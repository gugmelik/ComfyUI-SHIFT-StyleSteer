"""
SHIFT reference-image style steering for FLUX.2 Klein inside ComfyUI.

Core math (same as src/steering/*_klein.py in github.com/gugmelik/SHIFT, branch flux-klein):

  extract   v[t, l, s][j] = mean_k h+[t, l, s][j] - mean_k h-[t, l, s][j]
            h+ = block output with  prompt + reference image   (teacher, Klein I2I)
            h- = block output with  prompt only                (student, Klein T2I)
            Reference tokens are cropped from the image stream before averaging,
            so h+ and h- have the same token layout.

  apply     h <- ||h|| * (h + a v_hat) / ||h + a v_hat||        ("renorm_add", the paper rule)
            or an exact rotation of h by `angle_deg` towards v   ("angle", revision-3 control)

Hooking is done with ComfyUI's own `patches_replace["dit"][("double_block", i)]`
mechanism, so no weights are copied and the patched MODEL works with any sampler,
LoRA or other model patch.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch

log = logging.getLogger("SHIFT-StyleSteer")

STREAMS = ("img", "txt")


# --------------------------------------------------------------------------------------
# Steering vector container
# --------------------------------------------------------------------------------------
class ShiftVector:
    """Per-(step, block, stream) steering fields, plus the sigma grid they were extracted on.

    tensors[(stream, step, block)] -> (T, D) float16 on CPU
    """

    def __init__(self, tensors: Dict[Tuple[str, int, int], torch.Tensor], sigmas: List[float], meta: dict):
        self.tensors = tensors
        self.sigmas = [float(s) for s in sigmas]
        self.meta = dict(meta)
        self._cache: Dict[tuple, torch.Tensor] = {}
        self._warned = set()

    # ---- bookkeeping -----------------------------------------------------------------
    @property
    def num_steps(self) -> int:
        return max((k[1] for k in self.tensors), default=-1) + 1

    @property
    def num_blocks(self) -> int:
        return max((k[2] for k in self.tensors), default=-1) + 1

    def describe(self) -> str:
        shapes = {}
        for (stream, step, block), t in self.tensors.items():
            shapes.setdefault(stream, tuple(t.shape))
        return (f"SHIFT vector: {self.num_steps} steps x {self.num_blocks} double blocks, "
                f"streams {shapes}, sigmas {[round(s, 4) for s in self.sigmas]}, "
                f"{self.meta.get('num_prompts', '?')} prompts, extracted at "
                f"{self.meta.get('width', '?')}x{self.meta.get('height', '?')}")

    def step_for_sigma(self, sigma: float) -> int:
        """Nearest step of the extraction grid (the final sigma=0 is not a model call)."""
        grid = self.sigmas[:-1] if len(self.sigmas) > 1 else self.sigmas
        return int(min(range(len(grid)), key=lambda i: abs(grid[i] - sigma)))

    def get(self, stream: str, step: int, block: int, n_tokens: int, device, dtype,
            pooled: bool) -> Optional[torch.Tensor]:
        """Return the field for this call, moved to the device once and cached.

        If the per-token field does not match the current token count (other resolution,
        background-token dropping, ...) the token-pooled vector is used instead."""
        raw = self.tensors.get((stream, step, block))
        if raw is None:
            return None
        use_pooled = pooled or raw.shape[0] != n_tokens
        if use_pooled and not pooled and (stream, raw.shape[0], n_tokens) not in self._warned:
            self._warned.add((stream, raw.shape[0], n_tokens))
            log.warning("[SHIFT] %s stream: vector has %d tokens, model call has %d -> using the "
                        "token-pooled vector. Re-extract at this resolution for the per-token field.",
                        stream, raw.shape[0], n_tokens)
        key = (stream, step, block, use_pooled, str(device), dtype)
        if key not in self._cache:
            v = raw.float()
            if use_pooled:
                v = v.mean(0, keepdim=True)
            self._cache[key] = v.to(device=device, dtype=dtype)
        return self._cache[key]

    # ---- (de)serialisation -----------------------------------------------------------
    def to_safetensors_dict(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
        tensors = {f"{s}.s{t}.b{b}": v.contiguous() for (s, t, b), v in self.tensors.items()}
        meta = {"shift_meta": json.dumps({**self.meta, "sigmas": self.sigmas})}
        return tensors, meta

    @classmethod
    def from_safetensors(cls, path: str) -> "ShiftVector":
        from safetensors import safe_open
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            meta = json.loads((f.metadata() or {}).get("shift_meta", "{}"))
            for key in f.keys():
                stream, s, b = key.split(".")
                tensors[(stream, int(s[1:]), int(b[1:]))] = f.get_tensor(key)
        sigmas = meta.pop("sigmas", [])
        return cls(tensors, sigmas, meta)


# --------------------------------------------------------------------------------------
# Steering rules
# --------------------------------------------------------------------------------------
def renorm_add(h: torch.Tensor, v: torch.Tensor, alpha: float) -> torch.Tensor:
    """h <- ||h|| (h + a v_hat) / ||h + a v_hat||  (per token, norm preserved)."""
    dtype = h.dtype
    hf = torch.nan_to_num(h.float())
    vf = torch.nan_to_num(v.float())
    norm = hf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v_hat = vf / vf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    s = hf + alpha * v_hat
    s = s / s.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.nan_to_num(s * norm).to(dtype)


def rotate_towards(h: torch.Tensor, v: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate every token by exactly `angle_deg` towards the part of v orthogonal to it."""
    dtype = h.dtype
    hf = torch.nan_to_num(h.float())
    vf = torch.nan_to_num(v.float()).expand_as(hf)
    norm = hf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    u = hf / norm
    w = vf - (vf * u).sum(-1, keepdim=True) * u
    w = w / w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    th = math.radians(angle_deg)
    return torch.nan_to_num(norm * (math.cos(th) * u + math.sin(th) * w)).to(dtype)


# --------------------------------------------------------------------------------------
# Helpers shared by the extraction and apply patches
# --------------------------------------------------------------------------------------
def current_sigma(transformer_options: dict) -> Optional[float]:
    s = transformer_options.get("sigmas")
    if s is None:
        return None
    return float(s.detach().flatten()[0])


def nearest_index(grid: List[float], sigma: float) -> int:
    grid = grid[:-1] if len(grid) > 1 else grid
    return int(min(range(len(grid)), key=lambda i: abs(grid[i] - sigma)))


def generated_token_count(img: torch.Tensor, transformer_options: dict) -> int:
    """Klein appends reference tokens after the generated tokens: [gen | ref_1 | ref_2 ...]."""
    refs = transformer_options.get("reference_image_num_tokens") or []
    return img.shape[1] - int(sum(refs))


def cond_row_mask(batch: int, transformer_options: dict, device) -> torch.Tensor:
    """True for rows that belong to the positive (cond) half of a CFG batch."""
    cou = transformer_options.get("cond_or_uncond") or [0]
    chunk = max(batch // len(cou), 1)
    mask = torch.zeros(batch, dtype=torch.bool, device=device)
    for i, c in enumerate(cou):
        if c == 0:
            mask[i * chunk:(i + 1) * chunk] = True
    return mask


def parse_index_list(text: str, n: int) -> Optional[set]:
    """'all' -> None (meaning every index). '0,1,3-4' -> {0,1,3,4}."""
    text = (text or "all").strip().lower()
    if text in ("", "all", "*"):
        return None
    out = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return {i for i in out if 0 <= i < n} if n > 0 else out


# --------------------------------------------------------------------------------------
# Extraction: running sums of (teacher - student) block outputs
# --------------------------------------------------------------------------------------
class ActivationRecorder:
    """Accumulates sum(h+) - sum(h-) per (stream, step, block) on the CPU in float32.

    Because every prompt is run once with and once without the reference, the counts of the
    two classes are equal and the mean difference is simply (sum+ - sum-) / N."""

    def __init__(self, sigmas: List[float], streams: Iterable[str] = STREAMS):
        self.sigmas = [float(s) for s in sigmas]
        self.streams = tuple(streams)
        self.sign = 1.0
        self.sums: Dict[Tuple[str, int, int], torch.Tensor] = {}
        self.count = {+1.0: 0, -1.0: 0}
        self.calls = 0

    def begin_pass(self, positive: bool):
        self.sign = 1.0 if positive else -1.0
        self.count[self.sign] += 1

    def _add(self, key, tensor: torch.Tensor):
        x = tensor.detach().float().sum(0).cpu() * self.sign  # sum over cond rows
        if key in self.sums:
            if self.sums[key].shape != x.shape:
                raise RuntimeError(f"[SHIFT] token layout changed during extraction for {key}: "
                                   f"{tuple(self.sums[key].shape)} vs {tuple(x.shape)}. "
                                   "Keep width/height fixed for all prompts.")
            self.sums[key] += x
        else:
            self.sums[key] = x

    def make_patch(self, block_idx: int):
        def patch(args, extra):
            out = extra["original_block"](args)
            to = args.get("transformer_options", {}) or {}
            sigma = current_sigma(to)
            if sigma is None:
                return out
            grid = self.sigmas
            step = nearest_index(grid, sigma)
            img, txt = out["img"], out["txt"]
            rows = cond_row_mask(img.shape[0], to, img.device)
            n_gen = generated_token_count(img, to)
            if "img" in self.streams:
                self._add(("img", step, block_idx), img[rows, :n_gen])
            if "txt" in self.streams:
                self._add(("txt", step, block_idx), txt[rows])
            self.calls += 1
            return out
        return patch

    def to_vector(self, meta: dict) -> ShiftVector:
        n_pos, n_neg = self.count[1.0], self.count[-1.0]
        if n_pos == 0 or n_pos != n_neg:
            raise RuntimeError(f"[SHIFT] unbalanced extraction: {n_pos} positive vs {n_neg} negative passes")
        tensors = {k: (v / n_pos).to(torch.float16) for k, v in self.sums.items()}
        return ShiftVector(tensors, self.sigmas, {**meta, "num_prompts": n_pos})


# --------------------------------------------------------------------------------------
# Apply patch
# --------------------------------------------------------------------------------------
def make_apply_patch(vec: ShiftVector, block_idx: int, *, mode: str, strength_img: float,
                     strength_txt: float, angle_deg: float, steps: Optional[set], pooled: bool,
                     cond_only: bool):
    def patch(args, extra):
        out = extra["original_block"](args)
        to = args.get("transformer_options", {}) or {}
        sigma = current_sigma(to)
        if sigma is None:
            return out
        step = vec.step_for_sigma(sigma)
        if steps is not None and step not in steps:
            return out

        img, txt = out["img"], out["txt"]
        rows = cond_row_mask(img.shape[0], to, img.device) if cond_only else None
        new = dict(out)

        # image stream: only generated tokens are steered, reference tokens are left alone
        if mode == "angle" or strength_img != 0:
            n_gen = generated_token_count(img, to)
            v = vec.get("img", step, block_idx, n_gen, img.device, img.dtype, pooled)
            if v is not None:
                gen = img[:, :n_gen]
                steered = rotate_towards(gen, v, angle_deg) if mode == "angle" else renorm_add(gen, v, strength_img)
                if rows is not None:
                    steered = torch.where(rows[:, None, None], steered, gen)
                new["img"] = torch.cat([steered, img[:, n_gen:]], dim=1) if n_gen < img.shape[1] else steered

        # text stream (off by default, the paper configuration steers the image stream only)
        if strength_txt != 0:
            v = vec.get("txt", step, block_idx, txt.shape[1], txt.device, txt.dtype, pooled)
            if v is not None:
                steered = renorm_add(txt, v, strength_txt)
                if rows is not None:
                    steered = torch.where(rows[:, None, None], steered, txt)
                new["txt"] = steered
        return new
    return patch
