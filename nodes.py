"""ComfyUI nodes: SHIFT reference-image style steering for FLUX.2 Klein (4B / 9B)."""

from __future__ import annotations

import logging
import math
import os

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import node_helpers

from .shift_core import (
    ActivationRecorder,
    ShiftVector,
    make_apply_patch,
    parse_index_list,
)

log = logging.getLogger("SHIFT-StyleSteer")

VECTOR_FOLDER = "shift_vectors"
CATEGORY = "SHIFT style steering"

DEFAULT_PROMPTS = """a red fox in the snow
a cat sitting on a windowsill
a horse in a meadow
an owl on a branch
a sea turtle swimming
a lighthouse on a rocky coast
a bowl of fruit on a wooden table
a city street at dusk"""


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def flux2_schedule(steps: int, width: int, height: int) -> torch.Tensor:
    """Same schedule as ComfyUI's Flux2Scheduler node (and diffusers' Flux2KleinPipeline)."""
    seq_len = round(width * height / (16 * 16))
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if seq_len > 4300:
        mu = a2 * seq_len + b2
    else:
        m_200 = a2 * seq_len + b2
        m_10 = a1 * seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * steps + b
    t = torch.linspace(1, 0, steps + 1)
    with torch.no_grad():
        shifted = torch.where(t > 0, math.exp(mu) / (math.exp(mu) + (1 / t.clamp(min=1e-12) - 1)), torch.zeros_like(t))
    return shifted


def zero_out(conditioning):
    out = []
    for t in conditioning:
        d = t[1].copy()
        if d.get("pooled_output") is not None:
            d["pooled_output"] = torch.zeros_like(d["pooled_output"])
        out.append([torch.zeros_like(t[0]), d])
    return out


def prepare_reference(image: torch.Tensor, max_megapixels: float) -> torch.Tensor:
    """Downscale (never upscale) to at most `max_megapixels`, sides multiple of 16, as Klein does."""
    img = image[:1, :, :, :3]
    h, w = img.shape[1], img.shape[2]
    scale = min(1.0, math.sqrt(max_megapixels * 1024 * 1024 / float(h * w)))
    nw = max(16, int(w * scale) // 16 * 16)
    nh = max(16, int(h * scale) // 16 * 16)
    if (nw, nh) != (w, h):
        img = comfy.utils.common_upscale(img.movedim(-1, 1), nw, nh, "area", "disabled").movedim(1, -1)
    return img


def double_block_count(model) -> int:
    dm = model.get_model_object("diffusion_model")
    blocks = getattr(dm, "double_blocks", None)
    if blocks is None:
        raise ValueError("[SHIFT] this MODEL has no `double_blocks`; load a FLUX.2 Klein diffusion model.")
    return len(blocks)


def vector_dir() -> str:
    return folder_paths.get_folder_paths(VECTOR_FOLDER)[0]


# --------------------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------------------
class SHIFTExtractStyleVector:
    """Runs every prompt twice from the same noise (with / without the reference image),
    records the double-stream block outputs and returns their mean difference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE", {"tooltip": "The style reference. Used only here, never at generation time."}),
                "prompts": ("STRING", {"multiline": True, "default": DEFAULT_PROMPTS,
                                       "tooltip": "One prompt per line (content only, no style words). Lines starting with # are skipped."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100, "tooltip": "4 for the distilled model, ~20-50 for base."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1,
                                  "tooltip": "1.0 for the distilled model. Only the positive (cond) rows are recorded."}),
                "seed": ("INT", {"default": 42000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "max_prompts": ("INT", {"default": 16, "min": 1, "max": 1000}),
                "ref_max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05}),
                "return_previews": ("BOOLEAN", {"default": True,
                                                "tooltip": "Decode teacher (with reference) and student (prompt only) images for a visual check."}),
            }
        }

    RETURN_TYPES = ("SHIFT_VECTOR", "IMAGE", "IMAGE")
    RETURN_NAMES = ("shift_vector", "teacher_i2i", "student_t2i")
    FUNCTION = "extract"
    CATEGORY = CATEGORY

    def extract(self, model, clip, vae, reference_image, prompts, width, height, steps, cfg, seed,
                max_prompts, ref_max_megapixels, return_previews):
        prompt_list = [p.strip() for p in prompts.splitlines() if p.strip() and not p.strip().startswith("#")]
        prompt_list = prompt_list[:max_prompts]
        if not prompt_list:
            raise ValueError("[SHIFT] no prompts given")

        n_blocks = double_block_count(model)
        sigmas = flux2_schedule(steps, width, height)
        ref = prepare_reference(reference_image, ref_max_megapixels)
        ref_latent = vae.encode(ref)
        log.info("[SHIFT] extracting on %d prompts, %d double blocks, ref %dx%d, sigmas %s",
                 len(prompt_list), n_blocks, ref.shape[2], ref.shape[1], [round(float(s), 4) for s in sigmas])

        recorder = ActivationRecorder(sigmas.tolist())
        m = model.clone()
        for i in range(n_blocks):
            m.set_model_patch_replace(recorder.make_patch(i), "dit", "double_block", i)

        sampler = comfy.samplers.sampler_object("euler")
        latent = torch.zeros([1, 128, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        latent = comfy.sample.fix_empty_latent_channels(m, latent)

        teacher, student = [], []
        pbar = comfy.utils.ProgressBar(2 * len(prompt_list))
        for idx, prompt in enumerate(prompt_list):
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            neg = zero_out(cond)
            noise = comfy.sample.prepare_noise(latent, seed + idx)
            for positive in (True, False):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if positive:
                    pos_c = node_helpers.conditioning_set_values(cond, {"reference_latents": [ref_latent]}, append=True)
                    neg_c = node_helpers.conditioning_set_values(neg, {"reference_latents": [ref_latent]}, append=True)
                else:
                    pos_c, neg_c = cond, neg
                recorder.begin_pass(positive)
                samples = comfy.sample.sample_custom(m, noise, cfg, sampler, sigmas, pos_c, neg_c, latent,
                                                     disable_pbar=True, seed=seed + idx)
                if return_previews:
                    (teacher if positive else student).append(vae.decode(samples)[:1])
                pbar.update(1)

        vec = recorder.to_vector({"width": width, "height": height, "steps": steps, "cfg": cfg,
                                  "num_double_blocks": n_blocks,
                                  "ref_size": [int(ref.shape[2]), int(ref.shape[1])]})
        log.info("[SHIFT] %s", vec.describe())

        blank = torch.zeros([1, 64, 64, 3])
        t_img = torch.cat(teacher, 0) if teacher else blank
        s_img = torch.cat(student, 0) if student else blank
        return (vec, t_img, s_img)


class SHIFTApplyStyleVector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "shift_vector": ("SHIFT_VECTOR",),
                "mode": (["renorm_add", "angle"], {"default": "renorm_add",
                         "tooltip": "renorm_add: h <- |h|(h + a v)/|h + a v| (paper rule). angle: rotate every token by a fixed angle."}),
                "strength_img": ("FLOAT", {"default": 6.0, "min": -200.0, "max": 200.0, "step": 0.5,
                                           "tooltip": "alpha for the image stream (paper default ~6 on Klein 9B; sweep 2-12)."}),
                "angle_deg": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 90.0, "step": 0.5,
                                        "tooltip": "Only used in angle mode."}),
                "strength_txt": ("FLOAT", {"default": 0.0, "min": -200.0, "max": 200.0, "step": 0.5,
                                           "tooltip": "alpha for the text stream (0 = image-stream only, the paper setting)."}),
                "blocks": ("STRING", {"default": "all", "tooltip": "Double blocks to steer: 'all' or e.g. '0-3' or '0,2,4'."}),
                "steps": ("STRING", {"default": "all", "tooltip": "Steps of the extraction grid to steer: 'all' or e.g. '0,1'."}),
                "vector_field": (["per_token", "pooled"], {"default": "per_token",
                                 "tooltip": "per_token = 2-D field (needs the extraction resolution). pooled = one vector per block, any resolution."}),
                "apply_to": (["cond_and_uncond", "cond_only"], {"default": "cond_and_uncond"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, shift_vector: ShiftVector, mode, strength_img, angle_deg, strength_txt,
              blocks, steps, vector_field, apply_to):
        n_model = double_block_count(model)
        n = min(n_model, shift_vector.num_blocks)
        if n_model != shift_vector.num_blocks:
            log.warning("[SHIFT] model has %d double blocks, vector has %d (extracted on another model?)",
                        n_model, shift_vector.num_blocks)
        block_set = parse_index_list(blocks, n)
        step_set = parse_index_list(steps, shift_vector.num_steps)
        m = model.clone()
        for i in range(n):
            if block_set is not None and i not in block_set:
                continue
            m.set_model_patch_replace(
                make_apply_patch(shift_vector, i, mode=mode, strength_img=strength_img, strength_txt=strength_txt,
                                 angle_deg=angle_deg, steps=step_set, pooled=(vector_field == "pooled"),
                                 cond_only=(apply_to == "cond_only")),
                "dit", "double_block", i)
        return (m,)


class SHIFTSaveStyleVector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shift_vector": ("SHIFT_VECTOR",),
                             "filename_prefix": ("STRING", {"default": "klein4b_style"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, shift_vector: ShiftVector, filename_prefix):
        from safetensors.torch import save_file
        out_dir = vector_dir()
        os.makedirs(out_dir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(out_dir, f"{filename_prefix}_{i:05d}.safetensors")):
            i += 1
        path = os.path.join(out_dir, f"{filename_prefix}_{i:05d}.safetensors")
        tensors, meta = shift_vector.to_safetensors_dict()
        save_file(tensors, path, metadata=meta)
        log.info("[SHIFT] saved %s", path)
        return {"ui": {"text": [path]}}


class SHIFTLoadStyleVector:
    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list(VECTOR_FOLDER)
        return {"required": {"vector_name": (files or ["(none - run the extract workflow first)"],)}}

    RETURN_TYPES = ("SHIFT_VECTOR",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, vector_name):
        path = folder_paths.get_full_path_or_raise(VECTOR_FOLDER, vector_name)
        vec = ShiftVector.from_safetensors(path)
        log.info("[SHIFT] loaded %s: %s", vector_name, vec.describe())
        return (vec,)


NODE_CLASS_MAPPINGS = {
    "SHIFTExtractStyleVector": SHIFTExtractStyleVector,
    "SHIFTApplyStyleVector": SHIFTApplyStyleVector,
    "SHIFTSaveStyleVector": SHIFTSaveStyleVector,
    "SHIFTLoadStyleVector": SHIFTLoadStyleVector,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SHIFTExtractStyleVector": "SHIFT Extract Style Vector (Klein)",
    "SHIFTApplyStyleVector": "SHIFT Apply Style Steering (Klein)",
    "SHIFTSaveStyleVector": "SHIFT Save Style Vector",
    "SHIFTLoadStyleVector": "SHIFT Load Style Vector",
}
