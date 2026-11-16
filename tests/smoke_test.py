"""
CPU smoke test for ComfyUI-SHIFT-StyleSteer (no weights, no GPU needed).

Builds a *tiny random* FLUX.2-Klein-shaped model inside a real ComfyUI checkout and runs:
  extract (with a reference image)  ->  save  ->  load  ->  apply  ->  sample
and checks that the vector is non-zero, that reference tokens were cropped, that steering
changes the output and that strength 0 is a no-op.

    python tests/smoke_test.py --comfy C:\\dev\\ComfyUI-shift
"""

import argparse
import hashlib
import os
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--comfy", required=True, help="path to a ComfyUI checkout")
cli = ap.parse_args()
sys.path.insert(0, os.path.abspath(cli.comfy))
sys.argv = [sys.argv[0], "--cpu"]
HERE = os.path.dirname(os.path.abspath(__file__))

import importlib.util  # noqa: E402

import torch  # noqa: E402

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
import folder_paths  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.supported_models  # noqa: E402

tmp_models = tempfile.mkdtemp()
folder_paths.models_dir = tmp_models

pkg_dir = os.path.dirname(HERE)  # repo root = the custom node package
spec = importlib.util.spec_from_file_location("shift_steer", os.path.join(pkg_dir, "__init__.py"),
                                              submodule_search_locations=[pkg_dir])
shift = importlib.util.module_from_spec(spec)
sys.modules["shift_steer"] = shift
spec.loader.exec_module(shift)
nodes = sys.modules["shift_steer.nodes"]

torch.manual_seed(0)

# ---- tiny Klein-shaped model ------------------------------------------------------------
unet_config = {
    "image_model": "flux2", "axes_dim": [32, 32, 32, 32], "num_heads": 2, "mlp_ratio": 3.0, "theta": 2000,
    "out_channels": 128, "global_modulation": True, "mlp_silu_act": True, "qkv_bias": False, "ops_bias": False,
    "default_ref_method": "index", "ref_index_scale": 10.0, "txt_ids_dims": [3], "patch_size": 1,
    "in_channels": 128, "hidden_size": 256, "context_in_dim": 64, "vec_in_dim": None,
    "depth": 3, "depth_single_blocks": 1, "guidance_embed": False, "txt_norm": False,
}
cfg = comfy.supported_models.Flux2(unet_config)
cfg.set_inference_dtype(torch.float32, None)
base = cfg.get_model({})
with torch.no_grad():
    for p in base.diffusion_model.parameters():
        p.copy_(torch.randn_like(p) * 0.05)
model = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


class FakeClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, text):
        g = torch.Generator().manual_seed(int(hashlib.md5(text.encode()).hexdigest()[:8], 16))
        n = 5 + len(text) % 7  # variable length, Flux2 pads to 512
        return [[torch.randn(1, n, 64, generator=g), {}]]


class FakeVAE:
    def encode(self, img):  # (B, H, W, 3) -> (B, 128, H/16, W/16)
        x = torch.nn.functional.avg_pool2d(img.movedim(-1, 1), 16)
        return x.repeat(1, 43, 1, 1)[:, :128] * 2 - 1

    def decode(self, lat):
        return torch.sigmoid(lat[:, :3]).repeat_interleave(16, 2).repeat_interleave(16, 3).movedim(1, -1)


W = H = 128  # 8x8 latent -> 64 generated tokens
ref_img = torch.rand(1, 80, 96, 3)  # -> 5x6 latent -> 30 reference tokens

# ---- 1) extract -------------------------------------------------------------------------
vec, teacher, student = nodes.SHIFTExtractStyleVector().extract(
    model, FakeClip(), FakeVAE(), ref_img, "a fox\na cat\na lighthouse", W, H, 4, 1.0, 7, 16, 1.0, True)
print(vec.describe())
assert vec.num_blocks == 3 and vec.num_steps == 4, (vec.num_blocks, vec.num_steps)
assert vec.tensors[("img", 0, 0)].shape == (64, 256), vec.tensors[("img", 0, 0)].shape  # ref tokens cropped
assert vec.tensors[("txt", 0, 0)].shape == (512, 256)
assert float(vec.tensors[("img", 3, 2)].float().abs().mean()) > 0, "vector is zero: reference had no effect"
assert teacher.shape[0] == 3 and student.shape[0] == 3
print("extract OK")

# ---- 2) save / load ---------------------------------------------------------------------
res = nodes.SHIFTSaveStyleVector().save(vec, "smoke")
path = res["ui"]["text"][0]
name = os.path.basename(path)
assert name in folder_paths.get_filename_list("shift_vectors"), folder_paths.get_filename_list("shift_vectors")
(vec2,) = nodes.SHIFTLoadStyleVector().load(name)
assert torch.equal(vec2.tensors[("img", 1, 1)], vec.tensors[("img", 1, 1)]) and vec2.sigmas == vec.sigmas
print("save/load OK ->", path)


# ---- 3) apply + sample ------------------------------------------------------------------
def run(m, seed=3):
    lat = comfy.sample.fix_empty_latent_channels(m, torch.zeros(1, 128, H // 16, W // 16))
    noise = comfy.sample.prepare_noise(lat, seed)
    cond = FakeClip().encode_from_tokens_scheduled("a horse")
    return comfy.sample.sample_custom(m, noise, 1.0, comfy.samplers.sampler_object("euler"),
                                      nodes.flux2_schedule(4, W, H), cond, nodes.zero_out(cond), lat,
                                      disable_pbar=True, seed=seed)


apply = nodes.SHIFTApplyStyleVector().apply
plain = run(model)
(m0,) = apply(model, vec2, "renorm_add", 0.0, 10.0, 0.0, "all", "all", "per_token", "cond_and_uncond")
assert torch.allclose(run(m0), plain, atol=1e-5), "strength 0 must be a no-op"
(m6,) = apply(model, vec2, "renorm_add", 6.0, 10.0, 0.0, "all", "all", "per_token", "cond_and_uncond")
d6 = (run(m6) - plain).abs().mean().item()
(ma,) = apply(model, vec2, "angle", 0.0, 15.0, 0.0, "0-1", "0,1", "per_token", "cond_only")
da = (run(ma) - plain).abs().mean().item()
(mp,) = apply(model, vec2, "renorm_add", 6.0, 10.0, 2.0, "all", "all", "pooled", "cond_and_uncond")
dp = (run(mp) - plain).abs().mean().item()
print(f"mean |steered - plain|: renorm_add a=6: {d6:.4f}  angle 15deg (blocks 0-1, steps 0-1): {da:.4f}  pooled+txt: {dp:.4f}")
assert d6 > 1e-4 and da > 1e-4 and dp > 1e-4

# other resolution -> falls back to the pooled vector instead of crashing
H = W = 160
_ = run(m6)
print("resolution fallback OK")
print("ALL SMOKE TESTS PASSED")
