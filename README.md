# ComfyUI-SHIFT-StyleSteer

Reference-image style steering for **FLUX.2 Klein** (4B / 9B) in ComfyUI.

The style comes from one reference image, and that image is used only once. Each prompt is run
twice from the same noise: once *with* the reference (Klein I2I, the teacher) and once *without*
it (plain T2I, the student). The node records the double-stream block outputs at every step,
crops the reference tokens, and takes the mean difference. That difference is the steering field.
Generation afterwards is plain text-to-image with **no reference image**: the node patches the
double blocks with

    h <- |h| * (h + a * v) / |h + a * v|        (per token, image stream)

This is the SHIFT method (arXiv:2604.09213) extended to reference images (gugmelik/SHIFT, branch
`flux-klein`). It runs as native ComfyUI model patches (`patches_replace["dit"]["double_block", i]`),
so no weights are copied, and it works together with LoRAs, other patches and any sampler.

---

## Install

### Option 1: ComfyUI-Manager, "Install via Git URL"

1. Open **Manager** and click **Install via Git URL**.
2. Paste the URL and confirm:
   ```
   https://github.com/gugmelik/ComfyUI-SHIFT-StyleSteer
   ```
3. Restart ComfyUI.

If Manager answers *"This action is not allowed with this security level configuration"*, it is
because installing from a Git URL counts as a high-risk action. To allow it:

1. Open Manager's `config.ini`:
   - ComfyUI v0.3.76 and newer: `ComfyUI/user/__manager/config.ini`
   - older versions: `ComfyUI/user/default/ComfyUI-Manager/config.ini`
2. Set `security_level = weak`.
3. Restart ComfyUI, install, then set it back to `normal`.

Git-URL installs also need ComfyUI to listen only on 127.0.0.1, so don't start it with `--listen`.

### Option 2: ComfyUI-Manager search (after the node is published to the Comfy Registry)

Search Manager for **SHIFT Style Steering** and click **Install**.

### Option 3: by hand

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gugmelik/ComfyUI-SHIFT-StyleSteer
# restart ComfyUI
```

The node has no extra Python dependencies: it only uses torch and safetensors, which come with ComfyUI.

### Models (FLUX.2 Klein 4B, Apache-2.0, ungated)

Download these three files into your ComfyUI `models` folders:

| File | Folder | Link |
| --- | --- | --- |
| `flux-2-klein-4b.safetensors` (7.2 GB) | `models/diffusion_models` | [download](https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors) |
| `qwen_3_4b.safetensors` (7.5 GB) | `models/text_encoders` | [download](https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors) |
| `flux2-vae.safetensors` (0.3 GB) | `models/vae` | [download](https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors) |

The example workflows carry these download links too, so ComfyUI's missing-models dialog can
offer them when you open a workflow.

---

## Use

1. Open an example workflow. The JSON files are in `example_workflows/`, and ComfyUI also lists them
   under **Templates** in this node pack's section. You can also drag a JSON file onto the canvas.
   - **`01_shift_klein4b_extract_and_generate`** extracts a vector, saves it, then renders an
     unsteered and a steered image with the same seed.
   - **`02_shift_klein4b_generate_from_saved_vector`** loads a saved vector and generates, without
     extracting again.
2. **Style reference** (Load Image): upload any image. `references/` contains CC0 examples; copy
   `monet_water_lilies.jpg` into `ComfyUI/input` to use the workflow's default.
3. Press **Run**. The *Teacher* previews should show the style. If they don't, the vector won't
   carry it either.
4. Tune **SHIFT Apply Style Steering**:
   - `strength_img`: start at 6 and sweep 2–12.
   - `mode = angle` with `angle_deg` 5–20: a fixed rotation per token.
   - `blocks` / `steps` (e.g. `0-2`, `0,1`): restrict steering to some blocks or steps.
   - `vector_field = pooled`: one vector per block, so it works at any resolution.

Vectors are saved to `ComfyUI/models/shift_vectors/`. Extract again for each style and each model
(4B and 9B vectors are not interchangeable).

**Hardware:** an NVIDIA GPU with 16 GB of VRAM or more is comfortable. With 8–12 GB, start ComfyUI
with `--lowvram`.

---

## Developer setup (optional)

The `scripts/` folder builds a separate ComfyUI next to this repo, with its own venv, CUDA torch
and models. It links this repo into `custom_nodes` (as a junction or symlink), so you edit the code
here and just restart ComfyUI.

```powershell
# Windows, from the repo folder
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1   # -> ..\ComfyUI-shift
powershell -ExecutionPolicy Bypass -File scripts\smoke_test_windows.ps1   # CPU test, tiny random model
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1          # http://127.0.0.1:8188
```

```bash
# Linux / RunPod
bash scripts/setup_linux.sh && bash scripts/run_linux.sh             # port 8188
```

Options: `-Cuda cu126` for older drivers, `-SkipModels` if the models are already downloaded,
`-ComfyDir D:\ComfyUI` to use another ComfyUI folder. On Linux, use `COMFY_DIR=/path`.

```
ComfyUI-SHIFT-StyleSteer/
├── __init__.py / nodes.py / shift_core.py   the node pack (ComfyUI imports __init__.py)
├── pyproject.toml                           Comfy Registry metadata
├── example_workflows/                       workflows shown in ComfyUI's template browser
├── references/  prompts/                    CC0 style images, 50 train / 100 test prompts
├── scripts/  tests/  tools/                 dev setup, CPU smoke test, workflow generator
└── .github/workflows/publish_action.yml     publishes to the Comfy Registry when pyproject.toml changes
```

---

## The nodes

| Node | Inputs | Output |
| --- | --- | --- |
| **SHIFT Extract Style Vector (Klein)** | MODEL, CLIP, VAE, reference IMAGE, prompts (one per line), width/height (1024), steps (4), cfg (1.0), seed, max_prompts, ref_max_megapixels | `SHIFT_VECTOR`, teacher images, student images |
| **SHIFT Apply Style Steering (Klein)** | MODEL, SHIFT_VECTOR, mode, strength_img, angle_deg, strength_txt, blocks, steps, vector_field, apply_to | patched MODEL |
| **SHIFT Save Style Vector** | SHIFT_VECTOR, filename_prefix | `models/shift_vectors/<prefix>_00001.safetensors` |
| **SHIFT Load Style Vector** | file name | SHIFT_VECTOR |

A few details:

- **Reference tokens are cropped.** Klein lays out the image stream as `[generated | reference]`.
  Only the generated tokens are recorded and steered. The token count comes from ComfyUI's
  `reference_image_num_tokens`.
- **Steps are matched by sigma.** Each model call is matched to the nearest sigma of the extraction
  grid. The 4-step grid at 1024² is (1.000, 0.967, 0.908, 0.767), the same as in the paper. Running
  with a different step count therefore still maps to the closest extracted step.
- **CFG > 1 is supported** (Klein *base*). Only the cond rows are recorded during extraction. With
  `apply_to = cond_only`, only the cond rows are steered.
- **The vector file** is safetensors with keys `img.s{step}.b{block}` and `txt.s{step}.b{block}`,
  plus metadata (sigmas, resolution, prompt count). At 1024² it is a few hundred MB to ~1 GB in fp16,
  depending on the model size.
- **Using 9B:** load `flux-2-klein-9b*.safetensors` + `qwen_3_8b*.safetensors` and extract again.
  Vectors are model-specific. Note that the 9B weights are non-commercial.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| SHIFT nodes are red / missing | Check the ComfyUI console for `IMPORT FAILED ... ComfyUI-SHIFT-StyleSteer`. Make sure `ComfyUI/custom_nodes/ComfyUI-SHIFT-StyleSteer/__init__.py` exists, then restart ComfyUI. |
| `this MODEL has no double_blocks` | The Load Diffusion Model node isn't pointing at a Klein file. |
| Loader nodes are red | The model files are missing. Download the three files from the Models table into `ComfyUI/models/{diffusion_models,text_encoders,vae}`. |
| Out of memory | Start with `--lowvram`, lower `max_prompts` to 8, or use RunPod. |
| Warning `using the token-pooled vector` | The generation resolution differs from the extraction resolution. Extract again at that size for the per-token field. |
| Steered image ≈ unsteered | Raise `strength_img`, and check that the teacher previews actually show the style. |

## Credits

- Method: N. Konovalova, A. Kuznetsov, A. Alanov, *SHIFT: Steering hidden intermediates in flow
  transformers*, arXiv:2604.09213 (MIT). Reference-image extension for Klein: SHIFT branch `flux-klein`.
- Reference images: see `references/SOURCES.md` (AIC Open Access, CC0).
- FLUX.2 [klein] 4B: Black Forest Labs, Apache-2.0.
