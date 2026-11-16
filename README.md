# SHIFT style steering for FLUX.2 Klein 4B in ComfyUI

This folder has everything needed to run the node: the custom node, two workflows, CC0 reference
images, prompt sets, setup and run scripts, and a CPU smoke test. ComfyUI, its Python environment
and the models get installed **inside this folder**, so nothing goes system-wide.

**How it works.** The style comes from one reference image, and that image is used only once.
Each prompt is run twice from the same noise: once *with* the reference (Klein I2I, the teacher) and
once *without* it (plain T2I, the student). The node records the double-stream block outputs at every
step, crops the reference tokens, and takes the mean difference. That difference is the steering
field. Generation afterwards is plain text-to-image with **no reference image**: the node patches
the double blocks with

    h <- |h| * (h + a * v) / |h + a * v|        (per token, image stream)

This is the SHIFT reference-image method (gugmelik/SHIFT, branch `flux-klein`). It runs here as
native ComfyUI model patches (`patches_replace["dit"]["double_block", i]`), so no weights are copied,
and it works together with LoRAs, other patches and any sampler.

```
C:\dev\shift-klein-comfyui\
├── ComfyUI-SHIFT-StyleSteer\     the custom node (linked into ComfyUI\custom_nodes)
│   ├── __init__.py               registers the nodes + models\shift_vectors folder
│   ├── nodes.py                  Extract / Apply / Save / Load nodes
│   └── shift_core.py             recorder, steering rules, vector format (safetensors)
├── workflows\
│   ├── 01_shift_klein4b_extract_and_generate.json     extract + side-by-side generate
│   └── 02_shift_klein4b_generate_from_saved_vector.json
├── references\                   CC0 / public-domain style images (+ SOURCES.md)
├── prompts\                      50 train / 100 test prompts from the paper
├── scripts\                      setup / run for Windows and Linux (RunPod)
├── tests\smoke_test.py           CPU test with a tiny random Klein-shaped model
└── tools\make_workflows.py       regenerates the workflow JSONs
```

---

## Requirements

- An NVIDIA GPU. **16 GB VRAM or more** is comfortable. With 8–12 GB it should still work:
  ComfyUI offloads the text encoder, so start it with `--lowvram`.
  If your laptop GPU is too small, use the Linux steps on a RunPod pod (section B).
- About 20 GB of free disk space: 7.2 GB Klein 4B, 7.5 GB Qwen3-4B text encoder, 0.3 GB VAE, plus the venv.
- Git for Windows and Python 3.12 (3.11 and 3.13 also work).
- The models are **FLUX.2 Klein 4B (Apache-2.0)**. They are ungated, so no Hugging Face login is needed.

---

## A. Run on Windows (this PC)

Open **PowerShell** in `C:\dev\shift-klein-comfyui`:

```powershell
cd C:\dev\shift-klein-comfyui

# 1) one-time setup: clones ComfyUI, creates .venv, installs CUDA torch, links the node,
#    copies the workflows and reference images, downloads the 3 model files (resumable)
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#    older NVIDIA driver?  add  -Cuda cu126     models already downloaded?  add  -SkipModels

# 2) (optional) 10-second CPU test of the node logic, no models needed
powershell -ExecutionPolicy Bypass -File scripts\smoke_test_windows.ps1

# 3) start ComfyUI
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1
#    small GPU:  ... run_windows.ps1 --lowvram
```

At the end of setup, check the line `CUDA available: True | <your GPU>`. If it says `NO GPU`, rerun
setup with a different `-Cuda` value, or use section B.

### Then in the browser (http://127.0.0.1:8188)

1. Open the **Workflows** panel (left sidebar, or press `W`) and open
   **`01_shift_klein4b_extract_and_generate`**.
2. Look at the purple nodes (`SHIFT ...`) and check that none of them is red. A red node means the
   custom node did not load; see Troubleshooting.
3. **Style reference** (Load Image): `monet_water_lilies.jpg` is preselected. To use your own style,
   upload any image.
4. Press **Run**. The workflow then does three things:
   - **Extract**: 16 prompts × 2 passes × 4 steps. This takes a few minutes on a 24 GB card.
     The two preview nodes show *Teacher* (with the reference) and *Student* (prompt only). If the
     teacher images don't look like the style, the vector won't either.
   - **Save**: writes `ComfyUI\models\shift_vectors\klein4b_monet_00001.safetensors`.
   - **Generate**: `a tiger in the jungle`, rendered **unsteered** and **steered** with the same seed.
     The results are saved to `ComfyUI\output\shift_klein\`.
5. Tune **SHIFT Apply Style Steering**:
   - `strength_img`: start at 6 and sweep 2–12. Too high and the reference's *content* starts to
     leak into the image.
   - `mode = angle` with `angle_deg` 5–20: a fixed rotation per token, the revision-3 control.
   - `blocks` (e.g. `0-2`) and `steps` (e.g. `0,1`): restrict steering to some blocks or steps,
     as in the ablations.
   - `vector_field = pooled`: one vector per block, so it works at any resolution.
6. Change the prompt in the **Prompt (Steered)** node and run again. Only the generate part reruns;
   ComfyUI caches the extraction.

Once you have a saved vector, open **`02_shift_klein4b_generate_from_saved_vector`**. It skips
extraction: pick the file in **SHIFT Load Style Vector** and generate.

---

## B. Run on a Linux GPU box (RunPod)

Use a pod with a CUDA PyTorch template, at least 24 GB VRAM, and port 8188 exposed:

```bash
cd /workspace
# copy this folder to the pod (scp / runpodctl send / git), then:
cd shift-klein-comfyui
bash scripts/setup_linux.sh
bash scripts/run_linux.sh          # open https://<pod-id>-8188.proxy.runpod.net
```

Steps 1–6 in the browser are the same as in section A.

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
| SHIFT nodes are red / missing | Check the ComfyUI console for `IMPORT FAILED ... ComfyUI-SHIFT-StyleSteer`. Make sure the junction `ComfyUI\custom_nodes\ComfyUI-SHIFT-StyleSteer` exists (setup creates it). |
| `this MODEL has no double_blocks` | The Load Diffusion Model node isn't pointing at a Klein file. |
| Loader nodes are red | The model files are missing. Rerun setup without `-SkipModels`, or put the files into `ComfyUI\models\{diffusion_models,text_encoders,vae}`. |
| Out of memory | Start with `--lowvram`, lower `max_prompts` to 8, or use RunPod. |
| Warning `using the token-pooled vector` | The generation resolution differs from the extraction resolution. Extract again at that size for the per-token field. |
| Steered image ≈ unsteered | Raise `strength_img`, and check that the teacher previews actually show the style. |

## Credits

- Method: N. Konovalova, A. Kuznetsov, A. Alanov, *SHIFT: Steering hidden intermediates in flow
  transformers*, arXiv:2604.09213 (MIT). Reference-image extension for Klein: SHIFT branch `flux-klein`.
- Reference images: see `references/SOURCES.md` (AIC Open Access, CC0).
- FLUX.2 [klein] 4B: Black Forest Labs, Apache-2.0.
