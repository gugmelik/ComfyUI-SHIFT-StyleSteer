#!/usr/bin/env bash
# Download FLUX.2 Klein models into an existing ComfyUI (resumable).
#   COMFY_DIR=/workspace/ComfyUI MODELS="4b 9b" HF_TOKEN=hf_... bash scripts/download_models.sh
# MODELS: 4b (default), 4b-base, 9b, 9b-base, all.
# 4b comes from Comfy-Org (Apache-2.0, ungated). 4b-base / 9b / 9b-base come from black-forest-labs
# repos: open each model page on Hugging Face once, accept the license, and export HF_TOKEN.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY="${COMFY_DIR:-$(cat "$ROOT/.comfy_dir" 2>/dev/null || echo "$(dirname "$ROOT")/ComfyUI-shift")}"
[ -f "$COMFY/main.py" ] || { echo "ComfyUI not found at $COMFY (set COMFY_DIR=/path/to/ComfyUI)"; exit 1; }
echo "ComfyUI: $COMFY"
# MODELS="4b 4b-base 9b 9b-base" (default: 4b). 9B is gated: accept the license on HF and export HF_TOKEN=hf_...
declare -A SEEN=()
dl() {  # dir name url   (resumable; each file once per run)
  [ -n "${SEEN[$2]:-}" ] && return; SEEN[$2]=1
  mkdir -p "$COMFY/models/$1"
  auth=(); case "$3" in *black-forest-labs*) [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN");; esac
  echo "  $2 -> models/$1"
  curl -L --fail -C - ${auth[@]+"${auth[@]}"} -o "$COMFY/models/$1/$2" "$3" || { echo "download failed: $3 (gated? accept license, set HF_TOKEN)"; exit 1; }
}
te4() { dl text_encoders qwen_3_4b.safetensors https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors; }
te9() { dl text_encoders qwen_3_8b.safetensors https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors; }
vae() { dl vae flux2-vae.safetensors https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors; }
[ "${MODELS:-4b}" = all ] && MODELS="4b 4b-base 9b 9b-base"
for m in ${MODELS:-4b}; do
  case "$m" in
    4b)      dl diffusion_models flux-2-klein-4b.safetensors https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors; te4; vae;;
    4b-base) dl diffusion_models flux-2-klein-base-4b-fp8.safetensors https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors; te4; vae;;
    9b)      dl diffusion_models flux-2-klein-9b.safetensors https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors; te9; vae;;
    9b-base) dl diffusion_models flux-2-klein-base-9b.safetensors https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/resolve/main/flux-2-klein-base-9b.safetensors; te9; vae;;
    *) echo "unknown model $m (4b 4b-base 9b 9b-base all)"; exit 1;;
  esac
done
echo "models ready in $COMFY/models"
