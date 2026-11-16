#!/usr/bin/env bash
# Same as setup_windows.ps1, for a Linux GPU box (e.g. a RunPod pod with a CUDA PyTorch image).
# ComfyUI goes to ../ComfyUI-shift next to this repo (override with COMFY_DIR=/path).
#   bash scripts/setup_linux.sh            # everything
#   SKIP_MODELS=1 bash scripts/setup_linux.sh
#   MODELS="4b 9b-base" HF_TOKEN=hf_... bash scripts/setup_linux.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY="${COMFY_DIR:-$(dirname "$ROOT")/ComfyUI-shift}"; VENV="$COMFY/.venv"; CUDA="${CUDA:-cu128}"

if [ -f "$COMFY/main.py" ]; then git -C "$COMFY" pull --ff-only; else git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY"; fi
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
$PY -m pip install --upgrade pip
$PY -c "import torch" 2>/dev/null || $PY -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA"
$PY -m pip install -r "$COMFY/requirements.txt"

ln -sfn "$ROOT" "$COMFY/custom_nodes/ComfyUI-SHIFT-StyleSteer"
mkdir -p "$COMFY/user/default/workflows" "$COMFY/models/shift_vectors"
cp "$ROOT"/example_workflows/*.json "$COMFY/user/default/workflows/"
cp "$ROOT"/references/*.jpg "$COMFY/input/"

[ -z "${SKIP_MODELS:-}" ] && COMFY_DIR="$COMFY" bash "$ROOT/scripts/download_models.sh"
echo "$COMFY" > "$ROOT/.comfy_dir"
echo "Done. Start: bash scripts/run_linux.sh   (then open port 8188)"
