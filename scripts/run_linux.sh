#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/ComfyUI/main.py" --listen 0.0.0.0 --port 8188 "$@"
