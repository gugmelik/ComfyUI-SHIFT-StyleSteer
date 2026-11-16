#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY="$(cat "$ROOT/.comfy_dir")"
exec "$COMFY/.venv/bin/python" "$COMFY/main.py" --listen 0.0.0.0 --port 8188 "$@"
