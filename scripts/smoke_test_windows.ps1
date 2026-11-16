# CPU smoke test (tiny random model, no downloads): checks the node logic inside your ComfyUI checkout.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $Root ".venv\Scripts\python.exe") (Join-Path $Root "tests\smoke_test.py") --comfy (Join-Path $Root "ComfyUI")
