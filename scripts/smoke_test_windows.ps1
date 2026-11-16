# CPU smoke test (tiny random model, no downloads): checks the node logic inside your ComfyUI checkout.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Comfy = (Get-Content (Join-Path $Root ".comfy_dir") -Raw).Trim()
& (Join-Path $Comfy ".venv\Scripts\python.exe") (Join-Path $Root "tests\smoke_test.py") --comfy $Comfy
