# Start ComfyUI from this folder. Extra arguments are passed to ComfyUI, e.g.
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 --lowvram
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py   = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run scripts\setup_windows.ps1 first." }
Write-Host "ComfyUI -> http://127.0.0.1:8188  (Workflows panel: 01_shift_klein4b_extract_and_generate)" -ForegroundColor Cyan
& $Py (Join-Path $Root "ComfyUI\main.py") --listen 127.0.0.1 --port 8188 @args
