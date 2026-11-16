# Start ComfyUI from this folder. Extra arguments are passed to ComfyUI, e.g.
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 --lowvram
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Cfg  = Join-Path $Root ".comfy_dir"
if (-not (Test-Path $Cfg)) { throw "Run scripts\setup_windows.ps1 first." }
$Comfy = (Get-Content $Cfg -Raw).Trim()
$Py   = Join-Path $Comfy ".venv\Scripts\python.exe"

Write-Host "ComfyUI -> http://127.0.0.1:8188  (Workflows panel: shift_klein_<variant>_extract_and_generate)" -ForegroundColor Cyan
& $Py (Join-Path $Comfy "main.py") --listen 127.0.0.1 --port 8188 @args
