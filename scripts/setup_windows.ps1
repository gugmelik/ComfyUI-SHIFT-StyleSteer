# Developer setup on Windows: a separate ComfyUI install next to this repo, with its own venv,
# CUDA torch, FLUX.2 Klein 4B models, and THIS repo linked into custom_nodes (edit here, restart ComfyUI).
# Nothing is installed system-wide. Default layout:
#   C:\dev\shift-klein-comfyui        <- this repo (any folder name works)
#   C:\dev\ComfyUI-shift              <- ComfyUI + .venv + models (created by this script)
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -SkipModels      # models already downloaded
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Cuda cu126       # older NVIDIA driver
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -ComfyDir D:\ComfyUI   # use / create another ComfyUI
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Models 4b,9b           # which Klein variants
#
# -Models: 4b (default), 4b-base, 9b, 9b-base, all.
# 9B models are gated (FLUX Non-Commercial License): accept the license on each Hugging Face page
# and set a read token first:  $env:HF_TOKEN = "hf_..."
param(
    [string[]]$Models = @("4b"),
    [string]$Cuda = "cu128",
    [switch]$SkipModels,
    [string]$Python = "",
    [string]$ComfyDir = ""
)
$ErrorActionPreference = "Stop"
$Root    = Split-Path -Parent $PSScriptRoot
if (-not $ComfyDir) { $ComfyDir = Join-Path (Split-Path -Parent $Root) "ComfyUI-shift" }
$Comfy   = $ComfyDir
$Venv    = Join-Path $Comfy ".venv"
$NodeSrc = $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# ---------------------------------------------------------------- prerequisites
Step "Checking git and Python"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git not found. Install Git for Windows: https://git-scm.com/download/win" }
if (-not $Python) {
    foreach ($v in "3.12", "3.13", "3.11") {
        try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $Python = "py -$v"; break } } catch {}
    }
    if (-not $Python) { $Python = "python" }
}
Write-Host "Using Python: $Python"
Invoke-Expression "$Python --version"

# ---------------------------------------------------------------- ComfyUI
if (Test-Path (Join-Path $Comfy "main.py")) {
    Step "Updating ComfyUI"
    git -C $Comfy pull --ff-only
} else {
    Step "Cloning ComfyUI into $Comfy"
    git clone https://github.com/comfyanonymous/ComfyUI.git $Comfy
}

# ---------------------------------------------------------------- venv + torch
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Step "Creating venv in $Venv"
    Invoke-Expression "$Python -m venv `"$Venv`""
}
$Py = Join-Path $Venv "Scripts\python.exe"
Step "Installing PyTorch ($Cuda) and ComfyUI requirements"
& $Py -m pip install --upgrade pip
& $Py -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$Cuda"
& $Py -m pip install -r (Join-Path $Comfy "requirements.txt")
& $Py -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# ---------------------------------------------------------------- link the custom node
Step "Linking the SHIFT node into ComfyUI\custom_nodes (junction, no copy)"
$Link = Join-Path $Comfy "custom_nodes\ComfyUI-SHIFT-StyleSteer"
if (-not (Test-Path $Link)) {
    New-Item -ItemType Junction -Path $Link -Target $NodeSrc | Out-Null
}
Write-Host "  $Link -> $NodeSrc"

# ---------------------------------------------------------------- workflows, references
Step "Copying workflows and reference images"
$Wf = Join-Path $Comfy "user\default\workflows"
New-Item -ItemType Directory -Force -Path $Wf | Out-Null
Copy-Item (Join-Path $Root "example_workflows\*.json") $Wf -Force
Copy-Item (Join-Path $Root "references\*.jpg") (Join-Path $Comfy "input") -Force
New-Item -ItemType Directory -Force -Path (Join-Path $Comfy "models\shift_vectors") | Out-Null

# ---------------------------------------------------------------- models
if (-not $SkipModels) {
    $TE4 = @{ dir = "text_encoders"; name = "qwen_3_4b.safetensors"; url = "https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" }
    $TE9 = @{ dir = "text_encoders"; name = "qwen_3_8b.safetensors"; url = "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors" }
    $VAE = @{ dir = "vae"; name = "flux2-vae.safetensors"; url = "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" }
    $Catalog = @{
        "4b"      = @(@{ dir = "diffusion_models"; name = "flux-2-klein-4b.safetensors"; url = "https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors" }, $TE4, $VAE)
        "4b-base" = @(@{ dir = "diffusion_models"; name = "flux-2-klein-base-4b-fp8.safetensors"; url = "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors" }, $TE4, $VAE)
        "9b"      = @(@{ dir = "diffusion_models"; name = "flux-2-klein-9b.safetensors"; url = "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors" }, $TE9, $VAE)
        "9b-base" = @(@{ dir = "diffusion_models"; name = "flux-2-klein-base-9b.safetensors"; url = "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/resolve/main/flux-2-klein-base-9b.safetensors" }, $TE9, $VAE)
    }
    $Wanted = @()
    foreach ($m in ($Models -split ",")) { $m = $m.Trim().ToLower(); if ($m -eq "all") { $Wanted += $Catalog.Keys } elseif ($m) { $Wanted += $m } }
    $Files = @{}
    foreach ($k in ($Wanted | Select-Object -Unique)) {
        if (-not $Catalog.ContainsKey($k)) { throw "Unknown model '$k'. Use: 4b, 4b-base, 9b, 9b-base, all" }
        foreach ($f in $Catalog[$k]) { $Files[$f.name] = $f }
    }
    Step ("Downloading models for: " + (($Wanted | Select-Object -Unique) -join ", ") + " (resumable)")
    foreach ($m in $Files.Values) {
        $dst = Join-Path $Comfy ("models\" + $m.dir + "\" + $m.name)
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Write-Host "  $($m.name) -> models\$($m.dir)"
        $curlArgs = @("-L", "--fail", "-C", "-", "-o", $dst)
        if ($m.url -like "*black-forest-labs*") {
            if (-not $env:HF_TOKEN) { Write-Warning "  $($m.name) is on a gated BFL repo: set `$env:HF_TOKEN and accept the license on Hugging Face if this fails." }
            else { $curlArgs += @("-H", "Authorization: Bearer $($env:HF_TOKEN)") }
        }
        & curl.exe @curlArgs $m.url
        if ($LASTEXITCODE -ne 0) { throw "download failed: $($m.url)  (gated repo? open it in the browser, accept the license, set HF_TOKEN)" }
    }
}

Set-Content (Join-Path $Root ".comfy_dir") $Comfy
Step "Done. Start ComfyUI with:  powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1"
