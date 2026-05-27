[CmdletBinding()]
param(
    [string]$VenvPath = ".venv",
    [string]$LoRATConfig = "B-224",
    [string]$WeightPath = "models/lorat/base.bin"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $Root "$VenvPath/Scripts/python.exe"
$LoRATRoot = Join-Path $Root "external/LoRAT-main"
$OutputDir = Join-Path $Root "outputs/lorat-smoke"
$ResolvedWeightPath = Join-Path $Root $WeightPath

if (-not (Test-Path $PythonExe)) {
    throw "Python environment not found at $PythonExe. Run scripts/setup-lorat-env.ps1 first."
}
if (-not (Test-Path $LoRATRoot)) {
    throw "LoRAT checkout not found at $LoRATRoot. Run scripts/fetch-assets.ps1 -Asset lorat-repo first."
}
if (-not (Test-Path $ResolvedWeightPath)) {
    throw "LoRAT weight not found at $ResolvedWeightPath. Run scripts/fetch-assets.ps1 -Asset lorat-models first."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Package/device check"
@'
import importlib
import torch
from pathlib import Path
from safetensors import safe_open

for name in ["torch", "torchvision", "timm", "yaml", "safetensors", "fvcore", "PIL", "turbojpeg"]:
    mod = importlib.import_module(name)
    print(f"{name}: ok {getattr(mod, '__version__', '')}")

print(f"torch cuda available: {torch.cuda.is_available()}")
print(f"torch cuda version: {torch.version.cuda}")
print(f"torch cuda device count: {torch.cuda.device_count()}")

for path in sorted(Path("models/lorat").glob("*.bin")):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        print(f"{path.name}: safetensors ok tensors={len(handle.keys())}")
'@ | & $PythonExe -

Write-Host ""
Write-Host "LoRAT dry-run"
$LogDir = Join-Path $Root "outputs/logs"
$LogPath = Join-Path $LogDir "verify-lorat-env.log"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Push-Location $LoRATRoot
try {
    & $PythonExe main.py LoRAT $LoRATConfig `
        --device cpu `
        --dry_run `
        --output_dir $OutputDir `
        --weight_path $ResolvedWeightPath `
        --mixin_config do_model_profiling_only *> $LogPath

    if ($LASTEXITCODE -ne 0) {
        throw "LoRAT dry-run failed with exit code $LASTEXITCODE. See $LogPath"
    }
}
finally {
    Pop-Location
}

$logText = Get-Content -Path $LogPath -Raw
if ($logText -match "Traceback|Error|AssertionError") {
    throw "LoRAT dry-run wrote an error to $LogPath"
}

Write-Host "LoRAT dry-run ok. Log written to:"
Write-Host "  $LogPath"
