[CmdletBinding()]
param(
    [string]$VenvPath = ".venv",
    [string]$TorchIndexUrl = "",
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvFullPath = Join-Path $Root $VenvPath
$PythonExe = Join-Path $VenvFullPath "Scripts/python.exe"
$LoRATRequirementsPath = Join-Path $Root "external/LoRAT-main/requirements.txt"
$ProjectRequirementsPath = Join-Path $Root "requirements.txt"

if (-not (Test-Path $LoRATRequirementsPath)) {
    throw "LoRAT requirements not found at $LoRATRequirementsPath. Run scripts/fetch-assets.ps1 -Asset lorat-repo first."
}

if (-not (Test-Path $ProjectRequirementsPath)) {
    throw "Project requirements not found at $ProjectRequirementsPath."
}

if ($ForceRecreate -and (Test-Path $VenvFullPath)) {
    Remove-Item -LiteralPath $VenvFullPath -Recurse -Force
}

if (-not (Test-Path $PythonExe)) {
    python -m venv $VenvFullPath
}

& $PythonExe -m pip install --upgrade pip setuptools wheel

# The default Windows PyPI wheels are CPU-only here. For CUDA benchmarking, run this
# script on a machine with an NVIDIA GPU and install the matching PyTorch CUDA wheel
# from pytorch.org before installing LoRAT's requirements.
if ([string]::IsNullOrWhiteSpace($TorchIndexUrl)) {
    & $PythonExe -m pip install torch torchvision
} else {
    & $PythonExe -m pip install torch torchvision --index-url $TorchIndexUrl
}
& $PythonExe -m pip install -r $LoRATRequirementsPath
& $PythonExe -m pip install -r $ProjectRequirementsPath

Write-Host ""
Write-Host "LoRAT Python environment is ready:"
Write-Host "  $PythonExe"
