[CmdletBinding()]
param(
    [string]$RepoId = "chengyenhsieh/TAO-Amodal",
    [string]$OutputDir = "data/raw/TAO_OW_SUBSET",
    [string[]]$Sources = @("YFCC100M"),
    [string[]]$Splits = @("train", "val"),
    [switch]$IncludeCharades,
    [switch]$AnnotationsOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$target = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $target | Out-Null

if ($IncludeCharades -and -not ($Sources -contains "Charades")) {
    $Sources += "Charades"
}

$includePatterns = @(
    "README.md",
    "annotations/*",
    "download_frames.sh",
    "unzip_video.py"
)

if (-not $AnnotationsOnly) {
    foreach ($split in $Splits) {
        foreach ($source in $Sources) {
            $includePatterns += "frames/$split/$source.zip"
        }
    }
}

$hf = Get-Command hf -ErrorAction SilentlyContinue
if (-not $hf) {
    throw "The Hugging Face CLI `hf` is required. Install/update with: pip install -U huggingface_hub"
}

$args = @(
    "download",
    $RepoId,
    "--repo-type",
    "dataset",
    "--local-dir",
    $target
)

foreach ($pattern in $includePatterns) {
    $args += @("--include", $pattern)
}

if ($DryRun) {
    $args += "--dry-run"
}

Write-Host "Target: $target"
Write-Host "Patterns:"
foreach ($pattern in $includePatterns) {
    Write-Host "  $pattern"
}
Write-Host ""
Write-Host "If this fails with access denied, run `hf auth login` and accept the dataset terms at:"
Write-Host "https://huggingface.co/datasets/$RepoId"
Write-Host ""

& $hf.Source @args
if ($LASTEXITCODE -ne 0) {
    throw "hf download failed with exit code $LASTEXITCODE"
}
