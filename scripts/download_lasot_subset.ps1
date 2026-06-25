param(
    [string]$RepoId = "l-lt/LaSOT",
    [string]$OutputRoot = "data/raw/LaSOT_subset",
    [string[]]$Classes = @("hand", "licenseplate", "person"),
    [switch]$NoExtract,
    [switch]$KeepArchives
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ResolvedOutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot
} else {
    Join-Path $ProjectRoot $OutputRoot
}
$ArchiveRoot = Join-Path $ResolvedOutputRoot "_archives"

New-Item -ItemType Directory -Force -Path $ResolvedOutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ArchiveRoot | Out-Null

Write-Host "LaSOT subset target: $ResolvedOutputRoot"
Write-Host "Classes: $($Classes -join ', ')"

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python is required for Hugging Face downloads."
}

$downloadScript = @"
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "$RepoId"
archive_root = Path(r"$ArchiveRoot")
classes = $($Classes | ConvertTo-Json -Compress)
if isinstance(classes, str):
    classes = [classes]

for name in classes:
    filename = f"{name}.zip"
    print(f"Downloading {filename} from {repo_id}...")
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(archive_root),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded: {path}")
"@

$downloadScript | python -

if (-not $NoExtract) {
    foreach ($className in $Classes) {
        $archivePath = Join-Path $ArchiveRoot "$className.zip"
        $classOutput = Join-Path $ResolvedOutputRoot $className

        if (-not (Test-Path $archivePath)) {
            throw "Missing downloaded archive: $archivePath"
        }

        if (Test-Path $classOutput) {
            $existingFiles = Get-ChildItem -Path $classOutput -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($existingFiles) {
                Write-Host "Already extracted, skipping: $classOutput"
                continue
            }
        }

        Write-Host "Extracting $archivePath -> $ResolvedOutputRoot"
        Expand-Archive -Path $archivePath -DestinationPath $ResolvedOutputRoot -Force
    }
}

if (-not $KeepArchives -and -not $NoExtract) {
    foreach ($className in $Classes) {
        $archivePath = Join-Path $ArchiveRoot "$className.zip"
        if (Test-Path $archivePath) {
            Remove-Item -LiteralPath $archivePath -Force
        }
    }
}

$summaryPath = Join-Path $ResolvedOutputRoot "subset_manifest.json"
$manifest = [ordered]@{
    name = "LaSOT_subset"
    source = "https://huggingface.co/datasets/$RepoId"
    classes = $Classes
    extracted_to = $ResolvedOutputRoot
    created_for = "V9 selected-target and small/part-like SOT training"
    note = "Class-level LaSOT subset. Use as SOT template/search supervision; not MOT identity supervision."
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host "Done. Manifest: $summaryPath"
