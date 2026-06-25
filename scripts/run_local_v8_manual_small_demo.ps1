[CmdletBinding()]
param(
    [string]$Sequence = "",
    [string]$Video = "",
    [ValidateSet("B-224", "L-224", "g-224")]
    [string]$LoRATConfig = "B-224",
    [ValidateSet("best", "latest")]
    [string]$HeadCheckpoint = "best",
    [string]$V8HeadWeights = "",
    [string]$DownloadsRoot = "$env:USERPROFILE\Downloads",
    [string]$CheckpointCacheRoot = "",
    [string]$OutputRoot = "",
    [string]$Device = "auto",
    [int]$MaxFrames = 0,
    [int]$MaxTracks = 0,
    [switch]$NoSlotDebug,
    [switch]$NoWeek2Proof,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Get-ConfigKey {
    param([string]$Config)
    return ($Config -replace "-", "_")
}

function Get-CheckpointNames {
    param([string]$Key)
    $primary = if ($HeadCheckpoint -eq "best") {
        "v8_head_${Key}_best_by_val_iou.pt"
    }
    else {
        "v8_head_${Key}_latest.pt"
    }
    $fallback = if ($HeadCheckpoint -eq "best") {
        "v8_head_${Key}_latest.pt"
    }
    else {
        "v8_head_${Key}_best_by_val_iou.pt"
    }
    return @($primary, $fallback)
}

function Join-ZipEntryPath {
    param(
        [string]$Root,
        [string]$EntryName
    )
    $path = $Root
    foreach ($part in ($EntryName -split "/")) {
        if (-not [string]::IsNullOrWhiteSpace($part)) {
            $path = Join-Path $path $part
        }
    }
    return $path
}

function Resolve-ExistingV8Head {
    param(
        [string]$RepoRoot,
        [string]$ConfigKey
    )

    $searchRoots = @(
        (Join-Path $RepoRoot "models\lorat\v8_local_checkpoints"),
        $DownloadsRoot
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_ -PathType Container) }

    foreach ($checkpointName in (Get-CheckpointNames $ConfigKey)) {
        $matches = New-Object System.Collections.Generic.List[object]
        foreach ($root in $searchRoots) {
            Get-ChildItem -LiteralPath $root -Recurse -File -Filter $checkpointName -ErrorAction SilentlyContinue |
                ForEach-Object { $matches.Add($_) }
        }
        $bestMatch = $matches | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -ne $bestMatch) {
            return $bestMatch.FullName
        }
    }

    return ""
}

function Expand-V8HeadFromLatestZip {
    param(
        [string]$RepoRoot,
        [string]$ConfigKey
    )

    if (-not (Test-Path -LiteralPath $DownloadsRoot -PathType Container)) {
        return ""
    }

    if ([string]::IsNullOrWhiteSpace($CheckpointCacheRoot)) {
        $CheckpointCacheRoot = Join-Path $RepoRoot "models\lorat\v8_local_checkpoints"
    }
    New-Item -ItemType Directory -Force -Path $CheckpointCacheRoot | Out-Null

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zips = Get-ChildItem -LiteralPath $DownloadsRoot -File -Filter "lorat-v8-train-results*.zip" |
        Sort-Object LastWriteTime -Descending

    foreach ($zipFile in $zips) {
        $zip = [System.IO.Compression.ZipFile]::OpenRead($zipFile.FullName)
        try {
            foreach ($checkpointName in (Get-CheckpointNames $ConfigKey)) {
                $entryName = "checkpoints/$ConfigKey/$checkpointName"
                $entry = $zip.GetEntry($entryName)
                if ($null -eq $entry) {
                    continue
                }

                $destinationRoot = Join-Path $CheckpointCacheRoot $zipFile.BaseName
                $target = Join-ZipEntryPath $destinationRoot $entryName
                $targetDir = Split-Path -Parent $target
                New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

                $needsExtract = $true
                if (Test-Path -LiteralPath $target -PathType Leaf) {
                    $existing = Get-Item -LiteralPath $target
                    $needsExtract = ($existing.Length -ne $entry.Length)
                }
                if ($needsExtract) {
                    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
                }
                return $target
            }
        }
        finally {
            $zip.Dispose()
        }
    }

    return ""
}

function Resolve-Python {
    param([string]$RepoRoot)
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        return $venvPython
    }
    return "python"
}

function Resolve-Device {
    param(
        [string]$Python,
        [string]$RequestedDevice
    )

    if ($RequestedDevice -ne "auto") {
        return $RequestedDevice
    }

    try {
        $detected = & $Python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')"
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($detected)) {
            return $detected.Trim()
        }
    }
    catch {
        Write-Warning "Could not auto-detect CUDA with torch; falling back to cpu."
    }
    return "cpu"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Resolve-Python $repoRoot
$configKey = Get-ConfigKey $LoRATConfig

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "outputs\local_v8_manual_small_demo"
}
if ([string]::IsNullOrWhiteSpace($Sequence) -and [string]::IsNullOrWhiteSpace($Video)) {
    $Sequence = Join-Path $repoRoot "data\raw\DanceTrack\val\val\dancetrack0065"
}

if (-not [string]::IsNullOrWhiteSpace($V8HeadWeights)) {
    $resolvedHead = (Resolve-Path -LiteralPath $V8HeadWeights).Path
}
else {
    $resolvedHead = Resolve-ExistingV8Head $repoRoot $configKey
    if ([string]::IsNullOrWhiteSpace($resolvedHead)) {
        $resolvedHead = Expand-V8HeadFromLatestZip $repoRoot $configKey
    }
}

if ([string]::IsNullOrWhiteSpace($resolvedHead) -or -not (Test-Path -LiteralPath $resolvedHead -PathType Leaf)) {
    throw "Could not find a trained V8 head checkpoint. Pass -V8HeadWeights or put/extract a lorat-v8-train-results zip in Downloads."
}

$resolvedDevice = Resolve-Device $python $Device
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $OutputRoot "manual_small_${configKey}_${timestamp}"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

$tracker = Join-Path $repoRoot "programs\bounding_box_v8_lorat_quality_batched.py"
$loratRoot = Join-Path $repoRoot "external\LoRAT-main"
$saveVideo = Join-Path $runRoot "manual_small_target_${configKey}.mp4"
$motOutput = Join-Path $runRoot "manual_small_target_${configKey}.txt"
$debugLog = Join-Path $runRoot "manual_small_target_${configKey}_debug.csv"
$manualEventLog = Join-Path $runRoot "manual_small_target_${configKey}_manual_events.csv"
$slotDebugLog = Join-Path $runRoot "manual_small_target_${configKey}_slot_debug.csv"
$week2ProofLog = Join-Path $runRoot "manual_small_target_${configKey}_week2_proof.csv"

$argsList = @(
    $tracker,
    "--lorat-root", $loratRoot,
    "--lorat-config", $LoRATConfig,
    "--device", $resolvedDevice,
    "--v8-head-weights", $resolvedHead,
    "--max-tracks", ([string]$MaxTracks),
    "--save-video", $saveVideo,
    "--output", $motOutput,
    "--debug-log", $debugLog,
    "--manual-event-log", $manualEventLog
)

if (-not [string]::IsNullOrWhiteSpace($Video)) {
    $argsList += @("--video", $Video)
}
else {
    $argsList += @("--sequence", $Sequence, "--sequence-fps", "30")
}
if ($MaxFrames -gt 0) {
    $argsList += @("--max-frames", ([string]$MaxFrames))
}
if (-not $NoSlotDebug) {
    $argsList += @("--slot-debug-log", $slotDebugLog)
}
if (-not $NoWeek2Proof) {
    $argsList += @("--week2-proof-log", $week2ProofLog)
}

Write-Host "Manual V8 small-target demo"
Write-Host "Repo: $repoRoot"
Write-Host "Python: $python"
Write-Host "Device: $resolvedDevice"
Write-Host "LoRAT config: $LoRATConfig"
Write-Host "V8 head: $resolvedHead"
if (-not [string]::IsNullOrWhiteSpace($Video)) {
    Write-Host "Video: $Video"
}
else {
    Write-Host "Sequence: $Sequence"
}
Write-Host "Output folder: $runRoot"
Write-Host ""
Write-Host "When the first frame opens, draw the small body/head box, then press Enter/Space in the OpenCV selector."
Write-Host "Controls during playback: q quit, p pause, a add boxes, r manual re-anchor."
Write-Host ""
Write-Host "Command:"
Write-Host "  $python $($argsList -join ' ')"

if ($DryRun) {
    exit 0
}

Push-Location $repoRoot
try {
    & $python @argsList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
