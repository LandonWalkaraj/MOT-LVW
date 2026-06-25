param(
    [string]$DownloadsRoot = "$env:USERPROFILE\Downloads",
    [string]$CheckpointSource = "",
    [string]$ExtractCacheRoot = "",
    [string]$OutputRoot = "",
    [string[]]$Sequences = @("dancetrack0065"),
    [string]$TrackCounts = "1,2,3,4,5",
    [string[]]$CompareConfigs = @("B-224"),
    [ValidateSet("best", "latest")]
    [string]$HeadCheckpoint = "best",
    [ValidateSet("largest", "smallest", "area-window", "middle")]
    [string]$InitSelection = "largest",
    [double]$InitMinArea = 0,
    [double]$InitMaxArea = 0,
    [int[]]$InitTrackId = @(),
    [int]$MaxFrames = 0,
    [string]$DatasetRoot = "",
    [string]$Split = "val",
    [string]$Device = "cuda:0",
    [switch]$NoVideo,
    [switch]$ForceRefresh,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Config-Key {
    param([string]$Config)
    return ($Config -replace "-", "_")
}

function Checkpoint-File-Names {
    param([string]$Key)
    return @(
        "v8_head_${Key}_best_by_val_iou.pt",
        "v8_head_${Key}_latest.pt"
    )
}

function Requested-Config-Keys {
    param([string[]]$Configs)
    $expanded = New-Object System.Collections.Generic.List[string]
    foreach ($config in $Configs) {
        if ($config -eq "all") {
            foreach ($item in @("B-224", "L-224", "g-224")) {
                $expanded.Add((Config-Key $item))
            }
        }
        else {
            $expanded.Add((Config-Key $config))
        }
    }
    return @($expanded | Select-Object -Unique)
}

function Test-Checkpoint-Root {
    param(
        [string]$Root,
        [string[]]$Keys
    )
    if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root -PathType Container)) {
        return $false
    }
    foreach ($key in $Keys) {
        $configDir = Join-Path $Root $key
        if (-not (Test-Path -LiteralPath $configDir -PathType Container)) {
            return $false
        }
        $found = $false
        foreach ($name in (Checkpoint-File-Names $key)) {
            if (Test-Path -LiteralPath (Join-Path $configDir $name) -PathType Leaf) {
                $found = $true
                break
            }
        }
        if (-not $found) {
            return $false
        }
    }
    return $true
}

function Resolve-Checkpoint-Root-From-Directory {
    param([string]$Path)
    if (Test-Path -LiteralPath (Join-Path $Path "checkpoints") -PathType Container) {
        return (Join-Path $Path "checkpoints")
    }
    return $Path
}

function Zip-Has-Checkpoints {
    param(
        [string]$ZipPath,
        [string[]]$Keys
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $names = @{}
        foreach ($entry in $zip.Entries) {
            $names[$entry.FullName] = $true
        }
        foreach ($key in $Keys) {
            $hasKey = $false
            foreach ($name in (Checkpoint-File-Names $key)) {
                if ($names.ContainsKey("checkpoints/$key/$name")) {
                    $hasKey = $true
                    break
                }
            }
            if (-not $hasKey) {
                return $false
            }
        }
        return $true
    }
    finally {
        $zip.Dispose()
    }
}

function Join-Entry-Path {
    param(
        [string]$Root,
        [string]$EntryName
    )
    $path = $Root
    foreach ($part in ($EntryName -split "/")) {
        if ([string]::IsNullOrWhiteSpace($part)) {
            continue
        }
        $path = Join-Path $path $part
    }
    return $path
}

function Expand-Needed-Checkpoints {
    param(
        [string]$ZipPath,
        [string]$Destination,
        [string[]]$Keys,
        [bool]$Refresh
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($key in $Keys) {
            foreach ($name in (Checkpoint-File-Names $key)) {
                $entryName = "checkpoints/$key/$name"
                $entry = $zip.GetEntry($entryName)
                if ($null -eq $entry) {
                    continue
                }
                $target = Join-Entry-Path $Destination $entryName
                $targetDir = Split-Path -Parent $target
                New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
                $needsExtract = $Refresh -or -not (Test-Path -LiteralPath $target -PathType Leaf)
                if (-not $needsExtract) {
                    $existing = Get-Item -LiteralPath $target
                    $needsExtract = ($existing.Length -ne $entry.Length)
                }
                if ($needsExtract) {
                    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
                }
            }
        }
    }
    finally {
        $zip.Dispose()
    }
    return (Join-Path $Destination "checkpoints")
}

function Resolve-Local-Checkpoint-Root {
    param(
        [string]$Source,
        [string]$Downloads,
        [string]$CacheRoot,
        [string[]]$Keys,
        [bool]$Refresh
    )

    if (-not [string]::IsNullOrWhiteSpace($Source)) {
        $resolvedSource = (Resolve-Path -LiteralPath $Source).Path
        if (Test-Path -LiteralPath $resolvedSource -PathType Container) {
            $root = Resolve-Checkpoint-Root-From-Directory $resolvedSource
            if (Test-Checkpoint-Root $root $Keys) {
                return $root
            }
            throw "Checkpoint source does not contain the requested V8 head files: $resolvedSource"
        }
        if ($resolvedSource.EndsWith(".zip", [StringComparison]::OrdinalIgnoreCase)) {
            if (-not (Zip-Has-Checkpoints $resolvedSource $Keys)) {
                throw "Checkpoint zip does not contain the requested V8 head files: $resolvedSource"
            }
            $destination = Join-Path $CacheRoot ([IO.Path]::GetFileNameWithoutExtension($resolvedSource))
            return (Expand-Needed-Checkpoints $resolvedSource $destination $Keys $Refresh)
        }
        throw "Unsupported checkpoint source: $resolvedSource"
    }

    $directoryCandidates = @()
    if (Test-Path -LiteralPath $Downloads -PathType Container) {
        $directoryCandidates = Get-ChildItem -LiteralPath $Downloads -Directory -Filter "lorat-v8-train-results*" |
            Sort-Object LastWriteTime -Descending
    }
    foreach ($candidate in $directoryCandidates) {
        $root = Resolve-Checkpoint-Root-From-Directory $candidate.FullName
        if (Test-Checkpoint-Root $root $Keys) {
            return $root
        }
    }

    $zipCandidates = @()
    if (Test-Path -LiteralPath $Downloads -PathType Container) {
        $zipCandidates = Get-ChildItem -LiteralPath $Downloads -File -Filter "lorat-v8-train-results*.zip" |
            Sort-Object LastWriteTime -Descending
    }
    foreach ($candidate in $zipCandidates) {
        if (Zip-Has-Checkpoints $candidate.FullName $Keys) {
            $destination = Join-Path $CacheRoot $candidate.BaseName
            return (Expand-Needed-Checkpoints $candidate.FullName $destination $Keys $Refresh)
        }
    }

    throw "Could not find V8 training checkpoints in $Downloads. Pass -CheckpointSource with a result folder or zip."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($ExtractCacheRoot)) {
    $ExtractCacheRoot = Join-Path $repoRoot "models\lorat\v8_local_checkpoints"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "outputs\benchmarks\lorat-v8-local"
}

$keys = Requested-Config-Keys $CompareConfigs
$checkpointRoot = Resolve-Local-Checkpoint-Root `
    -Source $CheckpointSource `
    -Downloads $DownloadsRoot `
    -CacheRoot $ExtractCacheRoot `
    -Keys $keys `
    -Refresh ([bool]$ForceRefresh)

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $python = $venvPython
}
else {
    $python = "python"
}

$benchmark = Join-Path $repoRoot "programs\benchmark_lorat_week3.py"
$loratRoot = Join-Path $repoRoot "external\LoRAT-main"

$argsList = @(
    $benchmark,
    "--split", $Split,
    "--lorat-root", $loratRoot,
    "--device", $Device,
    "--gpu-profile", "local",
    "--track-counts", $TrackCounts,
    "--v8-head-weights-root", $checkpointRoot,
    "--v8-head-checkpoint", $HeadCheckpoint,
    "--init-selection", $InitSelection,
    "--init-min-area", ([string]$InitMinArea),
    "--init-max-area", ([string]$InitMaxArea),
    "--max-frames", ([string]$MaxFrames),
    "--output-root", $OutputRoot
)

if (-not [string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $argsList += @("--dataset-root", $DatasetRoot)
}
foreach ($sequence in $Sequences) {
    $argsList += @("--sequence", $sequence)
}
foreach ($trackId in $InitTrackId) {
    $argsList += @("--init-track-id", ([string]$trackId))
}
$argsList += "--compare-configs"
foreach ($config in $CompareConfigs) {
    $argsList += $config
}
if (-not $NoVideo) {
    $argsList += "--save-video"
}

Write-Host "Using V8 checkpoint root: $checkpointRoot"
Write-Host "Output root: $OutputRoot"
Write-Host "Python: $python"
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
