[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Username,

    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222,
    [string]$BundleName = "theia_lorat_v8_train_bundle.tar.gz",
    [string]$RemoteWorkRoot = "",
    [string]$ControlPath = "",
    [string[]]$TrainSequences = @(),
    [string[]]$ValSequences = @(),
    [string]$TrainSplit = "train",
    [string]$ValSplit = "val",
    [string[]]$ModelFiles = @("base.bin", "large.bin", "giant.bin"),
    [int]$MaxTrainSequences = 0,
    [int]$MaxValSequences = 0,
    [bool]$IncludeMot17 = $true,
    [int]$MaxMot17Sequences = 0,
    [switch]$Upload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StageRoot = Join-Path $ProjectRoot "outputs\t8-stage"
$PayloadRoot = Join-Path $StageRoot "mot-v8-train"
$BundlePath = Join-Path $StageRoot $BundleName
$SbatchPath = Join-Path $ProjectRoot "scripts\theia_v8_train_heads.sbatch"

if ([string]::IsNullOrWhiteSpace($RemoteWorkRoot)) {
    $RemoteWorkRoot = "/work/$Username"
}

if (-not (Test-Path $SbatchPath)) {
    throw "Missing Slurm script: $SbatchPath"
}

if (Test-Path $StageRoot) {
    $ResolvedStage = (Resolve-Path $StageRoot).Path
    if (-not $ResolvedStage.StartsWith($ProjectRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove staging path outside project root: $ResolvedStage"
    }
    Remove-Item -LiteralPath $ResolvedStage -Recurse -Force
}

New-Item -ItemType Directory -Path $PayloadRoot -Force | Out-Null

function Copy-RequiredPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    $Source = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $Source)) {
        throw "Missing required path: $Source"
    }

    $Destination = Join-Path $PayloadRoot $RelativePath
    $DestinationParent = Split-Path $Destination -Parent
    New-Item -ItemType Directory -Path $DestinationParent -Force | Out-Null

    $SourceItem = Get-Item -LiteralPath $Source
    if ($SourceItem.PSIsContainer) {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
        & robocopy $Source $Destination /E /XD "__pycache__" ".git" /XF "*.pyc" /NFL /NDL /NJH /NJS /NP | Out-Host
        if ($LASTEXITCODE -ge 8) {
            throw "robocopy failed for $RelativePath with exit code $LASTEXITCODE"
        }
    } else {
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Resolve-SequenceNames {
    param(
        [Parameter(Mandatory = $true)][string]$Split,
        [string[]]$Names,
        [int]$MaxCount
    )

    if ($Names.Count -gt 0) {
        return $Names
    }

    $SplitRoot = Join-Path $ProjectRoot "data\raw\DanceTrack\$Split\$Split"
    if (-not (Test-Path $SplitRoot)) {
        throw "Missing DanceTrack split root: $SplitRoot"
    }

    $Resolved = Get-ChildItem -Path $SplitRoot -Directory |
        Sort-Object Name |
        Select-Object -ExpandProperty Name
    if ($MaxCount -gt 0) {
        $Resolved = $Resolved | Select-Object -First $MaxCount
    }
    return @($Resolved)
}

function Copy-DanceTrackSplit {
    param(
        [Parameter(Mandatory = $true)][string]$Split,
        [Parameter(Mandatory = $true)][string[]]$Sequences
    )

    $SourceRoot = Join-Path $ProjectRoot "data\raw\DanceTrack\$Split\$Split"
    $DestRoot = Join-Path $PayloadRoot "data\raw\DanceTrack\$Split\$Split"
    New-Item -ItemType Directory -Path $DestRoot -Force | Out-Null

    foreach ($Sequence in $Sequences) {
        $Source = Join-Path $SourceRoot $Sequence
        if (-not (Test-Path $Source)) {
            throw "Missing DanceTrack $Split sequence: $Source"
        }
        $Destination = Join-Path $DestRoot $Sequence
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
        & robocopy $Source $Destination /E /XD "__pycache__" ".git" /XF "*.pyc" /NFL /NDL /NJH /NJS /NP | Out-Host
        if ($LASTEXITCODE -ge 8) {
            throw "robocopy failed for DanceTrack $Split sequence $Sequence with exit code $LASTEXITCODE"
        }
    }
}

function Resolve-MOT17SequenceNames {
    param([int]$MaxCount)

    $MOT17Root = Join-Path $ProjectRoot "data\raw\MOTChallenge\MOT17\train"
    if (-not (Test-Path $MOT17Root)) {
        if ($IncludeMot17) {
            throw "Missing MOT17 train root: $MOT17Root"
        }
        return @()
    }

    $Resolved = Get-ChildItem -Path $MOT17Root -Directory |
        Sort-Object Name |
        Select-Object -ExpandProperty Name
    if ($MaxCount -gt 0) {
        $Resolved = $Resolved | Select-Object -First $MaxCount
    }
    return @($Resolved)
}

function Copy-SequenceSet {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestRoot,
        [Parameter(Mandatory = $true)][string[]]$Sequences,
        [Parameter(Mandatory = $true)][string]$Label
    )

    New-Item -ItemType Directory -Path $DestRoot -Force | Out-Null
    foreach ($Sequence in $Sequences) {
        $Source = Join-Path $SourceRoot $Sequence
        if (-not (Test-Path $Source)) {
            throw "Missing $Label sequence: $Source"
        }
        $Destination = Join-Path $DestRoot $Sequence
        if (Test-Path $Destination) {
            throw "Duplicate destination sequence in combined dataset: $Destination"
        }
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
        & robocopy $Source $Destination /E /XD "__pycache__" ".git" /XF "*.pyc" /NFL /NDL /NJH /NJS /NP | Out-Host
        if ($LASTEXITCODE -ge 8) {
            throw "robocopy failed for $Label sequence $Sequence with exit code $LASTEXITCODE"
        }
    }
}

Copy-RequiredPath "programs"
Copy-RequiredPath "requirements.txt"
Copy-RequiredPath "external\LoRAT-main"
Copy-RequiredPath "scripts\theia_v8_train_heads.sbatch"

$ResolvedTrainSequences = Resolve-SequenceNames -Split $TrainSplit -Names $TrainSequences -MaxCount $MaxTrainSequences
$ResolvedValSequences = Resolve-SequenceNames -Split $ValSplit -Names $ValSequences -MaxCount $MaxValSequences
$ResolvedMot17Sequences = @()
if ($IncludeMot17) {
    $ResolvedMot17Sequences = Resolve-MOT17SequenceNames -MaxCount $MaxMot17Sequences
}

$CombinedTrainRoot = Join-Path $PayloadRoot "data\raw\Combined\train\train"
$CombinedValRoot = Join-Path $PayloadRoot "data\raw\Combined\val\val"
$DanceTrackTrainRoot = Join-Path $ProjectRoot "data\raw\DanceTrack\$TrainSplit\$TrainSplit"
$DanceTrackValRoot = Join-Path $ProjectRoot "data\raw\DanceTrack\$ValSplit\$ValSplit"
$Mot17TrainRoot = Join-Path $ProjectRoot "data\raw\MOTChallenge\MOT17\train"

Copy-SequenceSet -SourceRoot $DanceTrackTrainRoot -DestRoot $CombinedTrainRoot -Sequences $ResolvedTrainSequences -Label "DanceTrack $TrainSplit"
if ($ResolvedMot17Sequences.Count -gt 0) {
    Copy-SequenceSet -SourceRoot $Mot17TrainRoot -DestRoot $CombinedTrainRoot -Sequences $ResolvedMot17Sequences -Label "MOT17 train"
}
Copy-SequenceSet -SourceRoot $DanceTrackValRoot -DestRoot $CombinedValRoot -Sequences $ResolvedValSequences -Label "DanceTrack $ValSplit"

$ModelDest = Join-Path $PayloadRoot "models\lorat"
New-Item -ItemType Directory -Path $ModelDest -Force | Out-Null
foreach ($ModelFile in $ModelFiles) {
    $Source = Join-Path $ProjectRoot "models\lorat\$ModelFile"
    if (-not (Test-Path $Source)) {
        throw "Missing model file: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $ModelDest -Force
}

Get-ChildItem -Path $PayloadRoot -Directory -Filter "__pycache__" -Recurse |
    Remove-Item -Recurse -Force
Get-ChildItem -Path $PayloadRoot -Directory -Filter ".git" -Recurse |
    Remove-Item -Recurse -Force
Get-ChildItem -Path $PayloadRoot -File -Filter "*.pyc" -Recurse |
    Remove-Item -Force

if (Test-Path $BundlePath) {
    Remove-Item -LiteralPath $BundlePath -Force
}

& tar -czf $BundlePath -C $StageRoot "mot-v8-train"
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

$BundleSizeMb = [Math]::Round((Get-Item $BundlePath).Length / 1MB, 1)
Write-Host "Created bundle: $BundlePath ($BundleSizeMb MB)"
Write-Host "Included train split '$TrainSplit' sequences: $($ResolvedTrainSequences -join ', ')"
Write-Host "Included val split '$ValSplit' sequences: $($ResolvedValSequences -join ', ')"
Write-Host "Included MOT17 train sequences: $($ResolvedMot17Sequences -join ', ')"
Write-Host "Payload dataset root: data/raw/Combined"
Write-Host "Included model files: $($ModelFiles -join ', ')"
Write-Host "Upload flag: $Upload"

if (-not $Upload) {
    Write-Host "Not uploading. Re-run with -Upload when ready."
    Write-Host "Future submit command after upload:"
    Write-Host "  sbatch $RemoteWorkRoot/theia_v8_train_heads.sbatch"
    return
}

Write-Host "Uploading bundle and Slurm script to ${Username}@${RemoteHost}:$RemoteWorkRoot/"
$ScpArgs = @("-P", $Port)
if (-not [string]::IsNullOrWhiteSpace($ControlPath)) {
    $ScpArgs += @("-o", "ControlMaster=auto", "-o", "ControlPath=$ControlPath")
}
$ScpArgs += @($BundlePath, $SbatchPath, "${Username}@${RemoteHost}:$RemoteWorkRoot/")
& scp @ScpArgs
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}

Write-Host "Upload complete."
Write-Host "Submit from the persistent SSH session with:"
Write-Host "  sbatch $RemoteWorkRoot/theia_v8_train_heads.sbatch"
