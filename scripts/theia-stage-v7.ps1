[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Username,

    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222,
    [string]$BundleName = "theia_lorat_v7_bundle.tar.gz",
    [string]$RemoteWorkRoot = "",
    [string]$ControlPath = "",
    [string[]]$Sequences = @("dancetrack0065"),
    [string[]]$ModelFiles = @("base.bin", "large.bin", "giant.bin"),
    [switch]$AllSequences,
    [switch]$NoUpload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StageRoot = Join-Path $ProjectRoot "outputs\theia-stage-v7"
$PayloadRoot = Join-Path $StageRoot "mot-v7"
$BundlePath = Join-Path $StageRoot $BundleName
$SbatchPath = Join-Path $ProjectRoot "scripts\theia_v7_benchmark.sbatch"

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

Copy-RequiredPath "programs"
Copy-RequiredPath "requirements.txt"
Copy-RequiredPath "external\LoRAT-main"
Copy-RequiredPath "scripts\theia_v7_benchmark.sbatch"

if ($AllSequences) {
    $Sequences = Get-ChildItem -Path (Join-Path $ProjectRoot "data\raw\DanceTrack\val\val") -Directory |
        Sort-Object Name |
        Select-Object -ExpandProperty Name
}

$ModelDest = Join-Path $PayloadRoot "models\lorat"
New-Item -ItemType Directory -Path $ModelDest -Force | Out-Null
foreach ($ModelFile in $ModelFiles) {
    $Source = Join-Path $ProjectRoot "models\lorat\$ModelFile"
    if (-not (Test-Path $Source)) {
        throw "Missing model file: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $ModelDest -Force
}

$SequenceDest = Join-Path $PayloadRoot "data\raw\DanceTrack\val\val"
New-Item -ItemType Directory -Path $SequenceDest -Force | Out-Null
foreach ($Sequence in $Sequences) {
    $Source = Join-Path $ProjectRoot "data\raw\DanceTrack\val\val\$Sequence"
    if (-not (Test-Path $Source)) {
        throw "Missing DanceTrack sequence: $Source"
    }
    $Destination = Join-Path $SequenceDest $Sequence
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    & robocopy $Source $Destination /E /XD "__pycache__" ".git" /XF "*.pyc" /NFL /NDL /NJH /NJS /NP | Out-Host
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed for DanceTrack sequence $Sequence with exit code $LASTEXITCODE"
    }
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

& tar -czf $BundlePath -C $StageRoot "mot-v7"
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

$BundleSizeMb = [Math]::Round((Get-Item $BundlePath).Length / 1MB, 1)
Write-Host "Created bundle: $BundlePath ($BundleSizeMb MB)"
Write-Host "Included sequences: $($Sequences -join ', ')"
Write-Host "Included model files: $($ModelFiles -join ', ')"

if ($NoUpload) {
    Write-Host "NoUpload was set; skipping scp."
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
Write-Host "  sbatch $RemoteWorkRoot/theia_v7_benchmark.sbatch"
