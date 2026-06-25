[CmdletBinding()]
param(
    [string]$Username = "landonvw",
    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222,
    [string]$RemoteWorkRoot = "/work/landonvw",
    [string]$HeadWeightsRoot = "",
    [ValidateSet("best", "latest")]
    [string]$HeadCheckpoint = "best",
    [string]$ResultRoot = "",
    [string]$Sequences = "dancetrack0065",
    [string]$TrackCounts = "1,2,3,4,5",
    [string]$CompareConfigs = "B-224",
    [ValidateSet("largest", "smallest", "area-window", "middle")]
    [string]$InitSelection = "largest",
    [double]$InitMinArea = 0,
    [double]$InitMaxArea = 0,
    [string]$InitTrackIds = "",
    [int]$MaxFrames = 0,
    [switch]$NoVideo,
    [string]$ControlledOcclusionDurations = "0,5,10,20,40,80"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RemoteUpdateRoot = "$RemoteWorkRoot/lorat-v8-live-updates"

$ProgramFiles = @(
    "programs\benchmark_lorat_v8.py",
    "programs\benchmark_lorat_week3.py",
    "programs\benchmark_lorat_mot.py",
    "programs\bounding_box_v8_lorat_quality_batched.py",
    "programs\mot_common.py",
    "programs\exercise_lorat_mot.py"
)

$ScriptFiles = @(
    "scripts\theia_v8_week3_benchmark.sbatch"
)

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    Write-Host ""
    Write-Host ">> $FilePath $($ArgumentList -join ' ')"
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function ShellQuote {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)
    return "'" + ($Value -replace "'", "'\''") + "'"
}

Push-Location $ProjectRoot
try {
    foreach ($Path in ($ProgramFiles + $ScriptFiles)) {
        if (-not (Test-Path $Path)) {
            throw "Missing local file: $Path"
        }
    }

    if ([string]::IsNullOrWhiteSpace($ResultRoot)) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $ResultRoot = "$RemoteWorkRoot/lorat-v8-week3-results-$stamp"
    }

    Write-Host "Submitting V8 Week 3 benchmark to Theia."
    Write-Host "Remote work root: $RemoteWorkRoot"
    Write-Host "Remote update overlay: $RemoteUpdateRoot"
    Write-Host "Head weights root: $(if ($HeadWeightsRoot) { $HeadWeightsRoot } else { 'auto-detect newest complete root' })"
    Write-Host "Head checkpoint: $HeadCheckpoint"
    Write-Host "Result root: $ResultRoot"
    Write-Host "Sequences: $Sequences"
    Write-Host "Track counts: $TrackCounts"
    Write-Host "Compare configs: $CompareConfigs"
    Write-Host "Init selection: $InitSelection"
    Write-Host "Init min area: $InitMinArea"
    Write-Host "Init max area: $InitMaxArea"
    Write-Host "Init track ids: $(if ($InitTrackIds) { $InitTrackIds } else { 'none' })"
    Write-Host "Max frames: $MaxFrames"
    Write-Host "Save video: $(-not $NoVideo)"

    Invoke-Checked "ssh" @(
        "-p", "$Port",
        "$Username@$RemoteHost",
        "mkdir -p '$RemoteUpdateRoot/programs' '$RemoteUpdateRoot/scripts' '$RemoteWorkRoot'"
    )

    $ScpProgramArgs = @("-P", "$Port")
    foreach ($Path in $ProgramFiles) {
        $ScpProgramArgs += (Resolve-Path $Path).Path
    }
    $ScpProgramArgs += "$Username@$RemoteHost`:$RemoteUpdateRoot/programs/"
    Invoke-Checked "scp" $ScpProgramArgs

    $ScpScriptArgs = @("-P", "$Port")
    foreach ($Path in $ScriptFiles) {
        $ScpScriptArgs += (Resolve-Path $Path).Path
    }
    $ScpScriptArgs += "$Username@$RemoteHost`:$RemoteUpdateRoot/scripts/"
    Invoke-Checked "scp" $ScpScriptArgs

    Invoke-Checked "scp" @(
        "-P", "$Port",
        (Resolve-Path "scripts\theia_v8_week3_benchmark.sbatch").Path,
        "$Username@$RemoteHost`:$RemoteWorkRoot/"
    )

    $remoteEnv = @(
        "RESULT_ROOT=$(ShellQuote $ResultRoot)",
        "HEAD_CHECKPOINT=$(ShellQuote $HeadCheckpoint)",
        "SEQUENCES=$(ShellQuote $Sequences)",
        "TRACK_COUNTS=$(ShellQuote $TrackCounts)",
        "COMPARE_CONFIGS=$(ShellQuote $CompareConfigs)",
        "INIT_SELECTION=$(ShellQuote $InitSelection)",
        "INIT_MIN_AREA=$(ShellQuote ([string]$InitMinArea))",
        "INIT_MAX_AREA=$(ShellQuote ([string]$InitMaxArea))",
        "INIT_TRACK_IDS=$(ShellQuote $InitTrackIds)",
        "MAX_FRAMES=$(ShellQuote ([string]$MaxFrames))",
        "SAVE_VIDEO=$(if ($NoVideo) { '0' } else { '1' })",
        "CONTROLLED_OCCLUSION_DURATIONS=$(ShellQuote $ControlledOcclusionDurations)"
    )
    if (-not [string]::IsNullOrWhiteSpace($HeadWeightsRoot)) {
        $remoteEnv += "HEAD_WEIGHTS_ROOT=$(ShellQuote $HeadWeightsRoot)"
        $remoteEnv += "HEAD_WEIGHTS_ROOT_AUTO=0"
    }
    $remoteCommand = "cd $(ShellQuote $RemoteWorkRoot) && " + ($remoteEnv -join " ") + " sbatch '$RemoteWorkRoot/theia_v8_week3_benchmark.sbatch'"

    Invoke-Checked "ssh" @(
        "-p", "$Port",
        "$Username@$RemoteHost",
        $remoteCommand
    )

    Write-Host ""
    Write-Host "Submitted. Check status with:"
    Write-Host "  ssh -p $Port $Username@$RemoteHost `"squeue -u $Username`""
}
finally {
    Pop-Location
}
