[CmdletBinding()]
param(
    [string]$Username = "landonvw",
    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222,
    [string]$RemoteWorkRoot = "/work/landonvw",
    [string]$HeadWeightsRoot = "/work/landonvw/lorat-v8-train-results/checkpoints",
    [ValidateSet("best", "latest")]
    [string]$HeadCheckpoint = "best",
    [string]$ResultRoot = "",
    [string]$Sequences = "dancetrack0065",
    [string]$TrackCounts = "1,2,3,4,5",
    [string]$CompareConfigs = "B-224",
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
    "scripts\theia_v8_week3_benchmark.sbatch",
    "scripts\theia_v8_week3_largest_smallest_benchmark.sbatch"
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
        $ResultRoot = "$RemoteWorkRoot/lorat-v8-week3-largest-smallest-$stamp"
    }

    Write-Host "Submitting V8 Week 3 largest+smallest benchmark to Theia."
    Write-Host "Remote work root: $RemoteWorkRoot"
    Write-Host "Remote update overlay: $RemoteUpdateRoot"
    Write-Host "Head weights root: $HeadWeightsRoot"
    Write-Host "Head weights auto-detect: prefer newest valid checkpoint root"
    Write-Host "Head checkpoint: $HeadCheckpoint"
    Write-Host "Result root base: $ResultRoot"
    Write-Host "Sequences: $Sequences"
    Write-Host "Track counts: $TrackCounts"
    Write-Host "Compare configs: $CompareConfigs"
    Write-Host "Max frames: $MaxFrames"
    Write-Host "Save video: $(-not $NoVideo)"
    Write-Host "Controlled occlusion durations: $(if ($ControlledOcclusionDurations) { $ControlledOcclusionDurations } else { 'disabled' })"
    Write-Host "Init order: largest then smallest"

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

    $ScpRootScriptArgs = @("-P", "$Port")
    foreach ($Path in $ScriptFiles) {
        $ScpRootScriptArgs += (Resolve-Path $Path).Path
    }
    $ScpRootScriptArgs += "$Username@$RemoteHost`:$RemoteWorkRoot/"
    Invoke-Checked "scp" $ScpRootScriptArgs

    $remoteEnv = @(
        "RESULT_ROOT=$(ShellQuote $ResultRoot)",
        "HEAD_WEIGHTS_ROOT=$(ShellQuote $HeadWeightsRoot)",
        "HEAD_WEIGHTS_ROOT_AUTO=1",
        "HEAD_CHECKPOINT=$(ShellQuote $HeadCheckpoint)",
        "SEQUENCES=$(ShellQuote $Sequences)",
        "TRACK_COUNTS=$(ShellQuote $TrackCounts)",
        "COMPARE_CONFIGS=$(ShellQuote $CompareConfigs)",
        "MAX_FRAMES=$(ShellQuote ([string]$MaxFrames))",
        "SAVE_VIDEO=$(if ($NoVideo) { '0' } else { '1' })",
        "CONTROLLED_OCCLUSION_DURATIONS=$(ShellQuote $ControlledOcclusionDurations)",
        "INIT_SELECTION_ORDER='largest smallest'",
        "INIT_MIN_AREA=0",
        "INIT_MAX_AREA=0",
        "INIT_TRACK_IDS=''"
    )
    $remoteCommand = "cd $(ShellQuote $RemoteWorkRoot) && " + ($remoteEnv -join " ") + " sbatch '$RemoteWorkRoot/theia_v8_week3_largest_smallest_benchmark.sbatch'"

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
