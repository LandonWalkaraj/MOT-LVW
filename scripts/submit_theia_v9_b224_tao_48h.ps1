[CmdletBinding()]
param(
    [string]$Username = "landonvw",
    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222,
    [string]$RemoteWorkRoot = "/work/landonvw",
    [string]$RemoteTaoRoot = "/work/landonvw/TAO_OW_SUBSET",
    [string]$RemoteLaSOTRoot = "/work/landonvw/LaSOT_subset",
    [string]$LocalLaSOTRoot = "data\raw\LaSOT_subset",
    [string]$RemoteLaSOTArchive = "/work/landonvw/LaSOT_subset.tar.gz",
    [switch]$SkipLaSOTUpload,
    [switch]$ForceRebuildLaSOTArchive
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RemoteUpdateRoot = "$RemoteWorkRoot/lorat-v9-live-updates"
$StageRoot = Join-Path $ProjectRoot "outputs\theia_uploads"
$LocalLaSOTArchive = Join-Path $StageRoot "LaSOT_subset.tar.gz"

$ProgramFiles = @(
    "programs\train_lorat_v9_local_search_head.py",
    "programs\bounding_box_v9_lorat_local_search.py",
    "programs\benchmark_lorat_v9.py",
    "programs\export_tao_to_mot_sequences.py",
    "programs\train_lorat_v8_head.py",
    "programs\bounding_box_v8_lorat_quality_batched.py",
    "programs\benchmark_lorat_v8.py",
    "programs\benchmark_lorat_mot.py",
    "programs\mot_common.py",
    "programs\exercise_lorat_mot.py"
)

$ScriptFiles = @(
    "scripts\theia_v9_train_b224_48h_tao.sbatch"
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

function Require-LocalPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Missing local path: $Path"
    }
    return (Resolve-Path $Path).Path
}

Push-Location $ProjectRoot
try {
    foreach ($Path in ($ProgramFiles + $ScriptFiles)) {
        Require-LocalPath $Path | Out-Null
    }

    Write-Host "Submitting V9 B-224 mixed DanceTrack/MOT17/TAO training to Theia."
    Write-Host "Remote work root: $RemoteWorkRoot"
    Write-Host "Remote update overlay: $RemoteUpdateRoot"
    Write-Host "Remote TAO root: $RemoteTaoRoot"
    Write-Host "Remote LaSOT root: $RemoteLaSOTRoot"
    Write-Host "Remote LaSOT archive: $RemoteLaSOTArchive"
    Write-Host "Upload mode: code/scripts plus LaSOT subset archive unless -SkipLaSOTUpload is set."
    Write-Host "You may be prompted for USC password/MFA by ssh/scp."

    if (-not $SkipLaSOTUpload) {
        $ResolvedLaSOTRoot = Require-LocalPath $LocalLaSOTRoot
        New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null
        if ($ForceRebuildLaSOTArchive -and (Test-Path -LiteralPath $LocalLaSOTArchive)) {
            Remove-Item -LiteralPath $LocalLaSOTArchive -Force
        }
        if (Test-Path -LiteralPath $LocalLaSOTArchive) {
            Write-Host ""
            Write-Host "Reusing existing LaSOT subset archive: $LocalLaSOTArchive"
        }
        else {
            Write-Host ""
            Write-Host "Creating LaSOT subset archive: $LocalLaSOTArchive"
            Invoke-Checked "tar" @(
                "-czf",
                $LocalLaSOTArchive,
                "-C",
                (Split-Path -Parent $ResolvedLaSOTRoot),
                (Split-Path -Leaf $ResolvedLaSOTRoot)
            )
        }
    }

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
        (Resolve-Path "scripts\theia_v9_train_b224_48h_tao.sbatch").Path,
        "$Username@$RemoteHost`:$RemoteWorkRoot/"
    )

    if (-not $SkipLaSOTUpload) {
        Invoke-Checked "scp" @(
            "-P", "$Port",
            (Resolve-Path $LocalLaSOTArchive).Path,
            "$Username@$RemoteHost`:$RemoteLaSOTArchive"
        )
    }

    Invoke-Checked "ssh" @(
        "-p", "$Port",
        "$Username@$RemoteHost",
        "cd '$RemoteWorkRoot' && TAO_ROOT='$RemoteTaoRoot' LASOT_ROOT='$RemoteLaSOTRoot' LASOT_ARCHIVE='$RemoteLaSOTArchive' sbatch '$RemoteWorkRoot/theia_v9_train_b224_48h_tao.sbatch'"
    )

    Write-Host ""
    Write-Host "Submitted. Check status with:"
    Write-Host "  ssh -p $Port $Username@$RemoteHost `"squeue -u $Username`""
}
finally {
    Pop-Location
}
