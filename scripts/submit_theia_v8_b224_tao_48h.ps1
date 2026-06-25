[CmdletBinding()]
param(
    [string]$Username = "landonvw",
    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222,
    [string]$RemoteWorkRoot = "/work/landonvw",
    [string]$LocalTaoRoot = "data\raw\TAO_OW_SUBSET",
    [string]$RemoteTaoRoot = "/work/landonvw/TAO_OW_SUBSET",
    [switch]$SkipTaoUpload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RemoteUpdateRoot = "$RemoteWorkRoot/lorat-v8-live-updates"

$ProgramFiles = @(
    "programs\train_lorat_v8_head.py",
    "programs\bounding_box_v8_lorat_quality_batched.py",
    "programs\mot_common.py",
    "programs\exercise_lorat_mot.py"
)

$ScriptFiles = @(
    "scripts\theia_v8_train_heads.sbatch",
    "scripts\theia_v8_train_b224_48h_tao.sbatch"
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

    $ResolvedTaoRoot = Require-LocalPath $LocalTaoRoot
    $TaoAnnotationDir = Require-LocalPath (Join-Path $ResolvedTaoRoot "annotations")
    $TaoManifest = Require-LocalPath (Join-Path $ResolvedTaoRoot "subset_manifest.json")
    $TaoTrainZip = Require-LocalPath (Join-Path $ResolvedTaoRoot "frames\train\YFCC100M.zip")
    $TaoValZip = Require-LocalPath (Join-Path $ResolvedTaoRoot "frames\val\YFCC100M.zip")

    Write-Host "Submitting V8 B-224 mixed DanceTrack/MOT + TAO/YFCC training to Theia."
    Write-Host "Remote work root: $RemoteWorkRoot"
    Write-Host "Remote update overlay: $RemoteUpdateRoot"
    Write-Host "Remote TAO root: $RemoteTaoRoot"
    Write-Host "TAO upload mode: $(if ($SkipTaoUpload) { 'skip' } else { 'annotations + YFCC zip archives' })"
    Write-Host "You may be prompted for USC password/MFA by ssh/scp."

    Invoke-Checked "ssh" @(
        "-p", "$Port",
        "$Username@$RemoteHost",
        "mkdir -p '$RemoteUpdateRoot/programs' '$RemoteUpdateRoot/scripts' '$RemoteWorkRoot' '$RemoteTaoRoot/frames/train' '$RemoteTaoRoot/frames/val'"
    )

    $ScpProgramArgs = @("-P", "$Port")
    foreach ($Path in $ProgramFiles) {
        $ScpProgramArgs += (Resolve-Path $Path).Path
    }
    $ScpProgramArgs += "$Username@$RemoteHost`:$RemoteUpdateRoot/programs/"
    Invoke-Checked "scp" $ScpProgramArgs

    $ScpUpdateScriptArgs = @("-P", "$Port")
    foreach ($Path in $ScriptFiles) {
        $ScpUpdateScriptArgs += (Resolve-Path $Path).Path
    }
    $ScpUpdateScriptArgs += "$Username@$RemoteHost`:$RemoteUpdateRoot/scripts/"
    Invoke-Checked "scp" $ScpUpdateScriptArgs

    Invoke-Checked "scp" @(
        "-P", "$Port",
        (Resolve-Path "scripts\theia_v8_train_heads.sbatch").Path,
        (Resolve-Path "scripts\theia_v8_train_b224_48h_tao.sbatch").Path,
        "$Username@$RemoteHost`:$RemoteWorkRoot/"
    )

    if (-not $SkipTaoUpload) {
        Invoke-Checked "scp" @(
            "-P", "$Port",
            "-r",
            $TaoAnnotationDir,
            "$Username@$RemoteHost`:$RemoteTaoRoot/"
        )
        Invoke-Checked "scp" @(
            "-P", "$Port",
            $TaoManifest,
            "$Username@$RemoteHost`:$RemoteTaoRoot/"
        )
        Invoke-Checked "scp" @(
            "-P", "$Port",
            $TaoTrainZip,
            "$Username@$RemoteHost`:$RemoteTaoRoot/frames/train/YFCC100M.zip"
        )
        Invoke-Checked "scp" @(
            "-P", "$Port",
            $TaoValZip,
            "$Username@$RemoteHost`:$RemoteTaoRoot/frames/val/YFCC100M.zip"
        )
    }

    Invoke-Checked "ssh" @(
        "-p", "$Port",
        "$Username@$RemoteHost",
        "cd '$RemoteWorkRoot' && TAO_ROOT='$RemoteTaoRoot' sbatch '$RemoteWorkRoot/theia_v8_train_b224_48h_tao.sbatch'"
    )

    Write-Host ""
    Write-Host "Submitted. Check status with:"
    Write-Host "  ssh -p $Port $Username@$RemoteHost `"squeue -u $Username`""
}
finally {
    Pop-Location
}
