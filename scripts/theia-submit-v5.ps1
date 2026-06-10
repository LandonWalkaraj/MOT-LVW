[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Username,

    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222,
    [string]$RemoteWorkRoot = "",
    [string]$ControlPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RemoteWorkRoot)) {
    $RemoteWorkRoot = "/work/$Username"
}

if ([string]::IsNullOrWhiteSpace($ControlPath)) {
    $ControlPath = Join-Path $env:TEMP "theia-codex-ssh-$Username@$RemoteHost-$Port"
}

$ControlPath = $ControlPath.Replace("\", "/")
$Target = "${Username}@${RemoteHost}"
$RemoteScript = "$RemoteWorkRoot/theia_v5_benchmark.sbatch"
$RemoteCommand = "sbatch '$RemoteScript' && squeue -u '$Username'"

$SshArgs = @(
    "-p", $Port,
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=$ControlPath",
    $Target,
    $RemoteCommand
)

& ssh @SshArgs
if ($LASTEXITCODE -ne 0) {
    throw "ssh/sbatch failed with exit code $LASTEXITCODE"
}
