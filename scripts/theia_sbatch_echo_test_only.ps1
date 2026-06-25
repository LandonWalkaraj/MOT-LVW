[CmdletBinding()]
param(
    [string]$Username = "landonvw",
    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $ProjectRoot "outputs\theia_scheduler_checks"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "theia_sbatch_test_only_$Stamp.log"

$RemoteCommand = 'sbatch --test-only --partition=gpu-A100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=03:00:00 --wrap="echo test"'
$RemoteBashCommand = "sbatch --test-only --partition=gpu-A100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=03:00:00 --wrap='echo test'"

Write-Host "Running on Theia login shell:"
Write-Host "  $RemoteCommand"
Write-Host "This is --test-only and will not submit or run a Slurm job."
Write-Host "Logging output to: $LogPath"
Write-Host "You may be prompted for USC password/MFA."

Start-Transcript -Path $LogPath -Force | Out-Null
try {
    & ssh -p $Port "$Username@$RemoteHost" "bash" "-lc" $RemoteBashCommand
    if ($LASTEXITCODE -ne 0) {
        throw "ssh/sbatch --test-only failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
    Write-Host "Log saved to: $LogPath"
}
