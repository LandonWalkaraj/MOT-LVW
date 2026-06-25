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
$LogPath = Join-Path $LogDir "theia_echo_test_submit_$Stamp.log"

$RemoteCommand = @'
set -euo pipefail
echo "=== scheduler context ==="
hostname
date
echo
echo "=== existing queue ==="
squeue -u landonvw || true
echo
probe_script="/work/landonvw/slurm-echo-test-submit-check.sbatch"
cat > "$probe_script" <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=slurm-echo-test
#SBATCH --partition=gpu-A100
#SBATCH --account=rc_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/work/landonvw/slurm-echo-test-%j.out

set -euo pipefail
echo "echo test"
hostname
date
SBATCH
echo "=== sbatch --test-only estimate ==="
sbatch --test-only "$probe_script"
echo
echo "=== actual tiny Slurm job submit ==="
jobid=$(sbatch --parsable "$probe_script")
echo "submitted_jobid=$jobid"
echo "output_file=/work/landonvw/slurm-echo-test-$jobid.out"
echo
echo "=== submitted job status ==="
squeue -j "$jobid" -o "%.18i %.9P %.24j %.8u %.2t %.12M %.6D %R" || true
'@

Write-Host "Logging scheduler output to: $LogPath"
Write-Host "You may be prompted for USC password/MFA."
Start-Transcript -Path $LogPath -Force | Out-Null
try {
    & ssh -p $Port "$Username@$RemoteHost" $RemoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "ssh/sbatch command failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
    Write-Host "Log saved to: $LogPath"
}
