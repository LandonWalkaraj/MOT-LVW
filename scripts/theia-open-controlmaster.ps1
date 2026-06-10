[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Username,

    [string]$RemoteHost = "login-theia.rc.sc.edu",
    [int]$Port = 222,
    [string]$Persist = "4h",
    [string]$ControlPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ControlPath)) {
    $ControlPath = Join-Path $env:TEMP "theia-codex-ssh-$Username@$RemoteHost-$Port"
}

$ControlPath = $ControlPath.Replace("\", "/")
$Target = "${Username}@${RemoteHost}"
$EscapedControlPath = $ControlPath.Replace("'", "''")
$EscapedTarget = $Target.Replace("'", "''")

$Command = @"
Write-Host 'Opening persistent Theia SSH master for $EscapedTarget'
Write-Host 'ControlPath: $EscapedControlPath'
Write-Host 'Leave this window open while Codex uploads/submits/checks jobs.'
ssh -M -S '$EscapedControlPath' -o ControlMaster=yes -o ControlPersist=$Persist -N -p $Port '$EscapedTarget'
Write-Host ''
Write-Host 'Theia SSH master exited.'
"@

Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
    $Command
)

Write-Host "Started visible SSH master window for $Target."
Write-Host "ControlPath: $ControlPath"
