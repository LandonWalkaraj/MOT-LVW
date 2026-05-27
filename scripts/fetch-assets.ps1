[CmdletBinding()]
param(
    [string[]]$Asset = @("week1-core"),
    [string]$Root = "",
    [switch]$NoExtract,
    [switch]$Force,
    [switch]$List
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $scriptDirectory = if ($PSScriptRoot) {
        $PSScriptRoot
    } else {
        Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    $Root = (Resolve-Path (Join-Path $scriptDirectory "..")).Path
}

$Asset = @(
    foreach ($entry in $Asset) {
        foreach ($part in ([string]$entry -split ",")) {
            $trimmed = $part.Trim()
            if ($trimmed) {
                $trimmed
            }
        }
    }
)

$ManifestPath = Join-Path $Root "manifests/assets.json"
if (-not (Test-Path $ManifestPath)) {
    throw "Could not find manifest: $ManifestPath"
}

$Manifest = Get-Content -Path $ManifestPath -Raw | ConvertFrom-Json

function Get-ObjectPropertyNames {
    param([Parameter(Mandatory = $true)]$Object)
    return @($Object.PSObject.Properties | ForEach-Object { $_.Name })
}

function Show-Manifest {
    Write-Host "Groups:"
    foreach ($name in (Get-ObjectPropertyNames $Manifest.groups | Sort-Object)) {
        Write-Host ("  {0}" -f $name)
    }

    Write-Host ""
    Write-Host "Assets:"
    foreach ($name in (Get-ObjectPropertyNames $Manifest.assets | Sort-Object)) {
        $item = $Manifest.assets.$name
        Write-Host ("  {0,-28} {1}" -f $name, $item.kind)
    }
}

function Expand-AssetSelection {
    param([string[]]$Names)

    $assetNames = Get-ObjectPropertyNames $Manifest.assets
    $groupNames = Get-ObjectPropertyNames $Manifest.groups
    $seen = [ordered]@{}

    function Add-One {
        param([string]$Name)

        if ($groupNames -contains $Name) {
            foreach ($child in @($Manifest.groups.$Name)) {
                Add-One -Name $child
            }
            return
        }

        if ($assetNames -contains $Name) {
            if (-not $seen.Contains($Name)) {
                $seen[$Name] = $true
            }
            return
        }

        throw "Unknown asset or group '$Name'. Run with -List to see valid names."
    }

    foreach ($name in $Names) {
        Add-One -Name $name
    }

    return @($seen.Keys)
}

function Get-Curl {
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $curl) {
        throw "curl.exe is required. It is included with recent Windows versions."
    }
    return $curl.Source
}

function Get-DownloadUrl {
    param($Item)

    if ($Item.googleDriveFileId) {
        return "https://drive.usercontent.google.com/download?id=$($Item.googleDriveFileId)&export=download&confirm=t"
    }

    return [string]$Item.url
}

function Invoke-AssetDownload {
    param(
        [Parameter(Mandatory = $true)]$Name,
        [Parameter(Mandatory = $true)]$Item,
        [Parameter(Mandatory = $true)][string]$CurlPath
    )

    if ($Item.manual) {
        Write-Warning ("{0}: manual download required. {1}" -f $Name, $Item.note)
        Write-Host ("  Source: {0}" -f $Item.source)
        return
    }

    $url = Get-DownloadUrl -Item $Item
    if ([string]::IsNullOrWhiteSpace($url)) {
        throw "$Name has no url or googleDriveFileId in the manifest."
    }

    $dest = Join-Path $Root ([string]$Item.path)
    $destDir = Split-Path -Parent $dest
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null

    $minBytes = [int64]0
    if ($Item.PSObject.Properties.Name -contains "minBytes") {
        $minBytes = [int64]$Item.minBytes
    }

    if ((Test-Path $dest) -and -not $Force) {
        $size = (Get-Item $dest).Length
        if ($size -gt 0 -and ($minBytes -eq 0 -or $size -ge $minBytes)) {
            Write-Host ("SKIP {0}: already exists ({1:n0} bytes)" -f $Name, $size)
            return
        }
        Write-Warning ("{0}: existing file is smaller than expected; resuming download." -f $Name)
    }

    Write-Host ("GET  {0}" -f $Name)
    Write-Host ("     {0}" -f $url)
    & $CurlPath --location --fail --retry 3 --retry-delay 3 --connect-timeout 30 --continue-at - --output $dest $url
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $Name with exit code $LASTEXITCODE"
    }

    $downloadedSize = (Get-Item $dest).Length
    if ($downloadedSize -eq 0) {
        throw "$Name downloaded to 0 bytes."
    }
    if ($minBytes -gt 0 -and $downloadedSize -lt $minBytes) {
        throw "$Name downloaded to only $downloadedSize bytes; expected at least $minBytes. The saved file may be an HTML login or interstitial page."
    }

    if ($Item.extractTo -and -not $NoExtract -and $dest.ToLowerInvariant().EndsWith(".zip")) {
        $extractTo = Join-Path $Root ([string]$Item.extractTo)
        New-Item -ItemType Directory -Force -Path $extractTo | Out-Null
        Write-Host ("UNZIP {0} -> {1}" -f $Name, $extractTo)
        Expand-Archive -Path $dest -DestinationPath $extractTo -Force
    }
}

if ($List) {
    Show-Manifest
    exit 0
}

$curlPath = Get-Curl
$resolved = Expand-AssetSelection -Names $Asset

Write-Host ("Root: {0}" -f $Root)
Write-Host ("Assets: {0}" -f ($resolved -join ", "))
Write-Host ""

foreach ($name in $resolved) {
    Invoke-AssetDownload -Name $name -Item $Manifest.assets.$name -CurlPath $curlPath
}

Write-Host ""
Write-Host "Done."
