[CmdletBinding()]
param(
    [string] $ProductRoot = (Join-Path $env:LOCALAPPDATA 'Mangalations\HSKMangaTranslator'),
    [Parameter(DontShow = $true)]
    [string] $RegistryPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts\local.mangalations.hsk_manga',
    [switch] $KeepCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Remove-DirectoryIfEmpty {
    param([Parameter(Mandatory = $true)][string] $Path)
    if (
        (Test-Path -LiteralPath $Path -PathType Container) -and
        $null -eq (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1)
    ) {
        Remove-Item -LiteralPath $Path -Force
    }
}

$resolvedProductRoot = [IO.Path]::GetFullPath($ProductRoot).TrimEnd('\', '/')
$driveRoot = [IO.Path]::GetPathRoot($resolvedProductRoot).TrimEnd('\', '/')
if ($resolvedProductRoot -eq $driveRoot) {
    throw "refusing to uninstall from a drive root: $resolvedProductRoot"
}

$appRoot = Join-Path $resolvedProductRoot 'app'
$resourceRoot = Join-Path $resolvedProductRoot 'resources'
$stateRoot = Join-Path $resolvedProductRoot 'browser-companion-v1'
$bundleManifestPath = Join-Path $appRoot 'bundle-manifest.json'
$hasInstalledApp = Test-Path -LiteralPath $appRoot -PathType Container
if ($hasInstalledApp) {
    if (-not (Test-Path -LiteralPath $bundleManifestPath -PathType Leaf)) {
        throw "refusing to remove an unmarked application directory: $appRoot"
    }
    $bundleManifest = Get-Content -LiteralPath $bundleManifestPath -Raw | ConvertFrom-Json
    if (
        $bundleManifest.bundleFormatVersion -ne 1 -or
        $bundleManifest.product -ne 'HSK Manga Translator' -or
        $bundleManifest.nativeHostName -ne 'local.mangalations.hsk_manga'
    ) {
        throw "refusing to remove an application directory with an unexpected marker: $appRoot"
    }
}

$daemonPath = Join-Path $appRoot 'companion\hsk-manga-browser-daemon.exe'
$daemonRecordPath = Join-Path $stateRoot 'daemon-state-v1.json'
if (
    (Test-Path -LiteralPath $daemonPath -PathType Leaf) -and
    (Test-Path -LiteralPath $daemonRecordPath -PathType Leaf)
) {
    try {
        $record = Get-Content -LiteralPath $daemonRecordPath -Raw | ConvertFrom-Json
        $process = Get-Process -Id ([int] $record.pid) -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            $processPath = $null
            try {
                $processPath = $process.Path
            }
            catch {
                Write-Warning "could not inspect daemon process $($record.pid); it was not stopped"
            }
            if (
                $null -ne $processPath -and
                [IO.Path]::GetFullPath($processPath).Equals(
                    [IO.Path]::GetFullPath($daemonPath),
                    [StringComparison]::OrdinalIgnoreCase
                )
            ) {
                Stop-Process -Id $process.Id -Force
                $process.WaitForExit(5000) | Out-Null
            }
        }
    }
    catch {
        Write-Warning "could not stop the recorded browser daemon: $($_.Exception.Message)"
    }
}

$unregisterCandidates = @(
    (Join-Path $appRoot 'native-host-registration\Unregister-NativeHost.ps1'),
    (Join-Path $PSScriptRoot 'native-host-registration\Unregister-NativeHost.ps1')
)
$unregisterScript = $unregisterCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if ($null -ne $unregisterScript) {
    & $unregisterScript -RegistryPath $RegistryPath
}
elseif (Test-Path -LiteralPath $RegistryPath) {
    throw 'the native-host registry key exists but the preserved unregistration script is unavailable'
}

if ($hasInstalledApp -and (Test-Path -LiteralPath $appRoot)) {
    Remove-Item -LiteralPath $appRoot -Recurse -Force
}
if ($hasInstalledApp -and (Test-Path -LiteralPath $resourceRoot)) {
    Remove-Item -LiteralPath $resourceRoot -Recurse -Force
}
if (-not $KeepCache -and (Test-Path -LiteralPath $stateRoot)) {
    Remove-Item -LiteralPath $stateRoot -Recurse -Force
}

Remove-DirectoryIfEmpty -Path (Join-Path $resolvedProductRoot 'native-host')
Remove-DirectoryIfEmpty -Path $resolvedProductRoot
Write-Output "Uninstalled HSK Manga Translator from $resolvedProductRoot"
