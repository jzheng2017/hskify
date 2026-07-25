[CmdletBinding()]
param(
    [string] $BundleRoot,
    [string] $ProductRoot = (Join-Path $env:LOCALAPPDATA 'Hskify\HSKMangaTranslator'),
    [Parameter(DontShow = $true)]
    [string] $RegistryPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts\local.hskify.hsk_manga'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($BundleRoot)) {
    $BundleRoot = $PSScriptRoot
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-BundleFile {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Root,
        [Parameter(Mandatory = $true)]
        [object] $Entry
    )

    $path = Join-Path $Root ([string] $Entry.path)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "bundle file is missing: $($Entry.path)"
    }
    $actual = Get-Sha256 -Path $path
    $expected = ([string] $Entry.sha256).ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "bundle file SHA-256 mismatch for $($Entry.path): expected $expected, got $actual"
    }
}

function Copy-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Source,
        [Parameter(Mandatory = $true)]
        [string] $Destination
    )

    [IO.Directory]::CreateDirectory($Destination) | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $Source -Force) {
        Copy-Item -LiteralPath $item.FullName -Destination $Destination -Recurse -Force
    }
}

$resolvedBundleRoot = (Resolve-Path -LiteralPath $BundleRoot).Path
$bundleManifestPath = Join-Path $resolvedBundleRoot 'bundle-manifest.json'
if (-not (Test-Path -LiteralPath $bundleManifestPath -PathType Leaf)) {
    throw "bundle-manifest.json is missing from $resolvedBundleRoot"
}
$bundleManifest = Get-Content -LiteralPath $bundleManifestPath -Raw | ConvertFrom-Json
if (
    $bundleManifest.bundleFormatVersion -ne 1 -or
    $bundleManifest.product -ne 'HSK Manga Translator' -or
    $bundleManifest.nativeHostName -ne 'local.hskify.hsk_manga' -or
    $bundleManifest.firefoxExtensionId -ne 'hsk-manga-translator@local.hskify'
) {
    throw 'the bundle manifest does not identify the frozen HSK Manga Translator product'
}
foreach ($entry in @($bundleManifest.files)) {
    Assert-BundleFile -Root $resolvedBundleRoot -Entry $entry
}

$resolvedProductRoot = [IO.Path]::GetFullPath($ProductRoot).TrimEnd('\', '/')
$driveRoot = [IO.Path]::GetPathRoot($resolvedProductRoot).TrimEnd('\', '/')
if ($resolvedProductRoot -eq $driveRoot) {
    throw "refusing to install into a drive root: $resolvedProductRoot"
}
$appRoot = Join-Path $resolvedProductRoot 'app'
$resourceRoot = Join-Path $resolvedProductRoot 'resources'
if ((Test-Path -LiteralPath $appRoot) -or (Test-Path -LiteralPath $resourceRoot)) {
    throw "an installation already exists under $resolvedProductRoot; run Uninstall.ps1 first"
}

$createdApp = $false
$createdResources = $false
$registered = $false
try {
    [IO.Directory]::CreateDirectory($appRoot) | Out-Null
    $createdApp = $true
    foreach ($name in @(
        'companion',
        'extension',
        'native-host-registration',
        'Install.ps1',
        'Uninstall.ps1',
        'README.md',
        'bundle-manifest.json'
    )) {
        Copy-Item -LiteralPath (Join-Path $resolvedBundleRoot $name) -Destination $appRoot -Recurse -Force
    }

    Copy-DirectoryContents -Source (Join-Path $resolvedBundleRoot 'resources') -Destination $resourceRoot
    $createdResources = $true

    foreach ($entry in @($bundleManifest.files)) {
        $relativePath = [string] $entry.path
        if ($relativePath.StartsWith('resources\', [StringComparison]::OrdinalIgnoreCase)) {
            $installedPath = Join-Path $resolvedProductRoot $relativePath
        }
        else {
            $installedPath = Join-Path $appRoot $relativePath
        }
        if (-not (Test-Path -LiteralPath $installedPath -PathType Leaf)) {
            throw "installed file is missing after copy: $relativePath"
        }
        $actual = Get-Sha256 -Path $installedPath
        $expected = ([string] $entry.sha256).ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "installed file SHA-256 mismatch for $relativePath"
        }
    }

    $registerScript = Join-Path $appRoot 'native-host-registration\Register-NativeHost.ps1'
    $nativeHostPath = Join-Path $appRoot 'companion\hsk-manga-native-host.exe'
    & $registerScript -NativeHostPath $nativeHostPath -RegistryPath $RegistryPath | Out-Null
    $registered = $true
}
catch {
    if ($registered) {
        $unregisterScript = Join-Path $appRoot 'native-host-registration\Unregister-NativeHost.ps1'
        & $unregisterScript -RegistryPath $RegistryPath
    }
    if ($createdApp -and (Test-Path -LiteralPath $appRoot)) {
        Remove-Item -LiteralPath $appRoot -Recurse -Force
    }
    if ($createdResources -and (Test-Path -LiteralPath $resourceRoot)) {
        Remove-Item -LiteralPath $resourceRoot -Recurse -Force
    }
    throw
}

Write-Output "Installed HSK Manga Translator under $resolvedProductRoot"
Write-Output "Firefox extension package: $(Join-Path $appRoot 'extension\hsk-manga-translator-firefox.zip')"
Write-Output "Uninstall command: powershell -ExecutionPolicy Bypass -File `"$(Join-Path $appRoot 'Uninstall.ps1')`""
