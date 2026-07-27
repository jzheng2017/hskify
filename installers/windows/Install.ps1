[CmdletBinding()]
param(
    [string] $BundleRoot,
    [string] $ProductRoot = (Join-Path $env:LOCALAPPDATA 'Hskify'),
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

function Write-ModelReadinessMarker {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Root,
        [Parameter(Mandatory = $true)]
        [object] $Bundle,
        [string] $InstallationRoot = $Root
    )

    $modelManifestPath = Join-Path $Root 'resources\model-packs\manifest.v1.json'
    $modelManifest = Get-Content -LiteralPath $modelManifestPath -Raw | ConvertFrom-Json
    $resourceIdentities = @(
        foreach ($identity in @($modelManifest.resourceIdentities)) {
            [ordered]@{
                id = [string] $identity.id
                repository = [string] $identity.repository
                repositoryRevision = [string] $identity.repositoryRevision
                filename = [string] $identity.filename
                bytes = [uint64] $identity.bytes
                sha256 = [string] $identity.sha256
            }
        }
    )
    $installations = @(
        foreach ($identity in @($modelManifest.resourceIdentities)) {
            $path = if ([string] $identity.id -ceq 'translation-model') {
                Join-Path $InstallationRoot 'resources\models\Qwen3.5-4B-Q4_K_M.gguf'
            }
            else {
                Join-Path $InstallationRoot "resources\models\resident\$($identity.id)\$($identity.filename)"
            }
            [ordered]@{
                id = [string] $identity.id
                path = $path
            }
        }
    )
    $marker = [ordered]@{
        buildFingerprint = [string] $Bundle.buildFingerprint
        resourceIdentities = $resourceIdentities
        installations = $installations
    }
    $markerDirectory = Join-Path $Root 'browser-companion\browser-cache\browser-runtime'
    [IO.Directory]::CreateDirectory($markerDirectory) | Out-Null
    [IO.File]::WriteAllText(
        (Join-Path $markerDirectory 'models.ready'),
        ($marker | ConvertTo-Json -Depth 8 -Compress),
        [Text.UTF8Encoding]::new($false)
    )
}

function Stop-InstalledHskifyProcesses {
    param([Parameter(Mandatory = $true)][string] $InstalledAppRoot)

    $companionRoot = [IO.Path]::GetFullPath(
        (Join-Path $InstalledAppRoot 'companion')
    ).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    foreach ($name in @('hsk-manga-native-host', 'hsk-manga-browser-daemon')) {
        foreach ($process in @(Get-Process -Name $name -ErrorAction SilentlyContinue)) {
            $path = try { $process.Path } catch { $null }
            if (
                -not [string]::IsNullOrWhiteSpace($path) -and
                [IO.Path]::GetFullPath($path).StartsWith(
                    $companionRoot,
                    [StringComparison]::OrdinalIgnoreCase
                )
            ) {
                Stop-Process -Id $process.Id -Force -ErrorAction Stop
                $process.WaitForExit(5000)
            }
        }
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
    $bundleManifest.product -ne 'Hskify' -or
    $bundleManifest.nativeHostName -ne 'local.hskify.hsk_manga' -or
    $bundleManifest.firefoxExtensionId -ne 'hsk-manga-translator@local.hskify'
) {
    throw 'the bundle manifest does not identify Hskify'
}
if (
    -not $bundleManifest.resources.hskBundled -or
    -not $bundleManifest.resources.dictionaryBundled -or
    -not $bundleManifest.resources.modelBundled -or
    -not $bundleManifest.resources.residentModelsBundled -or
    $bundleManifest.resources.residentModelCount -ne 5 -or
    -not $bundleManifest.resources.residentRuntimeBundled -or
    $bundleManifest.resources.residentRuntimeFileCount -ne 39
) {
    throw 'the bundle is missing mandatory HSK, dictionary, detector, OCR, translation, CUDA, or llama resources'
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
$updateId = [Guid]::NewGuid().ToString('N')
$stagingRoot = Join-Path $resolvedProductRoot ".update-$updateId"
$backupRoot = Join-Path $resolvedProductRoot ".previous-$updateId"
$stagedAppRoot = Join-Path $stagingRoot 'app'
$stagedResourceRoot = Join-Path $stagingRoot 'resources'
$activatedNames = [Collections.Generic.List[string]]::new()
$backedUpNames = [Collections.Generic.List[string]]::new()
$registryExistedBefore = Test-Path -LiteralPath $RegistryPath

try {
    [IO.Directory]::CreateDirectory($stagedAppRoot) | Out-Null
    foreach ($name in @(
        'companion',
        'extension',
        'provenance',
        'native-host-registration',
        'Install.ps1',
        'Uninstall.ps1',
        'README.md',
        'bundle-manifest.json'
    )) {
        Copy-Item -LiteralPath (Join-Path $resolvedBundleRoot $name) -Destination $stagedAppRoot -Recurse -Force
    }

    Copy-DirectoryContents -Source (Join-Path $resolvedBundleRoot 'resources') -Destination $stagedResourceRoot

    foreach ($entry in @($bundleManifest.files)) {
        $relativePath = [string] $entry.path
        if ($relativePath.StartsWith('resources\', [StringComparison]::OrdinalIgnoreCase)) {
            $installedPath = Join-Path $stagingRoot $relativePath
        }
        else {
            $installedPath = Join-Path $stagedAppRoot $relativePath
        }
        if (-not (Test-Path -LiteralPath $installedPath -PathType Leaf)) {
            throw "staged file is missing after copy: $relativePath"
        }
        $actual = Get-Sha256 -Path $installedPath
        $expected = ([string] $entry.sha256).ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "staged file SHA-256 mismatch for $relativePath"
        }
    }

    Write-ModelReadinessMarker `
        -Root $stagingRoot `
        -Bundle $bundleManifest `
        -InstallationRoot $resolvedProductRoot

    Stop-InstalledHskifyProcesses -InstalledAppRoot $appRoot
    [IO.Directory]::CreateDirectory($backupRoot) | Out-Null
    foreach ($name in @('app', 'resources', 'browser-companion')) {
        $installedPath = Join-Path $resolvedProductRoot $name
        if (Test-Path -LiteralPath $installedPath) {
            Move-Item -LiteralPath $installedPath -Destination (Join-Path $backupRoot $name)
            $backedUpNames.Add($name)
        }
    }
    foreach ($name in @('app', 'resources', 'browser-companion')) {
        Move-Item -LiteralPath (Join-Path $stagingRoot $name) -Destination (Join-Path $resolvedProductRoot $name)
        $activatedNames.Add($name)
    }

    $registerScript = Join-Path $appRoot 'native-host-registration\Register-NativeHost.ps1'
    $nativeHostPath = Join-Path $appRoot 'companion\hsk-manga-native-host.exe'
    & $registerScript -NativeHostPath $nativeHostPath -RegistryPath $RegistryPath | Out-Null

    Remove-Item -LiteralPath $backupRoot -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
}
catch {
    foreach ($name in @($activatedNames)) {
        $activatedPath = Join-Path $resolvedProductRoot $name
        if (Test-Path -LiteralPath $activatedPath) {
            Remove-Item -LiteralPath $activatedPath -Recurse -Force
        }
    }
    foreach ($name in @($backedUpNames)) {
        $backupPath = Join-Path $backupRoot $name
        if (Test-Path -LiteralPath $backupPath) {
            Move-Item -LiteralPath $backupPath -Destination (Join-Path $resolvedProductRoot $name)
        }
    }
    if (-not $registryExistedBefore -and (Test-Path -LiteralPath $RegistryPath)) {
        Remove-Item -LiteralPath $RegistryPath -Recurse -Force
    }
    if (Test-Path -LiteralPath $backupRoot) {
        Remove-Item -LiteralPath $backupRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
    throw
}

Write-Output "Hskify is up to date under $resolvedProductRoot"
Write-Output "Firefox extension package: $(Join-Path $appRoot 'extension\hskify-firefox.zip')"
Write-Output "Uninstall command: powershell -ExecutionPolicy Bypass -File `"$(Join-Path $appRoot 'Uninstall.ps1')`""
