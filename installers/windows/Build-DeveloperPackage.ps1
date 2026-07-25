[CmdletBinding()]
param(
    [string] $OutputDirectory,
    [string] $NativeHostPath,
    [string] $BrowserDaemonPath,
    [string] $FirefoxExtensionZipPath,
    [string] $HskArtifactPath,
    [string] $DictionaryArtifactPath,
    [string] $ModelPath,
    [string] $SansFontPath,
    [string] $SerifFontPath,
    [Parameter(DontShow = $true)]
    [string] $ModelManifestPath,
    [switch] $SkipNpmInstall,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repositoryRoot 'dist\hsk-manga-translator-windows'
}
if ([string]::IsNullOrWhiteSpace($ModelManifestPath)) {
    $ModelManifestPath = Join-Path $repositoryRoot 'data\model-packs\manifest.v1.json'
}
if ([string]::IsNullOrWhiteSpace($SansFontPath)) {
    $SansFontPath = Join-Path $env:WINDIR 'Fonts\NotoSansSC-VF.ttf'
}
if ([string]::IsNullOrWhiteSpace($SerifFontPath)) {
    $SerifFontPath = Join-Path $env:WINDIR 'Fonts\NotoSerifSC-VF.ttf'
}

function Resolve-LeafFile {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,
        [Parameter(Mandatory = $true)]
        [string] $Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label must be an existing file: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string] $Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-SafeOutputDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,
        [Parameter(Mandatory = $true)]
        [string] $Repository
    )

    $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $repositoryPath = [IO.Path]::GetFullPath($Repository).TrimEnd('\', '/')
    $rootPath = [IO.Path]::GetPathRoot($fullPath).TrimEnd('\', '/')
    if ($fullPath -eq $rootPath -or $fullPath -eq $repositoryPath) {
        throw "refusing to use a broad output directory: $fullPath"
    }
    if ($repositoryPath.StartsWith($fullPath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to use an ancestor of the repository as the output directory: $fullPath"
    }
    return $fullPath
}

function Assert-ReplaceableOutputDirectory {
    param([Parameter(Mandatory = $true)][string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "output path exists but is not a directory: $Path"
    }
    if ($null -eq (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1)) {
        return
    }

    $markerPath = Join-Path $Path 'bundle-manifest.json'
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
        throw "refusing to replace a populated directory without a valid prior HSK bundle marker: $Path"
    }
    try {
        $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    }
    catch {
        throw "refusing to replace a populated directory with an unreadable HSK bundle marker: $Path"
    }
    if (
        $marker.bundleFormatVersion -ne 1 -or
        $marker.product -ne 'HSK Manga Translator' -or
        $marker.nativeHostName -ne 'local.hskify.hsk_manga'
    ) {
        throw "refusing to replace a populated directory with an invalid HSK bundle marker: $Path"
    }
}

function Read-FirefoxManifestFromZip {
    param([Parameter(Mandatory = $true)][string] $Path)

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $entry = $archive.GetEntry('manifest.json')
        if ($null -eq $entry) {
            throw 'Firefox extension ZIP does not contain manifest.json at its root'
        }
        $reader = [IO.StreamReader]::new($entry.Open())
        try {
            return $reader.ReadToEnd() | ConvertFrom-Json
        }
        finally {
            $reader.Dispose()
        }
    }
    finally {
        $archive.Dispose()
    }
}

if ([string]::IsNullOrWhiteSpace($NativeHostPath) -ne [string]::IsNullOrWhiteSpace($BrowserDaemonPath)) {
    throw 'NativeHostPath and BrowserDaemonPath must be supplied together'
}
if ([string]::IsNullOrWhiteSpace($HskArtifactPath) -ne [string]::IsNullOrWhiteSpace($DictionaryArtifactPath)) {
    throw 'HskArtifactPath and DictionaryArtifactPath must be supplied together'
}

$resolvedModelManifest = Resolve-LeafFile -Path $ModelManifestPath -Label 'ModelManifestPath'
$resolvedSansFont = Resolve-LeafFile -Path $SansFontPath -Label 'SansFontPath'
$resolvedSerifFont = Resolve-LeafFile -Path $SerifFontPath -Label 'SerifFontPath'
if ([IO.Path]::GetExtension($resolvedSansFont) -ne '.ttf' -or [IO.Path]::GetExtension($resolvedSerifFont) -ne '.ttf') {
    throw 'SansFontPath and SerifFontPath must be TrueType font files'
}
foreach ($font in @($resolvedSansFont, $resolvedSerifFont)) {
    if ([uint64] (Get-Item -LiteralPath $font).Length -gt 32MB) {
        throw "packaged CJK font exceeds the 32 MiB browser bound: $font"
    }
}
$modelManifest = Get-Content -LiteralPath $resolvedModelManifest -Raw | ConvertFrom-Json
if ($modelManifest.manifestVersion -ne 1 -or $modelManifest.selection.status -ne 'selected') {
    throw 'the model manifest must be version 1 with a selected standard pack'
}
$standardPackId = [string] $modelManifest.selection.standardPackId
$standardPacks = @($modelManifest.packs | Where-Object { $_.id -eq $standardPackId })
if ($standardPacks.Count -ne 1) {
    throw "the selected standard pack '$standardPackId' was not found exactly once"
}
$standardPack = $standardPacks[0]
if ($standardPack.runtimeModelId -ne 'qwen3.5-4b') {
    throw "the selected standard pack must use qwen3.5-4b, found '$($standardPack.runtimeModelId)'"
}
$translationFiles = @($standardPack.files | Where-Object { $_.id -eq 'translation-model' })
if ($translationFiles.Count -ne 1) {
    throw 'the selected standard pack must contain one translation-model file'
}
$translationModel = $translationFiles[0]
if ($translationModel.filename -ne 'Qwen3.5-4B-Q4_K_M.gguf') {
    throw "the selected translation model filename is not the frozen Qwen 4B artifact: $($translationModel.filename)"
}
$expectedModelSha256 = ([string] $translationModel.sha256).ToLowerInvariant()
if ($expectedModelSha256 -notmatch '^[0-9a-f]{64}$') {
    throw 'the selected translation model has an invalid SHA-256'
}

$resolvedModel = $null
if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    $resolvedModel = Resolve-LeafFile -Path $ModelPath -Label 'ModelPath'
    $actualModelSha256 = Get-Sha256 -Path $resolvedModel
    if ($actualModelSha256 -ne $expectedModelSha256) {
        throw "model SHA-256 mismatch: expected $expectedModelSha256, got $actualModelSha256"
    }
    if ([uint64] (Get-Item -LiteralPath $resolvedModel).Length -ne [uint64] $translationModel.bytes) {
        throw "model byte count mismatch: expected $($translationModel.bytes), got $((Get-Item -LiteralPath $resolvedModel).Length)"
    }
}

$resolvedHskArtifact = $null
$resolvedDictionaryArtifact = $null
if (-not [string]::IsNullOrWhiteSpace($HskArtifactPath)) {
    $resolvedHskArtifact = Resolve-LeafFile -Path $HskArtifactPath -Label 'HskArtifactPath'
    $resolvedDictionaryArtifact = Resolve-LeafFile -Path $DictionaryArtifactPath -Label 'DictionaryArtifactPath'
}

if ([string]::IsNullOrWhiteSpace($NativeHostPath)) {
    Push-Location $repositoryRoot
    try {
        Invoke-CheckedCommand -FilePath 'cargo' -Arguments @(
            'build',
            '--release',
            '--package', 'browser-companion',
            '--bin', 'hsk-manga-native-host',
            '--bin', 'hsk-manga-browser-daemon'
        )
    }
    finally {
        Pop-Location
    }

    if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        $targetRoot = Join-Path $repositoryRoot 'target'
    }
    elseif ([IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) {
        $targetRoot = $env:CARGO_TARGET_DIR
    }
    else {
        $targetRoot = Join-Path $repositoryRoot $env:CARGO_TARGET_DIR
    }
    $NativeHostPath = Join-Path $targetRoot 'release\hsk-manga-native-host.exe'
    $BrowserDaemonPath = Join-Path $targetRoot 'release\hsk-manga-browser-daemon.exe'
}
$resolvedNativeHost = Resolve-LeafFile -Path $NativeHostPath -Label 'NativeHostPath'
$resolvedBrowserDaemon = Resolve-LeafFile -Path $BrowserDaemonPath -Label 'BrowserDaemonPath'
if ([IO.Path]::GetFileName($resolvedNativeHost) -ne 'hsk-manga-native-host.exe') {
    throw 'NativeHostPath must name hsk-manga-native-host.exe'
}
if ([IO.Path]::GetFileName($resolvedBrowserDaemon) -ne 'hsk-manga-browser-daemon.exe') {
    throw 'BrowserDaemonPath must name hsk-manga-browser-daemon.exe'
}

$extensionRoot = Join-Path $repositoryRoot 'extensions\firefox'
if ([string]::IsNullOrWhiteSpace($FirefoxExtensionZipPath)) {
    Push-Location $extensionRoot
    try {
        if (-not $SkipNpmInstall) {
            Invoke-CheckedCommand -FilePath 'npm' -Arguments @(
                'install',
                '--no-package-lock',
                '--no-audit',
                '--no-fund'
            )
        }
        Invoke-CheckedCommand -FilePath 'npm' -Arguments @('run', 'build')
        Invoke-CheckedCommand -FilePath 'npm' -Arguments @('run', 'zip')
    }
    finally {
        Pop-Location
    }

    $extensionArchives = @(
        Get-ChildItem -LiteralPath (Join-Path $extensionRoot '.output') -Filter '*-firefox.zip' -File
    )
    if ($extensionArchives.Count -ne 1) {
        throw "expected exactly one Firefox ZIP under extensions\firefox\.output, found $($extensionArchives.Count)"
    }
    $FirefoxExtensionZipPath = $extensionArchives[0].FullName
}
$resolvedExtensionZip = Resolve-LeafFile -Path $FirefoxExtensionZipPath -Label 'FirefoxExtensionZipPath'
$firefoxManifest = Read-FirefoxManifestFromZip -Path $resolvedExtensionZip
if ($firefoxManifest.manifest_version -ne 3) {
    throw 'Firefox extension archive must use Manifest V3'
}
$extensionId = [string] $firefoxManifest.browser_specific_settings.gecko.id
if ($extensionId -ne 'hsk-manga-translator@local.hskify') {
    throw "Firefox extension archive has the wrong permanent ID: $extensionId"
}

$resolvedOutput = Assert-SafeOutputDirectory -Path $OutputDirectory -Repository $repositoryRoot
if (Test-Path -LiteralPath $resolvedOutput) {
    if (-not $Force) {
        throw "output directory already exists; pass -Force to replace it: $resolvedOutput"
    }
    Assert-ReplaceableOutputDirectory -Path $resolvedOutput
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}

$companionDirectory = Join-Path $resolvedOutput 'companion'
$extensionDirectory = Join-Path $resolvedOutput 'extension'
$resourceDirectory = Join-Path $resolvedOutput 'resources'
$modelDirectory = Join-Path $resourceDirectory 'models'
$modelPackDirectory = Join-Path $resourceDirectory 'model-packs'
$fontDirectory = Join-Path $resourceDirectory 'fonts'
[IO.Directory]::CreateDirectory($companionDirectory) | Out-Null
[IO.Directory]::CreateDirectory($extensionDirectory) | Out-Null
[IO.Directory]::CreateDirectory($modelPackDirectory) | Out-Null
[IO.Directory]::CreateDirectory($fontDirectory) | Out-Null

$stagedNativeHost = Join-Path $companionDirectory 'hsk-manga-native-host.exe'
$stagedBrowserDaemon = Join-Path $companionDirectory 'hsk-manga-browser-daemon.exe'
$stagedExtension = Join-Path $extensionDirectory 'hsk-manga-translator-firefox.zip'
Copy-Item -LiteralPath $resolvedNativeHost -Destination $stagedNativeHost
Copy-Item -LiteralPath $resolvedBrowserDaemon -Destination $stagedBrowserDaemon
Copy-Item -LiteralPath $resolvedExtensionZip -Destination $stagedExtension
Copy-Item -LiteralPath $resolvedModelManifest -Destination (Join-Path $modelPackDirectory 'manifest.v1.json')
$stagedSansFont = Join-Path $fontDirectory 'NotoSansSC-VF.ttf'
$stagedSerifFont = Join-Path $fontDirectory 'NotoSerifSC-VF.ttf'
Copy-Item -LiteralPath $resolvedSansFont -Destination $stagedSansFont
Copy-Item -LiteralPath $resolvedSerifFont -Destination $stagedSerifFont

$fileEntries = @(
    [ordered]@{
        role = 'native-host'
        path = 'companion\hsk-manga-native-host.exe'
        sha256 = Get-Sha256 -Path $stagedNativeHost
    },
    [ordered]@{
        role = 'browser-daemon'
        path = 'companion\hsk-manga-browser-daemon.exe'
        sha256 = Get-Sha256 -Path $stagedBrowserDaemon
    },
    [ordered]@{
        role = 'firefox-extension'
        path = 'extension\hsk-manga-translator-firefox.zip'
        sha256 = Get-Sha256 -Path $stagedExtension
    },
    [ordered]@{
        role = 'cjk-sans-font'
        path = 'resources\fonts\NotoSansSC-VF.ttf'
        sha256 = Get-Sha256 -Path $stagedSansFont
    },
    [ordered]@{
        role = 'cjk-serif-font'
        path = 'resources\fonts\NotoSerifSC-VF.ttf'
        sha256 = Get-Sha256 -Path $stagedSerifFont
    }
)

$hskBundled = $false
$dictionaryBundled = $false
if ($null -ne $resolvedHskArtifact) {
    $stagedHsk = Join-Path $resourceDirectory 'hsk-2.0.normalized.json'
    $stagedDictionary = Join-Path $resourceDirectory 'cc-cedict.normalized.json'
    Copy-Item -LiteralPath $resolvedHskArtifact -Destination $stagedHsk
    Copy-Item -LiteralPath $resolvedDictionaryArtifact -Destination $stagedDictionary
    $fileEntries += [ordered]@{
        role = 'hsk-data'
        path = 'resources\hsk-2.0.normalized.json'
        sha256 = Get-Sha256 -Path $stagedHsk
    }
    $fileEntries += [ordered]@{
        role = 'dictionary-data'
        path = 'resources\cc-cedict.normalized.json'
        sha256 = Get-Sha256 -Path $stagedDictionary
    }
    $hskBundled = $true
    $dictionaryBundled = $true
}

$modelBundled = $false
if ($null -ne $resolvedModel) {
    [IO.Directory]::CreateDirectory($modelDirectory) | Out-Null
    $stagedModel = Join-Path $modelDirectory 'Qwen3.5-4B-Q4_K_M.gguf'
    Copy-Item -LiteralPath $resolvedModel -Destination $stagedModel
    $fileEntries += [ordered]@{
        role = 'translation-model'
        path = 'resources\models\Qwen3.5-4B-Q4_K_M.gguf'
        sha256 = $expectedModelSha256
    }
    $modelBundled = $true
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Install.ps1') -Destination (Join-Path $resolvedOutput 'Install.ps1')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Uninstall.ps1') -Destination (Join-Path $resolvedOutput 'Uninstall.ps1')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $resolvedOutput 'README.md')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'native-host-registration') -Destination $resolvedOutput -Recurse

$bundleManifest = [ordered]@{
    bundleFormatVersion = 1
    product = 'HSK Manga Translator'
    version = [string] $firefoxManifest.version
    nativeHostName = 'local.hskify.hsk_manga'
    firefoxExtensionId = 'hsk-manga-translator@local.hskify'
    standardPackId = $standardPackId
    resources = [ordered]@{
        hskBundled = $hskBundled
        dictionaryBundled = $dictionaryBundled
        modelBundled = $modelBundled
        hskInstallPath = 'resources\hsk-2.0.normalized.json'
        dictionaryInstallPath = 'resources\cc-cedict.normalized.json'
        modelInstallPath = 'resources\models\Qwen3.5-4B-Q4_K_M.gguf'
        sansFontInstallPath = 'resources\fonts\NotoSansSC-VF.ttf'
        serifFontInstallPath = 'resources\fonts\NotoSerifSC-VF.ttf'
        expectedModelSha256 = $expectedModelSha256
    }
    files = @($fileEntries)
}
$bundleManifestJson = $bundleManifest | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText(
    (Join-Path $resolvedOutput 'bundle-manifest.json'),
    $bundleManifestJson + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

Write-Output $resolvedOutput
