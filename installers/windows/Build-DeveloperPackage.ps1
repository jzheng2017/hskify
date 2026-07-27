[CmdletBinding()]
param(
    [string] $OutputDirectory,
    [string] $NativeHostPath,
    [string] $BrowserDaemonPath,
    [string] $BuildAttestationPath,
    [string] $FirefoxExtensionZipPath,
    [string] $HskArtifactPath,
    [string] $DictionaryArtifactPath,
    [string] $ModelPath,
    [string] $ResidentModelsDirectory,
    [string] $ResidentRuntimeDirectory,
    [string] $SansFontPath,
    [string] $SerifFontPath,
    [Parameter(DontShow = $true)]
    [string] $ModelManifestPath,
    [switch] $SkipPnpmInstall,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repositoryRoot 'scripts\PerformanceBuildAttestation.ps1')
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repositoryRoot 'dist\hskify-windows'
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
if ([string]::IsNullOrWhiteSpace($ResidentModelsDirectory)) {
    $ResidentModelsDirectory = Join-Path $repositoryRoot '.cache\resident-models'
}
if ([string]::IsNullOrWhiteSpace($ResidentRuntimeDirectory)) {
    $ResidentRuntimeDirectory = Join-Path $repositoryRoot '.cache\resident-runtime'
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

function Assert-ExactArtifact {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][string] $Label,
        [Parameter(Mandatory = $true)][uint64] $ExpectedBytes,
        [Parameter(Mandatory = $true)][string] $ExpectedSha256
    )

    $actualBytes = [uint64] (Get-Item -LiteralPath $Path).Length
    if ($actualBytes -ne $ExpectedBytes) {
        throw "$Label byte count mismatch: expected $ExpectedBytes, got $actualBytes"
    }
    $actualSha256 = Get-Sha256 -Path $Path
    if ($actualSha256 -ne $ExpectedSha256) {
        throw "$Label SHA-256 mismatch: expected $ExpectedSha256, got $actualSha256"
    }
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
        $marker.product -ne 'Hskify' -or
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
if (
    -not [string]::IsNullOrWhiteSpace($NativeHostPath) -and
    [string]::IsNullOrWhiteSpace($BuildAttestationPath)
) {
    throw 'BuildAttestationPath is required with explicitly supplied native binaries'
}
if (
    [string]::IsNullOrWhiteSpace($HskArtifactPath) -or
    [string]::IsNullOrWhiteSpace($DictionaryArtifactPath) -or
    [string]::IsNullOrWhiteSpace($ModelPath)
) {
    throw 'HskArtifactPath, DictionaryArtifactPath, and ModelPath are required; installable bundles are always self-contained'
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
Assert-ExactArtifact -Path $resolvedSansFont -Label 'NotoSansSC-VF.ttf' -ExpectedBytes 17773244 -ExpectedSha256 '763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a'
Assert-ExactArtifact -Path $resolvedSerifFont -Label 'NotoSerifSC-VF.ttf' -ExpectedBytes 25129160 -ExpectedSha256 'a4aed9985a5916fbf6690456f8732a9fccd517938e353165d4142b4f11a39280'
$modelManifest = Get-Content -LiteralPath $resolvedModelManifest -Raw | ConvertFrom-Json
$expectedManifestFields = @('generatedAt', 'manifestVersion', 'resourceIdentities', 'translationModelId')
$actualManifestFields = @($modelManifest.PSObject.Properties.Name | Sort-Object)
if (($actualManifestFields -join "`n") -cne ($expectedManifestFields -join "`n")) {
    throw 'the model manifest has unexpected fields'
}
if ($modelManifest.manifestVersion -ne 1) {
    throw 'the model manifest must be version 1'
}
if ([string] $modelManifest.generatedAt -cnotmatch '^\d{4}-\d{2}-\d{2}$') {
    throw 'the model manifest generatedAt must be an ISO calendar date'
}
if ([string] $modelManifest.translationModelId -cne 'qwen3.5-4b') {
    throw "the mandatory translation model must be qwen3.5-4b, found '$($modelManifest.translationModelId)'"
}
$requiredResourceIds = @(
    'comic-text-bubble-detector-config',
    'comic-text-bubble-detector-preprocessor-config',
    'comic-text-bubble-detector-weights',
    'pp-ocr-v5-english-recognizer-config',
    'pp-ocr-v5-english-recognizer-model',
    'translation-model'
)
$resourceIdentities = @($modelManifest.resourceIdentities)
if ($resourceIdentities.Count -ne $requiredResourceIds.Count) {
    throw 'the model manifest must contain exactly six resident resource identities'
}
$expectedIdentityFields = @('bytes', 'filename', 'id', 'repository', 'repositoryRevision', 'sha256', 'url')
for ($index = 0; $index -lt $resourceIdentities.Count; $index++) {
    $identity = $resourceIdentities[$index]
    $actualIdentityFields = @($identity.PSObject.Properties.Name | Sort-Object)
    if (($actualIdentityFields -join "`n") -cne ($expectedIdentityFields -join "`n")) {
        throw "the model manifest resource identity has unexpected fields: $($identity.id)"
    }
    if ([string] $identity.id -cne $requiredResourceIds[$index]) {
        throw "the model manifest resource identities are incomplete or out of order at '$($identity.id)'"
    }
    if (
        [string] $identity.repositoryRevision -cnotmatch '^[0-9a-f]{40}$' -or
        [string] $identity.sha256 -cnotmatch '^[0-9a-f]{64}$' -or
        [uint64] $identity.bytes -eq 0
    ) {
        throw "the model manifest resource identity is not pinned: $($identity.id)"
    }
    $expectedUrl = "https://huggingface.co/$($identity.repository)/resolve/$($identity.repositoryRevision)/$($identity.filename)"
    if ([string] $identity.url -cne $expectedUrl) {
        throw "the resource URL is not exactly pinned: $($identity.id)"
    }
}
$translationModel = $resourceIdentities[-1]
if ($translationModel.filename -ne 'Qwen3.5-4B-Q4_K_M.gguf') {
    throw "the translation model filename is not the frozen Qwen 4B artifact: $($translationModel.filename)"
}
$expectedModelSha256 = ([string] $translationModel.sha256).ToLowerInvariant()
if ($expectedModelSha256 -notmatch '^[0-9a-f]{64}$') {
    throw 'the translation model has an invalid SHA-256'
}

if (-not (Test-Path -LiteralPath $ResidentModelsDirectory -PathType Container)) {
    throw "ResidentModelsDirectory must contain the five pinned detector/OCR resources: $ResidentModelsDirectory"
}
$resolvedResidentModelsDirectory = (Resolve-Path -LiteralPath $ResidentModelsDirectory).Path
$resolvedResidentResources = @(
    foreach ($identity in @($resourceIdentities | Where-Object id -ne 'translation-model')) {
        $path = Join-Path $resolvedResidentModelsDirectory ([string] $identity.id)
        $path = Join-Path $path ([string] $identity.filename)
        $resolvedPath = Resolve-LeafFile -Path $path -Label ([string] $identity.id)
        Assert-ExactArtifact `
            -Path $resolvedPath `
            -Label ([string] $identity.id) `
            -ExpectedBytes ([uint64] $identity.bytes) `
            -ExpectedSha256 ([string] $identity.sha256)
        [ordered]@{
            identity = $identity
            path = $resolvedPath
        }
    }
)

$expectedResidentRuntimeFiles = @(
    'cuda\.installed',
    'cuda\cublas64_13.dll',
    'cuda\cublasLt64_13.dll',
    'cuda\cudart64_13.dll',
    'cuda\cudnn64_9.dll',
    'cuda\cudnn_adv64_9.dll',
    'cuda\cudnn_cnn64_9.dll',
    'cuda\cudnn_engines_precompiled64_9.dll',
    'cuda\cudnn_engines_runtime_compiled64_9.dll',
    'cuda\cudnn_graph64_9.dll',
    'cuda\cudnn_heuristic64_9.dll',
    'cuda\cudnn_ops64_9.dll',
    'cuda\cufft64_12.dll',
    'cuda\curand64_10.dll',
    'cuda\nvrtc64_130_0.dll',
    'cuda\nvrtc-builtins64_131.dll',
    'llama.cpp\b8935\windows-cuda13-x64\.installed',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-base.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-alderlake.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-cannonlake.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-cascadelake.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-cooperlake.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-haswell.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-icelake.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-ivybridge.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-piledriver.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-sandybridge.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-sapphirerapids.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-skylakex.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-sse42.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-x64.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cpu-zen4.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-cuda.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml-rpc.dll',
    'llama.cpp\b8935\windows-cuda13-x64\ggml.dll',
    'llama.cpp\b8935\windows-cuda13-x64\libomp140.x86_64.dll',
    'llama.cpp\b8935\windows-cuda13-x64\llama-common.dll',
    'llama.cpp\b8935\windows-cuda13-x64\llama.dll',
    'llama.cpp\b8935\windows-cuda13-x64\mtmd.dll'
)
if (-not (Test-Path -LiteralPath $ResidentRuntimeDirectory -PathType Container)) {
    throw "ResidentRuntimeDirectory must contain the pinned CUDA 13.1 and llama.cpp b8935 runtime: $ResidentRuntimeDirectory"
}
$resolvedResidentRuntimeDirectory = (Resolve-Path -LiteralPath $ResidentRuntimeDirectory).Path.TrimEnd('\', '/')
$actualResidentRuntimeFiles = @(
    Get-ChildItem -LiteralPath $resolvedResidentRuntimeDirectory -Recurse -File |
        ForEach-Object { $_.FullName.Substring($resolvedResidentRuntimeDirectory.Length + 1) } |
        Sort-Object
)
if (
    ($actualResidentRuntimeFiles -join "`n") -cne
    (@($expectedResidentRuntimeFiles | Sort-Object) -join "`n")
) {
    throw 'ResidentRuntimeDirectory does not contain exactly the pinned runtime file set'
}
$cudaInstallMarker = Get-Content -LiteralPath (Join-Path $resolvedResidentRuntimeDirectory 'cuda\.installed') -Raw
if ($cudaInstallMarker.Trim() -cne 'cuda;platform=win_amd64;wheels=nvidia-cuda-runtime/13.1.80,nvidia-cuda-nvrtc/13.1.80,nvidia-cublas/13.2.0.9,nvidia-cufft/12.1.0.31,nvidia-curand/10.4.1.81,nvidia-cudnn-cu13/9.17.0.29;extract=5') {
    throw 'ResidentRuntimeDirectory has the wrong CUDA install marker'
}
$llamaInstallMarker = Get-Content -LiteralPath (Join-Path $resolvedResidentRuntimeDirectory 'llama.cpp\b8935\windows-cuda13-x64\.installed') -Raw
if ($llamaInstallMarker.Trim() -cne 'llama-b8935-windows-cuda13-x64-extract-2') {
    throw 'ResidentRuntimeDirectory has the wrong llama.cpp install marker'
}
$resolvedResidentRuntimeFiles = @(
    foreach ($relativePath in $expectedResidentRuntimeFiles) {
        [ordered]@{
            relativePath = $relativePath
            path = Join-Path $resolvedResidentRuntimeDirectory $relativePath
        }
    }
)

$resolvedModel = Resolve-LeafFile -Path $ModelPath -Label 'ModelPath'
$actualModelSha256 = Get-Sha256 -Path $resolvedModel
if ($actualModelSha256 -ne $expectedModelSha256) {
    throw "model SHA-256 mismatch: expected $expectedModelSha256, got $actualModelSha256"
}
if ([uint64] (Get-Item -LiteralPath $resolvedModel).Length -ne [uint64] $translationModel.bytes) {
    throw "model byte count mismatch: expected $($translationModel.bytes), got $((Get-Item -LiteralPath $resolvedModel).Length)"
}

$resolvedHskArtifact = Resolve-LeafFile -Path $HskArtifactPath -Label 'HskArtifactPath'
$resolvedDictionaryArtifact = Resolve-LeafFile -Path $DictionaryArtifactPath -Label 'DictionaryArtifactPath'
Assert-ExactArtifact -Path $resolvedHskArtifact -Label 'hsk-2.0.normalized.json' -ExpectedBytes 1219917 -ExpectedSha256 'e603244c49d6a231426e9696574e98bd1e76fbea68f56e76ea98695d26ce478f'
Assert-ExactArtifact -Path $resolvedDictionaryArtifact -Label 'cc-cedict.normalized.json' -ExpectedBytes 28604488 -ExpectedSha256 '4011f023d27e576559ae0f2afe6fd0cc4458f96d225baa80f0ddbc9bb0344f33'

if ([string]::IsNullOrWhiteSpace($NativeHostPath)) {
    if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        $targetRoot = Join-Path $repositoryRoot 'target'
    }
    elseif ([IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) {
        $targetRoot = [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)
    }
    else {
        $targetRoot = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $env:CARGO_TARGET_DIR))
    }
    $releaseDirectory = Join-Path $targetRoot 'release'
    if ([string]::IsNullOrWhiteSpace($BuildAttestationPath)) {
        $BuildAttestationPath = Join-Path $releaseDirectory 'hskify-performance-build-attestation.json'
    }
    $performanceBuildScript = Join-Path $repositoryRoot 'scripts\Invoke-PerformanceBuild.ps1'
    & $performanceBuildScript -AttestationPath $BuildAttestationPath
    if (-not $?) {
        throw 'the exact performance-build wrapper failed'
    }
    $NativeHostPath = Join-Path $releaseDirectory 'hsk-manga-native-host.exe'
    $BrowserDaemonPath = Join-Path $releaseDirectory 'hsk-manga-browser-daemon.exe'
}
$resolvedNativeHost = Resolve-LeafFile -Path $NativeHostPath -Label 'NativeHostPath'
$resolvedBrowserDaemon = Resolve-LeafFile -Path $BrowserDaemonPath -Label 'BrowserDaemonPath'
$resolvedBuildAttestation = Resolve-LeafFile -Path $BuildAttestationPath -Label 'BuildAttestationPath'
if ([IO.Path]::GetFileName($resolvedNativeHost) -ne 'hsk-manga-native-host.exe') {
    throw 'NativeHostPath must name hsk-manga-native-host.exe'
}
if ([IO.Path]::GetFileName($resolvedBrowserDaemon) -ne 'hsk-manga-browser-daemon.exe') {
    throw 'BrowserDaemonPath must name hsk-manga-browser-daemon.exe'
}
$buildAttestation = Assert-HskifyPerformanceBuildAttestation `
    -AttestationPath $resolvedBuildAttestation `
    -NativeHostPath $resolvedNativeHost `
    -BrowserDaemonPath $resolvedBrowserDaemon `
    -RepositoryRoot $repositoryRoot `
    -VerifyCurrentSourceTree `
    -VerifyCurrentHardware `
    -VerifyCurrentToolchain

$onnxRuntimeProviderFiles = @(
    'onnxruntime_providers_shared.dll',
    'onnxruntime_providers_cuda.dll'
) | ForEach-Object {
    Resolve-LeafFile `
        -Path (Join-Path (Split-Path -Parent $resolvedBrowserDaemon) $_) `
        -Label $_
}

$vcRoot = (Get-Item -LiteralPath ([string] $buildAttestation.toolchain.msvc.path)).
    Directory.Parent.Parent.Parent.Parent.Parent.Parent.FullName
$vcRuntimeRoot = Join-Path $vcRoot 'Redist\MSVC'
$vcRuntimeDirectory = Get-ChildItem -LiteralPath $vcRuntimeRoot -Directory |
    Sort-Object Name -Descending |
    ForEach-Object { Join-Path $_.FullName 'x64\Microsoft.VC143.CRT' } |
    Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
    Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($vcRuntimeDirectory)) {
    throw "the MSVC x64 runtime directory is missing under $vcRuntimeRoot"
}
$vcRuntimeFiles = @(
    'msvcp140.dll',
    'msvcp140_1.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll'
) | ForEach-Object {
    Resolve-LeafFile -Path (Join-Path $vcRuntimeDirectory $_) -Label $_
}

$extensionRoot = Join-Path $repositoryRoot 'extensions\firefox'
if ([string]::IsNullOrWhiteSpace($FirefoxExtensionZipPath)) {
    Push-Location $extensionRoot
    try {
        if (-not $SkipPnpmInstall) {
            Invoke-CheckedCommand -FilePath 'pnpm' -Arguments @(
                'install',
                '--frozen-lockfile'
            )
        }
        Invoke-CheckedCommand -FilePath 'pnpm' -Arguments @('build')
        Invoke-CheckedCommand -FilePath 'pnpm' -Arguments @('zip')
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
$residentModelDirectory = Join-Path $modelDirectory 'resident'
$residentRuntimeDirectory = Join-Path $resourceDirectory 'runtime'
$modelManifestDirectory = Join-Path $resourceDirectory 'model-packs'
$fontDirectory = Join-Path $resourceDirectory 'fonts'
$provenanceDirectory = Join-Path $resolvedOutput 'provenance'
[IO.Directory]::CreateDirectory($companionDirectory) | Out-Null
[IO.Directory]::CreateDirectory($extensionDirectory) | Out-Null
[IO.Directory]::CreateDirectory($modelManifestDirectory) | Out-Null
[IO.Directory]::CreateDirectory($fontDirectory) | Out-Null
[IO.Directory]::CreateDirectory($provenanceDirectory) | Out-Null

$stagedNativeHost = Join-Path $companionDirectory 'hsk-manga-native-host.exe'
$stagedBrowserDaemon = Join-Path $companionDirectory 'hsk-manga-browser-daemon.exe'
$stagedExtension = Join-Path $extensionDirectory 'hskify-firefox.zip'
$stagedBuildAttestation = Join-Path $provenanceDirectory 'performance-build-attestation.json'
Copy-Item -LiteralPath $resolvedNativeHost -Destination $stagedNativeHost
Copy-Item -LiteralPath $resolvedBrowserDaemon -Destination $stagedBrowserDaemon
Copy-Item -LiteralPath $resolvedExtensionZip -Destination $stagedExtension
Copy-Item -LiteralPath $resolvedBuildAttestation -Destination $stagedBuildAttestation
Copy-Item -LiteralPath $resolvedModelManifest -Destination (Join-Path $modelManifestDirectory 'manifest.v1.json')
$stagedVcRuntimeFiles = @($vcRuntimeFiles | ForEach-Object {
    $destination = Join-Path $companionDirectory ([IO.Path]::GetFileName($_))
    Copy-Item -LiteralPath $_ -Destination $destination
    $destination
})
$stagedOnnxRuntimeProviderFiles = @($onnxRuntimeProviderFiles | ForEach-Object {
    $destination = Join-Path $companionDirectory ([IO.Path]::GetFileName($_))
    Copy-Item -LiteralPath $_ -Destination $destination
    $destination
})
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
        role = 'performance-build-attestation'
        path = 'provenance\performance-build-attestation.json'
        bytes = [int64] (Get-Item -LiteralPath $stagedBuildAttestation).Length
        sha256 = Get-Sha256 -Path $stagedBuildAttestation
    },
    [ordered]@{
        role = 'firefox-extension'
        path = 'extension\hskify-firefox.zip'
        sha256 = Get-Sha256 -Path $stagedExtension
    },
    [ordered]@{
        role = 'resident-model-manifest'
        path = 'resources\model-packs\manifest.v1.json'
        sha256 = Get-Sha256 -Path (Join-Path $modelManifestDirectory 'manifest.v1.json')
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
$fileEntries += @($stagedVcRuntimeFiles | ForEach-Object {
    [ordered]@{
        role = 'vc-runtime'
        path = 'companion\' + [IO.Path]::GetFileName($_)
        sha256 = Get-Sha256 -Path $_
    }
})
$fileEntries += @($stagedOnnxRuntimeProviderFiles | ForEach-Object {
    [ordered]@{
        role = 'onnx-runtime-provider'
        path = 'companion\' + [IO.Path]::GetFileName($_)
        bytes = [int64] (Get-Item -LiteralPath $_).Length
        sha256 = Get-Sha256 -Path $_
    }
})

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

[IO.Directory]::CreateDirectory($modelDirectory) | Out-Null
$stagedModel = Join-Path $modelDirectory 'Qwen3.5-4B-Q4_K_M.gguf'
Copy-Item -LiteralPath $resolvedModel -Destination $stagedModel
$fileEntries += [ordered]@{
    role = 'translation-model'
    path = 'resources\models\Qwen3.5-4B-Q4_K_M.gguf'
    sha256 = $expectedModelSha256
}

foreach ($resource in $resolvedResidentResources) {
    $identity = $resource.identity
    $relativePath = "resources\models\resident\$($identity.id)\$($identity.filename)"
    $destinationDirectory = Join-Path $residentModelDirectory ([string] $identity.id)
    [IO.Directory]::CreateDirectory($destinationDirectory) | Out-Null
    $destination = Join-Path $destinationDirectory ([string] $identity.filename)
    Copy-Item -LiteralPath $resource.path -Destination $destination
    $fileEntries += [ordered]@{
        role = 'resident-model-resource'
        path = $relativePath
        bytes = [int64] $identity.bytes
        sha256 = [string] $identity.sha256
    }
}

foreach ($runtimeFile in $resolvedResidentRuntimeFiles) {
    $relativePath = "resources\runtime\$($runtimeFile.relativePath)"
    $destination = Join-Path $residentRuntimeDirectory $runtimeFile.relativePath
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    Copy-Item -LiteralPath $runtimeFile.path -Destination $destination
    $fileEntries += [ordered]@{
        role = 'resident-runtime'
        path = $relativePath
        bytes = [int64] (Get-Item -LiteralPath $destination).Length
        sha256 = Get-Sha256 -Path $destination
    }
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Install.ps1') -Destination (Join-Path $resolvedOutput 'Install.ps1')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Uninstall.ps1') -Destination (Join-Path $resolvedOutput 'Uninstall.ps1')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $resolvedOutput 'README.md')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'native-host-registration') -Destination $resolvedOutput -Recurse

$bundleManifest = [ordered]@{
    bundleFormatVersion = 1
    product = 'Hskify'
    version = [string] $firefoxManifest.version
    nativeHostName = 'local.hskify.hsk_manga'
    firefoxExtensionId = 'hsk-manga-translator@local.hskify'
    buildFingerprint = [string] $buildAttestation.buildFingerprint
    performanceBuildAttestation = [ordered]@{
        schema = [string] $buildAttestation.schema
        path = 'provenance\performance-build-attestation.json'
        sha256 = Get-Sha256 -Path $stagedBuildAttestation
    }
    modelId = [string] $modelManifest.translationModelId
    resources = [ordered]@{
        hskBundled = $true
        dictionaryBundled = $true
        modelBundled = $true
        residentModelsBundled = $resolvedResidentResources.Count -eq 5
        residentModelCount = $resolvedResidentResources.Count
        residentRuntimeBundled = $resolvedResidentRuntimeFiles.Count -eq $expectedResidentRuntimeFiles.Count
        residentRuntimeFileCount = $resolvedResidentRuntimeFiles.Count
        hskInstallPath = 'resources\hsk-2.0.normalized.json'
        dictionaryInstallPath = 'resources\cc-cedict.normalized.json'
        modelInstallPath = 'resources\models\Qwen3.5-4B-Q4_K_M.gguf'
        runtimeInstallPath = 'resources\runtime'
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
