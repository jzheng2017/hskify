param(
    [string] $HskArtifactPath,
    [string] $DictionaryArtifactPath,
    [string] $SansFontPath,
    [string] $SerifFontPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$buildScript = Join-Path $PSScriptRoot 'Build-ReleasePackage.ps1'
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repositoryRoot 'scripts\PerformanceBuildAttestation.ps1')
if ([string]::IsNullOrWhiteSpace($HskArtifactPath)) {
    $HskArtifactPath = Join-Path $repositoryRoot '.cache\language-data-pinned\hsk-2.0.normalized.json'
}
if ([string]::IsNullOrWhiteSpace($DictionaryArtifactPath)) {
    $DictionaryArtifactPath = Join-Path $repositoryRoot '.cache\language-data-pinned\cc-cedict.normalized.json'
}
if ([string]::IsNullOrWhiteSpace($SansFontPath)) {
    $SansFontPath = Join-Path $env:WINDIR 'Fonts\NotoSansSC-VF.ttf'
}
if ([string]::IsNullOrWhiteSpace($SerifFontPath)) {
    $SerifFontPath = Join-Path $env:WINDIR 'Fonts\NotoSerifSC-VF.ttf'
}
foreach ($requiredArtifact in @($HskArtifactPath, $DictionaryArtifactPath, $SansFontPath, $SerifFontPath)) {
    if (-not (Test-Path -LiteralPath $requiredArtifact -PathType Leaf)) {
        throw "exact packaging-test artifact is missing: $requiredArtifact"
    }
}
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('hsk-manga-package-' + [Guid]::NewGuid().ToString('N'))
$testRegistryPath = 'HKCU:\Software\Hskify\Tests\' + [Guid]::NewGuid().ToString('N')
$previousLocalAppData = $env:LOCALAPPDATA

function Assert-True {
    param(
        [Parameter(Mandatory = $true)]
        [bool] $Condition,
        [Parameter(Mandatory = $true)]
        [string] $Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

try {
    [IO.Directory]::CreateDirectory($temporaryRoot) | Out-Null
    $env:LOCALAPPDATA = Join-Path $temporaryRoot 'local-app-data'
    [IO.Directory]::CreateDirectory($env:LOCALAPPDATA) | Out-Null

    $fixtureRoot = Join-Path $temporaryRoot 'fixtures'
    $extensionRoot = Join-Path $fixtureRoot 'extension'
    [IO.Directory]::CreateDirectory($extensionRoot) | Out-Null

    $nativeHostPath = Join-Path $fixtureRoot 'hsk-manga-native-host.exe'
    $browserDaemonPath = Join-Path $fixtureRoot 'hsk-manga-browser-daemon.exe'
    [IO.File]::WriteAllBytes($nativeHostPath, [byte[]](77, 90, 1, 2, 3))
    [IO.File]::WriteAllBytes($browserDaemonPath, [byte[]](77, 90, 4, 5, 6))
    [IO.File]::WriteAllBytes(
        (Join-Path $fixtureRoot 'onnxruntime_providers_shared.dll'),
        [byte[]](77, 90, 7, 8, 9)
    )
    [IO.File]::WriteAllBytes(
        (Join-Path $fixtureRoot 'onnxruntime_providers_cuda.dll'),
        [byte[]](77, 90, 10, 11, 12)
    )
    $fixtureAttestationPath = Join-Path $fixtureRoot 'performance-build-attestation.json'
    $cudaHome = Join-Path $repositoryRoot '.cache\tools\nvidia-python\nvidia\cu13'
    $toolchain = Get-HskifyPerformanceToolchainEvidence `
        -CmakePath (Join-Path $repositoryRoot '.cache\tools\cmake-4.4.0\cmake-4.4.0-windows-x86_64\bin\cmake.exe') `
        -NvccPath (Join-Path $cudaHome 'bin\nvcc.exe') `
        -MsvcPath (Get-HskifyMsvcCompilerPath)
    $hardware = Get-HskifyPerformanceGpuIdentity
    Assert-HskifyExactPerformanceGpu -Gpu $hardware
    $sourceIdentity = Get-HskifyPerformanceSourceTreeIdentity -RepositoryRoot $repositoryRoot
    $fixtureAttestation = New-HskifyPerformanceBuildAttestation `
        -SourceIdentity $sourceIdentity `
        -Hardware $hardware `
        -Toolchain $toolchain `
        -CudaHome $cudaHome `
        -NativeHostPath $nativeHostPath `
        -BrowserDaemonPath $browserDaemonPath
    [IO.File]::WriteAllText(
        $fixtureAttestationPath,
        ($fixtureAttestation | ConvertTo-Json -Depth 12) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )

    $extensionManifest = [ordered]@{
        manifest_version = 3
        name = 'Hskify smoke fixture'
        version = '0.1.0'
        browser_specific_settings = [ordered]@{
            gecko = [ordered]@{
                id = 'hsk-manga-translator@local.hskify'
            }
        }
    }
    [IO.File]::WriteAllText(
        (Join-Path $extensionRoot 'manifest.json'),
        ($extensionManifest | ConvertTo-Json -Depth 5),
        [Text.UTF8Encoding]::new($false)
    )
    $extensionZipPath = Join-Path $fixtureRoot 'fixture-firefox.zip'
    Compress-Archive -LiteralPath (Join-Path $extensionRoot 'manifest.json') -DestinationPath $extensionZipPath

    $hskPath = (Resolve-Path -LiteralPath $HskArtifactPath).Path
    $dictionaryPath = (Resolve-Path -LiteralPath $DictionaryArtifactPath).Path

    $modelPath = Join-Path $fixtureRoot 'tiny-model.gguf'
    [IO.File]::WriteAllBytes($modelPath, [Text.Encoding]::UTF8.GetBytes('tiny deterministic Qwen fixture'))
    $sansFontPath = (Resolve-Path -LiteralPath $SansFontPath).Path
    $serifFontPath = (Resolve-Path -LiteralPath $SerifFontPath).Path
    $modelHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $modelBytes = (Get-Item -LiteralPath $modelPath).Length
    $fixtureRevision = ('a' * 40) -join ''
    $residentModelsRoot = Join-Path $fixtureRoot 'resident-models'
    $residentDescriptors = @(
        [ordered]@{ id = 'comic-text-bubble-detector-config'; filename = 'config.json' },
        [ordered]@{ id = 'comic-text-bubble-detector-preprocessor-config'; filename = 'preprocessor_config.json' },
        [ordered]@{ id = 'comic-text-bubble-detector-weights'; filename = 'model.safetensors' },
        [ordered]@{ id = 'lama-manga-inpainter-weights'; filename = 'lama-manga.safetensors' },
        [ordered]@{ id = 'manga-text-segmentation-weights'; filename = 'model.safetensors' },
        [ordered]@{ id = 'pp-ocr-v6-small-detector-config'; filename = 'inference.yml' },
        [ordered]@{ id = 'pp-ocr-v6-small-detector-model'; filename = 'inference.onnx' },
        [ordered]@{ id = 'pp-ocr-v6-small-recognizer-config'; filename = 'inference.yml' },
        [ordered]@{ id = 'pp-ocr-v6-small-recognizer-model'; filename = 'inference.onnx' },
        [ordered]@{ id = 'speech-bubble-segmentation-config'; filename = 'config.json' },
        [ordered]@{ id = 'speech-bubble-segmentation-weights'; filename = 'model.safetensors' }
    )
    $resourceIdentities = @($residentDescriptors | ForEach-Object {
        $directory = Join-Path $residentModelsRoot $_.id
        [IO.Directory]::CreateDirectory($directory) | Out-Null
        $path = Join-Path $directory $_.filename
        [IO.File]::WriteAllBytes(
            $path,
            [Text.Encoding]::UTF8.GetBytes("resident fixture $($_.id)")
        )
        [ordered]@{
            id = $_.id
            repository = 'example/models'
            repositoryRevision = $fixtureRevision
            filename = $_.filename
            url = "https://huggingface.co/example/models/resolve/$fixtureRevision/$($_.filename)"
            bytes = (Get-Item -LiteralPath $path).Length
            sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    $residentRuntimeRoot = Join-Path $fixtureRoot 'resident-runtime'
    $runtimeFiles = @(
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
    foreach ($relativePath in $runtimeFiles) {
        $path = Join-Path $residentRuntimeRoot $relativePath
        [IO.Directory]::CreateDirectory((Split-Path -Parent $path)) | Out-Null
        $contents = switch ($relativePath) {
            'cuda\.installed' {
                'cuda;platform=win_amd64;wheels=nvidia-cuda-runtime/13.1.80,nvidia-cuda-nvrtc/13.1.80,nvidia-cublas/13.2.0.9,nvidia-cufft/12.1.0.31,nvidia-curand/10.4.1.81,nvidia-cudnn-cu13/9.17.0.29;extract=5'
            }
            'llama.cpp\b8935\windows-cuda13-x64\.installed' {
                'llama-b8935-windows-cuda13-x64-extract-2'
            }
            default { "runtime fixture $relativePath" }
        }
        [IO.File]::WriteAllText($path, $contents, [Text.UTF8Encoding]::new($false))
    }
    $resourceIdentities += [ordered]@{
        id = 'translation-model'
        repository = 'example/models'
        repositoryRevision = $fixtureRevision
        filename = 'Qwen3.5-4B-Q4_K_M.gguf'
        url = "https://huggingface.co/example/models/resolve/$fixtureRevision/Qwen3.5-4B-Q4_K_M.gguf"
        bytes = $modelBytes
        sha256 = $modelHash
    }
    $projectorId = 'translation-model-projector'
    $projectorDirectory = Join-Path $residentModelsRoot $projectorId
    [IO.Directory]::CreateDirectory($projectorDirectory) | Out-Null
    $projectorPath = Join-Path $projectorDirectory 'mmproj-BF16.gguf'
    [IO.File]::WriteAllBytes(
        $projectorPath,
        [Text.Encoding]::UTF8.GetBytes("resident fixture $projectorId")
    )
    $projectorBytes = (Get-Item -LiteralPath $projectorPath).Length
    $projectorHash = (Get-FileHash -LiteralPath $projectorPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $resourceIdentities += [ordered]@{
        id = $projectorId
        repository = 'example/models'
        repositoryRevision = $fixtureRevision
        filename = 'mmproj-BF16.gguf'
        url = "https://huggingface.co/example/models/resolve/$fixtureRevision/mmproj-BF16.gguf"
        bytes = $projectorBytes
        sha256 = $projectorHash
    }
    $modelManifest = [ordered]@{
        manifestVersion = 1
        generatedAt = '2026-07-24'
        translationModelId = 'qwen3.5-4b'
        resourceIdentities = @($resourceIdentities)
    }
    $modelManifestPath = Join-Path $fixtureRoot 'manifest.v1.json'
    [IO.File]::WriteAllText(
        $modelManifestPath,
        ($modelManifest | ConvertTo-Json -Depth 8),
        [Text.UTF8Encoding]::new($false)
    )

    try {
        & $buildScript `
            -OutputDirectory (Join-Path $temporaryRoot 'missing-attestation-bundle') `
            -NativeHostPath $nativeHostPath `
            -BrowserDaemonPath $browserDaemonPath `
            -FirefoxExtensionZipPath $extensionZipPath `
            -HskArtifactPath $hskPath `
            -DictionaryArtifactPath $dictionaryPath `
            -ModelPath $modelPath `
            -ResidentModelsDirectory $residentModelsRoot `
            -ResidentRuntimeDirectory $residentRuntimeRoot `
            -SansFontPath $sansFontPath `
            -SerifFontPath $serifFontPath `
            -ModelManifestPath $modelManifestPath
        throw 'the release package accepted explicitly supplied binaries without an attestation'
    }
    catch {
        if ($_.Exception.Message -notlike '*BuildAttestationPath is required*') {
            throw
        }
    }

    function Assert-RejectedAttestationClaim {
        param(
            [Parameter(Mandatory = $true)][string] $Name,
            [Parameter(Mandatory = $true)][scriptblock] $Mutate,
            [Parameter(Mandatory = $true)][string] $ExpectedMessage
        )

        $claim = Get-Content -LiteralPath $fixtureAttestationPath -Raw | ConvertFrom-Json
        & $Mutate $claim
        $claimPath = Join-Path $fixtureRoot "$Name-attestation.json"
        [IO.File]::WriteAllText(
            $claimPath,
            ($claim | ConvertTo-Json -Depth 12) + [Environment]::NewLine,
            [Text.UTF8Encoding]::new($false)
        )
        try {
            & $buildScript `
                -OutputDirectory (Join-Path $temporaryRoot "$Name-bundle") `
                -NativeHostPath $nativeHostPath `
                -BrowserDaemonPath $browserDaemonPath `
                -BuildAttestationPath $claimPath `
                -FirefoxExtensionZipPath $extensionZipPath `
                -HskArtifactPath $hskPath `
                -DictionaryArtifactPath $dictionaryPath `
                -ModelPath $modelPath `
                -ResidentModelsDirectory $residentModelsRoot `
                -ResidentRuntimeDirectory $residentRuntimeRoot `
                -SansFontPath $sansFontPath `
                -SerifFontPath $serifFontPath `
                -ModelManifestPath $modelManifestPath
            throw "the release package accepted the invalid $Name attestation claim"
        }
        catch {
            if ($_.Exception.Message -notlike "*$ExpectedMessage*") {
                throw
            }
        }
    }

    Assert-RejectedAttestationClaim `
        -Name 'fingerprint' `
        -Mutate { param($claim) $claim.buildFingerprint = 'arbitrary-build' } `
        -ExpectedMessage 'fingerprint mismatch'
    Assert-RejectedAttestationClaim `
        -Name 'target' `
        -Mutate { param($claim) $claim.build.target = 'x86_64-unknown-linux-gnu' } `
        -ExpectedMessage 'exact release target'
    Assert-RejectedAttestationClaim `
        -Name 'features' `
        -Mutate { param($claim) $claim.build.features = @('cuda', 'compatibility') } `
        -ExpectedMessage 'exact release target'
    Assert-RejectedAttestationClaim `
        -Name 'toolchain' `
        -Mutate { param($claim) $claim.toolchain.cargo.version = '0.0.0-forged' } `
        -ExpectedMessage 'current cargo version'
    Assert-RejectedAttestationClaim `
        -Name 'hardware' `
        -Mutate { param($claim) $claim.hardware.memoryTotalMiB = 16000 } `
        -ExpectedMessage 'requires CUDA device 0'

    $tamperedDaemonPath = Join-Path $fixtureRoot 'tampered\hsk-manga-browser-daemon.exe'
    [IO.Directory]::CreateDirectory((Split-Path -Parent $tamperedDaemonPath)) | Out-Null
    [IO.File]::WriteAllBytes($tamperedDaemonPath, [byte[]](77, 90, 9, 9, 9))
    try {
        & $buildScript `
            -OutputDirectory (Join-Path $temporaryRoot 'tampered-binary-bundle') `
            -NativeHostPath $nativeHostPath `
            -BrowserDaemonPath $tamperedDaemonPath `
            -BuildAttestationPath $fixtureAttestationPath `
            -FirefoxExtensionZipPath $extensionZipPath `
            -HskArtifactPath $hskPath `
            -DictionaryArtifactPath $dictionaryPath `
            -ModelPath $modelPath `
            -ResidentModelsDirectory $residentModelsRoot `
            -ResidentRuntimeDirectory $residentRuntimeRoot `
            -SansFontPath $sansFontPath `
            -SerifFontPath $serifFontPath `
            -ModelManifestPath $modelManifestPath
        throw 'the release package accepted a binary that did not match the attestation'
    }
    catch {
        if ($_.Exception.Message -notlike '*binary identity mismatch*') {
            throw
        }
    }

    $badModelPath = Join-Path $fixtureRoot 'bad-model.gguf'
    [IO.File]::WriteAllBytes($badModelPath, [Text.Encoding]::UTF8.GetBytes('wrong model bytes'))
    $rejectedOutput = Join-Path $temporaryRoot 'rejected-bundle'
    try {
        & $buildScript `
            -OutputDirectory $rejectedOutput `
            -NativeHostPath $nativeHostPath `
            -BrowserDaemonPath $browserDaemonPath `
            -BuildAttestationPath $fixtureAttestationPath `
            -FirefoxExtensionZipPath $extensionZipPath `
            -HskArtifactPath $hskPath `
            -DictionaryArtifactPath $dictionaryPath `
            -ModelPath $badModelPath `
            -ResidentModelsDirectory $residentModelsRoot `
            -ResidentRuntimeDirectory $residentRuntimeRoot `
            -SansFontPath $sansFontPath `
            -SerifFontPath $serifFontPath `
            -ModelManifestPath $modelManifestPath
        throw 'the release package accepted a model with the wrong SHA-256'
    }
    catch {
        if ($_.Exception.Message -notlike '*model SHA-256 mismatch*') {
            throw
        }
    }
    Assert-True -Condition (-not (Test-Path -LiteralPath $rejectedOutput)) -Message 'rejected model created a bundle'

    $unmarkedOutput = Join-Path $temporaryRoot 'unmarked-output'
    [IO.Directory]::CreateDirectory($unmarkedOutput) | Out-Null
    $unrelatedFile = Join-Path $unmarkedOutput 'keep-me.txt'
    [IO.File]::WriteAllText($unrelatedFile, 'unrelated user content')
    try {
        & $buildScript `
            -OutputDirectory $unmarkedOutput `
            -NativeHostPath $nativeHostPath `
            -BrowserDaemonPath $browserDaemonPath `
            -BuildAttestationPath $fixtureAttestationPath `
            -FirefoxExtensionZipPath $extensionZipPath `
            -HskArtifactPath $hskPath `
            -DictionaryArtifactPath $dictionaryPath `
            -ModelPath $modelPath `
            -ResidentModelsDirectory $residentModelsRoot `
            -ResidentRuntimeDirectory $residentRuntimeRoot `
            -SansFontPath $sansFontPath `
            -SerifFontPath $serifFontPath `
            -ModelManifestPath $modelManifestPath `
            -Force
        throw 'the release package replaced a populated unmarked directory'
    }
    catch {
        if ($_.Exception.Message -notlike '*without a valid prior HSK bundle marker*') {
            throw
        }
    }
    Assert-True `
        -Condition (Test-Path -LiteralPath $unrelatedFile -PathType Leaf) `
        -Message '-Force removed content from an unmarked directory'

    $bundleRoot = Join-Path $temporaryRoot 'bundle'
    & $buildScript `
        -OutputDirectory $bundleRoot `
        -NativeHostPath $nativeHostPath `
        -BrowserDaemonPath $browserDaemonPath `
        -BuildAttestationPath $fixtureAttestationPath `
        -FirefoxExtensionZipPath $extensionZipPath `
        -HskArtifactPath $hskPath `
        -DictionaryArtifactPath $dictionaryPath `
        -ModelPath $modelPath `
        -ResidentModelsDirectory $residentModelsRoot `
        -ResidentRuntimeDirectory $residentRuntimeRoot `
        -SansFontPath $sansFontPath `
        -SerifFontPath $serifFontPath `
        -ModelManifestPath $modelManifestPath | Out-Null

    foreach ($relativePath in @(
        'companion\hsk-manga-native-host.exe',
        'companion\hsk-manga-browser-daemon.exe',
        'companion\msvcp140.dll',
        'companion\msvcp140_1.dll',
        'companion\vcruntime140.dll',
        'companion\vcruntime140_1.dll',
        'companion\onnxruntime_providers_shared.dll',
        'companion\onnxruntime_providers_cuda.dll',
        'provenance\performance-build-attestation.json',
        'extension\hskify-firefox.zip',
        'resources\hsk-2.0.normalized.json',
        'resources\cc-cedict.normalized.json',
        'resources\models\Qwen3.5-4B-Q4_K_M.gguf',
        'resources\models\resident\comic-text-bubble-detector-config\config.json',
        'resources\models\resident\comic-text-bubble-detector-preprocessor-config\preprocessor_config.json',
        'resources\models\resident\comic-text-bubble-detector-weights\model.safetensors',
        'resources\models\resident\pp-ocr-v6-small-recognizer-config\inference.yml',
        'resources\models\resident\pp-ocr-v6-small-recognizer-model\inference.onnx',
        'resources\runtime\cuda\.installed',
        'resources\runtime\cuda\cudart64_13.dll',
        'resources\runtime\llama.cpp\b8935\windows-cuda13-x64\.installed',
        'resources\runtime\llama.cpp\b8935\windows-cuda13-x64\llama.dll',
        'resources\fonts\NotoSansSC-VF.ttf',
        'resources\fonts\NotoSerifSC-VF.ttf',
        'resources\model-packs\manifest.v1.json',
        'native-host-registration\Register-NativeHost.ps1',
        'native-host-registration\Unregister-NativeHost.ps1',
        'Install.ps1',
        'Uninstall.ps1',
        'bundle-manifest.json'
    )) {
        Assert-True `
            -Condition (Test-Path -LiteralPath (Join-Path $bundleRoot $relativePath) -PathType Leaf) `
            -Message "bundle file missing: $relativePath"
    }

    $bundleManifestPath = Join-Path $bundleRoot 'bundle-manifest.json'
    $bundleManifest = Get-Content -LiteralPath $bundleManifestPath -Raw | ConvertFrom-Json
    Assert-True -Condition $bundleManifest.resources.hskBundled -Message 'bundle did not record HSK data'
    Assert-True -Condition $bundleManifest.resources.dictionaryBundled -Message 'bundle did not record dictionary data'
    Assert-True -Condition $bundleManifest.resources.modelBundled -Message 'bundle did not record the translation model'
    Assert-True -Condition $bundleManifest.resources.residentModelsBundled -Message 'bundle did not record detector/OCR resources'
    Assert-True -Condition ($bundleManifest.resources.residentModelCount -eq 12) -Message 'bundle recorded the wrong resident resource count'
    Assert-True -Condition $bundleManifest.resources.residentRuntimeBundled -Message 'bundle did not record CUDA/llama runtime resources'
    Assert-True -Condition ($bundleManifest.resources.residentRuntimeFileCount -eq 39) -Message 'bundle recorded the wrong CUDA/llama runtime resource count'
    Assert-True `
        -Condition (@($bundleManifest.files | Where-Object role -eq 'onnx-runtime-provider').Count -eq 2) `
        -Message 'bundle manifest did not hash both ONNX Runtime CUDA provider DLLs'
    Assert-True `
        -Condition (@($bundleManifest.files | Where-Object role -eq 'resident-runtime').Count -eq 39) `
        -Message 'bundle manifest did not hash every CUDA/llama runtime file'
    Assert-True `
        -Condition ($bundleManifest.resources.expectedModelSha256 -eq $modelHash) `
        -Message 'bundle recorded the wrong mandatory model SHA-256'
    Assert-True `
        -Condition ($bundleManifest.modelId -eq 'qwen3.5-4b') `
        -Message 'bundle recorded the wrong mandatory model ID'
    Assert-True `
        -Condition ($bundleManifest.buildFingerprint -eq $script:HskifyPerformanceBuildFingerprint) `
        -Message 'bundle recorded the wrong performance-build fingerprint'
    Assert-True `
        -Condition ($bundleManifest.performanceBuildAttestation.schema -eq $script:HskifyPerformanceAttestationSchema) `
        -Message 'bundle recorded the wrong performance-build attestation schema'
    $bundledAttestationPath = Join-Path $bundleRoot $bundleManifest.performanceBuildAttestation.path
    Assert-True `
        -Condition (
            (Get-FileHash -LiteralPath $bundledAttestationPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq
            $bundleManifest.performanceBuildAttestation.sha256
        ) `
        -Message 'bundle recorded the wrong performance-build attestation hash'
    $bundledAttestation = Assert-HskifyPerformanceBuildAttestation `
        -AttestationPath $bundledAttestationPath `
        -NativeHostPath (Join-Path $bundleRoot 'companion\hsk-manga-native-host.exe') `
        -BrowserDaemonPath (Join-Path $bundleRoot 'companion\hsk-manga-browser-daemon.exe') `
        -VerifyCurrentHardware `
        -VerifyCurrentToolchain

    $productRoot = Join-Path $env:LOCALAPPDATA 'Hskify'
    & powershell `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File (Join-Path $bundleRoot 'Install.ps1') `
        -ProductRoot $productRoot `
        -RegistryPath $testRegistryPath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "documented install command failed with exit code $LASTEXITCODE"
    }

    $installedNativeHost = Join-Path $productRoot 'app\companion\hsk-manga-native-host.exe'
    $installedVcRuntime = @(
        'msvcp140.dll',
        'msvcp140_1.dll',
        'vcruntime140.dll',
        'vcruntime140_1.dll'
    ) | ForEach-Object { Join-Path $productRoot "app\companion\$_" }
    $installedOnnxRuntimeProviders = @(
        'onnxruntime_providers_shared.dll',
        'onnxruntime_providers_cuda.dll'
    ) | ForEach-Object { Join-Path $productRoot "app\companion\$_" }
    $installedHsk = Join-Path $productRoot 'resources\hsk-2.0.normalized.json'
    $installedDictionary = Join-Path $productRoot 'resources\cc-cedict.normalized.json'
    $installedModel = Join-Path $productRoot 'resources\models\Qwen3.5-4B-Q4_K_M.gguf'
    $installedProjector = Join-Path $productRoot 'resources\models\resident\translation-model-projector\mmproj-BF16.gguf'
    $installedResidentModels = @($residentDescriptors | ForEach-Object {
        Join-Path $productRoot "resources\models\resident\$($_.id)\$($_.filename)"
    })
    $installedRuntimeFiles = @(
        Join-Path $productRoot 'resources\runtime\cuda\.installed'
        Join-Path $productRoot 'resources\runtime\cuda\cudart64_13.dll'
        Join-Path $productRoot 'resources\runtime\llama.cpp\b8935\windows-cuda13-x64\.installed'
        Join-Path $productRoot 'resources\runtime\llama.cpp\b8935\windows-cuda13-x64\llama.dll'
    )
    $installedSansFont = Join-Path $productRoot 'resources\fonts\NotoSansSC-VF.ttf'
    $installedSerifFont = Join-Path $productRoot 'resources\fonts\NotoSerifSC-VF.ttf'
    $installedAttestation = Join-Path $productRoot 'app\provenance\performance-build-attestation.json'
    $installedReadinessMarker = Join-Path $productRoot 'browser-companion\browser-cache\browser-runtime\models.ready'
    foreach ($installedPath in @($installedNativeHost) + $installedVcRuntime + $installedOnnxRuntimeProviders + @($installedHsk, $installedDictionary, $installedModel, $installedProjector) + $installedResidentModels + $installedRuntimeFiles + @($installedSansFont, $installedSerifFont, $installedAttestation, $installedReadinessMarker)) {
        Assert-True -Condition (Test-Path -LiteralPath $installedPath -PathType Leaf) -Message "installed file missing: $installedPath"
    }
    $readinessMarker = Get-Content -LiteralPath $installedReadinessMarker -Raw | ConvertFrom-Json
    Assert-True `
        -Condition (
            $readinessMarker.buildFingerprint -eq $script:HskifyPerformanceBuildFingerprint -and
            @($readinessMarker.resourceIdentities).Count -eq 13 -and
            @($readinessMarker.installations).Count -eq 13
        ) `
        -Message 'installer wrote an invalid model-readiness marker'

    $nativeManifestPath = (Get-Item -LiteralPath $testRegistryPath).GetValue('')
    Assert-True `
        -Condition (Test-Path -LiteralPath $nativeManifestPath -PathType Leaf) `
        -Message 'install did not create the native-host manifest'
    $nativeManifest = Get-Content -LiteralPath $nativeManifestPath -Raw | ConvertFrom-Json
    Assert-True -Condition ($nativeManifest.name -eq 'local.hskify.hsk_manga') -Message 'wrong native-host name'
    Assert-True -Condition ($nativeManifest.path -eq $installedNativeHost) -Message 'native manifest does not point at the installed host'
    Assert-True `
        -Condition (@($nativeManifest.allowed_extensions).Count -eq 1 -and $nativeManifest.allowed_extensions[0] -eq 'hsk-manga-translator@local.hskify') `
        -Message 'native manifest allows the wrong Firefox extension'

    $obsoleteAppFile = Join-Path $productRoot 'app\obsolete-from-previous-build'
    $obsoleteResourceFile = Join-Path $productRoot 'resources\obsolete-from-previous-build'
    [IO.File]::WriteAllText($obsoleteAppFile, 'obsolete')
    [IO.File]::WriteAllText($obsoleteResourceFile, 'obsolete')
    & powershell `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File (Join-Path $bundleRoot 'Install.ps1') `
        -ProductRoot $productRoot `
        -RegistryPath $testRegistryPath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "documented update command failed with exit code $LASTEXITCODE"
    }
    Assert-True `
        -Condition (-not (Test-Path -LiteralPath $obsoleteAppFile)) `
        -Message 'update retained an obsolete app file'
    Assert-True `
        -Condition (-not (Test-Path -LiteralPath $obsoleteResourceFile)) `
        -Message 'update retained an obsolete resource file'
    Assert-True `
        -Condition (@(Get-ChildItem -LiteralPath $productRoot -Force | Where-Object Name -Like '.previous-*').Count -eq 0) `
        -Message 'update retained a previous-build directory'
    Assert-True `
        -Condition (@(Get-ChildItem -LiteralPath $productRoot -Force | Where-Object Name -Like '.update-*').Count -eq 0) `
        -Message 'update retained a staging directory'

    $stateCache = Join-Path $productRoot 'browser-companion\browser-cache'
    [IO.Directory]::CreateDirectory($stateCache) | Out-Null
    [IO.File]::WriteAllText((Join-Path $stateCache 'dummy-cache'), 'fixture')

    & powershell `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File (Join-Path $productRoot 'app\Uninstall.ps1') `
        -ProductRoot $productRoot `
        -RegistryPath $testRegistryPath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "installed uninstall command failed with exit code $LASTEXITCODE"
    }
    Assert-True -Condition (-not (Test-Path -LiteralPath $testRegistryPath)) -Message 'uninstall left the isolated registry key'
    Assert-True -Condition (-not (Test-Path -LiteralPath $nativeManifestPath)) -Message 'uninstall left the native-host manifest'
    Assert-True -Condition (-not (Test-Path -LiteralPath $productRoot)) -Message 'uninstall left product files or cache'

    Write-Output 'Windows release package checks passed'
}
finally {
    if (Test-Path -LiteralPath $testRegistryPath) {
        Remove-Item -LiteralPath $testRegistryPath -Force
    }
    $env:LOCALAPPDATA = $previousLocalAppData
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
