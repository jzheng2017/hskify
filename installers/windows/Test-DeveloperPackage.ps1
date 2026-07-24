$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$buildScript = Join-Path $PSScriptRoot 'Build-DeveloperPackage.ps1'
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('hsk-manga-package-' + [Guid]::NewGuid().ToString('N'))
$testRegistryPath = 'HKCU:\Software\Mangalations\HSKMangaTranslator\Tests\' + [Guid]::NewGuid().ToString('N')
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

    $extensionManifest = [ordered]@{
        manifest_version = 3
        name = 'HSK Manga Translator smoke fixture'
        version = '0.1.0'
        browser_specific_settings = [ordered]@{
            gecko = [ordered]@{
                id = 'hsk-manga-translator@local.mangalations'
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

    $hskPath = Join-Path $fixtureRoot 'tiny-hsk.json'
    $dictionaryPath = Join-Path $fixtureRoot 'tiny-dictionary.json'
    [IO.File]::WriteAllText($hskPath, '{"entries":[{"word":"fixture","level":1}]}', [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($dictionaryPath, '{"entries":[{"word":"fixture","gloss":"test"}]}', [Text.UTF8Encoding]::new($false))

    $modelPath = Join-Path $fixtureRoot 'tiny-model.gguf'
    [IO.File]::WriteAllBytes($modelPath, [Text.Encoding]::UTF8.GetBytes('tiny deterministic Qwen fixture'))
    $modelHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $modelBytes = (Get-Item -LiteralPath $modelPath).Length
    $modelManifest = [ordered]@{
        manifestVersion = 1
        selection = [ordered]@{
            status = 'selected'
            standardPackId = 'standard-v1'
        }
        packs = @(
            [ordered]@{
                id = 'standard-v1'
                runtimeModelId = 'qwen3.5-4b'
                files = @(
                    [ordered]@{
                        id = 'translation-model'
                        filename = 'Qwen3.5-4B-Q4_K_M.gguf'
                        sha256 = $modelHash
                        bytes = $modelBytes
                    }
                )
            }
        )
    }
    $modelManifestPath = Join-Path $fixtureRoot 'manifest.v1.json'
    [IO.File]::WriteAllText(
        $modelManifestPath,
        ($modelManifest | ConvertTo-Json -Depth 8),
        [Text.UTF8Encoding]::new($false)
    )

    $badModelPath = Join-Path $fixtureRoot 'bad-model.gguf'
    [IO.File]::WriteAllBytes($badModelPath, [Text.Encoding]::UTF8.GetBytes('wrong model bytes'))
    $rejectedOutput = Join-Path $temporaryRoot 'rejected-bundle'
    try {
        & $buildScript `
            -OutputDirectory $rejectedOutput `
            -NativeHostPath $nativeHostPath `
            -BrowserDaemonPath $browserDaemonPath `
            -FirefoxExtensionZipPath $extensionZipPath `
            -HskArtifactPath $hskPath `
            -DictionaryArtifactPath $dictionaryPath `
            -ModelPath $badModelPath `
            -ModelManifestPath $modelManifestPath
        throw 'the developer package accepted a model with the wrong SHA-256'
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
            -FirefoxExtensionZipPath $extensionZipPath `
            -ModelManifestPath $modelManifestPath `
            -Force
        throw 'the developer package replaced a populated unmarked directory'
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
        -FirefoxExtensionZipPath $extensionZipPath `
        -HskArtifactPath $hskPath `
        -DictionaryArtifactPath $dictionaryPath `
        -ModelPath $modelPath `
        -ModelManifestPath $modelManifestPath | Out-Null

    foreach ($relativePath in @(
        'companion\hsk-manga-native-host.exe',
        'companion\hsk-manga-browser-daemon.exe',
        'extension\hsk-manga-translator-firefox.zip',
        'resources\hsk-2.0.normalized.json',
        'resources\cc-cedict.normalized.json',
        'resources\models\Qwen3.5-4B-Q4_K_M.gguf',
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
    Assert-True `
        -Condition ($bundleManifest.resources.expectedModelSha256 -eq $modelHash) `
        -Message 'bundle recorded the wrong selected model SHA-256'

    $productRoot = Join-Path $env:LOCALAPPDATA 'Mangalations\HSKMangaTranslator'
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
    $installedHsk = Join-Path $productRoot 'resources\hsk-2.0.normalized.json'
    $installedDictionary = Join-Path $productRoot 'resources\cc-cedict.normalized.json'
    $installedModel = Join-Path $productRoot 'resources\models\Qwen3.5-4B-Q4_K_M.gguf'
    foreach ($installedPath in @($installedNativeHost, $installedHsk, $installedDictionary, $installedModel)) {
        Assert-True -Condition (Test-Path -LiteralPath $installedPath -PathType Leaf) -Message "installed file missing: $installedPath"
    }

    $nativeManifestPath = (Get-Item -LiteralPath $testRegistryPath).GetValue('')
    Assert-True `
        -Condition (Test-Path -LiteralPath $nativeManifestPath -PathType Leaf) `
        -Message 'install did not create the native-host manifest'
    $nativeManifest = Get-Content -LiteralPath $nativeManifestPath -Raw | ConvertFrom-Json
    Assert-True -Condition ($nativeManifest.name -eq 'local.mangalations.hsk_manga') -Message 'wrong native-host name'
    Assert-True -Condition ($nativeManifest.path -eq $installedNativeHost) -Message 'native manifest does not point at the installed host'
    Assert-True `
        -Condition (@($nativeManifest.allowed_extensions).Count -eq 1 -and $nativeManifest.allowed_extensions[0] -eq 'hsk-manga-translator@local.mangalations') `
        -Message 'native manifest allows the wrong Firefox extension'

    $stateCache = Join-Path $productRoot 'browser-companion-v1\browser-cache-v1'
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

    Write-Output 'Windows developer package checks passed'
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
