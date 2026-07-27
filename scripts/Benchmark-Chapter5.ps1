[CmdletBinding()]
param(
    [ValidateRange(20, 10000)]
    [int] $Iterations = 20,

    [ValidateRange(1024, 65535)]
    [int] $Port = 43100,

    [ValidateRange(1, 6)]
    [int] $HskLevel = 5,

    [ValidateRange(1, 240)]
    [int] $RunTimeoutMinutes = 60,

    [ValidateRange(100, 5000)]
    [int] $TelemetryIntervalMs = 1000,

    [string] $OutputDirectory = '',

    [string] $ExtensionPackagePath = '',

    [string] $FirefoxExecutable = '',

    [string] $ResourcesDirectory = '',

    [string] $HskResourcePath = '',

    [string] $DictionaryResourcePath = '',

    [string] $QwenModelPath = '',

    [string] $FontsDirectory = '',

    [string] $NativeHostExecutable = '',

    [string] $BuildAttestationPath = '',

    [string] $LiveSmokeChapterUrl = '',

    [switch] $PrerequisitesOnly,

    [switch] $Headed
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'PerformanceBuildAttestation.ps1')
$benchmarkId = '30-years-since-the-prologue-chapter-5'
$fixtureRoot = Join-Path $repositoryRoot "fixtures\benchmarks\$benchmarkId"
$manifestPath = Join-Path $fixtureRoot 'manifest.json'
$isLiveSmoke = -not [string]::IsNullOrWhiteSpace($LiveSmokeChapterUrl)
$driverPath = Join-Path $PSScriptRoot $(if ($isLiveSmoke) {
    'benchmark\Chapter5.LiveAsura.Firefox.mjs'
} else {
    'benchmark\Chapter5.Firefox.mjs'
})
$playwrightModule = Join-Path $repositoryRoot 'node_modules\playwright'
$modelManifestPath = Join-Path $repositoryRoot 'data\model-packs\manifest.v1.json'
$expectedBuildFingerprint = $script:HskifyPerformanceBuildFingerprint
$expectedNativeHost = 'local.hskify.hsk_manga'
$expectedExtensionId = 'hsk-manga-translator@local.hskify'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

if ($isLiveSmoke) {
    $parsedLiveUrl = $null
    if (
        -not [Uri]::TryCreate($LiveSmokeChapterUrl, [UriKind]::Absolute, [ref] $parsedLiveUrl) -or
        $parsedLiveUrl.Scheme -notin @('http', 'https') -or
        -not [string]::IsNullOrEmpty($parsedLiveUrl.UserInfo)
    ) {
        throw 'LiveSmokeChapterUrl must be an explicit credential-free HTTP or HTTPS URL'
    }
    $LiveSmokeChapterUrl = $parsedLiveUrl.AbsoluteUri
}

function Assert-NoNull {
    param(
        [AllowNull()] $Value,
        [Parameter(Mandatory = $true)][string] $JsonPath
    )

    if ($null -eq $Value) {
        throw "Null placeholder is not permitted at $JsonPath"
    }
    if ($Value -is [string] -or $Value -is [ValueType]) {
        return
    }
    if ($Value -is [System.Collections.IDictionary]) {
        foreach ($key in $Value.Keys) {
            Assert-NoNull -Value $Value[$key] -JsonPath "$JsonPath.$key"
        }
        return
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $index = 0
        foreach ($item in $Value) {
            Assert-NoNull -Value $item -JsonPath "$JsonPath[$index]"
            $index += 1
        }
        return
    }
    foreach ($property in $Value.PSObject.Properties) {
        Assert-NoNull -Value $property.Value -JsonPath "$JsonPath.$($property.Name)"
    }
}

function Assert-NormalizedPolygon {
    param(
        [Parameter(Mandatory = $true)] $Polygon,
        [Parameter(Mandatory = $true)][string] $JsonPath
    )

    $points = @($Polygon)
    if ($points.Count -lt 4) {
        throw "$JsonPath must contain at least four points"
    }
    for ($index = 0; $index -lt $points.Count; $index += 1) {
        $point = @($points[$index])
        if ($point.Count -ne 2) {
            throw "$JsonPath[$index] must be an [x, y] pair"
        }
        foreach ($coordinate in $point) {
            $number = [double] $coordinate
            if ($number -lt 0.0 -or $number -gt 1.0) {
                throw "$JsonPath[$index] contains an out-of-range coordinate: $number"
            }
        }
    }
}

function Read-AndValidateFixture {
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Benchmark manifest is missing: $manifestPath"
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
    Assert-NoNull -Value $manifest -JsonPath '$'

    if ($manifest.schemaVersion -ne 3) {
        throw "Expected manifest schemaVersion 3, got $($manifest.schemaVersion)"
    }
    if ($manifest.id -ne $benchmarkId) {
        throw "Unexpected benchmark id: $($manifest.id)"
    }
    if ($manifest.pageCount -lt 1 -or @($manifest.images).Count -ne $manifest.pageCount) {
        throw 'The authoritative benchmark pageCount must match its non-empty images array'
    }
    if (
        $manifest.annotationStatus.reviewedPageCount -ne $manifest.pageCount -or
        $manifest.annotationStatus.generatedPageCount -ne $manifest.pageCount -or
        $manifest.annotationStatus.requiredPageCount -ne $manifest.pageCount -or
        $manifest.annotationStatus.completedPageCount -lt 0 -or
        $manifest.annotationStatus.completedPageCount -gt $manifest.pageCount -or
        $manifest.annotationStatus.status -notin @('complete', 'incomplete')
    ) {
        throw 'The annotationStatus object is inconsistent with the canonical 36-page fixture'
    }
    if ($manifest.annotationStatus.status -eq 'complete') {
        if (
            $manifest.annotationStatus.completedPageCount -ne $manifest.pageCount -or
            @($manifest.annotationStatus.missingPages).Count -ne 0 -or
            $manifest.annotationStatus.totalMissingFieldCount -ne 0
        ) {
            throw 'A complete annotationStatus must have every page complete and no missing gold fields'
        }
    }
    elseif (
        $manifest.annotationStatus.completedPageCount -ge $manifest.pageCount -or
        @($manifest.annotationStatus.missingPages).Count -eq 0 -or
        $manifest.annotationStatus.totalMissingFieldCount -lt 1
    ) {
        throw 'An incomplete annotationStatus must identify unfinished pages and missing gold fields'
    }
    if ($manifest.totalExpectedRegionCount -lt 1) {
        throw 'The completed gold fixture must contain at least one reviewed region'
    }
    if (
        $manifest.totalExpectedDialogueBubbleCount -lt 0 -or
        $manifest.totalExpectedNarrationCount -lt 0 -or
        $manifest.totalExpectedDialogueBubbleCount +
            $manifest.totalExpectedNarrationCount -ne
            $manifest.totalExpectedRegionCount -or
        $manifest.totalExpectedEnglishTranslationTargetCount -lt 1 -or
        $manifest.totalExpectedUntouchedExclusionCount -lt 0 -or
        $manifest.totalExpectedEnglishTranslationTargetCount +
            $manifest.totalExpectedUntouchedExclusionCount -ne
            $manifest.totalExpectedRegionCount
    ) {
        throw 'The fixture must partition every reviewed region into an English target or untouched exclusion'
    }

    $schemaPath = Join-Path $fixtureRoot $manifest.annotationSchema
    if (-not (Test-Path -LiteralPath $schemaPath -PathType Leaf)) {
        throw "Annotation schema is missing: $schemaPath"
    }
    $schemaFile = Get-Item -LiteralPath $schemaPath
    if ($schemaFile.Length -ne $manifest.annotationSchemaBytes) {
        throw "Annotation schema byte size mismatch: expected $($manifest.annotationSchemaBytes), got $($schemaFile.Length)"
    }
    $schemaHash = (Get-FileHash -LiteralPath $schemaPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($schemaHash -ne $manifest.annotationSchemaSha256) {
        throw "Annotation schema SHA-256 mismatch: expected $($manifest.annotationSchemaSha256), got $schemaHash"
    }

    foreach ($asset in @($manifest.replicaAssets)) {
        $assetPath = Join-Path $fixtureRoot ($asset.path -replace '/', '\')
        if (-not (Test-Path -LiteralPath $assetPath -PathType Leaf)) {
            throw "Replica asset is missing: $assetPath"
        }
        $assetFile = Get-Item -LiteralPath $assetPath
        $assetHash = (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($assetFile.Length -ne $asset.bytes -or $assetHash -ne $asset.sha256) {
            throw "Replica asset identity mismatch for $($asset.path)"
        }
    }

    $sourceRoot = Join-Path $repositoryRoot ($manifest.localSourceDirectory -replace '/', '\')
    $totalRegions = 0
    $totalDetectorGoldRegions = 0
    $totalNarrationRegions = 0
    $totalTranslationTargets = 0
    $totalUntouchedExclusions = 0
    $totalSourceBytes = [int64] 0
    $totalSourcePixels = [int64] 0
    $validatedPages = [System.Collections.Generic.List[object]]::new()
    foreach ($image in @($manifest.images | Sort-Object order)) {
        if (
            $image.expectedEnglishTranslationTargetCount +
                $image.expectedUntouchedExclusionCount -ne
                $image.expectedRegionCount -or
            $image.expectedDialogueBubbleCount +
                $image.expectedNarrationCount -ne
                $image.expectedRegionCount
        ) {
            throw "Page $($image.order) manifest counts do not partition its reviewed regions"
        }
        $sourcePath = Join-Path $sourceRoot $image.file
        if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
            throw "Source asset is missing: $sourcePath"
        }
        $sourceFile = Get-Item -LiteralPath $sourcePath
        $sourceHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sourceFile.Length -ne $image.bytes -or $sourceHash -ne $image.sha256) {
            throw "Source identity mismatch for $($image.file)"
        }
        $totalSourceBytes += [int64] $sourceFile.Length
        $totalSourcePixels += [int64] $image.width * [int64] $image.height

        $annotationPath = Join-Path $fixtureRoot ($image.annotation -replace '/', '\')
        if (-not (Test-Path -LiteralPath $annotationPath -PathType Leaf)) {
            throw "Annotation is missing: $annotationPath"
        }
        $annotationFile = Get-Item -LiteralPath $annotationPath
        $annotationHash = (Get-FileHash -LiteralPath $annotationPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($annotationFile.Length -ne $image.annotationBytes -or $annotationHash -ne $image.annotationSha256) {
            throw "Annotation identity mismatch for $($image.annotation)"
        }
        $annotation = Get-Content -LiteralPath $annotationPath -Raw -Encoding utf8 | ConvertFrom-Json
        Assert-NoNull -Value $annotation -JsonPath $image.annotation
        if (
            $annotation.schemaVersion -ne 1 -or
            $annotation.page.order -ne $image.order -or
            $annotation.page.file -ne $image.file -or
            $annotation.page.width -ne $image.width -or
            $annotation.page.height -ne $image.height -or
            $annotation.page.sourceSha256 -ne $image.sha256
        ) {
            throw "Annotation page metadata does not match the manifest for $($image.annotation)"
        }
        $regions = @($annotation.regions)
        if ($regions.Count -ne $image.expectedRegionCount) {
            throw "Region count mismatch for $($image.annotation): expected $($image.expectedRegionCount), got $($regions.Count)"
        }
        $ids = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        $pageDetectorGoldRegions = 0
        $pageNarrationRegions = 0
        $pageTranslationTargets = 0
        $pageUntouchedExclusions = 0
        for ($regionIndex = 0; $regionIndex -lt $regions.Count; $regionIndex += 1) {
            $region = $regions[$regionIndex]
            $regionPath = "$($image.annotation).regions[$regionIndex]"
            $expectedRegionId = '30ysp-ch5-p{0:d3}-r{1:d2}' -f $image.order, $regionIndex
            if (
                $region.readingOrder -ne $regionIndex -or
                [string] $region.id -cne $expectedRegionId -or
                -not $ids.Add([string] $region.id)
            ) {
                throw "$regionPath has invalid reading order or a duplicate ID"
            }
            if ($region.kind -notin @('dialogue', 'thought', 'narration')) {
                throw "$regionPath has unsupported kind $($region.kind)"
            }
            if ($region.kind -eq 'narration') {
                $pageNarrationRegions += 1
            }
            else {
                $pageDetectorGoldRegions += 1
            }
            foreach ($name in @('sourceEnglish', 'normalizedEnglish')) {
                if ([string]::IsNullOrWhiteSpace([string] $region.$name)) {
                    throw "$regionPath.$name must not be empty"
                }
            }
            $alphabetic = @(
                ([string] $region.sourceEnglish).ToCharArray() |
                    Where-Object { [char]::IsLetter($_) }
            )
            $nonLatinAlphabetic = @($alphabetic | Where-Object {
                $codePoint = [int] $_
                -not (
                    ($codePoint -ge 0x0041 -and $codePoint -le 0x005a) -or
                    ($codePoint -ge 0x0061 -and $codePoint -le 0x007a) -or
                    ($codePoint -ge 0x00c0 -and $codePoint -le 0x024f) -or
                    ($codePoint -ge 0x1e00 -and $codePoint -le 0x1eff)
                )
            })
            $confidentEnglish = (
                $alphabetic.Count -gt 0 -and $nonLatinAlphabetic.Count -eq 0
            )
            $translationTargetProperty = $region.PSObject.Properties['translationTarget']
            if ($null -eq $translationTargetProperty) {
                if (-not $confidentEnglish) {
                    throw "$regionPath is not confident Latin English and must be marked translationTarget=false"
                }
                $pageTranslationTargets += 1
            }
            else {
                if ($region.translationTarget -ne $false -or $confidentEnglish) {
                    throw "$regionPath.translationTarget is valid only as false on a language-ambiguous non-English region"
                }
                $pageUntouchedExclusions += 1
            }
            Assert-NormalizedPolygon -Polygon $region.textPolygon -JsonPath "$regionPath.textPolygon"
            $bubblePolygonProperty = $region.PSObject.Properties['bubblePolygon']
            if ($null -ne $bubblePolygonProperty) {
                Assert-NormalizedPolygon -Polygon $region.bubblePolygon -JsonPath "$regionPath.bubblePolygon"
            }
            if ($region.eraseMask.encoding -ne 'normalized-polygon-v1') {
                throw "$regionPath.eraseMask uses unsupported encoding $($region.eraseMask.encoding)"
            }
            Assert-NormalizedPolygon -Polygon $region.eraseMask.polygon -JsonPath "$regionPath.eraseMask.polygon"
        }
        if (
            $pageDetectorGoldRegions -ne $image.expectedDialogueBubbleCount -or
            $pageNarrationRegions -ne $image.expectedNarrationCount -or
            $pageTranslationTargets -ne $image.expectedEnglishTranslationTargetCount -or
            $pageUntouchedExclusions -ne $image.expectedUntouchedExclusionCount
        ) {
            throw "Canonical region-kind or translation-eligibility count mismatch for $($image.annotation)"
        }
        $totalRegions += $regions.Count
        $totalDetectorGoldRegions += $pageDetectorGoldRegions
        $totalNarrationRegions += $pageNarrationRegions
        $totalTranslationTargets += $pageTranslationTargets
        $totalUntouchedExclusions += $pageUntouchedExclusions
        $validatedPages.Add([ordered]@{
            order = $image.order
            sourceSha256 = $sourceHash
            annotationSha256 = $annotationHash
            regionCount = $regions.Count
            dialogueBubbleCount = $pageDetectorGoldRegions
            detectedBubbleGoldCount = $pageDetectorGoldRegions
            narrationRegionCount = $pageNarrationRegions
            englishTranslationTargetCount = $pageTranslationTargets
            untouchedExclusionCount = $pageUntouchedExclusions
        })
    }
    if (
        $totalRegions -ne $manifest.totalExpectedRegionCount -or
        $totalDetectorGoldRegions -ne $manifest.totalExpectedDialogueBubbleCount -or
        $totalNarrationRegions -ne $manifest.totalExpectedNarrationCount -or
        $totalTranslationTargets -ne $manifest.totalExpectedEnglishTranslationTargetCount -or
        $totalUntouchedExclusions -ne $manifest.totalExpectedUntouchedExclusionCount -or
        $totalSourceBytes -ne $manifest.totalSourceBytes -or
        $totalSourcePixels -ne $manifest.totalSourcePixels
    ) {
        throw 'Completed fixture totals do not match the manifest'
    }
    return [ordered]@{
        manifest = $manifest
        sourceRoot = $sourceRoot
        schemaSha256 = $schemaHash
        totalRegions = $totalRegions
        totalDetectorGoldRegions = $totalDetectorGoldRegions
        totalNarrationRegions = $totalNarrationRegions
        totalTranslationTargets = $totalTranslationTargets
        totalUntouchedExclusions = $totalUntouchedExclusions
        totalSourceBytes = $totalSourceBytes
        totalSourcePixels = $totalSourcePixels
        pages = @($validatedPages)
    }
}

function Assert-CompleteTranslationGold {
    param(
        [Parameter(Mandatory = $true)] $Fixture
    )

    $status = $Fixture.manifest.annotationStatus
    if (
        $status.status -ne 'complete' -or
        $status.completedPageCount -ne $status.requiredPageCount -or
        $status.requiredPageCount -ne $Fixture.manifest.pageCount -or
        @($status.missingPages).Count -ne 0 -or
        $status.totalMissingFieldCount -ne 0
    ) {
        throw (
            "Chapter 5 release measurement is blocked by incomplete translation gold: " +
            "status=$($status.status), completedPageCount=$($status.completedPageCount), " +
            "requiredPageCount=$($status.requiredPageCount), reasonCode=$($status.reasonCode), " +
            "missingFieldCounts=$($status.missingFieldCounts | ConvertTo-Json -Compress)"
        )
    }
}

function Get-Percentile {
    param(
        [Parameter(Mandatory = $true)][double[]] $Values,
        [Parameter(Mandatory = $true)][ValidateRange(0.0, 1.0)][double] $Percentile
    )
    if ($Values.Count -eq 0) {
        throw 'Cannot calculate a percentile over an empty sample'
    }
    $sorted = @($Values | Sort-Object)
    $index = [Math]::Max(0, [Math]::Ceiling($Percentile * $sorted.Count) - 1)
    return [Math]::Round([double] $sorted[$index], 3)
}

function Resolve-DefaultFirefox {
    param([Parameter(Mandatory = $true)] $NodeCommand)
    $escaped = $playwrightModule -replace '\\', '/'
    $path = & $NodeCommand.Source -e "process.stdout.write(require('$escaped').firefox.executablePath())"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($path)) {
        return ''
    }
    return [string] $path
}

function Read-ExtensionPackage {
    param([Parameter(Mandatory = $true)][string] $Path)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $manifestEntry = @($archive.Entries | Where-Object { $_.FullName -eq 'manifest.json' })
        if ($manifestEntry.Count -ne 1) {
            throw 'Packaged extension must contain one root manifest.json'
        }
        $reader = New-Object IO.StreamReader($manifestEntry[0].Open(), [Text.Encoding]::UTF8)
        try {
            $manifest = $reader.ReadToEnd() | ConvertFrom-Json
        }
        finally {
            $reader.Dispose()
        }
        $fingerprintFound = $false
        foreach ($entry in @($archive.Entries | Where-Object { $_.FullName -like '*.js' })) {
            $scriptReader = New-Object IO.StreamReader($entry.Open(), [Text.Encoding]::UTF8)
            try {
                if ($scriptReader.ReadToEnd().Contains($expectedBuildFingerprint)) {
                    $fingerprintFound = $true
                    break
                }
            }
            finally {
                $scriptReader.Dispose()
            }
        }
        return [ordered]@{
            manifest = $manifest
            fingerprintFound = $fingerprintFound
            sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = (Get-Item -LiteralPath $Path).Length
        }
    }
    finally {
        $archive.Dispose()
    }
}

function Assert-CurrentExtensionPackage {
    param([Parameter(Mandatory = $true)] $Extension)
    $geckoId = [string] $Extension.manifest.browser_specific_settings.gecko.id
    if (
        $Extension.manifest.manifest_version -ne 3 -or
        $geckoId -ne $expectedExtensionId -or
        @($Extension.manifest.permissions) -notcontains 'nativeMessaging' -or
        -not $Extension.fingerprintFound
    ) {
        throw 'Packaged extension manifest/fingerprint does not match the direct performance build'
    }
}

function Build-CurrentExtensionPackage {
    param([Parameter(Mandatory = $true)][string] $PackagePath)
    $extensionRoot = Join-Path $repositoryRoot 'extensions\firefox'
    $wxtCommand = Join-Path $repositoryRoot 'node_modules\.bin\wxt.cmd'
    $buildRoot = Join-Path $extensionRoot '.output\firefox-mv3'
    if (Test-Path -LiteralPath $PackagePath) {
        throw "Refusing to overwrite an existing extension package: $PackagePath"
    }

    Push-Location $extensionRoot
    try {
        & $wxtCommand 'build' '-b' 'firefox'
        if ($LASTEXITCODE -ne 0) {
            throw "Current Firefox extension build failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath (Join-Path $buildRoot 'manifest.json') -PathType Leaf)) {
        throw "Current Firefox extension build did not produce a root manifest: $buildRoot"
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory(
        $buildRoot,
        $PackagePath,
        [IO.Compression.CompressionLevel]::Optimal,
        $false
    )
    $extension = Read-ExtensionPackage -Path $PackagePath
    Assert-CurrentExtensionPackage -Extension $extension
    return $extension
}

function Test-Prerequisites {
    param([Parameter(Mandatory = $true)] $Fixture)
    $failures = [System.Collections.Generic.List[string]]::new()
    $node = Get-Command 'node.exe' -ErrorAction SilentlyContinue
    $smi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $node) { $failures.Add('node.exe is missing') }
    if ($null -eq $smi) { $failures.Add('nvidia-smi.exe is missing') }
    if (-not (Test-Path -LiteralPath $playwrightModule -PathType Container)) {
        $failures.Add("Packaged Playwright is missing: $playwrightModule")
    }
    if (-not (Test-Path -LiteralPath $driverPath -PathType Leaf)) {
        $failures.Add("Firefox benchmark driver is missing: $driverPath")
    }

    $registryPath = "HKCU:\Software\Mozilla\NativeMessagingHosts\$expectedNativeHost"
    $nativeHostExecutable = if ([string]::IsNullOrWhiteSpace($NativeHostExecutable)) {
        Join-Path $repositoryRoot 'target\release\hsk-manga-native-host.exe'
    } else {
        [IO.Path]::GetFullPath($NativeHostExecutable)
    }
    $daemonExecutable = Join-Path (Split-Path -Parent $nativeHostExecutable) 'hsk-manga-browser-daemon.exe'
    $resolvedBuildAttestation = if ([string]::IsNullOrWhiteSpace($BuildAttestationPath)) {
        Join-Path (Split-Path -Parent $nativeHostExecutable) 'hskify-performance-build-attestation.json'
    } else {
        [IO.Path]::GetFullPath($BuildAttestationPath)
    }
    if (-not (Test-Path -LiteralPath $nativeHostExecutable -PathType Leaf)) {
        $failures.Add("Release native-host executable is missing: $nativeHostExecutable")
    }
    if (-not (Test-Path -LiteralPath $daemonExecutable -PathType Leaf)) {
        $failures.Add("Sibling release daemon is missing: $daemonExecutable")
    }
    if (
        $nativeHostExecutable -match '(?i)\\target\\debug\\' -or
        $daemonExecutable -match '(?i)\\target\\debug\\'
    ) {
        $failures.Add('Debug binaries are forbidden; select the release native-host executable')
    }
    $buildAttestation = $null
    if (-not (Test-Path -LiteralPath $resolvedBuildAttestation -PathType Leaf)) {
        $failures.Add("Performance-build attestation is missing: $resolvedBuildAttestation")
    }
    elseif (
        (Test-Path -LiteralPath $nativeHostExecutable -PathType Leaf) -and
        (Test-Path -LiteralPath $daemonExecutable -PathType Leaf)
    ) {
        try {
            $buildAttestation = Assert-HskifyPerformanceBuildAttestation `
                -AttestationPath $resolvedBuildAttestation `
                -NativeHostPath $nativeHostExecutable `
                -BrowserDaemonPath $daemonExecutable `
                -VerifyCurrentHardware `
                -VerifyCurrentToolchain
        }
        catch {
            $failures.Add("Performance-build attestation verification failed: $($_.Exception.Message)")
        }
    }

    $resolvedExtensionPackage = ''
    $extension = $null
    $extensionPlan = 'build-current-source'
    if (-not [string]::IsNullOrWhiteSpace($ExtensionPackagePath)) {
        $resolvedExtensionPackage = [IO.Path]::GetFullPath($ExtensionPackagePath)
        $extensionPlan = 'use-explicit-package'
        if (-not (Test-Path -LiteralPath $resolvedExtensionPackage -PathType Leaf)) {
            $failures.Add("Explicit Firefox extension package is missing: $resolvedExtensionPackage")
        } else {
            try {
                $extension = Read-ExtensionPackage -Path $resolvedExtensionPackage
                Assert-CurrentExtensionPackage -Extension $extension
            }
            catch {
                $failures.Add("Packaged extension is unreadable or incompatible: $($_.Exception.Message)")
            }
        }
    } else {
        $extensionRoot = Join-Path $repositoryRoot 'extensions\firefox'
        $wxtCommand = Join-Path $repositoryRoot 'node_modules\.bin\wxt.cmd'
        $configPath = Join-Path $extensionRoot 'wxt.config.ts'
        $contractPath = Join-Path $extensionRoot 'src\contracts\browser.ts'
        foreach ($required in @($wxtCommand, $configPath, $contractPath)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
                $failures.Add("Current-extension packaging prerequisite is missing: $required")
            }
        }
        if (Test-Path -LiteralPath $configPath -PathType Leaf) {
            $configText = Get-Content -LiteralPath $configPath -Raw -Encoding utf8
            if (-not $configText.Contains($expectedExtensionId)) {
                $failures.Add("Current Firefox source does not declare extension ID $expectedExtensionId")
            }
        }
        if (Test-Path -LiteralPath $contractPath -PathType Leaf) {
            $contractText = Get-Content -LiteralPath $contractPath -Raw -Encoding utf8
            if (-not $contractText.Contains($expectedBuildFingerprint)) {
                $failures.Add("Current Firefox source does not declare build fingerprint $expectedBuildFingerprint")
            }
        }
    }

    $resolvedFirefox = $FirefoxExecutable
    if ([string]::IsNullOrWhiteSpace($resolvedFirefox) -and $null -ne $node) {
        try { $resolvedFirefox = Resolve-DefaultFirefox -NodeCommand $node } catch {}
    }
    if (
        [string]::IsNullOrWhiteSpace($resolvedFirefox) -or
        -not (Test-Path -LiteralPath $resolvedFirefox -PathType Leaf)
    ) {
        $failures.Add('Playwright-compatible Firefox is missing; pass -FirefoxExecutable or install the repository Playwright browser')
    } else {
        $resolvedFirefox = [IO.Path]::GetFullPath($resolvedFirefox)
    }

    $resolvedResources = if ([string]::IsNullOrWhiteSpace($ResourcesDirectory)) {
        ''
    } else {
        [IO.Path]::GetFullPath($ResourcesDirectory)
    }
    if (
        -not [string]::IsNullOrWhiteSpace($resolvedResources) -and
        -not (Test-Path -LiteralPath $resolvedResources -PathType Container)
    ) {
        $failures.Add("Explicit resource root is missing: $resolvedResources")
    }

    $resolvedHsk = if (-not [string]::IsNullOrWhiteSpace($HskResourcePath)) {
        [IO.Path]::GetFullPath($HskResourcePath)
    } elseif (-not [string]::IsNullOrWhiteSpace($resolvedResources)) {
        Join-Path $resolvedResources 'hsk-2.0.normalized.json'
    } else {
        ''
    }
    $resolvedDictionary = if (-not [string]::IsNullOrWhiteSpace($DictionaryResourcePath)) {
        [IO.Path]::GetFullPath($DictionaryResourcePath)
    } elseif (-not [string]::IsNullOrWhiteSpace($resolvedResources)) {
        Join-Path $resolvedResources 'cc-cedict.normalized.json'
    } else {
        ''
    }
    $resolvedModel = if (-not [string]::IsNullOrWhiteSpace($QwenModelPath)) {
        [IO.Path]::GetFullPath($QwenModelPath)
    } elseif (-not [string]::IsNullOrWhiteSpace($resolvedResources)) {
        Join-Path $resolvedResources 'models\Qwen3.5-4B-Q4_K_M.gguf'
    } else {
        ''
    }
    $resolvedFonts = if (-not [string]::IsNullOrWhiteSpace($FontsDirectory)) {
        [IO.Path]::GetFullPath($FontsDirectory)
    } elseif (-not [string]::IsNullOrWhiteSpace($resolvedResources)) {
        Join-Path $resolvedResources 'fonts'
    } else {
        ''
    }

    $resourceRequests = @(
        [ordered]@{
            label = 'HSK resource'
            path = $resolvedHsk
            hint = 'pass -ResourcesDirectory or -HskResourcePath'
        },
        [ordered]@{
            label = 'CC-CEDICT resource'
            path = $resolvedDictionary
            hint = 'pass -ResourcesDirectory or -DictionaryResourcePath'
        },
        [ordered]@{
            label = 'Qwen model'
            path = $resolvedModel
            hint = 'pass -ResourcesDirectory or -QwenModelPath'
        }
    )
    foreach ($request in $resourceRequests) {
        if ([string]::IsNullOrWhiteSpace($request.path)) {
            $failures.Add("$($request.label) is unresolved; $($request.hint)")
        } elseif (-not (Test-Path -LiteralPath $request.path -PathType Leaf)) {
            $failures.Add("$($request.label) is missing: $($request.path)")
        }
    }
    if ([string]::IsNullOrWhiteSpace($resolvedFonts)) {
        $failures.Add('CJK font directory is unresolved; pass -ResourcesDirectory or -FontsDirectory')
    } elseif (-not (Test-Path -LiteralPath $resolvedFonts -PathType Container)) {
        $failures.Add("CJK font directory is missing: $resolvedFonts")
    } else {
        foreach ($font in @('NotoSansSC-VF.ttf', 'NotoSerifSC-VF.ttf')) {
            $fontPath = Join-Path $resolvedFonts $font
            if (-not (Test-Path -LiteralPath $fontPath -PathType Leaf)) {
                $failures.Add("CJK font resource is missing: $fontPath")
            }
        }
    }

    $resolvedRuntime = if ([string]::IsNullOrWhiteSpace($resolvedResources)) {
        ''
    } else {
        Join-Path $resolvedResources 'runtime'
    }
    $expectedRuntimeFiles = @(
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
    $runtimeFilePaths = @()
    if ([string]::IsNullOrWhiteSpace($resolvedRuntime)) {
        $failures.Add('CUDA/llama runtime root is unresolved; pass -ResourcesDirectory')
    }
    elseif (-not (Test-Path -LiteralPath $resolvedRuntime -PathType Container)) {
        $failures.Add("CUDA/llama runtime root is missing: $resolvedRuntime")
    }
    else {
        $actualRuntimeFiles = @(
            Get-ChildItem -LiteralPath $resolvedRuntime -Recurse -File |
                ForEach-Object { $_.FullName.Substring($resolvedRuntime.Length + 1) } |
                Sort-Object
        )
        if (
            ($actualRuntimeFiles -join "`n") -cne
            (@($expectedRuntimeFiles | Sort-Object) -join "`n")
        ) {
            $failures.Add('CUDA/llama runtime does not contain exactly the pinned file set')
        }
        else {
            $runtimeFilePaths = @(
                $expectedRuntimeFiles | ForEach-Object { Join-Path $resolvedRuntime $_ }
            )
            $cudaMarker = (Get-Content -LiteralPath (Join-Path $resolvedRuntime 'cuda\.installed') -Raw).Trim()
            $llamaMarker = (Get-Content -LiteralPath (Join-Path $resolvedRuntime 'llama.cpp\b8935\windows-cuda13-x64\.installed') -Raw).Trim()
            if ($cudaMarker -cne 'cuda;platform=win_amd64;wheels=nvidia-cuda-runtime/13.1.80,nvidia-cuda-nvrtc/13.1.80,nvidia-cublas/13.2.0.9,nvidia-cufft/12.1.0.31,nvidia-curand/10.4.1.81,nvidia-cudnn-cu13/9.17.0.29;extract=5') {
                $failures.Add('CUDA runtime install marker is not the pinned CUDA 13.1 identity')
            }
            if ($llamaMarker -cne 'llama-b8935-windows-cuda13-x64-extract-2') {
                $failures.Add('llama.cpp runtime install marker is not the pinned b8935 CUDA identity')
            }
        }
    }

    $translationModel = $null
    $resourceIdentities = @()
    $residentModelPaths = @()
    if (Test-Path -LiteralPath $modelManifestPath -PathType Leaf) {
        $modelManifest = Get-Content -LiteralPath $modelManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
        $expectedManifestProperties = @(
            'generatedAt',
            'manifestVersion',
            'resourceIdentities',
            'translationModelId'
        )
        $actualManifestProperties = @($modelManifest.PSObject.Properties.Name | Sort-Object)
        if (
            ($actualManifestProperties -join "`n") -cne
            ($expectedManifestProperties -join "`n")
        ) {
            $failures.Add('Pinned model manifest has unexpected fields')
        }
        if ($modelManifest.manifestVersion -ne 1) {
            $failures.Add('Pinned model manifest is not version 1')
        }
        if ([string] $modelManifest.translationModelId -cne 'qwen3.5-4b') {
            $failures.Add('Pinned model manifest does not require qwen3.5-4b')
        }
        $manifestResourceIdentities = @($modelManifest.resourceIdentities)
        $requiredIdentityIds = @(
            'comic-text-bubble-detector-config',
            'comic-text-bubble-detector-preprocessor-config',
            'comic-text-bubble-detector-weights',
            'pp-ocr-v5-english-recognizer-config',
            'pp-ocr-v5-english-recognizer-model',
            'translation-model'
        )
        if ($manifestResourceIdentities.Count -ne $requiredIdentityIds.Count) {
            $failures.Add('Pinned model manifest must contain exactly six resource identities')
        }
        else {
            $identityIds = [System.Collections.Generic.HashSet[string]]::new(
                [StringComparer]::Ordinal
            )
            for ($index = 0; $index -lt $manifestResourceIdentities.Count; $index++) {
                $identity = $manifestResourceIdentities[$index]
                if ($null -eq $identity) {
                    $failures.Add('Pinned model manifest has a null resource identity')
                    continue
                }
                $identityProperties = @($identity.PSObject.Properties.Name | Sort-Object)
                $expectedIdentityProperties = @(
                    'bytes',
                    'filename',
                    'id',
                    'repository',
                    'repositoryRevision',
                    'sha256',
                    'url'
                )
                $expectedUrl = "https://huggingface.co/$($identity.repository)/resolve/$($identity.repositoryRevision)/$($identity.filename)"
                if (
                    ($identityProperties -join "`n") -cne
                    ($expectedIdentityProperties -join "`n") -or
                    [string] $identity.id -cne $requiredIdentityIds[$index] -or
                    [string]::IsNullOrWhiteSpace([string] $identity.repository) -or
                    [string]::IsNullOrWhiteSpace([string] $identity.filename) -or
                    [string] $identity.repositoryRevision -notmatch '^[0-9a-f]{40}$' -or
                    [string] $identity.sha256 -notmatch '^[0-9a-f]{64}$' -or
                    [string] $identity.url -cne $expectedUrl -or
                    [int64] $identity.bytes -le 0 -or
                    -not $identityIds.Add([string] $identity.id)
                ) {
                    $failures.Add("Pinned model manifest has an invalid resource identity: $($identity.id)")
                }
                $resourceIdentities += [ordered]@{
                    id = [string] $identity.id
                    repository = [string] $identity.repository
                    repositoryRevision = [string] $identity.repositoryRevision
                    filename = [string] $identity.filename
                    bytes = [int64] $identity.bytes
                    sha256 = [string] $identity.sha256
                }
                if ([string] $identity.id -cne 'translation-model') {
                    $residentPath = if ([string]::IsNullOrWhiteSpace($resolvedResources)) {
                        ''
                    }
                    else {
                        Join-Path $resolvedResources "models\resident\$($identity.id)\$($identity.filename)"
                    }
                    if ([string]::IsNullOrWhiteSpace($residentPath)) {
                        $failures.Add("Resident detector/OCR resource root is unresolved for $($identity.id); pass -ResourcesDirectory")
                    }
                    elseif (-not (Test-Path -LiteralPath $residentPath -PathType Leaf)) {
                        $failures.Add("Resident detector/OCR resource is missing: $residentPath")
                    }
                    else {
                        $actualBytes = (Get-Item -LiteralPath $residentPath).Length
                        $actualSha256 = (Get-FileHash -LiteralPath $residentPath -Algorithm SHA256).Hash.ToLowerInvariant()
                        if (
                            $actualBytes -ne [int64] $identity.bytes -or
                            $actualSha256 -cne [string] $identity.sha256
                        ) {
                            $failures.Add("Resident detector/OCR resource identity mismatch: $($identity.id)")
                        }
                        else {
                            $residentModelPaths += [ordered]@{
                                id = [string] $identity.id
                                filename = [string] $identity.filename
                                path = $residentPath
                            }
                        }
                    }
                }
            }
            $translationModels = @(
                $manifestResourceIdentities |
                    Where-Object { [string] $_.id -ceq 'translation-model' }
            )
            if ($translationModels.Count -eq 1) {
                $translationModel = $translationModels[0]
            }
            else {
                $failures.Add('Pinned model manifest does not contain exactly one translation model')
            }
            if (
                $null -ne $translationModel -and
                [string] $translationModel.filename -cne 'Qwen3.5-4B-Q4_K_M.gguf'
            ) {
                $failures.Add('Pinned translation model is not Qwen3.5-4B-Q4_K_M.gguf')
            }
            if (
                $null -ne $translationModel -and
                -not [string]::IsNullOrWhiteSpace($resolvedModel) -and
                (Test-Path -LiteralPath $resolvedModel -PathType Leaf) -and
                (Get-Item -LiteralPath $resolvedModel).Length -ne [int64] $translationModel.bytes
            ) {
                $failures.Add("Qwen model size mismatch: $resolvedModel")
            }
        }
    } else {
        $failures.Add("Pinned model manifest is missing: $modelManifestPath")
    }

    $gpu = $null
    if ($null -ne $smi) {
        try {
            $gpu = Get-HskifyPerformanceGpuIdentity
            Assert-HskifyExactPerformanceGpu -Gpu $gpu
        }
        catch {
            $failures.Add($_.Exception.Message)
        }
    }

    $otherDaemons = @(
        Get-CimInstance Win32_Process -Filter "Name='hsk-manga-browser-daemon.exe'" -ErrorAction SilentlyContinue
    )
    if ($otherDaemons.Count -gt 0) {
        $failures.Add("A browser daemon is already running; stop it before isolated measurement (PIDs: $($otherDaemons.ProcessId -join ', '))")
    }

    return [ordered]@{
        ready = $failures.Count -eq 0
        failures = @($failures)
        node = if ($null -eq $node) { '' } else { $node.Source }
        nvidiaSmi = if ($null -eq $smi) { '' } else { $smi.Source }
        firefoxExecutable = $resolvedFirefox
        nativeRegistryPath = $registryPath
        nativeHostExecutable = $nativeHostExecutable
        daemonExecutable = $daemonExecutable
        buildAttestationPath = $resolvedBuildAttestation
        buildAttestation = $buildAttestation
        extensionPackagePath = $resolvedExtensionPackage
        extension = $extension
        extensionVersion = if ($null -eq $extension) { '' } else { [string] $extension.manifest.version }
        extensionPlan = $extensionPlan
        resourcesDirectory = $resolvedResources
        resourcePaths = [ordered]@{
            hsk = $resolvedHsk
            dictionary = $resolvedDictionary
            qwenModel = $resolvedModel
            fontsDirectory = $resolvedFonts
            sansFont = if ([string]::IsNullOrWhiteSpace($resolvedFonts)) { '' } else { Join-Path $resolvedFonts 'NotoSansSC-VF.ttf' }
            serifFont = if ([string]::IsNullOrWhiteSpace($resolvedFonts)) { '' } else { Join-Path $resolvedFonts 'NotoSerifSC-VF.ttf' }
            residentModels = @($residentModelPaths)
            runtimeRoot = $resolvedRuntime
            runtimeFiles = @($runtimeFilePaths)
        }
        translationModel = $translationModel
        resourceIdentities = $resourceIdentities
        gpu = $gpu
        fixture = $Fixture
    }
}

function Register-TemporaryNativeHost {
    param(
        [Parameter(Mandatory = $true)][string] $ManifestPath,
        [Parameter(Mandatory = $true)][string] $NativeHostPath
    )
    $manifest = [ordered]@{
        name = $expectedNativeHost
        description = 'Temporary Hskify chapter 5 release benchmark native host'
        path = [IO.Path]::GetFullPath($NativeHostPath)
        type = 'stdio'
        allowed_extensions = @($expectedExtensionId)
    }
    [IO.File]::WriteAllText(
        $ManifestPath,
        ($manifest | ConvertTo-Json -Depth 4) + "`n",
        $utf8NoBom
    )

    $registryParentPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts'
    $registryPath = "$registryParentPath\$expectedNativeHost"
    $parentExisted = Test-Path -LiteralPath $registryParentPath
    if (-not $parentExisted) {
        New-Item -Path $registryParentPath -Force | Out-Null
    }
    $keyExisted = Test-Path -LiteralPath $registryPath
    if (-not $keyExisted) {
        New-Item -Path $registryPath -Force | Out-Null
    }
    # Registry-provider Get-Item returns a read-only RegistryKey for an
    # existing key. Open the exact HKCU subkey writable so an installed
    # Hskify registration can be replaced temporarily and restored.
    $registrySubkey = 'Software\Mozilla\NativeMessagingHosts\' + $expectedNativeHost
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($registrySubkey, $true)
    if ($null -eq $key) {
        throw "Could not open the native-host registry key writable: $registryPath"
    }
    $registrationError = $null
    $previousDefaultExists = $false
    $previousDefaultValue = $null
    try {
        $valueNames = @($key.GetValueNames())
        $previousDefaultExists = $valueNames -contains ''
        $previousDefaultValue = if ($previousDefaultExists) {
            $key.GetValue('', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        } else {
            $null
        }
        $key.SetValue('', $ManifestPath, [Microsoft.Win32.RegistryValueKind]::String)
    }
    catch {
        $registrationError = $_
        if ($keyExisted) {
            if ($previousDefaultExists) {
                $key.SetValue(
                    '',
                    $previousDefaultValue,
                    [Microsoft.Win32.RegistryValueKind]::String
                )
            } else {
                $key.DeleteValue('', $false)
            }
        }
    }
    finally {
        $key.Dispose()
    }
    if ($null -ne $registrationError) {
        if (-not $keyExisted -and (Test-Path -LiteralPath $registryPath)) {
            Remove-Item -LiteralPath $registryPath
        }
        if (
            -not $parentExisted -and
            (Test-Path -LiteralPath $registryParentPath) -and
            @(Get-ChildItem -LiteralPath $registryParentPath).Count -eq 0
        ) {
            Remove-Item -LiteralPath $registryParentPath
        }
        throw $registrationError
    }
    return [ordered]@{
        registryPath = $registryPath
        registryParentPath = $registryParentPath
        parentExisted = $parentExisted
        keyExisted = $keyExisted
        previousDefaultExists = $previousDefaultExists
        previousDefaultValue = $previousDefaultValue
        temporaryDefaultValue = $ManifestPath
    }
}

function Restore-TemporaryNativeHost {
    param([Parameter(Mandatory = $true)] $Registration)
    if (-not (Test-Path -LiteralPath $Registration.registryPath)) {
        if ($Registration.keyExisted) {
            throw 'Temporary native-host key disappeared before its previous value could be restored'
        }
        if (
            -not $Registration.parentExisted -and
            (Test-Path -LiteralPath $Registration.registryParentPath) -and
            @(Get-ChildItem -LiteralPath $Registration.registryParentPath).Count -eq 0
        ) {
            Remove-Item -LiteralPath $Registration.registryParentPath
        }
        return
    }
    $registrySubkey = 'Software\Mozilla\NativeMessagingHosts\' + $expectedNativeHost
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($registrySubkey, $true)
    if ($null -eq $key) {
        throw "Could not open the native-host registry key writable: $($Registration.registryPath)"
    }
    try {
        $current = $key.GetValue('', $null, [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        if (-not [string]::Equals(
            [string] $current,
            [string] $Registration.temporaryDefaultValue,
            [StringComparison]::Ordinal
        )) {
            throw 'Exact native-host registration changed during the benchmark; refusing to overwrite that external change'
        }
        if ($Registration.keyExisted) {
            if ($Registration.previousDefaultExists) {
                $key.SetValue(
                    '',
                    $Registration.previousDefaultValue,
                    [Microsoft.Win32.RegistryValueKind]::String
                )
            } else {
                $key.DeleteValue('', $false)
            }
        }
    }
    finally {
        $key.Dispose()
    }
    if (-not $Registration.keyExisted) {
        Remove-Item -LiteralPath $Registration.registryPath
    }
    if (
        -not $Registration.parentExisted -and
        (Test-Path -LiteralPath $Registration.registryParentPath) -and
        @(Get-ChildItem -LiteralPath $Registration.registryParentPath).Count -eq 0
    ) {
        Remove-Item -LiteralPath $Registration.registryParentPath
    }
}

function Stop-IsolatedDaemon {
    param(
        [Parameter(Mandatory = $true)][string] $StateDirectory,
        [Parameter(Mandatory = $true)][string] $DaemonExecutable
    )
    $recordPath = Join-Path $StateDirectory 'daemon-state.json'
    if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) {
        return @()
    }
    $record = Get-Content -LiteralPath $recordPath -Raw -Encoding utf8 | ConvertFrom-Json
    if ([int] $record.pid -le 0) {
        throw "Isolated daemon state contains an invalid PID: $recordPath"
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$([int] $record.pid)" -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return @()
    }
    $expected = [IO.Path]::GetFullPath($DaemonExecutable)
    if (
        [string]::IsNullOrWhiteSpace([string] $process.ExecutablePath) -or
        -not [string]::Equals(
            [IO.Path]::GetFullPath([string] $process.ExecutablePath),
            $expected,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Refusing to stop PID $($record.pid); it is not the isolated benchmark daemon $expected"
    }
    $stopped = [ordered]@{
        pid = [int] $process.ProcessId
        executablePath = [string] $process.ExecutablePath
        creationDate = [string] $process.CreationDate
        instanceId = [string] $record.instanceId
    }
    Stop-Process -Id ([int] $record.pid) -Force
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds(10)
    do {
        if ($null -eq (Get-Process -Id ([int] $record.pid) -ErrorAction SilentlyContinue)) {
            return @($stopped)
        }
        Start-Sleep -Milliseconds 100
    } while ([DateTimeOffset]::UtcNow -lt $deadline)
    throw "Could not stop isolated benchmark daemon PID $($record.pid)"
}

function Stop-IsolatedFirefox {
    param(
        [Parameter(Mandatory = $true)][string] $ProfileDirectory,
        [Parameter(Mandatory = $true)][string] $FirefoxExecutable
    )
    $expectedExecutable = [IO.Path]::GetFullPath($FirefoxExecutable)
    $profileToken = [IO.Path]::GetFullPath($ProfileDirectory)
    $matches = @(
        Get-CimInstance Win32_Process -Filter "Name='firefox.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                -not [string]::IsNullOrWhiteSpace([string] $_.ExecutablePath) -and
                [string]::Equals(
                    [IO.Path]::GetFullPath([string] $_.ExecutablePath),
                    $expectedExecutable,
                    [StringComparison]::OrdinalIgnoreCase
                ) -and
                ([string] $_.CommandLine).IndexOf(
                    $profileToken,
                    [StringComparison]::OrdinalIgnoreCase
                ) -ge 0
            }
    )
    $stopped = @(
        foreach ($process in $matches) {
            [ordered]@{
                pid = [int] $process.ProcessId
                executablePath = [string] $process.ExecutablePath
                creationDate = [string] $process.CreationDate
            }
            Stop-Process -Id ([int] $process.ProcessId) -Force
        }
    )
    return $stopped
}

function Read-ActiveMarker {
    param([Parameter(Mandatory = $true)][string] $Path)
    try {
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
        }
    }
    catch {}
    return $null
}

function Read-RedactedDaemonState {
    param([Parameter(Mandatory = $true)][string] $Path)
    try {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
        $state = Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
        return [ordered]@{
            instanceId = $state.instanceId
            pid = $state.pid
            port = $state.port
            startedAtUnixMs = $state.startedAtUnixMs
            controlSecretRedacted = $true
        }
    }
    catch {
        return $null
    }
}

function Get-GpuTelemetry {
    param([Parameter(Mandatory = $true)][string] $SmiPath)
    $deviceRows = @(& $SmiPath '--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,clocks.mem' '--format=csv,noheader,nounits' 2>$null)
    $computeRows = @(& $SmiPath '--query-compute-apps=pid,process_name,used_gpu_memory' '--format=csv,noheader,nounits' 2>$null)
    $devices = @(
        foreach ($row in $deviceRows) {
            if ([string]::IsNullOrWhiteSpace($row)) { continue }
            $parts = @($row.Split(',') | ForEach-Object { $_.Trim() })
            if ($parts.Count -eq 9) {
                [ordered]@{
                    name = $parts[0]
                    memoryTotalMiB = [double] $parts[1]
                    memoryUsedMiB = [double] $parts[2]
                    gpuUtilizationPercent = [double] $parts[3]
                    memoryUtilizationPercent = [double] $parts[4]
                    powerDrawW = [double] $parts[5]
                    temperatureC = [double] $parts[6]
                    smClockMHz = [double] $parts[7]
                    memoryClockMHz = [double] $parts[8]
                }
            }
        }
    )
    $compute = @(
        foreach ($row in $computeRows) {
            if ([string]::IsNullOrWhiteSpace($row)) { continue }
            $parts = @($row.Split(',') | ForEach-Object { $_.Trim() })
            if (
                $parts.Count -eq 3 -and
                $parts[0] -match '^\d+$' -and
                $parts[2] -match '^\d+(?:\.\d+)?$'
            ) {
                [ordered]@{
                    pid = [int] $parts[0]
                    processName = $parts[1]
                    usedMemoryMiB = [double] $parts[2]
                }
            }
        }
    )
    return [ordered]@{ devices = $devices; computeApps = $compute }
}

function Get-SystemPagingTelemetry {
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
    if ($null -eq $memory) {
        return [ordered]@{
            available = $false
            reason = 'Win32_PerfFormattedData_PerfOS_Memory was unavailable'
        }
    }
    return [ordered]@{
        available = $true
        pagesOutputPerSec = [uint64] $memory.PagesOutputPersec
        pageWritesPerSec = [uint64] $memory.PageWritesPersec
    }
}

function Get-ProcessTelemetry {
    param(
        [Parameter(Mandatory = $true)][int] $DriverPid,
        [AllowNull()] $DaemonState
    )
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $ids = [System.Collections.Generic.HashSet[int]]::new()
    [void] $ids.Add($DriverPid)
    if ($null -ne $DaemonState) { [void] $ids.Add([int] $DaemonState.pid) }
    do {
        $added = $false
        foreach ($process in $all) {
            if ($ids.Contains([int] $process.ParentProcessId) -and $ids.Add([int] $process.ProcessId)) {
                $added = $true
            }
        }
    } while ($added)
    return @(
        foreach ($process in $all) {
            if (-not $ids.Contains([int] $process.ProcessId)) { continue }
            [ordered]@{
                pid = [int] $process.ProcessId
                parentPid = [int] $process.ParentProcessId
                name = [string] $process.Name
                executablePath = [string] $process.ExecutablePath
                creationDate = [string] $process.CreationDate
                kernelMode100ns = [uint64] $process.KernelModeTime
                userMode100ns = [uint64] $process.UserModeTime
                workingSetBytes = [uint64] $process.WorkingSetSize
                peakWorkingSetBytes = [uint64] $process.PeakWorkingSetSize
                privateBytes = [uint64] $process.PrivatePageCount
                pageFileUsageKiB = [uint64] $process.PageFileUsage
                peakPageFileUsageKiB = [uint64] $process.PeakPageFileUsage
                readOperations = [uint64] $process.ReadOperationCount
                writeOperations = [uint64] $process.WriteOperationCount
                readTransferBytes = [uint64] $process.ReadTransferCount
                writeTransferBytes = [uint64] $process.WriteTransferCount
            }
        }
    )
}

function Measure-Driver {
    param(
        [Parameter(Mandatory = $true)] $Process,
        [Parameter(Mandatory = $true)][string] $OutputRoot,
        [Parameter(Mandatory = $true)][string] $StateDirectory,
        [Parameter(Mandatory = $true)][string] $SmiPath
    )
    $telemetryPath = Join-Path $OutputRoot 'telemetry.raw.jsonl'
    $filesystemPath = Join-Path $OutputRoot 'filesystem-events.raw.jsonl'
    $telemetry = [System.Collections.Generic.List[object]]::new()
    $filesystemEvents = [System.Collections.Generic.List[object]]::new()
    $watcher = $null
    $subscriptions = @()
    try {
        [IO.Directory]::CreateDirectory($StateDirectory) | Out-Null
        $watcher = New-Object IO.FileSystemWatcher($StateDirectory)
        $watcher.IncludeSubdirectories = $true
        $watcher.NotifyFilter = [IO.NotifyFilters]'FileName, DirectoryName, LastWrite, Size'
        foreach ($eventName in @('Changed', 'Created', 'Deleted', 'Renamed')) {
            $source = "HskifyBenchmark.$([Guid]::NewGuid().ToString('N')).$eventName"
            $subscriptions += Register-ObjectEvent -InputObject $watcher -EventName $eventName -SourceIdentifier $source
        }
        $watcher.EnableRaisingEvents = $true
        do {
            $sampledAt = [DateTimeOffset]::UtcNow
            $daemonState = Read-RedactedDaemonState -Path (Join-Path $StateDirectory 'daemon-state.json')
            $sample = [ordered]@{
                sampledAtUtc = $sampledAt.ToString('o')
                sampledAtEpochMs = $sampledAt.ToUnixTimeMilliseconds()
                daemonState = $daemonState
                processes = @(Get-ProcessTelemetry -DriverPid $Process.Id -DaemonState $daemonState)
                gpu = Get-GpuTelemetry -SmiPath $SmiPath
                systemPaging = Get-SystemPagingTelemetry
            }
            $telemetry.Add($sample)

            foreach ($event in @(Get-Event | Where-Object { $_.SourceIdentifier -like 'HskifyBenchmark.*' })) {
                $args = $event.SourceEventArgs
                $entry = [ordered]@{
                    observedAtUtc = ([DateTimeOffset] $event.TimeGenerated).ToUniversalTime().ToString('o')
                    observedAtEpochMs = ([DateTimeOffset] $event.TimeGenerated).ToUnixTimeMilliseconds()
                    changeType = [string] $args.ChangeType
                    fullPath = [string] $args.FullPath
                    oldFullPath = if ($args -is [IO.RenamedEventArgs]) { [string] $args.OldFullPath } else { '' }
                }
                $filesystemEvents.Add($entry)
                Remove-Event -EventIdentifier $event.EventIdentifier
            }
            if (-not $Process.HasExited) {
                Start-Sleep -Milliseconds $TelemetryIntervalMs
                $Process.Refresh()
            }
        } while (-not $Process.HasExited)
    }
    finally {
        if ($null -ne $watcher) { $watcher.EnableRaisingEvents = $false }
        foreach ($event in @(Get-Event | Where-Object { $_.SourceIdentifier -like 'HskifyBenchmark.*' })) {
            $args = $event.SourceEventArgs
            $filesystemEvents.Add([ordered]@{
                observedAtUtc = ([DateTimeOffset] $event.TimeGenerated).ToUniversalTime().ToString('o')
                observedAtEpochMs = ([DateTimeOffset] $event.TimeGenerated).ToUnixTimeMilliseconds()
                changeType = [string] $args.ChangeType
                fullPath = [string] $args.FullPath
                oldFullPath = if ($args -is [IO.RenamedEventArgs]) { [string] $args.OldFullPath } else { '' }
            })
            Remove-Event -EventIdentifier $event.EventIdentifier
        }
        foreach ($subscription in $subscriptions) {
            Unregister-Event -SubscriptionId $subscription.Id -ErrorAction SilentlyContinue
        }
        Get-Event | Where-Object { $_.SourceIdentifier -like 'HskifyBenchmark.*' } |
            Remove-Event -ErrorAction SilentlyContinue
        if ($null -ne $watcher) { $watcher.Dispose() }
    }
    $telemetryLines = @($telemetry | ForEach-Object { $_ | ConvertTo-Json -Depth 12 -Compress })
    $filesystemLines = @($filesystemEvents | ForEach-Object { $_ | ConvertTo-Json -Depth 8 -Compress })
    [IO.File]::WriteAllText(
        $telemetryPath,
        $(if ($telemetryLines.Count) { ($telemetryLines -join "`n") + "`n" } else { '' }),
        $utf8NoBom
    )
    [IO.File]::WriteAllText(
        $filesystemPath,
        $(if ($filesystemLines.Count) { ($filesystemLines -join "`n") + "`n" } else { '' }),
        $utf8NoBom
    )
    return [ordered]@{
        telemetryPath = $telemetryPath
        filesystemPath = $filesystemPath
        samplesRetainedInMemoryUntilDriverExit = $telemetry.Count
        filesystemEventsRetainedInMemoryUntilDriverExit = $filesystemEvents.Count
    }
}

function Read-RunSamples {
    param(
        [Parameter(Mandatory = $true)][string] $TelemetryPath,
        [Parameter(Mandatory = $true)] $Run
    )
    $started = [int64] $Run.measuredPhaseStartedAtEpochMs
    $ended = [int64] $Run.measuredPhaseEndedAtEpochMs
    return @(
        Get-Content -LiteralPath $TelemetryPath -Encoding utf8 |
            ForEach-Object { $_ | ConvertFrom-Json } |
            Where-Object {
                [int64] $_.sampledAtEpochMs -ge $started -and
                [int64] $_.sampledAtEpochMs -le $ended
            }
    )
}

function Read-RunFilesystemEvents {
    param(
        [Parameter(Mandatory = $true)][string] $FilesystemPath,
        [Parameter(Mandatory = $true)] $Run
    )
    $started = [int64] $Run.measuredPhaseStartedAtEpochMs
    $ended = [int64] $Run.measuredPhaseEndedAtEpochMs
    return @(
        Get-Content -LiteralPath $FilesystemPath -Encoding utf8 |
            ForEach-Object { $_ | ConvertFrom-Json } |
            Where-Object {
                [int64] $_.observedAtEpochMs -ge $started -and
                [int64] $_.observedAtEpochMs -le $ended
            }
    )
}

function Measure-PagefileWriteSequences {
    param(
        [Parameter(Mandatory = $true)] $Samples,
        [Parameter(Mandatory = $true)][int] $SampleIntervalMs
    )
    $longestMs = [int64] 0
    $sustainedCount = 0
    $positiveCount = 0
    $positiveStartedAt = [int64] -1
    $lastPositiveAt = [int64] -1

    foreach ($sample in @($Samples | Sort-Object sampledAtEpochMs)) {
        $positive = (
            $sample.systemPaging.available -eq $true -and
            [uint64] $sample.systemPaging.pagesOutputPerSec -gt 0
        )
        if ($positive) {
            if ($positiveCount -eq 0) {
                $positiveStartedAt = [int64] $sample.sampledAtEpochMs
            }
            $positiveCount++
            $lastPositiveAt = [int64] $sample.sampledAtEpochMs
            continue
        }

        if ($positiveCount -ge 2) {
            $duration = $lastPositiveAt - $positiveStartedAt
            $longestMs = [Math]::Max($longestMs, $duration)
            if ($duration -ge 1000) {
                $sustainedCount++
            }
        }
        $positiveCount = 0
        $positiveStartedAt = -1
        $lastPositiveAt = -1
    }

    if ($positiveCount -ge 2) {
        $duration = $lastPositiveAt - $positiveStartedAt + $SampleIntervalMs
        $longestMs = [Math]::Max($longestMs, $duration)
        if ($duration -ge 1000) {
            $sustainedCount++
        }
    }

    return [ordered]@{
        longestMs = [int64] $longestMs
        sustainedSequenceCount = [int] $sustainedCount
    }
}

function Get-RunTelemetrySummary {
    param(
        [Parameter(Mandatory = $true)][string] $RunId,
        [Parameter(Mandatory = $true)] $Samples,
        [Parameter(Mandatory = $true)] $FilesystemEvents
    )
    if (@($Samples).Count -eq 0) {
        return [ordered]@{
            measurementAvailable = $false
            missingEvidenceReason = "No telemetry sample timestamp fell inside measured phase $RunId"
            sampleCount = 0
            runtimeCacheFilesystemEventCount = @($FilesystemEvents).Count
        }
    }
    $processes = @($Samples | ForEach-Object { @($_.processes) })
    $daemonProcesses = @($processes | Where-Object { $_.name -eq 'hsk-manga-browser-daemon.exe' })
    $daemonPids = @($daemonProcesses.pid | Sort-Object -Unique)
    $instanceIds = @(
        $Samples |
            Where-Object { $null -ne $_.daemonState } |
            ForEach-Object { $_.daemonState.instanceId } |
            Sort-Object -Unique
    )
    $measuredVramSamples = @(
        foreach ($sample in $Samples) {
            $samplePids = @($sample.processes.pid)
            $apps = @($sample.gpu.computeApps | Where-Object { $samplePids -contains $_.pid })
            if ($apps.Count -eq 0) { continue }
            [ordered]@{
                sampledAtEpochMs = [int64] $sample.sampledAtEpochMs
                usedMemoryMiB = [double] (($apps.usedMemoryMiB | Measure-Object -Sum).Sum)
                processes = @($apps)
            }
        }
    )
    $devices = @($Samples | ForEach-Object { @($_.gpu.devices) })
    $privateSamples = @(
        foreach ($sample in $Samples) {
            [ordered]@{
                sampledAtEpochMs = [int64] $sample.sampledAtEpochMs
                privateBytes = [uint64] (($sample.processes.privateBytes | Measure-Object -Sum).Sum)
            }
        }
    )
    $firstDaemon = @($daemonProcesses | Sort-Object readTransferBytes | Select-Object -First 1)
    $lastDaemon = @($daemonProcesses | Sort-Object readTransferBytes | Select-Object -Last 1)
    $readDelta = if ($firstDaemon.Count -eq 1 -and $lastDaemon.Count -eq 1) {
        [uint64] $lastDaemon[0].readTransferBytes - [uint64] $firstDaemon[0].readTransferBytes
    } else { 0 }
    $writeDelta = if ($firstDaemon.Count -eq 1 -and $lastDaemon.Count -eq 1) {
        [uint64] $lastDaemon[0].writeTransferBytes - [uint64] $firstDaemon[0].writeTransferBytes
    } else { 0 }
    $pagingAvailable = @(
        $Samples | Where-Object {
            $_.systemPaging.PSObject.Properties.Name -contains 'available' -and
            $_.systemPaging.available -eq $true
        }
    )
    $pagingSequences = Measure-PagefileWriteSequences `
        -Samples $Samples `
        -SampleIntervalMs $TelemetryIntervalMs
    $resultCacheEvents = @(
        $FilesystemEvents | Where-Object {
            $_.fullPath -match '(?i)\\browser-cache\\results(?:\\|$)'
        }
    )
    $intermediateEvents = @(
        $FilesystemEvents | Where-Object {
            $_.fullPath -notmatch '(?i)\\browser-cache\\results(?:\\|$)' -and
            $_.fullPath -notmatch '(?i)\\daemon-state\.json$' -and
            $_.fullPath -notmatch '(?i)\\daemon\.lock$' -and
            $_.fullPath -notmatch '(?i)\\\.daemon-state-[^\\]+\.tmp$' -and
            $_.fullPath -notmatch '(?i)\\browser-cache$'
        }
    )
    $controlStateEvents = @(
        $FilesystemEvents | Where-Object {
            $_.fullPath -match '(?i)\\daemon-state\.json$' -or
            $_.fullPath -match '(?i)\\daemon\.lock$' -or
            $_.fullPath -match '(?i)\\\.daemon-state-[^\\]+\.tmp$' -or
            $_.fullPath -match '(?i)\\browser-cache$'
        }
    )
    return [ordered]@{
        measurementAvailable = $true
        sampleCount = @($Samples).Count
        firstSampleUtc = @($Samples)[0].sampledAtUtc
        lastSampleUtc = @($Samples)[-1].sampledAtUtc
        daemonPids = $daemonPids
        daemonInstanceIds = $instanceIds
        peakPrivateBytesAllMeasuredProcesses = [uint64] (($privateSamples.privateBytes | Measure-Object -Maximum).Maximum)
        peakWorkingSetBytesAllMeasuredProcesses = [uint64] (($processes.workingSetBytes | Measure-Object -Maximum).Maximum)
        peakDaemonPrivateBytes = if ($daemonProcesses.Count) { [uint64] (($daemonProcesses.privateBytes | Measure-Object -Maximum).Maximum) } else { 0 }
        peakDaemonWorkingSetBytes = if ($daemonProcesses.Count) { [uint64] (($daemonProcesses.workingSetBytes | Measure-Object -Maximum).Maximum) } else { 0 }
        peakDaemonPageFileUsageKiB = if ($daemonProcesses.Count) { [uint64] (($daemonProcesses.peakPageFileUsageKiB | Measure-Object -Maximum).Maximum) } else { 0 }
        daemonReadTransferDeltaBytes = $readDelta
        daemonWriteTransferDeltaBytes = $writeDelta
        processVramMeasurementAvailable = $measuredVramSamples.Count -gt 0
        processVramSampleCount = $measuredVramSamples.Count
        peakMeasuredProcessVramMiB = if ($measuredVramSamples.Count) {
            [double] (($measuredVramSamples.usedMemoryMiB | Measure-Object -Maximum).Maximum)
        } else { 0 }
        peakGpuMemoryUsedMiB = if ($devices.Count) { [double] (($devices.memoryUsedMiB | Measure-Object -Maximum).Maximum) } else { 0 }
        peakGpuUtilizationPercent = if ($devices.Count) { [double] (($devices.gpuUtilizationPercent | Measure-Object -Maximum).Maximum) } else { 0 }
        peakGpuPowerW = if ($devices.Count) { [double] (($devices.powerDrawW | Measure-Object -Maximum).Maximum) } else { 0 }
        peakGpuTemperatureC = if ($devices.Count) { [double] (($devices.temperatureC | Measure-Object -Maximum).Maximum) } else { 0 }
        runtimeCacheFilesystemEventCount = @($FilesystemEvents).Count
        completedResultCacheFilesystemEventCount = $resultCacheEvents.Count
        daemonControlStateFilesystemEventCount = $controlStateEvents.Count
        synchronousIntermediateWriteEventCount = $intermediateEvents.Count
        synchronousIntermediateWritesObserved = $intermediateEvents.Count -gt 0
        systemPagingMeasurementAvailable = $pagingAvailable.Count -eq @($Samples).Count
        positivePageOutputSampleCount = @(
            $pagingAvailable | Where-Object {
                [uint64] $_.systemPaging.pagesOutputPerSec -gt 0
            }
        ).Count
        longestPagefileWriteSequenceMs = $pagingSequences.longestMs
        sustainedPagefileWriteSequenceCount = $pagingSequences.sustainedSequenceCount
    }
}

function New-MaximumGate {
    param(
        [Parameter(Mandatory = $true)][string] $Id,
        [Parameter(Mandatory = $true)][double] $Actual,
        [Parameter(Mandatory = $true)][double] $Limit,
        [Parameter(Mandatory = $true)][string] $Unit,
        [Parameter(Mandatory = $true)][string] $Scope
    )
    return [ordered]@{
        id = $Id
        status = if ($Actual -le $Limit) { 'pass' } else { 'fail' }
        actual = $Actual
        limit = $Limit
        unit = $Unit
        scope = $Scope
        reason = if ($Actual -le $Limit) { '' } else { "$Actual $Unit exceeds $Limit $Unit" }
    }
}

function New-BooleanEvidenceGate {
    param(
        [Parameter(Mandatory = $true)][string] $Id,
        [Parameter(Mandatory = $true)][bool] $Passed,
        [Parameter(Mandatory = $true)][string] $FailureReason,
        [Parameter(Mandatory = $true)] $Evidence
    )
    return [ordered]@{
        id = $Id
        status = if ($Passed) { 'pass' } else { 'fail' }
        evidence = $Evidence
        reason = if ($Passed) { '' } else { $FailureReason }
    }
}

function Write-CompletedEvidence {
    param(
        [Parameter(Mandatory = $true)][string] $OutputRoot,
        [Parameter(Mandatory = $true)] $Prerequisites,
        [Parameter(Mandatory = $true)] $TelemetryFiles,
        [Parameter(Mandatory = $true)] $Environment
    )
    $index = Get-Content -LiteralPath (Join-Path $OutputRoot 'run-index.json') -Raw -Encoding utf8 |
        ConvertFrom-Json
    $rawRuns = [System.Collections.Generic.List[object]]::new()
    $runs = [System.Collections.Generic.List[object]]::new()
    foreach ($entry in @($index.results | Sort-Object sequence)) {
        $raw = Get-Content -LiteralPath (Join-Path $OutputRoot $entry.rawFile) -Raw -Encoding utf8 |
            ConvertFrom-Json
        $rawRuns.Add($raw)
        $samples = @(Read-RunSamples -TelemetryPath $TelemetryFiles.telemetryPath -Run $raw)
        $events = @(Read-RunFilesystemEvents -FilesystemPath $TelemetryFiles.filesystemPath -Run $raw)
        $telemetry = Get-RunTelemetrySummary `
            -RunId $entry.runId `
            -Samples $samples `
            -FilesystemEvents $events
        $telemetryFile = "$($entry.runId).telemetry.json"
        [IO.File]::WriteAllText(
            (Join-Path $OutputRoot $telemetryFile),
            ([ordered]@{
                runId = $entry.runId
                summary = $telemetry
                samples = $samples
                filesystemEvents = $events
            } | ConvertTo-Json -Depth 20) + "`n",
            $utf8NoBom
        )
        $runs.Add([ordered]@{
            runId = $entry.runId
            kind = $entry.kind
            timing = if ($raw.PSObject.Properties.Name -contains 'timing') {
                $raw.timing
            } else { [ordered]@{} }
            correctness = if ($raw.PSObject.Properties.Name -contains 'correctness') {
                $raw.correctness.totals
            } else { [ordered]@{} }
            cancellationLatencyMs = if (
                $raw.PSObject.Properties.Name -contains 'daemonCancellationLatencyMs'
            ) { $raw.daemonCancellationLatencyMs } else { 0 }
            telemetry = $telemetry
            rawEvidence = $entry.rawFile
            telemetryEvidence = $telemetryFile
        })
    }
    $warm = @($runs | Where-Object { $_.kind -eq 'warm' })
    if ($warm.Count -ne $Iterations) {
        throw "Expected $Iterations complete measured warm runs, found $($warm.Count)"
    }
    foreach ($kind in @('installed-cold', 'warmup', 'cache-replay', 'cancellation')) {
        if (@($runs | Where-Object { $_.kind -eq $kind }).Count -ne 1) {
            throw "Required run evidence is missing or duplicated: $kind"
        }
    }

    $allSamples = @(
        Get-Content -LiteralPath $TelemetryFiles.telemetryPath -Encoding utf8 |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
    $allEvents = @(
        Get-Content -LiteralPath $TelemetryFiles.filesystemPath -Encoding utf8 |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
    $measuredSamples = @(
        foreach ($sample in $allSamples) {
            $inside = @($rawRuns | Where-Object {
                [int64] $sample.sampledAtEpochMs -ge [int64] $_.measuredPhaseStartedAtEpochMs -and
                [int64] $sample.sampledAtEpochMs -le [int64] $_.measuredPhaseEndedAtEpochMs
            }).Count -gt 0
            if ($inside) { $sample }
        }
    )
    $measuredEvents = @(
        foreach ($event in $allEvents) {
            $inside = @($rawRuns | Where-Object {
                [int64] $event.observedAtEpochMs -ge [int64] $_.measuredPhaseStartedAtEpochMs -and
                [int64] $event.observedAtEpochMs -le [int64] $_.measuredPhaseEndedAtEpochMs
            }).Count -gt 0
            if ($inside) { $event }
        }
    )
    $resource = Get-RunTelemetrySummary `
        -RunId 'all-measured-phases' `
        -Samples $measuredSamples `
        -FilesystemEvents $measuredEvents
    $gates = [System.Collections.Generic.List[object]]::new()
    foreach ($raw in $rawRuns) {
        if ($raw.PSObject.Properties.Name -contains 'performanceGates') {
            foreach ($gate in @($raw.performanceGates)) { $gates.Add($gate) }
        }
        if ($raw.PSObject.Properties.Name -contains 'correctness') {
            foreach ($gate in @($raw.correctness.gates)) { $gates.Add($gate) }
        }
        if ($raw.PSObject.Properties.Name -contains 'jobRequests') {
            foreach ($gate in @($raw.jobRequests.gates)) { $gates.Add($gate) }
        }
        if ($raw.PSObject.Properties.Name -contains 'routes') {
            foreach ($gate in @($raw.routes.resourceIdentityEvidence.gates)) {
                $gates.Add($gate)
            }
        }
        if ($raw.PSObject.Properties.Name -contains 'gates') {
            foreach ($gate in @($raw.gates)) { $gates.Add($gate) }
        }
        if ($raw.PSObject.Properties.Name -contains 'readerFeatures') {
            foreach ($gate in @($raw.readerFeatures.gates)) { $gates.Add($gate) }
            foreach ($gate in @($raw.overflow.gates)) { $gates.Add($gate) }
            foreach ($gate in @($raw.sourceReplacement.gates)) { $gates.Add($gate) }
        }
    }
    if (-not $resource.measurementAvailable) {
        foreach ($id in @(
            'peak-private-memory',
            'peak-vram',
            'synchronous-intermediate-writes',
            'sustained-pagefile-writes'
        )) {
            $gates.Add([ordered]@{
                id = $id
                status = 'missing'
                reason = $resource.missingEvidenceReason
            })
        }
    }
    else {
        $gates.Add((New-MaximumGate `
            -Id 'peak-private-memory' `
            -Actual ([double] $resource.peakDaemonPrivateBytes) `
            -Limit ([double] (8GB)) `
            -Unit 'bytes' `
            -Scope 'isolated Hskify daemon private commit; packaged Firefox and benchmark-driver memory are retained as separate diagnostics'))
        if ($resource.processVramMeasurementAvailable) {
            $gates.Add((New-MaximumGate `
                -Id 'peak-vram' `
                -Actual ([double] $resource.peakMeasuredProcessVramMiB) `
                -Limit ([double] (10 * 1024)) `
                -Unit 'MiB' `
                -Scope 'timestamp-aligned sum of nvidia-smi compute-app memory for measured process IDs'))
        }
        else {
            $gates.Add((New-MaximumGate `
                -Id 'peak-vram' `
                -Actual ([double] $resource.peakGpuMemoryUsedMiB) `
                -Limit ([double] (10 * 1024)) `
                -Unit 'MiB' `
                -Scope 'conservative device-wide nvidia-smi memory on Windows WDDM, where NVIDIA does not expose per-process GPU memory'))
        }
        $gates.Add((New-MaximumGate `
            -Id 'synchronous-intermediate-writes' `
            -Actual ([double] $resource.synchronousIntermediateWriteEventCount) `
            -Limit 0 `
            -Unit 'events' `
            -Scope 'isolated daemon-state watcher; completed results and daemon discovery/lock metadata are separate'))
        if ($resource.systemPagingMeasurementAvailable) {
            $gates.Add((New-MaximumGate `
                -Id 'sustained-pagefile-writes' `
                -Actual ([double] $resource.sustainedPagefileWriteSequenceCount) `
                -Limit 0 `
                -Unit 'sequences' `
                -Scope 'system-wide PagesOutputPerSec; sustained means at least 1000 ms of consecutive positive samples'))
        }
        else {
            $gates.Add([ordered]@{
                id = 'sustained-pagefile-writes'
                status = 'missing'
                reason = 'Windows formatted paging counters were unavailable for at least one measured sample'
            })
        }
    }

    $cacheRaw = @($rawRuns | Where-Object { $_.kind -eq 'cache-replay' })[0]
    $warmupRaw = @($rawRuns | Where-Object { $_.kind -eq 'warmup' })[0]
    $sessionDaemonInstanceIds = @(
        $allSamples |
            Where-Object { $null -ne $_.daemonState } |
            ForEach-Object { [string] $_.daemonState.instanceId } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Sort-Object -Unique
    )
    $sameDaemon = $sessionDaemonInstanceIds.Count -eq 1
    $inferenceProgressStages = @(
        'decoding',
        'detecting',
        'ocr',
        'translating',
        'hsk-validating',
        'packaging'
    )
    $exactCacheJobs = @($cacheRaw.routes.jobs | Where-Object {
        $_.terminal.message -eq 'Exact cached translation replayed' -and
        @(
            $_.stageCounts.PSObject.Properties |
                Where-Object { $_.Name -in $inferenceProgressStages }
        ).Count -eq 0
    }).Count
    $expectedCacheJobs = [int] $Prerequisites.fixture.manifest.pageCount
    $noInference = (
        $cacheRaw.exactCache -eq $true -and
        $exactCacheJobs -eq $expectedCacheJobs
    )
    $gates.Add((New-BooleanEvidenceGate `
        -Id 'cache-replay-same-daemon' `
        -Passed $sameDaemon `
        -FailureReason 'Telemetry did not retain exactly one non-empty daemon instance ID for the full driver session' `
        -Evidence ([ordered]@{
            sessionDaemonInstanceIds = $sessionDaemonInstanceIds
            sampleCount = $allSamples.Count
            warmupSequence = $warmupRaw.sequence
            cacheReplaySequence = $cacheRaw.sequence
        })))
    $gates.Add((New-BooleanEvidenceGate `
        -Id 'cache-replay-no-inference-model-load' `
        -Passed ($sameDaemon -and $noInference) `
        -FailureReason 'The unchanged daemon did not prove one exact-cache terminal per chapter image with zero inference progress stages' `
        -Evidence ([ordered]@{
            exactCacheTerminalCount = $exactCacheJobs
            inferenceProgressEventCount = @($cacheRaw.routes.jobs | ForEach-Object {
                @(
                    $_.stageCounts.PSObject.Properties |
                        Where-Object { $_.Name -in $inferenceProgressStages }
                )
            }).Count
            ignoredBookkeepingStages = @('queued')
            causalContract = 'exact-cache terminal is emitted only by the pre-pipeline cache branch'
        })))

    $failedGates = @($gates | Where-Object { $_.status -ne 'pass' })
    $gateEvaluation = [ordered]@{
        benchmarkId = $Prerequisites.fixture.manifest.id
        evaluatedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
        status = if ($failedGates.Count -eq 0) { 'pass' } else { 'fail' }
        gates = @($gates)
        writesDuringMeasuredPhases = [ordered]@{
            driverEvidenceWrites = 0
            powershellTelemetryWrites = 0
            policy = 'both processes retain measurements in memory and write evidence only after all measured phases'
        }
    }
    [IO.File]::WriteAllText(
        (Join-Path $OutputRoot 'gate-evaluation.json'),
        ($gateEvaluation | ConvertTo-Json -Depth 30) + "`n",
        $utf8NoBom
    )
    if ($failedGates.Count -gt 0) {
        $details = @($failedGates | ForEach-Object { " - $($_.id): $($_.reason)" })
        throw "Benchmark gates failed. summary.json was not created:`n$($details -join "`n")"
    }

    $warmTiming = [ordered]@{}
    foreach ($metric in @(
        'hudAcknowledgementMs',
        'firstVisibleRegionMs',
        'visibleRegionGroupMs',
        'firstLongImageCompleteMs',
        'allImagesCompleteMs'
    )) {
        $values = [double[]] @($warm | ForEach-Object { [double] $_.timing.$metric })
        $warmTiming[$metric] = [ordered]@{
            p50 = Get-Percentile -Values $values -Percentile 0.50
            p95 = Get-Percentile -Values $values -Percentile 0.95
            maximum = [Math]::Round([double] (($values | Measure-Object -Maximum).Maximum), 3)
        }
    }
    $installedCold = @($runs | Where-Object { $_.kind -eq 'installed-cold' })[0]
    $cacheReplay = @($runs | Where-Object { $_.kind -eq 'cache-replay' })[0]
    $cancellationRaw = @($rawRuns | Where-Object { $_.kind -eq 'cancellation' })[0]
    $summary = [ordered]@{
        status = 'pass'
        benchmarkId = $Prerequisites.fixture.manifest.id
        recordedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
        warmRunCount = $warm.Count
        percentileMethod = 'nearest-rank'
        warmTimingMs = $warmTiming
        installedColdTimingMs = $installedCold.timing
        exactCacheReplayTimingMs = $cacheReplay.timing
        cancellationTimingMs = [ordered]@{
            page = $cancellationRaw.pageCancellationLatencyMs
            daemon = $cancellationRaw.daemonCancellationLatencyMs
        }
        resourceEvidence = $resource
        residentReuseEvidence = [ordered]@{
            sameDaemonInstance = $sameDaemon
            exactCacheReplay = $noInference
            inferenceModelsLoadedDuringReplay = $false
        }
        correctness = $cacheRaw.correctness
        jobRequestEvidence = $cacheRaw.jobRequests
        runtimeModelResourceIdentityEvidence = $cacheRaw.routes.resourceIdentityEvidence
        readerFeatures = $cacheRaw.readerFeatures
        overflow = $cacheRaw.overflow
        sourceReplacement = $cacheRaw.sourceReplacement
        runs = @($runs)
        gateEvidence = 'gate-evaluation.json'
        environmentEvidence = 'environment.json'
        liveNetworkSmokeIncluded = $false
    }
    [IO.File]::WriteAllText(
        (Join-Path $OutputRoot 'summary.json'),
        ($summary | ConvertTo-Json -Depth 30) + "`n",
        $utf8NoBom
    )
    $artifactHashes = @(
        Get-ChildItem -LiteralPath $OutputRoot -File |
            Where-Object { $_.Name -ne 'evidence-hashes.json' } |
            Sort-Object Name |
            ForEach-Object {
                [ordered]@{
                    file = $_.Name
                    bytes = $_.Length
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            }
    )
    [IO.File]::WriteAllText(
        (Join-Path $OutputRoot 'evidence-hashes.json'),
        ([ordered]@{ files = $artifactHashes } | ConvertTo-Json -Depth 5) + "`n",
        $utf8NoBom
    )
    return $summary
}

$fixtureValidation = Read-AndValidateFixture
if (-not $PrerequisitesOnly) {
    Assert-CompleteTranslationGold -Fixture $fixtureValidation
}
$prerequisites = Test-Prerequisites -Fixture $fixtureValidation
if (-not $prerequisites.ready) {
    $detail = ($prerequisites.failures | ForEach-Object { " - $_" }) -join "`n"
    throw "Chapter 5 release benchmark prerequisites are not satisfied:`n$detail"
}

if ($PrerequisitesOnly) {
    $releaseMeasurementPermitted = (
        $fixtureValidation.manifest.annotationStatus.status -eq 'complete' -and
        $fixtureValidation.manifest.annotationStatus.totalMissingFieldCount -eq 0
    )
    [ordered]@{
        status = if ($releaseMeasurementPermitted) { 'ready' } else { 'blocked-incomplete-gold' }
        benchmarkId = $fixtureValidation.manifest.id
        releaseMeasurementPermitted = $releaseMeasurementPermitted
        annotationStatus = $fixtureValidation.manifest.annotationStatus
        warmIterations = $Iterations
        nativeHost = $prerequisites.nativeHostExecutable
        daemon = $prerequisites.daemonExecutable
        performanceBuildAttestation = [ordered]@{
            path = $prerequisites.buildAttestationPath
            schema = [string] $prerequisites.buildAttestation.schema
            buildFingerprint = [string] $prerequisites.buildAttestation.buildFingerprint
            sourceTreeSha256 = [string] $prerequisites.buildAttestation.source.aggregateSha256
        }
        extension = [ordered]@{
            plan = $prerequisites.extensionPlan
            package = $prerequisites.extensionPackagePath
            generatedOnlyForMeasurement = $prerequisites.extensionPlan -eq 'build-current-source'
        }
        nativeRegistration = [ordered]@{
            plan = 'temporary-exact-HKCU-registration'
            registryPath = $prerequisites.nativeRegistryPath
            restoredAfterMeasurement = $true
        }
        firefox = $prerequisites.firefoxExecutable
        resources = $prerequisites.resourcePaths
        runtimeModelResourceIdentities = $prerequisites.resourceIdentities
        isolation = [ordered]@{
            firefoxProfile = 'fresh profile under the evidence directory'
            localAppData = 'isolated LOCALAPPDATA under the evidence directory'
            daemonState = 'empty explicit HSK_MANGA_STATE_DIR under isolated LOCALAPPDATA'
            resources = 'verified packaged models, fonts, CUDA, and llama runtime used in place'
            discoveryRecord = 'daemon-state.json'
            discoveryLock = 'daemon.lock'
            runtimeCache = 'browser-cache'
            internalSessionRoute = '/browser-internal/session'
        }
        gpu = $prerequisites.gpu
        note = 'Read-only preflight complete. No package was built, registry value was written, or daemon, Firefox, model, or network measurement was started.'
    } | ConvertTo-Json -Depth 8
    exit 0
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $stamp = [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $relativeEvidenceRoot = if ($isLiveSmoke) {
        ".cache\benchmark-evidence\live-asura\$benchmarkId\$stamp"
    } else {
        ".cache\benchmark-evidence\$benchmarkId\$stamp"
    }
    $OutputDirectory = Join-Path $repositoryRoot $relativeEvidenceRoot
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (
    (Test-Path -LiteralPath $OutputDirectory -PathType Container) -and
    @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count -gt 0
) {
    throw "Benchmark evidence directory must be absent or empty: $OutputDirectory"
}
[IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$profileDirectory = Join-Path $OutputDirectory 'firefox-profile'
$isolatedLocalAppData = Join-Path $OutputDirectory 'isolated-local-appdata'
$stateDirectory = Join-Path $isolatedLocalAppData 'Hskify\browser-companion'
$runtimeResourcesDirectory = $prerequisites.resourcesDirectory
[IO.Directory]::CreateDirectory($stateDirectory) | Out-Null
if (@(Get-ChildItem -LiteralPath $stateDirectory -Force).Count -ne 0) {
    throw "Installed-cold isolated daemon state is not empty: $stateDirectory"
}

$packagedBuildAttestationPath = Join-Path $OutputDirectory 'performance-build-attestation.json'
Copy-Item -LiteralPath $prerequisites.buildAttestationPath -Destination $packagedBuildAttestationPath

if ($prerequisites.extensionPlan -eq 'build-current-source') {
    $prerequisites.extensionPackagePath = Join-Path $OutputDirectory 'hskify-current.xpi'
    $prerequisites.extension = Build-CurrentExtensionPackage -PackagePath $prerequisites.extensionPackagePath
    $prerequisites.extensionVersion = [string] $prerequisites.extension.manifest.version
}
Assert-CurrentExtensionPackage -Extension $prerequisites.extension

$resourceEvidenceInputs = @(
    [ordered]@{ id = 'hsk'; path = $prerequisites.resourcePaths.hsk },
    [ordered]@{ id = 'dictionary'; path = $prerequisites.resourcePaths.dictionary },
    [ordered]@{ id = 'translation-model'; path = $prerequisites.resourcePaths.qwenModel },
    [ordered]@{ id = 'sans-font'; path = $prerequisites.resourcePaths.sansFont },
    [ordered]@{ id = 'serif-font'; path = $prerequisites.resourcePaths.serifFont }
)
$resourceEvidenceInputs += @(
    $prerequisites.resourcePaths.residentModels | ForEach-Object {
        [ordered]@{ id = $_.id; path = $_.path }
    }
)
$resourceEvidenceInputs += @(
    $prerequisites.resourcePaths.runtimeFiles | ForEach-Object {
        [ordered]@{
            id = 'runtime:' + $_.Substring($prerequisites.resourcePaths.runtimeRoot.Length + 1)
            path = $_
        }
    }
)
$resourceEvidence = @(
    foreach ($resource in $resourceEvidenceInputs) {
        [ordered]@{
            id = $resource.id
            path = $resource.path
            bytes = (Get-Item -LiteralPath $resource.path).Length
            sha256 = (Get-FileHash -LiteralPath $resource.path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
)
$modelEvidence = @($resourceEvidence | Where-Object { $_.id -eq 'translation-model' })
if (
    $modelEvidence.Count -ne 1 -or
    $modelEvidence[0].bytes -ne [int64] $prerequisites.translationModel.bytes -or
    $modelEvidence[0].sha256 -ne $prerequisites.translationModel.sha256
) {
    throw 'Explicit Qwen model identity does not match the mandatory pinned translation model'
}

$gitCommit = (& git.exe -C $repositoryRoot rev-parse HEAD).Trim()
$gitStatus = @(& git.exe -C $repositoryRoot status --short)
$dirtyDiffHash = (& git.exe -C $repositoryRoot diff --binary HEAD | & git.exe hash-object --stdin).Trim()
$systemFirefox = 'C:\Program Files\Mozilla Firefox\firefox.exe'
$environmentEvidence = [ordered]@{
    benchmarkId = $fixtureValidation.manifest.id
    evidenceKind = if ($isLiveSmoke) { 'live-asura-packaged-firefox-smoke' } else { 'deterministic-local-replica-benchmark' }
    requestedLiveChapterUrl = if ($isLiveSmoke) { $LiveSmokeChapterUrl } else { '' }
    recordedAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
    git = [ordered]@{
        commit = $gitCommit
        dirty = $gitStatus.Count -gt 0
        status = $gitStatus
        dirtyDiffHash = $dirtyDiffHash
    }
    buildFingerprint = $expectedBuildFingerprint
    performanceBuildAttestation = [ordered]@{
        file = 'performance-build-attestation.json'
        sourcePath = $prerequisites.buildAttestationPath
        bytes = (Get-Item -LiteralPath $packagedBuildAttestationPath).Length
        sha256 = (Get-FileHash -LiteralPath $packagedBuildAttestationPath -Algorithm SHA256).Hash.ToLowerInvariant()
        schema = [string] $prerequisites.buildAttestation.schema
        source = $prerequisites.buildAttestation.source
        build = $prerequisites.buildAttestation.build
        cuda = $prerequisites.buildAttestation.cuda
        hardware = $prerequisites.buildAttestation.hardware
        toolchain = $prerequisites.buildAttestation.toolchain
        llamaCppTag = [string] $prerequisites.buildAttestation.llamaCppTag
        builtAtUtc = [string] $prerequisites.buildAttestation.builtAtUtc
    }
    binaries = @(
        foreach ($path in @($prerequisites.nativeHostExecutable, $prerequisites.daemonExecutable)) {
            [ordered]@{
                path = $path
                bytes = (Get-Item -LiteralPath $path).Length
                sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    )
    extension = [ordered]@{
        path = $prerequisites.extensionPackagePath
        version = $prerequisites.extensionVersion
        bytes = $prerequisites.extension.bytes
        sha256 = $prerequisites.extension.sha256
        id = $expectedExtensionId
        packaging = $prerequisites.extensionPlan
    }
    firefox = [ordered]@{
        playwrightExecutable = $prerequisites.firefoxExecutable
        playwrightSha256 = (Get-FileHash -LiteralPath $prerequisites.firefoxExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
        installedFirefox = if (Test-Path -LiteralPath $systemFirefox) { $systemFirefox } else { '' }
        installedFirefoxVersion = if (Test-Path -LiteralPath $systemFirefox) {
            [string] (Get-Item -LiteralPath $systemFirefox).VersionInfo.ProductVersion
        } else {
            ''
        }
    }
    gpu = $prerequisites.gpu
    resources = $resourceEvidence
    runtimeModelResourceIdentities = $prerequisites.resourceIdentities
    isolation = [ordered]@{
        firefoxProfile = $profileDirectory
        localAppData = $isolatedLocalAppData
        daemonState = $stateDirectory
        runtimeResourceRoot = $runtimeResourcesDirectory
        exactNativeHostRegistryPath = $prerequisites.nativeRegistryPath
        nativeRegistrationLifetime = 'measurement-process-only; previous exact-key value restored in finally'
        discoveryRecord = 'daemon-state.json'
        discoveryLock = 'daemon.lock'
        runtimeCache = 'browser-cache'
        internalSessionRoute = '/browser-internal/session'
    }
    fixture = $fixtureValidation
    installedCold = [ordered]@{
        isolatedStateInitiallyEmpty = $true
        preexistingDaemonProcesses = 0
        runtimeCachesDeleted = $false
        resourcesDeleted = $false
        downloadsInvoked = $false
    }
    command = [ordered]@{
        iterations = $Iterations
        hskLevel = $HskLevel
        port = $Port
        runTimeoutMinutes = $RunTimeoutMinutes
        telemetryIntervalMs = $TelemetryIntervalMs
        headed = [bool] $Headed
    }
    evidencePolicy = if ($isLiveSmoke) {
        [ordered]@{
            deterministicLocalReplicaAggregationEligible = $false
            liveNetworkSmoke = $true
            bearerTokensPersisted = $false
        }
    } else {
        [ordered]@{
            vanillaImageDecodeIsRuntimeMetric = $false
            liveNetworkSmokeIncluded = $false
            bearerTokensPersisted = $false
        }
    }
}
[IO.File]::WriteAllText(
    (Join-Path $OutputDirectory 'environment.json'),
    ($environmentEvidence | ConvertTo-Json -Depth 20) + "`n",
    $utf8NoBom
)

$configPath = Join-Path $OutputDirectory 'driver-config.json'
$driverConfig = [ordered]@{
    repositoryRoot = $repositoryRoot
    manifestPath = $manifestPath
    outputDirectory = $OutputDirectory
    stateDirectory = $stateDirectory
    profileDirectory = $profileDirectory
    extensionPackagePath = $prerequisites.extensionPackagePath
    extensionVersion = $prerequisites.extensionVersion
    firefoxExecutable = $prerequisites.firefoxExecutable
    playwrightModule = $playwrightModule
    iterations = $Iterations
    hskLevel = $HskLevel
    port = $Port
    runTimeoutMs = $RunTimeoutMinutes * 60 * 1000
    headed = [bool] $Headed
    expectedResourceIdentities = $prerequisites.resourceIdentities
    chapterUrl = if ($isLiveSmoke) { $LiveSmokeChapterUrl } else { '' }
    firefoxUserPrefs = if ($isLiveSmoke) {
        [ordered]@{
            'extensions.webextOptionalPermissionPrompts' = $false
        }
    } else {
        [ordered]@{}
    }
}
[IO.File]::WriteAllText(
    $configPath,
    ($driverConfig | ConvertTo-Json -Depth 8) + "`n",
    $utf8NoBom
)

$stdoutPath = Join-Path $OutputDirectory 'driver.stdout.log'
$stderrPath = Join-Path $OutputDirectory 'driver.stderr.log'
$startInfo = New-Object Diagnostics.ProcessStartInfo
$startInfo.FileName = $prerequisites.node
$startInfo.Arguments = "`"$driverPath`" `"$configPath`""
$startInfo.WorkingDirectory = $repositoryRoot
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = -not $Headed
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.EnvironmentVariables['LOCALAPPDATA'] = $isolatedLocalAppData
$startInfo.EnvironmentVariables['HSK_MANGA_STATE_DIR'] = $stateDirectory
$startInfo.EnvironmentVariables['HSK_MANGA_RESOURCES_DIR'] = $runtimeResourcesDirectory
$startInfo.EnvironmentVariables['HSK_MANGA_HSK_PATH'] = $prerequisites.resourcePaths.hsk
$startInfo.EnvironmentVariables['HSK_MANGA_DICTIONARY_PATH'] = $prerequisites.resourcePaths.dictionary
$startInfo.EnvironmentVariables['HSK_MANGA_QWEN_MODEL_PATH'] = $prerequisites.resourcePaths.qwenModel

$nativeManifestPath = Join-Path $OutputDirectory "$expectedNativeHost.json"
$registration = $null
$process = $null
$telemetryFiles = $null
$cleanup = [ordered]@{
    driverStopped = $false
    isolatedFirefoxProcessesStopped = @()
    isolatedDaemonProcessesStopped = @()
    nativeRegistrationRestored = $false
    errors = @()
}
$cleanupErrors = [System.Collections.Generic.List[string]]::new()
try {
    $registration = Register-TemporaryNativeHost `
        -ManifestPath $nativeManifestPath `
        -NativeHostPath $prerequisites.nativeHostExecutable
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw 'Could not start the packaged-Firefox benchmark driver'
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $telemetryFiles = Measure-Driver `
        -Process $process `
        -OutputRoot $OutputDirectory `
        -StateDirectory $stateDirectory `
        -SmiPath $prerequisites.nvidiaSmi
    $process.WaitForExit()
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    [IO.File]::WriteAllText($stdoutPath, $stdout, $utf8NoBom)
    [IO.File]::WriteAllText($stderrPath, $stderr, $utf8NoBom)
    if ($process.ExitCode -ne 0) {
        throw "Packaged-Firefox benchmark failed with exit code $($process.ExitCode). Evidence: $OutputDirectory`n$stderr"
    }
}
finally {
    try {
        if ($null -ne $process) {
            try {
                if (-not $process.HasExited) {
                    Stop-Process -Id $process.Id -Force
                    $cleanup.driverStopped = $true
                }
            }
            catch {
                $cleanupErrors.Add("Driver cleanup failed: $($_.Exception.Message)")
            }
        }
        try {
            $cleanup.isolatedFirefoxProcessesStopped = @(
                Stop-IsolatedFirefox `
                    -ProfileDirectory $profileDirectory `
                    -FirefoxExecutable $prerequisites.firefoxExecutable
            )
        }
        catch {
            $cleanupErrors.Add("Isolated Firefox cleanup failed: $($_.Exception.Message)")
        }
        try {
            $cleanup.isolatedDaemonProcessesStopped = @(
                Stop-IsolatedDaemon `
                    -StateDirectory $stateDirectory `
                    -DaemonExecutable $prerequisites.daemonExecutable
            )
        }
        catch {
            $cleanupErrors.Add("Isolated daemon cleanup failed: $($_.Exception.Message)")
        }
    }
    finally {
        if ($null -ne $registration) {
            try {
                Restore-TemporaryNativeHost -Registration $registration
                $cleanup.nativeRegistrationRestored = $true
            }
            catch {
                $cleanupErrors.Add("Native-host registration restoration failed: $($_.Exception.Message)")
            }
        }
        $cleanup.errors = @($cleanupErrors)
        [IO.File]::WriteAllText(
            (Join-Path $OutputDirectory 'cleanup.json'),
            ($cleanup | ConvertTo-Json -Depth 8) + "`n",
            $utf8NoBom
        )
    }
}
if ($cleanupErrors.Count -gt 0) {
    throw "Benchmark cleanup was incomplete:`n$(($cleanupErrors | ForEach-Object { " - $_" }) -join "`n")"
}

Write-Output "Raw evidence: $OutputDirectory"
if ($isLiveSmoke) {
    $liveEvidencePath = Join-Path $OutputDirectory 'live-asura-smoke.json'
    if (-not (Test-Path -LiteralPath $liveEvidencePath -PathType Leaf)) {
        throw "Live smoke driver did not write its separate evidence JSON: $liveEvidencePath"
    }
    Get-Content -LiteralPath $liveEvidencePath -Raw -Encoding utf8
} else {
    $summary = Write-CompletedEvidence `
        -OutputRoot $OutputDirectory `
        -Prerequisites $prerequisites `
        -TelemetryFiles $telemetryFiles `
        -Environment $environmentEvidence
    $summary | ConvertTo-Json -Depth 20
}
