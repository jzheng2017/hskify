[CmdletBinding()]
param(
    [string]$RunId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
    [string]$QwenCache = 'C:\Users\Jiankai\Documents\hskify\.cache\model-benchmark',
    [string]$CacheRoot = '',
    [ValidateSet('all', 'qwen3.5-4b-q4-k-m', 'qwen3.5-2b-q4-k-m', 'hy-mt2-1.8b-q4-k-m', 'assemble')]
    [string]$Candidate = 'all'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($CacheRoot)) {
    $CacheRoot = Join-Path $PSScriptRoot '..\.cache\translation-model-benchmark'
}
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$cacheRootResolved = [System.IO.Path]::GetFullPath($CacheRoot)
$modelsDirectory = Join-Path $cacheRootResolved 'models'
$runtimeRoot = Join-Path $repositoryRoot '.cache\runtime-benchmark'
$libclangDirectory = Join-Path $repositoryRoot '.cache\tools\libclang\clang\native'
$libclangDll = Join-Path $libclangDirectory 'libclang.dll'
$msvcToolsRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC'
$windowsKitsIncludeRoot = 'C:\Program Files (x86)\Windows Kits\10\Include'
$outputDirectory = Join-Path $cacheRootResolved (Join-Path 'runs' $RunId)
$fixtureRoot = Join-Path $repositoryRoot 'fixtures\benchmarks\30-years-since-the-prologue-chapter-5'
$manifestPath = Join-Path $fixtureRoot 'manifest.json'
$annotations = Join-Path $fixtureRoot 'annotations'
$qwen4b = Join-Path $QwenCache 'Qwen3.5-4B-Q4_K_M.gguf'
$qwen2b = Join-Path $QwenCache 'Qwen3.5-2B-Q4_K_M.gguf'
$hyMt2 = Join-Path $modelsDirectory 'Hy-MT2-1.8B-Q4_K_M.gguf'
$hyUrl = 'https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/1cd5208700acedef4ef93019b6cfc148b8522d45/Hy-MT2-1.8B-Q4_K_M.gguf'

$models = @(
    [pscustomobject]@{
        Id = 'qwen3.5-4b-q4-k-m'
        Path = $qwen4b
        Bytes = [int64]2740937888
        Sha256 = '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
        Revision = 'e87f176479d0855a907a41277aca2f8ee7a09523'
    },
    [pscustomobject]@{
        Id = 'qwen3.5-2b-q4-k-m'
        Path = $qwen2b
        Bytes = [int64]1280835840
        Sha256 = 'aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223'
        Revision = 'f6d5376be1edb4d416d56da11e5397a961aca8ae'
    },
    [pscustomobject]@{
        Id = 'hy-mt2-1.8b-q4-k-m'
        Path = $hyMt2
        Bytes = [int64]1133080448
        Sha256 = 'dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699'
        Revision = '1cd5208700acedef4ef93019b6cfc148b8522d45'
    }
)

function Test-LatinAlphabeticCharacter {
    param([Parameter(Mandatory)][char] $Character)

    $codePoint = [int] $Character
    return (
        ($codePoint -ge 0x0041 -and $codePoint -le 0x005a) -or
        ($codePoint -ge 0x0061 -and $codePoint -le 0x007a) -or
        ($codePoint -ge 0x00c0 -and $codePoint -le 0x024f) -or
        ($codePoint -ge 0x1e00 -and $codePoint -le 0x1eff)
    )
}

function Test-ConfidentEnglishText {
    param([Parameter(Mandatory)][string] $Text)

    $alphabetic = @($Text.ToCharArray() | Where-Object { [char]::IsLetter($_) })
    return (
        $alphabetic.Count -gt 0 -and
        @($alphabetic | Where-Object { -not (Test-LatinAlphabeticCharacter $_) }).Count -eq 0
    )
}

function Assert-TranslationFixtureEligibility {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
    if (
        $manifest.schemaVersion -ne 3 -or
        $manifest.id -ne '30-years-since-the-prologue-chapter-5' -or
        $manifest.pageCount -ne 36 -or
        @($manifest.images).Count -ne $manifest.pageCount -or
        $manifest.totalExpectedRegionCount -lt 1 -or
        $manifest.totalExpectedDialogueBubbleCount +
            $manifest.totalExpectedNarrationCount -ne
            $manifest.totalExpectedRegionCount -or
        $manifest.totalExpectedEnglishTranslationTargetCount -lt 1 -or
        $manifest.totalExpectedUntouchedExclusionCount -lt 0 -or
        $manifest.totalExpectedEnglishTranslationTargetCount +
            $manifest.totalExpectedUntouchedExclusionCount -ne
            $manifest.totalExpectedRegionCount
    ) {
        throw 'Translation benchmark requires the canonical reviewed chapter 5 fixture'
    }
    $status = $manifest.annotationStatus
    if (
        $status.status -ne 'complete' -or
        $status.completedPageCount -ne $manifest.pageCount -or
        $status.requiredPageCount -ne $manifest.pageCount -or
        @($status.missingPages).Count -ne 0 -or
        $status.totalMissingFieldCount -ne 0
    ) {
        throw (
            "Translation-model release measurement is blocked by incomplete gold: " +
            "completedPageCount=$($status.completedPageCount)/$($status.requiredPageCount), " +
            "reasonCode=$($status.reasonCode), " +
            "missingFieldCounts=$($status.missingFieldCounts | ConvertTo-Json -Compress)"
        )
    }

    $regionsTotal = 0
    $detectorGold = 0
    $narration = 0
    $targets = 0
    $untouched = 0
    foreach ($image in @($manifest.images | Sort-Object order)) {
        $annotationPath = Join-Path $fixtureRoot ($image.annotation -replace '/', '\')
        $annotationFile = Get-Item -LiteralPath $annotationPath
        $annotationHash = (Get-FileHash -LiteralPath $annotationPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if (
            $annotationFile.Length -ne $image.annotationBytes -or
            $annotationHash -ne $image.annotationSha256
        ) {
            throw "Annotation identity mismatch: $($image.annotation)"
        }
        $annotation = Get-Content -LiteralPath $annotationPath -Raw -Encoding utf8 |
            ConvertFrom-Json
        $regions = @($annotation.regions)
        if ($regions.Count -ne $image.expectedRegionCount) {
            throw "Reviewed-region count mismatch: $($image.annotation)"
        }
        $pageDetectorGold = 0
        $pageNarration = 0
        $pageTargets = 0
        $pageUntouched = 0
        foreach ($region in $regions) {
            if ($region.kind -in @('dialogue', 'thought')) {
                $pageDetectorGold += 1
            }
            elseif ($region.kind -eq 'narration') {
                $pageNarration += 1
            }
            else {
                throw "$($region.id) has unsupported kind $($region.kind)"
            }
            $confidentEnglish = Test-ConfidentEnglishText ([string] $region.normalizedEnglish)
            $translationTargetProperty = $region.PSObject.Properties['translationTarget']
            if ($null -eq $translationTargetProperty) {
                if (-not $confidentEnglish) {
                    throw "$($region.id) has no confident Latin English and cannot enter a translation-model batch"
                }
                foreach ($goldField in @('simplifiedChinese', 'pinyin', 'hskTokens')) {
                    $goldProperty = $region.PSObject.Properties[$goldField]
                    if (
                        $null -eq $goldProperty -or
                        ($goldField -eq 'hskTokens' -and @($goldProperty.Value).Count -eq 0) -or
                        ($goldField -ne 'hskTokens' -and
                            [string]::IsNullOrWhiteSpace([string] $goldProperty.Value))
                    ) {
                        throw "$($region.id).$goldField is missing; release measurement requires complete Chinese, pinyin, and HSK gold"
                    }
                }
                $pageTargets += 1
            }
            else {
                if ($region.translationTarget -ne $false -or $confidentEnglish) {
                    throw "$($region.id) has an invalid translationTarget marker"
                }
                $pageUntouched += 1
            }
        }
        if (
            $pageDetectorGold -ne $image.expectedDialogueBubbleCount -or
            $pageNarration -ne $image.expectedNarrationCount -or
            $pageTargets -ne $image.expectedEnglishTranslationTargetCount -or
            $pageUntouched -ne $image.expectedUntouchedExclusionCount
        ) {
            throw "Translation eligibility count mismatch: $($image.annotation)"
        }
        $regionsTotal += $regions.Count
        $detectorGold += $pageDetectorGold
        $narration += $pageNarration
        $targets += $pageTargets
        $untouched += $pageUntouched
    }
    if (
        $regionsTotal -ne $manifest.totalExpectedRegionCount -or
        $detectorGold -ne $manifest.totalExpectedDialogueBubbleCount -or
        $narration -ne $manifest.totalExpectedNarrationCount -or
        $targets -ne $manifest.totalExpectedEnglishTranslationTargetCount -or
        $untouched -ne $manifest.totalExpectedUntouchedExclusionCount
    ) {
        throw 'Translation fixture eligibility totals do not match the chapter 5 manifest'
    }
    [ordered]@{
        reviewedRegions = $regionsTotal
        detectedBubbleGold = $detectorGold
        narrationRegions = $narration
        englishTranslationTargets = $targets
        untouchedExclusions = $untouched
    }
}

$fixtureEligibility = Assert-TranslationFixtureEligibility
New-Item -ItemType Directory -Force -Path $modelsDirectory, $outputDirectory | Out-Null
$fixtureEligibility | ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $outputDirectory 'fixture-eligibility.json') -Encoding utf8
$requiresHyMt2 = $Candidate -in @('all', 'hy-mt2-1.8b-q4-k-m')
if ($requiresHyMt2 -and
    (-not (Test-Path -LiteralPath $hyMt2) -or
    (Get-Item -LiteralPath $hyMt2).Length -ne [int64]1133080448)) {
    & curl.exe -L --fail --retry 6 --retry-delay 5 -C - --output $hyMt2 $hyUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Hy-MT2 download failed with exit code $LASTEXITCODE"
    }
}

$modelsToVerify = if ($Candidate -eq 'all') {
    $models
}
elseif ($Candidate -eq 'assemble') {
    @()
}
else {
    @($models | Where-Object Id -eq $Candidate)
}
$verification = foreach ($model in $modelsToVerify) {
    if (-not (Test-Path -LiteralPath $model.Path)) {
        throw "Missing model: $($model.Path)"
    }
    $file = Get-Item -LiteralPath $model.Path
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $model.Path).Hash.ToLowerInvariant()
    if ($file.Length -ne $model.Bytes) {
        throw "$($model.Id) byte mismatch: expected $($model.Bytes), got $($file.Length)"
    }
    if ($hash -ne $model.Sha256) {
        throw "$($model.Id) SHA-256 mismatch: expected $($model.Sha256), got $hash"
    }
    [pscustomobject]@{
        id = $model.Id
        path = $file.FullName
        bytes = $file.Length
        sha256 = $hash
        repositoryRevision = $model.Revision
    }
}
if ($verification) {
    $verificationPath = Join-Path $outputDirectory "model-verification-$Candidate.json"
    @($verification) | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $verificationPath -Encoding utf8
    $allVerification = Get-ChildItem -LiteralPath $outputDirectory -Filter 'model-verification-*.json' |
        Sort-Object Name |
        ForEach-Object { @(Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json) } |
        Group-Object id |
        ForEach-Object { $_.Group | Select-Object -Last 1 }
    @($allVerification) | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $outputDirectory 'model-verification.json') -Encoding utf8
}

$gpu = & nvidia-smi.exe --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) {
    throw 'nvidia-smi failed; refusing to record a CUDA benchmark'
}
[pscustomobject]@{
    capturedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    repositoryRoot = $repositoryRoot
    gitCommit = (& git -C $repositoryRoot rev-parse HEAD)
    os = [System.Environment]::OSVersion.VersionString
    gpu = $gpu
    llamaCppTag = 'b8935'
    inferenceThreads = 6
    cargoBuildJobs = 4
} | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $outputDirectory 'environment.json') -Encoding utf8

$env:KOHARU_INFERENCE_THREADS = '6'
$env:KOHARU_DATA_ROOT = $runtimeRoot
$env:CARGO_BUILD_JOBS = '4'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '4'
if (-not (Test-Path -LiteralPath $libclangDll -PathType Leaf)) {
    throw "Missing portable libclang: $libclangDll"
}
$env:LIBCLANG_PATH = $libclangDirectory
$msvcInclude = Get-ChildItem -LiteralPath $msvcToolsRoot -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1 |
    ForEach-Object { Join-Path $_.FullName 'include' }
$windowsKitVersion = Get-ChildItem -LiteralPath $windowsKitsIncludeRoot -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $msvcInclude -or
    -not (Test-Path -LiteralPath (Join-Path $msvcInclude 'stdbool.h')) -or
    -not $windowsKitVersion) {
    throw 'MSVC and Windows SDK C headers are required for bindgen'
}
$bindgenIncludes = @(
    $msvcInclude,
    (Join-Path $windowsKitVersion.FullName 'ucrt'),
    (Join-Path $windowsKitVersion.FullName 'shared'),
    (Join-Path $windowsKitVersion.FullName 'um')
)
$env:BINDGEN_EXTRA_CLANG_ARGS = ($bindgenIncludes |
    ForEach-Object { '-I"' + $_ + '"' }) -join ' '

function Get-BenchmarkMemorySnapshot {
    $os = Get-CimInstance Win32_OperatingSystem
    $pageFiles = @(Get-CimInstance Win32_PageFileUsage)
    $performance = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
    $gpuLine = & nvidia-smi.exe `
        --query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or @($gpuLine).Count -ne 1) {
        throw 'nvidia-smi memory telemetry failed'
    }
    $gpuFields = $gpuLine.Split(',') | ForEach-Object { $_.Trim() }
    [pscustomobject]@{
        capturedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
        availablePhysicalMiB = [math]::Round($os.FreePhysicalMemory / 1024, 3)
        availableVirtualMiB = [math]::Round($os.FreeVirtualMemory / 1024, 3)
        pageFileAllocatedMiB = [int64](($pageFiles | Measure-Object AllocatedBaseSize -Sum).Sum)
        pageFileUsedMiB = [int64](($pageFiles | Measure-Object CurrentUsage -Sum).Sum)
        pagesInputPerSecond = if ($performance) { [int64]$performance.PagesInputPersec } else { 0 }
        pageReadsPerSecond = if ($performance) { [int64]$performance.PageReadsPersec } else { 0 }
        gpuMemoryUsedMiB = [int64]$gpuFields[0]
        gpuMemoryFreeMiB = [int64]$gpuFields[1]
        gpuUtilizationPercent = [int64]$gpuFields[2]
        gpuMemoryUtilizationPercent = [int64]$gpuFields[3]
    }
}

function Assert-BenchmarkPreflight {
    param(
        [Parameter(Mandatory)]
        [string]$CandidateId
    )
    $competing = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and (
            $_.Name -match '(?i)^(hskify.*|hsk-manga.*|translation_model_benchmark|llama.*|ollama.*|kobold.*|ocr.*)\.exe$' -or
            ($_.Name -match '(?i)^python(?:w)?\.exe$' -and
                $_.CommandLine -match '(?i)(hskify|ocr|llama|gguf|model-benchmark)')
        )
    })
    $snapshot = Get-BenchmarkMemorySnapshot
    [pscustomobject]@{
        candidateId = $CandidateId
        safePhysicalFloorMiB = 8192
        safeVirtualFloorMiB = 8192
        competingProcesses = $competing | Select-Object ProcessId, Name, ExecutablePath, CommandLine
        snapshot = $snapshot
    } | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $outputDirectory "preflight-$CandidateId.json") -Encoding utf8
    if ($competing.Count -ne 0) {
        throw "Competing hskify/OCR/llama model process detected before $CandidateId"
    }
    if ($snapshot.availablePhysicalMiB -lt 8192 -or $snapshot.availableVirtualMiB -lt 8192) {
        throw "Unsafe RAM/commit headroom before $CandidateId"
    }
    if ($snapshot.pageFileAllocatedMiB -gt 0 -and
        $snapshot.pageFileUsedMiB -ge [math]::Floor($snapshot.pageFileAllocatedMiB * 0.75)) {
        throw "Unsafe pagefile usage before $CandidateId"
    }
}

function Invoke-MeasuredBenchmark {
    param(
        [Parameter(Mandatory)]
        [string]$Executable,
        [Parameter(Mandatory)]
        [string[]]$Arguments,
        [Parameter(Mandatory)]
        [string]$CandidateId
    )
    Assert-BenchmarkPreflight -CandidateId $CandidateId
    $stdoutPath = Join-Path $outputDirectory "stdout-$CandidateId.log"
    $stderrPath = Join-Path $outputDirectory "stderr-$CandidateId.log"
    $telemetryPath = Join-Path $outputDirectory "resource-samples-$CandidateId.jsonl"
    $summaryPath = Join-Path $outputDirectory "resource-summary-$CandidateId.json"
    [System.IO.File]::WriteAllText(
        $telemetryPath,
        '',
        [System.Text.UTF8Encoding]::new($false)
    )
    $samples = [System.Collections.Generic.List[object]]::new()
    $pagingStreak = 0
    $terminatedForSustainedPaging = $false
    $terminatedBelowPhysicalFloor = $false
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Executable
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.Arguments = ($Arguments | ForEach-Object {
        if ($_ -notmatch '[\s"]') {
            $_
        }
        else {
            '"' + (($_ -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
        }
    }) -join ' '
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start benchmark process for $CandidateId"
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $monitor = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        do {
            $process.Refresh()
            $machine = Get-BenchmarkMemorySnapshot
            $sample = [pscustomobject]@{
                capturedAtUtc = $machine.capturedAtUtc
                elapsedMilliseconds = [math]::Round($monitor.Elapsed.TotalMilliseconds, 3)
                processId = $process.Id
                processPrivateBytes = if ($process.HasExited) { $null } else { $process.PrivateMemorySize64 }
                processWorkingSetBytes = if ($process.HasExited) { $null } else { $process.WorkingSet64 }
                availablePhysicalMiB = $machine.availablePhysicalMiB
                availableVirtualMiB = $machine.availableVirtualMiB
                pageFileAllocatedMiB = $machine.pageFileAllocatedMiB
                pageFileUsedMiB = $machine.pageFileUsedMiB
                pagesInputPerSecond = $machine.pagesInputPerSecond
                pageReadsPerSecond = $machine.pageReadsPerSecond
                gpuMemoryUsedMiB = $machine.gpuMemoryUsedMiB
                gpuMemoryFreeMiB = $machine.gpuMemoryFreeMiB
                gpuUtilizationPercent = $machine.gpuUtilizationPercent
                gpuMemoryUtilizationPercent = $machine.gpuMemoryUtilizationPercent
            }
            $samples.Add($sample)
            $sample | ConvertTo-Json -Compress |
                Add-Content -LiteralPath $telemetryPath -Encoding utf8
            if ($sample.availablePhysicalMiB -lt 8192 -and -not $process.HasExited) {
                $terminatedBelowPhysicalFloor = $true
                Stop-Process -Id $process.Id -Force
                break
            }
            $pagingNow = $sample.pagesInputPerSecond -ge 250 -or
                $sample.pageReadsPerSecond -ge 250
            if ($pagingNow) {
                $pagingStreak += 1
            }
            else {
                $pagingStreak = 0
            }
            if ($pagingStreak -ge 10 -and -not $process.HasExited) {
                $terminatedForSustainedPaging = $true
                Stop-Process -Id $process.Id -Force
                break
            }
            if (-not $process.HasExited) {
                Start-Sleep -Milliseconds 500
            }
        } while (-not $process.HasExited)
        $process.WaitForExit()
        $benchmarkExitCode = [int]$process.ExitCode
    }
    finally {
        $monitor.Stop()
        $process.Refresh()
    }
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    [System.IO.File]::WriteAllText(
        $stdoutPath,
        $stdout,
        [System.Text.UTF8Encoding]::new($false)
    )
    [System.IO.File]::WriteAllText(
        $stderrPath,
        $stderr,
        [System.Text.UTF8Encoding]::new($false)
    )
    $privateSamples = @($samples | Where-Object processPrivateBytes -ne $null)
    $peakPrivateBytes = if ($privateSamples.Count -gt 0) {
        [int64](($privateSamples | Measure-Object processPrivateBytes -Maximum).Maximum)
    }
    else {
        $null
    }
    $peakWorkingSetBytes = if ($privateSamples.Count -gt 0) {
        [int64](($privateSamples | Measure-Object processWorkingSetBytes -Maximum).Maximum)
    }
    else {
        $null
    }
    [pscustomobject]@{
        candidateId = $CandidateId
        requestedSampleIntervalMilliseconds = 500
        monitoringWallMilliseconds = [math]::Round($monitor.Elapsed.TotalMilliseconds, 3)
        meanObservedSampleIntervalMilliseconds = if ($samples.Count -gt 1) {
            [math]::Round(
                ($samples[$samples.Count - 1].elapsedMilliseconds -
                    $samples[0].elapsedMilliseconds) / ($samples.Count - 1),
                3
            )
        }
        else {
            $null
        }
        samples = $samples.Count
        processPeakPrivateBytes = $peakPrivateBytes
        processPeakWorkingSetBytes = $peakWorkingSetBytes
        deviceWideGpuBaselineUsedMiB = [int64]$samples[0].gpuMemoryUsedMiB
        deviceWideGpuPeakUsedMiB = [int64](($samples | Measure-Object gpuMemoryUsedMiB -Maximum).Maximum)
        minimumAvailablePhysicalMiB = [double](($samples | Measure-Object availablePhysicalMiB -Minimum).Minimum)
        minimumAvailableVirtualMiB = [double](($samples | Measure-Object availableVirtualMiB -Minimum).Minimum)
        maximumPageFileUsedMiB = [int64](($samples | Measure-Object pageFileUsedMiB -Maximum).Maximum)
        maximumPagesInputPerSecond = [int64](($samples | Measure-Object pagesInputPerSecond -Maximum).Maximum)
        maximumPageReadsPerSecond = [int64](($samples | Measure-Object pageReadsPerSecond -Maximum).Maximum)
        terminatedForSustainedPaging = $terminatedForSustainedPaging
        terminatedBelowPhysicalFloor = $terminatedBelowPhysicalFloor
        benchmarkExitCode = $benchmarkExitCode
    } | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $summaryPath -Encoding utf8
    @(
        '--- STDOUT ---'
        @(Get-Content -LiteralPath $stdoutPath -ErrorAction SilentlyContinue)
        '--- STDERR ---'
        @(Get-Content -LiteralPath $stderrPath -ErrorAction SilentlyContinue)
    ) | Set-Content -LiteralPath (Join-Path $outputDirectory "console-$CandidateId.log") -Encoding utf8
    Get-Content -LiteralPath $stdoutPath -ErrorAction SilentlyContinue | Write-Host
    Get-Content -LiteralPath $stderrPath -ErrorAction SilentlyContinue |
        Select-String -Pattern 'BENCHMARK_CANDIDATE_(START|DONE)|ggml_cuda_init: found|using device CUDA0' |
        ForEach-Object { Write-Host $_.Line }
    if ($terminatedForSustainedPaging) {
        throw "Benchmark stopped after sustained paging/thrash for $CandidateId"
    }
    if ($terminatedBelowPhysicalFloor) {
        throw "Benchmark stopped after available physical RAM fell below 8 GiB for $CandidateId"
    }
    return $benchmarkExitCode
}

Push-Location $repositoryRoot
try {
    & cargo.exe build --release --locked -p koharu-llm --example translation_model_benchmark
    if ($LASTEXITCODE -ne 0) {
        throw "Release benchmark build failed with exit code $LASTEXITCODE"
    }
    $executable = Join-Path $repositoryRoot 'target\release\examples\translation_model_benchmark.exe'
    $benchmarkArguments = @(
        '--annotations', $annotations,
        '--output', $outputDirectory,
        '--runtime-root', $runtimeRoot,
        '--qwen4b', $qwen4b,
        '--qwen2b', $qwen2b,
        '--hy-mt2', $hyMt2
    )
    if ($Candidate -eq 'assemble') {
        $benchmarkArguments += '--assemble-only'
    }
    elseif ($Candidate -ne 'all') {
        $benchmarkArguments += @('--candidate', $Candidate)
    }
    $command = @($executable) + $benchmarkArguments
    $commandName = if ($Candidate -eq 'all') { 'all' } else { $Candidate }
    ($command | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' ' |
        Set-Content -LiteralPath (Join-Path $outputDirectory "command-$commandName.txt") -Encoding utf8
    $benchmarkExitCode = Invoke-MeasuredBenchmark `
        -Executable $executable `
        -Arguments $benchmarkArguments `
        -CandidateId $commandName
    if ($benchmarkExitCode -ne 0) {
        throw "Benchmark failed with exit code $benchmarkExitCode"
    }
}
finally {
    Pop-Location
}

Write-Host "Raw evidence: $outputDirectory"
