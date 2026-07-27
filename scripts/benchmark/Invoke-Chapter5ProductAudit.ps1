[CmdletBinding()]
param(
    [string]$OutputDirectory = ".cache/ch5-product-audit/latest",
    [int]$HskLevel = 5,
    [int[]]$PageOrders = @()
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))
if (Test-Path -LiteralPath $outputRoot) {
    throw "Audit output already exists: $outputRoot"
}
$stateRoot = Join-Path $outputRoot "state"
$null = New-Item -ItemType Directory -Path $stateRoot -Force

$manifestPath = Join-Path $repoRoot "fixtures/benchmarks/30-years-since-the-prologue-chapter-5/manifest.json"
$sourceRoot = Join-Path $repoRoot ".cache/benchmarks/30-years-since-the-prologue-chapter-5/source"
$daemonPath = Join-Path $repoRoot "target/release/hsk-manga-browser-daemon.exe"
$origin = "moz-extension://00000000-0000-4000-8000-000000000005"
$fingerprint = "hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-26-r2"

foreach ($required in @($manifestPath, $sourceRoot, $daemonPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required audit input is missing: $required"
    }
}

function Send-Request {
    param(
        [System.Net.Http.HttpClient]$Client,
        [System.Net.Http.HttpRequestMessage]$Request
    )
    $response = $Client.SendAsync($Request).GetAwaiter().GetResult()
    $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $response.IsSuccessStatusCode) {
        throw "HTTP $([int]$response.StatusCode) $($response.ReasonPhrase): $body"
    }
    if ([string]::IsNullOrWhiteSpace($body)) {
        return $null
    }
    return $body | ConvertFrom-Json
}

function New-BrowserRequest {
    param(
        [string]$Method,
        [string]$Url,
        [string]$Token
    )
    $request = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::new($Method),
        $Url
    )
    $null = $request.Headers.TryAddWithoutValidation("Authorization", "Bearer $Token")
    $null = $request.Headers.TryAddWithoutValidation("x-hsk-manga-extension-origin", $origin)
    return $request
}

function New-JsonContent {
    param([string]$Json)
    $content = [System.Net.Http.ByteArrayContent]::new(
        [System.Text.Encoding]::UTF8.GetBytes($Json)
    )
    $null = $content.Headers.TryAddWithoutValidation("Content-Type", "application/json")
    return $content
}

$stdoutPath = Join-Path $outputRoot "daemon.stdout.log"
$stderrPath = Join-Path $outputRoot "daemon.stderr.log"
$daemon = Start-Process `
    -FilePath $daemonPath `
    -ArgumentList @("--state-dir", $stateRoot, "--idle-milliseconds", "3600000") `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$client = $null
try {
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(5)
    $statePath = Join-Path $stateRoot "daemon-state.json"
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        if ($daemon.HasExited) {
            throw "Audit daemon exited with code $($daemon.ExitCode)."
        }
        if (Test-Path -LiteralPath $statePath) {
            try {
                $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                if ($state.port -and $state.controlSecret) {
                    break
                }
            } catch {
                # The daemon replaces this small record atomically. Retry a
                # transient read while startup completes.
            }
        }
        Start-Sleep -Milliseconds 50
    } while ([DateTime]::UtcNow -lt $deadline)
    if (-not $state.port) {
        throw "Timed out waiting for the isolated daemon state record."
    }

    $baseUrl = "http://127.0.0.1:$($state.port)"
    $sessionRequest = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::Post,
        "$baseUrl/browser-internal/session"
    )
    $null = $sessionRequest.Headers.TryAddWithoutValidation(
        "x-hsk-manga-control",
        [string]$state.controlSecret
    )
    $sessionJson = @{ extensionOrigin = $origin } | ConvertTo-Json -Compress
    $sessionRequest.Content = New-JsonContent -Json $sessionJson
    $session = Send-Request -Client $client -Request $sessionRequest
    if ($session.buildFingerprint -ne $fingerprint) {
        throw "Unexpected daemon fingerprint: $($session.buildFingerprint)"
    }

    $healthRequest = New-BrowserRequest -Method "GET" -Url "$baseUrl/health" -Token $session.token
    $health = Send-Request -Client $client -Request $healthRequest
    if ($health.setupState -ne "ready") {
        $setupRequest = New-BrowserRequest `
            -Method "POST" `
            -Url "$baseUrl/setup/models" `
            -Token $session.token
        $setup = Send-Request -Client $client -Request $setupRequest
        $setupDeadline = [DateTime]::UtcNow.AddMinutes(10)
        while ($setup.state -notin @("ready", "failed") -and [DateTime]::UtcNow -lt $setupDeadline) {
            Start-Sleep -Milliseconds 100
            $setupRequest = New-BrowserRequest `
                -Method "GET" `
                -Url "$baseUrl/setup" `
                -Token $session.token
            $setup = Send-Request -Client $client -Request $setupRequest
        }
        if ($setup.state -ne "ready") {
            throw "Managed-resource verification failed: $($setup.errorCode) $($setup.message)"
        }
        $healthRequest = New-BrowserRequest `
            -Method "GET" `
            -Url "$baseUrl/health" `
            -Token $session.token
        $health = Send-Request -Client $client -Request $healthRequest
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $selectedImages = @($manifest.images | Sort-Object order)
    if ($PageOrders.Count -gt 0) {
        $requested = [Collections.Generic.HashSet[int]]::new()
        foreach ($order in $PageOrders) {
            if ($order -lt 1 -or $order -gt [int]$manifest.pageCount) {
                throw "PageOrders contains out-of-range page $order."
            }
            if (-not $requested.Add($order)) {
                throw "PageOrders contains duplicate page $order."
            }
        }
        $selectedImages = @($selectedImages | Where-Object {
            $requested.Contains([int]$_.order)
        })
    }
    $cases = [System.Collections.Generic.List[object]]::new()
    $totalAccepted = 0
    foreach ($image in $selectedImages) {
        $sourcePath = Join-Path $sourceRoot $image.file
        $sourceBytes = [System.IO.File]::ReadAllBytes($sourcePath)
        $metadata = [ordered]@{
            buildFingerprint = $fingerprint
            clientImageId = "chapter5-audit-$($image.order)"
            sourceSha256 = $image.sha256
            sourceMimeType = "image/webp"
            naturalWidth = $image.width
            naturalHeight = $image.height
            pageSessionId = "chapter5-product-audit"
            pageIndex = [int]$image.order - 1
            settings = [ordered]@{
                sourceLanguage = "en"
                targetLanguage = "zh-CN"
                hskStandard = "2.0"
                hskLevel = $HskLevel
                readingDirection = "auto"
                translateSoundEffects = $false
            }
            visibleRects = @(
                [ordered]@{ x = 0.0; y = 0.0; width = 1.0; height = 0.45 }
            )
        }
        $multipart = [System.Net.Http.MultipartFormDataContent]::new()
        $imageContent = [System.Net.Http.ByteArrayContent]::new($sourceBytes)
        $imageContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new(
            "image/webp"
        )
        $multipart.Add($imageContent, "image", [string]$image.file)
        $requestJson = $metadata | ConvertTo-Json -Depth 10 -Compress
        $requestContent = New-JsonContent -Json $requestJson
        $multipart.Add($requestContent, "request")

        $started = [System.Diagnostics.Stopwatch]::StartNew()
        $createRequest = New-BrowserRequest -Method "POST" -Url "$baseUrl/jobs" -Token $session.token
        $createRequest.Content = $multipart
        $created = Send-Request -Client $client -Request $createRequest
        $acceptanceMs = $started.ElapsedMilliseconds

        $sequence = 0L
        $updates = [System.Collections.Generic.List[object]]::new()
        $regions = [ordered]@{}
        $terminal = $null
        while (-not $terminal) {
            $pollRequest = New-BrowserRequest `
                -Method "GET" `
                -Url "$baseUrl/jobs/$($created.jobId)/updates?after=$sequence&waitMs=20000" `
                -Token $session.token
            $batch = Send-Request -Client $client -Request $pollRequest
            foreach ($update in $batch.updates) {
                $updates.Add($update)
                $sequence = [Math]::Max($sequence, [long]$update.sequence)
                if ($update.type -eq "regionReady") {
                    $regions[[string]$update.region.id] = $update.region
                } elseif ($update.type -eq "regionRefined" -and $regions.Contains([string]$update.regionId)) {
                    $region = $regions[[string]$update.regionId]
                    $region.displayedChinese = $update.displayedChinese
                    $region.pinyin = $update.pinyin
                    $region.hsk = $update.hsk
                } elseif ($update.type -in @("complete", "failed", "cancelled")) {
                    $terminal = $update
                }
            }
            if ($sequence -lt [long]$batch.nextSequence) {
                $sequence = [long]$batch.nextSequence
            }
        }
        $started.Stop()
        if ($terminal.type -ne "complete") {
            throw "Page $($image.file) ended as $($terminal.type): $($terminal.message)"
        }

        $accepted = @($regions.Values)
        $totalAccepted += $accepted.Count
        $case = [ordered]@{
            sourceFile = $image.file
            sourceSha256 = $image.sha256
            width = $image.width
            height = $image.height
            acceptanceMs = $acceptanceMs
            totalMs = $started.ElapsedMilliseconds
            acceptedRegions = $accepted
            terminal = $terminal
            updates = $updates
        }
        $cases.Add($case)
        $case | ConvertTo-Json -Depth 30 | Set-Content `
            -LiteralPath (Join-Path $outputRoot ("{0:D3}.json" -f [int]$image.order)) `
            -Encoding UTF8
        Write-Host ("{0:D3} {1,5} ms {2,3} regions" -f [int]$image.order, $started.ElapsedMilliseconds, $accepted.Count)
    }

    $times = @($cases | ForEach-Object { [long]$_.totalMs } | Sort-Object)
    $p95Index = [Math]::Max(0, [Math]::Ceiling($times.Count * 0.95) - 1)
    $summary = [ordered]@{
        auditedAtUtc = [DateTime]::UtcNow.ToString("o")
        evidenceMethod = "fresh isolated daemon state and authenticated unversioned product HTTP routes"
        buildFingerprint = $fingerprint
        health = $health
        totalAcceptedRegions = $totalAccepted
        totalMs = [long](($times | Measure-Object -Sum).Sum)
        p95PageMs = $times[$p95Index]
        maxPageMs = $times[-1]
        cases = $cases
    }
    $summary | ConvertTo-Json -Depth 30 | Set-Content `
        -LiteralPath (Join-Path $outputRoot "summary.json") `
        -Encoding UTF8
    Write-Host "Accepted regions: $totalAccepted"
    Write-Host "p95 page: $($summary.p95PageMs) ms"
    Write-Host "max page: $($summary.maxPageMs) ms"
} finally {
    if ($client) {
        $client.Dispose()
    }
    if ($daemon -and -not $daemon.HasExited) {
        Stop-Process -Id $daemon.Id -Force
        $daemon.WaitForExit()
    }
}
