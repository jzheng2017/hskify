[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [switch]$Refresh,
    [int]$SmokeIterations = 25
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot ".cache\language-data"
}
$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$hskRevision = "8a6f229c699e2fd7ac8708156c55c3afb2be7e1f"
$hskBaseUrl = "https://raw.githubusercontent.com/glxxyz/hskhsk.com/$hskRevision/data/lists"
$hskPins = @(
    @{ Level = 1; Sha256 = "fb8b3a6ce764b7f67e64ad17f49136a43a8dbfe74b2a0d008d28c27b9a34c728" },
    @{ Level = 2; Sha256 = "44c6cab7a3168aafef7a0cdede60c03005de5445120588861a0876a92099c3a2" },
    @{ Level = 3; Sha256 = "64d90d369299bcaad2f78ae91c754b8d72f2d8f0c44d1df624abe0a225ade8d1" },
    @{ Level = 4; Sha256 = "29a026f3866c615cc24bd783296e4d6383b2bd57f4fea8dc1105a208c9fb16cc" },
    @{ Level = 5; Sha256 = "46ef80692829b95514534be5e6d8d85d4044c11b2844dc7c96b81a46312cda10" },
    @{ Level = 6; Sha256 = "ef46d5598adbf28e8cb1e1c3576bccc002119aef35213e01d8225e925b2a489c" }
)

function Get-LowerSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-PinnedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Sha256
    )

    $needsDownload = $Refresh -or -not (Test-Path -LiteralPath $Path)
    if (-not $needsDownload) {
        $needsDownload = (Get-LowerSha256 -Path $Path) -ne $Sha256
    }
    if ($needsDownload) {
        Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
    }
    $actual = Get-LowerSha256 -Path $Path
    if ($actual -ne $Sha256) {
        throw "SHA-256 mismatch for $Url`: expected $Sha256, got $actual"
    }
}

$rawHskDirectory = Join-Path $outputRoot "raw-hsk"
New-Item -ItemType Directory -Force -Path $rawHskDirectory | Out-Null
foreach ($pin in $hskPins) {
    $level = [int]$pin.Level
    $name = "HSK Official With Definitions 2012 L$level.txt"
    $url = "$hskBaseUrl/$([uri]::EscapeDataString($name))"
    $path = Join-Path $rawHskDirectory "hsk-$level.txt"
    Get-PinnedFile -Url $url -Path $path -Sha256 ([string]$pin.Sha256)
}

# The upstream files contain a handful of repeated headwords. Retain the first
# occurrence while traversing levels in ascending order so the lowest level
# wins, then normalize every retained word with the exact runtime normalizer.
$rawHskByWord = [ordered]@{}
foreach ($level in 1..6) {
    $path = Join-Path $rawHskDirectory "hsk-$level.txt"
    foreach ($line in [System.IO.File]::ReadAllLines($path, [System.Text.Encoding]::UTF8)) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $fields = $line -split "`t"
        if ($fields.Count -lt 5) {
            throw "Malformed HSK level $level row: $line"
        }
        $simplified = $fields[0].Trim()
        $pinyin = $fields[3].Trim()
        $gloss = $fields[4].Trim()
        if (-not $simplified -or -not $pinyin -or -not $gloss) {
            throw "Incomplete HSK level $level row: $line"
        }
        if (-not $rawHskByWord.Contains($simplified)) {
            $rawHskByWord[$simplified] = [pscustomobject]@{
                Level = $level
                Simplified = $simplified
                Pinyin = $pinyin
                Gloss = $gloss
            }
        }
    }
}

$rawHskEntries = @($rawHskByWord.Values)
$hskHeadwordsPath = Join-Path $outputRoot "hsk-2.0-headwords.txt"
$hskNormalizedHeadwordsPath = Join-Path $outputRoot "hsk-2.0-headwords.normalized.txt"
[System.IO.File]::WriteAllLines(
    $hskHeadwordsPath,
    @($rawHskEntries | ForEach-Object { $_.Simplified }),
    $utf8NoBom
)
& cargo run --locked -p hsk-control --bin hsk-normalize -- `
    --source $hskHeadwordsPath `
    --output $hskNormalizedHeadwordsPath
if ($LASTEXITCODE -ne 0) {
    throw "HSK normalization failed with exit code $LASTEXITCODE"
}

$normalizedHeadwords = [System.IO.File]::ReadAllLines(
    $hskNormalizedHeadwordsPath,
    [System.Text.Encoding]::UTF8
)
if ($normalizedHeadwords.Count -ne $rawHskEntries.Count) {
    throw "HSK normalization changed the row count"
}

# Runtime-equivalent forms also collapse to their first (therefore lowest-level)
# entry. All decomposition flags remain false unless they receive a separate
# linguistic audit; direct HSK headwords still validate normally.
$hskByWord = [ordered]@{}
for ($index = 0; $index -lt $rawHskEntries.Count; $index += 1) {
    $entry = $rawHskEntries[$index]
    $simplified = $normalizedHeadwords[$index]
    if ([string]::IsNullOrWhiteSpace($simplified)) {
        throw "HSK normalization produced an empty headword at row $($index + 1)"
    }
    if (-not $hskByWord.Contains($simplified)) {
        $hskByWord[$simplified] = [pscustomobject]@{
            Level = $entry.Level
            Simplified = $simplified
            Pinyin = $entry.Pinyin
            Gloss = $entry.Gloss
        }
    }
}

$hskSourcePath = Join-Path $outputRoot "hsk-2.0-source.tsv"
$hskLines = [System.Collections.Generic.List[string]]::new()
$hskLines.Add("level`tsimplified`tpinyin`tgloss`tindependently_usable")
foreach ($entry in $rawHskEntries) {
    $safeGloss = $entry.Gloss.Replace("`t", " ").Replace("`r", " ").Replace("`n", " ")
    $hskLines.Add("$($entry.Level)`t$($entry.Simplified)`t$($entry.Pinyin)`t$safeGloss`tfalse")
}
[System.IO.File]::WriteAllLines($hskSourcePath, $hskLines, $utf8NoBom)

$levelCounts = @(0, 0, 0, 0, 0, 0)
foreach ($entry in $hskByWord.Values) {
    $levelCounts[[int]$entry.Level - 1] += 1
}
$hskSourceHash = Get-LowerSha256 -Path $hskSourcePath
$hskMetadataPath = Join-Path $outputRoot "hsk-2.0-import.json"
$hskMetadata = [ordered]@{
    schemaVersion = 1
    kind = "hsk-2.0"
    standard = "2.0"
    datasetRevision = "hskhsk-$hskRevision-runtime-normalized-lowest-level-v1"
    completeness = "complete"
    source = [ordered]@{
        name = "HSK Official With Definitions 2012 L1-L6 (runtime-normalized, lowest-level deduplication)"
        url = "https://github.com/glxxyz/hskhsk.com/tree/$hskRevision/data/lists"
        revision = $hskRevision
        sha256 = $hskSourceHash
    }
    licence = [ordered]@{
        spdxExpression = "MIT"
        url = "https://github.com/glxxyz/hskhsk.com/blob/$hskRevision/LICENSE"
        attribution = "HSK six-level files from glxxyz/hskhsk.com; duplicate and runtime-equivalent headwords keep their lowest listed level."
        redistributionAllowed = $true
    }
    expectedEntryCount = $hskByWord.Count
    expectedLevelCounts = $levelCounts
}
[System.IO.File]::WriteAllText(
    $hskMetadataPath,
    ($hskMetadata | ConvertTo-Json -Depth 8) + "`n",
    $utf8NoBom
)

$hskArtifactPath = Join-Path $outputRoot "hsk-2.0.normalized.json"
& cargo run --locked -p hsk-control --bin hsk-import -- `
    --source $hskSourcePath `
    --metadata $hskMetadataPath `
    --output $hskArtifactPath `
    --delimiter tab
if ($LASTEXITCODE -ne 0) {
    throw "HSK import failed with exit code $LASTEXITCODE"
}

$cedictUrl = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip"
$cedictZipPath = Join-Path $outputRoot "cc-cedict.zip"
if ($Refresh -or -not (Test-Path -LiteralPath $cedictZipPath)) {
    Invoke-WebRequest -Uri $cedictUrl -OutFile $cedictZipPath -UseBasicParsing
}
$cedictExtractDirectory = Join-Path $outputRoot "cc-cedict-extract"
New-Item -ItemType Directory -Force -Path $cedictExtractDirectory | Out-Null
Expand-Archive -LiteralPath $cedictZipPath -DestinationPath $cedictExtractDirectory -Force
$cedictSourcePath = Join-Path $cedictExtractDirectory "cedict_ts.u8"
if (-not (Test-Path -LiteralPath $cedictSourcePath -PathType Leaf)) {
    throw "CC-CEDICT archive did not contain cedict_ts.u8"
}

$cedictLines = [System.IO.File]::ReadAllLines($cedictSourcePath, [System.Text.Encoding]::UTF8)
$cedictEntries = $cedictLines |
    Where-Object {
        $trimmed = $_.Trim()
        $trimmed -and -not $trimmed.StartsWith("#")
    } |
    Sort-Object -Unique
$cedictDateLine = $cedictLines | Where-Object { $_.StartsWith("#! date=") } | Select-Object -First 1
$cedictRevision = if ($cedictDateLine) {
    $cedictDateLine.Substring("#! date=".Length)
} else {
    Get-LowerSha256 -Path $cedictSourcePath
}
$cedictSourceHash = Get-LowerSha256 -Path $cedictSourcePath
$cedictMetadataPath = Join-Path $outputRoot "cc-cedict-import.json"
$cedictMetadata = [ordered]@{
    schemaVersion = 1
    kind = "cc-cedict"
    standard = $null
    datasetRevision = "mdbg-$cedictRevision"
    completeness = "complete"
    source = [ordered]@{
        name = "CC-CEDICT MDBG release"
        url = $cedictUrl
        revision = $cedictRevision
        sha256 = $cedictSourceHash
    }
    licence = [ordered]@{
        spdxExpression = "CC-BY-SA-4.0"
        url = "https://creativecommons.org/licenses/by-sa/4.0/"
        attribution = "CC-CEDICT, community maintained and published by MDBG."
        redistributionAllowed = $true
    }
    expectedEntryCount = $cedictEntries.Count
    expectedLevelCounts = $null
}
[System.IO.File]::WriteAllText(
    $cedictMetadataPath,
    ($cedictMetadata | ConvertTo-Json -Depth 8) + "`n",
    $utf8NoBom
)

$cedictArtifactPath = Join-Path $outputRoot "cc-cedict.normalized.json"
& cargo run --locked -p hsk-control --bin cedict-import -- `
    --source $cedictSourcePath `
    --metadata $cedictMetadataPath `
    --output $cedictArtifactPath
if ($LASTEXITCODE -ne 0) {
    throw "CC-CEDICT import failed with exit code $LASTEXITCODE"
}

if ($SmokeIterations -gt 0) {
    & cargo run --locked -p hsk-control --bin resource-smoke -- `
        $hskArtifactPath `
        $cedictArtifactPath `
        $SmokeIterations
    if ($LASTEXITCODE -ne 0) {
        throw "Language resource smoke test failed with exit code $LASTEXITCODE"
    }
}

$manifestPath = Join-Path $outputRoot "language-resources.json"
$manifest = [ordered]@{
    schemaVersion = 1
    generatedAtUtc = [DateTime]::UtcNow.ToString("o")
    hsk = [ordered]@{
        path = $hskArtifactPath
        sha256 = Get-LowerSha256 -Path $hskArtifactPath
        entries = $hskByWord.Count
        levelCounts = $levelCounts
        sourceSha256 = $hskSourceHash
    }
    dictionary = [ordered]@{
        path = $cedictArtifactPath
        sha256 = Get-LowerSha256 -Path $cedictArtifactPath
        entries = $cedictEntries.Count
        sourceSha256 = $cedictSourceHash
        archiveSha256 = Get-LowerSha256 -Path $cedictZipPath
        revision = $cedictRevision
    }
}
[System.IO.File]::WriteAllText(
    $manifestPath,
    ($manifest | ConvertTo-Json -Depth 8) + "`n",
    $utf8NoBom
)

Write-Output "Local language data is ready:"
Write-Output "  HSK:        $hskArtifactPath ($($hskByWord.Count) entries)"
Write-Output "  dictionary: $cedictArtifactPath ($($cedictEntries.Count) entries)"
Write-Output "  manifest:   $manifestPath"
