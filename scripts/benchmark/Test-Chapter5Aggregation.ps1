$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$benchmarkScript = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..\Benchmark-Chapter5.ps1')
)
$source = Get-Content -LiteralPath $benchmarkScript -Raw

foreach ($metric in @(
    'firstVisibleRegionMs'
    'visibleRegionGroupMs'
)) {
    if ($source -notmatch ("'" + [regex]::Escape($metric) + "'")) {
        throw "Warm aggregation does not use current timing metric: $metric"
    }
}

foreach ($removedMetric in @(
    'firstVisibleBubbleMs'
    'visibleBubbleGroupMs'
)) {
    if ($source -match [regex]::Escape($removedMetric)) {
        throw "Removed warm timing metric is still referenced: $removedMetric"
    }
}

Write-Output 'Chapter 5 aggregation metric checks passed'
