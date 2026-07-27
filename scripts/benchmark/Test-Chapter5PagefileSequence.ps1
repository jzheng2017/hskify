$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$benchmarkScript = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot '..\Benchmark-Chapter5.ps1')
)
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $benchmarkScript,
    [ref] $tokens,
    [ref] $parseErrors
)
if ($parseErrors.Count -ne 0) {
    throw "Benchmark script did not parse: $($parseErrors[0].Message)"
}
$functionAst = $ast.Find(
    {
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Measure-PagefileWriteSequences'
    },
    $true
)
if ($null -eq $functionAst) {
    throw 'Measure-PagefileWriteSequences was not found'
}
Invoke-Expression $functionAst.Extent.Text

function New-PagefileSample {
    param(
        [Parameter(Mandatory = $true)][int64] $At,
        [Parameter(Mandatory = $true)][uint64] $Pages,
        [bool] $Available = $true
    )
    return [pscustomobject]@{
        sampledAtEpochMs = $At
        systemPaging = [pscustomobject]@{
            available = $Available
            pagesOutputPerSec = $Pages
        }
    }
}

function Assert-SequenceMeasurement {
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)] $Samples,
        [Parameter(Mandatory = $true)][int64] $ExpectedLongestMs,
        [Parameter(Mandatory = $true)][int] $ExpectedSustainedCount
    )
    $actual = Measure-PagefileWriteSequences -Samples $Samples -SampleIntervalMs 1000
    if (
        [int64] $actual.longestMs -ne $ExpectedLongestMs -or
        [int] $actual.sustainedSequenceCount -ne $ExpectedSustainedCount
    ) {
        throw "$Name failed: expected longest=$ExpectedLongestMs count=$ExpectedSustainedCount; got longest=$($actual.longestMs) count=$($actual.sustainedSequenceCount)"
    }
}

Assert-SequenceMeasurement `
    -Name 'isolated spike' `
    -Samples @(
        (New-PagefileSample -At 0 -Pages 0)
        (New-PagefileSample -At 1000 -Pages 1)
        (New-PagefileSample -At 2000 -Pages 0)
    ) `
    -ExpectedLongestMs 0 `
    -ExpectedSustainedCount 0

Assert-SequenceMeasurement `
    -Name 'two consecutive positives' `
    -Samples @(
        (New-PagefileSample -At 0 -Pages 1)
        (New-PagefileSample -At 1000 -Pages 1)
        (New-PagefileSample -At 2000 -Pages 0)
    ) `
    -ExpectedLongestMs 1000 `
    -ExpectedSustainedCount 1

Assert-SequenceMeasurement `
    -Name 'separated spikes' `
    -Samples @(
        (New-PagefileSample -At 0 -Pages 1)
        (New-PagefileSample -At 1000 -Pages 0)
        (New-PagefileSample -At 2000 -Pages 1)
        (New-PagefileSample -At 3000 -Pages 0)
    ) `
    -ExpectedLongestMs 0 `
    -ExpectedSustainedCount 0

Assert-SequenceMeasurement `
    -Name 'unavailable sample breaks sequence' `
    -Samples @(
        (New-PagefileSample -At 0 -Pages 1)
        (New-PagefileSample -At 1000 -Pages 1 -Available $false)
        (New-PagefileSample -At 2000 -Pages 1)
        (New-PagefileSample -At 3000 -Pages 0)
    ) `
    -ExpectedLongestMs 0 `
    -ExpectedSustainedCount 0

Assert-SequenceMeasurement `
    -Name 'terminal sequence extension' `
    -Samples @(
        (New-PagefileSample -At 0 -Pages 1)
        (New-PagefileSample -At 500 -Pages 1)
    ) `
    -ExpectedLongestMs 1500 `
    -ExpectedSustainedCount 1

Write-Output 'Chapter 5 pagefile sequence checks passed'
