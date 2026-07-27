[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [Uri] $ChapterUrl,

    [Parameter(Mandatory = $true)]
    [string] $ExtensionPackagePath,

    [string] $OutputDirectory = '',

    [string] $FirefoxExecutable = '',

    [string] $ResourcesDirectory = '',

    [string] $HskResourcePath = '',

    [string] $DictionaryResourcePath = '',

    [string] $QwenModelPath = '',

    [string] $FontsDirectory = '',

    [string] $NativeHostExecutable = '',

    [string] $BuildAttestationPath = '',

    [ValidateRange(1, 6)]
    [int] $HskLevel = 5,

    [ValidateRange(1, 240)]
    [int] $RunTimeoutMinutes = 60,

    [switch] $PrerequisitesOnly,

    [switch] $Headed
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (
    -not $ChapterUrl.IsAbsoluteUri -or
    $ChapterUrl.Scheme -notin @('http', 'https') -or
    -not [string]::IsNullOrEmpty($ChapterUrl.UserInfo)
) {
    throw 'ChapterUrl must be an explicit credential-free HTTP or HTTPS URL'
}

$benchmarkScript = Join-Path $PSScriptRoot 'Benchmark-Chapter5.ps1'
$arguments = @{
    ExtensionPackagePath = [IO.Path]::GetFullPath($ExtensionPackagePath)
    LiveSmokeChapterUrl = $ChapterUrl.AbsoluteUri
    HskLevel = $HskLevel
    RunTimeoutMinutes = $RunTimeoutMinutes
}
foreach ($item in @(
    @{ Name = 'OutputDirectory'; Value = $OutputDirectory },
    @{ Name = 'FirefoxExecutable'; Value = $FirefoxExecutable },
    @{ Name = 'ResourcesDirectory'; Value = $ResourcesDirectory },
    @{ Name = 'HskResourcePath'; Value = $HskResourcePath },
    @{ Name = 'DictionaryResourcePath'; Value = $DictionaryResourcePath },
    @{ Name = 'QwenModelPath'; Value = $QwenModelPath },
    @{ Name = 'FontsDirectory'; Value = $FontsDirectory },
    @{ Name = 'NativeHostExecutable'; Value = $NativeHostExecutable },
    @{ Name = 'BuildAttestationPath'; Value = $BuildAttestationPath }
)) {
    if (-not [string]::IsNullOrWhiteSpace($item.Value)) {
        $arguments[$item.Name] = $item.Value
    }
}
if ($PrerequisitesOnly) { $arguments.PrerequisitesOnly = $true }
if ($Headed) { $arguments.Headed = $true }

& $benchmarkScript @arguments
