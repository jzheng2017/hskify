$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$registerScript = Join-Path $repositoryRoot 'installers\windows\native-host-registration\Register-NativeHost.ps1'
$unregisterScript = Join-Path $repositoryRoot 'installers\windows\native-host-registration\Unregister-NativeHost.ps1'
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('hsk-manga-registration-' + [Guid]::NewGuid().ToString('N'))
$testRegistryPath = 'HKCU:\Software\Hskify\Tests\' + [Guid]::NewGuid().ToString('N')
$previousLocalAppData = $env:LOCALAPPDATA

try {
    [IO.Directory]::CreateDirectory($temporaryRoot) | Out-Null
    $env:LOCALAPPDATA = Join-Path $temporaryRoot 'local-app-data'
    $hostDirectory = Join-Path $temporaryRoot 'host'
    [IO.Directory]::CreateDirectory($hostDirectory) | Out-Null
    $hostPath = Join-Path $hostDirectory 'hsk-manga-native-host.exe'
    [IO.File]::WriteAllBytes($hostPath, [byte[]](77, 90))

    $manifestPath = & $registerScript -NativeHostPath $hostPath -RegistryPath $testRegistryPath
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw 'Windows registration did not create a manifest leaf file'
    }
    if ((Get-Item -LiteralPath $testRegistryPath).GetValue('') -ne $manifestPath) {
        throw 'Windows registration did not write the isolated manifest registry value'
    }

    & $unregisterScript -RegistryPath $testRegistryPath
    if ((Test-Path -LiteralPath $manifestPath) -or (Test-Path -LiteralPath $testRegistryPath)) {
        throw 'Windows unregistration did not remove the isolated manifest and registry key'
    }

    $directoryHost = Join-Path $temporaryRoot 'directory\hsk-manga-native-host.exe'
    [IO.Directory]::CreateDirectory($directoryHost) | Out-Null
    try {
        & $registerScript -NativeHostPath $directoryHost -RegistryPath $testRegistryPath
        throw 'Windows registration accepted a directory as the native host executable'
    }
    catch {
        if ($_.Exception.Message -notlike '*existing executable file*') {
            throw
        }
    }
    if ((Test-Path -LiteralPath $testRegistryPath) -or (Test-Path -LiteralPath $manifestPath)) {
        throw 'Rejected Windows registration unexpectedly changed installer state'
    }

    Write-Output 'Windows native-host registration checks passed'
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
