param(
    [Parameter(DontShow = $true)]
    [string] $RegistryPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts\local.hskify.hsk_manga'
)

$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $env:LOCALAPPDATA 'Hskify\native-host\local.hskify.hsk_manga.json'

if (Test-Path -LiteralPath $RegistryPath) {
    Remove-Item -LiteralPath $RegistryPath -Force
}
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    Remove-Item -LiteralPath $manifestPath -Force
}
