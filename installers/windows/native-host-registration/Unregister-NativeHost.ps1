param(
    [Parameter(DontShow = $true)]
    [string] $RegistryPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts\local.hskify.browser'
)

$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $env:LOCALAPPDATA 'Hskify\native-host\local.hskify.browser.json'

if (Test-Path -LiteralPath $RegistryPath) {
    Remove-Item -LiteralPath $RegistryPath -Force
}
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    Remove-Item -LiteralPath $manifestPath -Force
}
