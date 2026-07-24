param(
    [Parameter(DontShow = $true)]
    [string] $RegistryPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts\local.mangalations.hsk_manga'
)

$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $env:LOCALAPPDATA 'Mangalations\HSKMangaTranslator\native-host\local.mangalations.hsk_manga.json'

if (Test-Path -LiteralPath $RegistryPath) {
    Remove-Item -LiteralPath $RegistryPath -Force
}
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    Remove-Item -LiteralPath $manifestPath -Force
}
