$ErrorActionPreference = 'Stop'
$registryPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts\local.mangalations.hsk_manga'
$manifestPath = Join-Path $env:LOCALAPPDATA 'Mangalations\HSKMangaTranslator\native-host\local.mangalations.hsk_manga.json'

if (Test-Path -LiteralPath $registryPath) {
    Remove-Item -LiteralPath $registryPath -Force
}
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    Remove-Item -LiteralPath $manifestPath -Force
}
