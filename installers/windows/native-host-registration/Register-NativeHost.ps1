param(
    [Parameter(Mandatory = $true)]
    [string] $NativeHostPath
)

$ErrorActionPreference = 'Stop'
$resolvedHost = (Resolve-Path -LiteralPath $NativeHostPath).Path
if ([IO.Path]::GetFileName($resolvedHost) -ne 'hsk-manga-native-host.exe') {
    throw 'NativeHostPath must name hsk-manga-native-host.exe'
}

$manifestDirectory = Join-Path $env:LOCALAPPDATA 'Mangalations\HSKMangaTranslator\native-host'
$manifestPath = Join-Path $manifestDirectory 'local.mangalations.hsk_manga.json'
$registryPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts\local.mangalations.hsk_manga'

[IO.Directory]::CreateDirectory($manifestDirectory) | Out-Null
$manifest = [ordered]@{
    name = 'local.mangalations.hsk_manga'
    description = 'HSK Manga Translator local browser companion'
    path = $resolvedHost
    type = 'stdio'
    allowed_extensions = @('hsk-manga-translator@local.mangalations')
}
$json = $manifest | ConvertTo-Json -Depth 3
[IO.File]::WriteAllText($manifestPath, $json, [Text.UTF8Encoding]::new($false))

New-Item -Path $registryPath -Force | Out-Null
Set-Item -LiteralPath $registryPath -Value $manifestPath
Write-Output $manifestPath
