param(
    [Parameter(Mandatory = $true)]
    [string] $NativeHostPath,
    [Parameter(DontShow = $true)]
    [string] $RegistryPath = 'HKCU:\Software\Mozilla\NativeMessagingHosts\local.hskify.hsk_manga'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $NativeHostPath -PathType Leaf)) {
    throw 'NativeHostPath must be an existing executable file'
}
$resolvedHost = (Resolve-Path -LiteralPath $NativeHostPath).Path
if ([IO.Path]::GetFileName($resolvedHost) -ne 'hsk-manga-native-host.exe') {
    throw 'NativeHostPath must name hsk-manga-native-host.exe'
}

$manifestDirectory = Join-Path $env:LOCALAPPDATA 'Hskify\HSKMangaTranslator\native-host'
$manifestPath = Join-Path $manifestDirectory 'local.hskify.hsk_manga.json'

[IO.Directory]::CreateDirectory($manifestDirectory) | Out-Null
$manifest = [ordered]@{
    name = 'local.hskify.hsk_manga'
    description = 'HSK Manga Translator local browser companion'
    path = $resolvedHost
    type = 'stdio'
    allowed_extensions = @('hsk-manga-translator@local.hskify')
}
$json = $manifest | ConvertTo-Json -Depth 3
[IO.File]::WriteAllText($manifestPath, $json, [Text.UTF8Encoding]::new($false))

New-Item -Path $RegistryPath -Force | Out-Null
Set-Item -LiteralPath $RegistryPath -Value $manifestPath
Write-Output $manifestPath
