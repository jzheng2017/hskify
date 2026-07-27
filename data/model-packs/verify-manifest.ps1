param(
    [string] $ManifestPath = (Join-Path $PSScriptRoot 'manifest.v1.json')
)

$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

$expectedManifestFields = @(
    'generatedAt',
    'manifestVersion',
    'resourceIdentities',
    'translationModelId'
)
$actualManifestFields = @($manifest.PSObject.Properties.Name | Sort-Object)
if (($actualManifestFields -join "`n") -cne ($expectedManifestFields -join "`n")) {
    throw 'model manifest has unexpected fields'
}
if ($manifest.manifestVersion -ne 1) {
    throw 'unsupported manifest version'
}
if ([string] $manifest.generatedAt -cnotmatch '^\d{4}-\d{2}-\d{2}$') {
    throw 'generatedAt must be an ISO calendar date'
}
if ([string] $manifest.translationModelId -cne 'qwen3.5-4b') {
    throw 'translationModelId must be qwen3.5-4b'
}

$identityFields = @(
    'bytes',
    'filename',
    'id',
    'repository',
    'repositoryRevision',
    'sha256',
    'url'
)
$requiredResourceIds = @(
    'pp-ocr-v5-english-recognizer-config',
    'pp-ocr-v5-english-recognizer-model',
    'pp-ocr-v5-mobile-detector-model',
    'translation-model'
)
$resourceIdentities = @($manifest.resourceIdentities)
if ($resourceIdentities.Count -ne $requiredResourceIds.Count) {
    throw "resourceIdentities must contain exactly $($requiredResourceIds.Count) entries"
}

for ($index = 0; $index -lt $resourceIdentities.Count; $index++) {
    $identity = $resourceIdentities[$index]
    $expectedId = $requiredResourceIds[$index]
    $actualFields = @($identity.PSObject.Properties.Name | Sort-Object)
    if (($actualFields -join "`n") -cne ($identityFields -join "`n")) {
        throw "resource identity has unexpected fields: $($identity.id)"
    }
    if ([string] $identity.id -cne $expectedId) {
        throw "resourceIdentities must contain the four required identities in ordinal id order; expected $expectedId"
    }
    if (
        ([string] $identity.repository).Length -gt 256 -or
        [string] $identity.repository -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$'
    ) {
        throw "invalid resource repository: $($identity.id)"
    }
    if ([string] $identity.repositoryRevision -cnotmatch '^[0-9a-f]{40}$') {
        throw "unpinned resource repository revision: $($identity.id)"
    }
    if (
        ([string] $identity.filename).Length -gt 255 -or
        [string] $identity.filename -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$'
    ) {
        throw "invalid resource filename: $($identity.id)"
    }
    if (
        [uint64] $identity.bytes -eq 0 -or
        [uint64] $identity.bytes -gt 9007199254740991
    ) {
        throw "invalid byte size: $($identity.id)"
    }
    if ([string] $identity.sha256 -cnotmatch '^[0-9a-f]{64}$') {
        throw "invalid resource SHA-256: $($identity.id)"
    }
    $expectedUrl = "https://huggingface.co/$($identity.repository)/resolve/$($identity.repositoryRevision)/$($identity.filename)"
    if ([string] $identity.url -cne $expectedUrl) {
        throw "resource URL is not the exact pinned identity URL: $($identity.id)"
    }
}

$translationIdentity = $resourceIdentities[-1]
if ([string] $translationIdentity.filename -cne 'Qwen3.5-4B-Q4_K_M.gguf') {
    throw 'translation-model must be the pinned Qwen3.5-4B Q4_K_M artifact'
}

Write-Output "verified qwen3.5-4b and exactly $($resourceIdentities.Count) pinned resident resource identities"
