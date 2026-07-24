param(
    [string]$ManifestPath = (Join-Path $PSScriptRoot 'manifest.v1.json')
)

$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

if ($manifest.manifestVersion -ne 1) {
    throw 'unsupported manifest version'
}

$candidateIds = @{}
foreach ($candidate in $manifest.candidates) {
    if ($candidateIds.ContainsKey($candidate.id)) {
        throw "duplicate candidate id: $($candidate.id)"
    }
    $candidateIds[$candidate.id] = $true

    if (-not $candidate.runtimeModelId -or -not $candidate.licenceUrl) {
        throw "candidate metadata incomplete: $($candidate.id)"
    }

    foreach ($file in $candidate.files) {
        if ($file.repositoryRevision -notmatch '^[0-9a-f]{40}$') {
            throw "unpinned repository revision: $($candidate.id)"
        }
        if ($file.sha256 -notmatch '^[0-9a-f]{64}$') {
            throw "invalid SHA-256: $($candidate.id)"
        }
        if ([uint64]$file.bytes -eq 0) {
            throw "zero byte size: $($candidate.id)"
        }
        if ($file.url -notmatch [regex]::Escape("/resolve/$($file.repositoryRevision)/")) {
            throw "URL does not contain pinned revision: $($candidate.id)"
        }
    }
}

if ($manifest.selection.status -ne 'selected' -and $manifest.packs.Count -ne 0) {
    throw 'installable packs must remain empty before selection'
}

Write-Output "verified $($manifest.candidates.Count) pinned candidates; selection=$($manifest.selection.status)"
