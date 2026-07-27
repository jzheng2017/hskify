[CmdletBinding()]
param(
    [switch] $ValidateOnly,
    [switch] $SyncModelsOnly,
    [switch] $SmokeTest,
    [int] $NativeDebounceMilliseconds = 800
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') {
    throw 'Firefox development hot reload currently requires Windows.'
}
if ($NativeDebounceMilliseconds -lt 250) {
    throw 'NativeDebounceMilliseconds must be at least 250.'
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$extensionRoot = Join-Path $repositoryRoot 'extensions\firefox'
$performanceBuild = Join-Path $PSScriptRoot 'Invoke-PerformanceBuild.ps1'
$attestationPath = Join-Path $repositoryRoot 'target\release\hskify-performance-build-attestation.json'
$nativeHostPath = Join-Path $repositoryRoot 'target\release\hsk-manga-native-host.exe'
$nativeDaemonPath = Join-Path $repositoryRoot 'target\release\hsk-manga-browser-daemon.exe'
$registerScript = Join-Path $repositoryRoot 'installers\windows\native-host-registration\Register-NativeHost.ps1'
$unregisterScript = Join-Path $repositoryRoot 'installers\windows\native-host-registration\Unregister-NativeHost.ps1'
$installedRoot = Join-Path $env:LOCALAPPDATA 'Hskify'
$readinessMarker = Join-Path $installedRoot 'browser-companion\browser-cache\browser-runtime\models.ready'
$nativeManifest = Join-Path $installedRoot 'native-host\local.hskify.hsk_manga.json'
$productionNativeHost = Join-Path $installedRoot 'app\companion\hsk-manga-native-host.exe'
$developmentRecoveryRoot = Join-Path $installedRoot 'development-recovery'
$registrationBackup = Join-Path $developmentRecoveryRoot 'native-host-path.txt'
$readinessBackup = Join-Path $developmentRecoveryRoot 'models.ready'
$modelManifest = Join-Path $repositoryRoot 'data\model-packs\manifest.v1.json'
$wxtModule = Join-Path $extensionRoot 'node_modules\wxt\bin\wxt.mjs'

function Assert-DevelopmentInputs {
    foreach ($path in @(
        $extensionRoot,
        $performanceBuild,
        $registerScript,
        $unregisterScript,
        $modelManifest,
        $wxtModule
    )) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "development input is missing: $path"
        }
    }
    if (-not (Get-Command 'node.exe' -ErrorAction SilentlyContinue)) {
        throw 'Node.js is required for Firefox development hot reload.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $installedRoot 'resources') -PathType Container)) {
        throw 'Install one complete Hskify release before starting development hot reload.'
    }
}

function Assert-BuildFingerprintAgreement {
    if (-not (Test-Path -LiteralPath $attestationPath -PathType Leaf)) {
        throw "performance-build attestation is missing: $attestationPath"
    }
    $attestation = Get-Content -LiteralPath $attestationPath -Raw | ConvertFrom-Json
    $fingerprint = [string] $attestation.buildFingerprint
    if ([string]::IsNullOrWhiteSpace($fingerprint)) {
        throw 'performance-build attestation has no build fingerprint.'
    }
    $rustContract = Get-Content `
        -LiteralPath (Join-Path $repositoryRoot 'crates\browser-companion\src\contracts.rs') `
        -Raw
    $typescriptContract = Get-Content `
        -LiteralPath (Join-Path $extensionRoot 'src\contracts\browser.ts') `
        -Raw
    $attestationSource = Get-Content `
        -LiteralPath (Join-Path $PSScriptRoot 'PerformanceBuildAttestation.ps1') `
        -Raw
    foreach ($source in @($rustContract, $typescriptContract, $attestationSource)) {
        if ($source.IndexOf($fingerprint, [StringComparison]::Ordinal) -lt 0) {
            throw "development build fingerprint is not synchronized: $fingerprint"
        }
    }
    return $fingerprint
}

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string] $Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Sync-DevelopmentModelResources {
    $manifest = Get-Content -LiteralPath $modelManifest -Raw | ConvertFrom-Json
    $resources = Join-Path $installedRoot 'resources'
    $client = [Net.WebClient]::new()
    try {
        foreach ($identity in @($manifest.resourceIdentities)) {
            $destination = if ([string] $identity.id -ceq 'translation-model') {
                Join-Path $resources 'models\Qwen3.5-4B-Q4_K_M.gguf'
            }
            else {
                Join-Path $resources "models\resident\$($identity.id)\$($identity.filename)"
            }
            $ready = (
                (Test-Path -LiteralPath $destination -PathType Leaf) -and
                [uint64] (Get-Item -LiteralPath $destination).Length -eq [uint64] $identity.bytes
            )
            if ($ready) {
                continue
            }

            $directory = Split-Path -Parent $destination
            [IO.Directory]::CreateDirectory($directory) | Out-Null
            $temporary = Join-Path $directory ".$($identity.filename).download"
            if (Test-Path -LiteralPath $temporary) {
                Remove-Item -LiteralPath $temporary -Force
            }
            Write-Host "[Hskify dev] Updating $($identity.id)..."
            try {
                $client.DownloadFile([string] $identity.url, $temporary)
                if (
                    [uint64] (Get-Item -LiteralPath $temporary).Length -ne [uint64] $identity.bytes -or
                    (Get-FileSha256 -Path $temporary) -cne [string] $identity.sha256
                ) {
                    throw "downloaded model identity did not match the pinned manifest: $($identity.id)"
                }
                Move-Item -LiteralPath $temporary -Destination $destination -Force
            }
            finally {
                if (Test-Path -LiteralPath $temporary) {
                    Remove-Item -LiteralPath $temporary -Force
                }
            }
        }
    }
    finally {
        $client.Dispose()
    }
}

function Stop-HskifyProcessesFrom {
    param([Parameter(Mandatory = $true)][string[]] $CompanionDirectories)

    $roots = @(
        $CompanionDirectories |
            ForEach-Object {
                [IO.Path]::GetFullPath($_).TrimEnd('\', '/') +
                    [IO.Path]::DirectorySeparatorChar
            }
    )
    foreach ($name in @('hsk-manga-native-host', 'hsk-manga-browser-daemon')) {
        foreach ($process in @(Get-Process -Name $name -ErrorAction SilentlyContinue)) {
            $path = try { $process.Path } catch { $null }
            if ([string]::IsNullOrWhiteSpace($path)) {
                continue
            }
            $resolved = [IO.Path]::GetFullPath($path)
            if ($roots.Where({
                $resolved.StartsWith($_, [StringComparison]::OrdinalIgnoreCase)
            }).Count -gt 0) {
                Stop-Process -Id $process.Id -Force -ErrorAction Stop
                $null = $process.WaitForExit(5000)
            }
        }
    }
}

function Write-DevelopmentReadinessMarker {
    param([Parameter(Mandatory = $true)][string] $BuildFingerprint)

    $manifest = Get-Content -LiteralPath $modelManifest -Raw | ConvertFrom-Json
    $resourceIdentities = @(
        foreach ($identity in @($manifest.resourceIdentities)) {
            [ordered]@{
                id = [string] $identity.id
                repository = [string] $identity.repository
                repositoryRevision = [string] $identity.repositoryRevision
                filename = [string] $identity.filename
                bytes = [uint64] $identity.bytes
                sha256 = [string] $identity.sha256
            }
        }
    )
    $resources = Join-Path $installedRoot 'resources'
    $installations = @(
        foreach ($identity in @($manifest.resourceIdentities)) {
            [ordered]@{
                id = [string] $identity.id
                path = if ([string] $identity.id -ceq 'translation-model') {
                    Join-Path $resources 'models\Qwen3.5-4B-Q4_K_M.gguf'
                }
                else {
                    Join-Path $resources "models\resident\$($identity.id)\$($identity.filename)"
                }
            }
        }
    )
    $marker = [ordered]@{
        buildFingerprint = $BuildFingerprint
        resourceIdentities = $resourceIdentities
        installations = $installations
    }
    $markerDirectory = Split-Path -Parent $readinessMarker
    [IO.Directory]::CreateDirectory($markerDirectory) | Out-Null
    [IO.File]::WriteAllText(
        $readinessMarker,
        ($marker | ConvertTo-Json -Depth 8 -Compress),
        [Text.UTF8Encoding]::new($false)
    )
}

function Invoke-NativeHotReload {
    Write-Host '[Hskify dev] Rebuilding the native companion...'
    Stop-HskifyProcessesFrom -CompanionDirectories @(
        (Split-Path -Parent $nativeHostPath),
        (Join-Path $installedRoot 'app\companion')
    )
    & $performanceBuild
    if ($LASTEXITCODE -ne 0) {
        throw "native performance build failed with exit code $LASTEXITCODE"
    }
    foreach ($path in @($nativeHostPath, $nativeDaemonPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "native performance build did not create $path"
        }
    }
    $fingerprint = Assert-BuildFingerprintAgreement
    Sync-DevelopmentModelResources
    Write-DevelopmentReadinessMarker -BuildFingerprint $fingerprint
    & $registerScript -NativeHostPath $nativeHostPath | Out-Null
    Write-Host "[Hskify dev] Native companion reloaded: $fingerprint"
}

function New-SourceWatcher {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [Parameter(Mandatory = $true)][bool] $IncludeSubdirectories,
        [Parameter(Mandatory = $true)][string] $Identifier
    )

    $watcher = [IO.FileSystemWatcher]::new($Path)
    $watcher.IncludeSubdirectories = $IncludeSubdirectories
    $watcher.NotifyFilter =
        [IO.NotifyFilters]::FileName -bor
        [IO.NotifyFilters]::DirectoryName -bor
        [IO.NotifyFilters]::LastWrite
    $watcher.EnableRaisingEvents = $true
    foreach ($eventName in @('Changed', 'Created', 'Deleted', 'Renamed')) {
        Register-ObjectEvent `
            -InputObject $watcher `
            -EventName $eventName `
            -SourceIdentifier "$Identifier.$eventName" | Out-Null
    }
    return $watcher
}

function Test-NativeSourceEvent {
    param([Parameter(Mandatory = $true)][System.Management.Automation.PSEventArgs] $Event)

    $path = [string] $Event.SourceEventArgs.FullPath
    $extension = [IO.Path]::GetExtension($path)
    if ($path.StartsWith((Join-Path $repositoryRoot 'crates'), [StringComparison]::OrdinalIgnoreCase)) {
        return $extension -in @('.rs', '.toml')
    }
    if ($path.StartsWith((Join-Path $repositoryRoot 'scripts'), [StringComparison]::OrdinalIgnoreCase)) {
        return [IO.Path]::GetFileName($path) -in @(
            'Invoke-PerformanceBuild.ps1',
            'PerformanceBuildAttestation.ps1'
        )
    }
    if ($path.StartsWith((Join-Path $repositoryRoot 'data\model-packs'), [StringComparison]::OrdinalIgnoreCase)) {
        return $extension -eq '.json'
    }
    return [IO.Path]::GetFileName($path) -in @('Cargo.toml', 'Cargo.lock')
}

function Stop-StartedProcessTree {
    param([Parameter(Mandatory = $true)][int] $RootProcessId)

    $all = @(Get-CimInstance Win32_Process)
    $pending = [Collections.Generic.Queue[int]]::new()
    $pending.Enqueue($RootProcessId)
    $ids = [Collections.Generic.List[int]]::new()
    while ($pending.Count -gt 0) {
        $parent = $pending.Dequeue()
        $ids.Add($parent)
        foreach ($child in @($all | Where-Object ParentProcessId -eq $parent)) {
            $pending.Enqueue([int] $child.ProcessId)
        }
    }
    for ($index = $ids.Count - 1; $index -ge 0; $index -= 1) {
        Stop-Process -Id $ids[$index] -Force -ErrorAction SilentlyContinue
    }
}

function Stop-DevelopmentFirefox {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [Collections.Generic.HashSet[int]] $ExistingProcessIds
    )

    $temporaryProfilePrefix = (
        [IO.Path]::GetFullPath((Join-Path $env:TEMP 'firefox-profile'))
    )
    foreach ($process in @(Get-CimInstance Win32_Process -Filter "Name = 'firefox.exe'")) {
        if (
            -not $ExistingProcessIds.Contains([int] $process.ProcessId) -and
            $process.CommandLine -like '*-start-debugger-server*' -and
            $process.CommandLine.IndexOf(
                $temporaryProfilePrefix,
                [StringComparison]::OrdinalIgnoreCase
            ) -ge 0
        ) {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}

Assert-DevelopmentInputs
if ($SyncModelsOnly) {
    Sync-DevelopmentModelResources
    Write-Output 'Firefox development model resources are current.'
    return
}
if ($ValidateOnly) {
    Assert-BuildFingerprintAgreement | Out-Null
    Write-Output 'Firefox development hot-reload inputs are valid.'
    return
}

$originalManifest = if (Test-Path -LiteralPath $nativeManifest -PathType Leaf) {
    Get-Content -LiteralPath $nativeManifest -Raw | ConvertFrom-Json
}
else {
    $null
}
$originalReadiness = if (Test-Path -LiteralPath $readinessMarker -PathType Leaf) {
    Get-Content -LiteralPath $readinessMarker -Raw
}
else {
    $null
}
$originalNativeHost = if (
    $null -ne $originalManifest -and
    (Test-Path -LiteralPath ([string] $originalManifest.path) -PathType Leaf)
) {
    [IO.Path]::GetFullPath([string] $originalManifest.path)
}
else {
    $null
}
[IO.Directory]::CreateDirectory($developmentRecoveryRoot) | Out-Null
if (
    $null -ne $originalNativeHost -and
    $originalNativeHost -ceq [IO.Path]::GetFullPath($productionNativeHost)
) {
    if (-not (Test-Path -LiteralPath $registrationBackup -PathType Leaf)) {
        [IO.File]::WriteAllText(
            $registrationBackup,
            $originalNativeHost,
            [Text.UTF8Encoding]::new($false)
        )
    }
    if (
        $null -ne $originalReadiness -and
        -not (Test-Path -LiteralPath $readinessBackup -PathType Leaf)
    ) {
        [IO.File]::WriteAllText(
            $readinessBackup,
            $originalReadiness,
            [Text.UTF8Encoding]::new($false)
        )
    }
}
$watchers = [Collections.Generic.List[IO.FileSystemWatcher]]::new()
$wxtProcess = $null
$firefoxPidsBefore = [Collections.Generic.HashSet[int]]::new()
foreach ($process in @(Get-Process -Name 'firefox' -ErrorAction SilentlyContinue)) {
    $null = $firefoxPidsBefore.Add([int] $process.Id)
}

try {
    Invoke-NativeHotReload

    $watchers.Add((New-SourceWatcher -Path (Join-Path $repositoryRoot 'crates') -IncludeSubdirectories $true -Identifier 'HskifyCrates'))
    $watchers.Add((New-SourceWatcher -Path $repositoryRoot -IncludeSubdirectories $false -Identifier 'HskifyCargo'))
    $watchers.Add((New-SourceWatcher -Path (Join-Path $repositoryRoot 'data\model-packs') -IncludeSubdirectories $true -Identifier 'HskifyModels'))
    $watchers.Add((New-SourceWatcher -Path $PSScriptRoot -IncludeSubdirectories $false -Identifier 'HskifyBuildScripts'))

    $node = (Get-Command 'node.exe').Source
    $wxtProcess = Start-Process `
        -FilePath $node `
        -ArgumentList @($wxtModule, '-b', 'firefox') `
        -WorkingDirectory $extensionRoot `
        -NoNewWindow `
        -PassThru
    Write-Host '[Hskify dev] Firefox extension hot reload is running. Press Ctrl+C to stop.'
    if ($SmokeTest) {
        $deadline = [DateTime]::UtcNow.AddSeconds(8)
        while (-not $wxtProcess.HasExited -and [DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Milliseconds 250
            $wxtProcess.Refresh()
        }
        if ($wxtProcess.HasExited) {
            throw "Firefox hot reload exited during startup (exit code $($wxtProcess.ExitCode))."
        }
        Write-Output 'Firefox development hot reload passed its startup smoke test.'
        return
    }

    $nativeChangeAt = $null
    while (-not $wxtProcess.HasExited) {
        $event = Wait-Event -Timeout 1
        if ($null -ne $event) {
            foreach ($queued in @(Get-Event)) {
                if (Test-NativeSourceEvent -Event $queued) {
                    $nativeChangeAt = [DateTime]::UtcNow
                }
                Remove-Event -EventIdentifier $queued.EventIdentifier
            }
        }
        if (
            $null -ne $nativeChangeAt -and
            ([DateTime]::UtcNow - $nativeChangeAt).TotalMilliseconds -ge
                $NativeDebounceMilliseconds
        ) {
            try {
                Invoke-NativeHotReload
            }
            catch {
                Write-Error -ErrorRecord $_ -ErrorAction Continue
            }
            $nativeChangeAt = $null
        }
        $wxtProcess.Refresh()
    }
    if ($wxtProcess.ExitCode -ne 0) {
        $exitCode = if ($null -eq $wxtProcess.ExitCode) {
            'unknown'
        }
        else {
            [string] $wxtProcess.ExitCode
        }
        throw "Firefox hot reload stopped unexpectedly (exit code $exitCode)."
    }
}
finally {
    foreach ($subscription in @(Get-EventSubscriber | Where-Object {
        $_.SourceIdentifier -like 'Hskify*'
    })) {
        Unregister-Event -SubscriptionId $subscription.SubscriptionId -ErrorAction SilentlyContinue
    }
    foreach ($watcher in $watchers) {
        $watcher.Dispose()
    }
    if ($null -ne $wxtProcess -and -not $wxtProcess.HasExited) {
        Stop-StartedProcessTree -RootProcessId $wxtProcess.Id
    }
    Stop-DevelopmentFirefox -ExistingProcessIds $firefoxPidsBefore
    Stop-HskifyProcessesFrom -CompanionDirectories @((Split-Path -Parent $nativeHostPath))

    $readinessToRestore = if (Test-Path -LiteralPath $readinessBackup -PathType Leaf) {
        Get-Content -LiteralPath $readinessBackup -Raw
    }
    else {
        $originalReadiness
    }
    if ($null -ne $readinessToRestore) {
        [IO.File]::WriteAllText(
            $readinessMarker,
            $readinessToRestore,
            [Text.UTF8Encoding]::new($false)
        )
    }
    $nativeHostToRestore = if (Test-Path -LiteralPath $registrationBackup -PathType Leaf) {
        (Get-Content -LiteralPath $registrationBackup -Raw).Trim()
    }
    elseif (Test-Path -LiteralPath $productionNativeHost -PathType Leaf) {
        $productionNativeHost
    }
    else {
        $originalNativeHost
    }
    if (
        -not [string]::IsNullOrWhiteSpace($nativeHostToRestore) -and
        (Test-Path -LiteralPath $nativeHostToRestore -PathType Leaf)
    ) {
        & $registerScript -NativeHostPath $nativeHostToRestore | Out-Null
    }
    else {
        & $unregisterScript | Out-Null
    }
    foreach ($backup in @($registrationBackup, $readinessBackup)) {
        if (Test-Path -LiteralPath $backup -PathType Leaf) {
            Remove-Item -LiteralPath $backup -Force
        }
    }
    Write-Host '[Hskify dev] Development registration stopped.'
}
