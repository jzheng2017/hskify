Set-StrictMode -Version Latest

$script:HskifyPerformanceAttestationSchema = 'hskify.performance-build-attestation.v1'
$script:HskifyPerformanceBuildFingerprint = 'hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-26-r2'
$script:HskifyPerformanceTarget = 'x86_64-pc-windows-msvc'
$script:HskifyPerformanceGpuName = 'NVIDIA GeForce RTX 4080 SUPER'
$script:HskifyPerformanceGpuMemoryMiB = 16376
$script:HskifyPerformanceComputeCapability = '8.9'
$script:HskifyPerformanceDriverApi = '13.1'
$script:HskifyPerformanceToolkitVersion = '13.1'
$script:HskifyPerformanceNvccVersion = '13.1.80'
$script:HskifyPerformanceOrtCudaVersion = '13'
$script:HskifyPerformanceCudaArchitecture = 'sm_89'
$script:HskifyPerformanceLlamaCppTag = 'b8935'

function Get-HskifySha256Text {
    param([Parameter(Mandatory = $true)][string] $Text)

    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
        return ([BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-HskifyFileIdentity {
    param([Parameter(Mandatory = $true)][string] $Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "attested binary is missing: $Path"
    }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    return [ordered]@{
        path = $resolved
        bytes = [int64] (Get-Item -LiteralPath $resolved).Length
        sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Invoke-HskifyVersionCommand {
    param(
        [Parameter(Mandatory = $true)][string] $FilePath,
        [Parameter(Mandatory = $true)][string[]] $Arguments
    )

    $output = @(& $FilePath @Arguments 2>&1 | ForEach-Object { [string] $_ })
    if ($LASTEXITCODE -ne 0) {
        throw "version command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
    $text = ($output -join "`n").Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        throw "version command returned no output: $FilePath"
    }
    return $text
}

function Get-HskifyPerformanceGpuIdentity {
    $smi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $smi) {
        throw 'Hskify requires NVIDIA CUDA, but nvidia-smi.exe is unavailable'
    }

    $row = & $smi.Source `
        '--id=0' `
        '--query-gpu=name,memory.total,compute_cap,driver_version' `
        '--format=csv,noheader,nounits'
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($row)) {
        throw 'Hskify could not query CUDA device 0'
    }
    $rows = @($row)
    if ($rows.Count -ne 1) {
        throw "Hskify requires exactly one result for CUDA device 0; found $($rows.Count)"
    }
    $parts = @(([string] $rows[0]).Split(',') | ForEach-Object { $_.Trim() })
    if ($parts.Count -ne 4) {
        throw "unexpected CUDA device identity response: $row"
    }

    $summary = (& $smi.Source '--id=0' 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw 'Hskify could not query the NVIDIA CUDA driver API version'
    }
    $driverApiMatch = [regex]::Match($summary, 'CUDA Version:\s*(\d+\.\d+)')
    if (-not $driverApiMatch.Success) {
        throw 'Hskify could not parse the NVIDIA CUDA driver API version'
    }

    return [ordered]@{
        gpuIndex = 0
        name = $parts[0]
        memoryTotalMiB = [int] $parts[1]
        computeCapability = $parts[2]
        driverVersion = $parts[3]
        driverApiVersion = $driverApiMatch.Groups[1].Value
    }
}

function Assert-HskifyExactPerformanceGpu {
    param([Parameter(Mandatory = $true)] $Gpu)

    if (
        [int] $Gpu.gpuIndex -ne 0 -or
        [string] $Gpu.name -cne $script:HskifyPerformanceGpuName -or
        [int] $Gpu.memoryTotalMiB -ne $script:HskifyPerformanceGpuMemoryMiB -or
        [string] $Gpu.computeCapability -cne $script:HskifyPerformanceComputeCapability -or
        [string] $Gpu.driverApiVersion -cne $script:HskifyPerformanceDriverApi -or
        [string] $Gpu.driverVersion -notmatch '^\d+\.\d+$'
    ) {
        throw (
            'Hskify performance release requires CUDA device 0 to be exactly ' +
            "$($script:HskifyPerformanceGpuName), $($script:HskifyPerformanceGpuMemoryMiB) MiB, " +
            "compute $($script:HskifyPerformanceComputeCapability), driver API " +
            "$($script:HskifyPerformanceDriverApi); found $($Gpu.name), " +
            "$($Gpu.memoryTotalMiB) MiB, compute $($Gpu.computeCapability), " +
            "driver $($Gpu.driverVersion), API $($Gpu.driverApiVersion)"
        )
    }
}

function Get-HskifyPerformanceSourceTreeIdentity {
    param([Parameter(Mandatory = $true)][string] $RepositoryRoot)

    $root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
    $git = Get-Command 'git.exe' -ErrorAction SilentlyContinue
    if ($null -eq $git) {
        throw 'git.exe is required to attest the performance-build source tree'
    }

    $gitCommit = (& $git.Source -C $root rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $gitCommit -notmatch '^[0-9a-f]{40}$') {
        throw 'could not resolve the source-tree Git commit'
    }
    $gitTree = (& $git.Source -C $root rev-parse 'HEAD^{tree}').Trim()
    if ($LASTEXITCODE -ne 0 -or $gitTree -notmatch '^[0-9a-f]{40}$') {
        throw 'could not resolve the source-tree Git tree'
    }

    $tracked = @(& $git.Source -C $root -c core.quotepath=false ls-files)
    if ($LASTEXITCODE -ne 0) {
        throw 'could not enumerate tracked source files'
    }
    $untracked = @(& $git.Source -C $root -c core.quotepath=false ls-files --others --exclude-standard)
    if ($LASTEXITCODE -ne 0) {
        throw 'could not enumerate untracked nonignored source files'
    }
    $trackedStatus = @(& $git.Source -C $root -c core.quotepath=false status --short --untracked-files=no)
    if ($LASTEXITCODE -ne 0) {
        throw 'could not enumerate tracked source modifications'
    }

    [Array]::Sort([string[]] $tracked, [StringComparer]::Ordinal)
    [Array]::Sort([string[]] $untracked, [StringComparer]::Ordinal)
    [Array]::Sort([string[]] $trackedStatus, [StringComparer]::Ordinal)

    $canonical = [Text.StringBuilder]::new()
    $untrackedIdentities = [System.Collections.Generic.List[object]]::new()
    foreach ($entry in @(
        @($tracked | ForEach-Object { [ordered]@{ kind = 'tracked'; path = [string] $_ } })
        @($untracked | ForEach-Object { [ordered]@{ kind = 'untracked-nonignored'; path = [string] $_ } })
    )) {
        $relativePath = ([string] $entry.path).Replace('\', '/')
        if (
            [string]::IsNullOrWhiteSpace($relativePath) -or
            $relativePath.IndexOfAny([char[]] "`0`r`n") -ge 0
        ) {
            throw 'source-tree paths must be nonempty and cannot contain NUL or newlines'
        }
        $absolutePath = Join-Path $root ($relativePath.Replace('/', '\'))
        if (Test-Path -LiteralPath $absolutePath -PathType Leaf) {
            $file = Get-Item -LiteralPath $absolutePath
            $state = 'file'
            $bytes = [int64] $file.Length
            $sha256 = (Get-FileHash -LiteralPath $absolutePath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        elseif (Test-Path -LiteralPath $absolutePath) {
            throw "source-tree entry is not a regular file: $relativePath"
        }
        else {
            if ($entry.kind -ne 'tracked') {
                throw "untracked source file disappeared during identity capture: $relativePath"
            }
            $state = 'missing'
            $bytes = [int64] 0
            $sha256 = ''
        }
        [void] $canonical.Append($entry.kind)
        [void] $canonical.Append("`0")
        [void] $canonical.Append($relativePath)
        [void] $canonical.Append("`0")
        [void] $canonical.Append($state)
        [void] $canonical.Append("`0")
        [void] $canonical.Append($bytes)
        [void] $canonical.Append("`0")
        [void] $canonical.Append($sha256)
        [void] $canonical.Append("`n")

        if ($entry.kind -eq 'untracked-nonignored') {
            $untrackedIdentities.Add([ordered]@{
                path = $relativePath
                bytes = $bytes
                sha256 = $sha256
            })
        }
    }

    return [ordered]@{
        algorithm = 'sha256-path-kind-state-bytes-content-v1'
        gitCommit = $gitCommit
        gitTree = $gitTree
        aggregateSha256 = Get-HskifySha256Text -Text $canonical.ToString()
        trackedFileCount = [int] $tracked.Count
        trackedModificationStatus = @($trackedStatus)
        untrackedNonignoredFiles = @($untrackedIdentities)
    }
}

function Get-HskifyMsvcCompilerPath {
    $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw 'Visual Studio Build Tools with the MSVC x64 compiler are required'
    }
    $installation = & $vswhere `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installation)) {
        throw 'Visual Studio Build Tools with the MSVC x64 compiler are required'
    }
    $compilers = @(
        Get-ChildItem -LiteralPath (Join-Path $installation 'VC\Tools\MSVC') -Recurse -Filter 'cl.exe' |
            Where-Object { $_.FullName -match '\\bin\\Hostx64\\x64\\cl\.exe$' } |
            Sort-Object FullName -Descending
    )
    if ($compilers.Count -eq 0) {
        throw 'MSVC x64 cl.exe was not found'
    }
    return $compilers[0].FullName
}

function Get-HskifyPerformanceToolchainEvidence {
    param(
        [Parameter(Mandatory = $true)][string] $CmakePath,
        [Parameter(Mandatory = $true)][string] $NvccPath,
        [Parameter(Mandatory = $true)][string] $MsvcPath
    )

    foreach ($path in @($CmakePath, $NvccPath, $MsvcPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "attested toolchain executable is missing: $path"
        }
    }
    $cargo = Get-Command 'cargo.exe' -ErrorAction SilentlyContinue
    $rustc = Get-Command 'rustc.exe' -ErrorAction SilentlyContinue
    if ($null -eq $cargo -or $null -eq $rustc) {
        throw 'cargo.exe and rustc.exe are required for the performance build'
    }

    $cmakeText = Invoke-HskifyVersionCommand -FilePath $CmakePath -Arguments @('--version')
    $cmakeMatch = [regex]::Match($cmakeText, '(?m)^cmake version (\d+\.\d+\.\d+)$')
    if (-not $cmakeMatch.Success -or $cmakeMatch.Groups[1].Value -cne '4.4.0') {
        throw "performance build requires CMake 4.4.0; found: $cmakeText"
    }

    $nvccText = Invoke-HskifyVersionCommand -FilePath $NvccPath -Arguments @('--version')
    $nvccMatch = [regex]::Match($nvccText, '\bV(\d+\.\d+\.\d+)\b')
    if (
        -not $nvccMatch.Success -or
        $nvccMatch.Groups[1].Value -cne $script:HskifyPerformanceNvccVersion
    ) {
        throw "performance build requires nvcc $($script:HskifyPerformanceNvccVersion); found: $nvccText"
    }

    $cargoText = Invoke-HskifyVersionCommand -FilePath $cargo.Source -Arguments @('--version')
    $cargoMatch = [regex]::Match($cargoText, '^cargo ([^\s]+)')
    $rustcText = Invoke-HskifyVersionCommand -FilePath $rustc.Source -Arguments @('--version')
    $rustcMatch = [regex]::Match($rustcText, '^rustc ([^\s]+)')
    if (-not $cargoMatch.Success -or -not $rustcMatch.Success) {
        throw 'could not parse cargo or rustc version output'
    }

    $msvcFile = Get-Item -LiteralPath $MsvcPath
    $msvcVersion = [string] $msvcFile.VersionInfo.FileVersion
    if ([string]::IsNullOrWhiteSpace($msvcVersion)) {
        throw "could not resolve the MSVC compiler version: $MsvcPath"
    }

    return [ordered]@{
        msvc = [ordered]@{
            path = (Resolve-Path -LiteralPath $MsvcPath).Path
            version = $msvcVersion
        }
        cmake = [ordered]@{
            path = (Resolve-Path -LiteralPath $CmakePath).Path
            version = $cmakeMatch.Groups[1].Value
        }
        nvcc = [ordered]@{
            path = (Resolve-Path -LiteralPath $NvccPath).Path
            version = $nvccMatch.Groups[1].Value
        }
        cargo = [ordered]@{
            path = (Resolve-Path -LiteralPath $cargo.Source).Path
            version = $cargoMatch.Groups[1].Value
        }
        rustc = [ordered]@{
            path = (Resolve-Path -LiteralPath $rustc.Source).Path
            version = $rustcMatch.Groups[1].Value
        }
    }
}

function New-HskifyPerformanceBuildAttestation {
    param(
        [Parameter(Mandatory = $true)] $SourceIdentity,
        [Parameter(Mandatory = $true)] $Hardware,
        [Parameter(Mandatory = $true)] $Toolchain,
        [Parameter(Mandatory = $true)][string] $CudaHome,
        [Parameter(Mandatory = $true)][string] $NativeHostPath,
        [Parameter(Mandatory = $true)][string] $BrowserDaemonPath
    )

    $nativeHost = Get-HskifyFileIdentity -Path $NativeHostPath
    $browserDaemon = Get-HskifyFileIdentity -Path $BrowserDaemonPath
    return [ordered]@{
        schema = $script:HskifyPerformanceAttestationSchema
        buildFingerprint = $script:HskifyPerformanceBuildFingerprint
        source = $SourceIdentity
        build = [ordered]@{
            target = $script:HskifyPerformanceTarget
            profile = 'release'
            package = 'browser-companion'
            features = @('cuda')
        }
        cuda = [ordered]@{
            cudaHome = [IO.Path]::GetFullPath($CudaHome)
            toolkitVersion = $script:HskifyPerformanceToolkitVersion
            nvccVersion = $script:HskifyPerformanceNvccVersion
            ortCudaVersion = $script:HskifyPerformanceOrtCudaVersion
            architecture = $script:HskifyPerformanceCudaArchitecture
            computeCapabilityEnvironment = '89'
        }
        hardware = $Hardware
        toolchain = $Toolchain
        llamaCppTag = $script:HskifyPerformanceLlamaCppTag
        builtAtUtc = [DateTimeOffset]::UtcNow.ToString('o')
        binaries = @(
            [ordered]@{
                role = 'native-host'
                fileName = 'hsk-manga-native-host.exe'
                bytes = $nativeHost.bytes
                sha256 = $nativeHost.sha256
            },
            [ordered]@{
                role = 'browser-daemon'
                fileName = 'hsk-manga-browser-daemon.exe'
                bytes = $browserDaemon.bytes
                sha256 = $browserDaemon.sha256
            }
        )
    }
}

function Assert-HskifyExactProperties {
    param(
        [Parameter(Mandatory = $true)] $Value,
        [Parameter(Mandatory = $true)][string[]] $Expected,
        [Parameter(Mandatory = $true)][string] $Label
    )

    if ($null -eq $Value) {
        throw "$Label is missing"
    }
    $actual = @($Value.PSObject.Properties.Name | Sort-Object)
    $expectedSorted = @($Expected | Sort-Object)
    if (($actual -join "`n") -cne ($expectedSorted -join "`n")) {
        throw "$Label fields must be exactly: $($expectedSorted -join ', ')"
    }
}

function Assert-HskifyPerformanceBuildAttestation {
    param(
        [Parameter(Mandatory = $true)][string] $AttestationPath,
        [Parameter(Mandatory = $true)][string] $NativeHostPath,
        [Parameter(Mandatory = $true)][string] $BrowserDaemonPath,
        [string] $RepositoryRoot = '',
        [switch] $VerifyCurrentSourceTree,
        [switch] $VerifyCurrentHardware,
        [switch] $VerifyCurrentToolchain
    )

    if (-not (Test-Path -LiteralPath $AttestationPath -PathType Leaf)) {
        throw "performance-build attestation is missing: $AttestationPath"
    }
    try {
        $attestation = Get-Content -LiteralPath $AttestationPath -Raw -Encoding utf8 | ConvertFrom-Json
    }
    catch {
        throw "performance-build attestation is unreadable: $($_.Exception.Message)"
    }

    Assert-HskifyExactProperties -Value $attestation -Expected @(
        'schema',
        'buildFingerprint',
        'source',
        'build',
        'cuda',
        'hardware',
        'toolchain',
        'llamaCppTag',
        'builtAtUtc',
        'binaries'
    ) -Label 'attestation'
    if ([string] $attestation.schema -cne $script:HskifyPerformanceAttestationSchema) {
        throw "performance-build attestation schema mismatch: $($attestation.schema)"
    }
    if ([string] $attestation.buildFingerprint -cne $script:HskifyPerformanceBuildFingerprint) {
        throw (
            'performance-build fingerprint mismatch; reinstall the extension, native host, and daemon together: ' +
            "$($attestation.buildFingerprint)"
        )
    }

    Assert-HskifyExactProperties -Value $attestation.build -Expected @(
        'target', 'profile', 'package', 'features'
    ) -Label 'attestation.build'
    $features = @($attestation.build.features)
    if (
        [string] $attestation.build.target -cne $script:HskifyPerformanceTarget -or
        [string] $attestation.build.profile -cne 'release' -or
        [string] $attestation.build.package -cne 'browser-companion' -or
        $features.Count -ne 1 -or
        [string] $features[0] -cne 'cuda'
    ) {
        throw 'performance-build attestation does not describe the exact release target with only the required cuda feature'
    }

    Assert-HskifyExactProperties -Value $attestation.cuda -Expected @(
        'cudaHome',
        'toolkitVersion',
        'nvccVersion',
        'ortCudaVersion',
        'architecture',
        'computeCapabilityEnvironment'
    ) -Label 'attestation.cuda'
    if (
        [string]::IsNullOrWhiteSpace([string] $attestation.cuda.cudaHome) -or
        [string] $attestation.cuda.toolkitVersion -cne $script:HskifyPerformanceToolkitVersion -or
        [string] $attestation.cuda.nvccVersion -cne $script:HskifyPerformanceNvccVersion -or
        [string] $attestation.cuda.ortCudaVersion -cne $script:HskifyPerformanceOrtCudaVersion -or
        [string] $attestation.cuda.architecture -cne $script:HskifyPerformanceCudaArchitecture -or
        [string] $attestation.cuda.computeCapabilityEnvironment -cne '89'
    ) {
        throw 'performance-build attestation CUDA claims do not match CUDA 13.1, ORT CUDA 13, and sm_89'
    }

    Assert-HskifyExactProperties -Value $attestation.hardware -Expected @(
        'gpuIndex',
        'name',
        'memoryTotalMiB',
        'computeCapability',
        'driverVersion',
        'driverApiVersion'
    ) -Label 'attestation.hardware'
    Assert-HskifyExactPerformanceGpu -Gpu $attestation.hardware

    Assert-HskifyExactProperties -Value $attestation.source -Expected @(
        'algorithm',
        'gitCommit',
        'gitTree',
        'aggregateSha256',
        'trackedFileCount',
        'trackedModificationStatus',
        'untrackedNonignoredFiles'
    ) -Label 'attestation.source'
    if (
        [string] $attestation.source.algorithm -cne 'sha256-path-kind-state-bytes-content-v1' -or
        [string] $attestation.source.gitCommit -notmatch '^[0-9a-f]{40}$' -or
        [string] $attestation.source.gitTree -notmatch '^[0-9a-f]{40}$' -or
        [string] $attestation.source.aggregateSha256 -notmatch '^[0-9a-f]{64}$' -or
        [int] $attestation.source.trackedFileCount -le 0
    ) {
        throw 'performance-build attestation source-tree identity is invalid'
    }
    foreach ($file in @($attestation.source.untrackedNonignoredFiles)) {
        Assert-HskifyExactProperties -Value $file -Expected @('path', 'bytes', 'sha256') -Label 'attestation.source.untrackedNonignoredFiles[]'
        if (
            [string]::IsNullOrWhiteSpace([string] $file.path) -or
            [int64] $file.bytes -lt 0 -or
            [string] $file.sha256 -notmatch '^[0-9a-f]{64}$'
        ) {
            throw "invalid untracked source identity in attestation: $($file.path)"
        }
    }

    Assert-HskifyExactProperties -Value $attestation.toolchain -Expected @(
        'msvc', 'cmake', 'nvcc', 'cargo', 'rustc'
    ) -Label 'attestation.toolchain'
    foreach ($toolName in @('msvc', 'cmake', 'nvcc', 'cargo', 'rustc')) {
        $tool = $attestation.toolchain.$toolName
        Assert-HskifyExactProperties -Value $tool -Expected @('path', 'version') -Label "attestation.toolchain.$toolName"
        if (
            [string]::IsNullOrWhiteSpace([string] $tool.path) -or
            [string]::IsNullOrWhiteSpace([string] $tool.version)
        ) {
            throw "performance-build attestation has an incomplete $toolName toolchain claim"
        }
    }
    if (
        [string] $attestation.toolchain.cmake.version -cne '4.4.0' -or
        [string] $attestation.toolchain.nvcc.version -cne $script:HskifyPerformanceNvccVersion
    ) {
        throw 'performance-build attestation has the wrong pinned CMake or nvcc version'
    }
    if ([string] $attestation.llamaCppTag -cne $script:HskifyPerformanceLlamaCppTag) {
        throw "performance-build attestation has the wrong llama.cpp tag: $($attestation.llamaCppTag)"
    }
    $builtAt = [DateTimeOffset]::MinValue
    if (
        -not [DateTimeOffset]::TryParse(
            [string] $attestation.builtAtUtc,
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::RoundtripKind,
            [ref] $builtAt
        ) -or
        $builtAt.Offset -ne [TimeSpan]::Zero
    ) {
        throw 'performance-build attestation builtAtUtc must be an exact UTC timestamp'
    }

    $expectedBinaries = @(
        [ordered]@{
            role = 'native-host'
            fileName = 'hsk-manga-native-host.exe'
            path = $NativeHostPath
        },
        [ordered]@{
            role = 'browser-daemon'
            fileName = 'hsk-manga-browser-daemon.exe'
            path = $BrowserDaemonPath
        }
    )
    $binaries = @($attestation.binaries)
    if ($binaries.Count -ne $expectedBinaries.Count) {
        throw 'performance-build attestation must contain exactly two binary identities'
    }
    for ($index = 0; $index -lt $expectedBinaries.Count; $index += 1) {
        $claim = $binaries[$index]
        $expected = $expectedBinaries[$index]
        Assert-HskifyExactProperties -Value $claim -Expected @(
            'role', 'fileName', 'bytes', 'sha256'
        ) -Label "attestation.binaries[$index]"
        if (
            [string] $claim.role -cne $expected.role -or
            [string] $claim.fileName -cne $expected.fileName -or
            [int64] $claim.bytes -le 0 -or
            [string] $claim.sha256 -notmatch '^[0-9a-f]{64}$'
        ) {
            throw "invalid performance-build binary claim for $($expected.role)"
        }
        $actual = Get-HskifyFileIdentity -Path $expected.path
        if (
            [int64] $claim.bytes -ne [int64] $actual.bytes -or
            [string] $claim.sha256 -cne [string] $actual.sha256
        ) {
            throw (
                "performance-build binary identity mismatch for $($expected.role): " +
                "attested $($claim.bytes) bytes/$($claim.sha256), " +
                "actual $($actual.bytes) bytes/$($actual.sha256)"
            )
        }
    }

    if ($VerifyCurrentSourceTree) {
        if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
            throw 'RepositoryRoot is required when verifying the attested source tree'
        }
        $currentSource = Get-HskifyPerformanceSourceTreeIdentity -RepositoryRoot $RepositoryRoot
        if (
            [string] $currentSource.gitCommit -cne [string] $attestation.source.gitCommit -or
            [string] $currentSource.gitTree -cne [string] $attestation.source.gitTree -or
            [string] $currentSource.aggregateSha256 -cne [string] $attestation.source.aggregateSha256
        ) {
            throw 'current source tree does not match the performance-build attestation; rebuild before packaging'
        }
    }

    if ($VerifyCurrentHardware) {
        $currentHardware = Get-HskifyPerformanceGpuIdentity
        Assert-HskifyExactPerformanceGpu -Gpu $currentHardware
        foreach ($field in @(
            'gpuIndex',
            'name',
            'memoryTotalMiB',
            'computeCapability',
            'driverVersion',
            'driverApiVersion'
        )) {
            if ([string] $currentHardware.$field -cne [string] $attestation.hardware.$field) {
                throw "current hardware field $field does not match the performance-build attestation"
            }
        }
    }

    if ($VerifyCurrentToolchain) {
        $currentToolchain = Get-HskifyPerformanceToolchainEvidence `
            -CmakePath ([string] $attestation.toolchain.cmake.path) `
            -NvccPath ([string] $attestation.toolchain.nvcc.path) `
            -MsvcPath ([string] $attestation.toolchain.msvc.path)
        foreach ($toolName in @('msvc', 'cmake', 'nvcc', 'cargo', 'rustc')) {
            foreach ($field in @('path', 'version')) {
                if (
                    [string] $currentToolchain.$toolName.$field -cne
                    [string] $attestation.toolchain.$toolName.$field
                ) {
                    throw "current $toolName $field does not match the performance-build attestation"
                }
            }
        }
    }

    return $attestation
}
