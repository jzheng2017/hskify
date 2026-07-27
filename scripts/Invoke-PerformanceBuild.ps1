[CmdletBinding()]
param(
    [string] $AttestationPath = '',
    [switch] $PrerequisitesOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'PerformanceBuildAttestation.ps1')

$toolRoot = Join-Path $repositoryRoot '.cache\tools'
$cmakeVersion = '4.4.0'
$cmakeArchiveName = "cmake-$cmakeVersion-windows-x86_64.zip"
$cmakeDirectory = Join-Path $toolRoot "cmake-$cmakeVersion"
$cmakeExecutable = Join-Path $cmakeDirectory "cmake-$cmakeVersion-windows-x86_64\bin\cmake.exe"
$cudaPythonRoot = Join-Path $toolRoot 'nvidia-python'
$cudaRoot = Join-Path $cudaPythonRoot 'nvidia\cu13'
$cudaPackages = @(
    'nvidia-cuda-nvcc==13.1.80',
    'nvidia-cuda-runtime==13.1.80',
    'nvidia-cuda-crt==13.1.80',
    'nvidia-cuda-cccl==13.1.78',
    'nvidia-nvvm==13.1.80',
    'nvidia-cuda-nvrtc==13.1.80'
)
$libclangRoot = Join-Path $toolRoot 'libclang'
$libclangDll = Join-Path $libclangRoot 'clang\native\libclang.dll'
$libclangPackage = 'libclang==18.1.1'
$targetTriple = 'x86_64-pc-windows-msvc'

if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
    $targetRoot = Join-Path $repositoryRoot 'target'
}
elseif ([IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) {
    $targetRoot = [IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)
}
else {
    $targetRoot = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $env:CARGO_TARGET_DIR))
}
$releaseDirectory = Join-Path $targetRoot 'release'
$nativeHostPath = Join-Path $releaseDirectory 'hsk-manga-native-host.exe'
$browserDaemonPath = Join-Path $releaseDirectory 'hsk-manga-browser-daemon.exe'
if ([string]::IsNullOrWhiteSpace($AttestationPath)) {
    $AttestationPath = Join-Path $releaseDirectory 'hskify-performance-build-attestation.json'
}
elseif (-not [IO.Path]::IsPathRooted($AttestationPath)) {
    $AttestationPath = Join-Path $repositoryRoot $AttestationPath
}
$AttestationPath = [IO.Path]::GetFullPath($AttestationPath)

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string] $FilePath,
        [Parameter(Mandatory = $true)][string[]] $Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Install-PortableCmake {
    if (Test-Path -LiteralPath $cmakeExecutable -PathType Leaf) {
        return
    }

    [IO.Directory]::CreateDirectory($toolRoot) | Out-Null
    $archive = Join-Path $toolRoot $cmakeArchiveName
    $release = "https://github.com/Kitware/CMake/releases/download/v$cmakeVersion"
    $checksums = Invoke-RestMethod -Uri "$release/cmake-$cmakeVersion-SHA-256.txt"
    $expected = @(
        ($checksums -split "`n") |
            Where-Object { $_ -match ([regex]::Escape($cmakeArchiveName) + '$') } |
            ForEach-Object { ($_ -split '\s+')[0].ToLowerInvariant() }
    )
    if ($expected.Count -ne 1 -or $expected[0] -notmatch '^[0-9a-f]{64}$') {
        throw "could not resolve the published SHA-256 for $cmakeArchiveName"
    }
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        Invoke-WebRequest -Uri "$release/$cmakeArchiveName" -OutFile $archive
    }
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected[0]) {
        throw "portable CMake SHA-256 mismatch: expected $($expected[0]), got $actual"
    }

    [IO.Directory]::CreateDirectory($cmakeDirectory) | Out-Null
    Invoke-Checked -FilePath 'tar.exe' -Arguments @('-xf', $archive, '-C', $cmakeDirectory)
    if (-not (Test-Path -LiteralPath $cmakeExecutable -PathType Leaf)) {
        throw 'portable CMake extraction did not produce cmake.exe'
    }
}

function Install-PortableCudaCompiler {
    $required = @(
        (Join-Path $cudaRoot 'bin\nvcc.exe'),
        (Join-Path $cudaRoot 'include\cuda.h'),
        (Join-Path $cudaRoot 'include\crt\host_config.h'),
        (Join-Path $cudaRoot 'include\cccl\cuda\std\version'),
        (Join-Path $cudaRoot 'nvvm\bin\cicc.exe'),
        (Join-Path $cudaRoot 'nvvm\libdevice\libdevice.10.bc'),
        (Join-Path $cudaRoot 'bin\x86_64\nvrtc64_130_0.dll'),
        (Join-Path $cudaRoot 'bin\x86_64\nvrtc-builtins64_131.dll')
    )
    if (@($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }).Count -eq 0) {
        return
    }

    [IO.Directory]::CreateDirectory($cudaPythonRoot) | Out-Null
    $arguments = @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-input',
        '--no-deps',
        '--upgrade',
        '--force-reinstall',
        '--target', $cudaPythonRoot
    ) + $cudaPackages
    Invoke-Checked -FilePath 'python.exe' -Arguments $arguments
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "portable CUDA compiler is incomplete: missing $path"
        }
    }
}

function Install-PortableLibclang {
    if (Test-Path -LiteralPath $libclangDll -PathType Leaf) {
        return
    }

    [IO.Directory]::CreateDirectory($libclangRoot) | Out-Null
    Invoke-Checked -FilePath 'python.exe' -Arguments @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-input',
        '--no-deps',
        '--upgrade',
        '--force-reinstall',
        '--target', $libclangRoot,
        $libclangPackage
    )
    if (-not (Test-Path -LiteralPath $libclangDll -PathType Leaf)) {
        throw "portable libclang is incomplete: missing $libclangDll"
    }
}

function Assert-PerformanceBuildEnvironment {
    if (-not [Environment]::Is64BitProcess -or $env:OS -ne 'Windows_NT') {
        throw 'Hskify performance binaries require 64-bit Windows'
    }
    if (
        [string] $env:CUDA_HOME -cne $cudaRoot -or
        [string] $env:CUDA_PATH -cne $cudaRoot
    ) {
        throw 'CUDA_HOME and CUDA_PATH must both identify the pinned CUDA 13.1 toolchain'
    }
    if ([string] $env:ORT_CUDA_VERSION -cne '13') {
        throw 'ORT_CUDA_VERSION must be exactly 13'
    }
    if ([string] $env:CUDA_COMPUTE_CAP -cne '89') {
        throw 'CUDA_COMPUTE_CAP must be exactly 89'
    }
    if ([string] $env:LLAMA_CPP_TAG -cne 'b8935') {
        throw 'LLAMA_CPP_TAG must be exactly b8935'
    }
    foreach ($required in @(
        (Join-Path $env:CUDA_HOME 'bin\nvcc.exe'),
        (Join-Path $env:CUDA_HOME 'include\cuda.h')
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "pinned CUDA_HOME is incomplete: missing $required"
        }
    }
}

$hardware = Get-HskifyPerformanceGpuIdentity
Assert-HskifyExactPerformanceGpu -Gpu $hardware
$sourceBefore = Get-HskifyPerformanceSourceTreeIdentity -RepositoryRoot $repositoryRoot

Install-PortableCmake
Install-PortableCudaCompiler
Install-PortableLibclang
$msvcCompiler = Get-HskifyMsvcCompilerPath
$msvcDirectory = Split-Path -Parent $msvcCompiler

$env:CUDA_PATH = $cudaRoot
$env:CUDA_HOME = $cudaRoot
$env:ORT_CUDA_VERSION = '13'
$env:CUDA_COMPUTE_CAP = '89'
$env:LLAMA_CPP_TAG = 'b8935'
$env:NVCC_CCBIN = $msvcDirectory
$env:LIBCLANG_PATH = Split-Path -Parent $libclangDll
$env:CMAKE_GENERATOR = 'Visual Studio 17 2022'
$env:CMAKE_GENERATOR_PLATFORM = 'x64'
$env:PATH = "$msvcDirectory;$(Join-Path $cudaRoot 'bin');$(Join-Path $cudaRoot 'bin\x86_64');$(Split-Path -Parent $cmakeExecutable);$env:PATH"

Assert-PerformanceBuildEnvironment
$toolchain = Get-HskifyPerformanceToolchainEvidence `
    -CmakePath $cmakeExecutable `
    -NvccPath (Join-Path $cudaRoot 'bin\nvcc.exe') `
    -MsvcPath $msvcCompiler

if ($PrerequisitesOnly) {
    [ordered]@{
        status = 'ready'
        buildFingerprint = $script:HskifyPerformanceBuildFingerprint
        source = $sourceBefore
        build = [ordered]@{
            target = $targetTriple
            profile = 'release'
            features = @('cuda')
        }
        cuda = [ordered]@{
            cudaHome = $cudaRoot
            toolkitVersion = '13.1'
            ortCudaVersion = '13'
            architecture = 'sm_89'
        }
        hardware = $hardware
        toolchain = $toolchain
        llamaCppTag = 'b8935'
        attestationWritten = $false
    }
    return
}

$git = Get-Command 'git.exe' -ErrorAction Stop
& $git.Source -C $repositoryRoot check-ignore --quiet -- $AttestationPath
if ($LASTEXITCODE -ne 0) {
    throw "performance-build attestation path must be ignored by Git: $AttestationPath"
}
if (Test-Path -LiteralPath $AttestationPath) {
    Remove-Item -LiteralPath $AttestationPath -Force
}

$cargoArguments = @(
    'build',
    '--locked',
    '--release',
    '--package', 'browser-companion',
    '--no-default-features',
    '--features', 'cuda',
    '--bin', 'hsk-manga-native-host',
    '--bin', 'hsk-manga-browser-daemon',
    '-j', '6'
)
Push-Location $repositoryRoot
try {
    Invoke-Checked -FilePath 'cargo.exe' -Arguments $cargoArguments
}
finally {
    Pop-Location
}

foreach ($binary in @($nativeHostPath, $browserDaemonPath)) {
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        throw "successful cargo build did not produce the required release binary: $binary"
    }
}
$sourceAfter = Get-HskifyPerformanceSourceTreeIdentity -RepositoryRoot $repositoryRoot
if (
    [string] $sourceAfter.gitCommit -cne [string] $sourceBefore.gitCommit -or
    [string] $sourceAfter.gitTree -cne [string] $sourceBefore.gitTree -or
    [string] $sourceAfter.aggregateSha256 -cne [string] $sourceBefore.aggregateSha256
) {
    throw 'source tree changed during the release build; no attestation was written'
}

$attestation = New-HskifyPerformanceBuildAttestation `
    -SourceIdentity $sourceAfter `
    -Hardware $hardware `
    -Toolchain $toolchain `
    -CudaHome $cudaRoot `
    -NativeHostPath $nativeHostPath `
    -BrowserDaemonPath $browserDaemonPath
[IO.Directory]::CreateDirectory((Split-Path -Parent $AttestationPath)) | Out-Null
$temporaryAttestation = "$AttestationPath.$([Guid]::NewGuid().ToString('N')).tmp"
try {
    [IO.File]::WriteAllText(
        $temporaryAttestation,
        ($attestation | ConvertTo-Json -Depth 12) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
    if (Test-Path -LiteralPath $AttestationPath -PathType Leaf) {
        [IO.File]::Replace($temporaryAttestation, $AttestationPath, $null)
    }
    else {
        [IO.File]::Move($temporaryAttestation, $AttestationPath)
    }
}
finally {
    if (Test-Path -LiteralPath $temporaryAttestation) {
        Remove-Item -LiteralPath $temporaryAttestation -Force
    }
}

Write-Output $AttestationPath
