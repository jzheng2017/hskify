# Windows performance package

The current Hskify product target is Windows on an NVIDIA GeForce RTX 4080
SUPER 16 GB (compute capability 8.9). Production native binaries must come from
the CUDA-gated performance build:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Invoke-PerformanceBuild.ps1
```

That script rejects other GPUs, provisions the pinned CUDA 13.1 compiler
components, sets compute capability 8.9, and produces:

```text
target\x86_64-pc-windows-msvc\release\hsk-manga-native-host.exe
target\x86_64-pc-windows-msvc\release\hsk-manga-browser-daemon.exe
target\x86_64-pc-windows-msvc\release\hskify-performance-build-attestation.json
```

`Build-DeveloperPackage.ps1` invokes that exact performance wrapper when
binaries are omitted. Explicit binaries are accepted only with their matching
attestation; the packager verifies the exact source tree, target, release
profile, CUDA feature/toolchain, RTX 4080 SUPER hardware, fingerprint, and
binary byte/hash claims before staging:

```powershell
powershell -ExecutionPolicy Bypass -File .\installers\windows\Build-DeveloperPackage.ps1 `
  -NativeHostPath .\target\x86_64-pc-windows-msvc\release\hsk-manga-native-host.exe `
  -BrowserDaemonPath .\target\x86_64-pc-windows-msvc\release\hsk-manga-browser-daemon.exe `
  -BuildAttestationPath .\target\x86_64-pc-windows-msvc\release\hskify-performance-build-attestation.json `
  -HskArtifactPath C:\artifacts\hsk-2.0.normalized.json `
  -DictionaryArtifactPath C:\artifacts\cc-cedict.normalized.json `
  -ModelPath C:\models\Qwen3.5-4B-Q4_K_M.gguf `
  -Force
```

The packager verifies the mandatory Qwen3.5-4B artifact's byte count and
SHA-256 against the model manifest and stages the Firefox extension, exact
native host and daemon, release attestation, fonts, language resources, and
registration scripts. The bundle manifest hashes the copied attestation.

## Installed layout

The current-user installation root is:

```text
%LOCALAPPDATA%\Hskify
```

Production resources are:

```text
resources\hsk-2.0.normalized.json
resources\cc-cedict.normalized.json
resources\models\Qwen3.5-4B-Q4_K_M.gguf
resources\fonts\NotoSansSC-VF.ttf
resources\fonts\NotoSerifSC-VF.ttf
```

The registered native host is `local.hskify.hsk_manga`, and the only allowed
Firefox extension is `hsk-manga-translator@local.hskify`.

## Status

Packaging-script tests with dummy bytes do not establish CUDA execution,
installed-Firefox behavior, or chapter performance. A complete installed
performance-package run is **pending** under
[`docs/firefox-manual-test-checklist.md`](../../docs/firefox-manual-test-checklist.md)
and [`docs/chapter-5-benchmark.md`](../../docs/chapter-5-benchmark.md).
