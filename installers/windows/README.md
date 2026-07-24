# Windows developer package

This directory provides the deliberately small Windows packaging path for the
HSK Manga Translator. It builds the two release companion executables, builds
and zips the Firefox Manifest V3 extension, stages a portable directory, and
installs that directory for the current user. It does not introduce an MSI,
GUI bootstrapper, model downloader, or another runtime.

The existing scripts under `native-host-registration` remain the authority for
the exact Firefox native host:

```text
local.mangalations.hsk_manga
```

and its one allowed extension:

```text
hsk-manga-translator@local.mangalations
```

## Build

Install the Rust MSVC toolchain, Visual Studio C++ build tools and Windows SDK,
CMake, LLVM/libclang, Node.js, and npm. `cmake` should be on `PATH`; when
libclang is installed in a nonstandard location, set `LIBCLANG_PATH` to the
directory containing `libclang.dll`. Then run this from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\installers\windows\Build-DeveloperPackage.ps1
```

That command runs:

```text
cargo build --release --package browser-companion --bin hsk-manga-native-host --bin hsk-manga-browser-daemon
npm install --no-package-lock --no-audit --no-fund
npm run build
npm run zip
```

The npm commands run from `extensions/firefox`. Pass `-SkipNpmInstall` when its
dependencies are already present.

Production language/model files are never guessed or downloaded by the build.
Supply their existing paths explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\installers\windows\Build-DeveloperPackage.ps1 `
  -HskArtifactPath C:\artifacts\hsk-2.0.normalized.json `
  -DictionaryArtifactPath C:\artifacts\cc-cedict.normalized.json `
  -ModelPath C:\models\Qwen3.5-4B-Q4_K_M.gguf `
  -Force
```

The model must match the selected standard pack in
`data\model-packs\manifest.v1.json`. The build checks both its byte count and
SHA-256 (`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
for the current Qwen3.5 4B Q4_K_M selection) before copying it. No large file is
downloaded automatically.

Prebuilt binaries and an extension archive can be supplied with
`-NativeHostPath`, `-BrowserDaemonPath`, and `-FirefoxExtensionZipPath`. This is
the deterministic seam used by the smoke test; the two executable paths must
be supplied together.

The default output is:

```text
dist\hsk-manga-translator-windows\
  Install.ps1
  Uninstall.ps1
  README.md
  bundle-manifest.json
  companion\
    hsk-manga-native-host.exe
    hsk-manga-browser-daemon.exe
  extension\
    hsk-manga-translator-firefox.zip
  native-host-registration\
    Register-NativeHost.ps1
    Unregister-NativeHost.ps1
    local.mangalations.hsk_manga.json.template
    README.md
  resources\
    hsk-2.0.normalized.json                 # only when supplied
    cc-cedict.normalized.json               # only when supplied
    models\
      Qwen3.5-4B-Q4_K_M.gguf                # only when supplied
    model-packs\
      manifest.v1.json
```

## Install and uninstall

Install for the current user:

```powershell
powershell -ExecutionPolicy Bypass -File .\dist\hsk-manga-translator-windows\Install.ps1
```

The installed application is under:

```text
%LOCALAPPDATA%\Mangalations\HSKMangaTranslator\app
```

The production resource contract is:

```text
%LOCALAPPDATA%\Mangalations\HSKMangaTranslator\resources\hsk-2.0.normalized.json
%LOCALAPPDATA%\Mangalations\HSKMangaTranslator\resources\cc-cedict.normalized.json
%LOCALAPPDATA%\Mangalations\HSKMangaTranslator\resources\models\Qwen3.5-4B-Q4_K_M.gguf
```

The installer verifies staged file hashes, copies the bundle, and invokes the
preserved registration script with the installed native-host path. It prints
the packaged Firefox extension location.

Uninstall with:

```powershell
powershell -ExecutionPolicy Bypass -File "$env:LOCALAPPDATA\Mangalations\HSKMangaTranslator\app\Uninstall.ps1"
```

Uninstall stops only a recorded daemon whose executable path exactly matches
this installation, unregisters the native host, and removes the application,
resources, and browser cache. Pass `-KeepCache` to retain the browser cache.

## Smoke tests

The tests use tiny local dummy binaries, data, model bytes, and an isolated
per-test registry key. They do not compile the application or download a
model:

```powershell
powershell -ExecutionPolicy Bypass -File .\installers\test-native-host-registration.ps1
powershell -ExecutionPolicy Bypass -File .\installers\windows\Test-DeveloperPackage.ps1
```
