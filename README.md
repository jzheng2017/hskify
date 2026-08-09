# Hskify

> Read English manga, webtoons, and light novels in Chinese — locally in Firefox.

Hskify is a Firefox reading companion for learning Mandarin. It automatically
recognizes the current chapter as either a prose document or sequential art,
then renders selectable Simplified Chinese with the requested HSK 2.0 learning
mode. Pages that cannot be classified confidently are left unchanged.

Both readers share pinyin, teaching-term highlighting, local dictionary lookup,
Original/Chinese/hold-to-compare controls, and local Mandarin speech.

## Reading modes

For light-novel chapters, Hskify runs Mozilla Readability on a cloned document,
maps accepted story blocks back to the live page, and mounts a safe inline
Shadow DOM reader next to the original chapter. The complete story structure
appears immediately with English placeholders; each block changes only when
its final Chinese result is ready. The source DOM remains connected and is
restored exactly on Original mode, cancellation, mutation, or navigation.

For manga and webtoons, local vision models detect and recognize story text,
restore only the original lettering areas, and place selectable Chinese over
the untouched source images. Illustrations inside prose chapters are preserved
as illustrations and are never sent through OCR.

One shared local language service owns faithful translation, HSK validation
and repair, pinyin, teaching metadata, dictionary context, and caching for both
modes. Natural mode publishes faithful Chinese; strict mode applies the bounded
HSK realization policy before anything becomes visible.

## Supported setup

This is an intentionally focused Windows performance build:

- Windows x86-64;
- NVIDIA GeForce RTX 4080 SUPER with 16 GB VRAM;
- CUDA 13.1 and a compatible NVIDIA driver.

CPU-only, macOS, Linux, Vulkan, Metal, and remote-provider operation are out of
scope. Model and production resource files are not included in this repository.

## Build

From Windows PowerShell with the Rust MSVC toolchain, Visual Studio C++ tools,
Python, and the supported NVIDIA setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Invoke-PerformanceBuild.ps1
```

The build produces `hskify-native-host.exe` and
`hskify-browser-daemon.exe`. See the
[companion implementation guide](crates/browser-companion/IMPLEMENTATION.md)
for local resource requirements.

## Repository map

- `extensions/firefox` — chapter classification and the two browser renderers
- `crates/browser-companion` — local daemon and image/document pipelines
- `crates/hsk-control` — HSK validation, pinyin, and dictionary tools
- `crates/koharu-ml`, `crates/koharu-app`, `crates/koharu-runtime` — local ML
  and CUDA runtime code
- `scripts` — local build and benchmark tooling

Start with the [documentation index](docs/README.md), then read the
[architecture](docs/architecture.md) and exact
[browser contract](docs/browser-contract.md).
