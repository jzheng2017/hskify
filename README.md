# Hskify

> Read English manga in Chinese — locally in Firefox.

Hskify is a Firefox reading companion for learning Mandarin with manga and
comic pages. It finds English dialogue, thoughts, and narration, translates
them into selectable Simplified Chinese, and restores the artwork behind the
original lettering.

The page updates progressively, starting with the regions currently visible
in the browser. Hskify also supports pinyin, original/Chinese comparison,
dictionary lookup, and local Mandarin speech.

## How it works

1. Firefox sends a selected page image to the local Hskify companion.
2. Local vision models find text and identify which regions are story text.
3. A local language model translates the accepted regions to the requested
   HSK 2.0 level.
4. Hskify cleans only the original text areas and places the Chinese text over
   the page as it becomes ready.

Sound effects, credits, branding, promotion, artwork, and non-English text are
left alone.

## Supported setup

This is an intentionally focused Windows performance build, not a
cross-platform release. The supported target is:

- Windows x86-64;
- an NVIDIA GeForce RTX 4080 SUPER with 16 GB of VRAM;
- the CUDA 13.1 toolchain and a compatible NVIDIA driver.

CPU-only, macOS, Linux, Vulkan, Metal, and remote-provider operation are
outside the current product scope. Model and production resource files are
not included in this repository.

## Build

From a Windows PowerShell prompt with the Rust MSVC toolchain, Visual Studio C++
tools, Python, and the supported NVIDIA setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Invoke-PerformanceBuild.ps1
```

The build script checks the hardware and provisions the pinned build tools. See
[the companion implementation guide](crates/browser-companion/IMPLEMENTATION.md)
for model resources and environment configuration.

## Project status

Hskify is an experimental, hardware-specific performance build. The repository
contains the browser extension, native companion, local inference pipeline, HSK
validation, and benchmark harness, but it does not ship a ready-made model
bundle or claim published latency and quality results.

## Repository map

- `extensions/firefox` — Firefox page discovery and progressive rendering
- `crates/browser-companion` — local daemon and translation pipeline
- `crates/hsk-control` — HSK validation, pinyin, and dictionary tools
- `crates/koharu-ml`, `crates/koharu-app`, `crates/koharu-runtime` — local ML and CUDA runtime code
- `scripts` — build and benchmark tooling

For the deeper technical material, start with the [documentation index](docs/README.md),
then see the [architecture overview](docs/architecture.md) and the
[real-reader-v2 release corpus guide](docs/real-reader-v2.md).
