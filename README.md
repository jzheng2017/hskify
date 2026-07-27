# Hskify

Hskify is a local Firefox reading companion optimized for one production
target: Windows with an NVIDIA GeForce RTX 4080 SUPER 16 GB (compute capability
8.9). The performance build is CUDA-only. CPU, Vulkan, Metal, remote-provider,
desktop-project, and compatibility-mode operation are outside this product
shape.

The current build translates eligible English story text directly into HSK
2.0-targeted Simplified Chinese. This includes dialogue, thought balloons, and
story narration while excluding sound effects, credits, branding, promotion,
and non-English text. It does not create a full-page translated image.
Instead, the daemon progressively publishes small transparent cleanup patches
and selectable Chinese text for each accepted region.

## Current architecture

1. Firefox discovers reader images after an explicit user action and sends the
   selected image to a native companion on the same machine.
2. A one-shot native host validates the installed Firefox identity, starts or
   discovers the loopback daemon, and returns a short-lived authenticated
   session.
3. The daemon decodes the source once, schedules overlapping detector tiles
   with visible tiles first, recognizes English story text, groups related
   lines without merging differently colored emphasis, and constructs a
   region-local cleanup patch.
4. Qwen3.5 4B semantically excludes non-story OCR and translates small
   story-region batches directly to the requested HSK level. A deterministic
   gate accepts exclusion only when the OCR itself supports a credit,
   release-note, SFX, or gibberish decision. Vocabulary and meaning validation
   accepts the translation or sends only the rejected item through one
   targeted repair.
5. The browser fetches and decodes each PNG patch, installs it before the
   corresponding selectable text, and can continue rendering while later
   regions are still running.

There is no versioned browser API, job-result endpoint, full cleaned-page
payload, project/history store, page-wide translation pass, or retranslation
route. The extension, native host, daemon, and contract fixtures instead share
the exact build fingerprint `hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-26-r2`; a mismatch is a
hard failure, not a negotiation opportunity.

Pinyin, longest-match local dictionary lookup, original/Chinese comparison,
and local Mandarin speech through Firefox remain part of the reader
experience.

## Performance target

The supported performance build is intentionally hardware-specific:

- NVIDIA GeForce RTX 4080 SUPER;
- exactly 16,376 MiB reported GPU memory on device 0;
- CUDA compute capability 8.9;
- NVIDIA driver API 13.1, CUDA toolkit 13.1, ORT CUDA 13, and `sm_89`;
- Qwen3.5 4B Q4_K_M plus the pinned detector, OCR, HSK 2.0, dictionary, and
  font resources.

`scripts/Invoke-PerformanceBuild.ps1` rejects a different GPU before building.
The browser-companion crate enables CUDA by default, and the resident runtime
uses GPU-preferred model loading. This repository does not claim a supported
fallback tier.

## Build

From a Windows PowerShell prompt with the Rust MSVC toolchain, Visual Studio
C++ tools, Python, and a compatible NVIDIA driver:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Invoke-PerformanceBuild.ps1
```

The script verifies the exact target GPU and driver API, provisions pinned
portable CMake and CUDA compiler components under `.cache`, and builds only the
Windows x86_64 MSVC release binaries with the required `cuda` feature. After a
successful build it writes the ignored
`target\release\hskify-performance-build-attestation.json`.
That attestation freezes the complete tracked/untracked source-tree identity,
tool versions, CUDA/llama.cpp configuration, exact hardware identity, and both
binary hashes. A failed build never produces a new attestation.

Production language/model resources are local files. See
[the companion implementation](crates/browser-companion/IMPLEMENTATION.md) for
the exact paths and environment overrides.

## Verification status

The documentation in this branch records architecture and test methodology,
not completed performance evidence. The sole canonical fixture is *30 Years
Since the Prologue* chapter 5: 36 hash-pinned images covering varied balloon,
lettering, background, and narration styles. All 36 pages now have reviewed
region geometry and complete translation gold: 218 story regions, 214
translation targets, reviewed Chinese and pinyin, and deterministic token-level
HSK annotations. Release evidence still requires the isolated exact-product
install to complete the cold, warm-up, 20-or-more warm, cache-replay, and
cancellation sequence.

Do not quote latency, throughput, memory, GPU utilization, quality, or
installed-Firefox results for this build until raw evidence is captured by the
method in [Chapter 5 benchmark and evidence](docs/chapter-5-benchmark.md).
The workload is a diverse regression corpus, not a chapter-specific tuning
target: production logic may not key on its text, names, URLs, colors,
coordinates, or hashes.

## Repository map

| Path | Purpose |
| --- | --- |
| `extensions/firefox` | Discovery, progressive update consumption, patch-first rendering, comparison, lookup, and speech |
| `crates/browser-companion` | Native launcher, authenticated loopback daemon, flat job log, direct progressive pipeline |
| `crates/hsk-control` | Deterministic HSK 2.0 validation, pinyin, and dictionary lookup |
| `crates/koharu-ml`, `crates/koharu-app`, `crates/koharu-runtime` | Reused local-model and CUDA runtime primitives |
| `fixtures/contracts` | Shared exact-build contract fixtures |
| `fixtures/benchmarks/30-years-since-the-prologue-chapter-5` | Sole canonical 36-image benchmark manifest, annotations, and local reader replica |
| `scripts/Invoke-PerformanceBuild.ps1` | RTX 4080 SUPER/CUDA-only build gate |
| `scripts/Benchmark-Chapter5.ps1` | Packaged-Firefox release E2E benchmark and raw evidence harness |

Start with [the documentation index](docs/README.md), [the architecture
overview](docs/architecture.md), and [the unversioned browser
contract](docs/browser-contract.md).
