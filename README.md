# Hskify

Hskify is a local-first Firefox reading companion for learning Mandarin from
English manga and webtoons. It detects English dialogue inside speech bubbles,
cleans only the accepted dialogue pixels, translates the page to Simplified
Chinese, and overlays selectable text constrained to a chosen HSK 2.0 level.

The result keeps the original page in the reader while adding:

- faithful and HSK-constrained Simplified Chinese;
- pinyin, definitions, and HSK level lookup for selected text;
- local Mandarin pronunciation when Firefox can access a suitable system
  voice;
- stable region geometry and cached cleaning results for fast retranslation;
  and
- bounded, local processing with no telemetry or manga-image egress.

Hskify currently targets the Firefox extension plus native companion workflow.
The Windows developer package has the strongest end-to-end validation; Linux
and macOS registration assets exist but still need the platform smoke tests
listed in the [Firefox manual test checklist](docs/firefox-manual-test-checklist.md).

## How it works

1. The extension discovers likely chapter images after an explicit user action.
2. Firefox native messaging starts or discovers a per-user Hskify daemon.
3. The daemon validates and bounds the image before invoking the reused Koharu
   pipeline for dialogue detection, OCR, and cleaning.
4. A local Qwen model produces a faithful translation and an HSK-targeted
   rewrite.
5. `hsk-control` validates vocabulary and supplies pinyin and dictionary data.
6. The extension renders selectable Chinese over the cleaned page.

See the [architecture overview](docs/architecture.md) for the component
boundaries, data flow, security model, and resource limits.

## Project layout

| Path | Responsibility |
| --- | --- |
| `extensions/firefox` | Page discovery, explicit-origin permissions, job UI, overlays, selection, lookup, and local pronunciation |
| `crates/browser-companion` | Native messaging launcher, authenticated loopback daemon, browser protocol, job lifecycle, and Koharu adapter |
| `crates/hsk-control` | HSK 2.0 validation, pinyin normalization, and CC-CEDICT-compatible lookup |
| `crates/koharu-*` | Inherited and selectively extended Koharu pipeline, runtime, ML, rendering, and shared application layers |
| `data` | Manifests, licences, and small project-authored seeds; production language/model artifacts remain outside Git |
| `installers` | Per-platform native-host registration and developer packaging |
| `docs` | Hskify design and maintenance docs plus inherited Koharu multilingual docs |

## Getting started

Hskify does not yet advertise a general end-user release. For a development
installation, start with the
[Windows developer package guide](installers/windows/README.md). Production
language data and model files are intentionally not committed or guessed by
the build.

To prepare reproducible local HSK and dictionary artifacts on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_local_language_data.ps1
```

Common verification commands are:

```text
cargo test -p browser-companion --all-targets -j 1
cargo test -p hsk-control --all-targets -j 1
bun install
bun run typecheck:firefox
bun run test:firefox
bun run build:firefox
```

ML builds may also require CMake, LLVM/libclang, platform build tools, and an
appropriate GPU runtime. Consult the inherited
[Koharu build guide](docs/en-US/how-to/build-from-source.md) when changing
shared pipeline crates.

## Documentation

Start at the [documentation index](docs/README.md).

- [Architecture overview](docs/architecture.md)
- [Maintainer guide and upstream synchronization](docs/maintainer-guide.md)
- [Architecture decisions](docs/architecture-decisions/)
- [Implementation notes](docs/implementation-notes/)
- [Firefox manual test checklist](docs/firefox-manual-test-checklist.md)
- [Model benchmark](docs/model-benchmark.md)
- [Licence inventory](docs/licence-inventory.md)

The localized `docs/en-US`, `docs/ja-JP`, `docs/pt-BR`, and `docs/zh-CN`
directories describe inherited Koharu functionality. They are retained as
upstream reference material and should not be read as Hskify installation or
support documentation.

## Koharu attribution

Hskify is a GPL-3.0-only fork of
[Koharu](https://github.com/mayocream/koharu), the local-first Rust manga
translation application created by Koharu's contributors. Hskify reuses
Koharu's application, ML, runtime, rendering, and project-storage layers while
adding a Firefox-specific security boundary and HSK learning workflow.

The pinned upstream revision and reuse boundary are recorded in
[ADR 0001](docs/architecture-decisions/0001-koharu-upstream-pin.md). See the
[maintainer guide](docs/maintainer-guide.md) before merging a new Koharu
revision.

## Privacy and security

Hskify binds its browser API to a random IPv4 loopback port, authenticates each
extension session, accepts only the registered extension origin, and exposes
only the dedicated `/browser/v1` surface. Source images, OCR text, translations,
and dictionary lookups remain on the user's machine. The browser path does not
initialize Koharu's remote providers.

Mandarin playback is also local-only: it uses Firefox's Web Speech integration
with an installed Simplified Chinese voice and has no cloud fallback. See
[ADR 0005](docs/architecture-decisions/0005-mandarin-pronunciation-voice-selection.md)
for selection rules and limitations.

## Contributing and licence

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a change. Hskify is
licensed under [GPL-3.0-only](LICENSE); model, font, HSK, and dictionary inputs
have their own terms recorded in the [licence inventory](docs/licence-inventory.md).
