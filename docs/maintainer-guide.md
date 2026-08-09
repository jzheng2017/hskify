# Maintainer guide

Hskify is one current-chapter Firefox product with exclusive document and image
modes. Keep modality-specific work at the edges and keep language correctness,
transport, replay, teaching tools, and lifecycle ownership shared.

## Product invariants

- Supported target: Windows x86-64, RTX 4080 SUPER 16 GB, compute capability
  8.9, CUDA 13.1.
- Exact fingerprint:
  `hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-08-09-r8`.
- Exact identities: extension `hskify@local.hskify`, native host
  `local.hskify.browser`, and binaries `hskify-native-host` /
  `hskify-browser-daemon`.
- Unversioned exact routes; no compatibility aliases, legacy schema, or settings
  migration.
- One automatic classifier selects `document`, `image`, or `unsupported`.
- Readability runs only on a clone. A valid document descriptor wins over image
  detection and must map to a live root below `body`.
- Source pages remain connected. Every temporary attribute and Hskify-owned node
  is restored or removed on Original mode, cancellation, mutation, navigation,
  fatal failure, and disposal.
- One shared `TranslationService`; prompts gain an OCR-correction instruction
  only for `ocr` provenance.
- Every document block is registered before dispatch. Canonical context and
  output order are `(sourceIndex, itemOrder)`, independent of focus priority.
- The 4,096-token language context uses the real tokenizer and batches at most
  six units. Only an individually oversized document block may split at ICU
  sentence boundaries, and it publishes only after joined validation.
- Natural mode publishes faithful Chinese with deterministic teaching metadata.
  Strict mode permits one bounded terminal repair. No provisional Chinese is
  visible.
- `LanguageRuntime` is sufficient for document jobs. Instrumentation must prove
  they do not initialize or invoke vision, OCR, projector, segmentation,
  inpainting, patch, or font paths.
- One append-only update log and acknowledgement/replay path serves both modes.
  Active records and persistent completed results use an exact tagged
  `image | document` union.
- Original/Chinese/hold-to-compare, pinyin, teaching terms, dictionary lookup,
  selection, and local Mandarin speech are shared.
- Reading direction appears only for image chapters.

## Review routing

| Change | Required evidence |
| --- | --- |
| Classifier or extraction | Semantic/div-heavy/noisy/hybrid/manga fixtures, live-DOM immutability, root safety, size bounds |
| Route, header, or JSON field | Browser contract plus exact TypeScript and Rust unknown-field/bounds/modality fixtures |
| Translation, prompt, tokenizer, or validator | Shared service tests, mid-chapter context, strict repair, joined-block validation, cache identity review |
| Document rendering | Final-only installation, teaching offsets, comparison, anchor, mutation/navigation/cancellation restoration tests |
| Image behavior | Existing image contracts, renderer, native, and real-reader regressions |
| Runtime loading | Instrumented proof of language-only document warm-up and full image warm-up |
| Performance claim | Raw local evidence under the exact fingerprint, browser, model resources, and supported workstation |
| GPU/toolchain change | Performance-build gate and a separate benchmark configuration |

## Verification commands

Use the exact performance wrapper for release binaries:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Invoke-PerformanceBuild.ps1
```

Run relevant local checks from the repository root:

```text
cargo fmt --all -- --check
cargo test -p browser-companion --all-targets -j 1
cargo test -p koharu-app --all-targets -j 1
cargo test -p koharu-llm --all-targets -j 1
cargo test -p hsk-control --all-targets -j 1
cargo clippy -p browser-companion -p koharu-app -p koharu-llm --all-targets -j 1 -- -D warnings
pnpm --dir extensions/firefox typecheck
pnpm --dir extensions/firefox test
pnpm --dir extensions/firefox build
node --test scripts/benchmark/document-chapter-gates.test.mjs scripts/run-document-chapter-benchmark.test.mjs
powershell -File installers/test-native-host-registration.ps1
powershell -File installers/windows/Test-ReleasePackage.ps1
git diff --check
```

These are commands to run, not pass claims. Record raw output for release
evidence. Do not copy measurements across fingerprints, model revisions,
resource hashes, hardware, browser builds, or fixture changes. Hskify has no CI;
verification is explicit and local.

## Performance gates

For a representative 300-block/100,000-character light-novel chapter:

- extraction plus reader skeleton p95 is below 100 ms;
- no scroll-time synchronous layout loop or task exceeds 50 ms;
- the first visible block dispatches alone, then tokenizer-aware batches contain
  at most six units;
- language-only warm-up allocates no vision resources.

Image language throughput may regress no more than 10 percent from the recorded
pre-refactor local baseline, and the complete image runtime with 4,096 language
tokens must remain within 16 GB VRAM.

### Runnable document benchmark

The local runner generates the fixture rather than accepting a hand-authored
chapter. Its served article contains exactly 300 mapped text blocks and 100,000
normalized characters. It launches the packaged extension through the same
Firefox BiDi harness as the real-reader regression, starts a real document job,
mounts all placeholders, performs a deterministic full-chapter scroll, records
the Hskify User Timing measures and scroll-window long-task evidence, then
cancels the unfinished translation before the next sample. It makes no external
network request; the reader binds only to a random IPv4 loopback port.

Firefox does not expose the
[Long Tasks API](https://developer.mozilla.org/docs/Web/API/PerformanceLongTaskTiming).
The harness uses that API when a test browser provides it and otherwise keeps a
four-millisecond timer pending during the run. A late timer records the
conservative upper bound
`eventLoopDelayMs + 4 ms`; any bound over 50 ms fails the scroll gate. Every raw
sample names the observation mode, so fallback evidence cannot be mistaken for
a native `PerformanceLongTaskTiming` entry.

Provide the existing packaged-Firefox harness configuration with
`--config <path>` or `HSKIFY_REAL_READER_BROWSER_CONFIG`. The JSON contains:

```json
{
  "extensionPackagePath": "C:\\path\\to\\hskify-firefox.zip",
  "firefoxExecutable": "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
  "playwrightModule": "C:\\path\\to\\node_modules\\@playwright\\test",
  "extensionVersion": "0.1.0",
  "profileDirectory": "C:\\path\\to\\benchmark-firefox-profile",
  "stateDirectory": "C:\\path\\to\\hskify-benchmark-state"
}
```

Run 20 samples, which is the default used for the local p95:

```powershell
node .\scripts\run-document-chapter-benchmark.mjs `
  --config .\.cache\document-browser-config.json `
  --samples 20
```

Native scheduler/resource instrumentation is not exposed through the browser
API. The runner initializes an empty native evidence envelope, exports its path
as `HSKIFY_BENCH_EVIDENCE_PATH` before Firefox starts, and joins native samples
to browser observations by exact tagged `(kind, jobId)` identity. The default is
`.cache/document-chapter-benchmark/native-evidence.json`; set the same
environment variable or the optional `nativeEvidencePath` browser-config field
to place it elsewhere. This is a native output, not a hand-authored input.

Before cancelling each sample, the runner polls that file until the matched
document job contains both its first visible-only dispatch and a post-first
dispatch. The native writer flushes after each dispatch and records the real
tokenizer counts plus vision/OCR/projector/segmentation/inpainting/patch/font
initialization and invocation counters.

The runner writes raw samples to
`.cache/document-chapter-benchmark/raw-samples.json` and the evaluated result to
the adjacent `evaluation.json`. The evaluator fails closed if native output is
missing or incomplete; it never infers zero resource use. It likewise fails if
neither the native Long Tasks observer nor the conservative event-loop probe
produced an explicit observation mode.

Re-evaluate an existing raw file with:

```powershell
npm run benchmark:document:gates
```

The evaluator fails on missing reader mounting, User Timing, scroll/long-task
coverage, job-matched native evidence, a non-representative workload, an invalid
first-visible/batch sequence, or any document-time vision initialization or
invocation.

### Runnable image throughput and VRAM benchmark

The image benchmark wraps the existing complete local real-reader-v2 packaged-
Firefox run. Native evidence supplies cumulative language-unit counts and wall
time measured strictly around admitted model generation calls. Each OCR-
authoritative unit is counted once at faithful generation. Strict-primary and
terminal-repair calls add duration without counting the source again; queue,
cache lookup, and validation time are excluded. In parallel,
the runner invokes `nvidia-smi` at a 100 ms target interval and retains every raw
sample, so the reported current throughput and full-runtime peak VRAM are
measurements rather than JSON fields supplied by a maintainer.

The configured `stateDirectory` must already have a valid installed-resource
marker. For every measurement the runner creates a unique new Firefox profile
and native state directory, copies only that marker, and leaves the result cache
empty. It refuses resource download during the measured run. Installed model
files remain shared through the normal resource directory (or
`HSKIFY_RESOURCES_DIR`), while no completed translation can be replayed from a
previous run. The fresh run directory and marker digest are retained in raw
evidence.

First record the pre-refactor baseline from the baseline build with the same
browser configuration, corpus selection, and workstation:

```powershell
npm run benchmark:image -- `
  --config .\.cache\document-browser-config.json `
  --selection core `
  --record-baseline .\.cache\image-runtime-baseline.json
```

The command writes a measured baseline artifact containing raw native unit and
duration totals, the tokenizer identity, and a canonical workload hash. The
hash covers the ordered corpus source hashes, dimensions, surface kinds,
reading directions, and exact translation settings. Do not create this file by
hand.

Run the rebuilt implementation with that artifact as an explicit input:

```powershell
$env:HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH = `
  (Resolve-Path .\.cache\image-runtime-baseline.json)
npm run benchmark:image -- `
  --config .\.cache\document-browser-config.json `
  --selection core
```

Use `HSKIFY_NVIDIA_SMI_PATH` when `nvidia-smi.exe` is not on `PATH`, and
`HSKIFY_BENCH_GPU_INDEX` when the supported GPU is not index zero. The runner
uses the same deterministic native-evidence flow as the document benchmark.
It writes `.cache/image-runtime-benchmark/raw-samples.json`, the underlying
real-reader summary, and `evaluation.json`. Re-evaluate the raw evidence with:

```powershell
npm run benchmark:image:gates
```

The image gate requires a passed packaged-Firefox regression, an exact native
sample for every browser image job, positive measured language work, raw GPU
samples, matching workload and tokenizer identities, at least 90 percent of
baseline language throughput, and peak VRAM no greater than 16,384 MiB. A JSON
object containing only claimed rates or memory values is rejected.
