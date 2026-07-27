# 30 Years Since the Prologue chapter 5 release benchmark

This is the sole canonical end-to-end benchmark for the direct progressive
Firefox build. It replaces every earlier chapter workload.

Release performance is valid only when the packaged-Firefox harness finishes
every required run while retaining its raw evidence. Faster source-level
diagnostics are recorded separately and never substituted for browser results.

## Why this workload

Chapter 5 contains 36 ordered source images and substantially more visual
variety than a conventional white-balloon sample. It includes light, dark,
gradient, textured, outlined, and strongly colored story regions; black,
white, and colored lettering; irregular balloon contours; narration; and
non-story material that must remain untouched.

The implementation must not recognize a fixed palette or memorize this
chapter. Detection, OCR acceptance, erase-mask generation, background
reconstruction, and text styling are evaluated by geometry, confidence,
language, local image structure, and semantic role. A change that improves
this fixture through chapter-specific coordinates, phrases, names, colors,
URLs, or hashes is invalid.

The fixture is a demanding regression corpus, not a production allowlist.
The extension must continue to operate on arbitrary chapters and arbitrary
foreground/background colors.

## Frozen workload and current status

`fixtures/benchmarks/30-years-since-the-prologue-chapter-5/manifest.json`
freezes:

- 36 source identities in reader order;
- 20,254,940 total encoded bytes;
- 300,102,900 total decoded pixels;
- exact byte counts, dimensions, and SHA-256 hashes;
- the annotation and evidence schemas; and
- a deterministic local HTTP reader replica.

The original source bytes are stored only in the ignored directory
`.cache/benchmarks/30-years-since-the-prologue-chapter-5/source`.

The committed manifest marks the annotation set **complete**. All 36 pages
contain reviewed geometry for 218 story regions, including 60 manual misses
from the first product pass. All 214 translation targets have reviewed Chinese,
pinyin, and deterministic token-level HSK annotations. Benchmark preflight may
therefore use this gold, but no latency, recall, OCR, cleaning, memory, or VRAM
result is release evidence until the exact packaged build completes the
required run sequence below.

Gold review must distinguish:

- eligible English dialogue, thought, and story narration;
- ambiguous text requiring an explicit review decision; and
- excluded sound effects, credits, scanlation promotion, branding,
  non-English text, and OCR gibberish.

The source-image hash is provenance, never an inference feature.

## Current product-path diagnostic

A fresh isolated-daemon diagnostic was run against all 36 canonical source
images after the RT-DETR replacement, fixed OCR crop guard, semantic exclusion
guard, and deterministic question-punctuation restoration were installed. It
used the unversioned product HTTP routes and exact fingerprint
`hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-26-r2`:

| Diagnostic | Result |
| --- | ---: |
| Published progressive regions | 214 |
| Spatially matched reviewed targets | 209 / 214 |
| Story-region publication recall | 97.66% |
| Matched English OCR CER | 1.88% |
| False translations | 0 / 214 |
| Per-image completion p95 | 3,832 ms |
| Slowest image | 4,347 ms |
| Full 36-image diagnostic wall time | 76.9 s |

The ignored raw diagnostic is
`.cache/ch5-product-audit/run-46-final-full/summary.json`. This is useful runtime
evidence, but it is deliberately not called release evidence: it does not
include packaged Firefox acquisition/rendering, 20 warm iterations,
cache-replay, memory/VRAM sampling, cancellation, DOM ordering, overflow, or
the final browser-rendered patch audit.

## Current packaged release evidence

The exact package recorded on 2026-07-27 completed one installed-cold run, one
warm-up, 20 measured warm inference runs, exact-cache replay, reader-feature
checks, restoration, cancellation, resource sampling, and disk auditing. All
426 gates passed. Nearest-rank warm results were:

| Metric | Result | Required |
| --- | ---: | ---: |
| HUD acknowledgement p95 | 3 ms | <=100 ms |
| First visible region p95 | 1,698 ms | <=2,000 ms |
| Visible three-region group p95 | 1,717 ms | <=5,000 ms |
| First long image complete p95 | 1,855 ms | <=12,000 ms |
| Complete 36-image chapter p95 | 35,352 ms | <=90,000 ms |
| Exact cached first viewport | 162 ms | <=250 ms |

Installed-cold first visible region was 4,021 ms, the first long image completed
in 6,774 ms, and the full chapter completed in 75,852 ms. Correctness and
resource evidence was:

| Gate | Result |
| --- | ---: |
| Story-region publication recall | 210 / 214 (98.13%) |
| English OCR CER | 232 / 8,565 (2.71%) |
| Non-English/non-story false translations | 1 / 216 (0.46%) |
| Independent source-glyph pixels erased | 6,898,569 / 6,898,569 |
| Changed pixels outside accepted regions | 0 / 25,640,363 |
| Overflow across zoom/resize checks | 0 / 864 |
| Page / daemon cancellation | 6 ms / 12 ms |
| Peak daemon private memory | 8,124,383,232 bytes (7.57 GiB) |
| Peak device-wide VRAM | 5,239 MiB |
| Synchronous intermediate writes | 0 |
| Positive page-output samples / sustained sequences | 0 / 0 |

Exact-cache replay used the same daemon, published the initial visible group in
162 ms, and loaded no inference model. Pinyin, dictionary lookup, comparison,
Mandarin speech, patch-before-Chinese ordering, source restoration, and all
supported zoom/resize scenarios passed.

The authoritative ignored output is
`.cache/benchmark-evidence/30-years-since-the-prologue-chapter-5/f36/summary.json`;
`gate-evaluation.json` beside it contains all 426 gate records, and the raw
per-run evidence remains in the same directory.

## Superseded packaged baseline

An earlier packaged build completed one cold run, one warm-up, and 20 measured
warm inference runs before the cache read-path and direct-pipeline
optimizations. These numbers are a superseded diagnostic baseline, not current
release evidence:

| Metric | Result |
| --- | ---: |
| Installed-cold first visible region | 4,426 ms |
| Installed-cold visible three-region group | 4,464 ms |
| Installed-cold first long image | 7,075 ms |
| Installed-cold complete 36-image chapter | 76,707 ms |
| Warm HUD acknowledgement p95 | 3 ms |
| Warm first visible region p95 | 1,819 ms |
| Warm visible three-region group p95 | 1,842 ms |
| Warm first long image p95 | 2,003 ms |
| Warm complete 36-image chapter p95 | 33,769 ms |
| Story-region recall | 97.66% |
| English OCR CER | 1.88% |
| False translations | 0 / 214 |

All 20 inference runs passed the gates defined for that earlier build. The same
sequence exposed a 453 ms exact-cache first-viewport result. Source fetch was
7 ms, but the daemon fully decoded the 16,000-pixel-tall image before checking
the exact result key. Cache reads now verify the uploaded bytes, SHA-256,
format, MIME, safety limits, and header dimensions, then open only their exact
SHA-keyed entry. A hit reuses content that was fully decoded when the entry was
created; only a miss performs full pixel decoding. Byte accounting and
eviction remain on the atomic store path.

The runner creates `summary.json` only if the complete cold, warm-up, 20 warm,
exact-cache, reader-feature, restoration, cancellation, memory, VRAM, and
disk-write sequence passes. Raw evidence remains beside it.

## Deterministic preflight

The benchmark runner is:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\Benchmark-Chapter5.ps1 `
  -PrerequisitesOnly `
  -ResourcesDirectory C:\absolute\path\to\resources
```

Preflight verifies the complete fixture, every source and annotation hash,
schema hashes, replica assets, release binaries, matching performance-build
attestation, packaged extension identity, Firefox, runtime resources, exact
model identities, the RTX 4080 SUPER/CUDA environment, and the absence of an
already-running Hskify daemon.

Preflight must remain read-only. It must not build or package the extension,
change native-host registration, create evidence, launch Firefox, load a
model, or download anything.

## Measured product path

The harness installs the packaged XPI into a fresh Firefox profile and serves
the exact 36 images through the local replica. It measures the product path,
not a Rust pipeline shortcut:

1. Firefox acquires, validates, hashes, and decodes chapter images in reader
   order.
2. The extension submits strict `POST /jobs` requests and immediately supplies
   the visible rectangles.
3. The daemon prioritizes the current viewport, then the rest of the current
   image, then exactly one image ahead, before remaining offscreen work.
4. Resident CUDA detection and OCR accept eligible English story regions
   without assuming foreground or background colors.
5. Direct HSK translation and deterministic validation publish progressive
   region updates.
6. Firefox fetches and decodes each transparent cleanup patch, commits it
   before the corresponding Chinese, and continues while offscreen regions
   run.

The deterministic timing viewport is 1280 x 1080 and is selected from committed
gold before Firefox starts. It must contain 3-6 independent English story
regions; the current fixture selects three regions without consulting runtime
output. The measured clock begins at the benchmark-only content start message
and includes image acquisition, hashing, upload, daemon work, patch retrieval,
browser decoding, and DOM commit. Optional live-site navigation and CDN time
are recorded separately.

## Required gates

Every measured warm run must meet:

| Metric | Required p95 |
| --- | ---: |
| HUD acknowledgement | <=100 ms |
| Exact cached first viewport | <=250 ms |
| First visible region | <=2 s |
| Initially visible 3-6 regions | <=5 s |
| First long image complete | <=12 s |
| All 36 images complete | <=90 s |
| Cancellation stops active compute and restores the page | <=500 ms |
| Peak private memory | <=8 GiB |
| Peak VRAM | <=10 GiB |
| Synchronous intermediate disk writes | 0 |
| Sustained pagefile writes | 0 |

Installed-but-process-cold limits, excluding downloads, are 8 seconds to the
first visible region, 20 seconds for the first long image, and 120 seconds for
the complete 36-image chapter.

Correctness gates are:

- accepted story-region precision at least 99% and recall at least 95%;
- English OCR character error rate at most 3%;
- false translation of excluded or non-English content at most 1%;
- no original glyphs remaining inside accepted erase regions;
- no changed pixels outside erase masks;
- no palette-specific rejection or conversion of a colored region to white;
- preservation of local color, gradients, texture, contour, and styling
  outside the glyph mask;
- patch installation before corresponding Chinese;
- zero Chinese text overflow at supported resize and zoom states;
- exact-cache replay without loading inference models;
- exact original restoration on cancellation, navigation, or source
  replacement; and
- working comparison, pinyin, dictionary, selection popover, and local
  Mandarin speech.

Correctness is scored on the progressive regions actually published to and
committed by Firefox. This deliberately covers the complete detector, OCR,
semantic acceptance, patch, and rendering path; raw proposal precision is not
a product metric because the detector intentionally emits broad
`text_bubble` and `text_free` candidates.

## Required run sequence

Once all gold is complete, a release invocation must:

1. start with an empty isolated daemon state and result cache;
2. run one installed-but-process-cold complete chapter;
3. run one excluded warm-up chapter while retaining resident models;
4. run at least 20 complete measured warm chapters, removing only completed
   result-cache entries between inference runs;
5. run exact-cache replay in the same daemon instance;
6. test source replacement and same-tab navigation restoration; and
7. issue in-flight cancellation and measure both daemon cancellation and exact
   DOM restoration.

The minimum command is:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\Benchmark-Chapter5.ps1 `
  -Iterations 20 `
  -ResourcesDirectory C:\absolute\path\to\resources
```

Raw evidence belongs under:

```text
.cache/benchmark-evidence/30-years-since-the-prologue-chapter-5/<UTC stamp>/
```

`summary.json` may be created only after all required gates pass. Failed or
interrupted runs retain their raw failure evidence but are not completed
summaries.

## Separate live-site smoke

The deterministic release benchmark uses only repository-local bytes. A live
Asura smoke test is optional and separate because redirects, reader markup,
CDN state, consent pages, and network latency can change.

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\Invoke-LiveAsuraSmoke.ps1 `
  -ChapterUrl 'https://<current-reader-host>/<chapter-5-path>' `
  -ExtensionPackagePath 'C:\absolute\path\to\hskify-current.xpi' `
  -ResourcesDirectory 'C:\absolute\path\to\resources'
```

Report requested/final URLs and network timing independently. Live smoke
evidence is never merged into deterministic cold/warm percentiles.
