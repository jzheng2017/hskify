# 30 Years Since the Prologue chapter 5 release benchmark

This is the sole canonical end-to-end benchmark for the direct progressive
Firefox build. It replaces every earlier chapter workload.

No performance or correctness result is committed here. A result is valid only
after the fixture has complete reviewed gold for all 36 images and the harness
finishes every required run while retaining its raw evidence.

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
images after the post-group OCR gate, semantic exclusion guard, and
letter-adjacent OCR-digit handling were installed. It used the unversioned
product HTTP routes and exact fingerprint
`hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-26-r2`:

| Diagnostic | Result |
| --- | ---: |
| Accepted progressive regions | 201 |
| Sum of per-image completion times | 66,803 ms |
| Per-image p95 | 3,691 ms |
| Slowest image | 4,288 ms |
| Credits-cover accepted regions | 0 |

The ignored raw diagnostic is
`.cache/ch5-product-audit/run-30-final/summary.json`. This is useful runtime
evidence, but it is deliberately not called release evidence: it does not
include packaged Firefox acquisition/rendering, 20 warm iterations,
cache-replay, memory/VRAM sampling, cancellation, DOM ordering, overflow, or
the benchmark's final one-to-one correctness scoring.

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

The measured clock begins at the benchmark-only content start message and
includes image acquisition, hashing, upload, daemon work, patch retrieval,
browser decoding, and DOM commit. Optional live-site navigation and CDN time
are recorded separately.

## Required gates

Every measured warm run must meet:

| Metric | Required p95 |
| --- | ---: |
| HUD acknowledgement | <=100 ms |
| Exact cached first viewport | <=250 ms |
| First visible region | <=2 s |
| All initially visible regions (one or more) | <=5 s |
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

For detector-only evidence, retain one raw detector JSON per source image and
score it with:

```powershell
python .\scripts\benchmark\score_chapter5_detector.py `
  --predictions C:\absolute\path\to\raw-detector-json `
  --sources C:\absolute\path\to\source-webps `
  --output C:\absolute\path\to\detector-evidence.json
```

The scorer must derive denominators from the committed manifest and reviewed
annotations. No page, region, or category count may be hard-coded in the
runner.

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
  -DetectorEvidencePath C:\absolute\path\to\detector-evidence.json `
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
