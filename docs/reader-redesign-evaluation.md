# Reader redesign verification and release evaluation

The redesign changes source ownership, sentence-group publication, discovery, native admission, validation, retry, caching and dictionary lookup together. Its exact build fingerprint is `hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10`; result-cache schema is `hskify-source-revision-result-2026-10-01-v2`. Old packages must be replaced as a unit. See [architecture](architecture.md) and [contracts](browser-contract.md).

## Verification that can run from source

Run extension typechecking and unit tests from `extensions/firefox`:

```powershell
pnpm typecheck
pnpm test
```

Run production-module browser regressions from the repository root:

```powershell
node scripts/test-redesign-firefox.mjs
node --test scripts/benchmark/generalization-gates.test.mjs scripts/benchmark/document-chapter-gates.test.mjs scripts/run-document-chapter-benchmark.test.mjs scripts/run-image-runtime-benchmark.test.mjs
node scripts/benchmark/structural-gate-self-tests.mjs
```

The Firefox regressions exercise complete live text coverage; media, list and node retention; empty children and queued edits; ancestor removal and benign attributes; English classification and explicit short-region selection; sentence groups; source changes during hashing and before mounting; shared image eligibility; reused/empty canvases; incremental style-read counts; window and nested anchors; initial viewport focus halfway through a chapter; computed CSS edge offsets; same-origin frame capture coordinates and restoration; hidden story elements; publisher drawing-context ownership; discarded WebGL framebuffer fallback with transactional overlay restoration; and failure notices with long hidden source text. The representative prose case measures extraction plus mounting over 30 warm samples of 300 blocks and 100,000 characters against the existing 100 ms p95 bound. Browser evidence is written to `.cache/redesign/firefox-regressions.json`.

Initialize the pinned native toolchain before compiling. CPU tests check policy and transport without loading resident models:

```powershell
. .\scripts\Invoke-PerformanceBuild.ps1 -PrerequisitesOnly | Out-Null
cargo test -p browser-companion -p koharu-app -p koharu-llm -p hsk-control --no-default-features --features hsk-control/test-seeds --lib
cargo test -p browser-companion --no-default-features --test contract_fixtures
cargo test -p hsk-control --features test-seeds --tests
cargo test -p hsk-control --test performance_smoke -- --ignored
```

Native fault regressions include a lost job-creation response and concurrent retries, conflicting request-ID reuse, initial viewport focus before dispatch, hidden-document suspension and cancellation/replay, optional persistence failures, immutable context, revision removal of obsolete OCR tails, immediate out-of-order document publication, and dictionary interaction after 129 image jobs. LLM runtime tests explicitly ignored for lack of initialized runtime libraries are not counted as passes. The binary handshake requires the CUDA build:

```powershell
cargo test -p browser-companion --release --no-default-features --features cuda --test native_handshake
```

## Matching package and hardware

Use `scripts/Invoke-PerformanceBuild.ps1` for the final locked release build and its source/binary/toolchain/GPU attestation, then `installers/windows/Build-ReleasePackage.ps1` with the verified lexical artifacts, Qwen model, resident models/runtime and fonts. The builder verifies input identities and writes file hashes to `bundle-manifest.json`; check the assembled files against that manifest. Run packaging and installer regressions with `installers/windows/Test-ReleasePackage.ps1`. Build attestation must match the source tree at packaging; changing source invalidates it. Packaging and native framing do not establish translation quality or latency.

The document benchmark now accepts `--mode natural|strict`, `--hsk-level 1..6` and `--start-position top|middle`; its default is 30 samples. It records actual readable translated viewport commits. Native dispatch events, offscreen Chinese, pending source-shaped placeholders and preserved failures do not count as readable translation. The existing extraction, scroll-task, throughput and VRAM gates remain in place. Keep cold startup, resident misses, partial hits and full hits separate. A cache state must be established by the run setup and corresponding inference evidence, never inferred from a fast result alone.

## Observed local results on the matching r9 package

The local checks passed: 221 extension unit tests and typechecking; 21 production-module Firefox regressions; 146 companion, 93 application and 51 LLM library tests; 9 native contract fixtures; 34 HSK tests and its performance smoke; 26 harness/gate tests; release native framing; Windows PowerShell packaging/installer regressions; and all 68 assembled package file hashes. Ten LLM tests require separately initialized runtime libraries and remain ignored. Extension lint reports zero errors/notices and four warnings in bundled Readability's disposable-clone `innerHTML` handling.

The 30 warm extraction/mount samples measured 50 ms p95 for 300 blocks and 100,000 characters. Window and nested reading anchors stayed fixed. These measurements do not include model inference.

Matching packaged Firefox/native GPU smoke runs used synthetic sources with the RTX 4080 SUPER. The model/runtime were warmed before starting each source. These are individual observations, not p95 latency or independent translation-quality measurements:

| Source/mode | Observed selectable viewport result | Outcome |
| --- | --- | --- |
| Prose, natural level 3 | 3,172 ms | All 9 source groups committed Chinese |
| Same prose, strict level 1 | No viewport Chinese | 8 groups explicitly failed lexical policy/repair; 1 offscreen heading translated |
| Same prose, natural level 4 | 371 ms | All 9 groups committed Chinese after the level change |
| Selected image, natural level 3 | 6,827 ms | One selectable Chinese region, no overflow |
| Same image, strict level 3 | 48 ms | Cached analysis/reference reused; strict lexical validation passed |
| Same image, natural level 4 | 47 ms | One selectable Chinese region, no overflow |

Raw local evidence is in `.cache/redesign/firefox-regressions.json`, `.cache/redesign/packaged-smoke/report.json`, `.cache/redesign/packaged-image-smoke/report.json` and `.cache/redesign/package-integrity.json`. Natural prose had no structural omissions in this smoke case. The selected image required the explicit region-selection UI because a single illustration is ambiguous. The image miss exceeds the proposed five-second natural target, and the strict-level-1 passage falls far below the proposed coverage target. Both require further investigation and evaluation; the package is not release-qualified.

## r10 segmentation optimization and packaged measurements

Profiling a fresh resident-model miss of the same 700 by 640 synthetic image on r9 measured 6,933 ms to actual selectable viewport Chinese. Glyph-segmentation forward consumed 4,463 ms, compared with 597 ms detection, 313 ms page-role analysis and 982 ms faithful generation. The principal bottleneck was the segmentation encoder's channel-by-channel grouped CUDA convolution. r10 replaces its depthwise operations with a fused kernel; weights, resolution, BF16 storage and translation policy are unchanged. The pipeline cache identity changes to `immutable-context-fused-depthwise-pipeline-v3-2026-10-01`.

The matching packaged r10 Firefox/native run measured **2,768 ms** for a fresh resident-model miss of that unchanged image. Segmentation forward took 78 ms, with 10 ms preprocessing and 37 ms postprocessing. Detection took 694 ms, page-role analysis 329 ms and faithful generation about 1,055 ms. The displayed translation remained `我会等你。不要忘记我。`, with no translated-region overflow. Subsequent analysis/reference-cache reuse took 51 ms in strict level 3 and 46 ms in natural level 4. The 8,520 ms resident warmup occurred before measurement and is excluded. These three observations are not a p95 estimate.

Independent CPU-reference GPU tests cover F32 and BF16, batch and storage offsets, non-contiguous views, strides, padding and dilation. Direct old/new segmentation comparisons over the synthetic image and three repository image fixtures produced finite probabilities throughout. Three masks were identical. The 770 by 1,080 fixture had 88 changed pixels out of 831,600 and 99.796% foreground-mask intersection over union; mean probability difference was 0.00011235 and maximum difference 0.11328125. Small accumulation differences can change threshold decisions, so these checks do not establish identical quality on unseen artwork. First-use compilation is included in the isolated 397 ms first segmentation call; warmed synthetic calls took 150–158 ms including mask conversion.

Two further packaged runs used unique source bytes and text on one illustration, layout and Arial font, with the RTX 4080 SUPER and resident models. Every source was fully visible and selected through the region-selection UI. Browser observers measured actual selectable Chinese commits; native evidence confirms fresh generation for each source. Source quantities varied across samples, avoiding faithful-result reuse. No failed samples were discarded:

| Synthetic source, level 3 | Attempts | Readable translations | Failures | First-readable timing |
| --- | --- | --- | --- | --- |
| Natural: `I WILL WAIT N DAYS. DO NOT FORGET ME.`, N=1–30 | 30 | 30 | 0 | p95 **3,060 ms**; range 2,300–3,073 ms |
| Strict: same day-count dialogue, N=31–60 | 30 | 0 | 30 | No readable result; latency gate fails |
| Strict: `I HAVE N FRIENDS. DO NOT FORGET ME.`, N=1–30 | 30 | 19 | 11 | Published results took 2,080–3,266 ms; failures prevent a passing latency claim |

All published strict output passed lexical validation. Natural output is not required to do so; a sampled day-count translation contained `天`, which the packaged validator rejects. OCR sometimes merged the first two English words or confused `I` with `L`. These measurements have no independent bilingual quality assessment. The strict failures remain explicit and retryable. The timed runs exposed failure-notice overflow: long transparent source text consumed flex space and could clip Retry. The final renderer removes that text from flex layout and wraps the notice. Firefox regression tests verify contained, clickable Retry buttons in 350 px and 84 px source regions while preserving source metadata and the original image. Successful translated regions in the timed runs did not overflow. The total GPU-memory peak was 13,455 MiB, sampled nominally every 250 ms and including other desktop GPU users; it is not an isolated process allocation measurement.

Current checks passed: 221 extension unit tests and typechecking, 22 production-module regressions in Firefox 151, 146 companion library tests, 9 contract fixtures, 68 ML library tests, 26 harness/gate tests, and the release native handshake. The latest 30-sample prose extraction/mount p95 remains 50 ms. The separately invoked CUDA reference test and geometry test both passed; the default ML suite still lists three ignored tests. The packaged native and extension artifacts use r10 together. Timing reports above precede the final failure-notice CSS correction; that correction is verified separately through production modules in Firefox.

Run the hardware reference test using the existing packaged runtime:

```powershell
. .\scripts\Invoke-PerformanceBuild.ps1 -PrerequisitesOnly | Out-Null
$env:HSKIFY_RESOURCES_DIR = (Resolve-Path 'dist/hskify-windows-r10/resources').Path
cargo test --locked -p koharu-ml --release --no-default-features --features cuda --lib manga_text_segmentation_2025::depthwise::tests -- --include-ignored
```

Raw local evidence is in `.cache/redesign/segmentation-fused-comparison.json`, `.cache/redesign/packaged-image-profile/report.json`, `.cache/redesign/packaged-image-r10/report.json`, `.cache/redesign/packaged-image-r10-warm/` and `.cache/redesign/packaged-image-r10-strict-warm/`. Each repeated-run directory retains its report, metrics, native inference evidence, daemon log and GPU samples. The natural image timing meets five seconds for this controlled warm-miss workload. Strict coverage, other readers/modalities/cache states and the frozen out-of-sample release gates remain unqualified.

## Frozen out-of-sample corpus

`fixtures/reader-evaluation/manifest.json` seeds the existing 13 image development references. It deliberately has no invented source digests, strata, series IDs, annotation digests or human assessments. It is not evaluation-ready.

Required corpus: 13 development plus 13 blind image chapters and 12 development plus 12 blind prose chapters, with at least three wholly unseen providers per modality. Hold out entire series and providers. Record artwork, fonts, genre, layout and reader class, and include negative controls. Use licensed or otherwise authorized local source snapshots; freeze bilingual annotations before evaluation and tune only on development chapters.

Each manifest chapter supplies `id`, `modality` (`image` or `document`), `split` (`development` or `blind`), `provider`, `series`, `readerClass`, `negativeControl`, `strata` (`artwork`, `font`, `genre`, `layout`) and `annotation` (`path`, SHA-256 `sha256`, ISO `frozenAt`). Annotation paths are relative to the manifest. Frozen annotation JSON contains `chapterId`, `sourceSha256`, and unique `items` with `id`, `eligible`, and `critical` flags. Images additionally need independently transcribed source text and geometry in the annotation used by the browser/OCR runner. Negative controls must be annotated as ineligible; eligibility must not be decided from extension output.

Evidence JSON supplies `buildFingerprint`, ISO `evaluatedAt`, `runs` and `latencySamples`. Every chapter needs exactly one run for `natural` and each of `strict-1` through `strict-6`. Each run supplies `chapterId`, `mode`, frozen `sourceSha256` and `items`. Each eligible item must appear exactly once as `translated` or `failed`. Translated items require independent bilingual assessment (`method: independent-bilingual`, at least two distinct reviewer IDs, boolean `faithful` and `grammarAppropriate`). Strict output requires `strictLexicalValid: true`. Every eligible item supplies explicit `restorationCorrupted: false`, `staleOverlay: false` and integer `renderingDefects`. Image items also supply OCR `referenceCharacters` and `editDistance`, including failures; omission of failed OCR must not bias OCR error rates.

Cover unseen names, commands, negation, participant reversal, repeated/signed quantities, fragments, quotations, long prose and critical meaning regressions. Structural checks are evidence of structure only. Vocabulary validation does not establish grammar difficulty, semantic fidelity, role recognition or OCR calibration. Human assessment must use source and output, independently of the model that produced the translation.

Run the combined gate after collecting real evidence:

```powershell
node scripts/benchmark/generalization-gates.mjs fixtures/reader-evaluation/manifest.json EVIDENCE.json REPORT.json
```

The gate verifies frozen annotation digests and identities, corpus splits, provider/series separation, complete item accounting and independent review. It reports story coverage, faithful accepted rate, grammar assessment, HSK compliance, abstention, OCR error and rendering defects separately by split, modality, reader class and mode. Coverage targets are 95% natural and 90% strict; faithful accepted translations must reach 95%. Critical semantic cases must translate faithfully. Source corruption, stale overlays, silent omissions and published strict lexical violations fail release.

Latency samples require a unique `id`, `modality`, `mode` (`natural` or `strict`), verified `extensionFingerprint` and `nativeFingerprint`, Firefox `browser.name/version`, GPU identity `NVIDIA GeForce RTX 4080 SUPER`, `packaged: true`, `buildAttestationVerified: true`, verified `cacheState` (`cold`, `resident-miss`, `partial-hit`, `full-hit`), `startPosition` and `startPerformanceMs`. Supply chronological or timestamped browser `events`, all expected `sourceItems` (`id`, boolean `visibleAtStart`) and matching `itemCompletions` (`id`, `visibleAtStart`, `performanceMs`). Missing source accounting fails the gate. First-readable latency is the earliest visible/readable translated selectable-text commit after start. Each modality/mode requires at least 30 warm samples, all four cache states and starts halfway through chapters. Warm p95 limits are five seconds natural and ten seconds strict; initially visible work must finish before background work.

## Remaining evidence and acquisition limits

Synthetic and unit regressions cannot substitute for the missing annotated/blind corpus or independent bilingual review. The complete release quality and latency targets remain unverified until those evaluations are completed, despite the controlled r10 natural-image latency improvement above. Two preprocessing views of the same OCR recognizer still need development calibration before their agreement can be interpreted as reliable confidence.

Firefox rendered-region fallback requires a fully visible selected surface; a tall offscreen WebGL surface may need a smaller region or must fail visibly. Same-origin frames use shared discovery; inaccessible cross-origin frames abstain. Normal remote images verify current bounded bytes through the existing authenticated background acquisition, while captured surfaces verify captured content. Animated/continuously changing surfaces may abstain rather than commit stale output. Image language context is frozen from available neighboring OCR at admission; it can be sparse when starting in the middle of an unseen chapter. Faithful generation now constrains response structure and Latin leakage using the existing grammar sampler. English capitalization is required for a protected name exception; ambiguous lowercase names may receive no exception or cause strict abstention. Capitalization does not prove that a word is a name, especially in uppercase OCR. These reader classes and conditions must be represented in held-out evaluation.
