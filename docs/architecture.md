# Hskify chapter architecture

Hskify translates a current English chapter into selectable Simplified Chinese in Firefox, using one authenticated local companion and one resident Qwen3.5-4B model. The current browser/native fingerprint is `hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10`. Contracts and caches change together; no compatibility parser or schema migration is retained.

## Source acquisition and ownership

`classifyChapter` supplies the same classification to passive warmup and active translation. Readability selects a candidate prose region from a disposable clone; extraction then walks eligible live text in that entire region. Direct section text, nested lists, inline media and original node identities are retained. Bounded samples from the beginning, middle and end must consistently classify as English with bundled `franc-min`. Short or ambiguous sources support explicit region selection. Limits reject oversized snapshots and allow selection of a smaller region.

The canonical document consists of immutable original Text values, disjoint source ranges, stable parent identities and ordered sentence groups. The default group bound is 600 UTF-16 code units; native token planning additionally checks actual completion capacity. Rendering owns inserted spans and explicitly tracked Text writes. It never replaces an original element's child list. Pending source-shaped spans retain layout. Original and Compare restore the actual source nodes, including external edits.

The reader drains queued external mutations before disconnecting its observer for owned writes. Every install checks the source's revision and connectivity. Ancestor removal invalidates the reader; empty added children and lazy illustration URLs do not adopt Chinese as original or cancel prose. Re-extraction retains translations only when both the item and its bounded source neighborhood still match. Window and nested scrolling keep a semantic reading anchor during mounting, publication and comparison.

`LiveSurfaceDiscovery` is the sole image/background/canvas/frame registry. Initial admission inspects the DOM, with shared control/owned/hidden exclusions and cached ancestor eligibility. Subsequent scans inspect changed subtrees and source attributes. IntersectionObserver tracks actual visibility; nearby prefetch is separate. Extension-owned mutations do not trigger discovery. Same-origin frames use the same adapters and observers.

Surface identity belongs to the element; `chapterSourceOrder` separately supplies current canonical DOM order, including pages inserted before existing work. Captured bytes have SHA-256 revisions. Active visible canvases and backgrounds are sampled at one-second intervals, at most two per interval; hidden tabs suspend sampling. Acquisition never chooses a drawing context on a publisher canvas. Transparent/uniform captures abstain. Default WebGL buffers and unsupported image/background geometry fall back to Firefox rendered-region capture with overlays transactionally hidden and restored. This fallback requires the entire selected surface to be visible; unsupported captures remain visibly retryable.

## Scheduling, publication and lifecycle

`ChapterController` owns classification/navigation. `ChapterRunController` owns the explicit run phase, generation, cancellation signal, chapter lifetime and update stream. The mode adapters own only source-specific discovery, rendering and item state. A settled run remains open for retry and new content. The HUD says all discovered content is processed; it does not declare an unknown lazy chapter permanently complete.

Both pipelines reuse the existing token planner, serialized CUDA language lane, separate vision lane and HSK control. Visible document groups are dispatched before background groups, and each validated group is published immediately. A completed prefix beginning at block zero is not required. Stable parent/sub-item metadata lets publication arrive out of order while DOM order remains unchanged. Focus changes update queued priorities and hidden-tab activity.

The glyph-segmentation encoder uses one fused CUDA launch for each depthwise convolution instead of Candle's separate convolution and temporary tensor per channel. The private operation preserves model weights, input resolution and BF16 storage, accumulates in FP32, and rounds at output. It validates geometry and contiguous storage before launching. NVRTC compiles the embedded kernel once per process using the existing packaged runtime; device modules reuse Candle's cache. Ordinary convolutions and CPU inference use the existing operations. No additional deployment files or configuration are required.

Document jobs register the full source context before generation. Image jobs freeze currently available surrounding English context once at admission. Generation and result-cache lookup/storage use that same immutable snapshot, independent of concurrent completion. Context traversal visits only bounded neighboring utterances. Image context is therefore reproducible for a job, but newly available neighboring OCR can improve a later job; quality evaluation must cover this limitation.

## Language policy and caches

The shared translation service checks HSK-specific cache entries before inference. A separate faithful cache reuses Chinese and source-anchored protected names across levels. Image page-analysis caching reuses successful OCR, role/bubble plans, cleanup decisions and faithful text across HSK changes and targeted retries. Failed OCR or cleanup is reprocessed. Each memory cache is bounded to 64 MiB.

Natural mode accepts valid faithful Chinese directly. Strict mode also accepts it directly when lexical policy and image layout limits already pass. Otherwise HSK realization permits at most one terminal repair. Generation reports stop versus token-limit termination; truncated output is rejected. The token planner must reserve adequate completion capacity instead of shrinking it silently.

Faithful generation uses the existing grammar sampler to constrain numbered rows, Chinese text and name metadata. This constrains transport and Latin leakage, not semantic or grammatical fidelity. Unsupported lowercase name metadata cannot grant vocabulary exceptions. Protected names include exact source and Chinese forms and a reason; they remain anchored through repair, validation, caching, teaching metadata and lookup. Explicit question punctuation and normalized numeric values/signs/multiplicity are structural evidence. They do not establish preservation of negation, participants, clauses or grammar difficulty. Independent bilingual assessment supplies semantic/grammar evidence; model self-scoring is insufficient.

Full-result persistence is optional. One bounded writer handles completion writes; persistence failure cannot fail an already translated item. Disk pruning uses a size/age index initialized once instead of scanning the directory after each result. Invalid entries become cache misses. The current result-cache schema is `hskify-source-revision-result-2026-10-01-v2`.

The r10 pipeline identity is `immutable-context-fused-depthwise-pipeline-v3-2026-10-01`. The changed build and pipeline identities invalidate earlier cached results without a schema migration.

## Transport, retry and interaction

Job creation requires a stable `clientRequestId`. Identical retries reuse a single job; conflicting reuse is rejected. The bounded creation ledger serializes concurrent retries. The complete source remains in targeted retry requests, while `retryItemIds` selects failed items for reprocessing. Successfully displayed items survive failure and retry. Image exclusions have an explicit `excluded` disposition; translation failures have `failed`, a visible notice and Retry. Prose failures offer Retry and Original/Compare.

The append-only log and poll/install/ack loop are shared. An acknowledgement follows successful DOM work. A newly recovered renderer replays from zero because its installed state is empty. Source changes, cancellation and navigation revoke result ownership and cancel native work.

Dictionary lookup uses bounded renderer-supplied item context through the existing pure local lookup implementation. It does not depend on retained native jobs. Job eviction can release blobs and native state while displayed Chinese, pinyin, dictionary lookup and local Mandarin speech remain usable.

## Bounds and verification

Document requests remain bounded to 1 MiB UTF-8 JSON, 2,000 groups and 16 KiB per group, with at most 64 focused IDs. Image requests retain the 20 MiB/25-million-pixel/16,384-dimension bounds. The language context remains 4,096 tokens. Windows/Firefox/CUDA 13.1/sm89 deployment and one shared resident resource pack are unchanged.

See [browser contracts](browser-contract.md), [release evaluation](reader-redesign-evaluation.md) and the production-module Firefox regressions in `scripts/test-redesign-firefox.mjs`. Extraction/mounting, scroll tasks, VRAM, throughput, first-readable latency and independent out-of-sample quality are distinct release gates. Passing unit or synthetic browser tests does not establish model quality on unseen chapters.
