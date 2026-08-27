# Hskify browser companion implementation

`browser-companion` is the Windows/CUDA native half of Hskify's current-chapter
Firefox reader. It builds two executables:

- `hskify-native-host`: the one-shot Firefox native-messaging launcher;
- `hskify-browser-daemon`: the authenticated loopback job daemon.

The product supports one exclusive `image` or `document` job per detected
source. Image jobs translate manga/webtoon raster text; document jobs translate
the ordered DOM text snapshot extracted from a light-novel chapter. They share
one language model, correctness policy, context ordering, cache identities,
lookup ownership, update log, cancellation, and acknowledgement model.

## Fixed identity and target

The TypeScript and Rust contracts contain exactly:

```text
hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-08-09-r8
```

The install identities are:

```text
native host: local.hskify.browser
Firefox ID:  hskify@local.hskify
```

There is no protocol negotiation, route alias, old-host cleanup, stored-schema
migration, or compatibility parser. A mismatched fingerprint fails before a
job starts.

The production feature set is Windows x86-64 MSVC, CUDA 13.1, compute
capability 8.9, and the supported RTX 4080 SUPER 16 GB workstation. The release
wrapper builds both binaries, records their exact hashes in the source/hardware
attestation, and packages one resource pack for both modes.

## Native launch and trust boundary

The Firefox registration manifest permits only `hskify@local.hskify` and points
to an absolute `hskify-native-host.exe`. The host validates its own executable
and manifest identity, starts or discovers the daemon, and requests a short-
lived bearer session bound to the canonical `moz-extension://` origin.

The daemon binds a random IPv4 loopback port. Browser requests must have the
exact loopback Host, bearer token, and active extension origin. Privileged
Firefox requests use `X-Hskify-Extension-Origin` when standard `Origin` is
absent. Only the native launcher may call `/browser-internal/session`, using
`X-Hskify-Control`; that endpoint is not CORS-enabled.

## Runtime ownership

`KoharuPipeline` has two independent lazy residents:

- `LanguageRuntime`: the one Qwen model instance, CUDA runtime manager, HSK
  data, dictionary data, and shared translation/cache dependencies;
- the image/vision resident: detector, OCR detector and recognizer, text and
  bubble segmenters, inpainter, multimodal projector/page adjudicator, and
  layout/style resources, all reusing `LanguageRuntime`.

`POST /warmup` takes `{"kind":"document"}` or `{"kind":"image"}`.
Document warm-up calls only the language resident. Image warm-up first ensures
language readiness and then initializes the vision resident. Instrumented
document tests must observe zero detector, OCR, projector, segmentation,
inpainting, patch, and font loads or invocations.

`GET /setup` is deliberately limited to verified resource installation state;
it does not allocate either resident. After a detected page becomes resource-
ready, Firefox issues the tagged `POST /warmup` and retries a previously
detected kind through installation or an in-progress warm-up.

There is one serialized cancellation-safe CUDA language lane and one serialized
vision lane. Document and image translation never create a second Qwen model.

## Exact browser API

The daemon mounts these unversioned routes:

| Method | Route | Meaning |
| --- | --- | --- |
| `GET` | `/health` | fingerprint, readiness, and exact resource identities |
| `GET` | `/setup` | installed resource state |
| `POST` | `/setup/models` | start/report managed resource setup |
| `POST` | `/warmup` | initialize the selected runtime boundary |
| `POST` | `/jobs/image` | multipart raster plus image metadata |
| `POST` | `/jobs/document` | complete JSON document snapshot |
| `PUT` | `/jobs/{jobId}/focus` | tagged image rectangles or visible block IDs |
| `GET` | `/jobs/{jobId}/updates` | sequence replay/long poll |
| `DELETE` | `/jobs/{jobId}` | cancel and release a job |
| `DELETE` | `/chapters/{pageSessionId}` | release canonical chapter context |
| `POST` | `/lookup` | item-owned local dictionary lookup |
| `GET` | `/blobs/{blobId}` | authorized image cleanup patch |
| `GET` | `/fonts/{fontId}` | allowlisted installed image font |

Serde contracts deny unknown fields. The server rejects a focus variant that
does not match the job modality.

## Document request and pipeline

`POST /jobs/document` accepts at most 1 MiB of UTF-8 JSON. The strict request
contains `buildFingerprint`, `pageSessionId`, `sourceSha256`, `settings`, and
`blocks`. There may be at most 2,000 blocks and one block may contain at most
16 KiB of UTF-8 text.

Every block contains:

- stable `itemId`;
- `sourceIndex` and `itemOrder`;
- kind `prose`, `heading`, `dialogue`, `caption`, `thought`, or `sfx`;
- provenance `dom`; and
- authoritative normalized English `text`.

Document contracts reject OCR provenance and layout constraints. The daemon
repeats canonical normalization and SHA-256 calculation over the complete
ordered block sequence; a different `sourceSha256`, duplicate item/order,
invalid bound, empty source, or unsorted source fails before publication.

All blocks are registered before translation. The pipeline builds canonical
context by `(sourceIndex, itemOrder)` and then applies focus priority. Up to 64
unique `visibleBlockIds` may be reported. If a visible block exists, its first
dispatch is a one-block batch; later batches contain at most six real-tokenizer
units. Dispatch order never changes surrounding context or terminal output
order.

The resident context budget is 4,096 tokens. A block that fits alone is never
split. An individually oversized block is segmented at ICU sentence boundaries
while line-break separators are retained. Its pieces can be translated in
bounded batches, but the daemon joins and validates the complete block before
one update is visible. If splitting, translation, joining, or final validation
fails, `documentBlockPreserved` records a source-preserving terminal block. The
browser withholds that source English in Chinese mode while Original mode keeps
the untouched site content.

## Image request and pipeline

`POST /jobs/image` accepts the raster as multipart `image` and the strict JSON
metadata as multipart `request`. Metadata includes client image identity,
source hash/MIME/dimensions, page session and canonical source order, surface
kind, reading direction, learning settings, and initial normalized rectangles.

The server checks body, encoded type, sniffed type, SHA-256, declared/decoded
dimensions, pixel count, and decoder allocation. The image pipeline owns only:

1. tile planning and viewport priority;
2. text proposal detection and OCR;
3. visual story/furniture/artwork adjudication;
4. text/bubble segmentation and local inpainting;
5. patch, typography, style, and layout construction.

Accepted source spans use provenance `ocr`, so the shared language adapter adds
only the OCR-correction instruction. Source layout is attached only to image
items. A translated image item stores and authorizes a valid transparent PNG
before publishing `imageRegionReady`. A terminal non-translated item publishes
`imageRegionPreserved`, leaving source pixels intact.

## Shared language correctness

`TranslationService` accepts generic ordered source spans and contains no comic,
panel, bubble, or OCR language in its shared prompt. The two-stage policy is:

1. establish faithful Simplified Chinese;
2. for `natural`, publish it with deterministic HSK teaching metadata;
3. for `strict`, perform HSK realization and at most one terminal repair;
4. publish no provisional Chinese.

DOM text remains authoritative. OCR inputs alone receive correction guidance.
Deterministic validation covers output structure, source echo/Latin leakage,
names, numbers, question intent, HSK validity, pinyin, and teaching-term ranges.

The shared terminal `TranslatedText` contains `sourceText`, `baseChinese`,
`displayedChinese`, `pinyin`, and final HSK state. It is embedded in
`imageRegionReady` and `documentBlockReady`. Preserved variants contain source
identity/text and a reason, never a draft translation.

## Updates, completion, and recovery

Each job has one append-only sequence beginning at 1. Valid update types are:

- `progress`;
- `imageRegionReady` and `imageRegionPreserved`;
- `documentBlockReady` and `documentBlockPreserved`;
- terminal `complete`, `failed`, or `cancelled`.

`complete` reports exact `translatedCount` and `preservedCount`. The job store
rejects duplicate item publication, regressive progress, updates after a
terminal event, and modality-incompatible updates.

`GET /jobs/{jobId}/updates?after=N&waitMs=M` returns only later updates and
waits at most the contract limit. Each response is one contiguous page of at
most 1,024 updates from a log retaining at most 10,000. The browser acknowledges
after DOM install and requests the next page; there is no separate status or
result model. An unacknowledged update may replay after MV3 suspension and is
installed once by `itemId`.

Job/artifact identity includes an exact `image | document` source tag and
source hash. A changed document hash or different modality cannot recover stale
output. Fatal job failure and cancellation release job-owned patches/contexts;
the browser restores the live source page.

## Lookup and cache identity

Dictionary hover lookup is owned by `jobId` and `itemId`. The daemon resolves a
Unicode offset against its canonical final displayed Chinese and returns the
longest expression starting at that character. Selection lookup is bounded and
does not trust browser-supplied translated context.

The completed-result cache has one current schema with tagged image/document
entries. Image entries contain final image items, patches, preserved items, and
lookup contexts. Document entries contain final ready/preserved blocks and
lookup contexts. The key includes modality, complete source hash, learning mode,
level, surrounding canonical context, model/prompt/validator identities,
tokenizer and HSK/dictionary resources, and output-affecting pipeline resources.
There is no reader for an earlier schema.

## Resource installation and environment

The packager installs one verified resource pack under `%LOCALAPPDATA%\Hskify`.
Supported overrides use only generic names:

- `HSKIFY_STATE_DIR`;
- `HSKIFY_RESOURCES_DIR`;
- `HSKIFY_HSK_PATH`;
- `HSKIFY_DICTIONARY_PATH`;
- `HSKIFY_QWEN_MODEL_PATH`.

Debug timing/rejection flags also use the `HSKIFY_*` prefix. There are no
novel-specific resources or deployment settings.

Local acceptance runs set `HSKIFY_BENCH_EVIDENCE_PATH` to an initialized
`{"samples":[]}` file. Native upserts document dispatch/runtime counters and
image language-unit/generation-duration measurements by tagged `(kind, jobId)`;
the browser benchmark runner consumes this native output directly. Image units
are counted once at faithful OCR-authoritative generation. Timed wall time
covers admitted faithful, strict-primary, and terminal-repair model calls only;
strict/repair do not increment units, and queue, cache, and validation time are
excluded. Cache replay intentionally produces no generation sample.

## Verification

Relevant local checks include:

```text
cargo test -p browser-companion --all-targets -j 1
cargo test -p koharu-app --all-targets -j 1
cargo test -p koharu-llm --all-targets -j 1
cargo clippy -p browser-companion -p koharu-app -p koharu-llm --all-targets -j 1 -- -D warnings
```

Contract fixtures cover image/document creation, focus modality, ready and
preserved events, completion counts, unknown fields, limits, hashes, and replay.
Native document tests cover token packing, mid-chapter context, joined-block
validation, strict repair, per-block preservation, cache separation, and the
language-only runtime boundary. Existing image and real-reader regressions
remain mandatory under the generic identities.
