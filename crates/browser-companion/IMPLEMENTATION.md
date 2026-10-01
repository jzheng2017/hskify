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
hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10
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

## Pipeline and release verification

The exact r9 source/retry/publication/lookup contracts are documented in [browser-contract.md](../../docs/browser-contract.md). Implementation responsibilities and bounds are documented in [architecture.md](../../docs/architecture.md); these are the maintained descriptions of the redesigned pipeline.

`server.rs` owns authenticated admission, creation-id deduplication, update logs, cancellation, job eviction and a bounded optional persistence writer. `pipeline_adapter.rs` owns shared language policy, resident runtime admission, immediate validated publication, immutable context and separate HSK/faithful/page-analysis caches. `result_cache.rs` owns the current exact disk schema and indexed pruning. `chapter_session.rs` retains bounded source context in canonical chapter order.

CPU library verification uses `cargo test --lib -p browser-companion -p koharu-app -p koharu-llm -p hsk-control --no-default-features --locked`. GPU binaries require the `cuda` feature; the native handshake integration test must be run against built CUDA binaries. `scripts/Invoke-PerformanceBuild.ps1` supplies the pinned build environment and records matching source/binary/hardware attestation. Do not describe CPU library tests as packaged/GPU performance evidence.

The fault regressions include concurrent/lost-response creation, cancellation and replay, optional cache-write failure, immutable-context keys and dictionary lookup after more than 128 image jobs. The existing contract fixtures deny unknown fields and bind results to the full source identity.

Independent annotated blind chapters and matching packaged Firefox/RTX 4080 SUPER latency samples remain release requirements in [reader-redesign-evaluation.md](../../docs/reader-redesign-evaluation.md).
