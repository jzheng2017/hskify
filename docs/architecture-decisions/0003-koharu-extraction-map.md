# ADR 0003: Koharu extraction map

## Problem

Browser mode needs a headless image-processing path, resumable progress, local
model loading, content-addressed blobs, and selectable-text metadata. Rebuilding
those systems beside Koharu would create two incompatible pipelines.

## Evidence

The pinned Koharu revision exposes the following reusable production surfaces:

| Need | Existing surface | Browser decision |
| --- | --- | --- |
| Pipeline invocation | `koharu_app::pipeline::run`, `PipelineSpec`, `Scope`, and `PipelineRunOptions` in `crates/koharu-app/src/pipeline` | Reuse unchanged behind a browser job adapter. |
| Headless proof path | `crates/koharu-app/bin/pipeline.rs` imports one image, invokes the production pipeline, and writes all artifacts | Keep as the Gate 0/3 diagnostic executable. |
| Engine discovery/loading | inventory-backed `Registry` and `EngineInfo` in `pipeline/engine.rs` | Reuse; browser mode selects the pinned defaults rather than adding engines. |
| Default vision stack | `pp-doclayout-v3`, `yuzumarker-font-detection`, `comic-text-detector-seg`, `speech-bubble-segmentation`, `paddle-ocr-vl-1.6`, and `lama-manga` in `config.rs` | Baseline before any evidence-backed replacement. |
| Progress/cancellation | `ProgressTick`, `WarningTick`, `AtomicBool`, `PipelineProgress`, `JobSummary`, and SSE snapshot events | Adapt to immutable browser status snapshots; retain Koharu cancellation. |
| Runtime downloads | `RuntimeManager`, `DownloadProgress`, and the application download registry | Reuse byte counts and states; add checksum policy in the browser model manifest. |
| Blob storage | `koharu_app::blobs::BlobStore` using immutable BLAKE3 references and a decoded-image LRU | Reuse within a browser-only cache root. Add the revision-keyed cache index above it. |
| OCR geometry | `TextData::line_polygons`, `Transform`, `rotation_deg`, and `confidence` | Normalize to protocol polygons and stable region IDs. |
| Style metadata | `TextData::style`, `font_prediction`, `detected_font_size_px`, and Koharu renderer types | Map only validated, browser-safe style fields into the result contract. |
| Local GGUF loading | `koharu_app::llm::Model`, `koharu_llm::ModelId`, `load_from_request`, and llama.cpp grammar support | Reuse with local targets only; browser mode never initializes provider targets. |
| Fonts | the renderer font book and `GoogleFontService` | Browser mode serves only installed/cached, licence-audited font bytes. |

The following browser-required information is not represented as a complete,
stable result by the pinned revision:

- bubble polygons are represented primarily as masks, not retained as one
  normalized polygon per logical region;
- OCR confidence is stored on a text node, but engine-specific alternatives and
  retry provenance are not a stable public contract;
- browser status polling needs a latest immutable snapshot rather than relying
  only on SSE delivery;
- the existing blob LRU is not a persistent, revision-keyed browser cache;
- the built-in local-model catalogue does not pin repository revision, byte
  size, checksum, or all inherited licence restrictions;
- full-page translation, two-pass HSK rewriting, and protocol-complete ordered
  region IDs are not existing Koharu behaviours.

Koharu's normal HTTP router is not suitable as the browser boundary: it exposes
the desktop/API surface, permissive CORS, and a much larger body limit than the
extension needs.

## Decision

Use `koharu_app::pipeline::run` as the only production image-pipeline driver.
The browser companion owns a thin adapter that:

1. imports verified image bytes into a hidden browser project;
2. invokes the existing selected engines;
3. snapshots progress for polling;
4. converts scene nodes, masks, blobs, and styles to protocol v1;
5. adds browser-only stable IDs, cache revisions, and HSK/model metadata; and
6. exposes those results only through the dedicated authenticated
   `/browser/v1` router.

Do not route browser requests through Koharu's general RPC router. Do not add a
second detector, OCR engine, inpainting stack, blob database, job runner, or
LLM runtime.

## Consequences

- Improvements and regressions in Koharu's production engines are shared with
  browser mode.
- Browser-only state remains isolated from a desktop project.
- Gate 3 must extend the adapter for retained bubble geometry and OCR
  provenance before claiming live cleaning acceptance.
- Model installation remains disabled until a licence-filtered benchmark and
  real fluent-reader review selects the standard and low-memory packs.
