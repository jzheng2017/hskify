# Hskify performance architecture

Hskify is a local Firefox-to-native pipeline specialized for low time-to-first
translated dialogue on one CUDA machine class. The browser and daemon exchange
an append-only sequence of small region updates; they never wait for or
transfer a reconstructed page.

## System shape

```mermaid
flowchart LR
    Reader["Firefox reader page"]
    Extension["Hskify extension"]
    Host["One-shot native host"]
    Daemon["Loopback daemon"]
    Scheduler["Viewport-first tile scheduler"]
    Vision["Resident CUDA detector and OCR"]
    Translator["Resident Qwen3.5 4B direct HSK translator"]
    Control["HSK validation, pinyin, dictionary"]
    Patch["Region-local transparent PNG patch"]
    Overlay["Patch-first selectable overlay"]
    Speech["Local Mandarin Web Speech voice"]

    Reader -->|"explicit action"| Extension
    Extension -->|"native handshake"| Host
    Host -->|"start or discover"| Daemon
    Extension -->|"authenticated unversioned routes"| Daemon
    Daemon --> Scheduler --> Vision
    Vision --> Patch
    Vision --> Translator --> Control
    Patch --> Daemon
    Control --> Daemon
    Daemon -->|"flat sequenced updates"| Extension
    Extension --> Overlay --> Reader
    Extension -->|"resolved Chinese"| Speech
```

## Build affinity and trust boundary

`hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-28-r7` is compiled into the TypeScript and Rust
contracts. Native handshake requests and responses, health responses, and job
creation metadata must carry that exact value. A different value is rejected.
There is no protocol-version header, range negotiation, compatibility shim, or
migration adapter.

The Windows release wrapper writes a post-success, ignored JSON attestation
covering the complete tracked and untracked-nonignored source identity, exact
x86_64 MSVC release/CUDA configuration, pinned toolchain and llama.cpp tag,
device-0 hardware/driver claims, and SHA-256/byte identities for both native
binaries. Packaging and benchmark preflight reject missing, stale, mutated, or
nonmatching attestations.

The native host accepts only the registered
`local.hskify.hsk_manga` manifest whose executable resolves to the running
binary and whose sole allowed extension is
`hsk-manga-translator@local.hskify`. It asks the daemon for a fresh 256-bit
bearer token bound to the exact canonical `moz-extension://` origin.

The daemon binds to a random `127.0.0.1` port. Browser requests require the
exact `Host`, the active extension origin (standard `Origin` and/or
`X-HSK-Manga-Extension-Origin`), and the bearer token before request-body
polling. CORS allows only that origin and the required GET, POST, PUT, and
DELETE methods. No general application router, URL fetch, telemetry, provider
credential, or remote translation path is mounted.

## Direct progressive data flow

1. The extension uploads one raster plus strict metadata to `POST /jobs`.
   Byte, MIME, hash, declared dimension, decoded dimension, pixel, and decoder
   allocation checks occur before the job begins.
2. The source is decoded once. The adapter divides it into 2,048-pixel tiles
   with 410-pixel overlap and reprioritizes remaining tiles whenever the
   viewport changes.
3. The pinned RT-DETR-v2 comic detector processes true CUDA batches of up to
   six tiles at its trained 640-pixel input size. Both `text_bubble` and
   `text_free` proposals continue to recognition; bubble rectangles are not a
   prerequisite. This covers dialogue, thoughts, captions, and unballooned
   story text while leaving the final semantic decision to OCR. Text proposals
   are spatially deduplicated at tile overlaps.
4. PP-OCRv5 English recognition runs in batches of eight. Mechanically valid
   Latin OCR at confidence 0.45 or higher remains eligible; hard-coded content
   word lists do not decide whether a line is story text.
5. Learned bubble segmentation assigns lines to real bubble identities so an
   entire balloon is processed atomically. Learned manga-text segmentation
   produces one stitched probability field that also drives OCR line discovery
   and per-line appearance sampling. Shared geometry expansion grows its mask
   around the detected glyphs, and manga LaMa restores the source artwork.
   Transparent region patches take alpha only from that semantic mask. Layout uses the
   measured, eroded bubble contour rather than a fixed detector-box inset.
6. Visible accepted regions jump ahead of off-screen work at every batch
   boundary. Translation batches contain up to six regions and normally flush
   once three are pending; an undersized visible tail waits no longer than
   75 ms so visibility changes do not create one-item GPU calls.
7. Qwen3.5 4B makes one contextual semantic decision per accepted OCR region:
   translate it directly to HSK 2.0-targeted Simplified Chinese, or return the
   typed `[NON-STORY]` disposition for unrelated page furniture such as a
   publisher/site credit, watermark, advertisement, or navigation label.
   Excluded regions retain their original pixels and never enter repair. The
   protocol forbids exclusion for story dialogue, narration, thoughts,
   captions, in-story text, names, roles, fragments, and emphasis.
   `hsk-control` validates vocabulary; the direct
   protocol validator checks protected names, standalone numbers, question
   intent, and output structure. Digits embedded in Latin OCR tokens such as
   `IDENTIT4` are not treated as semantic numbers. Only rejected items may
   enter one targeted repair batch, whose distinct bounded strategies run
   unless an earlier attempt succeeds. Each rejected candidate refreshes a
   typed validator avoid-list that strict repair must remove. Natural repair
   remains Natural across all attempts so it cannot discard an indispensable
   story concept merely to achieve a strict score. Natural learning accepts a bounded number
   of indispensable advanced terms after simplification and publishes their
   exact offsets, pinyin, definitions, and required level for hover teaching;
   strict mode accepts only level-valid non-name vocabulary. Explicit
   higher-HSK headwords stay atomic, while ordinary dictionary phrases made
   entirely from selected-level HSK headwords are counted by
   those surface words instead of being misclassified as advanced vocabulary.
   This is not a page-wide faithful pass followed by an HSK rewrite.
8. For each completed region, the daemon stores the patch blob first and then
   appends `regionReady`, which carries the patch descriptor, geometry, source
   text, base/direct Chinese, displayed Chinese, pinyin, style, layout, and HSK
   status. Ordered color bands preserve real foreground/outline changes between
   source lines, and Firefox keeps that band count while fitting the translation.
9. Firefox fetches and validates the PNG, decodes it, inserts it in the patch
   layer, and only then inserts the selectable text. Later updates continue
   independently.

Completion is a terminal event in the same log. It does not unlock a separate
result representation.

## Live-page rendering

The renderer never reparents, replaces, hides, or rewrites the reader's source
`img`. One document-anchored shadow-DOM portal shares the image's scroll
coordinate space and contains only transparent patch and selectable-text
layers. Normal document scrolling therefore stays compositor-only. Nested
scrollers trigger a position-only update, while resize and responsive layout
changes trigger a complete geometry/text refit. Cancellation or navigation
removes the portal and leaves the untouched source DOM in place.

The original/Chinese/hold-to-compare controls live in a separate fixed
shadow-DOM host at the viewport edge, so they remain reachable while reading a
long chapter. Pointer hit-testing maps a hovered rendered glyph to a Unicode
character offset. The daemon then performs a dictionary longest-match anchored
at that exact offset; selection remains an explicit fallback. The explanation
is placed outside the resolved glyph range when space permits, clamped to the
viewport, and dismissed on scroll, resize, or pointer departure.

## Scheduling and cache identity

The daemon stays warm for a 30-minute idle window and uses four Tokio workers,
at most eight general blocking threads, one serialized priority CUDA
scheduler, and one dedicated six-thread Rayon pool for browser image
preprocessing. The comic detector, OCR recognizer, local LLM
application state, and HSK control data are lazy `OnceCell` residents, so later
jobs reuse loaded state.

The 64 MiB byte-bounded in-memory translation cache is keyed by:

- normalized OCR text;
- up to six preceding dialogue utterances;
- the complete proper-name glossary;
- requested HSK level;
- natural or strict learning mode;
- model ID and exact model revision;
- prompt hash;
- validator hash; and
- the full HSK/dictionary control revision.

Changing any output-affecting dependency invalidates the cache. There is no
project cache, page history, stored page reconstruction, or level-change
retranslation endpoint.

Decoded images use a 512 MiB byte-bounded LRU. Completed per-image region
results and PNG patches also have a byte-bounded
2 GiB persistent cache. Its key includes the complete strict job request,
source hash, exact build fingerprint, and a fingerprint of every
output-affecting model, prompt, validator, dictionary, and pipeline resource.
Entries are atomically installed only after visible processing completes.
Stores enforce the 2 GiB bound and perform eviction once; reads open the exact
SHA-keyed entry directly instead of rescanning the cache directory for every
image. Every upload still validates its byte limit, SHA-256, encoded format,
MIME, declared limits, and header dimensions. An exact hit then reuses the
previously fully decoded/validated result; full pixel decoding occurs only on
a miss. No detector, OCR, translation, or patch intermediate is written to
disk.

The job log is append-only, starts at sequence 1, rejects regressive overall
progress, rejects duplicate region publication, and permits one terminal
`complete`, `failed`, or `cancelled` event. Clients long-poll after the last
acknowledged sequence, so extension background suspension does not require a
second status/result model.

## Resource envelope

| Resource | Default |
| --- | ---: |
| Image multipart field | 20 MiB |
| JSON metadata field | 64 KiB |
| Complete HTTP body | 21 MiB |
| Decoded pixels | 25,000,000 |
| Either dimension | 16,384 px |
| Decoder allocation | 128 MiB |
| Decoded-image LRU | 512 MiB |
| In-memory translation cache | 64 MiB |
| Persistent completed-result cache | 2 GiB |
| One patch blob | 16 MiB |
| Retained jobs | 128 |
| Retained sources and patches | 256 MiB |
| Authenticated in-flight requests | 64 |
| Updates per job | 10,000 |
| One update long-poll | 20 seconds maximum |
| Idle daemon window | 30 minutes |

Terminal inactive jobs are evicted oldest-first when job or byte capacity is
needed. Active jobs are never eviction candidates. Patch blobs are owned by
one job and removed with it.

## Hardware boundary

The performance build is CUDA-only and gated to an NVIDIA GeForce RTX 4080
SUPER with at least 16,000 MiB, compute capability 8.9, and the pinned CUDA
13.1 compiler packages. This is an intentional optimization boundary, not a
recommended tier among several. Results from another GPU, a CPU path, or a
different model revision are not evidence for this build.

## Reader features retained

The progressive architecture preserves:

- selectable Chinese with displayed pinyin;
- position-anchored hover explanations with local longest-match dictionary
  definitions and HSK overlay;
- region context showing direct/displayed Chinese and source English;
- original/Chinese/hold-to-compare controls; and
- local-only Mandarin pronunciation using an eligible Firefox/OS voice.

These browser tools do not delay offscreen inference. Region order is stable
page order followed by within-page reading order, while current-viewport work
may overtake queued offscreen work at detector, OCR, and translation batch
boundaries.

See [the browser contract](browser-contract.md) for exact routes and event
shapes and [the Chapter 5 evidence plan](chapter-5-benchmark.md) for the
complete 36-image gold corpus and the passing packaged release measurements.
