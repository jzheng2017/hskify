# Browser companion implementation

Status: direct progressive performance path. This note describes the current
Rust implementation, not the retired project-backed page pipeline.

## Executables and lifetime

The crate builds:

- `hsk-manga-native-host`, a one-shot Firefox native-messaging process that
  validates the manifest path, permanent add-on ID, manifest executable, and
  one bounded little-endian JSON frame; and
- `hsk-manga-browser-daemon`, a detached per-user process that owns the
  loopback HTTP service and resident CUDA models.

The daemon takes an exclusive per-user lock, binds literally to
`127.0.0.1:0`, writes a control-secret-protected discovery record, and uses a
30-minute idle window. Its Tokio runtime has four workers and at most eight
blocking threads. Windows detached creation requests
`CREATE_BREAKAWAY_FROM_JOB`, `CREATE_NEW_PROCESS_GROUP`, and
`CREATE_NO_WINDOW`; Unix uses `setsid()`.

```text
browser-companion/
  daemon.lock
  daemon-state.json
  browser-cache/
    browser-runtime/
    results/
```

The cache contains runtime/model state. There are no hidden translation
projects, page-history records, cleaned-page blobs, or versioned pipeline
markers.

## Exact build contract

Rust and TypeScript compile the same fingerprint:

```text
hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-26-r2
```

The native handshake request/response, health response, job metadata, and job
creation response validate or echo that exact value. Mismatch is rejected.
There is no protocol-number header, compatibility range, downgrade, migration
adapter, or legacy response parser.

The permanent identities are:

```text
native host: local.hskify.hsk_manga
Firefox ID:  hsk-manga-translator@local.hskify
```

## Loopback routes

The browser surface is unversioned:

```text
GET    /health
GET    /setup
POST   /setup/models
POST   /jobs
DELETE /jobs/{job_id}
PUT    /jobs/{job_id}/viewport
GET    /jobs/{job_id}/updates
POST   /lookup
GET    /blobs/{patch_id}
GET    /fonts/{font_id}
```

`POST /browser-internal/session` is control-secret protected and called only
by the native launcher. It is not CORS-enabled or part of the browser contract.
No general application API, MCP, UI, page-result, cleaned-image, or
retranslation route is mounted.

All browser routes require the exact loopback `Host`, an active canonical
extension origin, and its bearer token before handler/body extraction. The
explicit `X-HSK-Manga-Extension-Origin` header covers privileged Firefox
requests that omit `Origin`. Preflight permits only GET, POST, PUT, and DELETE
plus Authorization, Content-Type, and that extension-origin header.

## Job storage and progressive log

An accepted upload reserves:

- the immutable source bytes until its active job finishes;
- an atomic cancellation flag;
- the current viewport revision;
- a bounded append-only `Vec<JobUpdate>`;
- the region IDs already published;
- adapter-native region context for dictionary lookup; and
- job-owned PNG blobs.

Every update is assigned the next sequence while holding the job-log lock.
Sequence 0 is invalid. Overall progress may not regress. A region ID can be
published once; refinement requires an existing region. `complete`, `failed`,
and `cancelled` are terminal, and publication after terminal state is rejected.
The maximum is 10,000 updates per job.

`GET /jobs/{job_id}/updates` replays entries strictly after `after`. It rejects
a cursor beyond the latest sequence, long-polls for no more than 20 seconds,
and returns an empty batch without advancing the cursor on timeout. This single
log replaces separate progress, status, and result representations.

Terminal inactive jobs remain available until job-count or retained-byte
admission needs space. Eviction is deterministic oldest-first. Active jobs are
not evictable, and evicting a job removes its owned patch blobs.

## Upload and decode boundary

`POST /jobs` accepts exactly `image` and `request` multipart fields. The request
field must be JSON. Before source retention, the server validates:

- exact build fingerprint and semantic request contract;
- supported English-to-Simplified-Chinese/HSK 2.0 settings;
- sound-effect translation disabled;
- multipart, declared, and sniffed MIME agreement;
- declared SHA-256;
- non-zero declared and decoded dimensions;
- 20 MiB image, 64 KiB metadata, and 21 MiB complete-body limits;
- 25,000,000 pixels and 16,384 pixels on either dimension; and
- a 128 MiB decoder allocation budget.

The direct adapter decodes the retained source once more for inference and
rechecks that its dimensions match job metadata.

## Resident CUDA path

The browser-companion crate enables its `cuda` feature by default. The
performance build script accepts only an NVIDIA GeForce RTX 4080 SUPER with at
least 16,000 MiB and compute capability 8.9, installs pinned CUDA 13.1 compiler
components, and sets `CUDA_COMPUTE_CAP=89`.

The adapter lazily initializes two `OnceCell` values:

- the official PP-OCRv5 CUDA text detector, selected batched English
  recognizer, resident `RuntimeManager`, and local Qwen3.5 4B application
  state; and
- complete `hsk-control` data.

The detector/translator runtime uses `ComputePolicy::CudaRequired`. The English
recognizer uses `ort = 2.0.0-rc.12` with a mandatory CUDA execution provider,
fatal provider-registration errors, and environment providers disabled. ONNX
Runtime may place unsupported shape/control nodes on its built-in CPU provider;
disabling that normal fallback makes the selected PP-OCRv5 graph impossible to
load. Its warmed session, zero-copy input buffer, and caller-owned dynamic output
allocations are reused across jobs.
Output allocations use an LRU capped at four shapes and 32 MiB of host memory;
least-recent shapes are evicted and no GPU output memory remains cached.

Default resource discovery is:

```text
%LOCALAPPDATA%\Hskify\resources\
  hsk-2.0.normalized.json
  cc-cedict.normalized.json
  models\Qwen3.5-4B-Q4_K_M.gguf
  models\resident\pp-ocr-v5-mobile-detector-model\inference.onnx
  models\resident\pp-ocr-v5-english-recognizer-config\inference.yml
  models\resident\pp-ocr-v5-english-recognizer-model\inference.onnx
  fonts\NotoSansSC-VF.ttf
  fonts\NotoSerifSC-VF.ttf
```

The detector is frozen to
`PaddlePaddle/PP-OCRv5_mobile_det_onnx@e6f4fa85f00e168c862bc462aebca69eef9b3d3d`
and the recognizer to
`PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx@3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b`;
setup verifies their exact byte counts and SHA-256 identities before the
resident session is created.

The explicit overrides are `HSK_MANGA_RESOURCES_DIR`,
`HSK_MANGA_HSK_PATH`, `HSK_MANGA_DICTIONARY_PATH`, and
`HSK_MANGA_QWEN_MODEL_PATH`.

## Viewport-first region pipeline

1. Split the decoded page into 2,048-pixel tiles with 410-pixel overlap and
   letterbox detector input to at most 1,280 pixels.
2. Before each detector batch, reprioritize remaining tiles against the current
   `visibleRects` and active state.
3. Run the official PP-OCRv5 mobile text detector in true CUDA batches of at
   most six tiles.
4. Convert tile-local line geometry to source coordinates, enforce tile ownership,
   and spatially deduplicate overlapping candidates.
5. Run PP-OCRv5 English line recognition in batches of at most eight, and yield
   after each candidate chunk so newly visible translation work can overtake
   off-screen OCR.
6. Accept OCR lines at confidence 0.45 or higher, group adjacent lines by
   geometry and inferred foreground color, then apply the Latin-English,
   story-text, SFX, credit, branding, metadata, and ambiguity gates to the
   complete group. Keep differently colored emphasis and separated balloons
   independent. When a tile has explicit credit/release context, reject its
   isolated uppercase credit-name labels without applying that rule to normal
   story tiles.
7. Construct one region-local cleanup patch from the grouped line masks and compute deterministic
   reading order plus visibility.
8. Queue accepted regions for translation and reprioritize visible work at
   every OCR or detector boundary. Ready batches begin at three pending
   regions, contain at most six, and an undersized tail becomes eligible when
   its hard 75 ms batching deadline expires (or at the final forced drain).
   Boundary checks never sleep, and no page-wide translation call exists.

Sound effects, punctuation-only OCR, non-Latin text, credits, branding,
promotion, metadata, and low-confidence OCR are excluded before translation
and patch publication. Eligible narration and other story text outside a
balloon remain in scope.

## Region-local cleanup

Cleanup does not reconstruct or encode the source page. For one accepted story
region, the adapter:

- expands a small bounded patch rectangle;
- combines the recognizer's polarity-independent line masks, attaches
  punctuation and antialiased edges, dilates the ink locally, and keeps the
  alpha region bounded to the accepted text geometry;
- propagates surrounding known colors into masked pixels, with a median-color
  fast path for flat regions and local multiscale diffusion for gradients; and
- encodes a PNG whose alpha is 255 only at masked pixels and 0 elsewhere.

The server validates the PNG and its normalized rectangle, enforces a 16 MiB
per-patch limit and 256 MiB total retained source/patch budget, stores it under
the owning job, and returns a blob descriptor.

`publish_region` calls `store_patch_png` before appending `regionReady`.
Firefox then fetches and validates the patch, decodes it, inserts it into the
patch layer, and inserts selectable text synchronously afterward. This ordering
prevents Chinese text from appearing over uncleaned English.

## Direct HSK translation

The primary generation request sends up to six English dialogue utterances
directly to Qwen3.5 4B with the requested cumulative HSK 2.0 level and at most
six preceding dialogue utterances. It does not generate a page-wide faithful
translation and then rewrite it.

The primary numbered-line protocol permits `[SKIP]` for credits, branding,
release or scanner notes, SFX, non-English OCR, and gibberish. The parser
accepts that marker only when deterministic source evidence independently
supports exclusion. A marker on ordinary prose, including prose containing
OCR substitutions such as `WH4`, is rejected and the item is repaired.
Standalone numbers remain exact-preservation requirements; digits embedded in
Latin OCR tokens do not.

`hsk-control` validates each returned story item. Items that already pass are
accepted. Rejected items alone may be sent once to the targeted repair call
with their rejected Chinese and exact deterministic problems. The repair is
bounded to one batch; it is not a second general translation pass and never
restarts the page.

Pinyin is derived after the accepted/rejected final state by local
longest-match lookup. A progressive region carries:

- source English;
- the direct generation as `baseChinese`;
- the displayed post-validation/repair Chinese;
- pinyin;
- OCR confidence and reading order;
- normalized text/bubble/patch geometry;
- browser-safe style and layout; and
- requested level, strict validity, above-level tokens, and repair state.

## Translation cache

The daemon holds an in-memory direct-translation cache. Its SHA-256 key covers:

```text
schema
OCR text
last six preceding utterances
HSK level
model ID
model revision
prompt hash
validator hash
HSK/dictionary control revision
proper-name glossary
```

The key prevents reuse when dialogue context, level, model bytes/revision,
prompt behavior, validation logic, or language data changes. The cache is not a
project, browser history, persistent page artifact, or retranslation facility.

The separate 2 GiB persistent result cache stores only complete per-image
regions and their patch PNGs. Its key covers the strict request, build
fingerprint, source identity, and all output-affecting resource identities.
Each entry is installed with one atomic rename after visible processing
finishes; no tile, detector, OCR, translation, or patch intermediate is
persisted.

## Retained reader tools

`POST /lookup` uses the same local `hsk-control` instance for longest-match
tokens, pinyin, definitions, HSK level overlay, proper-name state, and optional
region context. The extension owns the original/Chinese comparison control.
Mandarin speech is also extension-only and uses an eligible local Web Speech
voice; neither comparison nor speech adds a daemon result endpoint.

## Default bounds

| Limit | Value |
| --- | ---: |
| Authenticated in-flight requests | 64 |
| Retained jobs | 128 |
| Retained source and patch bytes | 256 MiB |
| One patch | 16 MiB |
| One font | 32 MiB |
| Visible rectangles | 64 |
| Preceding context entries accepted by contract | 6 |
| Preceding entries used by direct translation/cache | 6 |
| Update long-poll | 20 s |
| Idle lifetime | 30 min |

The authenticated request permit is retained through response-body transfer,
so stalled blob/font consumers remain counted. Idle shutdown latches only when
there are no admitted requests and no active jobs.

## Evidence status

Architecture claims above are traced to the current code and contract
fixtures. No end-to-end latency, memory, VRAM, throughput, quality, or
packaged-Firefox result has yet been recorded for this direct progressive
build. The sole canonical workload is the 36-image *30 Years Since the
Prologue* chapter 5 fixture. Its 218-region geometry review and 214-target
translation, pinyin, and token-level HSK gold are complete. Follow
[the benchmark evidence method](../../docs/chapter-5-benchmark.md), keep raw
outputs, and do not add chapter-specific tuning or reuse measurements from a
different workload or the retired page-result pipeline.
