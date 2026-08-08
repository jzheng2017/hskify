# Browser companion implementation

Status: direct chapter-aware performance path. This note describes the current
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
hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-28-r7
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
DELETE /chapters/{page_session_id}
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

## Job storage and chapter log

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
published once. `regionReady` rejects pending validation state, so a browser
receives exactly one terminal translation per region. `complete`, `failed`,
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
- multipart, declared, and sniffed MIME agreement;
- declared SHA-256;
- non-zero declared and decoded dimensions;
- 20 MiB image, 64 KiB metadata, and 21 MiB complete-body limits;
- 25,000,000 pixels and 16,384 pixels on either dimension; and
- a 128 MiB decoder allocation budget.

The direct adapter decodes the retained source once more for inference and
rechecks that its dimensions match job metadata.

## Learning policy

Every job selects one explicit learning mode, and that mode is part of the
translation-cache identity:

- `natural` publishes the text translator's faithful Chinese reference
  directly. Deterministic vocabulary validation adds teaching metadata but
  never launches a second rewrite or repair generation.
- `strict` requires every non-exempt lexical token to pass the selected HSK
  vocabulary level before the result is accepted.

Names are always rendered naturally in Chinese and unapproved Latin output is
rejected. There is no browser name mode or protected-Latin exception. Every
above-level occurrence is emitted as a bounded `teachingTerm` with exact
Unicode character offsets, pinyin, local
dictionary definitions, its known required HSK level, and whether it is above
the selected level or outside the HSK list. The extension therefore teaches
the actual final wording; it does not rely on a separate annotation model or a
hard-coded list of story phrases.

Vocabulary validation treats explicit higher-level HSK headwords as atomic
violations. A dictionary phrase that is fully decomposable into HSK headwords
at the selected level is counted by those allowed surface words
instead of becoming a shadow HSK violation merely because the dictionary also
stores the phrase. Semantic composition and meaning preservation remain model
responsibilities; the deterministic validator controls the selected
vocabulary inventory.

## Resident CUDA path

The resident model pack includes the Qwen3.5-4B `mmproj-BF16.gguf` projector.
The daemon requires this matching projector and the translation model as one
capability; it never silently falls back to a text-only semantic path. The
projector is attached only after the resident text model is loaded and receives
one immutable page surface plus its ordered OCR/layout evidence. It returns
only visual role and optional continuation metadata; OCR remains the transcript
authority and a separate text-only generation owns faithful Chinese. Mandatory
role/translation failures preserve only the affected region, while invalid
optional continuation metadata is discarded independently.

The browser-companion crate enables its `cuda` feature by default. The
performance build script accepts only an NVIDIA GeForce RTX 4080 SUPER with at
least 16,000 MiB and compute capability 8.9, installs pinned CUDA 13.1 compiler
components, and sets `CUDA_COMPUTE_CAP=89`.

The adapter lazily initializes two `OnceCell` values:

- the pinned CUDA RT-DETR-v2 comic/bubble detector, batched PP-OCRv6-small
  recognizer, learned text and bubble segmenters, manga LaMa inpainter,
  resident `RuntimeManager`, and local Qwen3.5 4B application state;
  and
- complete `hsk-control` data.

The detector/translator runtime uses `ComputePolicy::CudaRequired`. The English
recognizer uses `ort = 2.0.0-rc.12` with a mandatory CUDA execution provider,
fatal provider-registration errors, and environment providers disabled. ONNX
Runtime may place unsupported shape/control nodes on its built-in CPU provider;
disabling that normal fallback makes the selected PP-OCRv6-small graph impossible to
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
  models\resident\comic-text-bubble-detector-config\config.json
  models\resident\comic-text-bubble-detector-preprocessor-config\preprocessor_config.json
  models\resident\comic-text-bubble-detector-weights\model.safetensors
  models\resident\lama-manga-inpainter-weights\lama-manga.safetensors
  models\resident\manga-text-segmentation-weights\model.safetensors
  models\resident\pp-ocr-v6-small-detector-config\inference.yml
  models\resident\pp-ocr-v6-small-detector-model\inference.onnx
  models\resident\pp-ocr-v6-small-recognizer-config\inference.yml
  models\resident\pp-ocr-v6-small-recognizer-model\inference.onnx
  models\resident\speech-bubble-segmentation-config\config.json
  models\resident\speech-bubble-segmentation-weights\model.safetensors
  fonts\NotoSansSC-VF.ttf
  fonts\NotoSerifSC-VF.ttf
```

The comic topology detector is frozen to
`ogkalu/comic-text-and-bubble-detector@16e8a622f91fabc6b5b65c96d32d1183f8843546`
; independent text-line detection is frozen to
`PaddlePaddle/PP-OCRv6_small_det_onnx@28fe5895c24fd108c19eb3e8479f4ab385fbfc62`;
the recognizer is frozen to
`PaddlePaddle/PP-OCRv6_small_rec_onnx@b8f84f0b80c529de40b4fbb3544b84fa7233a513`;
setup verifies their exact byte counts and SHA-256 identities before the
resident session is created.

When the daemon sees that the managed resources are ready, it starts one
background warm-up task for the resident detector, OCR, segmenters, inpainter,
HSK data, and translation model. The same `OnceCell` initialization futures are
shared with incoming jobs, so a quick user action safely joins the in-flight
load while the normal case uses the otherwise idle time before the Translate
click. Failed warm-up is retryable, and resource paths are rediscovered after a
first-time model installation instead of being frozen as missing for the
daemon's lifetime.

The explicit overrides are `HSK_MANGA_RESOURCES_DIR`,
`HSK_MANGA_HSK_PATH`, and `HSK_MANGA_DICTIONARY_PATH`.

## Viewport-first region pipeline

1. Split the decoded page into 2,048-pixel tiles with 410-pixel overlap. The
   comic detector uses its trained 640-pixel canvas; the PP-OCR detector keeps
   aspect ratio and scans tall inputs through overlapping near-native windows
   instead of squeezing a long page into one square.
2. Before each detector batch, reprioritize remaining tiles against the current
   `visibleRects` and active state. When the visible frontier is smaller than
   six tiles, submit only that frontier first instead of filling the batch with
   off-screen tiles.
3. Run the comic and PP-OCR text detectors in bounded CUDA batches. PP-OCR
   polygons are preferred line geometry; the comic detector contributes
   independent recovery proposals where PP-OCR has no overlapping line.
   Convert both streams to source coordinates, enforce overlap ownership, and
   spatially deduplicate them.
4. Run PP-OCRv6-small recognition in batches of at most eight. Each detector
   line is read from the original crop and a contrast-normalized grayscale
   crop. Consensus is decided per line before region concatenation, so one
   disagreeing background glyph is discarded without erasing neighboring
   dialogue that agrees in both views.
5. Accept mechanically valid Latin OCR at the calibrated 0.55 confidence
   floor. A detector-backed bubble becomes finalizable only after every
   unprocessed tile-ownership cell that could contain another line in that
   bubble is gone. Visible finalized groups enter semantic analysis and
   publication immediately; non-visible groups share one bounded page-tail
   pass. Rejected proposals become terminal source-preserving unreadable
   regions instead of disappearing.
6. Run multimodal page adjudication over the immutable source surface (or a
   geometry-derived evidence viewport) and numbered OCR polygons. The model
   classifies story, SFX, embedded furniture, and decorative artwork and may
   link a story continuation. It cannot emit or alter the OCR transcript or
   translate text. Invalid optional continuation metadata is dropped
   independently.
7. Only admitted story and SFX regions enter cleanup and translation. Segment
   source glyphs and start one page-level LaMa cleanup task. In parallel, one
   text-only generation translates at most six role-filtered OCR regions. Its
   role-position index is separate from the numbered source lines so metadata
   cannot be copied into Chinese. Context is refreshed when language work is
   dispatched, after time spent in OCR/vision, without waiting for unfinished
   pages.
8. Verify cleanup with direct safety invariants: some masked pixels changed,
   residual edge energy stays bounded, the patch-to-source seam stays below
   its error limit, and every protected pixel is byte-identical. A clipped
   half-glyph therefore fails through its seam; unrelated mask density and
   unstable smooth-area residual ratios are not used as proxies.
9. Natural mode validates and publishes the faithful reference directly once
   cleanup succeeds. Strict mode sends ordered HSK rewrite batches: the first
   visible region dispatches alone, throughput batches start at three and
   contain at most six, and the remaining tail flushes when the page frontier
   closes. Boundary checks never sleep.
10. Vision and language workloads have separate serialized CUDA lanes, so two
     page jobs can pipeline semantic vision for the next page alongside HSK
     realization for the current page. Queue registration is scoped to the
     acquiring future: aborting cleanup removes its waiter immediately instead
     of leaving an ownerless lane head.
11. Firefox keeps at most two image jobs in flight within a two-image decoded
    pixel budget. Pending work is reprioritized as the viewport moves, but an
    admitted page is never cancelled and restarted merely because it scrolled
    off screen. Automatic image retry is absent; retry is an explicit user
    action after a terminal failure.

Low-confidence, undecodable, and non-Latin OCR is never translated or painted.
Detector proposals that cannot reach terminal consensus remain visible source
pixels with an `UnreadableRegion` hover target. Content is not rejected by
hard-coded story, credit, role, or sound-effect word lists. Eligible narration
and other story text outside a balloon remain in scope.

## Model-backed cleanup

Cleanup does not paint inferred background colors. For each source image, the
adapter:

- predicts source-text pixels with the pinned manga text segmenter;
- predicts real speech-bubble contours and assigns lines by contour identity;
- constrains the semantic text mask to accepted OCR geometry and expands glyphs
  with the shared model pipeline's measured text-region rules;
- runs the manga-trained LaMa inpainter over the union mask; and
- emits transparent per-region PNGs whose alpha follows only the expanded
  semantic mask.

The stitched learned text-probability field is produced once per detector tile
and reused by OCR line discovery, per-line palette extraction, punctuation
support, cleanup masking, and inpainting. Bubble grouping keeps those ordered
appearance bands instead of replacing a mixed-color bubble with the style of
its longest line. Layout uses the complete connected bubble component rather
than the OCR or detector rectangle. Its safe text polygon is a shape-preserving
erosion whose clearance is bounded by both measured glyph height and the
bubble's own smaller dimension; oversized source lettering therefore cannot
collapse a large balloon into a tiny replacement-text area.

An empty or unsafe semantic mask preserves that region's source pixels and
publishes an unreadable notice without retrying the page. It never silently
leaves half a balloon translated and never substitutes a painted text-sized
rectangle.

The server validates the PNG and its normalized rectangle, enforces a 16 MiB
per-patch limit and 256 MiB total retained source/patch budget, stores it under
the owning job, and returns a blob descriptor.

`publish_region` calls `store_generated_patch_png` before appending `regionReady`.
Firefox then fetches and validates the patch, decodes it, inserts it into the
patch layer, and inserts selectable text synchronously afterward. This ordering
prevents Chinese text from appearing over uncleaned English.

## Faithful natural and strict HSK translation

After visual role filtering, the resident text model supplies one complete,
unconstrained faithful Chinese reference for each story or SFX region. Roles
are carried as a separate position index, not labels beside source or output
text. Outputs containing role labels, Latin words, punctuation-only text,
source echoes, or malformed/missing positions are rejected per region. Natural
mode uses the faithful reference as final text and adds HSK coverage, pinyin,
and teaching terms deterministically. Strict mode sends up to six ordered
references plus immutable English OCR structure to Qwen3.5 4B with the
requested cumulative HSK 2.0 level and at most six accepted chapter-context
utterances.

The requested level controls syntax as well as vocabulary. Levels 1-2 prefer
short direct clauses, explicit referents, everyday wording, and no avoidable
idioms, formal nominalization, nested clauses, or passive constructions.
Levels 3-4 permit familiar compound sentences while simplifying dense
embedding and formal synonyms; levels 5-6 permit natural advanced grammar.
Names are always rendered naturally in Chinese, using established Chinese
forms or consistent phonetic transliteration. No Latin name survives as an HSK
exception, and page understanding does not own a glossary or entity-memory
contract.

The page pass receives compact normalized geometry, enclosure topology, and
every OCR item in its bounded evidence window. Regions are merged in canonical
reading order, so an arbitrary tile boundary cannot split a connected phrase.
It explicitly distinguishes reader-facing dialogue/narration from text painted
on signs, books, clothing, interfaces, credits, and title artwork. Furniture
and artwork emit a silent terminal preservation event and never enter
segmentation, inpainting, or patch encoding. A surviving region is thereafter
authoritatively story content: HSK generation cannot independently reclassify
or skip it. Standalone numbers remain exact-preservation requirements; digits
embedded in Latin OCR tokens do not.

`hsk-control` validates every displayed story item. In natural mode, its
vocabulary findings are emitted immediately as teaching terms and the faithful
reference is terminal. In strict mode, above-level vocabulary and deterministic
meaning/preservation failures such as missing output, source echo, number
loss, question-intent loss, and excessive expansion enter one
logical batched repair. A strict item can receive at most one repair attempt;
repair output is never recursively fed into another strategy.
Up to six rejected regions enter one logical numbered repair request. Before
generation, the translator measures each candidate subbatch with the resident
model's real tokenizer and chat template, chooses the largest ordered prefix
whose prompt plus desired output fits the actual context window, and merges
the results by application ID. Parsing, validation, and avoid-lists remain
isolated, so one malformed sibling cannot authorize
or invalidate another and an oversized logical batch never burns retries on a
known context overflow. The deterministic validator supplies a typed avoid-list
for strict vocabulary repairs. Natural mode never enters this rewrite path.
The repair never restarts the page. If one OCR region remains unsafe to
publish, its original pixels remain untouched and the other regions still
complete; deterministic per-region validation exhaustion is not promoted into
a retry of the whole image.

Pending meaning-valid primaries can provide internal discourse context to
later ordered batches, but they never cross the browser contract. Pinyin is
derived after the accepted/rejected final state by local
longest-match lookup. A terminal `TranslatedRegion` carries:

- source English;
- the faithful reference as `baseChinese`;
- the normalized natural or post-repair strict Chinese;
- pinyin;
- OCR confidence and reading order;
- normalized text/bubble/patch geometry;
- browser-safe style and layout; and
- requested level, learning mode, level coverage, exact teaching terms, strict
  validity, above-level tokens, and repair state.

Source color bands remain vertical appearance samples. They are mapped onto
the fitted output lines after layout and never force the Chinese translation
to retain the source line count. Polygon geometry determines the largest
non-overflowing layout; a 12 CSS-pixel/one-percent-of-image accessibility floor,
not an arbitrary fraction of oversized source lettering, decides whether the
result is readable.

## Chapter translation cache

The daemon holds a 64 MiB byte-bounded in-memory strict-HSK translation cache. Its
SHA-256 key covers:

```text
schema
OCR text
faithful Chinese reference and utterance kind
canonical preceding and following chapter context
HSK level
learning mode
model ID
model revision
prompt hash
validator hash
HSK/dictionary control revision
```

The key prevents reuse when chapter context, level, model
bytes/revision, prompt behavior, validation logic, or language data changes.
The cache is not a
project, browser history, persistent page artifact, or retranslation facility.

The separate 2 GiB persistent result cache stores only complete terminal
chapter-region results and their patch PNGs. Its key covers the strict request,
build fingerprint, source identity, all output-affecting resource identities,
the semantic/OCR/cleanup pipeline revision, and the HSK normalization,
segmentation, lookup, Jieba, and Unicode-table policy revisions. A pipeline or
validator change therefore cannot replay regions assessed by the previous
policy.
Each entry is installed with one atomic rename after visible processing
finishes. Size accounting and eviction occur on that store path. A replay
computes the exact key and opens only that entry; it does not scan all cached
chapter images before every hit. The upload's bytes, SHA-256, format, MIME,
limits, and header dimensions are still checked before lookup. Because a hit
identifies content that was fully decoded when the entry was created, only a
miss performs full pixel decoding. No tile, detector, OCR, translation, or
patch intermediate is persisted.

## Retained reader tools

`POST /lookup` uses the same local `hsk-control` instance for longest-match
tokens, pinyin, definitions, HSK level overlay, proper-name state, and optional
region context. Selection lookup tokenizes the selected text. Hover lookup
accepts only an owning region and Unicode character offset, then longest-matches
from that exact position in the daemon's canonical displayed Chinese. It
returns one expression and never advances across punctuation. The extension
owns the original/Chinese comparison control.
Mandarin speech is also extension-only and uses an eligible local Web Speech
voice; neither comparison nor speech adds a daemon result endpoint.

## Default bounds

| Limit | Value |
| --- | ---: |
| Authenticated in-flight requests | 64 |
| Retained jobs | 128 |
| Retained source and patch bytes | 256 MiB |
| Decoded-image LRU | 512 MiB |
| In-memory translation cache | 64 MiB |
| Persistent completed-result cache | 2 GiB |
| One patch | 16 MiB |
| One font | 32 MiB |
| Visible rectangles | 64 |
| Chapter context entries used by direct translation/cache | 6 |
| Update long-poll | 20 s |
| Idle lifetime | 30 min |

The authenticated request permit is retained through response-body transfer,
so stalled blob/font consumers remain counted. Idle shutdown latches only when
there are no admitted requests and no active jobs.

## Evidence status

Architecture claims above are traced to the current code and contract
fixtures. Release evidence belongs to the complete packaged-Firefox
real-reader-v2 corpus described in
[the real-reader v2 method](../../docs/real-reader-v2.md). The tracked
manifest is currently capture-required, so no source-only diagnostic or
retired benchmark can be reported as a release pass. Keep raw outputs and
never add chapter-specific tuning.
