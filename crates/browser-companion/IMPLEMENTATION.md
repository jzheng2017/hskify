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

## Learning policy

Every job selects one explicit learning mode, and that mode is part of the
translation-cache identity:

- `natural` simplifies vocabulary and grammar first, then permits a small,
  level-dependent number of indispensable story terms. Levels 1-3 target at
  least 90% level-appropriate lexical tokens, level 4 targets 93%, and levels
  5-6 target 95%. Short dialogue may keep one useful term where a longer
  paraphrase would be less readable.
- `strict` requires every non-exempt lexical token to pass the selected HSK
  vocabulary level before the result is accepted.

Protected proper names remain separate from this policy and follow the
reader's Names setting. Every accepted above-level occurrence is emitted as a
bounded `teachingTerm` with exact Unicode character offsets, pinyin, local
dictionary definitions, its known required HSK level, and whether it is above
the selected level or outside the HSK list. The extension therefore teaches
the actual final wording; it does not rely on a second model call or a
hard-coded list of story phrases.

Vocabulary validation treats explicit higher-level HSK headwords as atomic
violations. A dictionary phrase that is fully decomposable into HSK headwords
at the selected level is counted by those allowed surface words
instead of becoming a shadow HSK violation merely because the dictionary also
stores the phrase. Semantic composition and meaning preservation remain model
responsibilities; the deterministic validator controls the selected
vocabulary inventory.

## Resident CUDA path

The browser-companion crate enables its `cuda` feature by default. The
performance build script accepts only an NVIDIA GeForce RTX 4080 SUPER with at
least 16,000 MiB and compute capability 8.9, installs pinned CUDA 13.1 compiler
components, and sets `CUDA_COMPUTE_CAP=89`.

The adapter lazily initializes two `OnceCell` values:

- the pinned CUDA RT-DETR-v2 comic text detector, batched English PP-OCRv5
  recognizer, learned text and bubble segmenters, manga LaMa inpainter,
  resident `RuntimeManager`, and local Qwen3.5 4B application state;
  and
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
  models\resident\comic-text-bubble-detector-config\config.json
  models\resident\comic-text-bubble-detector-preprocessor-config\preprocessor_config.json
  models\resident\comic-text-bubble-detector-weights\model.safetensors
  models\resident\lama-manga-inpainter-weights\lama-manga.safetensors
  models\resident\manga-text-segmentation-weights\model.safetensors
  models\resident\pp-ocr-v5-english-recognizer-config\inference.yml
  models\resident\pp-ocr-v5-english-recognizer-model\inference.onnx
  models\resident\speech-bubble-segmentation-config\config.json
  models\resident\speech-bubble-segmentation-weights\model.safetensors
  fonts\NotoSansSC-VF.ttf
  fonts\NotoSerifSC-VF.ttf
```

The detector is frozen to
`ogkalu/comic-text-and-bubble-detector@16e8a622f91fabc6b5b65c96d32d1183f8843546`
and the recognizer to
`PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx@3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b`;
setup verifies their exact byte counts and SHA-256 identities before the
resident session is created.

The explicit overrides are `HSK_MANGA_RESOURCES_DIR`,
`HSK_MANGA_HSK_PATH`, `HSK_MANGA_DICTIONARY_PATH`, and
`HSK_MANGA_QWEN_MODEL_PATH`.

## Viewport-first region pipeline

1. Split the decoded page into 2,048-pixel tiles with 410-pixel overlap and
   resize detector input to the model's trained 640-pixel dimensions.
2. Before each detector batch, reprioritize remaining tiles against the current
   `visibleRects` and active state.
3. Run RT-DETR-v2 in true CUDA batches of at most six tiles and retain its
   `text_bubble` and `text_free` classes. Do not require a detected balloon.
4. Convert tile-local text geometry to source coordinates, enforce tile
   ownership, and spatially deduplicate overlapping candidates.
5. Run PP-OCRv5 English line recognition in batches of at most eight, and yield
   after each candidate chunk so newly visible translation work can overtake
   off-screen OCR.
6. Accept mechanically valid Latin OCR at confidence 0.45 or higher. Segment
   real bubble contours and group every accepted line by bubble identity, so a
   complete balloon is translated atomically even when detector boxes differ.
7. Segment source glyph pixels with the learned manga text model, constrain the
   mask to accepted text regions, expand it by measured source geometry, and
   restore the masked artwork with the manga-trained LaMa model.
8. Queue accepted regions for translation and reprioritize visible work at
   every OCR or detector boundary. Ready batches begin at three pending
   regions, contain at most six, and an undersized tail becomes eligible when
   its hard 75 ms batching deadline expires (or at the final forced drain).
   Boundary checks never sleep, and no page-wide translation call exists.

Low-confidence, undecodable, and non-Latin OCR are excluded before translation.
Content is not rejected by hard-coded story, credit, role, or sound-effect word
lists. Eligible narration and other story text outside a balloon remain in
scope.

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
its longest line.

An empty semantic mask fails the image and enters the extension's bounded
automatic retry state machine. It never silently leaves half a balloon
translated and never substitutes a painted text-sized rectangle.

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

The requested level controls syntax as well as vocabulary. Levels 1-2 prefer
short direct clauses, explicit referents, everyday wording, and no avoidable
idioms, formal nominalization, nested clauses, or passive constructions.
Levels 3-4 permit familiar compound sentences while simplifying dense
embedding and formal synonyms; levels 5-6 permit natural advanced grammar.
The job's name preference is part of generation, validation, and both cache
keys. `keep-original` preserves the source's exact Latin name spelling and
permits those matched source names as HSK exceptions. `chinese` uses approved
glossary forms first, then established Chinese names when certain and
otherwise consistent phonetic transliteration. Neither mode translates a
name by its dictionary meaning.

The same contextual Qwen decision that translates a region may return the
typed `[NON-STORY]` disposition for an entire publisher/site credit, watermark,
advertisement, or navigation label. The protocol explicitly forbids that
disposition for dialogue, narration, thoughts, captions, in-story signs or
letters, titles, names, roles, fragments, and stylized emphasis. Excluded
regions publish neither cleanup pixels nor replacement text, so the source
image remains untouched. Standalone numbers remain exact-preservation
requirements; digits embedded in Latin OCR tokens do not.

`hsk-control` validates each returned story item. Items that already pass are
accepted. Rejected items alone receive at most four prompt-changing targeted
repair attempts with their rejected Chinese and exact deterministic problems.
Every distinct bounded strategy runs unless an earlier attempt succeeds; an
unchanged answer cannot prevent the later source-first rewrite strategies.
The deterministic validator also supplies a typed avoid-list that is refreshed
from each rejected candidate. Strict repair must emit none of those exact terms.
Natural repair remains governed by Natural learning on every attempt: it must
simplify the avoid-list while retaining at most the level-specific budget of
indispensable story terms. It never silently escalates to Strict and discards a
core story concept merely to improve the vocabulary score.
The repair never restarts the page. If one OCR region remains unsafe to
publish, its original pixels remain untouched and the other regions still
complete; deterministic per-region validation exhaustion is not promoted into
a retry of the whole image.

Pinyin is derived after the accepted/rejected final state by local
longest-match lookup. A progressive region carries:

- source English;
- the direct generation as `baseChinese`;
- the displayed post-validation/repair Chinese;
- pinyin;
- OCR confidence and reading order;
- normalized text/bubble/patch geometry;
- browser-safe style and layout; and
- requested level, learning mode, level coverage, exact teaching terms, strict
  validity, above-level tokens, and repair state.

## Translation cache

The daemon holds a 64 MiB byte-bounded in-memory direct-translation cache. Its
SHA-256 key covers:

```text
schema
OCR text
last six preceding utterances
HSK level
learning mode
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
fingerprint, source identity, all output-affecting resource identities, and
the HSK normalization, segmentation, lookup, Jieba, and Unicode-table policy
revisions. A validator-code change therefore cannot replay regions assessed by
the previous policy.
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
| Preceding context entries accepted by contract | 6 |
| Preceding entries used by direct translation/cache | 6 |
| Update long-poll | 20 s |
| Idle lifetime | 30 min |

The authenticated request permit is retained through response-body transfer,
so stalled blob/font consumers remain counted. Idle shutdown latches only when
there are no admitted requests and no active jobs.

## Evidence status

Architecture claims above are traced to the current code and contract
fixtures. The sole canonical workload is the 36-image *30 Years Since the
Prologue* chapter 5 fixture. Its 218-region geometry review and 214-target
translation, pinyin, and token-level HSK gold are complete. Release claims
require the complete packaged-Firefox run sequence in
[the benchmark evidence method](../../docs/chapter-5-benchmark.md); source-only
diagnostics are labeled separately. The current exact package passed all 426
gates across one installed-cold run and 20 measured warm runs; the benchmark
document records the timings, quality, memory, VRAM, cache, cancellation, and
disk evidence. Keep raw outputs, do not add chapter-specific tuning, and never
reuse measurements from a different workload or the retired page-result
pipeline.
