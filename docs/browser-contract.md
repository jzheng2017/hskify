# Unversioned browser contract

The browser daemon mounts one exact API at its random IPv4 loopback origin.
There is no version prefix, content negotiation, compatibility route, result
download, or legacy parser.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Exact fingerprint, engine readiness, and sorted resident-resource identities |
| `GET` | `/setup` | Current installable-resource state |
| `POST` | `/setup/models` | Start or report resource setup |
| `POST` | `/warmup` | Initialize exactly the requested `document` or `image` runtime |
| `POST` | `/jobs/image` | Create an image job from multipart raster bytes and strict metadata |
| `POST` | `/jobs/document` | Create a document job from one JSON chapter snapshot |
| `PUT` | `/jobs/{jobId}/focus` | Replace tagged image-rectangle or visible-block focus |
| `GET` | `/jobs/{jobId}/updates` | Replay or long-poll updates after a sequence |
| `DELETE` | `/jobs/{jobId}` | Cancel and release one job |
| `DELETE` | `/chapters/{pageSessionId}` | Release ordered chapter context |
| `POST` | `/lookup` | Local pinyin/dictionary lookup owned by an `itemId` |
| `GET` | `/blobs/{blobId}` | Fetch an authorized job-owned image patch |
| `GET` | `/fonts/{fontId}` | Fetch one permitted installed font |

The native launcher alone calls `/browser-internal/session` with
`X-Hskify-Control`. That endpoint is not CORS-enabled and is not part of the
browser API.

## Authentication and identity

Every browser route requires the exact loopback `Host`, an active canonical
extension origin, and `Authorization: Bearer <session-token>`. Privileged
Firefox fetches that omit standard `Origin` send
`X-Hskify-Extension-Origin` with the same canonical origin.

The native handshake, health response, setup readiness, and both job requests
must agree on:

```text
hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-08-09-r8
```

The registered native host is `local.hskify.browser`; its sole allowed Firefox
extension is `hskify@local.hskify`. Unknown JSON fields, duplicate security
headers, modality mismatches, and a different fingerprint are rejected.

`GET /setup` reports only verified installable-resource state and never loads a
model. Once resources are ready, page detection issues `POST /warmup` with the
detected kind. A document request initializes only the language runtime; an
image request initializes language and vision. A detected page retains its kind
and retries through missing/installing/warming states, so finishing first-run
installation cannot strand that tab without a tagged warm-up.

## Shared job fields

Both creation requests identify the exact source, page session, HSK 2.0 level
1–6, and learning mode (`natural` or `strict`). Image metadata additionally
identifies canonical chapter source order. The only language pair is English to
Simplified Chinese. A successful creation returns HTTP 202 with the exact build
fingerprint and `jobId`.

Active-job and page-artifact records have one exact `source.kind` discriminator:
`image` or `document`. The record keeps `sourceSha256` beside that tagged
source, and its image variant retains only the image fields required for
recovery. Output from one kind cannot be replayed into the other.

## Image creation

`POST /jobs/image` accepts exactly two multipart fields:

- `image`: PNG, JPEG, WebP, or GIF bytes;
- `request`: `application/json` metadata containing exactly
  `buildFingerprint`, `clientImageId`, `sourceSha256`, `sourceMimeType`,
  `naturalWidth`, `naturalHeight`, `pageSessionId`, `sourceIndex`,
  `chapterSourceOrder`, `surfaceKind`, `readingDirection`, `settings`, and
  `visibleRects`.

The daemon verifies byte count, MIME, sniffed format, SHA-256, declared and
decoded dimensions, pixel count, and decoder allocation before starting.
Layout constraints occur only on `ocr` source spans.

## Document creation

`POST /jobs/document` accepts `application/json` with exactly
`buildFingerprint`, `pageSessionId`, `sourceSha256`, `settings`, and `blocks`.
Every block has exactly `itemId`, `sourceIndex`, `itemOrder`, `kind`,
`provenance`, `text`, and an optional `layout`. Document blocks require
`provenance: "dom"` and reject a present `layout`; DOM text is authoritative.

The daemon applies these limits before registration:

| Input | Limit |
| --- | ---: |
| Complete UTF-8 JSON body | 1 MiB |
| Text blocks | 2,000 |
| One normalized block | 16 KiB UTF-8 |
| Visible block IDs in one focus update | 64 |

It repeats deterministic normalization and recomputes the canonical document
hash over the complete ordered text snapshot. A mismatch or oversized chapter
is rejected; the daemon never truncates it. Browser extraction measures the
complete compact `/jobs/document` JSON envelope, not only its block text or
reader snapshot, before accepting the descriptor.

Every block is registered in canonical `(sourceIndex, itemOrder)` order before
translation starts. This makes context independent of viewport scheduling.

## Focus

`PUT /jobs/{jobId}/focus` accepts exactly one variant matching the job:

```json
{
  "kind": "image",
  "active": true,
  "visibleRects": [{ "x": 0, "y": 0, "width": 1, "height": 0.5 }]
}
```

or:

```json
{
  "kind": "document",
  "active": true,
  "visibleBlockIds": ["block-id"]
}
```

Normalized image rectangles must be finite and bounded. Document IDs must be
unique registered blocks and are capped at 64. Sending the wrong variant is a
modality error.

## Ordered updates

`GET /jobs/{jobId}/updates?after=N&waitMs=M` returns the exact `jobId`, the
last returned sequence, and later updates. Sequences start at 1 and strictly
increase without gaps from `after + 1`. An empty long-poll retains the supplied
cursor. One response contains at most 1,024 updates; the browser advances the
cursor and immediately requests the next contiguous page when more remain. The
native log retains at most 10,000 updates. The maximum wait is 20 seconds.

The update union is tagged by `type`:

| Type | Meaning |
| --- | --- |
| `progress` | Current stage and bounded progress/count fields |
| `imageRegionReady` | Final translated image item plus authorized cleanup patch and layout |
| `documentBlockReady` | Final translated document block |
| `imageRegionPreserved` | Terminal source-preserving image item; source pixels remain visible |
| `documentBlockPreserved` | Terminal source-preserving document block; the reader keeps English |
| `complete` | Terminal translated/preserved counts |
| `failed` | Terminal code, message, and retryability |
| `cancelled` | Terminal cancellation |

There is no provisional or pending text update. One shared `TranslatedText`
payload contains:

- authoritative source text;
- faithful/base Chinese and final displayed Chinese;
- pinyin; and
- final HSK state, including teaching-term ranges and repair state.

Each ready wrapper owns its `itemId`. `imageRegionReady` composes
`TranslatedText` with OCR confidence, reading order, style, layout constraints,
and a stored PNG patch descriptor.
`documentBlockReady` composes it with semantic block identity. An
`imageRegionPreserved` contains only `itemId`, text polygon, source text,
confidence, item order, and terminal reason; it has no patch or Chinese text. A
`documentBlockPreserved` contains the item identity, source text, and terminal
reason but no Chinese candidate.

Natural mode publishes faithful Chinese with deterministic teaching metadata.
Strict mode publishes only after HSK realization and at most one terminal
repair. Joined pieces of an oversized individual document block are validated
as a whole before its one block update is appended.

## Replay and acknowledgement

`after` is the last page-installed acknowledgement, not merely the last update
read by the background worker. After MV3 suspension, an unacknowledged update
can replay; the content controller installs the same `itemId` once and advances
the acknowledgement only after all associated DOM work succeeds.

Recovery requires the exact source kind, hash, and `TranslationSettings`; image
recovery additionally requires the exact reading direction. Cancellation,
mutation, or navigation invalidates ownership and tears down the complete
render target. Fatal document failure restores the exact original source-root
attributes.

## Lookup, patches, and speech

`POST /lookup` supports bounded selection lookup or a hover offset owned by a
`jobId` and `itemId`. The daemon resolves the hover against its canonical final
Chinese and returns the longest dictionary expression beginning at that Unicode
offset. It never trusts a browser-provided translated substring.

Image patch blobs are authorized only after their `imageRegionReady` update and
are removed with the owning job. The extension validates and decodes a PNG
before inserting its Chinese text. Document jobs do not create or fetch blobs
or fonts.

Original/Chinese comparison is local browser state. Mandarin playback uses an
eligible local Simplified-Chinese Firefox/OS voice and sends no daemon request.
