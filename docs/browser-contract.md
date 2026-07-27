# Unversioned browser contract

The browser daemon mounts routes directly at its random loopback origin. There
is no `/v1` or `/api` prefix and no separate result resource.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Exact build fingerprint, engine version, readiness, and sorted resident-resource identities |
| `GET` | `/setup` | Current local-resource setup state |
| `POST` | `/setup/models` | Start or report managed resource setup |
| `POST` | `/jobs` | Validate multipart image + JSON metadata and create a job |
| `DELETE` | `/jobs/{job_id}` | Cancel the job |
| `PUT` | `/jobs/{job_id}/viewport` | Replace visible normalized rectangles and active state |
| `GET` | `/jobs/{job_id}/updates` | Replay or long-poll flat updates after a sequence |
| `POST` | `/lookup` | Local pinyin/dictionary lookup, optionally bound to a job region |
| `GET` | `/blobs/{patch_id}` | Fetch one job-owned `image/png` cleanup patch |
| `GET` | `/fonts/{font_id}` | Fetch one permitted installed font |

The launcher alone calls `/browser-internal/session` with the discovery
control secret. That path is an implementation-private bootstrap endpoint: it
is not CORS-enabled, is not exposed as the browser API, and does not imply
protocol negotiation.

## Authentication and build identity

Every browser route requires:

- `Host: 127.0.0.1:<actual-port>`;
- one active canonical extension origin;
- `Authorization: Bearer <session-token>`; and
- `X-HSK-Manga-Extension-Origin` when privileged Firefox fetches omit the
  standard `Origin` header.

There is deliberately no protocol header. The exact fingerprint
`hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-26-r2` is validated in the native handshake and job
request and echoed by native readiness, health, and job creation. Unknown JSON
fields are rejected by the contracts.

## Job creation

`POST /jobs` accepts exactly two multipart fields:

- `image`: PNG, JPEG, WebP, or GIF bytes whose multipart type, declared type,
  sniffed type, SHA-256, and decoded dimensions agree;
- `request`: `application/json` metadata containing the exact build
  fingerprint, source identity, dimensions, page identity, HSK 2.0 level 1–6,
reading direction, visible rectangles, up to six preceding utterances, and an
optional bounded proper-name glossary.

The only supported language pair is English to Simplified Chinese. Sound-effect
translation must be false. A successful request returns HTTP 202 with only the
build fingerprint and `jobId`.

## Flat progressive updates

`GET /jobs/{job_id}/updates?after=N&waitMs=M` returns:

```json
{
  "jobId": "job-id",
  "nextSequence": 12,
  "updates": [
    {
      "type": "progress",
      "sequence": 11,
      "stage": "translating",
      "overallProgress": 0.42,
      "message": "Translating English directly into HSK-targeted Chinese"
    },
    {
      "type": "regionReady",
      "sequence": 12,
      "region": {}
    }
  ]
}
```

`after` is the last acknowledged sequence. The daemon returns only later
entries, waits at most 20 seconds, and returns an empty batch on timeout.
Sequences start at 1 and strictly increase. `nextSequence` is the last returned
sequence, or the supplied cursor for an empty batch.

The update union is flat and tagged by `type`:

| Type | Meaning |
| --- | --- |
| `progress` | Current stage plus optional stage/overall fraction and count |
| `regionReady` | Complete renderable region and its stored patch descriptor |
| `regionRefined` | Replacement displayed Chinese, pinyin, and HSK status for an already published region |
| `complete` | Successful terminal event |
| `failed` | Terminal error code, message, and retryability |
| `cancelled` | Cancelled terminal event |

Stages are `queued`, `decoding`, `detecting`, `ocr`, `inpainting`,
`translating`, `hsk-validating`, `styling`, and `packaging`. Clients must not
infer a page-result phase from them. Accurate primary Chinese is published
immediately even when it contains marked above-level vocabulary. The one
permitted singular background repair may later publish `regionRefined`; it
never restarts page-wide generation.

A `regionReady` contains normalized text and optional bubble polygons, a
normalized PNG patch rectangle and blob ID, source English, direct/base and
displayed Chinese, pinyin, OCR confidence, reading order, validated style and
layout, and HSK status. The status carries requested level, strict validity,
above-level tokens, and one of `not-needed`, `pending`, `accepted`, or
`rejected`.

## Patch-before-text invariant

The daemon stores a valid PNG blob before appending its `regionReady` event.
The extension authorizes the blob only after receiving that event, fetches and
validates it, decodes it, and synchronously inserts the patch image before the
selectable text node. If patch loading fails, that region is not installed as
text over uncleaned English.

The original page image is never replaced with a cleaned-page response. The
reader result is the original image plus a progressive patch layer and a text
layer.

## Lookup, comparison, and speech

`POST /lookup` accepts up to 256 selected characters. Job and region IDs must
be supplied together when region context is requested. Results contain
longest-match tokens with Simplified spelling, pinyin, definitions, optional
HSK level, and explicit proper-name state.

Original/Chinese comparison is entirely in the extension's layered renderer.
Mandarin playback sends no daemon request: Firefox speaks selected Chinese
through the best eligible local Simplified-Chinese voice.
