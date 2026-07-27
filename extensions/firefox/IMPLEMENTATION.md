# Firefox extension implementation

The Firefox MV3 extension is a direct client of the local, unversioned
progressive companion API. There is no legacy result download or full cleaned
image path.

## Companion contract

Every native and HTTP handshake is pinned to the build fingerprint
`hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-26-r2`. A different fingerprint is a hard failure,
not a negotiated compatibility mode.

The background worker uses these loopback routes:

- `POST /jobs` uploads the original image and JSON metadata as multipart
  `image` and `request` parts.
- `PUT /jobs/{jobId}/viewport` sends normalized visible source rectangles and
  whether the image is actively being processed.
- `GET /jobs/{jobId}/updates?after={sequence}&waitMs=20000` long-polls
  monotonic updates.
- `GET /blobs/{patchId}` downloads a region's transparent PNG patch.
- `DELETE /jobs/{jobId}` cancels and releases a job.

Setup, dictionary, and font requests remain authenticated root routes:
`/setup`, `/setup/models`, `/lookup`, and `/fonts/{fontId}`.

`JobUpdate` is the discriminated union `progress`, `regionReady`,
`regionRefined`, `complete`, `failed`, and `cancelled`. `regionReady` carries
geometry, patch identity and rectangle, English/base/displayed Chinese,
pinyin, OCR confidence, reading order, typography/layout, and HSK validation
and repair state. `regionRefined` can change only displayed Chinese, pinyin,
and HSK state.

## MV3 recovery and ownership

Active job records are stored in `browser.storage.local`. They contain the
tab/frame/page/source identity, the last delivered sequence, the last
page-acknowledged sequence, and the region/patch/font IDs observed so far.
The page acknowledges a batch only after every patch and text mutation in it
has completed. After background suspension, recovery resumes from that
installed sequence; an unacknowledged batch is safely replayed.

All job, patch, font, and lookup messages are checked against the owning
tab/frame/document URL and persisted source URL/hash/dimensions. Runtime
messages use strict allowlists and bounded binary payloads. Page navigation,
source replacement, cancellation, disposal, or ownership mismatch uses one
idempotent restore-all path: completed and partial overlays are both destroyed,
every original image returns to its exact parent/sibling position with its
original attributes intact, and the companion job is released.

## Progressive rendering

The renderer keeps the exact original `<img>` node connected and visible. A
layout-preserving wrapper adds a Shadow DOM containing:

- a transparent patch layer;
- selectable Chinese text;
- Original, Chinese, and hold-to-compare controls; and
- the dictionary/pinyin/Mandarin-speech popover.

For each `regionReady`, the patch blob is downloaded and decoded completely
off-DOM. Only then is the patch synchronously installed, followed by its text
node. A corrupt, stale, or cancelled patch can therefore never expose Chinese
over source lettering. `regionRefined` replaces text-node content and updates
pinyin/HSK metadata without changing geometry, styling, or the installed
patch.

Original and compare modes hide only the overlay. They never hide or replace
the page image. Destroying the renderer restores the original node to its
exact parent and sibling position.

## Geometry, viewport priority, and fitting

Image geometry accounts for borders, padding, `object-fit`, and
`object-position`. Visible source rectangles also account for cover cropping
and browser viewport intersection. Scroll, resize, visibility, and image-size
changes are coalesced into viewport updates at roughly 100 ms.

Text fitting tests nearby legal Chinese line breaks against the safe polygon.
Model fitting and final DOM measurement both use bounded binary searches.
The final measurement pass checks scroll dimensions, stays inside the
subpixel boundary, and has a zero-size fallback only for degenerate geometry,
so selectable text never overflows its region.

## Verification

From `extensions/firefox`:

```text
npm run typecheck
npm test -- --run
npm run test:e2e
npm run build
```

The Vitest suite covers strict progressive contracts, exact root endpoints,
update acknowledgement/recovery, patch ownership, atomic patch installation,
refinement, viewport messages, measured fitting, selection, dictionary
pinyin, and Mandarin speech. The Playwright Firefox harness covers real image
decode, normalized geometry, object-fit mapping, compare modes, navigation,
selection, vertical text, and long WebP dimensions.
