# Firefox Gate 1 and bridge-client implementation

## Scope

This branch implements Workstreams A and E through Gate 1, plus the Firefox
half of Gate 2. It does not change the frozen protocol parser or shared contract
fixtures.

The extension uses WXT Manifest V3 with plain TypeScript, DOM, and CSS. The
translator page bundle is an unlisted `translator.js` artifact and is injected
only after a popup action with `activeTab` and `scripting`. No install-time page
content script or required broad webpage host permission is present.

## Runtime shape

- The popup persists one global cumulative HSK 1–6 setting and exposes visible
  and all-image actions plus cancellation and recovered state.
- The background performs the one-shot
  `local.mangalations.hsk_manga` native handshake, keeps the bearer endpoint in
  `storage.session`, and retries an authenticated localhost request once after
  a 401 with a fresh handshake.
- Active job identity, tab/frame/page ownership, source hash, page index,
  fixture flag, timestamp, and decoded dimensions are recoverable from
  `storage.local`. Image/font bytes and bearer tokens are not stored there.
- Remote image acquisition requests an optional permission for the exact
  origin, follows at most three validated HTTP(S) redirects, retries credentials
  only after 401/403, bounds streamed bytes, sniffs MIME signatures, reads
  decoded dimensions from image headers, enforces byte/pixel/dimension limits,
  hashes with Web Crypto SHA-256, and uploads image plus JSON metadata as
  multipart form data. The companion never receives an untrusted source URL.
- The content runtime discovers conservative loaded `<img>` and `<picture>`
  candidates, observes lazy/SPA mutation and intersection state, sorts visible
  images first, and runs one job at a time. Polling is one ordinary message per
  snapshot (1 second while visible, 4 seconds while hidden). Cancel removes
  queued work and cancels the current job; failed images stay original and show
  one Retry action.

## Fixture mode

Local pages opt into Gate 1 fixture mode with
`<html data-hmt-fixture="true">` or `?hmtFixture=1`. The same background job
store, polling messages, queue, binary result delivery, renderer, progress UI,
selection code, and cancellation path are used. Only the companion-side status,
result, lookup, clean-image, and font payloads are deterministic fixtures.

Fixture job status is derived from persisted creation metadata, so closing the
popup or reconstructing the background does not stop or reset it. The adapter
transfers clean-image and font payloads as `ArrayBuffer`; the fixture font is
intentionally invalid to exercise the measured local-system-font fallback.

Synthetic CC0 fixture pages under `fixtures/browser-pages` cover responsive and
lazy images, navigation links, fixed/max widths, object-fit contain/cover,
transform rejection, SPA replacement, cross-origin simulation, selection,
resize/zoom, and renderer modes.

## Selectable renderer

The wrapper is created only after a complete validated result and after font
loading/fallback resolves. It moves the original live `<img>` or `<picture>`
rather than cloning it, verifies wrapping changes layout by no more than two CSS
pixels, and restores the original on failure or teardown.

The wrapper keeps the original as the layout/click anchor. Its Shadow DOM owns
the clean Blob image, normalized selectable text layer, isolated controls,
dictionary popover, and CSS. Chinese enters only through `textContent`;
`innerHTML` is not used. The geometry mapper accounts for border, padding,
object-fit, object-position, contain letterboxing, and cover cropping.
`ResizeObserver` refits on responsive changes and zoom.

The browser fitter starts with the companion's suggested size/lines, performs a
rectangle fit, then checks usable polygon spans while respecting Chinese
opening/closing punctuation. The DOM preserves the exact `displayedChinese`
text without synthetic newline nodes. Validated style application includes
local font family, colour, bounded weight/slant, outline, shadow, rotation,
alignment, spacing, and horizontal/vertical writing mode. Overflow is marked
`data-fit="degraded"` for diagnostics.

Original and Chinese are persistent modes; Compare is press-and-hold. Controls
and dictionary interactions cannot trigger an enclosing reader link. A normal
click without selection still bubbles to it, while a click carrying a
non-collapsed translated-text selection is stopped. Default copy is normalized
to exactly the selected Chinese and includes no source English or metadata.
Focused regions support Ctrl/Cmd+A keyboard selection.

## Local verification

Run from `extensions/firefox`:

```text
wxt build -b firefox
tsc --noEmit
vitest run
playwright test
web-ext lint --source-dir .output/firefox-mv3
```

Latest local Gate 1 evidence:

- WXT Firefox MV3 production build: passed; `translator.js` is packaged but
  absent from the manifest's static content scripts.
- TypeScript strict typecheck: passed.
- Vitest: 56 tests passed across 15 files.
- Playwright Firefox renderer harness: 5 tests passed. Playwright intentionally
  tests the regular-page renderer harness because it cannot load Firefox
  extensions.
- `web-ext lint`: 0 errors, 0 notices, 2 warnings. Both warnings are the known
  compatibility interaction between the retained Firefox 128 minimum and the
  newer Firefox data-collection declaration.
- `web-ext run`: a bounded headless system-Firefox temporary-profile smoke
  reached a live session and was then terminated; no pre-existing Firefox
  process was stopped.

## Remaining integration gaps

- Workstream B must supply and test the installed native launcher/daemon,
  loopback-only binding, origin rejection, bearer rejection, daemon lifetime,
  real endpoints, and native-host registration. This branch tests the client
  handshake, authentication headers, one 401 refresh, recovery, multipart
  shape, and response bounds with deterministic fakes.
- The cross-origin optional-permission prompt and actual extension runtime
  structured-clone ceiling still need an installed-extension manual run. Unit
  coverage clones an 8 MiB clean image and the client test transfers a 5 MiB
  response without copying it into Firefox storage.
- Real redistributable companion font bytes are pending the model/data pack.
  The successful `FontFace` cache and failed-font fallback are unit tested, and
  the binary font message path is implemented.
- Playwright does not validate popup injection or native messaging because it
  cannot install a Firefox extension. The regular Firefox `web-ext run` manual
  checklist must cover popup → injection → fixture translation and the exact
  permission prompt before integration Gate 2 is accepted.
- Golden-set clipping, actual OCR/inpainting/translation quality, strict HSK
  vocabulary, and cache reuse are later companion/model gates and are not
  claimed here.
