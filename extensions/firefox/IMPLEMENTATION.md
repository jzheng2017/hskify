# Firefox production extension implementation

## Scope

This directory contains the Firefox side of the HSK Manga Translator, including
the browser interaction layer, secure native/loopback client, recovery queue,
selectable renderer, setup UI, and packaged-extension assets. It retains the
frozen protocol parser and shared contract fixtures.

The extension uses WXT Manifest V3, TypeScript, DOM, and CSS. The page runtime
is built as the unlisted `translator.js` artifact and is injected only after a
popup action using `activeTab` and `scripting`. It is not a static content
script, and the manifest does not request broad webpage access at install time.

## Permission and message boundary

- Opening the popup injects the page runtime and precomputes the cross-origin
  image hosts required for visible and all-image actions.
- Firefox match patterns are exact, deduplicated, and portless
  (`https://cdn.example/*`). The manifest merely declares the optional
  HTTP/HTTPS pattern scope; a concrete subset is requested at runtime.
- `browser.permissions.request()` is invoked directly in the translate button's
  click stack. The popup does not await background or content work before that
  call, and it suppresses duplicate starts while the request is pending.
- Background acquisition only checks `permissions.contains()`. It never tries
  to open an optional-permission prompt from an asynchronous message handler.
- If a redirect or newly loaded image reveals another hostname, that exact
  pattern is retained in `storage.session` and merged into the next popup
  click's permission plan.
- Every known background and content message is parsed into an exact shape.
  Unknown fields, invalid bounds, malformed hashes, oversized runtime buffers,
  and page-controlled fixture switches are rejected.
- Active-job operations require the originating tab, frame, and document URL.
  Completed lookup/font operations additionally require a small
  owner-and-artifact record that lists the result's allowed region and font
  IDs.

Production has no page-controlled fixture mode. The deterministic backend is a
constructor-injected test dependency, and no fixture adapter or fixture marker
is present in the production WXT output.

## Image and companion lifecycle

The content runtime attempts bounded byte acquisition for same-origin,
`data:`, and `blob:` sources. It streams those responses and stops before
materializing a payload over 25 MiB. When background acquisition is needed, it:

1. validates HTTP(S) URLs and each redirect;
2. verifies a pre-granted exact host pattern for each cross-origin hop;
3. fetches without credentials first and retries with credentials only after
   401/403;
4. rejects unsafe credentialed cross-origin redirects;
5. streams with a byte ceiling;
6. sniffs PNG, JPEG, WebP, or GIF signatures and checks declared MIME,
   dimensions, pixel count, and configured limits;
7. hashes the actual bytes with Web Crypto SHA-256; and
8. uploads bytes plus the frozen request metadata as multipart form data.

The companion never receives a page-controlled remote URL to fetch.

Native session endpoints remain in `storage.session`. A background instance
that reuses a cached endpoint first validates `/health`. A failed transport or
401 invalidates the lease, performs one fresh native handshake, and retries
once. Clean-image and font responses are streamed with 25 MiB and 32 MiB
ceilings respectively before an `ArrayBuffer` is created.

Before result delivery, the background verifies:

- the caller's full page/source identity;
- result job ID, source hash, and decoded source dimensions;
- cleaned-image signature and declared MIME; and
- cleaned-image dimensions equal to the submitted source.

Only then is the result transferred to the page runtime.

## Recovery, navigation, cancellation, and retry

`storage.local` holds only small active-job recovery records. Each record
includes tab, frame, page session, normalized document/source URLs, source
hash, decoded dimensions, page index, selected HSK level, and creation time.
Completed-result authorization records use `storage.session`, so tab IDs
cannot accidentally inherit them across browser restarts.

Recovery is scoped to tab, frame, page session, and document URL. URL,
dimensions, and SHA-256 must match the live image. For an HTTP(S) candidate
whose content script cannot supply a hash, the background reacquires and hashes
the bytes. DOM index is only an additional mapping key; it is never sufficient
identity by itself.

The page runtime checks generation, page-session ID, document URL, source URL,
intrinsic dimensions, connectivity, and cancellation after every awaited
operation and immediately before renderer commit. If navigation or source
replacement happens while submission is awaited, the returned job identity is
retained long enough to cancel that newly created companion job.

Full and same-document navigation cancel the old page session, remove its
completed authorization records, restore rendered originals, and create a new
page-session ID. `tabs.onUpdated` URL changes and `tabs.onRemoved` provide a
background-side cleanup path as well.

The queue is one-at-a-time and visible-first. Failed queue IDs remain failed;
visibility and mutation callbacks cannot silently enqueue them again. Only the
image Retry action clears the failed state. Source removal/replacement aborts
the affected active item and updates `current`, `failed`, `completed`, and
`total` without double-counting.

## Selectable renderer

The original live `<img>` or `<picture>` remains unchanged while work is in
progress. The renderer:

- browser-decodes the cleaned Blob image and verifies its actual intrinsic
  dimensions before creating a wrapper;
- awaits result-owned font loading and `document.fonts.ready`;
- rechecks the stale-render guard after every wait and immediately before DOM
  mutation;
- moves the original live owner instead of cloning it;
- rejects a wrapper that changes controlled layout by more than two CSS pixels;
- keeps the original as the layout anchor and restores it on every failure;
- uses a Shadow DOM for the clean image, text, controls, and lookup popover; and
- inserts Chinese only with `textContent`.

Suggested line breaks are represented by real `.hmt-region-line` spans while
the region's text content remains exactly `displayedChinese`. A genuinely
inset companion safe polygon is preferred. When the companion repeats the
complete bubble hull as its safe polygon, the renderer instead uses the
original OCR text polygon so outlines and speech tails cannot be treated as
text space. Fitting reserves an inner margin, permits up to six legal CJK
lines, rejects whitespace at line boundaries, and keeps font scaling
proportional instead of enforcing an eight-pixel floor. After the font is
loaded, real DOM overflow is measured and the font is reduced in small steps;
the region clips any residual ink rather than allowing it to spill outside.

An unselected primary click in translated text dispatches exactly one
non-bubbling click to the original image, preserving direct image listeners.
The overlay click itself continues to the reader ancestor exactly once.
Selected text, controls, and the dictionary popover suppress reader navigation.
Copy reads the selected range's text content so visual line spans add no hidden
English or layout whitespace.

The lookup popover also has a Listen/Stop control for the exact selected
Chinese. It uses the Web Speech API with `zh-CN`, preferring a Mainland
Mandarin voice and then natural/neural voice variants when Firefox exposes
them. Playback is delegated to Firefox and the operating system, so no speech
model, service key, extension permission, or companion memory is added. The
installed voice ultimately determines audio quality.

## Synthetic browser fixtures

The browser pages use actual generated PNG/WebP assets, not SVG page inputs or
header-only fake buffers. The source artwork is original synthetic geometry.

`fixtures/browser-pages/webtoon.html` models a long reader with:

- 20 query-string WebP chapter images at 900×16,000 intrinsic pixels;
- one cover and 133 generated comment avatars, for 154 page images total;
- page-image width capped at 720 CSS pixels; and
- a 465 CSS-pixel responsive width at a 480-pixel viewport.

Discovery regression coverage selects exactly the 20 chapter pages while
excluding the cover and comment/user images. Firefox Playwright also decodes
the long WebP at its production dimensions.

## Local verification

Run from `extensions/firefox`:

```text
npm run typecheck
npm run test
npm run test:e2e
npm run build
npm run lint:extension
```

Latest evidence for this branch:

- strict TypeScript/WXT typecheck: passed;
- Vitest: 91 tests passed across 20 files;
- Playwright Firefox renderer harness: 6 tests passed;
- WXT Firefox MV3 production build: passed;
- production-output fixture-marker scan: no matches;
- `web-ext lint`: 0 errors, 0 notices, 2 compatibility warnings;
- bounded `web-ext run` packaged launch: passed using a temporary profile,
  pre-installed extension, headless Firefox, and an auto-exiting screenshot
  smoke.
- live Nano Machine chapter 100 page 1: 11 English speech-bubble regions,
  non-bubble sound effects retained, and 11 real selectable Chinese regions
  rendered with no degraded fit. A post-correction replay measured zero DOM
  overflow, at least 6.9 CSS pixels of inset inside every conservative text
  region, and reduced the crowded long-bubble fonts from about 50–52 pixels to
  37–39 pixels.
- installed packaged-extension E2E: a fresh disposable Firefox profile used a
  trusted click on the real popup, the registered native host, the installed
  daemon, cached clean-image reuse, and HSK 5 correction to finish 11 regions
  with zero degraded fits in 186.7 seconds.

The two lint warnings are the known compatibility interaction between retained
Firefox 128 support and the newer
`browser_specific_settings.gecko.data_collection_permissions` declaration.

## Unclaimed integration work

The primary Windows installed path is proven, but the implementation is not
claiming every release edge case. In particular:

- the installed native launcher/daemon and real companion endpoints completed
  one full Firefox translation; explicit duplicate-launch, idle-cleanup,
  authentication-rejection, suspension, and reconnect probes remain;
- practical 5–20 MiB Firefox extension runtime-message ceilings still require
  an installed packaged-extension probe; unit cloning is not evidence for that
  browser limit;
- popup permission UI, denial, and a redirect to a newly required hostname
  still need an interactive packaged Firefox run;
- the packaged Noto CJK font bytes and production companion endpoints are
  implemented, but broader font-category and golden-set clipping review
  remains pending; and
- live OCR, speech-bubble cleanup, local translation, bounded HSK correction,
  and cache reuse are implemented, while representative fluent-reader quality
  review remains release work.

The exact manual matrix is maintained in
`docs/firefox-manual-test-checklist.md`.
