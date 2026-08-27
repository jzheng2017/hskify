# Firefox extension implementation

The Firefox MV3 extension owns one current-chapter run and selects exactly one
rendering mode. The fixed build fingerprint is
`hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-08-09-r8`; a different native
build fails closed.

## Chapter selection

The page classifier attempts a site-independent document descriptor before
image discovery. It marks a cloned document, runs `@mozilla/readability` 0.6.0
with a DOM serializer, and maps accepted semantic nodes back to the live DOM.
The clone is disposable and Readability output is never injected.

Document mode requires at least five story blocks, 1,000 normalized characters,
predominantly English alphabetic text, complete marker mapping, and a content
root below `body`. If it passes, it wins and illustrations remain untouched.
Otherwise the existing sequential-art detector may choose image mode. An
uncertain page is unsupported; there are no site selectors or manual mode
setting.

Whitespace, stable block IDs, and the ordered snapshot hash are deterministic.
The snapshot includes titles, leveled headings, paragraphs, blockquotes, list
items, figure captions, illustrations, and separators while excluding site UI,
navigation, ads, metadata/bylines, comments, and forms. Oversize snapshots are
rejected rather than partially translated; the 1 MiB check measures the
complete compact `/jobs/document` request envelope.

## Shared ownership

The controller owns native-session discovery, job creation, one update poll,
acknowledgement, cancellation, replay, HUD, source/navigation monitoring, and
chapter release. It delegates modality behavior to `DocumentChapterMode` or
`ImageChapterMode`. Active jobs and artifacts store an exact tagged
`image | document` source identity.

Storage keys, classes, and owned data attributes use the `hskify` prefix. Page
acknowledgement advances only after every update in the batch is installed. An
unacknowledged final block may replay after background suspension, but stable
`itemId` installation is idempotent. Recovery also requires the same source
kind, hash, and exact translation settings. Image recovery additionally
requires the same reading direction. Returned update sequences must be
contiguous from the requested acknowledgement cursor.

Before using an existing chapter tab, the background health-checks its content
runtime. It replaces and verifies a missing or stale runtime, while the popup
retries transient preparation failures on its normal refresh interval. WXT or
add-on reloads therefore do not require reloading the chapter page.

The background calls:

- `POST /jobs/image` for multipart raster jobs;
- `POST /jobs/document` for JSON snapshots;
- `PUT /jobs/{jobId}/focus` for tagged image rectangles or visible blocks;
- `POST /warmup` with the detected chapter kind;
- the common update, cancellation, chapter release, lookup, blob, font, and
  setup routes documented in `docs/browser-contract.md`.

`GET /setup` remains a resource-only status read. Passive content detection
retains the detected kind if resources are not installed yet and retries its
tagged warm-up after installation. The popup likewise POSTs one tagged warm-up
for the detected kind before enabling translation; it does not make setup
status polling allocate a runtime. The always-on passive probe runs the
Readability clone/parser only once at `document_idle`; its short discovery
window repeats only cheap image checks on unsupported pages.

## Document reader

Mapped live block elements remain connected in their site-defined positions.
The renderer retains their original child nodes and attributes, then installs
only safe final Chinese text and teaching spans inside those same elements. It
adds no chapter surface, background, typography, or arbitrary Readability HTML.
Pending and `documentBlockPreserved` items use empty line-box placeholders, so
Chinese mode never mixes source English into translated prose.

Original mode, cancellation, source mutation, SPA navigation, fatal failure,
and disposal restore the retained source nodes and remove only Hskify metadata
and overlay UI. Mode changes preserve a block-relative scroll anchor. The
dictionary overlay is viewport-fixed and anchored to the selected character or
range rather than the containing paragraph.

An `IntersectionObserver` tracks the in-place blocks and a 100 ms coalescer reports at
most 64 visible block IDs. Scroll handlers perform no geometry reads. The first
visible block is eligible for a single-item native dispatch before token-aware
batches of at most six.

## Image reader

Image mode retains the established patch-before-text renderer. Source images
stay connected. An image job's transparent cleanup patch is fetched, validated,
decoded off-DOM, and installed before its final selectable Chinese. Normal
scrolling does not trigger a synchronous layout loop. Reading direction is an
image-only popup setting.

## Shared interaction layer

Both renderers implement the same small render-target interface for one
singleton Original/Chinese/hold-to-compare controller. They also share teaching
term markup, position-aware local dictionary lookup, pinyin display, selection,
and local Mandarin speech. Lookup ownership is `itemId` in both modes.

The popup reports `document`, `image`, or `unsupported` detection and retains
one “Translate chapter” action. It shows reading direction only for image
chapters.

## Verification

From `extensions/firefox`:

```text
pnpm typecheck
pnpm test
pnpm build
```

Fixtures cover document extraction without live-DOM mutation, unsafe root and
input rejection, hybrid/manga classification, progressive final-only rendering,
exact restoration, comparison and teaching tools, focus coalescing, SPA
navigation, cancellation, and MV3 replay. Existing image contract, geometry,
renderer, and real-reader regressions remain required under the new identities.
