# Firefox implementation

The extension implements the r9 source-ownership contract described in [architecture](../../docs/architecture.md) and [browser contract](../../docs/browser-contract.md).

- `discovery/chapter.ts` shares classification between warmup and active translation. `document/extraction.ts` owns region mapping, complete eligible-text extraction, language sampling, immutable revisions and sentence-group identity. `document/region-selection.ts` handles ambiguous/short sources.
- `document/reader.ts` owns disjoint translated text slots and canonical original Text nodes. Original elements, illustrations and nested lists remain connected. Mutation records are drained before owned writes; installs validate source ownership.
- `discovery/surfaces.ts` is the incremental source registry for images, backgrounds, canvases and same-origin frames. `discovery/images.ts` supplies shared eligibility and lazy-image rules. Visibility is actual IntersectionObserver visibility, distinct from prefetch.
- `acquisition/rendered-region.ts` serializes Firefox fallback captures and restores overlay visibility on success, failure or cancellation. `rendering/surface-transform.ts` and `rendering/geometry.ts` share coordinate decisions between acquisition and rendering.
- `page/chapter-run-controller.ts` owns run phases and the shared poll/install/ack lifecycle. Mode adapters preserve successful items on retry and keep discovery active after settlement. `messaging/background.ts` owns authenticated transport, source capture, bounded acquisition and recovery replay.
- The existing renderer, HSK controls, comparison tools, local pinyin/dictionary and Mandarin speech remain shared. Lookup supplies bounded item context and remains usable after native job eviction.

From this directory run `pnpm typecheck`, `pnpm test`, `pnpm zip`, and `pnpm lint:extension`. From the repository run `node scripts/test-redesign-firefox.mjs` for real Firefox tests of production modules. These module tests do not substitute for packaged native/GPU latency or annotated blind quality evaluation; see [release gates](../../docs/reader-redesign-evaluation.md).
