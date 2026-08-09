# Firefox performance-build manual checklist

All rows are pending unless an evidence bundle records the exact fingerprint,
binary and extension hashes, Firefox version, RTX 4080 SUPER/CUDA environment,
commands, timestamps, and raw artifacts.

## Installation and trust

- [ ] Build through `scripts/Invoke-PerformanceBuild.ps1` on the supported GPU.
- [ ] Verify the matching attestation covers the exact source, Windows MSVC
  release/CUDA configuration, toolchain, hardware, and both executable hashes.
- [ ] Install the matching extension, host, daemon, language data, model pack,
  fonts, and licences.
- [ ] Verify Firefox invokes only `local.hskify.browser` for
  `hskify@local.hskify`.
- [ ] Verify a different build fingerprint fails closed.
- [ ] Verify the daemon rejects wrong Host, origin, bearer token,
  `X-Hskify-Extension-Origin`, `X-Hskify-Control`, duplicate headers, and
  unsupported preflight.

## Mode selection

- [ ] Semantic and div-heavy novel chapters select `document`.
- [ ] Navigation, ads, comments, metadata, and unrelated prose do not enter the
  document snapshot.
- [ ] A mapped `body` root, fewer than five blocks, fewer than 1,000 normalized
  characters, non-English prose, oversize input, or incomplete marker mapping
  is rejected.
- [ ] Hybrid prose with illustrations selects `document` and never OCRs its
  illustrations.
- [ ] Manga/webtoon pages with unrelated surrounding prose select `image`.
- [ ] A page that satisfies neither detector reports `unsupported` and is not
  modified.
- [ ] Readability parsing leaves the live DOM byte-for-byte/attribute-for-
  attribute unchanged before the reader mounts.

## Document reader

- [ ] A complete semantic reader skeleton mounts adjacent to the source root in
  under the benchmark bound, with English in every pending placeholder.
- [ ] Only final `documentBlockReady` Chinese becomes visible; a preserved block
  remains English and no block visibly revises.
- [ ] Teaching-term offsets, pinyin, hover dictionary, selection, and Mandarin
  speech resolve against the joined final block.
- [ ] Original, Chinese, and hold-to-compare preserve a stable scroll anchor.
- [ ] Source mutation, same-tab SPA navigation, cancellation, fatal failure, and
  repeated disposal remove Hskify nodes and restore every original root
  attribute exactly.
- [ ] An unacknowledged block replays after MV3 suspension and installs once.
- [ ] A document-hash or modality mismatch cannot recover stale output.
- [ ] Scrolling causes no synchronous geometry loop; focus changes are
  coalesced at 100 ms and contain at most 64 visible block IDs.
- [ ] Instrumentation records no vision, OCR, projector, segmentation,
  inpainting, patch, or font initialization/invocation for document work.

## Image reader

- [ ] Run the complete real-reader-v2 core/stress selection in reader order.
- [ ] Visible tiles/regions overtake off-screen queued work without changing
  canonical reading/context order.
- [ ] Every `imageRegionReady` patch is authorized, fetched, validated, decoded,
  and inserted before selectable Chinese.
- [ ] Source images stay connected and exact artwork outside accepted erase
  masks remains unchanged.
- [ ] Cancellation, navigation, source replacement, and repeated disposal
  restore every original image and remove every Hskify marker.
- [ ] Image output matches the recorded pre-refactor visual and language
  regressions under the renamed contract.
- [ ] Reading-direction controls appear for image chapters and not documents.

## Shared correctness and replay

- [ ] Names, pronouns, numbers, questions, dialogue continuity, and mid-chapter
  viewport jumps retain faithful meaning.
- [ ] Natural mode publishes faithful Chinese with exact teaching metadata.
- [ ] Strict mode performs at most one terminal repair and invalid joined blocks
  preserve their English source.
- [ ] Lookup is owned by `itemId` and cannot cross jobs, modalities, or source
  hashes.
- [ ] Completion counts translated and preserved items correctly.
- [ ] Persistent cache entries cannot cross image/document kind, level, mode,
  source/context hash, model/prompt/validator identity, or HSK resources.

## Performance evidence

- [ ] On a representative 300-block/100,000-character document, extraction plus
  skeleton p95 is below 100 ms and no scroll-time task exceeds 50 ms.
- [ ] The first visible block dispatches alone; later language batches contain
  at most six real-tokenizer units.
- [ ] Run `npm run benchmark:document`; its native counters prove language-only
  document warm-up initializes and invokes no vision path.
- [ ] Record the pre-refactor image baseline with
  `npm run benchmark:image -- --record-baseline <path>`, then pass that exact
  artifact to the rebuilt run with `--baseline <path>` or
  `HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH`.
- [ ] The measured image language throughput regresses no more than 10 percent,
  and the 100 ms `nvidia-smi` trace keeps the complete 4,096-token image runtime
  inside 16 GB VRAM.
- [ ] Preserve raw samples and hashes outside gold fixture data.
