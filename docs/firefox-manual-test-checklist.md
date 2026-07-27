# Firefox performance-build manual checklist

All rows are pending for the current direct progressive build unless an
evidence bundle records the exact fingerprint, binary/extension hashes,
Firefox version, RTX 4080 SUPER/CUDA environment, commands, timestamps, and raw
artifacts. Results from the retired page-result build do not satisfy this list.

## Installation and trust

- [ ] Build through `scripts/Invoke-PerformanceBuild.ps1` on the exact GPU.
- [ ] Preserve and verify the matching performance-build attestation; confirm
  source-tree identity, Windows x86_64 MSVC release/CUDA feature, CUDA
  13.1/ORT 13/`sm_89`, tool versions, device-0 16,376 MiB/API 13.1 identity,
  and both executable byte/hash claims.
- [ ] Package and install the matching extension, native host, daemon, data,
  attestation, fonts, and Qwen3.5 4B revision.
- [ ] Verify Firefox invokes only `local.hskify.hsk_manga` for
  `hsk-manga-translator@local.hskify`.
- [ ] Verify a mismatched build fingerprint fails closed.
- [ ] Verify the daemon binds only to a random IPv4 loopback port and rejects
  wrong Host, origin, token, duplicate headers, and unsupported preflight.
- [ ] Verify the detached daemon survives one-shot host exit and cleans up
  after the 30-minute idle window.

## Progressive reader behavior

- [x] Complete independent region and geometry review for all 36 Chapter 5
  images.
- [x] Complete the remaining Chinese, pinyin, and HSK-token gold before
  treating any run as release evidence.
- [ ] Run all 36 hash-pinned Chapter 5 images in reader order.
- [ ] Verify visible tiles/regions arrive ahead of off-screen work.
- [ ] For every `regionReady`, verify the PNG is fetched, validated, decoded,
  and inserted before selectable Chinese.
- [ ] Verify no whole cleaned-page request or response occurs.
- [ ] Verify region updates survive extension background suspension and resume
  from the last acknowledged sequence without duplication.
- [ ] After at least one image has completed and another is partial, verify
  cancellation produces one terminal event, no later patch/text, and restores
  the exact original chapter DOM (`src`, `srcset`, `sizes`, class, style,
  attributes, siblings, and image order) with no Hskify wrapper or marker.
- [ ] Verify a same-document, same-tab history navigation and an active-image
  source replacement each restore every original chapter image exactly.
- [ ] Verify disposing the content controller twice is idempotent and leaves
  the same exact original chapter snapshot.
- [ ] Verify original, Chinese, and hold-to-compare controls preserve page
  geometry and navigation.
- [ ] Verify selection lookup returns pinyin, definitions, HSK overlay, and the
  correct bound region context.
- [ ] Verify local Mandarin speech chooses an eligible local Simplified-Chinese
  voice, cancels correctly, and exposes a clear unavailable state without
  network fallback.

## Story-region gating

- [ ] Annotated English dialogue, thought, and story narration is detected and
  OCRed.
- [ ] Sound effects, credits, promotion, branding, non-English text, ambiguous
  regions, and sub-0.45 OCR do not create patches or translations.
- [ ] Light, dark, gradient, textured, outlined, and arbitrarily colored
  regions work without hue-specific rules.
- [ ] Patch alpha is confined to the intended glyph mask.
- [ ] Artwork and pixels outside the erase mask remain unchanged; local color,
  gradients, texture, contours, and styling are preserved.
- [ ] Direct Chinese is meaningful and natural; strict/repair/rejected HSK
  states match deterministic validation.

## Performance evidence

- [ ] Capture cold, warm-up, and at least 20 measured warm chapter runs.
- [ ] Capture time to first region, first installed patch, first selectable
  text, 50 percent regions, and terminal completion.
- [ ] Define cancellation latency from cancel issuance to the exact DOM
  restoration observer and daemon `cancelled` update observation; keep later
  health, replay, and patch evidence outside that measured interval.
- [ ] Capture per-page/chapter p50 and p95, RAM, private bytes, VRAM, GPU
  utilization/power/temperature, CPU, errors, cache hits, and repair counts.
- [ ] Preserve raw samples and hashes under an evidence path outside gold data.
- [ ] Keep optional live-site network timing separate.

See [the benchmark method](chapter-5-benchmark.md). A checked box without its
linked raw evidence is not a passed release check.
