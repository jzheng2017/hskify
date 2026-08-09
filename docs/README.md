# Hskify documentation

These documents describe the current Windows/CUDA Firefox product. Hskify has
one chapter controller and two exclusive reading modes: a structured document
reader for light novels and the existing image renderer for manga and
webtoons. There is no compatibility API, migrated storage schema, or alternate
provider path.

## Read in this order

- [Architecture](architecture.md): classification, shared language service,
  image/document pipelines, scheduling, cache, and runtime boundaries.
- [Browser contract](browser-contract.md): exact unversioned routes, payload
  limits, focus updates, and final-only job events.
- [Firefox implementation](../extensions/firefox/IMPLEMENTATION.md): cloned-DOM
  extraction, render ownership, recovery, and interaction behavior.
- [Browser companion implementation](../crates/browser-companion/IMPLEMENTATION.md):
  native runtime and pipeline details.
- [Real-reader v2 corpus](real-reader-v2.md): image-reader correctness and
  performance evidence.
- [Manga pipeline audit](manga-pipeline-audit.md): image-pipeline correctness
  findings and evidence requirements.
- [Model benchmark](model-benchmark.md): translation quality requirements.
- [External component evaluation](component-evaluation.md): retained and
  rejected image-processing components.
- [Firefox manual checklist](firefox-manual-test-checklist.md): packaged checks
  for both chapter modes.
- [Maintainer guide](maintainer-guide.md): invariants and verification.
- [Licence inventory](licence-inventory.md): shipped code, data, model, font,
  and browser dependency obligations.

## Accepted decisions

- [Historical story-region processing decision](architecture-decisions/0004-progressive-story-regions.md)
- [Local Mandarin voice selection](architecture-decisions/0005-mandarin-pronunciation-voice-selection.md)
