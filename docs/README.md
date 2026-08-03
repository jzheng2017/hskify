# Hskify documentation

These documents describe only the direct, performance-only Firefox build in
the current code. The removed multilingual desktop documentation covered a
different product surface: general RPC APIs, projects and history, provider
configuration, broad hardware fallbacks, and page-wide translation workflows.
It must not be used as Hskify documentation.

## Read in this order

- [Architecture](architecture.md): components, data flow, scheduling, caches,
  security, and the RTX 4080 SUPER/CUDA-only boundary.
- [Browser contract](browser-contract.md): exact unversioned routes, flat
  chapter events, strict build fingerprint, and patch-first rendering.
- [Browser companion implementation](../crates/browser-companion/IMPLEMENTATION.md):
  code-level daemon and pipeline behavior.
- [Real-reader v2 corpus and evidence](real-reader-v2.md): the local,
  content-addressed chapter contract and packaged Firefox release gate.
- [Model benchmark](model-benchmark.md): the locked translation model and
  quality-evaluation requirements.
- [External component evaluation](component-evaluation.md): established work
  retained, rejected candidates, and cleanup policy.
- [Firefox manual checklist](firefox-manual-test-checklist.md): packaged
  browser checks that cannot be replaced by unit tests.
- [Maintainer guide](maintainer-guide.md): invariants and documentation update
  rules.
- [Licence inventory](licence-inventory.md): runtime resource and data audit
  status.

## Accepted decisions

- [Historical story-region processing decision](architecture-decisions/0004-progressive-story-regions.md)
- [Local Mandarin voice selection](architecture-decisions/0005-mandarin-pronunciation-voice-selection.md)

Historical gate notes and inherited desktop documentation were removed because
they described a versioned, project-backed, page-result architecture that is
not present in the current browser build.
