# Hskify documentation

This directory contains two intentionally separate documentation sets:

1. Hskify documentation for the Firefox reading companion, its local native
   service, HSK learning behavior, packaging, and maintenance.
2. Inherited Koharu documentation retained for the shared manga pipeline and
   desktop application layers.

Use the Hskify documents first. A statement in inherited Koharu documentation
does not automatically describe Hskify's browser permissions, packaging,
security boundary, supported platforms, or release status.

## Hskify design and operation

- [Architecture overview](architecture.md) — system boundaries, data flow,
  security, and resource constraints.
- [Maintainer guide](maintainer-guide.md) — upstream synchronization,
  verification, and the fork-versus-package/service decision.
- [Firefox manual test checklist](firefox-manual-test-checklist.md) — release
  checks that require real Firefox and platform integration.
- [Model benchmark](model-benchmark.md) — model evaluation criteria and
  results.
- [Licence inventory](licence-inventory.md) — code, models, fonts, datasets,
  and redistribution status.

Implementation details live close to the component they describe:

- [`browser-companion` implementation](../crates/browser-companion/IMPLEMENTATION.md)
- [`hsk-control` overview](../crates/hsk-control/README.md)
- [`hsk-control` implementation](../crates/hsk-control/IMPLEMENTATION.md)
- [Firefox extension implementation](../extensions/firefox/IMPLEMENTATION.md)
- [Windows developer package](../installers/windows/README.md)

## Architecture decisions

- [ADR 0001: Koharu upstream pin](architecture-decisions/0001-koharu-upstream-pin.md)
- [ADR 0002: Gate 0 contract and Firefox clarifications](architecture-decisions/0002-gate-zero-contract-clarifications.md)
- [ADR 0003: Koharu extraction map](architecture-decisions/0003-koharu-extraction-map.md)
- [ADR 0004: Dialogue-only webtoon cleaning](architecture-decisions/0004-dialogue-only-webtoon-cleaning.md)
- [ADR 0005: Mandarin pronunciation and voice selection](architecture-decisions/0005-mandarin-pronunciation-voice-selection.md)

## Implementation and compatibility notes

- [Gate 0 evidence](implementation-notes/gate-0.md)
- [Asura Comics structural compatibility](implementation-notes/asura-compatibility.md)

These notes are point-in-time evidence, not evergreen support promises. Check
their dates and the manual test checklist before relying on them for a release.

## Inherited Koharu documentation

The multilingual trees below document upstream Koharu concepts and workflows:

- [English (United States)](en-US/index.md)
- [Japanese](ja-JP/index.md)
- [Portuguese (Brazil)](pt-BR/index.md)
- [Simplified Chinese](zh-CN/index.md)

They are useful when working on the inherited `crates/koharu-*` layers,
especially the pipeline, models, rendering, API, and desktop build. Preserve
Koharu's name and upstream links inside those trees unless Hskify has actually
forked and verified the specific page.
