# ADR 0001: Koharu upstream pin

## Problem

The browser companion must extend Koharu's existing local manga pipeline without
duplicating its detector, OCR, inpainting, renderer, local-LLM runtime, progress
registry, or blob store. A reproducible upstream revision is required before
browser adapters can be written.

## Evidence

- Upstream repository: <https://github.com/mayocream/koharu>
- Licence: `GPL-3.0-only`
- Pinned revision: `2107843f0c7e2458de5a329980c78575401babb5`
- Upstream revision date: 2026-07-11
- Upstream version at the pin: `0.61.2`
- The pinned workspace exposes `koharu-app`, `koharu-core`, `koharu-llm`,
  `koharu-ml`, `koharu-renderer`, `koharu-rpc`, and `koharu-runtime`.
- The existing HTTP surface already supports multipart page import, pipeline
  execution, operation cancellation and polling, blob reads, font discovery,
  model downloads, and SSE progress.

## Decision

This repository is a pinned fork of the revision above. The remote named
`upstream` points to the authoritative Koharu repository. Browser-specific
contracts and adapters are isolated under `crates/browser-companion` and
`extensions/firefox`.

The permanent Firefox extension ID and native host name are:

```text
hsk-manga-translator@local.mangalations
local.mangalations.hsk_manga
```

Protocol version 1 is frozen by the fixtures in `fixtures/contracts`.

## Consequences

- Upstream changes are merged deliberately after reviewing contract and
  pipeline changes.
- Browser mode reuses Koharu's public application and pipeline layers.
- Any required change to a shared Koharu type is made at the nearest shared
  layer and covered by a regression test.
- Model and dataset redistribution still requires a separate item-by-item
  licence audit.
