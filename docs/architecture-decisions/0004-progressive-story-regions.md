# ADR 0004: Progressive story regions

- Status: Accepted
- Updated: 2026-07-26

## Context

The performance product must show useful story text before an entire tall page
finishes. A page-wide clean/translate/render artifact delays first output,
multiplies memory traffic, and makes excluded-content preservation harder to
express.

## Decision

The browser path processes and publishes independent regions:

1. Overlapping detector tiles are reprioritized from the current viewport.
2. Detected text lines enter English recognition. Spatial, color, confidence,
   language, and story-role gates accept dialogue, thoughts, and narration
   without requiring a white balloon.
3. OCR must meet the fixed confidence floor and contain Latin alphabetic text
   only.
4. Sound effects, credits, scanlation promotion, branding, non-English text,
   and ambiguous OCR are rejected before cleanup or translation.
5. Cleanup produces one transparent PNG patch whose opaque pixels are limited
   to the inferred erase/glyph mask. Local image structure, not a fixed color
   list, determines foreground, background, fill, and styling.
6. English is translated directly to HSK-targeted Simplified Chinese in small
   batches; only invalid items may use one bounded targeted repair.
7. The patch blob is stored before `regionReady`, and Firefox installs the
   decoded patch before its selectable Chinese.
8. Completion is a terminal log event, not a separate page result.

## Consequences

- The original image remains the stable page coordinate system.
- Useful visible story text can render while off-screen tiles continue.
- There is no full cleaned image, browser project, page history, or
  retranslation artifact.
- Patch quality is intentionally local. Complex textured backgrounds may be
  less visually complete than page-wide neural inpainting, but pixels outside
  the erase mask must remain exact and colored regions must not be whitened.
- Excluded non-story regions remain untranslated even if they contain readable
  English.
- Contract stage names such as `inpainting` remain wire vocabulary; they do not
  imply a page-wide image stage.

## Evidence

The code and contract tests establish the structural invariants. Chapter-wide
latency, memory, VRAM, recall, OCR, patch quality, and installed-Firefox
evidence are pending under the
[real-reader v2 method](../real-reader-v2.md). The corpus is a regression
sample, not permission for chapter-specific rules.
