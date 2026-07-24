# ADR 0004: Dialogue-only webtoon cleaning

## Problem

The baseline browser pipeline ran independent full-page bubble segmentation,
text segmentation, OCR, and neural inpainting stages. Very tall webtoon pages
made that shape expensive and unreliable. An early Nano Machine chapter 100
run retained only two regions, treated Korean sound effects as dialogue, and
painted Chinese over surviving English. A later neural cleanup removed the
right regions but spent 42 seconds in inpainting and left faint grey marks in
otherwise flat white speech bubbles.

Browser mode needs a simpler invariant: only English text proven to be inside
a speech bubble may enter OCR, translation, or the erase mask, and no other
pixel may change.

## Evidence

- The source page was 800 by 11,470 pixels.
- The corrected joint detector produced 11 English dialogue regions instead of
  two.
- Korean effects, the non-bubble English `CLENCH` effect, and a punctuation-only
  `?` bubble all remained outside the accepted erase mask.
- The baseline neural cleanup stage took about 42 seconds on this page.
- The deterministic cleanup stage completed between the 26- and 30-second job
  polls.
- The complete four-core job finished in 217 seconds with a 4.75 GiB peak
  working set and at least 16.34 GiB of physical RAM free.

## Decision

Keep `koharu_app::pipeline::run` as the only production pipeline driver, but
use a browser-specific sequence of Koharu engines and artifacts:

1. The sliced comic detector emits both text boxes and a distinct-ID
   conservative speech-bubble mask.
2. Geometry is accepted only when its centre lies in a bubble or at least 20
   percent of its area overlaps one.
3. OCR runs only on that reduced geometry.
4. OCR results must contain ASCII letters and must not contain non-ASCII
   alphabetic text.
5. The erase mask contains only the accepted English dialogue boxes and is
   extended into their confirmed bubble IDs.
6. `dialogue-bubble-fill` verifies that every erase pixel belongs to a bubble,
   computes one median background colour per bubble, and fills exactly the
   accepted erase pixels.

The detector channel capacity is one. The daemon admits one cleaning or
retranslation pipeline at a time and bounds inference threads separately from
HTTP/runtime threads.

## Consequences

- Original English dialogue is removed before selectable Chinese is rendered.
- Sound effects, captions outside bubbles, artwork, and punctuation-only
  bubbles remain unchanged by construction.
- Browser cleanup no longer loads a neural inpainting model or buffers several
  tall detector slices.
- Flat speech bubbles are handled predictably. A highly textured or
  transparent bubble may receive a median-colour fill rather than synthesized
  texture; that is an explicit v1 trade-off for bounded resource use and the
  no-non-bubble-mutation guarantee.
- Koharu's general `lama-manga` engine remains available to desktop pipelines.
  This decision adds no parallel browser image stack.
