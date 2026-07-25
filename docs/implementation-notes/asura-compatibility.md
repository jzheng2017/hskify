# Asura Comics compatibility note

Date: 2026-07-24

This is a transient structural compatibility check requested by the user. No
chapter artwork, screenshots, response bodies, or derived text are committed.
Only public page structure and response metadata are recorded.

## Observed reader shape

`https://asuracomic.net/` redirected to `https://asurascans.com/`. A current
public chapter reader exposed 20 webtoon pages as ordinary
`<img data-page-index>` elements. The page images:

- came from the separate `https://cdn.asurascans.com` origin;
- used WebP URLs with query strings;
- were not wrapped in `<picture>` elements or navigation links;
- had intrinsic widths of 900 or 1,200 pixels;
- reached 16,000 pixels high and 14.4 million decoded pixels;
- rendered at a 720-pixel desktop maximum and resized proportionally to 465
  pixels in a 480-pixel viewport;
- appeared in one approximately 156,000-CSS-pixel vertical document.

Three representative CDN responses declared `image/webp`, returned HTTP 200,
and were 174,580, 240,076, and 925,978 bytes. Those samples and the maximum
observed decoded size fit both extension and companion input bounds.

## Discovery result

The extension's current conservative discovery predicate was evaluated
read-only against the live DOM. Among 154 total images, it accepted exactly the
20 chapter pages and rejected covers, avatars, comment media, and controls.
At the top of the chapter, the first two long images intersected the viewport.
After scrolling 13,500 CSS pixels, only page index 2 intersected, which matches
the visible-first queue model.

## Implemented compatibility work

The popup now requests the exact, portless page/CDN origins directly from the
translate click. Background acquisition only checks an already granted
permission, and a newly observed redirect host is deferred to the next popup
click. The permanent site-neutral regression fixture exists at
`fixtures/browser-pages/webtoon.html`; it preserves long query-string WebP
pages, a separate image origin, `data-page-index`, responsive sizing, a very
tall document, and distracting cover/avatar/comment images.

Automated discovery still accepts exactly the 20 fixture chapter pages among
154 images. Firefox also decodes the 900 by 16,000 long-page fixture at its
production dimensions and keeps renderer geometry within two CSS pixels after
resize. The interactive packaged-Firefox permission prompt remains a manual
release check.

## Nano Machine chapter 100 acceptance

The current Nano Machine chapter 100 reader exposed 10 WebP page images. Their
intrinsic sizes were:

```text
800x11470  800x10865  800x10445  800x10780  800x11655
800x11925  800x11005  800x11830  800x11270  800x3477
```

The first 800 by 11,470 page was submitted to the release browser daemon with
four inference threads and one pipeline job. It completed in 217 seconds:
detection reached OCR at 22 seconds, deterministic dialogue cleanup finished
at 30 seconds, and local translation plus bounded HSK correction consumed the
remaining time. Peak daemon working set was 4.75 GiB, peak private memory was
7.89 GiB, and at least 16.34 GiB of physical RAM remained free.

The detector/OCR gate produced 11 English dialogue regions. Visual inspection
of the actual clean WebP confirmed:

- every accepted English speech-bubble region was emptied before rendering;
- Korean sound effects were byte-for-byte outside the erase mask and remained
  visible;
- the English `CLENCH` sound effect outside a bubble remained visible;
- the punctuation-only `?` bubble remained untouched; and
- the production Firefox renderer placed 11 selectable Chinese DOM regions
  with no degraded fit.

After rebuilding and installing the final package, a second acceptance used a
fresh disposable regular-Firefox profile and the extension's real toolbar
popup. The trusted popup click injected the page runtime, launched the
registered native host and installed daemon, reused the cached Koharu cleaning
artifacts, ran local translation plus HSK 5 correction, and rendered the same
11 selectable regions in 186.7 seconds. The final DOM had zero degraded fits,
the original image opacity changed to zero only after completion, and the page
HUD reported `1 of 1 images translated`. This run did not touch the user's
normal Firefox profile.

The renderer was subsequently tightened after human review found that using
the complete bubble hull made several Chinese overlays visually oversized.
Replaying the actual page now falls back to each original OCR text polygon when
the supplied safe polygon is identical to the bubble hull, reserves an inner
margin, and permits proportional shrinking below the old fixed floor. All 11
regions measured zero DOM overflow and zero degraded fits; the minimum inset
inside the conservative text regions was 6.9 CSS pixels, and the crowded long
bubbles dropped from roughly 50–52 pixel fonts to 37–39 pixels.

Acceptance images and chapter bytes remain in the ignored local `.cache`
directory; no chapter artwork or derived translation is committed.

The remaining installed-Firefox check should verify:

1. the popup's direct user action requests only the page and CDN origins;
2. visible mode starts with the intersecting pages and all mode keeps DOM order;
3. cross-origin WebP bytes are streamed and validated before upload;
4. a 900 by 16,000 page is accepted without clipping or layout drift;
5. scrolling and responsive resizing keep translated regions aligned;
6. cancel, navigation, and source replacement never hide or translate a stale
   original;
7. original, Chinese, selection, copy, and direct image-click behavior remain
   usable.
