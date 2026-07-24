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

## Required regression and acceptance work

This check does not claim the extension works end to end on the site. An
independent review found that the initial Firefox implementation requested the
CDN optional permission after the popup gesture had already crossed
asynchronous messaging, which Firefox may reject. It also generated invalid
port-bearing match patterns for development origins. Gate 1 remains rejected
until the permission flow is moved into the direct popup action and tested in
packaged Firefox.

The permanent regression fixture must be synthetic and site-neutral while
preserving the relevant shape: multiple long WebP-like raster pages, query
strings, a separate fixture origin, `data-page-index`, a responsive 720-pixel
maximum, a very tall scroll document, and distracting cover/avatar/comment
images. Real Asura assets must not be copied into the repository.

After those fixes, the manual Firefox check should verify:

1. the popup's direct user action requests only the page and CDN origins;
2. visible mode starts with the intersecting pages and all mode keeps DOM order;
3. cross-origin WebP bytes are streamed and validated before upload;
4. a 900 by 16,000 page is accepted without clipping or layout drift;
5. scrolling and responsive resizing keep translated regions aligned;
6. cancel, navigation, and source replacement never hide or translate a stale
   original;
7. original, Chinese, selection, copy, and direct image-click behavior remain
   usable.
