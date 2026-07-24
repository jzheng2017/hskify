# Firefox extension manual test checklist

Last updated: 2026-07-24

This checklist separates completed automated evidence from checks that require
an installed Firefox extension, an interactive permission prompt, or the
installed native companion. A pending row is not release evidence.

## Automated evidence completed

| Check | Status | Evidence |
| --- | --- | --- |
| Strict TypeScript/WXT typecheck | Passed | `npm run typecheck` |
| Firefox MV3 production build | Passed | `npm run build` |
| Production fixture isolation | Passed | No `data-hmt-fixture`, `hmtFixture`, fixture service, fixture response, or `structuredClone` test marker in `.output/firefox-mv3` |
| Unit/component suite | Passed | 74 Vitest tests across 18 files |
| Firefox renderer harness | Passed | 6 Playwright tests |
| Long raster decode | Passed | Firefox decoded the query-string 900×16,000 WebP |
| Long-reader discovery | Passed | Exactly 20 chapter images selected among 154 images; cover/comments/avatars excluded |
| Direct and ancestor click delivery | Passed | Original image listener once and reader ancestor once; selected text suppresses navigation |
| Corrupt/stale renderer safety | Passed | Decode failure and an awaited stale font path leave the original unchanged |
| Clean/font response streaming caps | Passed | Unit streams stop over the configured limit before full materialization |
| Cached native-session health and retry | Passed | Unit coverage for `/health`, transport refresh, and repeated failure invalidation |
| Extension lint | Passed with known warnings | 0 errors, 0 notices, 2 Firefox minimum-version/data-collection warnings |
| Bounded packaged startup smoke | Passed | `web-ext run` pre-installed `.output/firefox-mv3` in a temporary headless profile and exited successfully after an automatic screenshot |

The Playwright suite is a regular-page Firefox harness. It proves browser DOM,
selection, decode, layout, and click behaviour, but it does not install the
extension or exercise native messaging. The bounded `web-ext run` smoke proves
that the built manifest starts; it does not prove the popup/native workflow.

## Interactive packaged-extension checks

### Exact optional permission prompt

Status: **Pending manual run**

1. Start two local origins with the reader page and image host on different
   hostnames.
2. Open the popup and choose **Translate visible manga**.
3. Verify Firefox asks only for the exact portless image-host pattern shown by
   the page, not `<all_urls>` and not an install-time permission.
4. Deny once and verify every original remains visible with one clear retry
   path.
5. Grant on retry and verify translation starts without a second prompt.
6. Repeat with **Translate all manga** and verify only the all-image host union
   is requested.

### Redirect to a new hostname

Status: **Pending manual run**

1. Serve an image URL that redirects to a different hostname.
2. Verify the background does not fetch that hop without a pre-granted exact
   host permission and never attempts an asynchronous permission prompt.
3. Reopen the popup, verify the remembered redirect hostname is included in
   the exact permission plan, grant it through the next translate click, and
   verify acquisition succeeds.
4. Repeat with an authenticated response and verify a credentialed redirect to
   another origin is rejected.

### Real Firefox runtime-message binary ceiling

Status: **Pending; Gate 0 binary-transfer evidence is not complete**

1. Use the packaged extension and a local companion that returns valid,
   browser-decodable clean images at representative 5, 10, 15, and 20 MiB
   payload sizes.
2. For each size, compare byte length and SHA-256 at the companion, background,
   and content boundaries.
3. Verify the image decodes, dimensions match, the page remains responsive,
   and cancellation/navigation during transfer leaves the original visible.
4. Repeat several times while recording Firefox peak memory and any message
   failure threshold.
5. If any representative size is unreliable, implement and retest a bounded
   fixed-size chunk protocol.

Do not substitute a same-realm or unit-test `structuredClone()` call for this
probe. It does not measure Firefox extension messaging.

## Installed native-companion checks

### Launcher and daemon security

Status: **Blocked on installed Workstream B companion**

- Native-host registration accepts only the permanent Firefox extension ID.
- One-shot launch returns a random loopback port and short-lived bearer token.
- The daemon binds only `127.0.0.1`.
- Missing/invalid bearer, extension origin, and protocol headers are rejected.
- Duplicate launchers discover one daemon.
- Manga bytes never leave loopback.

### Suspension, restart, and reconnect

Status: **Blocked on installed Workstream B companion**

- Close the popup while a job runs; processing continues.
- Allow the MV3 background to suspend; the next content poll reconstructs
  state from small metadata and validates the cached endpoint through
  `/health`.
- Stop the daemon between polls; one fresh native handshake and retry succeeds.
- Reject the token twice; the session is invalidated and the error remains
  visible/retryable.
- Restart Firefox and verify the expected fresh launcher handshake.

### Live cancellation and navigation

Status: **Blocked on installed Workstream B companion**

- Cancel during upload, processing, polling, clean-image transfer, font load,
  and immediately before render commit.
- Replace `currentSrc`, remove the image, use `history.pushState`, change the
  query/hash route, perform a full navigation, and close the tab during a job.
- Verify each stale job is cancelled or discarded, counters remain accurate,
  Blob URLs are revoked, and every incomplete original remains unchanged.
- Retry a failed image and verify visibility/mutation events alone never
  retrigger it.

### Corrupt companion artifacts in the installed extension

Status: **Blocked on installed Workstream B companion**

- Return a bad clean-image signature, MIME mismatch, truncated stream, wrong
  decoded dimensions, oversized body with and without `Content-Length`, and a
  malformed font.
- Verify the original is not wrapped or hidden, no partial Chinese is shown,
  the response is bounded, and the user receives one retry action.

## Later-gate visual and model checks

Status: **Not part of Firefox Gate 1 acceptance**

- Real redistributable CJK font bank and font-category quality.
- Golden-set polygon fitting with no unreported clipping.
- OCR, masks, inpainting, translation, strict HSK vocabulary, names, numbers,
  and negation.
- OCR/inpainting cache reuse after HSK-level changes.
- Final fully local network inspection with the production model pack.
