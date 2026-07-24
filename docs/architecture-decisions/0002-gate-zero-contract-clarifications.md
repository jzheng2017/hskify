# ADR 0002: Gate 0 contract and Firefox clarifications

## Problem

The implementation plan leaves several details undefined that are required for
a secure, testable Gate 0:

- native-host manifests launch an executable path, not an executable plus
  browser-mode arguments;
- a user-triggered, non-persistent content script needs the Firefox `scripting`
  permission in addition to `activeTab`;
- the native response calls the token short-lived but does not communicate its
  expiry;
- several required HTTP endpoints have no response/request/error shapes;
- Koharu's normal router is intentionally suitable for its local UI, but uses
  permissive CORS and a much larger upload limit than browser mode needs.

## Evidence

- Firefox native messaging launches the registered host executable and keeps
  the native message framing pipes attached to that process.
- Static WXT content scripts require matching host access. An unlisted bundle
  injected with `browser.scripting.executeScript()` after a user action
  preserves the plan's no-install-time-`<all_urls>` requirement.
- The pinned Koharu router applies `CorsLayer::very_permissive()` and a 1 GiB
  body limit.
- The pinned Koharu main process initializes optional remote integrations and
  desktop concerns that are not part of browser mode.

## Decision

1. Build two small executables from `browser-companion`:
   `hsk-manga-native-host` and `hsk-manga-browser-daemon`.
2. Package the content script as an unlisted WXT bundle and inject it only
   after a popup action. Add the `scripting` permission; continue to request
   image-CDN access only for the exact origin at runtime.
3. Add `sessionExpiresAtUnixMs` to the native ready response.
4. Define and fixture-test `HealthResponse`, `BrowserJobCreated`,
   `RetranslateRequest`, `LookupRequest`, and a common `ErrorResponse`.
5. Validate all style colors as hexadecimal colors before they reach CSS.
   Model output is never accepted as arbitrary CSS.
6. Expose a dedicated `/browser/v1` router. Do not mount Koharu's `/api/v1`,
   `/mcp`, or UI routes in the browser daemon.
7. Bind literally to `127.0.0.1:0`, authenticate before reading uploads, cap
   native and HTTP bodies, and recompute source SHA-256.

## Consequences

- The native launcher remains small and cannot accidentally initialize the
  desktop UI or telemetry.
- The extension gains no page access until the user acts or grants an exact
  origin permission.
- Protocol v1 clients can proactively refresh expired sessions.
- Browser mode has a narrower attack surface than normal Koharu headless mode.
- These additive fields and types are part of the protocol v1 freeze and are
  covered by both Rust and TypeScript fixtures.
