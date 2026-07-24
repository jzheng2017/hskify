# Browser companion: secure fixture-backed Gate 2

Status: Workstream B implementation through Gate 2 (fixture adapter; no live ML
pipeline).

## Delivered shape

The crate builds two executables:

- `hsk-manga-native-host`: validates Firefox's manifest-path and permanent
  add-on-ID arguments, reads one bounded little-endian native frame, discovers
  or launches the daemon, asks it for a fresh browser session, writes one
  bounded response frame, and exits.
- `hsk-manga-browser-daemon`: holds a per-user exclusive lock, binds literally
  to `127.0.0.1:0`, publishes a control-secret-protected discovery record,
  serves only the browser router, and exits after an idle period with no active
  jobs.

Windows detached creation requests `CREATE_BREAKAWAY_FROM_JOB`,
`CREATE_NEW_PROCESS_GROUP`, and `CREATE_NO_WINDOW`. Unix creation calls
`setsid()` in the child before exec. Standard streams are detached from the
native host in both cases.

The per-user state root contains:

```text
daemon-v1.lock
daemon-state-v1.json
browser-cache-v1/
```

On Unix, the directories are mode 0700 and the state/lock files are mode 0600.
On Windows they inherit the current user's profile ACL. The state record has a
version, instance UUID, PID, random loopback port, start timestamp, and an
independent 256-bit control secret. Stale records are replaced only while
holding the daemon lock, and shutdown removes a record only when its instance
UUID still matches.

## Security boundary

- The native host accepts only the manifest whose name is
  `local.mangalations.hsk_manga`, whose executable resolves to the running
  binary, and whose sole allowed add-on is
  `hsk-manga-translator@local.mangalations`.
- Every handshake issues a newly generated, unpadded base64url 256-bit token
  with an explicit expiration. Browser tokens are bound to the exact canonical
  `moz-extension://` origin from that handshake.
- The internal session endpoint uses the separate discovery control secret.
  It is not CORS-enabled and is never returned to Firefox.
- Browser middleware verifies the exact `127.0.0.1:<bound-port>` Host, an
  active extension origin, protocol header `1`, and bearer token before a
  handler or body extractor runs. Secret comparison uses constant-time byte
  equality.
- CORS permits only an active extension origin, methods GET/POST/DELETE, and
  headers Authorization, Content-Type, and X-HSK-Manga-Protocol. No wildcard
  origins, credentials, or permissive fallback are present.
- `/api/v1`, `/mcp`, UI assets, and all non-browser namespaces return 404 and
  are not mounted.
- There is no telemetry, remote fetch, cloud credential, URL-fetch endpoint,
  or manga egress path.

## Fixture adapter and limits

All required `/browser/v1` endpoints are present. Health, setup, job result,
progress, lookup, and retranslation consume the frozen protocol-v1 fixtures.
Uploaded bytes are used as the clean PNG blob after full safe decoding; other
supported raster inputs are decoded and re-encoded as PNG. Fixture jobs remain
daemon-owned while Firefox backgrounds suspend, progress monotonically, can be
polled with a refreshed session, and use synchronized cancellation so a worker
cannot revive a cancelled state.

Default limits are:

- 20 MiB image field;
- 64 KiB metadata field;
- 21 MiB complete multipart body;
- 25,000,000 decoded pixels;
- 16,384 pixels on either dimension;
- 128 MiB decoder allocation budget;
- 64 MiB clean blob;
- 128 retained jobs and 256 MiB retained fixture blobs.

Multipart fields are streamed against their individual limits. The daemon
recomputes SHA-256, matches declared/multipart/sniffed MIME types, checks
declared dimensions, decodes under resource limits, and compares decoded
dimensions before retaining bytes.

The font endpoint returns a valid, project-generated Gate-2 TrueType fixture
containing only `.notdef` and space. Browsers therefore load it successfully
and use normal CJK fallback. It intentionally is not the licensed production
CJK font bank planned for Gate 6.

## Koharu reuse boundary

The existing fork was inspected before writing the adapter:

- `koharu-app::BlobStore` is the content-addressed persistent blob layer;
- `koharu-core::events` contains pipeline/download status snapshots;
- `koharu-app::pipeline` owns cancellable detector/OCR/inpaint/LLM/render
  execution;
- `koharu-app::GoogleFontService` and the renderer own installed font bytes and
  layout.

Gate 2 intentionally keeps only bounded in-memory fixture jobs/blobs. Gate 3
must replace that adapter with the existing Koharu session, blob, pipeline, and
event layers rather than make this fixture executor persistent or add a second
job engine.

## Registration assets

`installers/{windows,linux,macos}/native-host-registration` contains exact-ID
manifest templates plus per-user register/unregister scripts. They accept only
an absolute executable with the frozen native-host filename and do not expose
ports or manual server configuration.

## Verification

Run from the repository root:

```text
cargo fmt -p browser-companion -- --check
cargo test -p browser-companion --all-targets
cargo clippy -p browser-companion --all-targets -- -D warnings
```

Coverage includes framing bounds and endianness, native caller validation, an
end-to-end native binary handshake against the daemon binary, fresh/expired
tokens, constant-time authentication use, pre-body rejection, Host/origin/
protocol/CORS restrictions, every required route, SHA-256/MIME/byte/pixel
limits, progress/result/blob/font transfer, cancellation races, duplicate
locks, stale state replacement, random IPv4 loopback binding, and idle cleanup.

On the Windows development harness, direct use of
`CREATE_BREAKAWAY_FROM_JOB` returned Win32 error 5 because the harness's outer
job does not grant `JOB_OBJECT_LIMIT_BREAKAWAY_OK`. The test verifies the exact
production flag and then exercises discovery, control auth, duplicate
prevention, and idle cleanup in the harness job. A regular-Firefox packaging
smoke test remains required to verify actual breakaway from Firefox's native
host job. Unix `setsid()` and the macOS/Linux registration scripts were not
runtime-tested on this Windows machine.
