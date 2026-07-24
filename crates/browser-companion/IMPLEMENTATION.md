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

Retention is deterministic and admission-driven. Each accepted job receives a
monotonic daemon-local sequence. When either retention bound is under pressure,
the daemon evicts the oldest inactive terminal job until the new job fits.
Running jobs are never eviction candidates. An evicted job's clean blob is
removed only after the last retained job reference disappears; this also keeps
the source blob valid when a retranslation replaces its completed source job.
Completed and cancelled jobs therefore remain queryable until later admission
pressure reclaims them, after which their job endpoint (and any now-orphaned
blob endpoint) returns 404.

Clean PNG payloads are deduplicated by SHA-256, MIME type, and exact byte
equality. Identical uploads and equivalent cleaned inputs share one blob
allocation. Capacity can still return a retryable 429 while all reclaimable
space is held by active jobs, but terminal history can no longer leave a
long-lived daemon permanently saturated.

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
ports or manual server configuration. The Unix scripts reject ASCII control
bytes while accepting valid non-ASCII UTF-8 path bytes; the generated JSON
retains the UTF-8 executable path.

## Verification

Run from the repository root:

```text
cargo fmt -p browser-companion -- --check
cargo check -p browser-companion
cargo test -p browser-companion --all-targets
cargo clippy -p browser-companion --all-targets -- -D warnings
sh -n installers/linux/native-host-registration/register.sh \
  installers/linux/native-host-registration/unregister.sh \
  installers/macos/native-host-registration/register.sh \
  installers/macos/native-host-registration/unregister.sh \
  installers/test-native-host-registration.sh
sh installers/test-native-host-registration.sh
```

Coverage includes framing bounds and endianness, native caller validation, an
end-to-end native binary handshake against a deliberately prestarted daemon,
fresh/expired tokens, constant-time authentication use, pre-body rejection,
Host/origin/protocol/CORS restrictions, every required route, SHA-256/MIME/
byte/pixel limits, progress/result/blob/font transfer, cancellation races,
active saturation, oldest-terminal eviction, exact-byte SHA deduplication,
completion/cancellation orphan cleanup, shared/retranslation blob references,
duplicate locks, stale state replacement, random IPv4 loopback binding, and
idle cleanup. The prestarted-daemon handshake test covers framing, caller
validation, and authenticated session issuance; it does not exercise launcher
spawn.

### Evidence captured 2026-07-24

On the Windows Codex development harness:

- `cargo fmt -p browser-companion -- --check`, `cargo check -p
  browser-companion`, and `cargo clippy -p browser-companion --all-targets -- -D
  warnings` passed.
- `cargo test -p browser-companion --all-targets` passed 31 library tests, 7
  contract-fixture tests, the prestarted-daemon native handshake, and the
  explicitly in-job Windows lifecycle test. The production breakaway probe was
  reported as ignored.
- Git Bash `sh -n` passed for both Unix register/unregister pairs and the
  registration regression script. `sh
  installers/test-native-host-registration.sh` passed both Linux and macOS
  layouts using `翻訳ツール/hsk-manga-native-host`, and verified that a newline
  in the path is rejected specifically as a control character. Its temporary
  `HOME` prevents writes to the real Firefox profile.
- PowerShell's parser reported zero syntax errors for
  `Register-NativeHost.ps1` and `Unregister-NativeHost.ps1`.
- The explicitly requested production probe,
  `cargo test -p browser-companion --test windows_lifecycle
  production_breakaway_launch_probe_covers_detached_lifecycle -- --ignored
  --exact --nocapture`, failed immediately with Win32 error 5 (`Access is
  denied`). This outer job does not grant `JOB_OBJECT_LIMIT_BREAKAWAY_OK`; the
  test no longer falls back to an in-job child or reports detached coverage.

### Remaining platform smoke requirements

- **Windows / real Firefox:** install the packaged manifest and signed binaries,
  invoke the native host from regular Firefox, close the one-shot native host,
  and verify the production daemon remains alive, serves the returned token,
  rejects a duplicate daemon, and performs idle cleanup. This is the required
  evidence that `CREATE_BREAKAWAY_FROM_JOB` works specifically from Firefox's
  native-host job; the Codex outer job and the direct ignored probe cannot
  substitute for it.
- **Linux / real Firefox:** run the registration regression plus an actual
  per-user install from a non-ASCII UTF-8 path, invoke it from Firefox, verify
  manifest permissions and origin/ID enforcement, and confirm the `setsid()`
  daemon survives native-host exit before idle cleanup.
- **macOS / real Firefox:** repeat the non-ASCII registration and Firefox
  native-message launch against the packaged, signed/notarized binaries; verify
  the manifest under `~/Library/Application
  Support/Mozilla/NativeMessagingHosts`, executable/quarantine behavior,
  `setsid()` survival, and idle cleanup.
