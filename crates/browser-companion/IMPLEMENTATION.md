# Browser companion: secure Koharu-backed Gate 3

Status: Workstream B implementation through Gate 3 (real local detection,
bubble/text masks, English OCR, and inpainting; no translation or HSK rewrite).

## Delivered shape

The crate builds two executables:

- `hsk-manga-native-host`: validates Firefox's manifest-path and permanent
  add-on-ID arguments, reads one bounded little-endian native frame, discovers
  or launches the daemon, asks it for a fresh browser session, writes one
  bounded response frame, and exits.
- `hsk-manga-browser-daemon`: holds a per-user exclusive lock, binds literally
  to `127.0.0.1:0`, publishes a control-secret-protected discovery record,
  serves only the browser router, and exits after an idle period with no active
  jobs or admitted authenticated requests.

Windows detached creation requests `CREATE_BREAKAWAY_FROM_JOB`,
`CREATE_NEW_PROCESS_GROUP`, and `CREATE_NO_WINDOW`. Unix creation calls
`setsid()` in the child before exec. Standard streams are detached from the
native host in both cases.

The per-user state root contains:

```text
daemon-v1.lock
daemon-state-v1.json
browser-cache-v1/
  koharu-data/
  cleaning-projects-v1/<source-sha256>.khrproj/
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

## Koharu cleaning adapter and limits

`POST /browser/v1/jobs` now retains the validated source under the existing
daemon bounds and runs the production `koharu_app::pipeline::run` path. The
Gate-3 spec uses Koharu's configured detector, text segmenter, bubble
segmenter, English OCR, and manga inpainter. It deliberately omits the
translator, HSK rewrite, and renderer. Retranslation therefore returns the
explicit `TRANSLATION_NOT_AVAILABLE` error until Gate 4.

Each source hash owns a persistent Koharu project. Successful runs compact the
project and write a versioned pipeline marker; a later identical upload can
package the cached detector/OCR/mask/inpaint artifacts without initializing
the ML runtime. The browser result contains the real Koharu inpainted PNG/WebP,
OCR text and confidence, normalized text geometry, and bubble/safe polygons
derived from Koharu's distinct-ID bubble mask. Region IDs hash the source and
normalized text geometry, so cache hits preserve IDs. OCR confidence below
0.60 emits a region-scoped `LOW_OCR_CONFIDENCE` warning.

The frozen protocol requires translation-shaped fields for dialogue regions.
At Gate 3 those fields contain the source English as an explicit identity
fallback, pinyin is empty, vocabulary is non-strict, and
`translationHit=false`; no translation or HSK claim is made. Jobs remain
daemon-owned while Firefox backgrounds suspend, progress monotonically, can be
polled with a refreshed session, and share the same cancellation atomic used
by Koharu so a worker cannot revive a cancelled state.

Default limits are:

- 20 MiB image field;
- 64 KiB metadata field;
- 21 MiB complete multipart body;
- 25,000,000 decoded pixels;
- 16,384 pixels on either dimension;
- 128 MiB decoder allocation budget;
- 64 MiB clean blob;
- 128 retained jobs and 256 MiB retained source/clean blobs;
- four concurrent authenticated requests, including response-body transfer.

Multipart fields are streamed against their individual limits. The daemon
recomputes SHA-256, matches declared/multipart/sniffed MIME types, checks
declared dimensions, decodes under resource limits, and compares decoded
dimensions before retaining bytes.

Authentication and protocol checks run before a request can acquire one of the
four global permits or poll its body. Saturation returns retryable HTTP 429 with
`REQUEST_CAPACITY_EXHAUSTED`. A permit remains owned through multipart
consumption and safe image decode even if the HTTP task is cancelled,
and complete response-body transfer. Idle shutdown uses the same synchronized
admission state, latches only when both admitted requests and active jobs are
zero, and refuses new admission after latching.

Koharu output is rejected before retention if it exceeds the configured clean
blob cap. Blob GET bodies retain the stored `Arc<[u8]>` through
`Bytes::from_owner`, avoiding a per-request copy of up to 64 MiB; the response
holds its global permit until the body is consumed or dropped. Tall narrow
reader images remain accepted when they fit the byte, dimension, pixel,
decoder-allocation, and output limits.

Retention is deterministic and admission-driven. Each accepted job receives a
monotonic daemon-local sequence. When either retention bound is under pressure,
the daemon evicts the oldest inactive terminal job until the new job fits.
Running jobs are never eviction candidates. An evicted job's clean blob is
removed only after the last retained job reference disappears. Completed and
cancelled jobs therefore remain queryable until later admission pressure
reclaims them, after which their job endpoint (and any now-orphaned blob
endpoint) returns 404.

Clean PNG/WebP payloads are deduplicated by SHA-256, MIME type, and exact byte
equality. Identical uploads and equivalent cleaned inputs share one blob
allocation. Capacity can still return a retryable 429 while all reclaimable
space is held by active jobs, but terminal history can no longer leave a
long-lived daemon permanently saturated.

The font endpoint returns a valid, project-generated Gate-2 TrueType fixture
containing only `.notdef` and space. Browsers therefore load it successfully
and use normal CJK fallback. It intentionally is not the licensed production
CJK font bank planned for Gate 6.

## Koharu reuse boundary

Gate 3 reuses `ProjectSession` and its content-addressed `BlobStore`,
`RuntimeManager`, Koharu's engine registry, and
`koharu_app::pipeline::run`. The browser layer only adapts progress,
cancellation, scene artifacts, stable protocol geometry, warnings, and bounded
blob retention. It does not introduce a second detector/OCR/inpaint engine.

Health/setup, the lightweight lookup scaffold, and the fallback font remain
the frozen protocol fixtures from Gate 2. Translation/HSK work begins at Gate
4, while production font/style integration remains Gate 6.

## Registration assets

`installers/{windows,linux,macos}/native-host-registration` contains exact-ID
manifest templates plus per-user register/unregister scripts. They accept only
an absolute regular/leaf executable with the frozen native-host filename and do
not expose ports or manual server configuration. The Unix scripts reject ASCII
control bytes while accepting valid non-ASCII UTF-8 path bytes; the generated
JSON retains the UTF-8 executable path.

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
powershell -File installers/test-native-host-registration.ps1
```

Coverage includes framing bounds and endianness, native caller validation, an
end-to-end native binary handshake against a deliberately prestarted daemon,
fresh/expired tokens, constant-time authentication use, pre-body rejection,
Host/origin/protocol/CORS restrictions, every required route, SHA-256/MIME/
byte/pixel limits, progress/result/blob/font transfer, cancellation races,
authenticated high-concurrency saturation before body polling, stalled-request
idle exclusion, clean-output bounds, zero-copy/budgeted blob transfer, active
saturation, oldest-terminal eviction, exact-byte SHA deduplication,
completion/cancellation orphan cleanup, shared blob references,
duplicate locks, stale state replacement, random IPv4 loopback binding, and
idle cleanup. A 256 by 4096 synthetic webtoon scene verifies stable IDs,
normalized OCR geometry, instance-mask-derived bubble/safe polygons,
low-confidence warnings, identity translation fields, cache flags, cleaned
image dimensions, and whole-result protocol validation. The prestarted-daemon
handshake test covers framing, caller validation, and authenticated session
issuance; it does not exercise launcher spawn or download production models.

### Evidence captured 2026-07-24

On the Windows Codex development harness:

- `cargo fmt -p browser-companion -- --check`, `cargo check -p
  browser-companion --lib`, and `cargo clippy -p browser-companion
  --all-targets -- -D warnings` passed with the Visual Studio 2019 developer
  environment, the repository's verified LLVM 22.1.0 cache, and AWS-LC's
  checked-in prebuilt NASM objects.
- `cargo test -p browser-companion --all-targets` passed 37 library tests, 7
  contract-fixture tests, the prestarted-daemon native handshake, and the
  explicitly non-breakaway Windows lifecycle test. The production breakaway
  probe was reported as ignored.
- Git Bash `sh -n` passed for both Unix register/unregister pairs and the
  registration regression script. `sh
  installers/test-native-host-registration.sh` passed both Linux and macOS
  layouts using `翻訳ツール/hsk-manga-native-host`, and verified that a newline
  in the path is rejected specifically as a control character, directories are
  rejected, and unregister removes the isolated manifest. Its temporary `HOME`
  prevents writes to the real Firefox profile.
- PowerShell's parser reported zero syntax errors for both Windows registration
  scripts and their regression script.
  `installers/test-native-host-registration.ps1` passed isolated
  register/unregister and directory-rejection checks.
- The explicitly requested production probe,
  `cargo test -p browser-companion --test windows_lifecycle
  production_breakaway_launch_probe_covers_detached_lifecycle -- --ignored
  --exact --nocapture`, failed immediately with Win32 error 5 (`Access is
  denied`) after verifying that the parent process is in a Windows job. This
  outer job does not grant `JOB_OBJECT_LIMIT_BREAKAWAY_OK`; the test no longer
  falls back to an in-job child or reports detached coverage. On a permitting
  parent it additionally requires `IsProcessInJob` to report that the spawned
  child escaped every Windows job.

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
