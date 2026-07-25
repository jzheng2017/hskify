# Browser companion: secure Koharu-backed production pipeline

Status: production browser companion with real local speech-bubble/text
detection, English OCR, dialogue-only cleanup, full-page faithful translation,
HSK rewrite/validation, packaged fonts, and dictionary lookup.

## Delivered shape

The crate builds two executables:

- `hsk-manga-native-host`: validates Firefox's manifest-path and permanent
  add-on-ID arguments, reads one bounded little-endian native frame, discovers
  or launches the daemon, asks it for a fresh browser session, writes one
  bounded response frame, and exits.
- `hsk-manga-browser-daemon`: holds a per-user exclusive lock, binds literally
  to `127.0.0.1:0`, publishes a control-secret-protected discovery record,
  serves only the browser router, and exits after two idle minutes with no
  active jobs or admitted authenticated requests. Its Tokio runtime uses two
  workers and four blocking threads. On Windows it runs below normal priority
  and defaults to half the available CPUs with a six-core cap;
  `KOHARU_INFERENCE_THREADS` provides an explicit override.

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
  `local.hskify.hsk_manga`, whose executable resolves to the running
  binary, and whose sole allowed add-on is
  `hsk-manga-translator@local.hskify`.
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
  headers Authorization, Content-Type, X-HSK-Manga-Extension-Origin, and
  X-HSK-Manga-Protocol. The explicit origin header covers privileged extension
  fetches that omit the standard `Origin` header. No wildcard origins,
  credentials, or permissive fallback are present.
- `/api/v1`, `/mcp`, UI assets, and all non-browser namespaces return 404 and
  are not mounted.
- There is no telemetry, remote fetch, cloud credential, URL-fetch endpoint,
  or manga egress path.

## Koharu cleaning and translation adapter

`POST /browser/v1/jobs` now retains the validated source under the existing
daemon bounds and runs the production `koharu_app::pipeline::run` path. The
cleaning spec runs Koharu's sliced joint comic text/bubble detector first,
rejects text without meaningful speech-bubble overlap, OCRs only that reduced
geometry, and then rejects OCR that is not English. A distinct-ID bubble mask
and accepted dialogue boxes form the erase mask. Koharu's
`dialogue-bubble-fill` engine fills only those pixels with the median
background colour of their own bubble; pixels outside the accepted erase mask,
including sound effects, are never changed. The cleaned page then runs one
full-page faithful English-to-Simplified-Chinese request through the local
Qwen model, followed by an HSK-targeted rewrite through that same loaded model.

Each source hash owns a persistent Koharu project. Successful runs compact the
project and write a versioned pipeline marker; a later identical upload can
package the cached detector/OCR/mask/inpaint artifacts without initializing
the ML runtime. The browser result contains the real Koharu inpainted PNG/WebP,
OCR text and confidence, normalized text geometry, and bubble/safe polygons
derived from Koharu's distinct-ID bubble mask. Region IDs hash the source and
normalized text geometry, so cache hits preserve IDs. OCR confidence below
0.60 emits a region-scoped `LOW_OCR_CONFIDENCE` warning.

Every HSK candidate is checked by `hsk-control`. Only invalid regions are sent
back with exact vocabulary and preservation feedback, and the loop stops after
the initial rewrite plus at most two corrections. Numbers, explicit protected
names, and negation markers are preservation requirements. Results carry the
faithful Chinese, normalized displayed Chinese, pinyin, strict-vocabulary
status, explicit proper-name exceptions, and a visible
`HSK_REWRITE_FAILED` warning when the bounded loop cannot produce a strict
candidate.

`POST /browser/v1/jobs/{jobId}/retranslate` retains the clean blob and
detection/OCR/inpaint cache entries. A changed HSK level reruns only the HSK
rewrite and validator; an identical level and dialogue context is a translation
cache hit. `POST /browser/v1/lookup` uses the same complete HSK and dictionary
resources for longest-match pinyin, definitions, HSK overlay, and optional
region context.

Jobs remain daemon-owned while Firefox backgrounds suspend, progress
monotonically, can be polled with a refreshed session, and share the same
cancellation atomic used by Koharu so a worker cannot revive a cancelled
state.

Translation resources are loaded lazily from
`%LOCALAPPDATA%\Hskify\HSKMangaTranslator\resources`:

```text
hsk-2.0.normalized.json
cc-cedict.normalized.json
models/Qwen3.5-4B-Q4_K_M.gguf
```

`HSK_MANGA_RESOURCES_DIR`, `HSK_MANGA_HSK_PATH`,
`HSK_MANGA_DICTIONARY_PATH`, and `HSK_MANGA_QWEN_MODEL_PATH` provide explicit
local path overrides. Handshake, health, and setup status report missing
resources until all three files are present.

## Limits

Default limits are:

- 20 MiB image field;
- 64 KiB metadata field;
- 21 MiB complete multipart body;
- 25,000,000 decoded pixels;
- 16,384 pixels on either dimension;
- 128 MiB decoder allocation budget;
- 64 MiB clean blob;
- 128 retained jobs and 256 MiB retained source/clean blobs;
- four concurrent authenticated requests, including response-body transfer;
- one active cleaning/retranslation pipeline; and
- bounded detector/model threads, with the browser daemon's two-minute idle
  shutdown releasing loaded model memory.

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

The font endpoint serves the installed, hash-verified Noto Sans SC and Noto
Serif SC variable font bytes from the developer package. Unknown font IDs and
missing resources fail closed; browser fallback remains available if a
supported font cannot be loaded.

## Koharu reuse boundary

The browser companion reuses `ProjectSession` and its content-addressed `BlobStore`,
`RuntimeManager`, Koharu's engine registry, and
`koharu_app::pipeline::run`. The browser layer only adapts progress,
cancellation, scene artifacts, stable protocol geometry, warnings, and bounded
blob retention. The speech-bubble filter and deterministic dialogue fill are
registered Koharu pipeline engines and artifacts, not a parallel browser-only
image stack.

Faithful and HSK prompts share Koharu's in-process local model state.
Vocabulary validation and lookup reuse `hsk-control`; no parallel tokenizer or
dictionary implementation exists in the browser crate. Production font bytes
are served from the same installed resource pack used by setup verification.

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
cargo fmt --all -- --check
cargo test -p browser-companion --all-targets -j 1
cargo test -p koharu-app --all-targets -j 1
cargo test -p koharu-llm --all-targets -j 1
cargo test -p hsk-control --all-targets -j 1
cargo clippy -p browser-companion -p koharu-app -p koharu-llm \
  --all-targets -j 1 -- -D warnings
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

- `cargo test -p browser-companion --all-targets -j 1` passed 54 library
  tests, the daemon resource test, 7 contract-fixture tests, the native
  handshake test, and the non-breakaway lifecycle test. The production
  breakaway probe remained explicitly ignored.
- `cargo test -p koharu-app --all-targets -j 1` passed 64 tests, including the
  joint bubble-mask and deterministic dialogue-fill regressions; the opt-in
  real-model smoke remained ignored.
- `cargo test -p koharu-llm --all-targets -j 1` passed 30 runtime tests; tests
  requiring initialized native runtime fixtures remained explicitly ignored.
- `cargo test -p hsk-control --all-targets -j 1` passed all unit,
  remediation, reproducibility, and workstream tests; only the explicit
  full-scale performance smoke remained ignored.
- `cargo fmt --all`, `git diff --check`, and `cargo clippy -p
  browser-companion -p koharu-app -p koharu-llm --all-targets -j 1 -- -D
  warnings` passed with the Visual Studio 2019 developer environment, the
  repository's verified LLVM 22.1.0 cache, and AWS-LC's prebuilt NASM path.
- A release daemon processed Nano Machine chapter 100 page 1 in 217 seconds,
  producing 11 English speech-bubble regions. Peak working set was 4.75 GiB;
  Korean and non-bubble English sound effects remained unchanged, and the real
  Firefox renderer reported 11 selectable regions with zero degraded fits.
- The packaged extension was temporarily installed in a fresh disposable
  Firefox profile and exercised through its real popup. The registered native
  host launched the installed daemon, which reused the cached clean image and
  completed HSK 5 correction for all 11 regions in 186.7 seconds with zero
  degraded fits.
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

- **Windows / real Firefox:** the installed native host and daemon completed a
  real popup-triggered Firefox translation after the one-shot host returned.
  Dedicated duplicate-daemon, idle-cleanup, and forced reconnect probes remain.
  The Codex outer job and the direct ignored probe still cannot substitute for
  explicit breakaway coverage on every supported Firefox/Windows combination.
- **Linux / real Firefox:** run the registration regression plus an actual
  per-user install from a non-ASCII UTF-8 path, invoke it from Firefox, verify
  manifest permissions and origin/ID enforcement, and confirm the `setsid()`
  daemon survives native-host exit before idle cleanup.
- **macOS / real Firefox:** repeat the non-ASCII registration and Firefox
  native-message launch against the packaged, signed/notarized binaries; verify
  the manifest under `~/Library/Application
  Support/Mozilla/NativeMessagingHosts`, executable/quarantine behavior,
  `setsid()` survival, and idle cleanup.
