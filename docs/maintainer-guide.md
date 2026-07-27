# Maintainer guide

The Hskify browser product is the direct progressive performance path, even
though it reuses implementation crates whose names and general capabilities
are broader.

## Product invariants

- Windows + NVIDIA GeForce RTX 4080 SUPER 16 GB + compute capability 8.9.
- CUDA-only performance build through
  `scripts/Invoke-PerformanceBuild.ps1`.
- Ignored post-success build attestation with the exact source tree,
  Windows x86_64 MSVC release/CUDA configuration, hardware/toolchain claims,
  and native-host/daemon identities. Packaging and measurement fail closed
  without it.
- Exact build fingerprint shared by extension, native host, daemon, and
  fixtures; no negotiated protocol version.
- Unversioned loopback browser routes.
- One append-only progressive job log; no status/result dual model.
- Region-local PNG patches; no reconstructed cleaned page.
- Patch installed before selectable text.
- Confirmed English dialogue, thought, and eligible story narration only;
  sound effects, credits, promotion, branding, non-English text, and ambiguous
  OCR remain excluded.
- Color-agnostic local cleanup; no chapter, phrase, coordinate, URL, hash, hue,
  foreground-color, or background-color allowlists.
- Proposal and OCR acceptance must work for arbitrary foreground/background
  colors; cleanup must preserve the accepted region's local color, texture,
  gradients, contours, and styling outside the erase mask.
- Direct English-to-HSK Chinese primary generation, with at most one targeted
  invalid-item repair batch.
- Story inclusion is deterministic before the translation-only LLM; the LLM
  returns either translated story text or the typed non-story disposition for
  unrelated page furniture.
- Viewport priority controls processing and publication only. Stable
  `readingOrder` remains document order followed by within-image reading order.
- No browser projects, history, page-level pipeline markers, or level-change
  retranslation.
- Local pinyin, dictionary, comparison, and Mandarin speech retained.

If code changes one of these invariants, update the architecture, browser
contract, companion implementation, fixtures README, and benchmark method in
the same change.

## Review routing

| Change | Required documentation/evidence |
| --- | --- |
| Route, header, or JSON contract | `browser-contract.md`, Rust/TS fixtures, exact fingerprint decision |
| Tile, batch, viewport, cache, or patch behavior | `architecture.md`, companion implementation, chapter benchmark fields |
| Admission limit, cache byte ceiling, or thread limit | Architecture resource table, companion defaults, stress/boundary tests |
| Model, prompt, validator, HSK, or dictionary revision | Cache-key review, model record, evidence environment |
| GPU/toolchain/runtime change | Performance-build gate and a new separate benchmark configuration |
| Reader lookup/comparison/speech change | Architecture, Firefox checklist, accessibility/privacy checks |
| Claimed performance or quality result | Raw hashed evidence bundle and completed fixture audit |

The sole canonical release workload is the 36-image *30 Years Since the
Prologue* chapter 5 fixture. It is intentionally visually diverse, but it is
never a production allowlist or a license for benchmark-specific tuning.

## Verification commands

Use the performance build for release artifacts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\Invoke-PerformanceBuild.ps1
```

The command writes binaries and
`hskify-performance-build-attestation.json` under `target\release` only after
Cargo succeeds. Run the
read-only hardware/source/toolchain gate without compiling via
`-PrerequisitesOnly`.

The normal automated suite should include, as applicable:

```text
cargo fmt --all -- --check
cargo test -p browser-companion --all-targets -j 1
cargo test -p koharu-app --all-targets -j 1
cargo test -p koharu-llm --all-targets -j 1
cargo test -p hsk-control --all-targets -j 1
cargo clippy -p browser-companion -p koharu-app -p koharu-llm --all-targets -j 1 -- -D warnings
bun run typecheck:firefox
bun run test:firefox
bun run build:firefox
git diff --check
```

These are commands to run, not claims that the current branch passed them.
Record command, tool versions, exit status, and raw output when creating
evidence.

## Documentation discipline

Do not copy measurements forward across fingerprints, model revisions,
hardware, browser builds, fixture hashes, or architecture changes. Mark an
unrun check as pending. A failed or blocked run remains failed or blocked until
new raw evidence exists.

The broader reused crates may retain project, desktop, remote-provider, or
fallback capabilities for other binaries. Do not describe those surfaces as
Hskify browser features unless the browser companion actually mounts and tests
them.
