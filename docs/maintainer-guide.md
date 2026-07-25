# Hskify maintainer guide

Hskify is maintained as a deliberate fork of Koharu. The fork preserves a
reviewable upstream relationship while allowing Hskify to change shared
pipeline behavior and ship a tightly integrated local Firefox companion.

Read the [architecture overview](architecture.md) and
[ADR 0001](architecture-decisions/0001-koharu-upstream-pin.md) before changing
the upstream boundary.

## Remote and pin policy

The expected remotes are:

```text
origin    https://github.com/jzheng2017/hskify.git
upstream  https://github.com/mayocream/koharu.git
```

`origin` is authoritative for Hskify. `upstream` is authoritative for Koharu.
ADR 0001 records the last reviewed Koharu revision, upstream version, and
licence. Never describe the moving upstream branch as the pin.

Verify the remotes and current pin before a synchronization:

```bash
git remote -v
git fetch upstream --tags
git show --no-patch --format=fuller 2107843f0c7e2458de5a329980c78575401babb5
```

If a remote URL changes, update this guide and ADR 0001 in the same review.

## Synchronizing Koharu

Use a merge-based update on a dedicated branch. Rebasing the long-lived fork
rewrites Hskify history and makes later provenance and conflict review harder.

1. Start with a clean working tree and create a branch such as
   `sync/koharu-<version>`.
2. Fetch `upstream` and identify an immutable commit or signed release tag.
3. Review the upstream range before merging:

   ```bash
   git log --oneline --decorate <old-pin>..<new-pin>
   git diff --stat <old-pin>..<new-pin>
   git diff <old-pin>..<new-pin> -- Cargo.toml Cargo.lock crates ui docs
   ```

4. Merge without committing so conflicts and generated changes can be reviewed
   together:

   ```bash
   git merge --no-commit --no-ff <new-pin>
   ```

5. Resolve conflicts according to ownership:

   - keep Hskify browser contracts, authentication, resource bounds, HSK policy,
     package identity, and installer behavior unless an explicit Hskify
     decision changes them;
   - take upstream fixes in inherited Koharu code when they do not violate a
     Hskify ADR or browser invariant;
   - port shared Hskify pipeline changes onto the new upstream shape instead of
     replacing an entire upstream file;
   - keep localized Koharu docs attributed to Koharu, and keep Hskify docs in
     the top-level `docs` set; and
   - regenerate lockfiles or generated clients only with their documented
     tools—never hand-resolve generated output blindly.

6. Audit changes in dependency licences, model/runtime downloads, network
   behavior, persisted schemas, API contracts, CLI flags, and minimum toolchain
   versions.
7. Run the verification matrix below and record failures honestly.
8. Update ADR 0001 with the new immutable revision, version, date, reusable
   surfaces, and any changed consequences. Add a new ADR when the integration
   strategy or security boundary changes; do not rewrite the history of an
   accepted decision.
9. Update the architecture overview, licence inventory, implementation notes,
   and user-facing docs affected by the merge.
10. Review the final diff against both parents before committing:

    ```bash
    git diff --check
    git diff --stat HEAD
    git log --left-right --cherry-pick --oneline HEAD...upstream/main
    ```

Do not mix an upstream synchronization with unrelated feature work. Small,
separate follow-up commits make regressions and future merges easier to locate.

## Verification matrix

At minimum, run:

```text
cargo fmt --all -- --check
cargo test -p browser-companion --all-targets -j 1
cargo test -p koharu-app --all-targets -j 1
cargo test -p koharu-llm --all-targets -j 1
cargo test -p hsk-control --all-targets -j 1
cargo clippy -p browser-companion -p koharu-app -p koharu-llm --all-targets -j 1 -- -D warnings
bun install
bun run typecheck:firefox
bun run test:firefox
bun run build:firefox
```

Also run the native-host registration tests for the affected platforms and
repeat every applicable item in the
[Firefox manual test checklist](firefox-manual-test-checklist.md). A unit test
or mocked native host does not replace the real packaged-Firefox launch,
permission, reconnect, and process-lifecycle checks.

For ML changes, verify with pinned fixtures before using production models.
Never commit downloaded model weights, language datasets, reader images, cache
directories, secrets, or machine-specific build paths.

## Why Hskify is a fork

The current integration needs more than a client of Koharu:

- Hskify invokes `koharu_app::pipeline::run` and shares project sessions, blob
  storage, runtime/model state, cancellation, engines, and scene artifacts.
- Dialogue-only cleaning adds registered pipeline engines and retained geometry
  at shared layers.
- Browser jobs need stable conversions from internal scene/mask types, not only
  Koharu's public desktop or permissive general RPC surface.
- A single installed process and resource pack avoid duplicate multi-gigabyte
  model state and inconsistent caches.
- Security fixes sometimes need to cross launcher, daemon, application, and
  pipeline boundaries atomically.

A repository fork makes those changes testable together and keeps the exact
upstream provenance visible.

## When a package would be better

Prefer normal package dependencies if Koharu eventually publishes stable,
versioned Rust crates for the required application/pipeline surfaces and
Hskify no longer patches their internals. A package split is justified when:

- the needed APIs and artifact schemas have semver guarantees;
- Hskify-specific engines can be registered without modifying Koharu crates;
- upstream and Hskify test suites can run independently;
- dependency features can exclude desktop, remote-provider, and unused runtime
  surfaces; and
- one dependency graph can still share model state and storage without copying
  large artifacts.

At that point, extract only the Hskify-owned extension, companion, protocol, and
HSK layers. Do not publish a package merely to disguise a continuously patched
git dependency.

## When a separate service would be better

A separately deployed service is appropriate only if independent scaling,
multi-user scheduling, stronger process isolation, or a language-agnostic
network API becomes more important than Hskify's one-user local-first design.
That choice would require a new threat model covering network authentication,
TLS, tenancy, storage isolation, data retention, upgrades, observability, and
potential image/text egress.

The current browser daemon is not that service: it is per-user, loopback-only,
short-lived, and intentionally exposes a minimal authenticated API. Moving
Koharu behind a separately installed or remote service today would add version
skew, duplicate lifecycle management, and a larger security surface without a
demonstrated operational benefit.

Any transition from fork to package or service requires a new ADR, a migration
plan for projects/cache/resources, and compatibility tests for the frozen
browser protocol.
