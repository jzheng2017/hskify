# Cleanup patch correctness audit

This small CPU-only crate compiles the browser companion's production
`pipeline_adapter/patch.rs` directly. It exists separately from the main
workspace so patch correctness does not require native OCR, translation, CUDA,
or `libclang` build dependencies.

Run the focused synthetic tests:

```powershell
$env:CARGO_TARGET_DIR = (Resolve-Path .cache).Path + '\cleanup-patch-audit-target'
cargo test --manifest-path scripts/benchmark/cleanup-patch-audit/Cargo.toml --jobs 6
```

The canonical corpus is
`fixtures/benchmarks/30-years-since-the-prologue-chapter-5`, whose manifest
currently marks all 36 gold annotation pages incomplete. There is therefore no
corpus-level cleanup-patch correctness command or per-region evidence output
yet. Add that audit only after the manifest declares complete, hash-pinned
annotation geometry; do not derive gold regions from detector output.
