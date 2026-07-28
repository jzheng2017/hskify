# Local real-reader regression corpus

This manifest describes 20 real reader pages from seven chapters across Asura
and WEBTOON. The source image bytes are intentionally not committed or
redistributed. They live locally under:

```text
local-corpus/real-reader-v1/objects/<sha256>.<extension>
```

The tracked manifest is deterministic and contains provenance, byte length,
dimensions, SHA-256, quality focus, and semantic expectations. Chapter URLs
are attribution metadata only. Regression commands never fetch them.

Verify the manifest without local images:

```powershell
node scripts/real-reader-corpus.mjs manifest
```

Verify the local smoke corpus:

```powershell
node scripts/real-reader-corpus.mjs verify --selection smoke
```

Missing objects are failures with exit code 2 and a machine-readable
`requiredPaths` list when `--json` is used. Restore the exact files from an
authorized local source; the runner will not download them.

Run the current release daemon and pipeline:

```powershell
node scripts/run-real-reader-regression.mjs --selection smoke
```

The pipeline command writes a timestamped evidence directory under `runs/`.
It requires the current `target/release/hsk-manga-browser-daemon.exe`, submits
only verified local objects, and asserts:

- every job reaches `complete`;
- every expected story region is present;
- final HSK repair state is never `pending`;
- patch descriptors are valid, overlap source text, and resolve to PNG bytes;
- annotated names remain unchanged with `keep-original`;
- sound effects excluded by settings do not become translation regions;
- scan-credit splash pages remain zero-region non-story controls;
- multiplier notation such as `X3` preserves its numeric value on ultra-tall strips;
- the dense page differs between HSK 2 and HSK 5 on at least one shared
  region, with every HSK 2 result routed through level 2 validation.

`summary.json` is the machine-readable release gate. Per-job update streams
are preserved next to it for diagnosis.

The Firefox renderer harness also resolves its source images from this local
corpus. Its mocked regions remain a deterministic renderer contract test;
they are not pipeline or visual-quality evidence.
