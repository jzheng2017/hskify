# hsk-control

Pure Rust HSK 2.0 vocabulary validation and local CC-CEDICT-compatible
dictionary lookup for the browser companion.

The normal constructor rejects incomplete data. The repository deliberately
contains only tiny, project-authored test seeds. Their embedded resources and
`HskControl::from_embedded_test_seed()` are compiled only with the non-default
`test-seeds` Cargo feature and must not be presented to users as the HSK
vocabulary or a complete dictionary.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the API boundary, algorithms,
licence audit, reproducible import commands, and known limitations.

For a functional local development setup on Windows, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_local_language_data.ps1
```

The script downloads pinned HSK 2.0 inputs plus the current CC-CEDICT release,
generates production-loadable artifacts under `.cache/language-data`, and runs
the full-resource smoke binary. Large generated resources stay outside Git.
