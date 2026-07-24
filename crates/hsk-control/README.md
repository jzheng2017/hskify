# hsk-control

Pure Rust HSK 2.0 vocabulary validation and local CC-CEDICT-compatible
dictionary lookup for the browser companion.

The normal constructor rejects incomplete data. The repository deliberately
contains only a tiny, project-authored test seed; it is available through
`HskControl::from_embedded_test_seed()` and must not be presented to users as
the HSK vocabulary or a complete dictionary.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the API boundary, algorithms,
licence audit, reproducible import commands, and known limitations.
