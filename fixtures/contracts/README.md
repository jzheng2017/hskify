# Browser build-contract fixtures

These files are the shared source of truth for the Firefox and Rust contract
parsers. Valid fixtures must parse and pass semantic validation in both
languages. Files under `invalid/` must be rejected.

Coordinates are normalized against the decoded source image. The contract has
no negotiated protocol version: the extension, native host, and daemon require
the same exact build fingerprint. HSK standard, languages, enum values,
monotonic update sequences, progress bounds, region identity, and terminal
events are validated rather than trusted.
