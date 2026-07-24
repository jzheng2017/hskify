# Browser protocol v1 fixtures

These files are the shared source of truth for the Firefox and Rust contract
parsers. Valid fixtures must parse and pass semantic validation in both
languages. Files under `invalid/` must be rejected.

Coordinates are normalized against the decoded source image. Protocol version,
HSK standard, languages, enum values, progress bounds, unique region IDs, and
terminal state/stage combinations are validated rather than trusted.
