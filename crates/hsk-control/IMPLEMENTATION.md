# Workstream C implementation note

## Outcome

`hsk-control` is a pure Rust API for HSK 2.0 vocabulary control and local
Chinese lookup. It contains no browser, HTTP, model, async, or Koharu pipeline
code.

Production construction uses:

```rust
HskControl::from_json(hsk_artifact, dictionary_artifact)
```

and requires both generated artifacts to declare `complete`. The repository
contains only project-authored `test-seed` artifacts. The normal constructor
returns `HskControlError::DatasetIncomplete` for them. Fixture code must opt in
by calling `HskControl::from_embedded_test_seed()` or passing
`LoadPolicy::AllowIncompleteTestSeed`.

This is deliberate: the current branch cannot be mistaken for a complete HSK
or dictionary product.

## Data and cache identity

Generated artifacts record:

- schema version and dataset revision;
- HSK standard (`2.0`) where applicable;
- exact source name, URL, revision, and SHA-256;
- SPDX expression, licence URL, attribution, and an affirmative redistribution
  audit;
- completeness state;
- audited total entry count and, for HSK, counts for levels 1 through 6.

The importer rejects a source hash mismatch, missing attribution, a negative
redistribution audit, or a complete dataset without expected counts. Runtime
loading rechecks normalized forms and audited counts.

`HskControl::cache_revision()` is a SHA-256 identity over the normalized HSK
and dictionary contents plus the normalization, Jieba segmentation, compound
guard, lookup, and dependency revisions. It changes deterministically when any
of those inputs changes.

## Normalization

The pipeline is:

1. Unicode NFKC;
2. removal of soft-hyphen/zero-width formatting controls;
3. `opencc-fmmseg` 0.8.0 `tw2sp`, then `hk2s`, for mainland Simplified output;
4. canonical Chinese punctuation, decimal-point preservation, collapsed
   whitespace, and edge trimming.

`opencc-fmmseg` is a cross-platform, pure Rust OpenCC-compatible converter with
bundled lexicons, so the companion does not need a system C++ OpenCC install.
Version 0.8.0 is pinned because 0.11.x hard-pins Rayon 1.10 and cannot resolve
inside this workspace, which already pins Rayon 1.12.

Limitations:

- this is an OpenCC-compatible port, not the official C++ implementation;
- regional phrase conversion is deterministic (`tw2sp` followed by `hk2s`) but
  cannot infer every context-sensitive regional meaning;
- OpenCC does not return source alignment for length-changing phrase mappings.
  Therefore every `HskViolation.start_char/end_char` is exact in Unicode scalar
  offsets into `ValidationReport.normalized_text`, not the unnormalized input.

The last point is explicit in the API and regression-tested with non-BMP emoji.

## Validation algorithm

At engine construction, every HSK word from levels 1–6 and every local
dictionary headword is added to `jieba-rs`'s mature default lexicon. The
selected HSK level does not bias primary segmentation.

For each normalized region not occupied by an explicit protected name:

1. Jieba produces the primary segmentation.
2. A longest-known-compound guard scans Jieba token boundaries using the full
   HSK/dictionary trie.
3. A known higher-level HSK word is rejected as one span even when its
   characters are lower-level words.
4. A known dictionary headword outside the allowed HSK set is also rejected as
   one span. This intentionally prefers false rejection for lexicalized
   compounds.
5. Only an unknown token may pass allowed-word dynamic-programming
   decomposition, and only with complete coverage by independently usable
   allowed HSK words.
6. Chinese/Arabic numeric forms and numeric-plus-allowed-measure-word
   decompositions are permitted.
7. Unknown Chinese spans and unprotected non-Chinese lexical tokens fail.

Protected person names, place names, titles, and unavoidable proper nouns are
supplied explicitly. Each occurrence is returned with an exact span and reason;
it is never an unreported strict pass.

Suggestions are deterministic and capped at three. Curated lower-level words
from the artifact come first, followed by allowed decomposition components and
lower-level entries with overlapping non-stopword English glosses, ranked by
gloss match, HSK level, optional frequency rank, and lexical order.

## Lookup

The CC-CEDICT importer accepts:

```text
traditional simplified [pin1 yin1] /definition one/definition two/
```

It validates structure, normalizes the Simplified headword, converts numbered
pinyin to tone marks, sorts/deduplicates entries, and writes canonical JSON.

Lookup normalizes the selection and uses longest-match segmentation over the
combined dictionary/HSK trie, with a single-character fallback. Results merge
dictionary pinyin/definitions with HSK pinyin/gloss/level metadata. Proper names
are marked only when the caller explicitly protects them; dictionary wording
never silently creates an HSK exception.

## Bounded correction support

`HskControl::correction_loop()` validates the initial rewrite and permits at
most two subsequent correction requests. Feedback includes exact HSK
violations plus deterministic preservation failures for:

- numeric forms;
- protected names present in the faithful reference;
- added or removed Chinese negation.

After the third invalid evaluation it returns `Failed`; later evaluations return
`Terminated`. The crate never initiates or owns a model request.

## Reproducible import

HSK:

```text
cargo run -p hsk-control --bin hsk-import -- \
  --source SOURCE.tsv \
  --metadata AUDIT.json \
  --output hsk-2.0.normalized.json \
  --delimiter tab
```

Dictionary:

```text
cargo run -p hsk-control --bin cedict-import -- \
  --source cedict_ts.u8 \
  --metadata AUDIT.json \
  --output cc-cedict.normalized.json
```

The two committed test-seed outputs are regenerated byte-for-byte in
`tests/import_reproducibility.rs`.

For an installed complete dataset, run:

```text
cargo run -p hsk-control --bin resource-smoke -- \
  hsk-2.0.normalized.json cc-cedict.normalized.json 10000
```

## Licence audit

Committed data:

- HSK test seed: project-authored, `GPL-3.0-only`; explicitly not an official
  HSK list.
- dictionary test seed: project-authored CC-CEDICT-format parser fixture,
  `GPL-3.0-only`; no CC-CEDICT entries copied.

Direct lexical/normalization dependencies:

- `opencc-fmmseg` 0.8.0: MIT; its bundled OpenCC-compatible lexicons are
  credited to OpenCC.
- OpenCC source lexicons: Apache-2.0
  (<https://github.com/BYVoid/OpenCC>).
- `jieba-rs` 0.10.1 and its default distribution: MIT
  (<https://github.com/messense/jieba-rs>).
- `unicode-normalization` 0.1.25: MIT OR Apache-2.0.

Candidate upstream data not committed:

- CC-CEDICT is published under CC BY-SA 4.0 and requires attribution and
  share-alike handling
  (<https://www.mdbg.net/chinese/dictionary?page=cc-cedict>). A release owner
  must pin a release, preserve its data licence/attribution, and approve its
  combined-distribution treatment before vendoring a generated artifact.
- Several repositories publish HSK-derived lists under permissive repository
  licences, but their HSK level provenance and/or inherited dictionary
  definitions are not sufficiently clear for this branch to certify the full
  data. No full HSK list is committed until that audit is resolved.

## Verification evidence (2026-07-24, Windows, Rust 1.95.0)

- `cargo fmt -p hsk-control -- --check`: pass.
- `cargo clippy -p hsk-control --all-targets -- -D warnings`: pass.
- `cargo test -p hsk-control --all-targets`: 20 passed, 0 failed, one explicit
  performance test ignored by the normal suite.
- `cargo test -p hsk-control --test performance_smoke -- --ignored --nocapture`:
  pass; loaded 5,000 synthetic HSK records plus 125,000 synthetic dictionary
  records and completed 1,000 validation+lookup iterations; total test time
  22.08 seconds in an unoptimized test build.

The synthetic scale test contains no third-party words. A measured benchmark
against real complete datasets remains blocked on the two data audits above.

## Integration handoff

This workstream was forbidden from changing the root manifest or `Cargo.lock`.
The crate manifest introduces `opencc-fmmseg` and `unicode-normalization` (and
uses the already selected `sha2`/CSV ecosystem). The integrator must regenerate
and review the root lockfile when merging. No existing Koharu crate, frozen
fixture, browser companion, extension file, root manifest, or committed
lockfile is changed by this branch.
