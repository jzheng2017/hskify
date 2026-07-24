# HSK data

No full HSK list is committed here because a redistribution-compatible,
revision-pinned HSK 2.0 source has not yet passed the project's licence audit.

`test-seed-source.tsv` is a small project-authored control-flow fixture. Its
level assignments are only test inputs. It is incomplete, is not official HSK
data, and is rejected by the production load policy.

To import an audited source deterministically:

```text
cargo run -p hsk-control --bin hsk-import -- \
  --source path/to/source.tsv \
  --metadata path/to/audit.json \
  --output path/to/hsk-2.0.normalized.json \
  --delimiter tab
```

The source must contain `level`, `simplified`, `pinyin`, and `gloss` headers.
Optional headers are `simpler_words`, `independently_usable`, and
`frequency_rank`. List fields use `|`. CSV is supported with
`--delimiter comma`.

The metadata records the exact source URL, revision, SHA-256, SPDX expression,
attribution, redistribution decision, completeness, expected entry count, and
per-level counts. A claimed complete dataset without audited expected counts is
rejected.
