# Local dictionary data

The importer accepts the standard CC-CEDICT text shape:

```text
traditional simplified [pin1 yin1] /definition one/definition two/
```

Generation is deterministic and converts numbered pinyin to tone marks:

```text
cargo run -p hsk-control --bin cedict-import -- \
  --source path/to/cedict_ts.u8 \
  --metadata path/to/audit.json \
  --output path/to/cc-cedict.normalized.json
```

CC-CEDICT is available under CC BY-SA 4.0 and requires attribution/share-alike
handling. The project does not vendor it in this workstream; a release owner
must pin a release, record its exact SHA-256 and entry count, preserve the
required attribution, and confirm combined-distribution obligations before
committing generated data.

`test-seed-cedict.u8` is a small project-authored format fixture. It is marked
`test-seed`, is not CC-CEDICT, and is rejected by the production load policy.

Official download/licence page:
<https://www.mdbg.net/chinese/dictionary?page=cc-cedict>
