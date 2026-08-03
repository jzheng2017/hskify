# Performance-build licence inventory

This inventory covers the Hskify browser product's mandatory runtime resources.
It contains only the required production resource identities.

| Resource | Required identity | Licence/audit state | Distribution state |
| --- | --- | --- | --- |
| Hskify code and project-authored fixtures | current repository revision | GPL-3.0-only | repository |
| Qwen3.5 4B Q4_K_M | exact revision and SHA-256 in the model manifest | Apache-2.0 inherited from Qwen; exact file identity must be preserved | mandatory local resource; bundling depends on release packaging review |
| HSK 2.0 artifact | generated normalized artifact | provenance, redistribution, attribution, revision, and completeness audit pending | not committed as production data |
| CC-CEDICT-compatible artifact | generated normalized artifact | CC BY-SA 4.0 source obligations and combined-distribution review pending | not committed as production data |
| Noto Sans SC / Noto Serif SC variable fonts | installed files verified by package manifest | SIL Open Font License 1.1; package must preserve licence text and exact hashes | local packaged resource when supplied |
| Comic text and bubble detector | `ogkalu/comic-text-and-bubble-detector@16e8a622f91fabc6b5b65c96d32d1183f8843546` config, preprocessor, and weights plus manifest SHA-256 values | upstream model terms and exact identities must be preserved | mandatory local resource |
| PP-OCRv6-small detector and recognizer | exact official model revisions and manifest SHA-256 values | upstream Paddle model terms and exact identities must be preserved | mandatory local resource |
| Real-reader v2 chapter source pages | content-addressed local objects from the reviewed core/stress set | source provenance and redistribution review required before release | capture-required; never fetched by the release runner |
| Real-reader v2 annotations | project-created factual evaluation data | exhaustive geometry, entity, style, cleanup, HSK, and exclusion review required | capture-required |

The repository's small HSK and dictionary test seeds are control-flow fixtures,
not production language data. They must never be packaged or described as a
complete HSK list or dictionary.

A performance evidence bundle records exact hashes but does not itself grant
redistribution rights. Release approval requires the corresponding licence,
NOTICE/attribution, provenance, and distribution decision for every shipped
byte.
