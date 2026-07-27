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
| PP-OCRv5 mobile detector | `PaddlePaddle/PP-OCRv5_mobile_det_onnx@e6f4fa85f00e168c862bc462aebca69eef9b3d3d` plus manifest SHA-256 | upstream Paddle model terms and exact identity must be preserved | mandatory local resource |
| English PP-OCRv5 mobile recognizer | `PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx@3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b` plus manifest SHA-256 | upstream Paddle model terms and exact identity must be preserved | mandatory local resource |
| *30 Years Since the Prologue* chapter 5 source pages | 36 remote source identities and local ignored bytes | benchmark-only evaluation; redistribution permission is not asserted | never committed |
| Chapter 5 annotations | project-created factual evaluation data | audit records source-page identity and review provenance | 36 pages, 218 reviewed regions, and complete translation/pinyin/token gold committed |

The repository's small HSK and dictionary test seeds are control-flow fixtures,
not production language data. They must never be packaged or described as a
complete HSK list or dictionary.

A performance evidence bundle records exact hashes but does not itself grant
redistribution rights. Release approval requires the corresponding licence,
NOTICE/attribution, provenance, and distribution decision for every shipped
byte.
