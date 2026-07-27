# External component evaluation

Hskify uses an established component when it satisfies the product contract
and removes candidates that do not. The selection criterion is not whether a
project can translate one manga page; it is whether it can supply
color-agnostic English story regions progressively, preserve the live source
image, and produce region-local cleanup suitable for patch-before-text
rendering.

## Retained

| Component | Decision | Production role |
| --- | --- | --- |
| PaddlePaddle PP-OCRv5 mobile detector | Retained | CUDA-batched text-line detection |
| PaddlePaddle English PP-OCRv5 mobile recognizer | Retained | CUDA-batched English recognition |
| `paddle-ocr-rs` preprocessing and DB postprocessing | Retained as an implementation reference | Apache-2.0 reference behavior adapted to the workspace's exact ONNX Runtime build |
| Qwen3.5 4B Q4_K_M | Retained | Resident direct English-to-HSK-Chinese generation |

The two Paddle model revisions and hashes are frozen in
`data/model-packs/manifest.v1.json`. Hskify adds viewport scheduling,
spatial deduplication, color-aware line grouping, English/story-role gates,
local erase masks, direct HSK translation, and the browser overlay. Those
product-specific stages are not available as one maintained drop-in package.

## Evaluated and rejected

| Candidate | Reason it was not used |
| --- | --- |
| `manga-image-translator-rust` detector paths | Useful reference project, but its full pipeline and data model do not match the progressive live-image contract; its alternate DBNet/CTD candidates missed too much varied story text in visual review |
| Older comic RT-DETR detector | Missed colored, stylized, and unballooned story text needed by chapter 5 |
| Kiuyha Manga-Bubble-YOLO | Balloon-only detection did not solve narration/unballooned text and missed visually diverse regions |
| Qwen3.5 2B Q4_K_M | Faster, but the controlled chapter-5 comparison did not qualify it as a naturalness/meaning-equivalent replacement |
| Hy-MT2 1.8B Q4_K_M | Faster, but lower structured success and more critical proxy failures disqualified it |

Preliminary detector counts produced before the final geometry audit are not
release evidence and are intentionally not quoted here. The authoritative
correctness result must come from the packaged chapter-5 benchmark against the
corrected committed annotations.

Rejected model/runtime directories, superseded detector bundles, superseded
benchmark runs, and incomplete experiment outputs are local cache artifacts
only. They have been removed; no fallback selector, compatibility adapter, or
dormant rejected production resource remains in the package.
