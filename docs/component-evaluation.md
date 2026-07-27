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
| `ogkalu/comic-text-and-bubble-detector` RT-DETR-v2 R50 | Retained | CUDA-batched `text_bubble` and `text_free` proposals |
| PaddlePaddle English PP-OCRv5 mobile recognizer | Retained | CUDA-batched English recognition |
| Qwen3.5 4B Q4_K_M | Retained | Resident direct English-to-HSK-Chinese generation |

The detector, Paddle recognizer, and translation model revisions and hashes are
frozen in `data/model-packs/manifest.v1.json`. Hskify adds viewport scheduling,
spatial deduplication, English/story-role gates, color-aware local erase masks,
direct HSK translation, and the browser overlay. Those product-specific stages
are not available as one maintained drop-in package.

The retained RT-DETR model comes from the established Comic Translate work,
but Hskify does not embed that application's page pipeline or project model.
A complete drop-in translator was not selected because the candidates produce
reconstructed pages, assume their own job/project lifecycle, or cannot publish
patch-before-text regions in viewport priority. Reusing their full pipeline
would therefore remove required behavior rather than replace custom code.

## Evaluated and rejected

| Candidate | Reason it was not used |
| --- | --- |
| PaddlePaddle PP-OCRv5 mobile detector | Its general-purpose DB proposals missed too much colored, stylized, and low-contrast chapter-5 story text even at larger and multiscale inputs |
| `manga-image-translator-rust` detector paths | Useful reference project, but its full pipeline and data model do not match the progressive live-image contract; its alternate DBNet/CTD candidates missed too much varied story text in visual review |
| Kiuyha Manga-Bubble-YOLO | Balloon-only detection did not solve narration/unballooned text and missed visually diverse regions |
| Qwen3.5 2B Q4_K_M | Faster, but the controlled chapter-5 comparison did not qualify it as a naturalness/meaning-equivalent replacement |
| Hy-MT2 1.8B Q4_K_M | Faster, but lower structured success and more critical proxy failures disqualified it |

At the production threshold, the retained RT-DETR `text_bubble` plus
`text_free` proposals provide the strongest reviewed coverage of the varied
chapter-5 story regions among the evaluated candidates. Raw proposal precision
is intentionally not a product gate: recognition and the story-role filter must
reject non-English text, SFX, credits, and branding. The authoritative
accepted-region precision, OCR, rendering, and performance results come from
the packaged browser benchmark, whose retained evidence records the exact
threshold, build, and source hashes.

Rejected model/runtime directories, superseded detector bundles, superseded
benchmark runs, and incomplete experiment outputs are local cache artifacts
only. They have been removed; no fallback selector, compatibility adapter, or
dormant rejected production resource remains in the package.
