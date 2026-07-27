# PP-OCR recognition benchmark

`ocr_models_chapter5.py` compares the official ONNX exports of:

- `PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx` at
  `3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b`; and
- `PaddlePaddle/PP-OCRv6_small_rec_onnx` at
  `b8f84f0b80c529de40b4fbb3544b84fa7233a513`.

The script pins every Python dependency and validates the byte count and
SHA-256 of each ONNX/config artifact. It also validates every chapter source
WebP and annotation against the frozen manifest before doing any work.

Preparation never creates an ONNX Runtime session:

```powershell
uv run .\scripts\benchmark\ocr_models_chapter5.py prepare `
  --inspect-montage .\.cache\ocr-benchmark\inspection.png
uv run .\scripts\benchmark\ocr_models_chapter5.py self-test
```

The canonical fixture is
`fixtures/benchmarks/30-years-since-the-prologue-chapter-5`. Its manifest is
the authority for page, annotation, OCR-target, and exclusion counts. If
`annotationStatus.status` is not `complete`, corpus preparation, segmentation
audit, detector scoring, and model execution fail before reading or inventing
gold annotations. The committed Chapter 5 fixture is complete.

Narration/caption regions (`kind: narration`) are OCR and translation targets,
but are excluded from speech-bubble detector gold and its precision/recall
denominators. Dialogue and thought regions remain detector gold. Credits,
scanlation promotions, sound effects, non-English/gibberish text, and
title/series branding remain outside the annotated target corpus.

`self-test` remains available because it exercises model artifact validation,
preprocessing, CTC decoding, scoring, and synthetic segmentation without
loading the gold corpus or creating an ONNX Runtime session.

GPU execution is deliberately gated. After the caller has explicitly cleared a
serialized candidate run, run:

```powershell
uv run .\scripts\benchmark\ocr_models_chapter5.py run `
  --gpu-clearance EXPLICITLY_CLEARED_SERIALIZED_GPU_RUN
```

Each candidate runs in its own child process. Once gold is complete, the
evidence contains predictions for every manifest-declared confident-English
OCR target, strict micro CER, case-insensitive diagnostic CER, exact artifact
revisions and hashes, line-recognizer and production-contract latency for batch
sizes 1/2/4/8, and sampled peak process RAM/device VRAM. Manifest-declared
punctuation-only exclusions remain available to non-model audits and never
enter OCR inference.

The candidates are text-line models while the Rust browser OCR boundary
supplies multiline text-block crops. The harness uses the same text-polygon
bounds expanded by three pixels and splits lines with an annotation-independent
Otsu/horizontal-projection algorithm. Gold line breaks are used only to report
a splitter diagnostic. They never influence crops or inference.
