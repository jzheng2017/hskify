# Shared language-model benchmark record

## Frozen comparison protocol

Every candidate uses the same generic ordered English-to-Simplified-Chinese
source-span protocol and the real tokenizer:

- resident context: 4,096 tokens;
- microbatch: at most six ordered units;
- decoding: greedy and unpenalized;
- natural mode: faithful Chinese plus deterministic teaching metadata;
- strict mode: HSK realization and at most one terminal repair;
- publication: final-only.

The shared prompt describes meaning preservation and generic span kinds only.
It contains no manga, bubble, panel, OCR, website, chapter phrase, coordinate,
color, URL, hash, or trigger language. The adapter adds an OCR-correction
instruction only for `ocr` provenance; `dom` text is authoritative. Exact model,
prompt, validator, tokenizer, HSK, and dictionary identities are recorded in
each evidence bundle and cache key.

## Canonical workloads

Language qualification requires both:

- the complete ordered local real-reader-v2 core/stress image corpus; and
- reviewed light-novel chapters covering semantic and div-heavy extraction,
  dialogue, headings, quotes, lists, captions, names, pronouns, numbers,
  questions, long blocks, mid-chapter visible-first jumps, and joined-block
  validation.

The packaged Firefox path is authoritative. Model-only replay is diagnostic and
cannot replace final reader rendering, restoration, and update-replay checks.
Image throughput comparisons use the local image-runtime runner documented in
the maintainer guide. It creates a fresh native result-cache state and Firefox
profile for each baseline/current measurement while reusing the verified
installed resource pack, joins native language timing by browser-observed job
ID, and samples full-runtime NVIDIA memory directly.

## Candidates and qualification

When gold fixtures are complete, compare in one controlled GPU sequence:

1. Qwen3.5 4B Q4_K_M
2. Qwen3.5 2B Q4_K_M
3. Hy-MT2 1.8B Q4_K_M

Each candidate receives identical registered source order, surrounding context,
span metadata, tokenizer packing, prompt, validator, decoding, warm-up, and
resource monitoring. Raw evidence preserves model hashes, commands,
environment, per-item outputs, timing, preservation reasons, and repair counts.

A smaller model qualifies only if it:

- adds no critical meaning errors under blinded human review;
- renders names in Chinese and preserves numbers/question intent;
- maintains cross-block names, pronouns, and dialogue continuity during
  visible-first execution;
- passes joined-block and strict HSK validators; and
- meets both document and image latency/resource gates.

Automated structure, vocabulary, entity, number, negation, and question checks
are useful diagnostics, not substitutes for meaning and naturalness review.

## Selection

Qwen3.5 4B Q4_K_M remains the sole resident production language model. Document
warm-up loads this language runtime without vision; image mode reuses the same
instance when loading its vision runtime. Smaller candidates are not packaged.
