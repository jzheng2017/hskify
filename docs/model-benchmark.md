# Direct HSK model benchmark record

## Frozen comparison protocol

Every candidate must use the same generic direct English-to-HSK-Chinese
protocol:

```text
revision: direct-hsk-en-zh-generic-v15-2026-07-26
prompt SHA-256: 8faf04bf2c4e5a1d2b43be426f93c157cd62ce1dc18676aee79d22b678bb396f
validator SHA-256: 81c581cc7af0c97f2672ad2624135d983eb01a22d64ab8322dc64f7b957a461d
decoding: greedy, unpenalized
repair: at most one targeted repair per rejected item
```

The prompt contains only generic meaning-preservation instructions and
application-supplied glossary entries. It must not contain chapter-specific
terms, translations, source phrases, character names, coordinates, colors,
URLs, hashes, or trigger rules.

The production protocol has since advanced to
`direct-hsk-en-zh-generic-v17-2026-07-27` with prompt hash
`sha256:ec287e2d5f7ba898f70f80852b98b67e9d2bc25f9e3b0a1fddf1041baab6ef2a`
and validator hash
`sha256:ca74f50314d77f0048e0a49a5ef050e3a7a7f4e942c6eb7c7e22da75ada6d7d1`.
That revision adds generic non-story classification plus deterministic
exclusion support and OCR-digit validation; it contains no Chapter 5 phrase,
name, coordinate, color, URL, or hash trigger. The controlled three-model
record below remains the translation-quality selection record for the 214
story targets, while product-path exclusion behavior is verified separately.

## Canonical comparison workload

The sole release comparison workload is the reviewed English target set from
the 36-image *30 Years Since the Prologue* chapter 5 fixture:

```text
fixtures/benchmarks/30-years-since-the-prologue-chapter-5/
```

That workload was chosen for its varied dialogue, thought, narration,
lettering, foreground colors, backgrounds, and visual styles. The diversity is
a regression challenge, not permission to tune a model or prompt to one
chapter.

The committed manifest contains reviewed geometry for all 36 pages and 218
story regions. All 214 translation targets have approved Chinese, pinyin, and
token-level HSK annotations. The final automated comparison used the corrected
reading order and all 214 targets.

## Candidates and qualification

When gold is complete, compare these candidates in one controlled GPU sequence:

1. Qwen3.5 4B Q4_K_M
2. Qwen3.5 2B Q4_K_M
3. Hy-MT2 1.8B Q4_K_M

Each candidate must receive exactly the same ordered target rows, batching,
preceding context, glossary, prompt, validator, decoding settings, warm-up,
and resource monitoring. Raw evidence must preserve model hashes, commands,
environment, per-row outputs, timing samples, and failure classifications.

A smaller model qualifies only if it:

- adds no critical meaning errors under human review;
- preserves protected names and numbers at least 99%;
- matches the 4B model's naturalness under blinded fluent-reader review; and
- satisfies the structural and deterministic validation gates.

Automated checks for output structure, names, numbers, negation, and question
intent are useful diagnostics. They are not substitutes for critical-meaning
and naturalness review.

## Final automated comparison

The controlled run completed on the RTX 4080 SUPER with no sustained paging:

| Candidate | Warm total | Batch p50 | Batch p95 | Structured | Critical proxy items | Character unigram F1 | Character bigram F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B Q4_K_M | 18.54 s | 506 ms | 694 ms | 214/214 | 1 | 0.612 | 0.427 |
| Qwen3.5 2B Q4_K_M | 10.13 s | 281 ms | 373 ms | 214/214 | 4 | 0.519 | 0.327 |
| Hy-MT2 1.8B Q4_K_M | 14.26 s | 286 ms | 849 ms | 203/214 | 13 | 0.447 | 0.336 |

The run peaked at 4.81 GiB private bytes and 4,313 MiB device-wide VRAM use.
The complete ignored evidence is
`.cache/translation-model-benchmark/runs/chapter5-final-20260726-r2`;
`benchmark.json` is 887,954 bytes with SHA-256
`cf8b97cf489b1c2231bbc6222d9c571df9749b76dae535cece87ae6cfed7f90d`.

The fixture contains no approved source-English-to-Chinese proper-name
glossary and no ASCII-number preservation cases, so the nominal empty-set
preservation rates are not treated as a 99% qualification result. Human
naturalness review is also not invented.

## Selection

Neither smaller model qualifies: both add critical proxy failures, both score
materially below the 4B reference, Hy-MT2 also fails structure on 11 items, and
the required human naturalness/name-number evidence is absent. The plan's
fallback therefore applies directly: Qwen3.5 4B Q4_K_M is the production
model. The rejected model files are not packaged and are removed from the
local evaluation cache after this evidence is retained.
