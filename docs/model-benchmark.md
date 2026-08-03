# Direct HSK model benchmark record

## Frozen comparison protocol

Every candidate must use the same generic direct English-to-HSK-Chinese
protocol:

```text
revision: direct-hsk-en-zh-ordered-connected-regions-v81-2026-08-02
prompt SHA-256: 13ce6d86028dfcb13d8c7a0fca8306f6108dee4c92f67a40dbe6e29e34d64d5c
validator SHA-256: 887ad273362f1005ff495a74ffa9487a4de524e47c51bcfc210e1b9ced7ab1c9
decoding: greedy, unpenalized
repair: one bounded batch; at most one new-evidence attempt per rejected item
```

The prompt contains only generic meaning-preservation instructions and
application-supplied glossary entries. It must not contain chapter-specific
terms, translations, source phrases, character names, coordinates, colors,
URLs, hashes, or trigger rules.

The current revision adds ordered chapter context and bounded following-source
context at microbatch boundaries. The protocol contains no chapter phrase,
name, coordinate, color, URL, or hash trigger. Translation quality is
qualified only through the ordered real-reader-v2 corpus; no deleted chapter
fixture or model-only replay is a release substitute.

## Canonical comparison workload

The release comparison workload is the complete ordered local real-reader-v2
core/stress set. Its manifest and annotations are capture-required until every
page, region, exclusion, entity, style, cleanup allowance, and HSK alternative
has been independently reviewed. The packaged browser runner is the only
qualification path.

## Candidates and qualification

When gold is complete, compare these candidates in one controlled GPU sequence:

1. Qwen3.5 4B Q4_K_M
2. Qwen3.5 2B Q4_K_M
3. Hy-MT2 1.8B Q4_K_M

Each candidate must receive exactly the same ordered target rows, batching,
chapter context/entity memory, prompt, validator, decoding settings, warm-up,
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

No final automated comparison is claimed until the v2 corpus is complete.

Resource identity, timings, quality, and human review must be collected from
the packaged Firefox v2 run and retained with its raw evidence bundle.

## Selection

Qwen3.5 4B Q4_K_M remains the resident production model. Smaller candidates
are not packaged; any future comparison must use the v2 corpus and the same
terminal browser gates.
