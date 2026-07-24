# Local translation model benchmark

Status: **awaiting execution and fluent-reader review**

This document is the benchmark protocol and audit record. It contains no
invented quality scores and does not select a production model.

## Licence filter

The exact GGUF artifacts in `data/model-packs/manifest.v1.json` were inspected
before download:

| Candidate | Koharu model ID | Size | Licence disposition |
| --- | --- | ---: | --- |
| Qwen3.5 4B Q4_K_M | `qwen3.5-4b` | 2,740,937,888 bytes | Apache-2.0; eligible for benchmark |
| Qwen3.5 2B Q4_K_M | `qwen3.5-2b` | 1,280,835,840 bytes | Apache-2.0; eligible for benchmark |
| Hunyuan-MT 7B Q4_K_M | `hunyuan-mt-7b` | 4,702,111,200 bytes | excluded: inherited Tencent terms exclude use in the EU, UK, and South Korea |
| LFM2.5 1.2B Q4_K_M | `lfm2.5-1.2b-instruct` | 730,895,168 bytes | conditional commercial licence; excluded from an unrestricted default pending legal/product decision |

The Hunyuan exclusion is material for this development environment in the
Netherlands. A model-card metadata label is not treated as overriding the
repository's actual `License.txt`.

## Fixed evaluation protocol

Every eligible candidate must receive the same ordered full-image region
payload and prompt revision. Each run records:

- repository revision, exact filename, SHA-256, and quantization;
- runtime/Koharu revision and compute backend;
- prompt revision and decoding configuration;
- structured-output success and retry count;
- name, pronoun, number, and negation preservation;
- first-pass and corrected HSK compliance;
- wall-clock latency, peak process RAM, and peak VRAM;
- all raw outputs needed to create the blinded packet.

The faithful pass and HSK rewrite are scored separately. No candidate sees a
different source order or additional context.

## Human rubric

At least one fluent Chinese reader scores anonymized outputs from 1 to 5:

1. meaning preservation;
2. natural Chinese manga dialogue;
3. character voice, tone, and humour;
4. names and pronouns;
5. number and negation fidelity.

A score of 1 means unusable or meaning-changing; 3 means understandable with
noticeable editing; 5 means fluent and faithful. Reviewers also mark any
critical meaning reversal. The candidate key remains separate until all score
sheets are complete.

Automated checks may reject a candidate, but they do not choose the winner.
The aggregation report includes count, mean, median, per-sample minimum,
critical failures, structured success rate, HSK success rate, latency, and
memory. Ties are resolved by human meaning/naturalness first, then resource
cost.

## Selection record

| Output | State |
| --- | --- |
| Standard model pack | not selected |
| Low-memory model pack | not selected |
| Hardware thresholds | not established |
| Translation prompt revision | `benchmark-en-zh-v1` (evaluation only) |
| Human score sheets | awaiting |

Until real scores and hardware measurements are recorded, the installable
`packs` array is intentionally empty and installer code must report that no
approved model pack is available.
