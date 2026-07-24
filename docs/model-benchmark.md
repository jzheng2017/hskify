# Local translation model benchmark

Status: **bootstrap execution complete; awaiting representative corpus and
fluent-reader review**

This document is the benchmark protocol and audit record. It contains measured
machine results but no invented human quality scores, and it does not select a
production model.

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

## Bootstrap execution

Both eligible Qwen artifacts were downloaded from the exact revision URLs in
the manifest. Their local byte counts and SHA-256 digests matched before either
was loaded:

| Candidate | Verified bytes | Verified SHA-256 |
| --- | ---: | --- |
| Qwen3.5 4B Q4_K_M | 2,740,937,888 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Qwen3.5 2B Q4_K_M | 1,280,835,840 | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` |

The first frozen request is
`fixtures/golden-evaluation/prompts/benchmark-en-zh-v1.json`. It sends all
three dialogue regions from the synthetic page in one ordered request.
Generation used the pinned Koharu revision, llama.cpp tag `b8935`, a 2,048
token context, a fixed seed of `299792458`, disabled thinking, and greedy
sampling through Koharu's local GGUF executable.

Two defects in that executable were found and fixed before recording results:
the runtime bindings were prepared but not initialized, and `--disable-gpu`
left llama.cpp's auto-offload default enabled. The latter now explicitly sets
zero GPU layers, matching the production `Llm` path.

Machine: Windows, Ryzen-class `zen4` runtime build, 32 GiB RAM, RTX 4080 SUPER
16 GiB. Values below are warm single-process runs of the same short prompt.
Peak process memory came from Windows process counters sampled during the run.
WDDM does not expose per-process VRAM, so GPU memory is only an approximate
global `nvidia-smi` delta over the immediately sampled baseline.

| Candidate/backend | Wall | Load | Decode | Peak working set | Peak private bytes | Approx. GPU delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 2B / CUDA | 1.91 s | 0.86 s | 67 tokens at 207.38 t/s | 1,938,345,984 | 3,924,303,872 | 2,043 MiB |
| Qwen3.5 4B / CUDA | 3.26 s | 2.03 s | 68 tokens at 131.54 t/s | 3,403,534,336 | 5,472,055,296 | 3,502 MiB |
| Qwen3.5 2B / CPU-only | 4.46 s | 1.40 s | 68 tokens at 28.65 t/s | 1,565,274,112 | 1,736,404,992 | not used |
| Qwen3.5 4B / CPU-only | 7.80 s | 1.75 s | 68 tokens at 13.00 t/s | 3,077,185,536 | 1,914,425,344 | not used |

Both candidates returned valid JSON with every requested region ID exactly
once and no unknown IDs. The outputs are stored only under randomized labels
in `fixtures/golden-evaluation/blinded-review`; the identity key remains
ignored until a fluent reader completes the score sheet.

These numbers are an engineering smoke result, not pack thresholds. The
bootstrap set has one synthetic page, no names, no numbers, no negation, and no
HSK rewrite pass. The raw harness also tests prompt-only JSON compliance; Gate
4 still requires llama.cpp-compatible grammar/schema-constrained decoding.

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
| Bootstrap machine execution | complete for eligible 2B and 4B artifacts |
| Blinded bootstrap outputs | generated under randomized labels |
| Human score sheets | awaiting a real fluent Chinese reader |

Until real scores and hardware measurements are recorded, the installable
`packs` array is intentionally empty and installer code must report that no
approved model pack is available.
