# Manga translation pipeline audit

Audit date: 2026-08-08

This record applies to the `image` pipeline and its pre-rebuild local baseline.
It does not qualify document extraction or the light-novel reader. Contract
names in current code follow the generic image/document API.

## Verdict

The former pipeline had several structural correctness and latency failures,
not just tuning problems. It could lose text before OCR, let a visual model
rewrite the OCR transcript, omit admitted sound effects, leak role labels into
Chinese, start later pages with stale chapter context, deadlock a CUDA lane
after cancellation, and serialize page admission in Firefox. Those paths have
been removed.

The current pipeline is materially better and reaches a smooth warm steady
state on the out-of-sample chapter: every translated page from page 8 onward
was ready before a simulated reader reached it. It does not yet satisfy the
project's stricter cold-start promise. Five early translated pages still caused
more than one second of waiting, so the honest verdict is **substantially
improved, but not yet zero-wait from the beginning of a cold chapter**.

## Scope and evidence

The audit followed the complete packaged-Firefox path: discovery and admission,
native messaging, upload validation, detector proposal flow, OCR, visual role
classification, faithful translation, HSK handling, cleanup, terminal event
publication, patch-first rendering, and chapter scheduling.

The out-of-sample input was Webtoon's *Chaotic*, chapter 1, captured as 23
independent 800 by 1280 images. It was not used to design content rules. The
capture is content-addressed under `local-corpus/real-reader-v2/objects`, and
its integrity record is `temp/chaotic-holdout-capture-result.json`.
The final observational output is
`.cache/chaotic-holdout-probe-30/summary.json`.

This is an **unannotated capture probe**, not release-quality translation
evidence. It supports timing, terminal-state, rendering, source-preservation,
and manual-output observations. It cannot establish detector recall or
translation accuracy percentages without exhaustive human annotations.

## Structural findings and corrections

| Finding | Failure mode | Structural correction |
| --- | --- | --- |
| Comic-detector proposals were the only route into OCR | Unballooned captions, narration, and detector misses could disappear without a terminal region | Union independent comic-detector and PP-OCR detector proposals; retain PP-OCR line geometry and spatially deduplicate the union |
| OCR recovery was page/group coupled | One bad glyph or alternate view could erase neighboring valid dialogue | Run original and contrast-normalized views and decide consensus independently per detected line |
| Rejected OCR could vanish | The reader could not distinguish “no text” from “text failed” | Emit a terminal source-preserving unreadable region for rejected proposals |
| The multimodal model owned transcript, role, and translation | It could alter OCR text, hallucinate content, or discard valid siblings when one record was malformed | Make OCR immutable; restrict visual output to `story`, `sfx`, `furniture`, `artwork`, and optional continuation; parse numbered records independently |
| Role labels were placed beside source text | Labels such as “sound effect” leaked into rendered Chinese | Carry roles in a separate position index; reject role-label output deterministically |
| Translation policy had optional SFX and source-language-name branches | Valid SFX could be silently excluded and Latin names could bypass the Chinese-only invariant | Remove both policies from browser and shared translator contracts; every admitted SFX is translated and every name is rendered in Chinese |
| Faithful meaning and HSK simplification happened in one visual generation | Vocabulary constraints could cause omissions before a complete semantic reference existed | Translate admitted OCR with a text-only faithful pass; natural mode publishes it, while strict mode alone performs one bounded HSK rewrite/repair |
| Page jobs captured chapter context at job start | A page completing OCR later could still miss translations that finished meanwhile | Refresh canonical preceding context when language work is dispatched |
| Cleanup was repeated and serialized with language work | Redundant model work increased page latency and blocked translation | Run one page-level cleanup task and overlap it with text translation on separate Vision and Language lanes |
| Dropping a queued CUDA acquire left its waiter at the lane head | Cancellation could permanently deadlock all later GPU work | Use scoped waiter registration whose drop removes the waiter; cover the abort race with a deterministic scheduler test |
| Firefox admitted pages serially and restarted preempted work | Readers repeatedly reached untranslated pages and paid duplicate inference cost | Admit two pages immediately within a two-page pixel budget, continuously reprioritize pending work, and never cancel/restart an admitted page for scrolling |
| Rendering could expose Chinese before cleanup or silently accept unreadable fit | English and Chinese could overlap, or text could overflow | Store/decode/install the verified patch before selectable text; make fit failure terminal and source-preserving |
| Persistent cache reads scanned broadly and identities omitted behavior changes | Cache lookup could become chapter-size work or replay stale semantics | Open the exact SHA-keyed entry and include model, prompt, validator, control-data, pipeline, context, role, level, and learning-mode identities |

No chapter title, source phrase, credit word, or other content allowlist was
added. The remaining decisions are based on geometry, confidence, model
contracts, and deterministic output invariants.

## Out-of-sample timing

The same 23-page capture and six-second-per-page reader simulation were used
for the baseline and final implementation.

| Metric | Baseline | Final measured run | Change |
| --- | ---: | ---: | ---: |
| Chapter completion | 132,135 ms | 113,784 ms | 13.9% faster |
| First visible translated text | 16,053 ms | 9,139 ms | 43.1% faster |
| Translated pages ready before viewport | 3/19 (15.8%) | 13/18 (72.2%) | +56.4 percentage points |
| p95/max simulated reader stall | 12,112 ms | 4,190 ms | 65.4% lower |
| Pages with more than one second of waiting | not separately recorded | 2, 3, 4, 5, and 7 | strict gate still fails |

The final run completed all 23 jobs and published 59 terminal regions: 45
translated and 14 source-preserving. Patch-before-text, one terminal DOM commit
per region, readable fit, route integrity, wrapper count, and packaged resource
identity checks all passed. The sole failed assertion was viewport readiness,
whose required threshold is at least 95% of translated pages ready in advance,
p95 stall at most 500 ms, and no stall above one second.

## Manual output observations

- The cover, title page, and end promotion were preserved rather than treated
  as dialogue.
- Cross-page text retained its meaning: “OH...A LITTLE-” became “哦…有点…”,
  followed by “HARDHEADED” as “固执的…”.
- Sound effects reached translation: `KICK` became “踢”, `STEP` became “一步”,
  and OCR's `BAML!` became “巴姆！”. These prove the SFX path is no longer
  skipped, although “一步” is not especially natural onomatopoeia and the
  OCR-derived “巴姆” is weaker than “砰”.
- No visual role label appeared in rendered output.
- One device-control `Play` on page 8 was still classified as story text and
  translated to “播放”. This is the clearest remaining visual-role false
  positive; no chapter-specific suppression rule was added.

## Remaining risks

1. Cold model start and the first detector/vision queue remain too slow for the
   strict real-time gate. The warm pipeline is smooth; chapter entry is not.
2. Visual role classification can still confuse embedded interface furniture
   with story text, as shown by `Play`.
3. OCR and literal faithful translation do not always produce natural Chinese
   sound symbolism. SFX quality needs annotated examples, not a wordlist.
4. Strict HSK mode can only preserve a proper name that is already correctly
   represented in the faithful Chinese reference. Its deterministic vocabulary
   validator has no independent entity oracle.
5. The holdout has no exhaustive text/role/translation annotations. Release
   claims still require the completed real-reader v2 corpus and blinded human
   review described in [Real-reader v2](real-reader-v2.md) and
   [Model benchmark](model-benchmark.md).

## Verification

The implementation is covered by Rust suites for the shared prompt protocol,
application translator, HSK data/control, companion scheduler/pipeline/server,
and cache behavior; TypeScript type checking and Firefox unit tests; corpus and
structural-gate self-tests; the exact CUDA release build; packaged extension
creation; and the packaged-Firefox capture probe above.
