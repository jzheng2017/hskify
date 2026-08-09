//! Shared compact prompt protocol for faithful Chinese HSK realization.
//!
//! Product and benchmark callers deliberately use these same builders. The
//! model sees only temporary one-based positions; application IDs remain with
//! the caller.

use std::fmt::Write as _;

pub const DIRECT_HSK_PROMPT_REVISION: &str = "ordered-span-hsk-realization-v2-2026-08-09";

/// Canonical protocol description whose SHA-256 is
/// [`DIRECT_HSK_PROMPT_HASH`].
///
/// Keep this material synchronized with the builders below. The unit test pins
/// the digest so a prompt-semantic change cannot silently reuse cache entries
/// or benchmark evidence.
pub const DIRECT_HSK_PROMPT_FINGERPRINT_MATERIAL: &str = r#"ordered-span-hsk-realization-v2-2026-08-09
input=generic ordered source spans paired with complete faithful Chinese references; image spans alone may carry measured character and line budgets
provenance=DOM text is authoritative and must not be corrected; OCR spans alone receive a bounded obvious-recognition-error correction instruction
chapter-context=daemon-owned preceding Chinese and bounded neighboring English are reference only; preserve canonical sourceIndex and itemOrder and never emit context-only spans
names=render every name in Chinese, using the faithful reference as authority; no source-language name preservation mode exists
translation=realize each faithful Chinese reference at the requested HSK level; preserve its complete meaning, participant roles, agency, attachment, causality, modality, quantities, negation, tone, ambiguity, and numeric values while simplifying vocabulary and grammar
natural-learning=target 90% coverage for levels 1-3, 93% for level 4, and 95% for levels 5-6; retain only indispensable above-level terms and expose them as teaching metadata
strict-learning=avoid every above-level term unless the faithful Chinese name form makes it unavoidable
layout=honor maximum Chinese characters and line count only when image constraints are supplied
output=one terminal numbered Chinese line per input span, no labels, explanations, markup, IDs, source-language leakage, or provisional text
decoding=deterministic greedy generation with a source-sized output budget capped at 1024 tokens; context-aware packing shrinks the batch before exceeding the resident model context
repair=the same ordered context is supplied to one bounded terminal repair; rejected candidates stay hidden until repair or a source-preserving terminal result"#;

// Filled from the exact UTF-8 bytes of
// DIRECT_HSK_PROMPT_FINGERPRINT_MATERIAL.
pub const DIRECT_HSK_PROMPT_HASH: &str =
    "sha256:0204bc6cd0c4a4a4d5dbf5356a1fdf53da6d4893e599d1023c574097f1d45d09";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectHskLearningMode {
    Natural,
    Strict,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectSourceProvenance {
    Dom,
    Ocr,
}

/// Shared identity of numbered-line parsing and deterministic preservation
/// validation used by production and release evidence.
pub const DIRECT_HSK_VALIDATOR_FINGERPRINT_MATERIAL: &str = "numbered-output-v4|ordered-span-count-and-id-coverage|source-language-and-Chinese-only-gates|numeric-and-critical-term-preservation|kind-label-rejection|hsk-natural-teaching-metadata-and-strict-vocabulary|optional-image-layout-character-and-line-budget|terminal-only-publication-v3|repair-evidence-is-item-local-and-contextual";

pub const DIRECT_HSK_VALIDATOR_HASH: &str =
    "sha256:729765101be1dcde612cec18613e8a01e8a1b70ae7898a74928ee49d8cfe4e68";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectHskContext<'a> {
    pub source_english: &'a str,
    pub chinese: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectHskSource<'a> {
    pub source_english: &'a str,
    pub faithful_chinese: &'a str,
}

#[must_use]
pub fn primary_system_prompt_for_source(
    level: u8,
    count: usize,
    learning_mode: DirectHskLearningMode,
    provenance: DirectSourceProvenance,
) -> String {
    let level_style = level_style_instruction(level);
    let semantic_admission_instruction = "Source registration is already complete: every supplied \
        span is admitted content that must receive a translation. Do not classify, exclude, skip, \
        merge, or reorder any span.";
    let name_instruction = "Treat person, place, organization, and other proper names as names: never translate \
        their dictionary meaning. Preserve the Chinese form in the faithful reference, or use an established \
        Chinese name when certain and otherwise a phonetic Chinese transliteration. Keep it consistent with \
        preceding context. Never emit Latin name spellings.";
    let learning_instruction = match learning_mode {
        DirectHskLearningMode::Natural => natural_learning_instruction(level),
        DirectHskLearningMode::Strict => {
            "Use strict HSK policy. Rewrite every avoidable above-level word and grammar pattern with \
            level-appropriate language even when the result is less elegant. Only protected proper \
            names and exact required glossary forms may remain outside the selected level."
                .to_owned()
        }
    };
    let provenance_instruction = match provenance {
        DirectSourceProvenance::Dom => {
            "The English was extracted directly from the document DOM and is authoritative; preserve it exactly as the source and do not rewrite or correct it before translation."
        }
        DirectSourceProvenance::Ocr => {
            "The English came from OCR and can contain minor recognition errors in letters, spacing, or punctuation. Silently correct only an obvious OCR error when grammar and neighboring context make the intended English clear; do not invent content when it is genuinely ambiguous."
        }
    };
    format!(
        "Realize the {count} numbered faithful Chinese references in their supplied source order at the requested HSK level. \
{semantic_admission_instruction} {provenance_instruction} The faithful Chinese reference is the semantic authority: preserve all of its meaning, add nothing, and omit nothing; use the English only to resolve names and source structure. Rewrite each reference into concise, \
natural Simplified Chinese for a reader targeting cumulative HSK 2.0 level {level}. \
Use the supplied preceding translations and neighboring numbered spans to resolve pronouns, omitted \
subjects, ellipsis, and sentences split across source spans. Adjacent spans are context, not extra \
content: each numbered output must contain only the meaning carried by its own source span, while using \
the surrounding sequence to make that portion coherent. Preserve a genuinely standalone fragment as a \
fragment; when several spans form one sentence, translate each span as its corresponding portion of \
that sentence without duplicating or inventing meaning. Actively rewrite vocabulary, grammar, clause structure, and idioms to \
        suit the requested level—not vocabulary alone. {level_style} {learning_instruction} Prefer the simplest natural wording \
        that preserves the complete meaning; do not keep advanced grammar merely because its vocabulary \
        passes the HSK list. \
Preserve complete meaning: every clause and detail; speaker, addressee, and participant roles; \
who acts on whom and whether agency is intentional or accidental; attachment, cause and result; \
modality, certainty, and conditions; quantities and comparisons; negation; question intent; tone \
and humour; relationships and pronoun referents; ambiguity as resolved by preceding context, or \
        the ambiguity itself when unresolved; and self-corrections in their original order. Preserve \
        numeric values. When a numbered line has an image layout budget, stay within its maximum \
        Chinese-character and line counts, choosing a concise equivalent before allowing overflow. {name_instruction} Your response must start \
with `1\t` and contain exactly {count} non-empty lines numbered 1 through {count} in order. On \
        every line, write the position, one tab, and only its Simplified Chinese translation. \
Do not write headings, labels, explanations, Markdown, JSON, or application IDs."
    )
}

fn level_style_instruction(level: u8) -> &'static str {
    match level {
        1 | 2 => {
            "Use basic everyday words and short, direct subject-verb-object clauses. Make referents \
            explicit when natural. Prefer two simple clauses over one nested clause. Avoid idioms, \
            literary or formal wording, nominalization, dense modifiers, and avoidable 把/被 or passive \
            constructions."
        }
        3 | 4 => {
            "Use common conversational words and familiar connectors. Moderate compound sentences are \
            fine, but replace advanced idioms, formal synonyms, nominalization, and deeply nested clauses \
            with clearer everyday phrasing."
        }
        _ => {
            "Natural advanced grammar and precise vocabulary are allowed, while concise everyday wording \
            is still preferred when equally accurate."
        }
    }
}

fn natural_learning_instruction(level: u8) -> String {
    let (coverage, term_limit) = match level {
        1..=3 => (90, 1),
        4 => (93, 2),
        5 => (95, 2),
        _ => (95, 3),
    };
    format!(
        "Use the simplify-preserve-teach policy. First simplify advanced vocabulary and grammar \
        wherever an everyday expression preserves the complete meaning naturally. Target at least \
        {coverage}% level-appropriate lexical occurrences and retain no more than {term_limit} \
        above-level occurrence in this complete line. Retain one only when paraphrasing it would \
        become awkward, childish, repetitive, or materially less precise. Prefer a useful recurring \
        content word over a decorative literary synonym. The application will identify and teach \
        retained terms; do not add explanations or markup."
    )
}

#[must_use]
pub fn primary_user_prompt(
    context: &[DirectHskContext<'_>],
    sources: &[DirectHskSource<'_>],
) -> String {
    let mut prompt = String::new();
    if !context.is_empty() {
        prompt.push_str("Previous translations (reference only; do not output):\n");
        prompt.push_str(&context_budget_text(context));
        prompt.push('\n');
    }
    prompt.push_str("Faithful Chinese references (semantic authority):\n");
    for (index, source) in sources.iter().enumerate() {
        writeln!(
            &mut prompt,
            "{}\t{}",
            index + 1,
            compact(source.faithful_chinese)
        )
        .expect("writing to String cannot fail");
    }
    prompt.push_str("\nEnglish source lines (name and structure reference):\n");
    for (index, source) in sources.iter().enumerate() {
        let source = compact(source.source_english);
        writeln!(&mut prompt, "{}\t{source}", index + 1).expect("writing to String cannot fail");
    }
    prompt
}

#[must_use]
pub fn repair_system_prompt_for_source(
    level: u8,
    count: usize,
    learning_mode: DirectHskLearningMode,
    provenance: DirectSourceProvenance,
) -> String {
    let level_style = level_style_instruction(level);
    let name_instruction = "Never translate proper names by dictionary meaning. Preserve their Chinese form from the \
        faithful reference, or use an established Chinese form when certain and otherwise a phonetic Chinese \
        transliteration. Never emit Latin name spellings.";
    let learning_instruction = match learning_mode {
        DirectHskLearningMode::Natural => natural_learning_instruction(level),
        DirectHskLearningMode::Strict => {
            "Replace every listed above-level term and grammar pattern with level-appropriate wording. \
            Treat every exact term in the Validator avoid-list as a forbidden Chinese substring: \
            check the completed answer and emit none of them. A name form required to preserve the \
            faithful reference is the only exception."
                .to_owned()
        }
    };
    let scope_instruction = if count == 1 {
        "Repair this one English-to-Simplified-Chinese translation".to_owned()
    } else {
        format!(
            "Repair each of the {count} numbered English-to-Simplified-Chinese translations independently"
        )
    };
    let output_instruction = if count == 1 {
        "Return exactly one non-empty line containing only the corrected translation; write no position or tab."
            .to_owned()
    } else {
        format!(
            "Return exactly {count} non-empty lines numbered 1 through {count} in order. On every line write the position, one tab, and only that corrected translation."
        )
    };
    let provenance_instruction = match provenance {
        DirectSourceProvenance::Dom => {
            "The DOM source is authoritative; do not correct or reinterpret its English before translation."
        }
        DirectSourceProvenance::Ocr => {
            "Correct only an obvious OCR recognition error when grammar and context make the intended English clear; do not carry nonsensical recognition fragments into Chinese."
        }
    };
    format!(
        "{scope_instruction} for a reader targeting \
        cumulative HSK 2.0 level {level}. {provenance_instruction} Fix every listed problem. The faithful Chinese reference is the semantic authority: preserve all of its meaning, add nothing, and omit nothing. Actively rewrite vocabulary, grammar, \
        clause structure, and idioms for the requested level—not vocabulary alone. {level_style} {learning_instruction} Preserve \
        every clause and detail, participant roles, \
agency, cause and result, modality, quantities and comparisons, negation, question intent, tone \
        and humour, ambiguity, pronoun referents, self-corrections, Chinese name forms \
        already present in the source, and numeric values. {output_instruction} \
Semantic role classification is already complete; always return a translation. {name_instruction} Write no headings, labels, explanations, Markdown, \
JSON, or application IDs."
    )
}

#[must_use]
pub fn repair_user_prompt(
    source_english: &str,
    faithful_chinese: &str,
    rejected_chinese: Option<&str>,
    problems: &[&str],
) -> String {
    repair_user_prompt_with_constraints(
        source_english,
        faithful_chinese,
        rejected_chinese,
        problems,
        &[],
    )
}

#[must_use]
pub fn repair_user_prompt_with_constraints(
    source_english: &str,
    faithful_chinese: &str,
    rejected_chinese: Option<&str>,
    problems: &[&str],
    avoid_chinese: &[String],
) -> String {
    format!(
        "{}\nAnswer:",
        repair_item_constraints(
            source_english,
            faithful_chinese,
            rejected_chinese,
            problems,
            avoid_chinese,
        )
    )
}

#[must_use]
pub fn repair_item_constraints(
    source_english: &str,
    faithful_chinese: &str,
    rejected_chinese: Option<&str>,
    problems: &[&str],
    avoid_chinese: &[String],
) -> String {
    let rejected = rejected_chinese
        .map(compact)
        .unwrap_or_else(|| "<missing>".to_owned());
    let problems = problems
        .iter()
        .map(|problem| compact(problem))
        .collect::<Vec<_>>()
        .join(" | ");
    let source = compact(source_english);
    let avoid = if avoid_chinese.is_empty() {
        "<none>".to_owned()
    } else {
        avoid_chinese
            .iter()
            .map(|term| compact(term))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "English source: {}\nFaithful Chinese reference: {}\nRejected: {rejected}\nValidator avoid-list: {avoid}\nProblems: {problems}",
        source,
        compact(faithful_chinese),
    )
}

/// Render exactly the context records included in the primary user prompt.
///
/// Callers tokenize this string while enforcing the separate 256-token
/// preceding-context limit.
#[must_use]
pub fn context_budget_text(context: &[DirectHskContext<'_>]) -> String {
    let mut rendered = String::new();
    for item in context {
        writeln!(
            &mut rendered,
            "- {} => {}",
            compact(item.source_english),
            compact(item.chinese)
        )
        .expect("writing to String cannot fail");
    }
    rendered
}

#[must_use]
pub fn compact(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_protocol_has_exact_numbered_input_and_chinese_name_policy() {
        let context = [DirectHskContext {
            source_english: "Earlier English",
            chinese: "之前的中文",
        }];
        let sources = [
            DirectHskSource {
                source_english: "Captain Rowan Finch is here.",
                faithful_chinese: "罗文·芬奇队长在这里。",
            },
            DirectHskSource {
                source_english: "ROWAN FINCH has 2 questions.",
                faithful_chinese: "罗文·芬奇有两个问题。",
            },
        ];

        let system = primary_system_prompt_for_source(
            5,
            sources.len(),
            DirectHskLearningMode::Strict,
            DirectSourceProvenance::Ocr,
        );
        let user = primary_user_prompt(&context, &sources);

        assert!(system.contains("start with `1\t`"));
        assert!(system.contains("exactly 2 non-empty lines"));
        assert!(system.contains("numbered 1 through 2 in order"));
        assert!(system.contains("Source registration is already complete"));
        assert!(system.contains("every supplied span is admitted content"));
        assert!(system.contains("Never emit Latin name spellings"));
        assert!(!system.contains("keep-original"));
        assert!(!system.contains("placeholder"));
        assert_eq!(
            user,
            "Previous translations (reference only; do not output):\n\
- Earlier English => 之前的中文\n\
\n\
Faithful Chinese references (semantic authority):\n\
1\t罗文·芬奇队长在这里。\n\
2\t罗文·芬奇有两个问题。\n\
\n\
English source lines (name and structure reference):\n\
1\tCaptain Rowan Finch is here.\n\
2\tROWAN FINCH has 2 questions.\n"
        );
    }

    #[test]
    fn low_and_high_hsk_levels_receive_materially_different_style_rules() {
        let low = primary_system_prompt_for_source(
            2,
            1,
            DirectHskLearningMode::Strict,
            DirectSourceProvenance::Ocr,
        );
        let high = primary_system_prompt_for_source(
            5,
            1,
            DirectHskLearningMode::Strict,
            DirectSourceProvenance::Ocr,
        );
        let low_repair = repair_system_prompt_for_source(
            2,
            1,
            DirectHskLearningMode::Strict,
            DirectSourceProvenance::Ocr,
        );

        assert!(low.contains("short, direct subject-verb-object clauses"));
        assert!(low.contains("Prefer two simple clauses over one nested clause"));
        assert!(low.contains("Avoid idioms"));
        assert!(low_repair.contains("short, direct subject-verb-object clauses"));
        assert!(high.contains("Natural advanced grammar and precise vocabulary are allowed"));
        assert!(!high.contains("Prefer two simple clauses over one nested clause"));
    }

    #[test]
    fn learning_modes_have_distinct_controlled_vocabulary_policies() {
        let natural = primary_system_prompt_for_source(
            3,
            1,
            DirectHskLearningMode::Natural,
            DirectSourceProvenance::Ocr,
        );
        let strict = primary_system_prompt_for_source(
            3,
            1,
            DirectHskLearningMode::Strict,
            DirectSourceProvenance::Ocr,
        );
        let natural_repair = repair_system_prompt_for_source(
            3,
            1,
            DirectHskLearningMode::Natural,
            DirectSourceProvenance::Ocr,
        );

        assert!(natural.contains("simplify-preserve-teach"));
        assert!(natural.contains("90% level-appropriate lexical occurrences"));
        assert!(natural.contains("no more than 1 above-level occurrence"));
        assert!(strict.contains("strict HSK policy"));
        assert!(strict.contains("Rewrite every avoidable above-level word"));
        assert!(natural_repair.contains("90% level-appropriate lexical occurrences"));
        assert_ne!(natural, strict);
    }

    #[test]
    fn repair_protocol_keeps_source_and_constraints_separate() {
        let prompt = repair_user_prompt_with_constraints(
            "Alice does not have 2 tickets.",
            "爱丽丝没有两张票。",
            Some("她有票。"),
            &["preserve 2", "preserve negation"],
            &["女神".to_owned(), "注定".to_owned()],
        );

        assert_eq!(
            prompt,
            "English source: Alice does not have 2 tickets.\n\
Faithful Chinese reference: 爱丽丝没有两张票。\n\
Rejected: 她有票。\n\
Validator avoid-list: 女神, 注定\n\
Problems: preserve 2 | preserve negation\n\
Answer:"
        );
        assert!(!prompt.contains("Previous translations"));
        assert!(!prompt.lines().any(|line| line.starts_with("1\t")));
    }

    #[test]
    fn prompt_and_validator_sha256_values_match_their_exact_material() {
        let prompt = sha256_hex(DIRECT_HSK_PROMPT_FINGERPRINT_MATERIAL.as_bytes());
        let validator = sha256_hex(DIRECT_HSK_VALIDATOR_FINGERPRINT_MATERIAL.as_bytes());

        assert_eq!(format!("sha256:{prompt}"), DIRECT_HSK_PROMPT_HASH);
        assert_eq!(format!("sha256:{validator}"), DIRECT_HSK_VALIDATOR_HASH);
    }

    fn sha256_hex(input: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a_2f98,
            0x7137_4491,
            0xb5c0_fbcf,
            0xe9b5_dba5,
            0x3956_c25b,
            0x59f1_11f1,
            0x923f_82a4,
            0xab1c_5ed5,
            0xd807_aa98,
            0x1283_5b01,
            0x2431_85be,
            0x550c_7dc3,
            0x72be_5d74,
            0x80de_b1fe,
            0x9bdc_06a7,
            0xc19b_f174,
            0xe49b_69c1,
            0xefbe_4786,
            0x0fc1_9dc6,
            0x240c_a1cc,
            0x2de9_2c6f,
            0x4a74_84aa,
            0x5cb0_a9dc,
            0x76f9_88da,
            0x983e_5152,
            0xa831_c66d,
            0xb003_27c8,
            0xbf59_7fc7,
            0xc6e0_0bf3,
            0xd5a7_9147,
            0x06ca_6351,
            0x1429_2967,
            0x27b7_0a85,
            0x2e1b_2138,
            0x4d2c_6dfc,
            0x5338_0d13,
            0x650a_7354,
            0x766a_0abb,
            0x81c2_c92e,
            0x9272_2c85,
            0xa2bf_e8a1,
            0xa81a_664b,
            0xc24b_8b70,
            0xc76c_51a3,
            0xd192_e819,
            0xd699_0624,
            0xf40e_3585,
            0x106a_a070,
            0x19a4_c116,
            0x1e37_6c08,
            0x2748_774c,
            0x34b0_bcb5,
            0x391c_0cb3,
            0x4ed8_aa4a,
            0x5b9c_ca4f,
            0x682e_6ff3,
            0x748f_82ee,
            0x78a5_636f,
            0x84c8_7814,
            0x8cc7_0208,
            0x90be_fffa,
            0xa450_6ceb,
            0xbef9_a3f7,
            0xc671_78f2,
        ];
        let mut data = input.to_vec();
        let bit_len = (data.len() as u64) * 8;
        data.push(0x80);
        while data.len() % 64 != 56 {
            data.push(0);
        }
        data.extend_from_slice(&bit_len.to_be_bytes());

        let mut state = [
            0x6a09_e667_u32,
            0xbb67_ae85,
            0x3c6e_f372,
            0xa54f_f53a,
            0x510e_527f,
            0x9b05_688c,
            0x1f83_d9ab,
            0x5be0_cd19,
        ];
        for chunk in data.chunks_exact(64) {
            let mut words = [0_u32; 64];
            for (index, bytes) in chunk.chunks_exact(4).enumerate() {
                words[index] = u32::from_be_bytes(bytes.try_into().expect("four bytes"));
            }
            for index in 16..64 {
                let s0 = words[index - 15].rotate_right(7)
                    ^ words[index - 15].rotate_right(18)
                    ^ (words[index - 15] >> 3);
                let s1 = words[index - 2].rotate_right(17)
                    ^ words[index - 2].rotate_right(19)
                    ^ (words[index - 2] >> 10);
                words[index] = words[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(words[index - 7])
                    .wrapping_add(s1);
            }

            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
            for index in 0..64 {
                let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let choice = (e & f) ^ (!e & g);
                let temp1 = h
                    .wrapping_add(sum1)
                    .wrapping_add(choice)
                    .wrapping_add(K[index])
                    .wrapping_add(words[index]);
                let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let majority = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = sum0.wrapping_add(majority);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }
            for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
                *slot = slot.wrapping_add(value);
            }
        }
        state.iter().map(|word| format!("{word:08x}")).collect()
    }
}
