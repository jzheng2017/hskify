//! Shared compact prompt protocol for direct English-to-HSK Chinese translation.
//!
//! Product and benchmark callers deliberately use these same builders. The
//! model sees only temporary one-based positions; application IDs remain with
//! the caller.

use std::fmt::Write as _;

pub const DIRECT_HSK_PROMPT_REVISION: &str =
    "direct-hsk-en-zh-semantic-disposition-inline-names-v29-2026-07-27";

/// Canonical protocol description whose SHA-256 is
/// [`DIRECT_HSK_PROMPT_HASH`].
///
/// Keep this material synchronized with the builders below. The unit test pins
/// the digest so a prompt-semantic change cannot silently reuse cache entries
/// or benchmark evidence.
pub const DIRECT_HSK_PROMPT_FINGERPRINT_MATERIAL: &str = "\
    direct-hsk-en-zh-semantic-disposition-inline-names-v29-2026-07-27
primary-system=classify and translate each of exactly {count} numbered OCR sources independently; output [NON-STORY] only when the complete source is unrelated page furniture such as a publisher/site credit, watermark, advertisement, or navigation label; never exclude dialogue, narration, thoughts, captions, signs, letters, titles within the story, names, roles, sentence fragments, or stylized emphasis; only supplied preceding translations are reference; translate only meaning explicitly present in that line; preserve sentence and styled-emphasis fragments as fragments; never complete a fragment from another numbered source; response starts 1+tab; exactly {count} non-empty ordered lines 1..{count}; position+one-tab+Chinese-or-[NON-STORY]; actively rewrite vocabulary, grammar, clause structure, and idioms for HSK2.0 level {level} according to a level-specific style rule, while preserving meaning; at levels 1-2 use basic everyday words, short subject-verb-object clauses, explicit referents, and avoid idioms, literary/formal wording, nominalization, nested clauses, and avoidable passive/把/被 constructions; at levels 3-4 allow common compound sentences and familiar connectors but replace advanced idioms, formal synonyms, and dense embedding; at levels 5-6 allow natural advanced grammar and precise vocabulary; preserve every clause/detail, speaker/addressee/participant roles, agency, attachment, causality, modality/certainty/condition, quantities/comparisons, negation, question intent, tone/humour, context-resolved ambiguity, unresolved ambiguity, relationships, pronoun referents, self-corrections in order, and numeric values; name handling is part of this same contextual translation decision: in keep-original mode preserve only semantic proper names that identify a particular entity, wrap every retained name as ⟦exact boundary-aligned source spelling⟧, and translate all unmarked English including descriptions and roles; in Chinese-name mode use approved/established/phonetic Chinese forms without dictionary-meaning translation; follow optional line-local approved-glossary notes only for matching position; no headings/labels/explanation/markdown/json/IDs
primary-user=optional readable preceding-translations heading and dash source=>Chinese reference lines; optional line-specific-note heading with one-based position-tagged notes generated solely from application-supplied approved glossary entries whose exact ASCII English forms occur on that source line, selecting longest nonoverlapping matches; blank separators; English-lines heading; one-based position+tab+untouched-English
repair-system=one accurate natural HSK2.0 level {level} Chinese line; actively apply the same level-specific vocabulary, grammar, clause-structure, and idiom rule as primary generation; fix all problems; apply the same contextual proper-name decision and exact-source inline name markup in keep-original mode, or Chinese name forms without dictionary-meaning translation in Chinese-name mode; preserve every clause/detail, roles, agency, causality, modality, quantities/comparisons, negation, question intent, tone/humour, ambiguity, self-corrections, approved-substituted-glossary-forms, and numeric values; no position/label/tab/explanation/markdown/json/ID
repair-user=Source/Rejected/Problems/Answer readable fields; matching approved Chinese glossary forms substituted directly in Source
decoding=greedy-unpenalized
batch=3..6-production-max6
context=max6-preceding-utterances-and-max256-context-tokens
repair=at-most-one-per-rejected-bubble-no-context-no-primary-retry";

// Filled from the exact UTF-8 bytes of
// DIRECT_HSK_PROMPT_FINGERPRINT_MATERIAL.
pub const DIRECT_HSK_PROMPT_HASH: &str =
    "sha256:55190968a85b2619aca2d48087d9a52e22c48a881aee959aa69cbe25904dc558";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectHskNameStyle {
    KeepOriginal,
    Chinese,
}

/// Shared identity of numbered-line parsing and deterministic preservation
/// validation used by production and release evidence.
pub const DIRECT_HSK_VALIDATOR_FINGERPRINT_MATERIAL: &str = "numbered-tab-or-space-parser|\
fullwidth-ascii|strict-primary-ascii-repair-trigger|source-guided-final-numeric-value-validation-v3-\
    ignore-letter-adjacent-ocr-digits-v1|deterministic-question-punctuation-v1|\
    semantic-non-story-disposition-v1|inline-exact-source-name-markup-v1|unmarked-latin-rejected-v1|names|question-fragment-aware-v2|\
excessive-han-expansion-v1";
pub const DIRECT_HSK_VALIDATOR_HASH: &str =
    "sha256:1c23256323cef94f965c4d1c093392a3515f61249eba5d79cd73aa6689a4a1b1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectHskContext<'a> {
    pub source_english: &'a str,
    pub chinese: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectHskName<'a> {
    pub source_english: &'a str,
    pub chinese: &'a str,
}

#[must_use]
pub fn primary_system_prompt(level: u8, count: usize) -> String {
    primary_system_prompt_with_name_style(level, count, DirectHskNameStyle::Chinese)
}

#[must_use]
pub fn primary_system_prompt_with_name_style(
    level: u8,
    count: usize,
    name_style: DirectHskNameStyle,
) -> String {
    let level_style = level_style_instruction(level);
    let name_instruction = match name_style {
        DirectHskNameStyle::KeepOriginal => {
            "As part of translating the complete sentence, decide from meaning and context which \
            source spans are proper names: lexicalized identifiers for particular people, places, \
            organizations, named events, or unique entities. A capitalized description, relationship, \
            occupation, rank, title, species, ordinary noun phrase, or clause is not a proper name and \
            must be translated. Keep each proper name exactly as written in the English source and wrap \
            it as `⟦exact source spelling⟧`. The text inside each marker pair must be one exact, \
            boundary-aligned source span. Do not put markers around translated text. Translate every \
            English word outside those marked names."
        }
        DirectHskNameStyle::Chinese => {
            "Treat person, place, organization, and other proper names as names: never translate \
            their dictionary meaning. Use an approved glossary form when supplied; otherwise use an \
            established Chinese name only when certain, or a phonetic Chinese transliteration, and \
            keep it consistent with preceding context."
        }
    };
    format!(
        "Classify and translate each of the {count} numbered OCR source lines independently. \
For a complete source that is unrelated page furniture—a publisher or site credit, watermark, \
advertisement, or navigation label—output exactly `[NON-STORY]`. Never use `[NON-STORY]` for \
dialogue, narration, thoughts, captions, signs, letters, titles within the story, proper names, \
roles, sentence fragments, or stylized emphasis. Translate every story source into concise, \
natural Simplified Chinese for a reader targeting cumulative HSK 2.0 level {level}. \
Use only supplied preceding translations as reference; do not use another numbered source to \
change a line's meaning. Translate only meaning explicitly present in that numbered line. If it is a \
sentence fragment or styled emphasis fragment, keep it as a fragment and never complete it from \
        another numbered source. Actively rewrite vocabulary, grammar, clause structure, and idioms to \
        suit the requested level—not vocabulary alone. {level_style} Prefer the simplest natural wording \
        that preserves the complete meaning; do not keep advanced grammar merely because its vocabulary \
        passes the HSK list. \
Preserve complete meaning: every clause and detail; speaker, addressee, and participant roles; \
who acts on whom and whether agency is intentional or accidental; attachment, cause and result; \
modality, certainty, and conditions; quantities and comparisons; negation; question intent; tone \
and humour; relationships and pronoun referents; ambiguity as resolved by preceding context, or \
the ambiguity itself when unresolved; and self-corrections in their original order. Preserve \
        numeric values. {name_instruction} If line-specific approved-glossary notes appear, follow only notes carrying that \
line's position; never apply a note to another line or output the notes. Your response must start \
with `1\t` and contain exactly {count} non-empty lines numbered 1 through {count} in order. On \
        every line, write the position, one tab, and only its Simplified Chinese translation plus \
        the required temporary name markers when keep-original mode is active, or the exact \
        `[NON-STORY]` disposition. \
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

#[must_use]
pub fn primary_user_prompt(
    context: &[DirectHskContext<'_>],
    names: &[DirectHskName<'_>],
    sources: &[&str],
) -> String {
    let mut prompt = String::new();
    if !context.is_empty() {
        prompt.push_str("Previous translations (reference only; do not output):\n");
        prompt.push_str(&context_budget_text(context));
        prompt.push('\n');
    }
    let notes = sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| {
            let notes = line_specific_notes(source, names);
            (!notes.is_empty()).then(|| format!("- {}: {}", index + 1, notes.join("; ")))
        })
        .collect::<Vec<_>>();
    if !notes.is_empty() {
        prompt.push_str("Line-specific translation notes (reference only; do not output):\n");
        for note in notes {
            writeln!(&mut prompt, "{note}").expect("writing to String cannot fail");
        }
        prompt.push('\n');
    }
    prompt.push_str("English lines:\n");
    for (index, source) in sources.iter().enumerate() {
        writeln!(&mut prompt, "{}\t{}", index + 1, compact(source))
            .expect("writing to String cannot fail");
    }
    prompt
}

fn line_specific_notes(source: &str, names: &[DirectHskName<'_>]) -> Vec<String> {
    let compact_source = compact(source);
    line_specific_names(&compact_source, names)
        .into_iter()
        .map(|name| {
            format!(
                "approved glossary \"{}\" => \"{}\" (use this exact form)",
                compact(name.source_english),
                compact(name.chinese)
            )
        })
        .collect()
}

fn line_specific_names<'a>(source: &str, names: &[DirectHskName<'a>]) -> Vec<DirectHskName<'a>> {
    let lower = source.to_ascii_lowercase();
    let mut occurrences = names
        .iter()
        .filter(|name| !name.source_english.is_empty() && name.source_english.is_ascii())
        .flat_map(|name| {
            let needle = name.source_english.to_ascii_lowercase();
            let needle_len = needle.len();
            lower
                .match_indices(&needle)
                .map(move |(start, _)| (start, start + needle_len, *name))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    occurrences.sort_by(|left, right| {
        (right.1 - right.0)
            .cmp(&(left.1 - left.0))
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.2.source_english.cmp(right.2.source_english))
    });

    let mut occupied = Vec::<(usize, usize)>::new();
    let mut selected = Vec::<(usize, DirectHskName<'a>)>::new();
    for (start, end, name) in occurrences {
        if occupied
            .iter()
            .any(|(used_start, used_end)| start < *used_end && *used_start < end)
        {
            continue;
        }
        occupied.push((start, end));
        if !selected
            .iter()
            .any(|(_, existing)| existing.source_english == name.source_english)
        {
            selected.push((start, name));
        }
    }
    selected.sort_by_key(|(start, _)| *start);
    selected.into_iter().map(|(_, name)| name).collect()
}

#[must_use]
pub fn repair_system_prompt(level: u8) -> String {
    repair_system_prompt_with_name_style(level, DirectHskNameStyle::Chinese)
}

#[must_use]
pub fn repair_system_prompt_with_name_style(level: u8, name_style: DirectHskNameStyle) -> String {
    let level_style = level_style_instruction(level);
    let name_instruction = match name_style {
        DirectHskNameStyle::KeepOriginal => {
            "Decide proper names from the complete source meaning, not capitalization. A proper name \
            is a lexicalized identifier for a particular entity; descriptions, relationships, \
            occupations, ranks, titles, species, noun phrases, and clauses must be translated. Keep \
            each proper name exactly as written in the source and wrap it as `⟦exact source \
            spelling⟧`. Each marked value must be one exact, boundary-aligned source span. Translate \
            every English word outside marked names."
        }
        DirectHskNameStyle::Chinese => {
            "Never translate proper names by dictionary meaning; use approved or established forms \
            and otherwise a phonetic Chinese transliteration."
        }
    };
    format!(
        "Repair this one English-to-Simplified-Chinese translation for a reader targeting \
        cumulative HSK 2.0 level {level}. Fix every listed problem. Actively rewrite vocabulary, grammar, \
        clause structure, and idioms for the requested level—not vocabulary alone. {level_style} Preserve \
        every clause and detail, participant roles, \
agency, cause and result, modality, quantities and comparisons, negation, question intent, tone \
        and humour, ambiguity, pronoun referents, self-corrections, approved glossary forms \
        already present in the source, and numeric values. Return exactly one non-empty line containing \
only the corrected translation. {name_instruction} Write no position, label, tab, explanation, Markdown, \
JSON, or application ID."
    )
}

#[must_use]
pub fn repair_user_prompt(
    source_english: &str,
    rejected_chinese: Option<&str>,
    problems: &[&str],
    names: &[DirectHskName<'_>],
) -> String {
    let rejected = rejected_chinese
        .map(compact)
        .unwrap_or_else(|| "<missing>".to_owned());
    let problems = problems
        .iter()
        .map(|problem| compact(problem))
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        "Source: {}\nRejected: {rejected}\nProblems: {problems}\nAnswer:",
        substitute_approved_names(source_english, names)
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
pub fn substitute_approved_names(source: &str, names: &[DirectHskName<'_>]) -> String {
    let mut ordered = names.to_vec();
    ordered.sort_by(|left, right| {
        right
            .source_english
            .len()
            .cmp(&left.source_english.len())
            .then_with(|| left.source_english.cmp(right.source_english))
    });

    ordered.into_iter().fold(compact(source), |text, name| {
        replace_ascii_case_insensitive(&text, name.source_english, name.chinese)
    })
}

#[must_use]
pub fn compact(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn replace_ascii_case_insensitive(haystack: &str, needle: &str, replacement: &str) -> String {
    if needle.is_empty() || !needle.is_ascii() {
        return haystack.to_owned();
    }
    let lower_haystack = haystack.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    let mut cursor = 0;
    let mut output = String::with_capacity(haystack.len());
    while let Some(relative) = lower_haystack[cursor..].find(&lower_needle) {
        let start = cursor + relative;
        let end = start + needle.len();
        output.push_str(&haystack[cursor..start]);
        output.push_str(replacement);
        cursor = end;
    }
    output.push_str(&haystack[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_protocol_starts_at_one_and_has_exact_readable_input_lines() {
        let context = [DirectHskContext {
            source_english: "Earlier English",
            chinese: "之前的中文",
        }];
        let names = [
            DirectHskName {
                source_english: "Captain Rowan Finch",
                chinese: "罗文·芬奇队长",
            },
            DirectHskName {
                source_english: "Rowan Finch",
                chinese: "罗文·芬奇",
            },
        ];
        let sources = [
            "Captain Rowan Finch is here.",
            "ROWAN FINCH has 2 questions.",
        ];

        let system = primary_system_prompt(5, sources.len());
        let user = primary_user_prompt(&context, &names, &sources);

        assert!(system.contains("start with `1\t`"));
        assert!(system.contains("exactly 2 non-empty lines"));
        assert!(system.contains("numbered 1 through 2 in order"));
        assert!(system.contains("Classify and translate"));
        assert!(system.contains("publisher or site credit"));
        assert!(system.contains("[NON-STORY]"));
        assert!(system.contains("Never use `[NON-STORY]` for"));
        assert_eq!(
            user,
            "Previous translations (reference only; do not output):\n\
- Earlier English => 之前的中文\n\
\n\
Line-specific translation notes (reference only; do not output):\n\
- 1: approved glossary \"Captain Rowan Finch\" => \"罗文·芬奇队长\" (use this exact form)\n\
- 2: approved glossary \"Rowan Finch\" => \"罗文·芬奇\" (use this exact form)\n\
\n\
English lines:\n\
1\tCaptain Rowan Finch is here.\n\
2\tROWAN FINCH has 2 questions.\n"
        );
        assert!(!user.contains("INPUT\t"));
        assert!(!user.contains("\nC\t"));
        assert!(!user.contains("\nN\t"));
        assert!(!user.contains("Approved Chinese names"));
        assert!(system.contains("source lines independently"));
        assert!(system.contains("only supplied preceding translations as reference"));
        assert!(system.contains("keep it as a fragment"));
        assert!(system.contains("never complete it from another numbered source"));
        assert!(system.contains("simplest natural wording"));
        assert!(system.contains("speaker, addressee, and participant roles"));
        assert!(system.contains("agency is intentional or accidental"));
        assert!(system.contains("quantities and comparisons"));
        assert!(system.contains("tone and humour"));
        assert!(system.contains("self-corrections in their original order"));
        assert!(system.contains("never translate their dictionary meaning"));
    }

    #[test]
    fn low_and_high_hsk_levels_receive_materially_different_style_rules() {
        let low = primary_system_prompt(2, 1);
        let high = primary_system_prompt(5, 1);
        let low_repair = repair_system_prompt(2);

        assert!(low.contains("short, direct subject-verb-object clauses"));
        assert!(low.contains("Prefer two simple clauses over one nested clause"));
        assert!(low.contains("Avoid idioms"));
        assert!(low_repair.contains("short, direct subject-verb-object clauses"));
        assert!(high.contains("Natural advanced grammar and precise vocabulary are allowed"));
        assert!(!high.contains("Prefer two simple clauses over one nested clause"));
    }

    #[test]
    fn arbitrary_sources_receive_no_special_phrase_notes() {
        let sources = [
            "The telescope is beside my neighbor's red cart.",
            "If I might have arrived earlier, would the result differ?",
            "No, wait—I meant the eastern entrance.",
        ];
        let prompt = primary_user_prompt(&[], &[], &sources);

        assert!(!prompt.contains("Line-specific translation notes"));
        assert!(!prompt.lines().any(|line| line.starts_with("- ")));
    }

    #[test]
    fn only_matching_supplied_glossary_forms_receive_line_local_notes() {
        let names = [
            DirectHskName {
                source_english: "River",
                chinese: "河",
            },
            DirectHskName {
                source_english: "River Stone",
                chinese: "河石",
            },
            DirectHskName {
                source_english: "Never Present",
                chinese: "不会出现",
            },
        ];
        let sources = [
            "River Stone met River.",
            "No approved glossary form occurs here.",
        ];
        let prompt = primary_user_prompt(&[], &names, &sources);

        assert!(prompt.contains(
            "- 1: approved glossary \"River Stone\" => \"河石\" (use this exact form); approved glossary \"River\" => \"河\" (use this exact form)"
        ));
        assert!(!prompt.contains("\"Never Present\""));
        assert!(!prompt.contains("- 2:"));
    }

    #[test]
    fn system_prompt_and_fingerprint_material_are_generic() {
        let system = primary_system_prompt(5, 3);
        let material = DIRECT_HSK_PROMPT_FINGERPRINT_MATERIAL;
        assert!(system.contains("Preserve complete meaning"));
        assert!(system.contains("ambiguity itself when unresolved"));
        assert!(material.contains("application-supplied approved glossary entries"));
        assert!(material.contains("publisher/site credit"));
        assert!(material.contains("[NON-STORY]"));
        assert!(!system.contains("chapter"));
        assert!(!material.contains("chapter"));
    }

    #[test]
    fn name_style_explicitly_switches_between_original_and_chinese_forms() {
        let original =
            primary_system_prompt_with_name_style(3, 1, DirectHskNameStyle::KeepOriginal);
        let chinese = primary_system_prompt_with_name_style(3, 1, DirectHskNameStyle::Chinese);
        let original_repair =
            repair_system_prompt_with_name_style(3, DirectHskNameStyle::KeepOriginal);

        assert!(original.contains("exactly as written in the English source"));
        assert!(original.contains("⟦exact source spelling⟧"));
        assert!(original.contains("Translate every English word outside"));
        assert!(original_repair.contains("Decide proper names from the complete source meaning"));
        assert!(original_repair.contains("boundary-aligned source span"));
        assert!(chinese.contains("phonetic Chinese transliteration"));
        assert!(!chinese.contains("including its original Latin spelling"));
    }

    #[test]
    fn longer_overlapping_names_are_substituted_first_without_application_ids() {
        let names = [
            DirectHskName {
                source_english: "Mira",
                chinese: "米拉",
            },
            DirectHskName {
                source_english: "Professor Mira",
                chinese: "米拉教授",
            },
        ];

        assert_eq!(
            substitute_approved_names("Professor Mira met Mira.", &names),
            "米拉教授 met 米拉."
        );
    }

    #[test]
    fn repair_protocol_substitutes_names_and_omits_context_and_numbered_framing() {
        let names = [DirectHskName {
            source_english: "Alice",
            chinese: "爱丽丝",
        }];
        let prompt = repair_user_prompt(
            "Alice does not have 2 tickets.",
            Some("她有票。"),
            &["preserve 2", "preserve negation"],
            &names,
        );

        assert_eq!(
            prompt,
            "Source: 爱丽丝 does not have 2 tickets.\n\
Rejected: 她有票。\n\
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
