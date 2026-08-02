//! Shared compact prompt protocol for direct English-to-HSK Chinese translation.
//!
//! Product and benchmark callers deliberately use these same builders. The
//! model sees only temporary one-based positions; application IDs remain with
//! the caller.

use std::fmt::Write as _;

pub const DIRECT_HSK_PROMPT_REVISION: &str =
    "direct-hsk-en-zh-ordered-connected-regions-v79-2026-08-02";

/// Canonical protocol description whose SHA-256 is
/// [`DIRECT_HSK_PROMPT_HASH`].
///
/// Keep this material synchronized with the builders below. The unit test pins
/// the digest so a prompt-semantic change cannot silently reuse cache entries
/// or benchmark evidence.
pub const DIRECT_HSK_PROMPT_FINGERPRINT_MATERIAL: &str = "\
    direct-hsk-en-zh-ordered-connected-regions-v79-2026-08-02
ocr-correction=OCR may contain minor letter, spacing, or punctuation errors; silently correct an obvious error when grammar, neighboring lines, and story context make the intended English clear, never carry nonsensical fragments into Chinese, never invent genuinely ambiguous content, and do not omit concrete nouns or modifiers merely because their exact term is above the requested level; paraphrase them with a simpler expression instead
translation-test=when deciding whether a discovered span is an opaque name, test whether translating its ordinary component meanings would preserve its role; color-plus-role, adjective-plus-occupation, species-plus-title, relationship words, and institutional job titles are descriptive even when they identify one character; reserve opaque for arbitrary personal names, surnames, coined places, coined organizations, and coined entities
page-title-admission=a standalone prominent work or series title-like noun phrase with no finite clause or other story context is page furniture; do not force a readable cover title into story
opaque-lexical-gate=a one-token or multi-token span with no ordinary English dictionary meaning is an opaque identifier even inside a sentence; unknown transliterated-looking strings are not descriptive
semantic-analysis=separate bounded pre-cleanup same-model response schemas ensure entity output cannot corrupt page-role output; semantic response budgets are derived from the maximum serialized schema size rather than average model formatting; the page-role pass first emits an explicit all-furniture-page-section decision and then proposes story, pure standalone SFX, unrelated furniture, or visually integral illustrated lettering; an invalid model-level ARTWORK page role is normalized to page furniture only for a complete all-non-story section containing an independent furniture anchor, while incomplete, mixed, and artwork-only sections remain region-scoped and fail safe to story; every independently visually admissible short-form region then requires a separate focused batched PRESERVE-or-TRANSLATE adjudication regardless of the broad page-role proposal, using the complete ordered page-section context and each candidate's source position so isolated interjections and story fragments remain translatable while named technique artwork stays preserved; any candidate that plausibly functions as a spoken or narrated utterance in context translates, including greetings, toasts, interjections, reactions, warnings, observations, and announcements, because dramatic wording or typography alone never makes story text artwork; quantities, rewards, items, missions, instructions, interface labels, ordinary roles, and uncertain cases translate even when large or stylized, and malformed or omitted decisions translate by default; short non-sentence tokens repeated across separate OCR regions are strong generic SFX evidence unless their context clearly makes them dialogue or a named label; terminal sentence punctuation, ASCII numeric notation, and long clause structure are deterministic story evidence and cannot enter artwork adjudication; keep-original entity handling uses typed candidate discovery followed by one batched contextual OPAQUE-or-DESCRIPTIVE lexical-function adjudication only when candidates exist; discovery emits shortest exact source spans labelled PERSON, PLACE, ORGANIZATION, EVENT, or ENTITY; candidate fields accept pipe or tab separation and TYPE=SPAN or SPAN=TYPE order; malformed, omitted, or hallucinated name entries fail closed per numbered line without discarding valid siblings, while an exact boundary-aligned untyped discovery span is retained as UNKNOWN solely for independent contextual adjudication and can never become protected by discovery alone; each proposed type, complete source occurrences, and generic syntactic attachment evidence remains attached through adjudication so ordinary phenomena, actions, objects, qualities, roles, titles, and transparent descriptions are DESCRIPTIVE while arbitrary personal, place, organization, event, and entity identifiers are OPAQUE; a PERSON span preceded by a definite article is explicitly adjudicated as likely DESCRIPTIVE unless its words are demonstrably an opaque identifier; adjudication has a bounded response budget sized for every candidate and its parser accepts exactly one unambiguous OPAQUE-or-DESCRIPTIVE token per numbered line despite harmless formatting tokens; only boundary-aligned OPAQUE candidates approved by both stages, remembered chapter entities, or configured glossary entries become placeholders and protected names; deterministic admissibility rejects closed-class spans and possessive-attached common nouns without rejecting predicate-position names; page-role analysis consumes the complete available section plus fallible layout, relative font height, color-band, and outline evidence; an ARTWORK region is protected only when independent scale-or-styled-freeform visual evidence and focused semantic adjudication agree, so interface text cannot be removed merely because one model response emitted ARTWORK and technique lettering cannot be lost merely because the broad classifier emitted STORY; ARTWORK requires both illustrated visual function and the semantics of a named attack, technique, form, spell, transformation, or title, and nearby fragments may form one such label; finite clauses, sentences, questions, conversational statements, pronoun-led lines, and ordinary verb phrases are always story regardless of emphasis; malformed or uncertain role decisions fail safe to story; SFX follows user policy; retained story is authoritative and translation and repair cannot create additional names, preserve unsupplied Latin text, or later discard it
name-adjudication=when an opaque candidate is followed by a translatable kinship or group noun such as twins, brothers, sisters, family, clan, or house, protect only the candidate and translate the noun; a multi-token personal-name candidate remains opaque when a rank, title, or role follows it; a leading definite article or surrounding descriptive phrase does not downgrade the exact identifier; preserve every approved opaque name once for each source occurrence and never duplicate it; numbered verifier parsing may discard only a leading numeric formatting field before the strict OPAQUE or DESCRIPTIVE decision
primary-system=translate exactly {count} preclassified story OCR regions in supplied comic reading order; never output a semantic exclusion marker because page function was already decided; use supplied preceding translations and neighboring numbered regions to resolve pronouns, omitted subjects, ellipsis, and sentences split across connected bubbles; neighboring regions provide context but each output contains only its own region's portion without duplication or invention; preserve genuinely standalone fragments; response starts 1+tab; exactly {count} non-empty ordered lines 1..{count}; position+one-tab+Chinese; actively rewrite vocabulary, grammar, clause structure, and idioms for HSK2.0 level {level} according to a level-specific style rule, while preserving meaning; natural-learning mode simplifies first and enforces a numeric level-appropriate lexical-coverage target plus a numeric per-line exception ceiling, retaining only indispensable story terms when paraphrase would become awkward, childish, repetitive, or imprecise, and leaves teaching metadata to the application; strict mode rewrites every avoidable advanced term and grammar pattern except protected names and required glossary forms; at levels 1-2 use basic everyday words, short subject-verb-object clauses, explicit referents, and avoid idioms, literary/formal wording, nominalization, nested clauses, and avoidable passive/把/被 constructions; at levels 3-4 allow common compound sentences and familiar connectors but replace advanced idioms, formal synonyms, and dense embedding; at levels 5-6 allow natural advanced grammar and precise vocabulary; preserve every clause/detail, speaker/addressee/participant roles, agency, attachment, causality, modality/certainty/condition, quantities/comparisons, negation, question intent, tone/humour, context-resolved ambiguity, unresolved ambiguity, relationships, pronoun referents, self-corrections in order, and numeric values; in keep-original mode copy only supplied opaque approved-name placeholders exactly and translate every other Latin word including relationships, honorifics, roles, ranks, titles, work or chapter titles, furniture that survived classification, and uncertain OCR tokens; never invent a name, marker, or unsupplied Latin span because dedicated NER already supplied the complete approved name set; in Chinese-name mode use approved/established/phonetic Chinese forms without dictionary-meaning translation; follow optional line-local approved-glossary notes only for matching position; no headings/labels/explanation/markdown/json/IDs
primary-user=optional readable preceding-translations heading and dash source=>Chinese reference lines; optional line-specific-note heading with one-based position-tagged notes generated solely from application-supplied approved glossary entries whose exact ASCII English forms occur on that source line, selecting longest nonoverlapping boundary-aligned matches; blank separators; English-lines heading; one-based position+tab+English with only already-approved remembered or glossary keep-original spans replaced by opaque per-line ordered markers such as ⟦N1⟧
repair-system=one or up to six independently numbered accurate natural HSK2.0 level {level} Chinese repairs for preclassified story regions; semantic role is authoritative and repair always returns a translation; actively apply the same level-specific vocabulary, grammar, clause-structure, and idiom rule as primary generation; fix all item-local problems; natural-learning mode simplifies every listed term that has a natural level-safe expression and must meet the same numeric coverage target and per-line exception ceiling as primary generation; strict mode treats each deterministic validator avoid-list as exact forbidden Chinese substrings and emits none of them except protected names and required glossary forms; in keep-original mode copy only verified opaque placeholders and translate every other Latin word, or use Chinese name forms without dictionary-meaning translation in Chinese-name mode; preserve every clause/detail, roles, agency, causality, modality, quantities/comparisons, negation, question intent, tone/humour, ambiguity, self-corrections, approved-substituted-glossary-forms, and numeric values; singular output has no position or tab while batch output uses exact ordered position+tab lines
repair-user=each item has Source/Rejected/Validator-avoid-list/Problems readable fields; singular adds Answer while batch adds one Corrected-numbered-lines response boundary; the typed validator avoid-list is refreshed from each rejected candidate; Chinese-name mode substitutes matching approved Chinese glossary forms; keep-original mode replaces only independently verified exact boundary-aligned source spans with opaque per-line ordered markers; parser restores markers to exact OCR spelling before deterministic item-local validation
decoding=greedy-unpenalized
batch=production-max6-with-exact-chat-template-token-capacity-planning; choose-largest-fitting-ordered-prefix-before-generation; never retry-a-known-oversized-prompt; preserve-streaming-and-application-id-order-across-subbatches
context=max6-preceding-utterances-and-max256-context-tokens; remove-oldest-context-first-when-needed-for-the-current-batch; when-context-removal-is-insufficient-split-the-ordered-batch; one-utterance-output-budget-may-use-exact-remaining-capacity-but-never-less-than-24-tokens
repair=one batched validate-and-revise generation with no context and no primary retry; up to six rejected bubbles share the generation while parsing and validation remain isolated by application ID; each item receives compact exact deterministic validator spans and an exact typed avoid-list, without context-free dictionary substitutions; the model rewrites those spans only in full sentence context; natural-learning keeps an indispensable story concept when removing it would damage meaning; text requiring revision is withheld from the browser until this final result is terminal, while already accepted primary text may publish immediately";

// Filled from the exact UTF-8 bytes of
// DIRECT_HSK_PROMPT_FINGERPRINT_MATERIAL.
pub const DIRECT_HSK_PROMPT_HASH: &str =
    "sha256:6230f03fbc36622d2f69bdfaae1cd77e4f77f0505e8742479814fb91380bdafc";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectHskNameStyle {
    KeepOriginal,
    Chinese,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectHskLearningMode {
    Natural,
    Strict,
}

/// Shared identity of numbered-line parsing and deterministic preservation
/// validation used by production and release evidence.
pub const DIRECT_HSK_VALIDATOR_FINGERPRINT_MATERIAL: &str = "numbered-tab-or-space-parser|\
fullwidth-ascii|single-item-primary-direct-chinese-line-v1|source-guided-ascii-multiplier-to-unicode-before-latin-validation-v1|strict-primary-ascii-repair-trigger|source-guided-final-numeric-value-validation-v3-\
    ignore-letter-adjacent-ocr-digits-with-boundary-multiplier-notation-v2|deterministic-question-punctuation-v1|translation-reclassification-markers-malformed-v1|\
    semantic-non-story-disposition-v6-furniture-anchored-complete-page-normalization-three-way-artwork-consensus-sentence-number-clause-story-gates-and-standalone-network-identifier-exclusion|semantic-sfx-policy-disposition-v1|semantic-ner-typed-exact-source-spans-v12-opaque-vs-descriptive-lexical-function-adjudication-generic-syntactic-attachment-evidence-pipe-or-tab-fields-and-reversible-type-pairs-untyped-exact-spans-unknown-until-independent-adjudication-per-line-fail-closed-order-normalized-type-retained-unambiguous-decision-token-with-closed-class-and-possessive-attachment-gates|opaque-approved-name-placeholder-restoration-v1|unapproved-name-markup-rejected-v2|known-protected-unmarked-latin-source-line-scoped-v3|all-other-latin-rejected-in-every-name-mode-v3|names|question-fragment-aware-v2|\
     excessive-han-expansion-v1|protected-name-occurrence-count-v1";
pub const DIRECT_HSK_VALIDATOR_HASH: &str =
    "sha256:e17e5f267a4672bdf912ed06a9ddac060fb8d7725dfd33166374f166c4626a15";

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
    primary_system_prompt_with_policy(level, count, name_style, true)
}

#[must_use]
pub fn primary_system_prompt_with_policy(
    level: u8,
    count: usize,
    name_style: DirectHskNameStyle,
    translate_sound_effects: bool,
) -> String {
    primary_system_prompt_with_learning_policy(
        level,
        count,
        name_style,
        translate_sound_effects,
        DirectHskLearningMode::Strict,
    )
}

#[must_use]
pub fn primary_system_prompt_with_learning_policy(
    level: u8,
    count: usize,
    name_style: DirectHskNameStyle,
    _translate_sound_effects: bool,
    learning_mode: DirectHskLearningMode,
) -> String {
    let level_style = level_style_instruction(level);
    let semantic_admission_instruction = "Semantic role classification is already complete: every \
        supplied region is admitted story content that must receive a translation. Do not classify, \
        exclude, skip, merge, or reorder any region.";
    let name_instruction = match name_style {
        DirectHskNameStyle::KeepOriginal => {
            "Pretranslation semantic analysis has already supplied the complete approved proper-name set. Copy \
            every supplied opaque approved-name placeholder such as `⟦N1⟧` exactly once and unchanged \
            in its grammatical position; the application restores its exact source spelling. Never \
            invent another name, name marker, or unsupplied Latin span. Translate every other Latin \
            word, including capitalized descriptions, relationships, honorifics, occupations, roles, \
            ranks, titles, species, work or chapter titles, interface or game categories, ordinary noun \
            phrases, and uncertain OCR tokens."
        }
        DirectHskNameStyle::Chinese => {
            "Treat person, place, organization, and other proper names as names: never translate \
            their dictionary meaning. Use an approved glossary form when supplied; otherwise use an \
            established Chinese name only when certain, or a phonetic Chinese transliteration, and \
            keep it consistent with preceding context."
        }
    };
    let learning_instruction = match learning_mode {
        DirectHskLearningMode::Natural => natural_learning_instruction(level),
        DirectHskLearningMode::Strict => {
            "Use strict HSK policy. Rewrite every avoidable above-level word and grammar pattern with \
            level-appropriate language even when the result is less elegant. Only protected proper \
            names and exact required glossary forms may remain outside the selected level."
                .to_owned()
        }
    };
    format!(
        "Translate the {count} numbered OCR regions in their supplied comic reading order. \
{semantic_admission_instruction} OCR can contain minor recognition errors in letters, spacing, or \
punctuation. Silently correct an obvious OCR error when grammar, neighboring lines, and story context \
make the intended English clear; do not carry nonsensical OCR fragments into the Chinese translation, \
and do not invent content when the source is genuinely ambiguous. Translate each source into concise, \
natural Simplified Chinese for a reader targeting cumulative HSK 2.0 level {level}. \
Use the supplied preceding translations and neighboring numbered regions to resolve pronouns, omitted \
subjects, ellipsis, and sentences split across connected bubbles. Adjacent regions are context, not extra \
content: each numbered output must contain only the meaning carried by its own source region, while using \
the surrounding sequence to make that portion coherent. Preserve a genuinely standalone fragment as a \
fragment; when several regions form one sentence, translate each region as its corresponding portion of \
that sentence without duplicating or inventing meaning. Actively rewrite vocabulary, grammar, clause structure, and idioms to \
        suit the requested level—not vocabulary alone. {level_style} {learning_instruction} Prefer the simplest natural wording \
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
        the required temporary name markers when keep-original mode is active. \
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
    names: &[DirectHskName<'_>],
    sources: &[&str],
) -> String {
    primary_user_prompt_with_name_style(context, names, sources, DirectHskNameStyle::Chinese)
}

#[must_use]
pub fn primary_user_prompt_with_name_style(
    context: &[DirectHskContext<'_>],
    names: &[DirectHskName<'_>],
    sources: &[&str],
    name_style: DirectHskNameStyle,
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
        let source = match name_style {
            DirectHskNameStyle::KeepOriginal => mark_approved_names(source, names),
            DirectHskNameStyle::Chinese => compact(source),
        };
        writeln!(&mut prompt, "{}\t{source}", index + 1).expect("writing to String cannot fail");
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
    repair_system_prompt_with_policy(level, name_style, true)
}

#[must_use]
pub fn repair_system_prompt_with_policy(
    level: u8,
    name_style: DirectHskNameStyle,
    translate_sound_effects: bool,
) -> String {
    repair_system_prompt_with_learning_policy(
        level,
        name_style,
        translate_sound_effects,
        DirectHskLearningMode::Strict,
    )
}

#[must_use]
pub fn repair_system_prompt_with_learning_policy(
    level: u8,
    name_style: DirectHskNameStyle,
    translate_sound_effects: bool,
    learning_mode: DirectHskLearningMode,
) -> String {
    repair_system_prompt_for_count(level, 1, name_style, translate_sound_effects, learning_mode)
}

#[must_use]
pub fn repair_batch_system_prompt_with_learning_policy(
    level: u8,
    count: usize,
    name_style: DirectHskNameStyle,
    translate_sound_effects: bool,
    learning_mode: DirectHskLearningMode,
) -> String {
    assert!(count > 0, "repair batch must not be empty");
    repair_system_prompt_for_count(
        level,
        count,
        name_style,
        translate_sound_effects,
        learning_mode,
    )
}

fn repair_system_prompt_for_count(
    level: u8,
    count: usize,
    name_style: DirectHskNameStyle,
    _translate_sound_effects: bool,
    learning_mode: DirectHskLearningMode,
) -> String {
    let level_style = level_style_instruction(level);
    let name_instruction = match name_style {
        DirectHskNameStyle::KeepOriginal => {
            "Pretranslation semantic analysis already supplied the complete approved proper-name set. Keep every \
            opaque approved-name placeholder such as `⟦N1⟧` exactly once and unchanged in its \
            grammatical position; the application restores the exact source spelling. Do not invent \
            additional names, markers, or unsupplied Latin spans. Translate every other Latin word, \
            including descriptions, relationships, honorifics, occupations, roles, ranks, titles, \
            species, work or chapter titles, interface or game categories, noun phrases, clauses, and \
            uncertain OCR tokens."
        }
        DirectHskNameStyle::Chinese => {
            "Never translate proper names by dictionary meaning; use approved or established forms \
            and otherwise a phonetic Chinese transliteration."
        }
    };
    let learning_instruction = match learning_mode {
        DirectHskLearningMode::Natural => natural_learning_instruction(level),
        DirectHskLearningMode::Strict => {
            "Replace every listed above-level term and grammar pattern with level-appropriate wording. \
            Treat every exact term in the Validator avoid-list as a forbidden Chinese substring: \
            check the completed answer and emit none of them. Only protected names and required \
            glossary forms are exceptions."
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
    format!(
        "{scope_instruction} for a reader targeting \
        cumulative HSK 2.0 level {level}. OCR can contain minor recognition errors in letters, spacing, or \
punctuation. Silently correct an obvious OCR error when grammar and context make the intended English \
clear; do not carry nonsensical OCR fragments into Chinese. Fix every listed problem. Actively rewrite vocabulary, grammar, \
        clause structure, and idioms for the requested level—not vocabulary alone. {level_style} {learning_instruction} Preserve \
        every clause and detail, participant roles, \
agency, cause and result, modality, quantities and comparisons, negation, question intent, tone \
        and humour, ambiguity, pronoun referents, self-corrections, approved glossary forms \
        already present in the source, and numeric values. {output_instruction} \
Semantic role classification is already complete; always return a translation. {name_instruction} Write no headings, labels, explanations, Markdown, \
JSON, or application IDs."
    )
}

#[must_use]
pub fn repair_user_prompt(
    source_english: &str,
    rejected_chinese: Option<&str>,
    problems: &[&str],
    names: &[DirectHskName<'_>],
) -> String {
    repair_user_prompt_with_name_style(
        source_english,
        rejected_chinese,
        problems,
        names,
        DirectHskNameStyle::Chinese,
    )
}

#[must_use]
pub fn repair_user_prompt_with_name_style(
    source_english: &str,
    rejected_chinese: Option<&str>,
    problems: &[&str],
    names: &[DirectHskName<'_>],
    name_style: DirectHskNameStyle,
) -> String {
    repair_user_prompt_with_constraints(
        source_english,
        rejected_chinese,
        problems,
        names,
        name_style,
        &[],
    )
}

#[must_use]
pub fn repair_user_prompt_with_constraints(
    source_english: &str,
    rejected_chinese: Option<&str>,
    problems: &[&str],
    names: &[DirectHskName<'_>],
    name_style: DirectHskNameStyle,
    avoid_chinese: &[String],
) -> String {
    format!(
        "{}\nAnswer:",
        repair_item_constraints(
            source_english,
            rejected_chinese,
            problems,
            names,
            name_style,
            avoid_chinese,
        )
    )
}

#[must_use]
pub fn repair_item_constraints(
    source_english: &str,
    rejected_chinese: Option<&str>,
    problems: &[&str],
    names: &[DirectHskName<'_>],
    name_style: DirectHskNameStyle,
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
    let source = match name_style {
        DirectHskNameStyle::KeepOriginal => mark_approved_names(source_english, names),
        DirectHskNameStyle::Chinese => substitute_approved_names(source_english, names),
    };
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
        "Source: {}\nRejected: {rejected}\nValidator avoid-list: {avoid}\nProblems: {problems}",
        source,
    )
}

#[must_use]
pub fn mark_approved_names(source: &str, names: &[DirectHskName<'_>]) -> String {
    replace_approved_name_occurrences(source, names, |_, _, ordinal| {
        format!("\u{27e6}N{ordinal}\u{27e7}")
    })
}

#[must_use]
pub fn restore_approved_name_placeholders(
    source: &str,
    translation: &str,
    names: &[DirectHskName<'_>],
) -> String {
    let mut restored = translation.to_owned();
    let _ = replace_approved_name_occurrences(source, names, |matched, _, ordinal| {
        let placeholder = format!("\u{27e6}N{ordinal}\u{27e7}");
        restored = restored.replace(&placeholder, &format!("\u{27e6}{matched}\u{27e7}"));
        String::new()
    });
    restored
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
    replace_approved_name_occurrences(source, names, |_, name, _| name.chinese.to_owned())
}

fn replace_approved_name_occurrences(
    source: &str,
    names: &[DirectHskName<'_>],
    mut render: impl FnMut(&str, DirectHskName<'_>, usize) -> String,
) -> String {
    let source = compact(source);
    let source_ref = source.as_str();
    let lower = source.to_ascii_lowercase();
    let mut occurrences = names
        .iter()
        .filter(|name| !name.source_english.is_empty() && name.source_english.is_ascii())
        .flat_map(|name| {
            let needle = name.source_english.to_ascii_lowercase();
            lower
                .match_indices(&needle)
                .filter_map(move |(start, matched)| {
                    let end = start + matched.len();
                    let starts_at_boundary =
                        start == 0 || !source_ref.as_bytes()[start - 1].is_ascii_alphanumeric();
                    let ends_at_boundary = end == source_ref.len()
                        || !source_ref.as_bytes()[end].is_ascii_alphanumeric();
                    (starts_at_boundary && ends_at_boundary).then_some((start, end, *name))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    occurrences.sort_by(|left, right| {
        (right.1 - right.0)
            .cmp(&(left.1 - left.0))
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut selected = Vec::<(usize, usize, DirectHskName<'_>)>::new();
    for occurrence in occurrences {
        if selected
            .iter()
            .any(|used| occurrence.0 < used.1 && used.0 < occurrence.1)
        {
            continue;
        }
        selected.push(occurrence);
    }
    selected.sort_by_key(|occurrence| occurrence.0);

    let mut output = String::with_capacity(source.len());
    let mut cursor = 0;
    for (ordinal, (start, end, name)) in selected.into_iter().enumerate() {
        output.push_str(&source[cursor..start]);
        output.push_str(&render(&source[start..end], name, ordinal + 1));
        cursor = end;
    }
    output.push_str(&source[cursor..]);
    output
}

#[must_use]
pub fn compact(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
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
        assert!(system.contains("Translate the 2 numbered OCR regions"));
        assert!(system.contains("Semantic role classification is already complete"));
        assert!(!system.contains("[NON-STORY]"));
        assert!(!system.contains("[SFX]"));
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
        assert!(system.contains("supplied comic reading order"));
        assert!(system.contains("sentences split across connected bubbles"));
        assert!(system.contains("each numbered output"));
        assert!(system.contains("genuinely standalone fragment"));
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
        assert!(material.contains("unrelated furniture"));
        assert!(material.contains("hallucinated name entries fail closed per numbered line"));
        assert!(material.contains("untyped discovery span is retained as UNKNOWN"));
        assert!(material.contains("semantic exclusion marker"));
        assert!(!system.contains("chapter"));
        assert!(!material.to_ascii_lowercase().contains("asura"));
        assert!(!material.to_ascii_lowercase().contains("webtoon"));
        assert!(!material.contains("Maysa"));
    }

    #[test]
    fn learning_modes_have_distinct_controlled_vocabulary_policies() {
        let natural = primary_system_prompt_with_learning_policy(
            3,
            1,
            DirectHskNameStyle::KeepOriginal,
            false,
            DirectHskLearningMode::Natural,
        );
        let strict = primary_system_prompt_with_learning_policy(
            3,
            1,
            DirectHskNameStyle::KeepOriginal,
            false,
            DirectHskLearningMode::Strict,
        );
        let natural_repair = repair_system_prompt_with_learning_policy(
            3,
            DirectHskNameStyle::KeepOriginal,
            false,
            DirectHskLearningMode::Natural,
        );

        assert!(natural.contains("simplify-preserve-teach"));
        assert!(natural.contains("90% level-appropriate lexical occurrences"));
        assert!(natural.contains("no more than 1 above-level occurrence"));
        assert!(natural.contains("application will identify and teach"));
        assert!(strict.contains("strict HSK policy"));
        assert!(strict.contains("Rewrite every avoidable above-level word"));
        assert!(natural_repair.contains("90% level-appropriate lexical occurrences"));
        assert!(natural_repair.contains("no more than 1 above-level occurrence"));
        assert_ne!(natural, strict);
    }

    #[test]
    fn name_style_explicitly_switches_between_original_and_chinese_forms() {
        let original =
            primary_system_prompt_with_name_style(3, 1, DirectHskNameStyle::KeepOriginal);
        let chinese = primary_system_prompt_with_name_style(3, 1, DirectHskNameStyle::Chinese);
        let original_repair =
            repair_system_prompt_with_name_style(3, DirectHskNameStyle::KeepOriginal);

        assert!(original.contains("complete approved proper-name set"));
        assert!(original.contains("Never invent another name"));
        assert!(original.contains("Translate every other Latin word"));
        assert!(original.contains("⟦N1⟧"));
        assert!(original.contains("Translate every other Latin word"));
        assert!(original_repair.contains("complete approved proper-name set"));
        assert!(original_repair.contains("opaque approved-name placeholder"));
        assert!(original_repair.contains("complete approved proper-name set"));
        assert!(chinese.contains("phonetic Chinese transliteration"));
        assert!(!chinese.contains("including its original Latin spelling"));
    }

    #[test]
    fn keep_original_marks_only_model_approved_boundary_aligned_source_spans() {
        let names = [
            DirectHskName {
                source_english: "Maysa",
                chinese: "Maysa",
            },
            DirectHskName {
                source_english: "Ann",
                chinese: "Ann",
            },
        ];
        let sources = ["Ann met MAYSA near Annette."];
        let primary = primary_user_prompt_with_name_style(
            &[],
            &names,
            &sources,
            DirectHskNameStyle::KeepOriginal,
        );
        let repair = repair_user_prompt_with_name_style(
            sources[0],
            Some("玛莎来了。"),
            &["preserve approved names"],
            &names,
            DirectHskNameStyle::KeepOriginal,
        );

        assert!(primary.contains("1\t⟦N1⟧ met ⟦N2⟧ near Annette."));
        assert!(repair.contains("Source: ⟦N1⟧ met ⟦N2⟧ near Annette."));
        assert!(!primary.contains("⟦N1⟧ette"));
        assert_eq!(
            restore_approved_name_placeholders(sources[0], "⟦N2⟧见到了⟦N1⟧。", &names),
            "⟦MAYSA⟧见到了⟦Ann⟧。"
        );
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
Validator avoid-list: <none>\n\
Problems: preserve 2 | preserve negation\n\
Answer:"
        );
        assert!(!prompt.contains("Previous translations"));
        assert!(!prompt.lines().any(|line| line.starts_with("1\t")));
    }

    #[test]
    fn repair_constraints_render_the_validator_avoid_list_as_a_separate_field() {
        let prompt = repair_user_prompt_with_constraints(
            "She is a goddess.",
            Some("她是女神。"),
            &["rewrite above-level vocabulary"],
            &[],
            DirectHskNameStyle::Chinese,
            &["女神".to_owned(), "注定".to_owned()],
        );

        assert!(prompt.contains("Validator avoid-list: 女神, 注定"));
        assert!(prompt.ends_with("\nAnswer:"));
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
