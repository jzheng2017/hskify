//! Direct English-to-HSK-targeted Simplified Chinese batch translation.
//!
//! The model only sees compact, one-based positions. Stable application IDs
//! are mapped onto those positions after generation, so the model never has
//! to copy opaque IDs or emit a verbose schema. Parsing and preservation
//! checks are per item: one malformed line does not discard valid siblings.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};
use koharu_llm::direct_hsk_protocol::{
    DirectHskContext, DirectHskLearningMode, DirectHskName, DirectHskNameStyle,
    context_budget_text, primary_system_prompt_with_learning_policy,
    primary_user_prompt_with_name_style, repair_batch_system_prompt_with_learning_policy,
    repair_item_constraints, repair_system_prompt_with_learning_policy,
    repair_user_prompt_with_constraints, restore_approved_name_placeholders,
};
use koharu_llm::{GenerateOptions, Language, ModelId};
use serde::{Deserialize, Serialize};

use super::{Model, State};

pub use koharu_llm::direct_hsk_protocol::{
    DIRECT_HSK_PROMPT_HASH as HSK_TRANSLATION_PROMPT_HASH,
    DIRECT_HSK_PROMPT_REVISION as HSK_TRANSLATION_PROMPT_REVISION,
    DIRECT_HSK_VALIDATOR_HASH as HSK_TRANSLATION_VALIDATOR_HASH,
};

pub const HSK_TRANSLATION_MODEL: ModelId = ModelId::Qwen3_5_4b;
pub const HSK_SEMANTIC_ANALYSIS_REVISION: &str =
    "page-wide-role-two-stage-typed-name-adjudication-v20-2026-07-29";
// Composite cache identity: repository@commit, filename, and exact file digest.
pub const HSK_TRANSLATION_MODEL_REVISION: &str = "unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523:Qwen3.5-4B-Q4_K_M.gguf:sha256=00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4";
pub const MAX_HSK_PRECEDING_UTTERANCES: usize = 6;
pub const MAX_HSK_CONTEXT_TOKENS: usize = 256;
pub const MAX_HSK_TRANSLATION_BATCH: usize = 6;
/// Semantic roles need the complete visual section so credits, covers, and
/// connected lettering are not misclassified after an arbitrary translation
/// batch boundary. Translation remains deliberately smaller and independent.
pub const MAX_HSK_SEMANTIC_PAGE_REGIONS: usize = 128;

/// Stable cache fingerprint for the direct and repair prompt templates, their
/// compact wire format, context bound, and greedy decoding policy.
#[must_use]
pub const fn direct_hsk_prompt_hash() -> &'static str {
    HSK_TRANSLATION_PROMPT_HASH
}

/// Stable cache fingerprint for numbered-line parsing, partial-result rules,
/// normalization, and deterministic preservation checks in this module.
///
/// The browser pipeline should additionally include the loaded
/// `hsk_control::HskControl::cache_revision()` because that resource-dependent
/// vocabulary fingerprint is intentionally owned outside the LLM layer.
#[must_use]
pub const fn direct_hsk_validator_hash() -> &'static str {
    HSK_TRANSLATION_VALIDATOR_HASH
}

const MIN_OUTPUT_TOKENS: usize = 24;
const MAX_OUTPUT_TOKENS: usize = 256;
const OUTPUT_TOKENS_PER_UTTERANCE: usize = 8;
const NON_STORY_MARKER: &str = "[NON-STORY]";
const SOUND_EFFECT_MARKER: &str = "[SFX]";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskTranslationBatchRequest {
    pub requested_level: u8,
    #[serde(default)]
    pub learning_mode: HskLearningMode,
    #[serde(default)]
    pub name_handling: HskNameHandling,
    #[serde(default = "default_translate_sound_effects")]
    pub translate_sound_effects: bool,
    pub utterances: Vec<HskSourceUtterance>,
    #[serde(default)]
    pub preceding_utterances: Vec<HskPrecedingUtterance>,
    #[serde(default)]
    pub protected_names: Vec<HskProtectedName>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HskNameHandling {
    KeepOriginal,
    #[default]
    Chinese,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HskLearningMode {
    #[default]
    Natural,
    Strict,
}

impl From<HskLearningMode> for DirectHskLearningMode {
    fn from(value: HskLearningMode) -> Self {
        match value {
            HskLearningMode::Natural => Self::Natural,
            HskLearningMode::Strict => Self::Strict,
        }
    }
}

const fn default_translate_sound_effects() -> bool {
    true
}

impl From<HskNameHandling> for DirectHskNameStyle {
    fn from(value: HskNameHandling) -> Self {
        match value {
            HskNameHandling::KeepOriginal => Self::KeepOriginal,
            HskNameHandling::Chinese => Self::Chinese,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskSourceUtterance {
    pub id: String,
    pub kind: HskUtteranceKind,
    pub source_english: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_layout: Option<HskSemanticLayout>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskSemanticLayout {
    pub detector_enclosed: bool,
    pub measured_font_height_millionths: u32,
    pub appearance_band_count: u32,
    pub distinct_text_color_count: u32,
    pub has_outline: bool,
    pub x0_millionths: u32,
    pub y0_millionths: u32,
    pub x1_millionths: u32,
    pub y1_millionths: u32,
    pub page_width: u32,
    pub page_height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HskUtteranceKind {
    Dialogue,
    Caption,
    Thought,
    Sfx,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskPrecedingUtterance {
    pub source_english: String,
    pub chinese: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskProtectedName {
    pub source_english: String,
    pub chinese: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskTranslationBatchResult {
    pub items: Vec<HskTranslationOutcome>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HskSemanticClassification {
    pub page_is_furniture: bool,
    pub regions: Vec<(String, HskTranslationDisposition)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskTranslationOutcome {
    pub id: String,
    #[serde(default)]
    pub disposition: HskTranslationDisposition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<HskTranslationIssue>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HskTranslationDisposition {
    #[default]
    Translate,
    ExcludeNonStory,
    ExcludeSoundEffect,
    PreserveArtwork,
}

impl HskTranslationOutcome {
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.disposition == HskTranslationDisposition::Translate
            && self.issues.is_empty()
            && self
                .text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty())
    }

    #[must_use]
    pub fn is_non_story(&self) -> bool {
        matches!(
            self.disposition,
            HskTranslationDisposition::ExcludeNonStory
                | HskTranslationDisposition::ExcludeSoundEffect
        ) && self.issues.is_empty()
            && self.text.is_none()
    }

    #[must_use]
    pub fn repair_problems(&self) -> Vec<String> {
        self.issues
            .iter()
            .map(HskTranslationIssue::description)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum HskTranslationIssue {
    MissingLine,
    DuplicateLine,
    MalformedLine,
    SourceEcho,
    EmptyTranslation,
    NumberMismatch {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    ProtectedNameMissing {
        source_english: String,
        chinese: String,
    },
    QuestionIntentMissing,
    ExcessiveExpansion {
        source_words: usize,
        chinese_characters: usize,
    },
    InvalidNameMarkup,
    UnmarkedLatinText,
}

impl HskTranslationIssue {
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::MissingLine => "no translation was returned".to_owned(),
            Self::DuplicateLine => "more than one translation was returned".to_owned(),
            Self::MalformedLine => "return only the Simplified Chinese translation".to_owned(),
            Self::SourceEcho => "translate the source instead of copying it".to_owned(),
            Self::EmptyTranslation => "translation is empty".to_owned(),
            Self::NumberMismatch { expected, actual } => {
                format!("preserve ASCII numbers exactly: expected {expected:?}, got {actual:?}")
            }
            Self::ProtectedNameMissing {
                source_english,
                chinese,
            } => format!("translate protected name `{source_english}` exactly as `{chinese}`"),
            Self::QuestionIntentMissing => "preserve the source question intent".to_owned(),
            Self::ExcessiveExpansion {
                source_words,
                chinese_characters,
            } => format!(
                "translate only this source fragment; {source_words} English words expanded to \
{chinese_characters} Chinese characters"
            ),
            Self::InvalidNameMarkup => {
                "copy only the supplied opaque approved-name placeholders; do not add name markers"
                    .to_owned()
            }
            Self::UnmarkedLatinText => {
                "translate every Latin word outside the supplied opaque approved-name placeholders"
                    .to_owned()
            }
        }
    }
}

/// A repair request contains exactly one candidate rejected by parsing,
/// preservation checks, or the caller's deterministic HSK vocabulary
/// validator. The caller owns the small bounded retry policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskTranslationRepairRequest {
    pub requested_level: u8,
    #[serde(default)]
    pub learning_mode: HskLearningMode,
    #[serde(default)]
    pub name_handling: HskNameHandling,
    #[serde(default = "default_translate_sound_effects")]
    pub translate_sound_effects: bool,
    pub utterance: HskRepairUtterance,
    #[serde(default)]
    pub preceding_utterances: Vec<HskPrecedingUtterance>,
    #[serde(default)]
    pub protected_names: Vec<HskProtectedName>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskTranslationRepairBatchRequest {
    pub requested_level: u8,
    #[serde(default)]
    pub learning_mode: HskLearningMode,
    #[serde(default)]
    pub name_handling: HskNameHandling,
    #[serde(default = "default_translate_sound_effects")]
    pub translate_sound_effects: bool,
    pub utterances: Vec<HskRepairUtterance>,
    #[serde(default)]
    pub preceding_utterances: Vec<HskPrecedingUtterance>,
    #[serde(default)]
    pub protected_names: Vec<HskProtectedName>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskRepairUtterance {
    pub id: String,
    pub kind: HskUtteranceKind,
    pub source_english: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_chinese: Option<String>,
    #[serde(default)]
    pub avoid_chinese: Vec<String>,
    pub problems: Vec<String>,
}

trait Generator {
    async fn token_count(&self, text: &str) -> Result<usize>;

    async fn constrained_completion_capacity(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        target_language: Language,
    ) -> Result<usize>;

    async fn generate_streaming(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        options: &GenerateOptions,
        target_language: Language,
        cancel: &AtomicBool,
        on_piece: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<String>;
}

/// Borrowed direct-translation facade over the application's already-loaded
/// model state.
///
/// It carries no model of its own; initial translation and targeted repair
/// therefore reuse the same loaded `Llm` held by [`Model`].
#[derive(Clone, Copy)]
pub struct DirectHskTranslator<'model> {
    model: &'model Model,
}

impl DirectHskTranslator<'_> {
    #[must_use]
    pub const fn model_id(&self) -> ModelId {
        HSK_TRANSLATION_MODEL
    }

    #[must_use]
    pub const fn model_revision(&self) -> &'static str {
        HSK_TRANSLATION_MODEL_REVISION
    }

    #[must_use]
    pub const fn prompt_hash(&self) -> &'static str {
        direct_hsk_prompt_hash()
    }

    #[must_use]
    pub const fn validator_hash(&self) -> &'static str {
        direct_hsk_validator_hash()
    }

    /// Translate directly from English to natural HSK-targeted Simplified
    /// Chinese in one greedy generation. Invalid model lines are returned
    /// beside successful items; the full batch is never retried.
    pub async fn translate_batch(
        &self,
        request: &HskTranslationBatchRequest,
        cancel: &AtomicBool,
    ) -> Result<HskTranslationBatchResult> {
        translate_with(self.model, request, cancel).await
    }

    /// Classify standalone sound effects with the same semantic model used for
    /// translation. The result is advisory and contains only application IDs;
    /// callers can exclude those regions before translation when the user has
    /// disabled sound effects.
    pub async fn classify_semantic_regions(
        &self,
        utterances: &[HskSourceUtterance],
        cancel: &AtomicBool,
    ) -> Result<HskSemanticClassification> {
        classify_semantic_regions_with(self.model, utterances, cancel).await
    }

    /// Discover opaque proper-name source spans after page-role
    /// classification. Keeping this response contract separate prevents
    /// entity output from corrupting story/artwork decisions.
    pub async fn detect_proper_names(
        &self,
        utterances: &[HskSourceUtterance],
        cancel: &AtomicBool,
    ) -> Result<Vec<HskProtectedName>> {
        detect_proper_names_with(self.model, utterances, cancel).await
    }

    /// Translate a batch while publishing each complete numbered line as soon
    /// as it is decoded. Application-owned IDs are restored before the
    /// callback runs.
    pub async fn translate_batch_streaming(
        &self,
        request: &HskTranslationBatchRequest,
        cancel: &AtomicBool,
        on_item: &mut dyn FnMut(&HskTranslationOutcome) -> Result<()>,
    ) -> Result<HskTranslationBatchResult> {
        translate_with_streaming(self.model, request, cancel, on_item).await
    }

    /// Perform the one terminal validation repair for a rejected bubble.
    pub async fn repair_invalid_item(
        &self,
        request: &HskTranslationRepairRequest,
        cancel: &AtomicBool,
    ) -> Result<HskTranslationOutcome> {
        repair_with(self.model, request, cancel).await
    }

    /// Repair several rejected bubbles in one numbered generation. Parsing
    /// and validation remain item-scoped so a malformed sibling cannot hide
    /// or invalidate a usable repair.
    pub async fn repair_invalid_batch(
        &self,
        request: &HskTranslationRepairBatchRequest,
        cancel: &AtomicBool,
    ) -> Result<HskTranslationBatchResult> {
        repair_batch_with(self.model, request, cancel).await
    }
}

async fn detect_proper_names_with<G>(
    generator: &G,
    utterances: &[HskSourceUtterance],
    cancel: &AtomicBool,
) -> Result<Vec<HskProtectedName>>
where
    G: Generator + ?Sized,
{
    check_cancelled(cancel)?;
    if utterances.is_empty() {
        return Ok(Vec::new());
    }
    if utterances.len() > MAX_HSK_TRANSLATION_BATCH {
        bail!(
            "proper-name detection batches may contain at most {MAX_HSK_TRANSLATION_BATCH} items"
        );
    }
    validate_common(
        utterances
            .iter()
            .map(|utterance| (utterance.id.as_str(), utterance.source_english.as_str())),
        &[],
        &[],
    )?;
    let mut user_prompt = String::from("Numbered English OCR lines:\n");
    for (index, utterance) in utterances.iter().enumerate() {
        use std::fmt::Write as _;
        writeln!(
            &mut user_prompt,
            "{}\t{}",
            index + 1,
            compact_field(&utterance.source_english),
        )
        .expect("writing to String cannot fail");
    }
    let system_prompt = format!(
        "Find opaque proper names in exactly {} numbered English comic OCR lines. A proper name is the \
        shortest exact source span that is a lexical identifier for one particular PERSON, PLACE, \
        ORGANIZATION, EVENT, or unique ENTITY. Do not output a complete sentence or clause as a name. \
        Pronouns, determiners, relationships, honorifics, occupations, roles, ranks, titles, species, \
        interface/game categories, ordinary noun phrases, and descriptive epithets remain translatable even when \
        capitalized or unique. A work/chapter title, publisher/site name, logo, watermark, and words \
        functioning as decorative page furniture are not in-story names. For each source line return \
        its position, one tab, then each name as TYPE=exact source spelling separated by ` | `, or `-` \
        when none exist. TYPE must be PERSON, PLACE, ORGANIZATION, EVENT, or ENTITY. Preserve source \
        casing. Return exactly {} ordered non-empty lines and nothing else.",
        utterances.len(),
        utterances.len(),
    );
    let raw = generator
        .generate_streaming(
            &system_prompt,
            &user_prompt,
            &GenerateOptions::greedy(
                16usize
                    .saturating_add(utterances.len().saturating_mul(8))
                    .clamp(MIN_OUTPUT_TOKENS, 96),
            ),
            Language::ChineseSimplified,
            cancel,
            &mut |_| Ok(()),
        )
        .await?;
    check_cancelled(cancel)?;
    trace_semantic_output("proper-name", &raw);
    let candidates = parse_detected_names(&raw, utterances)?;
    verify_proper_name_candidates_with(generator, utterances, &candidates, cancel).await
}

async fn verify_proper_name_candidates_with<G>(
    generator: &G,
    utterances: &[HskSourceUtterance],
    candidates: &[DetectedNameCandidate],
    cancel: &AtomicBool,
) -> Result<Vec<HskProtectedName>>
where
    G: Generator + ?Sized,
{
    check_cancelled(cancel)?;
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let system_prompt = format!(
        "Classify exactly {} candidate English spans by lexical function for keep-original name \
        handling. Return OPAQUE only when the span functions as an arbitrary identifier for a \
        particular person, place, organization, named event, or unique entity in its supplied source \
        occurrences, rather than describing what the referent is. Return DESCRIPTIVE for pronouns, \
        determiners, ordinary words, relationships, honorifics, occupations, \
        roles, ranks, titles, species, interface/game categories, descriptive epithets, and phrases \
        whose component meanings function as the description. The proposed entity type is fallible: \
        an ordinary phenomenon, action, object, or quality used with its normal lexical meaning is \
        DESCRIPTIVE, not a named event or entity. An opaque personal or family surname remains OPAQUE when \
        it identifies a particular family, clan, or house before the separately translatable role word. \
        For a PERSON candidate marked `syntax=definite-article-before`, treat the syntax as strong \
        evidence of a descriptive role, title, or epithet; return OPAQUE only when the candidate \
        itself is demonstrably an opaque lexical identifier rather than ordinary descriptive words. \
        Capitalization and uniqueness alone never make a name. Return exactly {} ordered lines \
        containing position, one tab, and OPAQUE or DESCRIPTIVE; write nothing else.",
        candidates.len(),
        candidates.len(),
    );
    let mut user_prompt = String::from("Candidates and complete source occurrences:\n");
    for (index, candidate) in candidates.iter().enumerate() {
        use std::fmt::Write as _;
        let occurrences = utterances
            .iter()
            .filter(|utterance| {
                source_contains_exact_span(
                    &utterance.source_english,
                    &candidate.name.source_english,
                )
            })
            .map(|utterance| compact_field(&utterance.source_english))
            .collect::<Vec<_>>();
        writeln!(
            &mut user_prompt,
            "{}\ttype={}\tspan={}\tsyntax={}\toccurrences={}",
            index + 1,
            candidate.entity_type,
            compact_field(&candidate.name.source_english),
            candidate_syntax(utterances, &candidate.name.source_english),
            occurrences.join(" || "),
        )
        .expect("writing to String cannot fail");
    }
    let raw = generator
        .generate_streaming(
            &system_prompt,
            &user_prompt,
            &GenerateOptions::greedy(
                32usize
                    .saturating_add(candidates.len().saturating_mul(8))
                    .clamp(48, 96),
            ),
            Language::ChineseSimplified,
            cancel,
            &mut |_| Ok(()),
        )
        .await?;
    check_cancelled(cancel)?;
    trace_semantic_output("proper-name-verifier", &raw);

    let mut keep = vec![false; candidates.len()];
    for raw_line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let Some((position, payload)) = parse_numbered_payload(raw_line) else {
            continue;
        };
        if position == 0 || position > candidates.len() {
            continue;
        }
        if let Some(decision) = parse_keep_original_decision(payload) {
            keep[position - 1] = decision;
        }
    }
    Ok(candidates
        .iter()
        .zip(keep)
        .filter_map(|(candidate, keep)| keep.then(|| candidate.name.clone()))
        .collect())
}

fn parse_keep_original_decision(payload: &str) -> Option<bool> {
    let mut decision = None;
    for token in payload.split(|character: char| !character.is_ascii_alphabetic()) {
        let current = if token.eq_ignore_ascii_case("OPAQUE") {
            Some(true)
        } else if token.eq_ignore_ascii_case("DESCRIPTIVE") {
            Some(false)
        } else {
            None
        };
        let Some(current) = current else {
            continue;
        };
        if decision.is_some_and(|previous| previous != current) {
            return None;
        }
        decision = Some(current);
    }
    decision
}

fn candidate_syntax(utterances: &[HskSourceUtterance], candidate: &str) -> &'static str {
    let candidate_lower = candidate.to_ascii_lowercase();
    let has_definite_article = utterances.iter().any(|utterance| {
        let source_lower = utterance.source_english.to_ascii_lowercase();
        source_lower
            .match_indices(&candidate_lower)
            .any(|(start, _)| {
                utterance.source_english[..start]
                    .split(|character: char| !character.is_ascii_alphabetic() && character != '\'')
                    .filter(|part| !part.is_empty())
                    .next_back()
                    .is_some_and(|word| word.eq_ignore_ascii_case("the"))
            })
    });
    if has_definite_article {
        "definite-article-before"
    } else {
        "neutral"
    }
}

async fn classify_semantic_regions_with<G>(
    generator: &G,
    utterances: &[HskSourceUtterance],
    cancel: &AtomicBool,
) -> Result<HskSemanticClassification>
where
    G: Generator + ?Sized,
{
    check_cancelled(cancel)?;
    if utterances.is_empty() {
        return Ok(HskSemanticClassification {
            page_is_furniture: false,
            regions: Vec::new(),
        });
    }
    if utterances.len() > MAX_HSK_SEMANTIC_PAGE_REGIONS {
        bail!(
            "semantic page classification may contain at most {MAX_HSK_SEMANTIC_PAGE_REGIONS} regions"
        );
    }
    validate_common(
        utterances
            .iter()
            .map(|utterance| (utterance.id.as_str(), utterance.source_english.as_str())),
        &[],
        &[],
    )?;
    let page_dimensions = utterances
        .iter()
        .find_map(|utterance| utterance.semantic_layout.as_ref())
        .map(|layout| format!("{}x{}", layout.page_width, layout.page_height))
        .unwrap_or_else(|| "unknown".to_owned());
    let mut user_prompt = format!(
        "OCR regions from one comic page section (page={page_dimensions}). Layout is fallible supporting evidence:\n"
    );
    for (index, utterance) in utterances.iter().enumerate() {
        use std::fmt::Write as _;
        if let Some(layout) = &utterance.semantic_layout {
            writeln!(
                &mut user_prompt,
                "{}\ttext={}\tdetector={}\tfont-height={:.2}%\tappearance-bands={}\ttext-colors={}\toutline={}\tbox={:.1}%,{:.1}%-{:.1}%,{:.1}%",
                index + 1,
                compact_field(&utterance.source_english),
                if layout.detector_enclosed {
                    "enclosed"
                } else {
                    "free"
                },
                layout.measured_font_height_millionths as f32 / 10_000.0,
                layout.appearance_band_count,
                layout.distinct_text_color_count,
                if layout.has_outline { "yes" } else { "no" },
                layout.x0_millionths as f32 / 10_000.0,
                layout.y0_millionths as f32 / 10_000.0,
                layout.x1_millionths as f32 / 10_000.0,
                layout.y1_millionths as f32 / 10_000.0,
            )
            .expect("writing to String cannot fail");
        } else {
            writeln!(
                &mut user_prompt,
                "{}\ttext={}",
                index + 1,
                compact_field(&utterance.source_english),
            )
            .expect("writing to String cannot fail");
        }
    }
    let system_prompt = format!(
        "Classify one comic page section and exactly {} numbered English OCR regions by semantic function. \
        First output `PAGE`, one tab, then FURNITURE only when the entire section is a cover, work/chapter \
        card, credits/branding cluster, advertisement, navigation, or unreadable OCR with no coherent \
        in-story content; otherwise output `PAGE`, one tab, then STORY. A title plus publisher/staff/branding \
        regions is an all-furniture section even when the title itself is readable. Any coherent dialogue, \
        thought, narration, in-story caption, sign, or world content makes the page section STORY. Then classify each target's \
        own role independently. Layout and detector enclosure are fallible evidence, never hard rules. \
        Output position, one tab, then ARTWORK for readable in-story words whose original letterforms \
        are intentionally part of the illustration—such as a large standalone attack, technique, spell, \
        transformation, or dramatic title treatment outside ordinary dialogue containers—when replacing \
        those letterforms with a standard text overlay would destroy the visual composition. A short \
        fragment may be ARTWORK when nearby regions visibly combine into one named move or form. ARTWORK \
        requires both illustrated visual function and the semantics of a named move, form, spell, \
        transformation, or title. A finite clause, sentence, question, conversational statement, pronoun-led \
        line, or ordinary verb phrase is always STORY regardless of capitalization, font size, border, \
        enclosure, or dramatic emphasis. Ordinary dialogue, narration, captions, signs, interface labels, \
        and plain story text remain STORY. When uncertain between ARTWORK and STORY, output STORY. \
        Output SFX only for a standalone onomatopoeia, sound cue, \
        or nonverbal auditory effect; output FURNITURE only for an unrelated publisher/site credit, \
        watermark, advertisement, reader navigation, work/series title or logo, chapter title/card, or \
        irrecoverable OCR letter soup with no coherent language to translate; output STORY for dialogue, thoughts, \
        narration, in-story labels and titles, names, roles, and ordinary language. Judge meaning \
        in page context rather than word shape, capitalization, or any single coordinate. A prominent title \
        on a cover/card is FURNITURE even when detector-enclosed and centered; a story caption is STORY even \
        when placed at an edge. After the PAGE line, return exactly {} ordered numbered lines, each \
        containing only its position, one tab, and SFX, FURNITURE, ARTWORK, or STORY. Write nothing else.",
        utterances.len(),
        utterances.len(),
    );
    let desired_output_tokens = 20usize
        .saturating_add(utterances.len().saturating_mul(5))
        .clamp(40, 768);
    let minimum_output_tokens = 8usize
        .saturating_add(utterances.len().saturating_mul(3))
        .clamp(24, desired_output_tokens);
    let completion_capacity = generator
        .constrained_completion_capacity(&system_prompt, &user_prompt, Language::ChineseSimplified)
        .await?;
    if completion_capacity < minimum_output_tokens {
        bail!(
            "semantic page section with {} regions cannot fit the resident model context",
            utterances.len()
        );
    }
    let raw = generator
        .generate_streaming(
            &system_prompt,
            &user_prompt,
            &GenerateOptions::greedy(completion_capacity.min(desired_output_tokens)),
            Language::ChineseSimplified,
            cancel,
            &mut |_| Ok(()),
        )
        .await?;
    check_cancelled(cancel)?;
    trace_semantic_output("exclusion", &raw);
    let mut classification = parse_semantic_classification(&raw, utterances);
    if classification.page_is_furniture {
        return Ok(classification);
    }
    let semantic_exclusions = classification
        .regions
        .iter()
        .filter(|(_, disposition)| {
            matches!(
                disposition,
                HskTranslationDisposition::ExcludeNonStory
                    | HskTranslationDisposition::ExcludeSoundEffect
            )
        })
        .map(|(id, _)| id.as_str())
        .collect::<HashSet<_>>();
    let proposed_artwork = utterances
        .iter()
        .filter(|utterance| !semantic_exclusions.contains(utterance.id.as_str()))
        .filter(|utterance| artwork_claim_is_admissible(utterance))
        .collect::<Vec<_>>();
    let approved_artwork =
        verify_artwork_candidates_with(generator, utterances, &proposed_artwork, cancel).await?;
    let existing_dispositions = classification
        .regions
        .drain(..)
        .filter(|(_, disposition)| *disposition != HskTranslationDisposition::PreserveArtwork)
        .collect::<HashMap<_, _>>();
    classification.regions = utterances
        .iter()
        .filter_map(|utterance| {
            if approved_artwork.contains(&utterance.id) {
                Some((
                    utterance.id.clone(),
                    HskTranslationDisposition::PreserveArtwork,
                ))
            } else {
                existing_dispositions
                    .get(&utterance.id)
                    .copied()
                    .map(|disposition| (utterance.id.clone(), disposition))
            }
        })
        .collect();
    Ok(classification)
}

fn trace_semantic_output(stage: &str, output: &str) {
    if std::env::var_os("HSKIFY_TRACE_REJECTED_OCR").is_some_and(|value| value == "1") {
        eprintln!("hskify-semantic-{stage}-output={output:?}");
    }
}

async fn verify_artwork_candidates_with<G>(
    generator: &G,
    page_context: &[HskSourceUtterance],
    candidates: &[&HskSourceUtterance],
    cancel: &AtomicBool,
) -> Result<HashSet<String>>
where
    G: Generator + ?Sized,
{
    check_cancelled(cancel)?;
    if candidates.is_empty() {
        return Ok(HashSet::new());
    }
    let system_prompt = format!(
        "Adjudicate exactly {} proposed comic ARTWORK regions. Return PRESERVE only when the words \
        are visually integral illustrated lettering for a named attack, technique, form, spell, \
        transformation, or dramatic story title and replacing their original letterforms with a \
        standard overlay would damage the composition. Use the complete ordered section context \
        to distinguish an illustrated label from a short spoken or narrated story fragment. A \
        candidate that can plausibly be a character's utterance or narration in that context must \
        be TRANSLATE, including greetings, toasts, interjections, reactions, warnings, observations, \
        and announcements. Dramatic wording or typography alone never makes such story text artwork. \
        Return TRANSLATE for dialogue, narration, \
        finite clauses, quantities, rewards, item names or descriptions, missions, instructions, \
        interface or game labels, ordinary signs, roles, titles of people, sound effects, credits, \
        watermarks, and uncertain cases, even when large, outlined, colorful, capitalized, or \
        detector-enclosed. Return exactly {} ordered lines containing position, one tab, and \
        PRESERVE or TRANSLATE; write nothing else.",
        candidates.len(),
        candidates.len(),
    );
    let mut user_prompt = String::from("Complete ordered section context:\n");
    for (index, utterance) in page_context.iter().enumerate() {
        use std::fmt::Write as _;
        writeln!(
            &mut user_prompt,
            "{}\ttext={}",
            index + 1,
            compact_field(&utterance.source_english),
        )
        .expect("writing to String cannot fail");
    }
    user_prompt.push_str("\nProposed illustrated-lettering regions:\n");
    for (index, candidate) in candidates.iter().enumerate() {
        use std::fmt::Write as _;
        let layout = candidate
            .semantic_layout
            .as_ref()
            .expect("artwork candidates require visual evidence");
        let page_position = page_context
            .iter()
            .position(|utterance| utterance.id == candidate.id)
            .map_or(0, |position| position + 1);
        writeln!(
            &mut user_prompt,
            "{}\tpage-position={}\ttext={}\tdetector={}\tfont-height={:.2}%\tappearance-bands={}\ttext-colors={}\toutline={}",
            index + 1,
            page_position,
            compact_field(&candidate.source_english),
            if layout.detector_enclosed {
                "enclosed"
            } else {
                "free"
            },
            layout.measured_font_height_millionths as f32 / 10_000.0,
            layout.appearance_band_count,
            layout.distinct_text_color_count,
            if layout.has_outline { "yes" } else { "no" },
        )
        .expect("writing to String cannot fail");
    }
    let raw = generator
        .generate_streaming(
            &system_prompt,
            &user_prompt,
            &GenerateOptions::greedy(
                32usize
                    .saturating_add(candidates.len().saturating_mul(8))
                    .clamp(48, 96),
            ),
            Language::ChineseSimplified,
            cancel,
            &mut |_| Ok(()),
        )
        .await?;
    check_cancelled(cancel)?;
    trace_semantic_output("artwork-verifier", &raw);
    let mut approved = HashSet::new();
    for raw_line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let Some((position, payload)) = parse_numbered_payload(raw_line) else {
            continue;
        };
        if position == 0 || position > candidates.len() {
            continue;
        }
        let mut decision = None;
        for token in payload.split(|character: char| !character.is_ascii_alphabetic()) {
            let current = if token.eq_ignore_ascii_case("PRESERVE") {
                Some(true)
            } else if token.eq_ignore_ascii_case("TRANSLATE") {
                Some(false)
            } else {
                None
            };
            let Some(current) = current else {
                continue;
            };
            if decision.is_some_and(|previous| previous != current) {
                decision = None;
                break;
            }
            decision = Some(current);
        }
        if decision == Some(true) {
            approved.insert(candidates[position - 1].id.clone());
        }
    }
    Ok(approved)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DetectedNameCandidate {
    name: HskProtectedName,
    entity_type: String,
}

fn parse_detected_names(
    output: &str,
    utterances: &[HskSourceUtterance],
) -> Result<Vec<DetectedNameCandidate>> {
    const ENTITY_TYPES: [&str; 5] = ["PERSON", "PLACE", "ORGANIZATION", "EVENT", "ENTITY"];
    let mut slots = vec![None::<Vec<(String, String)>>; utterances.len()];
    for raw_line in output.lines().filter(|line| !line.trim().is_empty()) {
        let Some((position, payload)) = parse_numbered_payload(raw_line) else {
            continue;
        };
        if position == 0 || position > utterances.len() || slots[position - 1].is_some() {
            continue;
        }
        let source = &utterances[position - 1].source_english;
        let mut names = Vec::<(String, String)>::new();
        if payload.trim() != "-" {
            for discovered_candidate in payload
                .split(|character| matches!(character, '|' | '\t'))
                .map(str::trim)
            {
                let (entity_type, candidate) =
                    if let Some((left, right)) = discovered_candidate.split_once('=') {
                        let left_is_type = ENTITY_TYPES
                            .iter()
                            .any(|allowed| left.trim().eq_ignore_ascii_case(allowed));
                        let right_is_type = ENTITY_TYPES
                            .iter()
                            .any(|allowed| right.trim().eq_ignore_ascii_case(allowed));
                        if left_is_type {
                            (left.trim(), right.trim())
                        } else if right_is_type {
                            (right.trim(), left.trim())
                        } else {
                            continue;
                        }
                    } else {
                        // Discovery proposes candidates; contextual adjudication
                        // remains the authority. Preserve an exact untyped span
                        // as UNKNOWN so harmless schema drift cannot erase a
                        // valid candidate before the independent verifier sees
                        // every source occurrence.
                        ("UNKNOWN", discovered_candidate)
                    };
                let Some(candidate) = canonical_source_span(source, candidate) else {
                    continue;
                };
                if !opaque_name_span_is_admissible(source, candidate) {
                    continue;
                }
                if !names
                    .iter()
                    .any(|(name, _)| name.eq_ignore_ascii_case(candidate))
                {
                    names.push((candidate.to_owned(), entity_type.to_ascii_uppercase()));
                }
            }
        }
        slots[position - 1] = Some(names);
    }
    let mut detected = Vec::new();
    for names in slots.into_iter().flatten() {
        for (name, entity_type) in names {
            if detected.iter().any(|existing: &DetectedNameCandidate| {
                existing.name.source_english.eq_ignore_ascii_case(&name)
            }) {
                continue;
            }
            detected.push(DetectedNameCandidate {
                name: HskProtectedName {
                    source_english: name.clone(),
                    chinese: name,
                },
                entity_type,
            });
        }
    }
    Ok(detected)
}

fn opaque_name_span_is_admissible(source: &str, candidate: &str) -> bool {
    const CLOSED_CLASS_SINGLE_WORDS: &[&str] = &[
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "they",
        "us",
        "we",
        "you",
        "your",
        "yours",
        "aren't",
        "can't",
        "couldn't",
        "didn't",
        "doesn't",
        "don't",
        "hadn't",
        "hasn't",
        "haven't",
        "isn't",
        "shouldn't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
    ];
    const POSSESSIVE_DETERMINERS: &[&str] = &["my", "your", "his", "her", "our", "their", "its"];
    let trimmed = candidate.trim();
    let word_count = trimmed.split_whitespace().count();
    if word_count == 0
        || (word_count == 1
            && CLOSED_CLASS_SINGLE_WORDS
                .iter()
                .any(|word| trimmed.eq_ignore_ascii_case(word)))
    {
        return false;
    }
    let source_trimmed = source.trim();
    if trimmed.eq_ignore_ascii_case(source_trimmed)
        && (word_count > 6
            || (word_count > 1 && source_trimmed.ends_with('.'))
            || source_trimmed
                .chars()
                .any(|character| matches!(character, '!' | '?' | ';')))
    {
        return false;
    }
    let source_lower = source.to_ascii_lowercase();
    let candidate_lower = trimmed.to_ascii_lowercase();
    if let Some(start) = source_lower.find(&candidate_lower) {
        let previous = source[..start]
            .split(|character: char| !character.is_ascii_alphabetic() && character != '\'')
            .filter(|part| !part.is_empty())
            .next_back()
            .unwrap_or("");
        if POSSESSIVE_DETERMINERS
            .iter()
            .any(|word| previous.eq_ignore_ascii_case(word))
        {
            return false;
        }
    }
    true
}

fn artwork_claim_is_admissible(utterance: &HskSourceUtterance) -> bool {
    const PROMINENT_FONT_HEIGHT_MILLIONTHS: u32 = 100_000;
    const FREEFORM_STYLED_FONT_HEIGHT_MILLIONTHS: u32 = 60_000;
    const STORY_LEADING_WORDS: &[&str] = &[
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "who",
        "what",
        "why",
        "how",
        "when",
        "where",
        "am",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "can't",
        "couldn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "haven't",
        "hasn't",
        "hadn't",
    ];
    let trimmed = utterance.source_english.trim();
    if trimmed.ends_with('?')
        || trimmed.ends_with('.')
        || trimmed.chars().any(|character| character.is_ascii_digit())
        || english_word_count(trimmed) > 6
    {
        return false;
    }
    let first = trimmed
        .split(|character: char| !character.is_ascii_alphabetic() && character != '\'')
        .find(|part| !part.is_empty())
        .unwrap_or("");
    let semantics_are_admissible = !STORY_LEADING_WORDS
        .iter()
        .any(|word| first.eq_ignore_ascii_case(word));
    if !semantics_are_admissible {
        return false;
    }
    let Some(layout) = utterance.semantic_layout.as_ref() else {
        return false;
    };
    let visually_prominent =
        layout.measured_font_height_millionths >= PROMINENT_FONT_HEIGHT_MILLIONTHS;
    let styled_freeform = !layout.detector_enclosed
        && layout.measured_font_height_millionths >= FREEFORM_STYLED_FONT_HEIGHT_MILLIONTHS
        && (layout.has_outline
            || layout.distinct_text_color_count > 1
            || layout.appearance_band_count > 1);
    visually_prominent || styled_freeform
}

fn parse_semantic_classification(
    output: &str,
    utterances: &[HskSourceUtterance],
) -> HskSemanticClassification {
    let page_category = output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .find_map(|line| {
            let (label, category) = line.split_once('\t')?;
            if !label.trim().eq_ignore_ascii_case("PAGE") {
                return None;
            }
            Some(category.trim().to_ascii_uppercase())
        });
    let mut decisions = vec![None::<HskTranslationDisposition>; utterances.len()];
    let mut raw_categories = vec![None::<String>; utterances.len()];
    for raw_line in output.lines().filter(|line| !line.trim().is_empty()) {
        let Some((position, payload)) = parse_numbered_payload(raw_line) else {
            continue;
        };
        if position == 0 || position > utterances.len() || decisions[position - 1].is_some() {
            continue;
        }
        let mut columns = payload.split('\t');
        let category = columns
            .next()
            .unwrap_or(payload)
            .split_whitespace()
            .next()
            .unwrap_or(payload);
        let category = category.to_ascii_uppercase();
        raw_categories[position - 1] = Some(category.clone());
        decisions[position - 1] = match category.as_str() {
            "SFX" => Some(HskTranslationDisposition::ExcludeSoundEffect),
            "FURNITURE" => Some(HskTranslationDisposition::ExcludeNonStory),
            "ARTWORK" if artwork_claim_is_admissible(&utterances[position - 1]) => {
                Some(HskTranslationDisposition::PreserveArtwork)
            }
            "ARTWORK" => Some(HskTranslationDisposition::Translate),
            "STORY" => Some(HskTranslationDisposition::Translate),
            _ => None,
        };
    }
    let complete_non_story_artwork_page = page_category.as_deref() == Some("ARTWORK")
        && raw_categories
            .iter()
            .any(|category| category.as_deref() == Some("FURNITURE"))
        && raw_categories
            .iter()
            .all(|category| matches!(category.as_deref(), Some("SFX" | "FURNITURE" | "ARTWORK")));
    let page_is_furniture =
        page_category.as_deref() == Some("FURNITURE") || complete_non_story_artwork_page;
    let regions = decisions
        .into_iter()
        .enumerate()
        .filter_map(|(index, decision)| match decision {
            Some(
                disposition @ (HskTranslationDisposition::ExcludeSoundEffect
                | HskTranslationDisposition::ExcludeNonStory
                | HskTranslationDisposition::PreserveArtwork),
            ) => Some((utterances[index].id.clone(), disposition)),
            Some(HskTranslationDisposition::Translate) | None => None,
        })
        .collect();
    HskSemanticClassification {
        page_is_furniture,
        regions,
    }
}

fn parse_numbered_payload(line: &str) -> Option<(usize, &str)> {
    let line = line.trim_start();
    let digit_count = line
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 {
        return None;
    }
    let position = line[..digit_count].parse::<usize>().ok()?;
    let remainder = &line[digit_count..];
    let separator = remainder.chars().next()?;
    if !matches!(
        separator,
        '\t' | ' ' | '.' | '．' | ')' | '）' | ':' | '：' | '、'
    ) && !separator.is_whitespace()
    {
        return None;
    }
    let separator_length = separator.len_utf8();
    let payload = remainder[separator_length..].trim();
    (!payload.is_empty()).then_some((position, payload))
}

impl Generator for Model {
    async fn token_count(&self, text: &str) -> Result<usize> {
        let state = self.state.read().await;
        match &*state {
            State::ReadyLocal(llm) if llm.id() == HSK_TRANSLATION_MODEL => llm.token_count(text),
            State::ReadyLocal(llm) => bail!(
                "direct HSK translation requires local model `{}`, but `{}` is loaded",
                HSK_TRANSLATION_MODEL,
                llm.id()
            ),
            State::ReadyProvider { .. } => {
                bail!("direct HSK translation is local-only; remote providers are disabled")
            }
            State::Loading { .. } => bail!("direct HSK translation model is still loading"),
            State::Failed { error, .. } => {
                bail!("direct HSK translation model failed to load: {error}")
            }
            State::Empty => {
                bail!("direct HSK translation model `{HSK_TRANSLATION_MODEL}` is not loaded")
            }
        }
    }

    async fn constrained_completion_capacity(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        target_language: Language,
    ) -> Result<usize> {
        let state = self.state.read().await;
        match &*state {
            State::ReadyLocal(llm) if llm.id() == HSK_TRANSLATION_MODEL => {
                llm.constrained_completion_capacity(user_prompt, target_language, system_prompt)
            }
            State::ReadyLocal(llm) => bail!(
                "direct HSK translation requires local model `{}`, but `{}` is loaded",
                HSK_TRANSLATION_MODEL,
                llm.id()
            ),
            State::ReadyProvider { .. } => {
                bail!("direct HSK translation is local-only; remote providers are disabled")
            }
            State::Loading { .. } => bail!("direct HSK translation model is still loading"),
            State::Failed { error, .. } => {
                bail!("direct HSK translation model failed to load: {error}")
            }
            State::Empty => {
                bail!("direct HSK translation model `{HSK_TRANSLATION_MODEL}` is not loaded")
            }
        }
    }

    async fn generate_streaming(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        options: &GenerateOptions,
        target_language: Language,
        cancel: &AtomicBool,
        on_piece: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        let mut state = self.state.write().await;
        let llm = match &mut *state {
            State::ReadyLocal(llm) if llm.id() == HSK_TRANSLATION_MODEL => llm,
            State::ReadyLocal(llm) => bail!(
                "direct HSK translation requires local model `{}`, but `{}` is loaded",
                HSK_TRANSLATION_MODEL,
                llm.id()
            ),
            State::ReadyProvider { .. } => {
                bail!("direct HSK translation is local-only; remote providers are disabled")
            }
            State::Loading { .. } => bail!("direct HSK translation model is still loading"),
            State::Failed { error, .. } => {
                bail!("direct HSK translation model failed to load: {error}")
            }
            State::Empty => {
                bail!("direct HSK translation model `{HSK_TRANSLATION_MODEL}` is not loaded")
            }
        };

        llm.generate_constrained_streaming(
            user_prompt,
            options,
            target_language,
            system_prompt,
            cancel,
            on_piece,
        )
    }
}

impl Model {
    /// Access direct HSK translation without loading or copying model state.
    #[must_use]
    pub const fn direct_hsk_translator(&self) -> DirectHskTranslator<'_> {
        DirectHskTranslator { model: self }
    }
}

async fn translate_with<G>(
    generator: &G,
    request: &HskTranslationBatchRequest,
    cancel: &AtomicBool,
) -> Result<HskTranslationBatchResult>
where
    G: Generator + ?Sized,
{
    translate_with_streaming(generator, request, cancel, &mut |_| Ok(())).await
}

async fn translate_with_streaming<G>(
    generator: &G,
    request: &HskTranslationBatchRequest,
    cancel: &AtomicBool,
    on_item: &mut dyn FnMut(&HskTranslationOutcome) -> Result<()>,
) -> Result<HskTranslationBatchResult>
where
    G: Generator + ?Sized,
{
    check_cancelled(cancel)?;
    validate_translation_request(request)?;
    if request.utterances.is_empty() {
        return Ok(HskTranslationBatchResult { items: Vec::new() });
    }

    let mut remaining = request.utterances.as_slice();
    let mut rolling_context = request.preceding_utterances.clone();
    let mut items = Vec::with_capacity(request.utterances.len());
    while !remaining.is_empty() {
        check_cancelled(cancel)?;
        let (bounded_request, options) =
            plan_translation_subbatch(generator, request, remaining, &rolling_context).await?;
        let consumed = bounded_request.utterances.len();
        let result = translate_prepared_request_streaming(
            generator,
            &bounded_request,
            options,
            cancel,
            on_item,
        )
        .await?;
        for outcome in &result.items {
            if !outcome.is_valid() {
                continue;
            }
            let Some(source) = bounded_request
                .utterances
                .iter()
                .find(|utterance| utterance.id == outcome.id)
            else {
                continue;
            };
            rolling_context.push(HskPrecedingUtterance {
                source_english: source.source_english.clone(),
                chinese: outcome.text.clone().expect("valid outcome has text"),
            });
        }
        if rolling_context.len() > MAX_HSK_PRECEDING_UTTERANCES {
            rolling_context.drain(..rolling_context.len() - MAX_HSK_PRECEDING_UTTERANCES);
        }
        items.extend(result.items);
        remaining = &remaining[consumed..];
    }
    Ok(HskTranslationBatchResult { items })
}

async fn plan_translation_subbatch<G>(
    generator: &G,
    request: &HskTranslationBatchRequest,
    remaining: &[HskSourceUtterance],
    rolling_context: &[HskPrecedingUtterance],
) -> Result<(HskTranslationBatchRequest, GenerateOptions)>
where
    G: Generator + ?Sized,
{
    for count in (1..=remaining.len().min(MAX_HSK_TRANSLATION_BATCH)).rev() {
        let mut candidate = request.clone();
        candidate.utterances = remaining[..count].to_vec();
        candidate.preceding_utterances = bounded_context(generator, rolling_context).await?;
        loop {
            let prompt = build_translation_prompt(&candidate);
            let system_prompt = translation_system_prompt(
                candidate.requested_level,
                candidate.utterances.len(),
                candidate.name_handling,
                candidate.translate_sound_effects,
                candidate.learning_mode,
            );
            let desired_output_tokens = output_token_budget(
                candidate
                    .utterances
                    .iter()
                    .map(|utterance| utterance.source_english.as_str()),
                candidate.utterances.len(),
            );
            let completion_capacity = generator
                .constrained_completion_capacity(
                    &system_prompt,
                    &prompt,
                    Language::ChineseSimplified,
                )
                .await?;
            if completion_capacity >= desired_output_tokens {
                return Ok((candidate, GenerateOptions::greedy(desired_output_tokens)));
            }
            if !candidate.preceding_utterances.is_empty() {
                candidate.preceding_utterances.remove(0);
                continue;
            }
            if count == 1 && completion_capacity >= MIN_OUTPUT_TOKENS {
                return Ok((
                    candidate,
                    GenerateOptions::greedy(completion_capacity.min(desired_output_tokens)),
                ));
            }
            break;
        }
    }
    bail!(
        "one OCR utterance cannot fit the resident translation context even after removing preceding context"
    )
}

async fn translate_prepared_request_streaming<G>(
    generator: &G,
    bounded_request: &HskTranslationBatchRequest,
    options: GenerateOptions,
    cancel: &AtomicBool,
    on_item: &mut dyn FnMut(&HskTranslationOutcome) -> Result<()>,
) -> Result<HskTranslationBatchResult>
where
    G: Generator + ?Sized,
{
    let prompt = build_translation_prompt(&bounded_request);
    let expected = bounded_request
        .utterances
        .iter()
        .map(|utterance| ExpectedUtterance {
            id: &utterance.id,
            source_english: &utterance.source_english,
        })
        .collect::<Vec<_>>();
    let mut streamed_ids = HashSet::with_capacity(expected.len());
    let mut pending_line = String::new();
    let mut publish_piece = |piece: &str| -> Result<()> {
        pending_line.push_str(piece);
        while let Some(newline) = pending_line.find('\n') {
            let mut tail = pending_line.split_off(newline + 1);
            std::mem::swap(&mut tail, &mut pending_line);
            let completed = tail.strip_suffix('\n').unwrap_or(&tail);
            let completed = completed.strip_suffix('\r').unwrap_or(completed);
            if let Some(outcome) = parse_streamed_line(
                completed,
                &expected,
                &bounded_request.protected_names,
                bounded_request.name_handling,
                bounded_request.translate_sound_effects,
            ) && streamed_ids.insert(outcome.id.clone())
            {
                on_item(&outcome)?;
            }
        }
        Ok(())
    };
    let raw = generator
        .generate_streaming(
            &translation_system_prompt(
                bounded_request.requested_level,
                bounded_request.utterances.len(),
                bounded_request.name_handling,
                bounded_request.translate_sound_effects,
                bounded_request.learning_mode,
            ),
            &prompt,
            &options,
            Language::ChineseSimplified,
            cancel,
            &mut publish_piece,
        )
        .await?;
    check_cancelled(cancel)?;

    let result = parse_numbered_output_with_name_handling(
        &raw,
        &expected,
        &bounded_request.protected_names,
        bounded_request.name_handling,
        bounded_request.translate_sound_effects,
    );
    for outcome in &result.items {
        if streamed_ids.insert(outcome.id.clone()) {
            on_item(outcome)?;
        }
    }
    Ok(result)
}

async fn repair_with<G>(
    generator: &G,
    request: &HskTranslationRepairRequest,
    cancel: &AtomicBool,
) -> Result<HskTranslationOutcome>
where
    G: Generator + ?Sized,
{
    check_cancelled(cancel)?;
    validate_repair_request(request)?;

    let mut bounded_request = request.clone();
    bounded_request.preceding_utterances =
        bounded_context(generator, &request.preceding_utterances).await?;
    let prompt = build_repair_prompt(&bounded_request);
    let options = GenerateOptions::greedy(output_token_budget(
        std::iter::once(bounded_request.utterance.source_english.as_str()),
        1,
    ));
    let raw = generator
        .generate_streaming(
            &repair_system_prompt(
                bounded_request.requested_level,
                bounded_request.name_handling,
                bounded_request.translate_sound_effects,
                bounded_request.learning_mode,
            ),
            &prompt,
            &options,
            Language::ChineseSimplified,
            cancel,
            &mut |_| Ok(()),
        )
        .await?;
    check_cancelled(cancel)?;

    Ok(parse_repair_output_with_name_handling(
        &raw,
        &ExpectedUtterance {
            id: &bounded_request.utterance.id,
            source_english: &bounded_request.utterance.source_english,
        },
        &bounded_request.protected_names,
        bounded_request.name_handling,
        bounded_request.translate_sound_effects,
    ))
}

async fn repair_batch_with<G>(
    generator: &G,
    request: &HskTranslationRepairBatchRequest,
    cancel: &AtomicBool,
) -> Result<HskTranslationBatchResult>
where
    G: Generator + ?Sized,
{
    check_cancelled(cancel)?;
    validate_repair_batch_request(request)?;
    if request.utterances.is_empty() {
        return Ok(HskTranslationBatchResult { items: Vec::new() });
    }

    let mut remaining = request.utterances.as_slice();
    let mut items = Vec::with_capacity(request.utterances.len());
    while !remaining.is_empty() {
        check_cancelled(cancel)?;
        let (bounded_request, options) =
            plan_repair_subbatch(generator, request, remaining).await?;
        let consumed = bounded_request.utterances.len();
        items.extend(
            repair_prepared_batch(generator, &bounded_request, options, cancel)
                .await?
                .items,
        );
        remaining = &remaining[consumed..];
    }
    Ok(HskTranslationBatchResult { items })
}

async fn plan_repair_subbatch<G>(
    generator: &G,
    request: &HskTranslationRepairBatchRequest,
    remaining: &[HskRepairUtterance],
) -> Result<(HskTranslationRepairBatchRequest, GenerateOptions)>
where
    G: Generator + ?Sized,
{
    for count in (1..=remaining.len().min(MAX_HSK_TRANSLATION_BATCH)).rev() {
        let mut candidate = request.clone();
        candidate.utterances = remaining[..count].to_vec();
        // Repairs are deliberately self-contained: carrying prior Chinese
        // answers into this prompt encourages copying rejected output and
        // consumes context without helping the correction.
        candidate.preceding_utterances.clear();
        let prompt = build_repair_batch_prompt(&candidate);
        let system_prompt = repair_batch_system_prompt_with_learning_policy(
            candidate.requested_level,
            candidate.utterances.len(),
            candidate.name_handling.into(),
            candidate.translate_sound_effects,
            candidate.learning_mode.into(),
        );
        let desired_output_tokens = output_token_budget(
            candidate
                .utterances
                .iter()
                .map(|utterance| utterance.source_english.as_str()),
            candidate.utterances.len(),
        );
        let completion_capacity = generator
            .constrained_completion_capacity(&system_prompt, &prompt, Language::ChineseSimplified)
            .await?;
        if completion_capacity >= desired_output_tokens {
            return Ok((candidate, GenerateOptions::greedy(desired_output_tokens)));
        }
        if count == 1 && completion_capacity >= MIN_OUTPUT_TOKENS {
            return Ok((
                candidate,
                GenerateOptions::greedy(completion_capacity.min(desired_output_tokens)),
            ));
        }
    }
    bail!("one rejected OCR utterance cannot fit the resident translation context")
}

async fn repair_prepared_batch<G>(
    generator: &G,
    bounded_request: &HskTranslationRepairBatchRequest,
    options: GenerateOptions,
    cancel: &AtomicBool,
) -> Result<HskTranslationBatchResult>
where
    G: Generator + ?Sized,
{
    let prompt = build_repair_batch_prompt(&bounded_request);
    let raw = generator
        .generate_streaming(
            &repair_batch_system_prompt_with_learning_policy(
                bounded_request.requested_level,
                bounded_request.utterances.len(),
                bounded_request.name_handling.into(),
                bounded_request.translate_sound_effects,
                bounded_request.learning_mode.into(),
            ),
            &prompt,
            &options,
            Language::ChineseSimplified,
            cancel,
            &mut |_| Ok(()),
        )
        .await?;
    check_cancelled(cancel)?;
    trace_semantic_output("repair-batch", &raw);

    let expected = bounded_request
        .utterances
        .iter()
        .map(|utterance| ExpectedUtterance {
            id: &utterance.id,
            source_english: &utterance.source_english,
        })
        .collect::<Vec<_>>();
    Ok(parse_repair_batch_output(
        &raw,
        &expected,
        &bounded_request.protected_names,
        bounded_request.name_handling,
        bounded_request.translate_sound_effects,
    ))
}

async fn bounded_context<G>(
    generator: &G,
    context: &[HskPrecedingUtterance],
) -> Result<Vec<HskPrecedingUtterance>>
where
    G: Generator + ?Sized,
{
    let start = context.len().saturating_sub(MAX_HSK_PRECEDING_UTTERANCES);
    let mut bounded = context[start..].to_vec();
    while !bounded.is_empty()
        && generator
            .token_count(&render_context_for_budget(&bounded))
            .await?
            > MAX_HSK_CONTEXT_TOKENS
    {
        bounded.remove(0);
    }
    Ok(bounded)
}

fn render_context_for_budget(context: &[HskPrecedingUtterance]) -> String {
    let context = context
        .iter()
        .map(|utterance| DirectHskContext {
            source_english: &utterance.source_english,
            chinese: &utterance.chinese,
        })
        .collect::<Vec<_>>();
    context_budget_text(&context)
}

fn translation_system_prompt(
    level: u8,
    count: usize,
    name_handling: HskNameHandling,
    translate_sound_effects: bool,
    learning_mode: HskLearningMode,
) -> String {
    primary_system_prompt_with_learning_policy(
        level,
        count,
        name_handling.into(),
        translate_sound_effects,
        learning_mode.into(),
    )
}

fn repair_system_prompt(
    level: u8,
    name_handling: HskNameHandling,
    translate_sound_effects: bool,
    learning_mode: HskLearningMode,
) -> String {
    repair_system_prompt_with_learning_policy(
        level,
        name_handling.into(),
        translate_sound_effects,
        learning_mode.into(),
    )
}

fn build_translation_prompt(request: &HskTranslationBatchRequest) -> String {
    let context = request
        .preceding_utterances
        .iter()
        .map(|utterance| DirectHskContext {
            source_english: &utterance.source_english,
            chinese: &utterance.chinese,
        })
        .collect::<Vec<_>>();
    let names = request
        .protected_names
        .iter()
        .map(|name| DirectHskName {
            source_english: &name.source_english,
            chinese: &name.chinese,
        })
        .collect::<Vec<_>>();
    let sources = request
        .utterances
        .iter()
        .map(|utterance| utterance.source_english.as_str())
        .collect::<Vec<_>>();
    primary_user_prompt_with_name_style(&context, &names, &sources, request.name_handling.into())
}

fn build_repair_prompt(request: &HskTranslationRepairRequest) -> String {
    // A singular repair must not be able to copy a preceding Chinese answer.
    // The source and protected names are sufficient to repair one rejected
    // item; context remains a primary-generation aid.
    let utterance = &request.utterance;
    let problems = utterance
        .problems
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let names = request
        .protected_names
        .iter()
        .map(|name| DirectHskName {
            source_english: &name.source_english,
            chinese: &name.chinese,
        })
        .collect::<Vec<_>>();
    repair_user_prompt_with_constraints(
        &utterance.source_english,
        utterance.rejected_chinese.as_deref(),
        &problems,
        &names,
        request.name_handling.into(),
        &utterance.avoid_chinese,
    )
}

fn build_repair_batch_prompt(request: &HskTranslationRepairBatchRequest) -> String {
    let names = request
        .protected_names
        .iter()
        .map(|name| DirectHskName {
            source_english: &name.source_english,
            chinese: &name.chinese,
        })
        .collect::<Vec<_>>();
    let mut prompt = String::from("Rejected items:\n");
    for (index, utterance) in request.utterances.iter().enumerate() {
        use std::fmt::Write as _;
        let problems = utterance
            .problems
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let constraints = repair_item_constraints(
            &utterance.source_english,
            utterance.rejected_chinese.as_deref(),
            &problems,
            &names,
            request.name_handling.into(),
            &utterance.avoid_chinese,
        );
        writeln!(&mut prompt, "Item {}:\n{}", index + 1, constraints)
            .expect("writing to String cannot fail");
    }
    prompt.push_str("Corrected numbered lines:");
    prompt
}

fn compact_field(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn output_token_budget<'a>(
    sources: impl IntoIterator<Item = &'a str>,
    utterance_count: usize,
) -> usize {
    let source_chars = sources
        .into_iter()
        .map(str::chars)
        .map(Iterator::count)
        .sum::<usize>();
    let translated_text = source_chars.div_ceil(2);
    translated_text
        .saturating_add(
            utterance_count
                .saturating_mul(OUTPUT_TOKENS_PER_UTTERANCE)
                .saturating_add(8),
        )
        .clamp(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
}

fn validate_translation_request(request: &HskTranslationBatchRequest) -> Result<()> {
    validate_level(request.requested_level)?;
    if request.utterances.len() > MAX_HSK_TRANSLATION_BATCH {
        bail!("HSK translation batches may contain at most {MAX_HSK_TRANSLATION_BATCH} utterances");
    }
    validate_common(
        request
            .utterances
            .iter()
            .map(|utterance| (utterance.id.as_str(), utterance.source_english.as_str())),
        &request.preceding_utterances,
        &request.protected_names,
    )
}

fn validate_repair_request(request: &HskTranslationRepairRequest) -> Result<()> {
    validate_level(request.requested_level)?;
    validate_common(
        std::iter::once((
            request.utterance.id.as_str(),
            request.utterance.source_english.as_str(),
        )),
        &request.preceding_utterances,
        &request.protected_names,
    )?;
    validate_repair_utterance(&request.utterance)
}

fn validate_repair_batch_request(request: &HskTranslationRepairBatchRequest) -> Result<()> {
    validate_level(request.requested_level)?;
    if request.utterances.len() > MAX_HSK_TRANSLATION_BATCH {
        bail!("HSK repair batches may contain at most {MAX_HSK_TRANSLATION_BATCH} utterances");
    }
    validate_common(
        request
            .utterances
            .iter()
            .map(|utterance| (utterance.id.as_str(), utterance.source_english.as_str())),
        &request.preceding_utterances,
        &request.protected_names,
    )?;
    for utterance in &request.utterances {
        validate_repair_utterance(utterance)?;
    }
    Ok(())
}

fn validate_repair_utterance(utterance: &HskRepairUtterance) -> Result<()> {
    if utterance.problems.is_empty()
        || utterance
            .problems
            .iter()
            .any(|problem| problem.trim().is_empty())
    {
        bail!(
            "targeted HSK repair item `{}` requires non-empty problems",
            utterance.id
        );
    }
    if utterance
        .rejected_chinese
        .as_deref()
        .is_some_and(|text| text.trim().is_empty())
    {
        bail!(
            "targeted HSK repair item `{}` has an empty rejected translation",
            utterance.id
        );
    }
    if utterance.avoid_chinese.len() > 32
        || utterance
            .avoid_chinese
            .iter()
            .any(|term| term.trim().is_empty())
    {
        bail!(
            "targeted HSK repair item `{}` requires at most 32 non-empty validator avoid terms",
            utterance.id
        );
    }
    Ok(())
}

fn validate_level(level: u8) -> Result<()> {
    if !(1..=6).contains(&level) {
        bail!("HSK translation level must be from 1 through 6");
    }
    Ok(())
}

fn validate_common<'a>(
    utterances: impl IntoIterator<Item = (&'a str, &'a str)>,
    context: &[HskPrecedingUtterance],
    protected_names: &[HskProtectedName],
) -> Result<()> {
    let mut ids = HashSet::new();
    for (id, source) in utterances {
        if id.trim().is_empty() {
            bail!("HSK translation application ID must not be empty");
        }
        if !ids.insert(id) {
            bail!("duplicate HSK translation application ID `{id}`");
        }
        if source.trim().is_empty() {
            bail!("HSK translation item `{id}` has empty English text");
        }
    }

    for utterance in context {
        if utterance.source_english.trim().is_empty() || utterance.chinese.trim().is_empty() {
            bail!("preceding HSK context requires non-empty English and Chinese text");
        }
    }

    let mut names = HashMap::with_capacity(protected_names.len());
    for name in protected_names {
        let source = name.source_english.trim();
        let chinese = name.chinese.trim();
        if source.is_empty() || chinese.is_empty() {
            bail!("HSK protected names require non-empty English and Chinese forms");
        }
        let key = source.to_ascii_lowercase();
        if let Some(previous) = names.insert(key, chinese)
            && previous != chinese
        {
            bail!("HSK protected name `{source}` has conflicting Chinese forms");
        }
    }
    Ok(())
}

struct ExpectedUtterance<'a> {
    id: &'a str,
    source_english: &'a str,
}

enum ParsedLine {
    Candidate { text: String },
    ExcludeNonStory,
    ExcludeSoundEffect,
    Malformed,
}

#[cfg(test)]
fn parse_numbered_output(
    output: &str,
    expected: &[ExpectedUtterance<'_>],
    protected_names: &[HskProtectedName],
) -> HskTranslationBatchResult {
    parse_numbered_output_with_name_handling(
        output,
        expected,
        protected_names,
        HskNameHandling::Chinese,
        true,
    )
}

fn parse_numbered_output_with_name_handling(
    output: &str,
    expected: &[ExpectedUtterance<'_>],
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
    translate_sound_effects: bool,
) -> HskTranslationBatchResult {
    let mut slots = (0..expected.len())
        .map(|_| Vec::<ParsedLine>::new())
        .collect::<Vec<_>>();
    for raw_line in output.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            continue;
        }
        if let Some((position, parsed)) = parse_output_line(line, expected.len()) {
            if is_source_echo(&parsed, expected[position - 1].source_english) {
                continue;
            }
            slots[position - 1].push(parsed);
        }
    }
    if expected.len() == 1 && slots[0].is_empty() {
        return HskTranslationBatchResult {
            items: vec![parse_repair_output_with_name_handling(
                output,
                &expected[0],
                protected_names,
                name_handling,
                translate_sound_effects,
            )],
        };
    }

    let items = expected
        .iter()
        .zip(slots)
        .map(|(expected, lines)| {
            outcome_from_lines(
                expected,
                lines,
                protected_names,
                name_handling,
                translate_sound_effects,
            )
        })
        .collect();
    HskTranslationBatchResult { items }
}

fn parse_streamed_line(
    line: &str,
    expected: &[ExpectedUtterance<'_>],
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
    translate_sound_effects: bool,
) -> Option<HskTranslationOutcome> {
    let (position, parsed) = parse_output_line(line, expected.len())?;
    if is_source_echo(&parsed, expected[position - 1].source_english) {
        return None;
    }
    Some(outcome_from_lines(
        &expected[position - 1],
        vec![parsed],
        protected_names,
        name_handling,
        translate_sound_effects,
    ))
}

fn parse_repair_output_with_name_handling(
    output: &str,
    expected: &ExpectedUtterance<'_>,
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
    translate_sound_effects: bool,
) -> HskTranslationOutcome {
    let mut lines = output
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .filter(|line| !line.trim().is_empty());
    let Some(line) = lines.next() else {
        return outcome_from_lines(
            expected,
            Vec::new(),
            protected_names,
            name_handling,
            translate_sound_effects,
        );
    };
    if lines.next().is_some() || line.contains('\t') {
        return outcome_from_lines(
            expected,
            vec![ParsedLine::Malformed],
            protected_names,
            name_handling,
            translate_sound_effects,
        );
    }
    if compact_field(line) == compact_field(expected.source_english) {
        return HskTranslationOutcome {
            id: expected.id.to_owned(),
            disposition: HskTranslationDisposition::Translate,
            text: None,
            issues: vec![HskTranslationIssue::SourceEcho],
        };
    }
    if line.trim().eq_ignore_ascii_case(NON_STORY_MARKER) {
        return outcome_from_lines(
            expected,
            vec![ParsedLine::Malformed],
            protected_names,
            name_handling,
            translate_sound_effects,
        );
    }
    if line.trim().eq_ignore_ascii_case(SOUND_EFFECT_MARKER) {
        return outcome_from_lines(
            expected,
            vec![ParsedLine::ExcludeSoundEffect],
            protected_names,
            name_handling,
            translate_sound_effects,
        );
    }
    let mut outcome = outcome_from_lines(
        expected,
        vec![ParsedLine::Candidate {
            text: line.to_owned(),
        }],
        protected_names,
        name_handling,
        translate_sound_effects,
    );
    if let Some(text) = outcome.text.as_deref() {
        let markup_issues = outcome
            .issues
            .iter()
            .filter(|issue| {
                matches!(
                    issue,
                    HskTranslationIssue::InvalidNameMarkup | HskTranslationIssue::UnmarkedLatinText
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        outcome.issues = preservation_issues(expected.source_english, text, protected_names, true);
        outcome.issues.extend(markup_issues);
    }
    outcome
}

fn parse_numbered_repair_output(
    output: &str,
    expected: &[ExpectedUtterance<'_>],
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
    translate_sound_effects: bool,
) -> HskTranslationBatchResult {
    let mut result = parse_numbered_output_with_name_handling(
        output,
        expected,
        protected_names,
        name_handling,
        translate_sound_effects,
    );
    for outcome in &mut result.items {
        let Some(text) = outcome.text.as_deref() else {
            continue;
        };
        let Some(expected) = expected.iter().find(|expected| expected.id == outcome.id) else {
            continue;
        };
        let markup_issues = outcome
            .issues
            .iter()
            .filter(|issue| {
                matches!(
                    issue,
                    HskTranslationIssue::InvalidNameMarkup | HskTranslationIssue::UnmarkedLatinText
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        outcome.issues = preservation_issues(expected.source_english, text, protected_names, true);
        outcome.issues.extend(markup_issues);
    }
    result
}

fn parse_repair_batch_output(
    output: &str,
    expected: &[ExpectedUtterance<'_>],
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
    translate_sound_effects: bool,
) -> HskTranslationBatchResult {
    if let [utterance] = expected {
        return HskTranslationBatchResult {
            items: vec![parse_repair_output_with_name_handling(
                output,
                utterance,
                protected_names,
                name_handling,
                translate_sound_effects,
            )],
        };
    }
    parse_numbered_repair_output(
        output,
        expected,
        protected_names,
        name_handling,
        translate_sound_effects,
    )
}

#[cfg(test)]
fn parse_repair_output(
    output: &str,
    expected: &ExpectedUtterance<'_>,
    protected_names: &[HskProtectedName],
) -> HskTranslationOutcome {
    parse_repair_output_with_name_handling(
        output,
        expected,
        protected_names,
        HskNameHandling::Chinese,
        true,
    )
}

fn is_source_echo(line: &ParsedLine, source_english: &str) -> bool {
    matches!(
        line,
        ParsedLine::Candidate { text, .. }
            if compact_field(text) == compact_field(source_english)
    )
}

fn outcome_from_lines(
    expected: &ExpectedUtterance<'_>,
    mut lines: Vec<ParsedLine>,
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
    translate_sound_effects: bool,
) -> HskTranslationOutcome {
    if lines.is_empty() {
        return HskTranslationOutcome {
            id: expected.id.to_owned(),
            disposition: HskTranslationDisposition::Translate,
            text: None,
            issues: vec![HskTranslationIssue::MissingLine],
        };
    }
    if lines.len() > 1 {
        let text = lines.drain(..).find_map(|line| match line {
            ParsedLine::Candidate { text, .. } if !text.trim().is_empty() => Some(text),
            ParsedLine::Candidate { .. }
            | ParsedLine::ExcludeNonStory
            | ParsedLine::ExcludeSoundEffect
            | ParsedLine::Malformed => None,
        });
        return HskTranslationOutcome {
            id: expected.id.to_owned(),
            disposition: HskTranslationDisposition::Translate,
            text,
            issues: vec![HskTranslationIssue::DuplicateLine],
        };
    }

    let mut text = match lines.pop().expect("slot is non-empty") {
        ParsedLine::Candidate { text } => text,
        ParsedLine::ExcludeNonStory => {
            return HskTranslationOutcome {
                id: expected.id.to_owned(),
                disposition: HskTranslationDisposition::ExcludeNonStory,
                text: None,
                issues: Vec::new(),
            };
        }
        ParsedLine::ExcludeSoundEffect => {
            if translate_sound_effects {
                return HskTranslationOutcome {
                    id: expected.id.to_owned(),
                    disposition: HskTranslationDisposition::Translate,
                    text: None,
                    issues: vec![HskTranslationIssue::MalformedLine],
                };
            }
            return HskTranslationOutcome {
                id: expected.id.to_owned(),
                disposition: HskTranslationDisposition::ExcludeSoundEffect,
                text: None,
                issues: Vec::new(),
            };
        }
        ParsedLine::Malformed => {
            return HskTranslationOutcome {
                id: expected.id.to_owned(),
                disposition: HskTranslationDisposition::Translate,
                text: None,
                issues: vec![HskTranslationIssue::MalformedLine],
            };
        }
    };
    text = text.trim().to_owned();
    if text.is_empty() {
        return HskTranslationOutcome {
            id: expected.id.to_owned(),
            disposition: HskTranslationDisposition::Translate,
            text: None,
            issues: vec![HskTranslationIssue::EmptyTranslation],
        };
    }

    if name_handling == HskNameHandling::KeepOriginal {
        let names = protected_names
            .iter()
            .map(|name| DirectHskName {
                source_english: &name.source_english,
                chinese: &name.chinese,
            })
            .collect::<Vec<_>>();
        text = restore_approved_name_placeholders(expected.source_english, &text, &names);
    }
    normalize_multiplier_notation(expected.source_english, &mut text);
    let (mut text, mut markup_issues) = validate_and_strip_name_markup(
        expected.source_english,
        &text,
        name_handling,
        protected_names,
    );
    normalize_full_width_digits(&mut text);
    normalize_question_punctuation(expected.source_english, &mut text);
    let mut issues = preservation_issues(expected.source_english, &text, protected_names, false);
    issues.append(&mut markup_issues);
    HskTranslationOutcome {
        id: expected.id.to_owned(),
        disposition: HskTranslationDisposition::Translate,
        text: Some(text),
        issues,
    }
}

fn parse_output_line(line: &str, expected_count: usize) -> Option<(usize, ParsedLine)> {
    let digit_count = line
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digit_count == 0 {
        return None;
    }
    let digits = &line[..digit_count];
    let position = digits.parse::<usize>().ok()?;
    if position == 0 || position > expected_count || position.to_string() != digits {
        return None;
    }

    let remainder = &line[digit_count..];
    let text =
        if let Some(text) = remainder.strip_prefix('\t') {
            text
            // Qwen occasionally emits the requested compact numbered protocol
            // as a conventional numbered list. Once the bounded position and a
            // recognized separator are present, the mapping is unambiguous.
        } else if remainder.starts_with(' ') {
            remainder.trim_start_matches(' ')
        } else if remainder.chars().next().is_some_and(|separator| {
            matches!(separator, '.' | '．' | ')' | '）' | ':' | '：' | '、')
        }) {
            let separator_bytes = remainder.chars().next().map(char::len_utf8).unwrap_or(0);
            remainder[separator_bytes..].trim_start_matches(' ')
        } else {
            return Some((position, ParsedLine::Malformed));
        };
    if text.contains('\t') {
        return Some((position, ParsedLine::Malformed));
    }
    if text.trim().eq_ignore_ascii_case(NON_STORY_MARKER) {
        return Some((position, ParsedLine::ExcludeNonStory));
    }
    if text.trim().eq_ignore_ascii_case(SOUND_EFFECT_MARKER) {
        return Some((position, ParsedLine::ExcludeSoundEffect));
    }
    Some((
        position,
        ParsedLine::Candidate {
            text: text.to_owned(),
        },
    ))
}

fn validate_and_strip_name_markup(
    source_english: &str,
    translation: &str,
    name_handling: HskNameHandling,
    protected_names: &[HskProtectedName],
) -> (String, Vec<HskTranslationIssue>) {
    const OPEN: char = '⟦';
    const CLOSE: char = '⟧';

    let mut output = String::with_capacity(translation.len());
    let mut current_name = None::<String>;
    let mut invalid_markup = name_handling == HskNameHandling::Chinese
        && translation
            .chars()
            .any(|character| matches!(character, OPEN | CLOSE));
    for character in translation.chars() {
        match character {
            OPEN => {
                if current_name.is_some() {
                    invalid_markup = true;
                } else {
                    current_name = Some(String::new());
                }
            }
            CLOSE => {
                let Some(name) = current_name.take() else {
                    invalid_markup = true;
                    continue;
                };
                if name_handling != HskNameHandling::KeepOriginal
                    || name.is_empty()
                    || !source_contains_exact_span(source_english, &name)
                    || !protected_names.iter().any(|approved| {
                        approved.source_english == name
                            && source_contains_exact_span(source_english, &approved.source_english)
                    })
                {
                    invalid_markup = true;
                }
                output.push_str(&name);
            }
            _ => {
                if let Some(name) = current_name.as_mut() {
                    name.push(character);
                } else {
                    output.push(character);
                }
            }
        }
    }
    if let Some(name) = current_name {
        invalid_markup = true;
        output.push_str(&name);
    }

    let mut issues = Vec::new();
    if invalid_markup {
        issues.push(HskTranslationIssue::InvalidNameMarkup);
    }
    let allowed_names = protected_names
        .iter()
        .filter(|name| source_contains_exact_span(source_english, &name.source_english))
        .cloned()
        .collect::<Vec<_>>();
    if output
        .chars()
        .any(|character| character.is_ascii_alphabetic())
        && (name_handling == HskNameHandling::Chinese
            || !all_latin_is_protected(&output, &allowed_names))
    {
        issues.push(HskTranslationIssue::UnmarkedLatinText);
    }
    (output, issues)
}

fn all_latin_is_protected(translation: &str, protected_names: &[HskProtectedName]) -> bool {
    let mut covered = vec![false; translation.len()];
    for allowed in protected_names
        .iter()
        .map(|name| name.chinese.as_str())
        .filter(|name| !name.is_empty())
    {
        for (start, matched) in translation.match_indices(allowed) {
            for byte in covered.iter_mut().take(start + matched.len()).skip(start) {
                *byte = true;
            }
        }
    }
    translation.char_indices().all(|(index, character)| {
        !character.is_ascii_alphabetic() || covered.get(index).copied().unwrap_or(false)
    })
}

fn source_contains_exact_span(source: &str, name: &str) -> bool {
    source.match_indices(name).any(|(start, matched)| {
        let end = start + matched.len();
        let starts_at_boundary =
            start == 0 || !source.as_bytes()[start - 1].is_ascii_alphanumeric();
        let ends_at_boundary =
            end == source.len() || !source.as_bytes()[end].is_ascii_alphanumeric();
        starts_at_boundary && ends_at_boundary
    })
}

fn canonical_source_span<'a>(source: &'a str, candidate: &str) -> Option<&'a str> {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return None;
    }
    source
        .char_indices()
        .filter(|(start, _)| {
            *start == 0 || !source.as_bytes()[start.saturating_sub(1)].is_ascii_alphanumeric()
        })
        .find_map(|(start, _)| {
            let end = start.checked_add(candidate.len())?;
            let actual = source.get(start..end)?;
            let ends_at_boundary =
                end == source.len() || !source.as_bytes()[end].is_ascii_alphanumeric();
            (ends_at_boundary && actual.eq_ignore_ascii_case(candidate)).then_some(actual)
        })
}

fn preservation_issues(
    source_english: &str,
    chinese: &str,
    protected_names: &[HskProtectedName],
    accept_chinese_numerals: bool,
) -> Vec<HskTranslationIssue> {
    let mut issues = Vec::new();
    let expected_numbers = ascii_numbers(source_english);
    let actual_numbers = if accept_chinese_numerals {
        normalized_numbers_for_source(source_english, chinese)
    } else {
        ascii_numbers(chinese)
    };
    if actual_numbers != expected_numbers {
        issues.push(HskTranslationIssue::NumberMismatch {
            expected: expected_numbers,
            actual: actual_numbers,
        });
    }

    let source_lower = source_english.to_ascii_lowercase();
    for name in protected_names {
        if source_lower.contains(&name.source_english.to_ascii_lowercase())
            && !chinese.contains(&name.chinese)
        {
            issues.push(HskTranslationIssue::ProtectedNameMissing {
                source_english: name.source_english.clone(),
                chinese: name.chinese.clone(),
            });
        }
    }

    if has_question_intent(&source_lower) && !has_chinese_question_intent(chinese) {
        issues.push(HskTranslationIssue::QuestionIntentMissing);
    }
    let source_words = english_word_count(source_english);
    let chinese_characters = chinese_character_count(chinese);
    let maximum_chinese_characters = source_words.saturating_mul(4).saturating_add(4).max(12);
    if source_words <= 8 && chinese_characters > maximum_chinese_characters {
        issues.push(HskTranslationIssue::ExcessiveExpansion {
            source_words,
            chinese_characters,
        });
    }
    issues
}

fn english_word_count(text: &str) -> usize {
    text.split(|character: char| !character.is_ascii_alphabetic())
        .filter(|word| !word.is_empty())
        .count()
}

fn chinese_character_count(text: &str) -> usize {
    text.chars()
        .filter(|character| {
            matches!(
                *character,
                '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}'
            )
        })
        .count()
}

fn normalize_full_width_digits(text: &mut String) {
    *text = text
        .chars()
        .map(|character| match character {
            '０'..='９' => {
                char::from_u32(u32::from(character) - u32::from('０') + u32::from('0'))
                    .expect("full-width digit has an ASCII form")
            }
            _ => character,
        })
        .collect();
}

fn normalize_multiplier_notation(source_english: &str, text: &mut String) {
    let source = source_english.as_bytes();
    let source_has_multiplier = source.iter().enumerate().any(|(index, byte)| {
        matches!(byte, b'x' | b'X')
            && (index
                .checked_sub(1)
                .is_some_and(|previous| source[previous].is_ascii_digit())
                || source
                    .get(index + 1)
                    .is_some_and(|next| next.is_ascii_digit()))
    });
    if !source_has_multiplier {
        return;
    }
    let characters = text.chars().collect::<Vec<_>>();
    *text = characters
        .iter()
        .enumerate()
        .map(|(index, character)| {
            if matches!(character, 'x' | 'X')
                && (index
                    .checked_sub(1)
                    .is_some_and(|previous| characters[previous].is_ascii_digit())
                    || characters
                        .get(index + 1)
                        .is_some_and(|next| next.is_ascii_digit()))
            {
                '×'
            } else {
                *character
            }
        })
        .collect();
}

fn normalized_numbers_for_source(source_english: &str, text: &str) -> Vec<String> {
    let expected = ascii_numbers(source_english);
    let actual_ascii = ascii_numbers(text);
    if actual_ascii == expected || !actual_ascii.is_empty() {
        return actual_ascii;
    }
    expected
        .into_iter()
        .filter(|number| {
            chinese_number_variants(number)
                .iter()
                .any(|chinese| text.contains(chinese))
        })
        .collect()
}

fn chinese_number_variants(ascii: &str) -> Vec<String> {
    let digit_sequence = ascii
        .chars()
        .filter_map(|digit| match digit {
            '0' => Some('零'),
            '1' => Some('一'),
            '2' => Some('二'),
            '3' => Some('三'),
            '4' => Some('四'),
            '5' => Some('五'),
            '6' => Some('六'),
            '7' => Some('七'),
            '8' => Some('八'),
            '9' => Some('九'),
            _ => None,
        })
        .collect::<String>();
    let mut variants = vec![digit_sequence];
    if let Ok(value) = ascii.parse::<u16>()
        && value <= 9_999
    {
        let standard = chinese_integer_below_10_000(value);
        if !variants.contains(&standard) {
            variants.push(standard.clone());
        }
        if standard.starts_with("二百") || standard.starts_with("二千") {
            variants.push(format!("两{}", &standard['二'.len_utf8()..]));
        }
    }
    variants.sort_by_key(|variant| std::cmp::Reverse(variant.len()));
    variants
}

fn chinese_integer_below_10_000(value: u16) -> String {
    if value == 0 {
        return "零".to_owned();
    }
    let digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    let units = ["千", "百", "十", ""];
    let divisors = [1_000_u16, 100, 10, 1];
    let mut rendered = String::new();
    let mut zero_pending = false;
    for (index, divisor) in divisors.into_iter().enumerate() {
        let digit = usize::from(value / divisor % 10);
        if digit == 0 {
            zero_pending |= !rendered.is_empty() && value % divisor != 0;
            continue;
        }
        if zero_pending {
            rendered.push('零');
            zero_pending = false;
        }
        if !(digit == 1 && divisor == 10 && rendered.is_empty()) {
            rendered.push(digits[digit]);
        }
        rendered.push_str(units[index]);
    }
    rendered
}

fn ascii_numbers(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut numbers = Vec::new();
    let mut start = None;
    for (index, byte) in bytes.iter().copied().enumerate() {
        if byte.is_ascii_digit() {
            start.get_or_insert(index);
        } else if let Some(number_start) = start.take() {
            if ascii_number_is_semantic(bytes, number_start, index) {
                numbers.push(text[number_start..index].to_owned());
            }
        }
    }
    if let Some(number_start) = start {
        if ascii_number_is_semantic(bytes, number_start, bytes.len()) {
            numbers.push(text[number_start..].to_owned());
        }
    }
    numbers
}

fn ascii_number_is_semantic(bytes: &[u8], start: usize, end: usize) -> bool {
    let left_alpha = start
        .checked_sub(1)
        .is_some_and(|index| bytes[index].is_ascii_alphabetic());
    let right_alpha = bytes
        .get(end)
        .is_some_and(|byte| byte.is_ascii_alphabetic());
    if !left_alpha && !right_alpha {
        return true;
    }

    let left_multiplier = start.checked_sub(1).is_some_and(|marker| {
        matches!(bytes[marker], b'x' | b'X')
            && marker
                .checked_sub(1)
                .is_none_or(|before| !bytes[before].is_ascii_alphanumeric())
    });
    let right_multiplier = bytes.get(end).is_some_and(|marker| {
        matches!(marker, b'x' | b'X')
            && bytes
                .get(end + 1)
                .is_none_or(|after| !after.is_ascii_alphanumeric())
    });
    left_multiplier || right_multiplier
}

fn has_question_intent(source_lower: &str) -> bool {
    if source_lower.contains('?') {
        return true;
    }
    let trimmed = source_lower.trim_end();
    if trimmed.ends_with("...")
        || trimmed.ends_with(',')
        || trimmed.ends_with(';')
        || trimmed.ends_with(':')
        || trimmed.ends_with('-')
        || trimmed.ends_with('—')
        || trimmed.ends_with('…')
    {
        // Webtoon dialogue is often split across adjacent balloons. An
        // inverted auxiliary at the start of a comma-terminated fragment does
        // not require this fragment to carry the sentence's final question
        // mark; the mark may belong to the continuation.
        return false;
    }
    let mut words = source_lower
        .split(|character: char| !character.is_ascii_alphabetic())
        .filter(|word| !word.is_empty());
    let first_word = words.next();
    let second_word = words.next();
    if first_word == Some("do") && second_word == Some("not") {
        return false;
    }
    first_word.is_some_and(|word| {
        matches!(
            word,
            "am" | "are"
                | "can"
                | "could"
                | "did"
                | "do"
                | "does"
                | "had"
                | "has"
                | "have"
                | "how"
                | "is"
                | "may"
                | "might"
                | "must"
                | "shall"
                | "should"
                | "was"
                | "were"
                | "what"
                | "when"
                | "where"
                | "which"
                | "who"
                | "whom"
                | "whose"
                | "why"
                | "will"
                | "would"
        )
    })
}

fn has_chinese_question_intent(text: &str) -> bool {
    text.contains('?') || text.contains('？')
}

fn normalize_question_punctuation(source_english: &str, chinese: &mut String) {
    if !has_question_intent(&source_english.to_ascii_lowercase())
        || has_chinese_question_intent(chinese)
    {
        return;
    }

    let trimmed_len = chinese.trim_end().len();
    chinese.truncate(trimmed_len);
    while chinese.ends_with(['.', '!', '。', '！']) {
        chinese.pop();
    }
    chinese.push('？');
}

fn check_cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use anyhow::Context;

    use super::*;

    struct FakeGenerator {
        outputs: Mutex<VecDeque<String>>,
        system_prompts: Mutex<Vec<String>>,
        user_prompts: Mutex<Vec<String>>,
        options: Mutex<Vec<GenerateOptions>>,
        target_languages: Mutex<Vec<Language>>,
        calls: AtomicUsize,
    }

    struct MaxTwoUtteranceGenerator {
        inner: FakeGenerator,
    }

    struct MaxTwoRepairGenerator {
        inner: FakeGenerator,
    }

    impl MaxTwoUtteranceGenerator {
        fn new(outputs: impl IntoIterator<Item = &'static str>) -> Self {
            Self {
                inner: FakeGenerator::new(outputs),
            }
        }
    }

    impl Generator for MaxTwoUtteranceGenerator {
        async fn token_count(&self, text: &str) -> Result<usize> {
            self.inner.token_count(text).await
        }

        async fn constrained_completion_capacity(
            &self,
            _system_prompt: &str,
            user_prompt: &str,
            _target_language: Language,
        ) -> Result<usize> {
            let numbered_lines = user_prompt
                .lines()
                .filter(|line| {
                    line.split_once('\t')
                        .is_some_and(|(position, _)| position.parse::<usize>().is_ok())
                })
                .count();
            Ok(
                if numbered_lines <= 2 && !user_prompt.contains("Previous translations") {
                    usize::MAX
                } else {
                    0
                },
            )
        }

        async fn generate_streaming(
            &self,
            system_prompt: &str,
            user_prompt: &str,
            options: &GenerateOptions,
            target_language: Language,
            cancel: &AtomicBool,
            on_piece: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<String> {
            self.inner
                .generate_streaming(
                    system_prompt,
                    user_prompt,
                    options,
                    target_language,
                    cancel,
                    on_piece,
                )
                .await
        }
    }

    impl Generator for MaxTwoRepairGenerator {
        async fn token_count(&self, text: &str) -> Result<usize> {
            self.inner.token_count(text).await
        }

        async fn constrained_completion_capacity(
            &self,
            _system_prompt: &str,
            user_prompt: &str,
            _target_language: Language,
        ) -> Result<usize> {
            let items = user_prompt
                .lines()
                .filter(|line| {
                    line.strip_prefix("Item ")
                        .and_then(|suffix| suffix.strip_suffix(':'))
                        .is_some_and(|position| position.parse::<usize>().is_ok())
                })
                .count();
            Ok(if items <= 2 { usize::MAX } else { 0 })
        }

        async fn generate_streaming(
            &self,
            system_prompt: &str,
            user_prompt: &str,
            options: &GenerateOptions,
            target_language: Language,
            cancel: &AtomicBool,
            on_piece: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<String> {
            self.inner
                .generate_streaming(
                    system_prompt,
                    user_prompt,
                    options,
                    target_language,
                    cancel,
                    on_piece,
                )
                .await
        }
    }

    impl FakeGenerator {
        fn new(outputs: impl IntoIterator<Item = &'static str>) -> Self {
            Self {
                outputs: Mutex::new(outputs.into_iter().map(str::to_owned).collect()),
                system_prompts: Mutex::new(Vec::new()),
                user_prompts: Mutex::new(Vec::new()),
                options: Mutex::new(Vec::new()),
                target_languages: Mutex::new(Vec::new()),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Generator for FakeGenerator {
        async fn token_count(&self, text: &str) -> Result<usize> {
            Ok(text.chars().count())
        }

        async fn constrained_completion_capacity(
            &self,
            _system_prompt: &str,
            _user_prompt: &str,
            _target_language: Language,
        ) -> Result<usize> {
            Ok(usize::MAX)
        }

        async fn generate_streaming(
            &self,
            system_prompt: &str,
            user_prompt: &str,
            options: &GenerateOptions,
            target_language: Language,
            _cancel: &AtomicBool,
            on_piece: &mut dyn FnMut(&str) -> Result<()>,
        ) -> Result<String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.system_prompts
                .lock()
                .unwrap()
                .push(system_prompt.to_owned());
            self.user_prompts
                .lock()
                .unwrap()
                .push(user_prompt.to_owned());
            self.options.lock().unwrap().push(options.clone());
            self.target_languages.lock().unwrap().push(target_language);
            let output = self
                .outputs
                .lock()
                .unwrap()
                .pop_front()
                .context("fake output exhausted")?;
            for piece in output.split_inclusive('\n') {
                on_piece(piece)?;
            }
            Ok(output)
        }
    }

    #[tokio::test]
    async fn translation_cannot_self_declare_an_unapproved_name() -> Result<()> {
        let generator = FakeGenerator::new(["1\t我昨天见到了⟦Tarin Voss⟧。"]);
        let input = HskTranslationBatchRequest {
            requested_level: 3,
            learning_mode: HskLearningMode::Strict,
            name_handling: HskNameHandling::KeepOriginal,
            translate_sound_effects: false,
            utterances: vec![source("dialogue", "I saw Tarin Voss yesterday.")],
            preceding_utterances: Vec::new(),
            protected_names: Vec::new(),
        };

        let result = translate_with(&generator, &input, &AtomicBool::new(false)).await?;

        assert_eq!(
            result.items[0].text.as_deref(),
            Some("我昨天见到了Tarin Voss。")
        );
        assert_eq!(
            result.items[0].issues,
            [
                HskTranslationIssue::InvalidNameMarkup,
                HskTranslationIssue::UnmarkedLatinText,
            ]
        );
        assert_eq!(generator.calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            *generator.target_languages.lock().unwrap(),
            [Language::ChineseSimplified]
        );
        Ok(())
    }

    #[tokio::test]
    async fn ordinary_descriptions_need_no_code_vocabulary_to_translate() -> Result<()> {
        let generator = FakeGenerator::new(["1\t那位年长的管理员来了。"]);
        let input = HskTranslationBatchRequest {
            requested_level: 3,
            learning_mode: HskLearningMode::Strict,
            name_handling: HskNameHandling::KeepOriginal,
            translate_sound_effects: false,
            utterances: vec![source("dialogue", "THE SENIOR ADMINISTRATOR ARRIVED.")],
            preceding_utterances: Vec::new(),
            protected_names: Vec::new(),
        };

        let result = translate_with(&generator, &input, &AtomicBool::new(false)).await?;

        assert!(result.items[0].is_valid());
        Ok(())
    }

    #[tokio::test]
    async fn chinese_name_mode_rejects_every_remaining_latin_word() -> Result<()> {
        let generator = FakeGenerator::new(["1\tThe wife来了。"]);
        let mut input = request();
        input.name_handling = HskNameHandling::Chinese;
        input.utterances = vec![source("wife", "The wife arrived.")];
        input.protected_names.clear();

        let result = translate_with(&generator, &input, &AtomicBool::new(false)).await?;

        assert_eq!(
            result.items[0].issues,
            [HskTranslationIssue::UnmarkedLatinText]
        );
        Ok(())
    }

    #[tokio::test]
    async fn targeted_repairs_share_one_numbered_generation_without_cross_item_name_leaks()
    -> Result<()> {
        let generator = FakeGenerator::new(["1\t学生\n2\tNeris"]);
        let request = HskTranslationRepairBatchRequest {
            requested_level: 3,
            learning_mode: HskLearningMode::Natural,
            name_handling: HskNameHandling::KeepOriginal,
            translate_sound_effects: true,
            utterances: vec![
                HskRepairUtterance {
                    id: "student".to_owned(),
                    kind: HskUtteranceKind::Dialogue,
                    source_english: "The graduate student arrived.".to_owned(),
                    rejected_chinese: Some("研究生来了。".to_owned()),
                    avoid_chinese: vec!["研究生".to_owned()],
                    problems: vec!["use an easier word".to_owned()],
                },
                HskRepairUtterance {
                    id: "wife".to_owned(),
                    kind: HskUtteranceKind::Dialogue,
                    source_english: "My wife arrived.".to_owned(),
                    rejected_chinese: Some("My wife来了。".to_owned()),
                    avoid_chinese: Vec::new(),
                    problems: vec!["translate every ordinary Latin word".to_owned()],
                },
            ],
            preceding_utterances: Vec::new(),
            protected_names: vec![HskProtectedName {
                source_english: "Neris".to_owned(),
                chinese: "Neris".to_owned(),
            }],
        };

        let result = repair_batch_with(&generator, &request, &AtomicBool::new(false)).await?;

        assert_eq!(generator.calls.load(Ordering::Relaxed), 1);
        assert_eq!(result.items[0].text.as_deref(), Some("学生"));
        assert!(result.items[0].issues.is_empty());
        assert_eq!(
            result.items[1].issues,
            [HskTranslationIssue::UnmarkedLatinText]
        );
        Ok(())
    }

    #[tokio::test]
    async fn repair_batches_are_partitioned_before_generation_to_fit_model_context() -> Result<()> {
        let generator = MaxTwoRepairGenerator {
            inner: FakeGenerator::new([
                "1\t第一项\n2\t第二项",
                "1\t第三项\n2\t第四项",
                "1\t第五项",
            ]),
        };
        let request = HskTranslationRepairBatchRequest {
            requested_level: 3,
            learning_mode: HskLearningMode::Natural,
            name_handling: HskNameHandling::Chinese,
            translate_sound_effects: true,
            utterances: (1..=5)
                .map(|position| HskRepairUtterance {
                    id: format!("item-{position}"),
                    kind: HskUtteranceKind::Dialogue,
                    source_english: format!("Rejected English item {position}."),
                    rejected_chinese: Some(format!("rejected-{position}")),
                    avoid_chinese: Vec::new(),
                    problems: vec!["translate all ordinary words".to_owned()],
                })
                .collect(),
            preceding_utterances: Vec::new(),
            protected_names: Vec::new(),
        };

        let result = repair_batch_with(&generator, &request, &AtomicBool::new(false)).await?;

        assert_eq!(generator.inner.calls.load(Ordering::Relaxed), 3);
        assert_eq!(
            result
                .items
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            ["item-1", "item-2", "item-3", "item-4", "item-5"]
        );
        assert!(
            generator
                .inner
                .user_prompts
                .lock()
                .unwrap()
                .iter()
                .all(|prompt| prompt.matches("\nItem ").count() <= 2)
        );
        Ok(())
    }

    #[test]
    fn semantic_classification_discovers_unseen_names_without_preserving_roles() -> Result<()> {
        let utterances = vec![
            source("a", "The guild quartermaster Sable opened the gate."),
            source("b", "Our regent asked cartographer Ilyan to return."),
        ];

        let names = parse_detected_names("1\tPERSON=Sable\n2\tIlyan=PERSON", &utterances)?;

        assert_eq!(
            names
                .iter()
                .map(|candidate| (
                    candidate.name.source_english.as_str(),
                    candidate.entity_type.as_str(),
                ))
                .collect::<Vec<_>>(),
            [("Sable", "PERSON"), ("Ilyan", "PERSON")]
        );
        Ok(())
    }

    #[test]
    fn semantic_classification_discards_hallucinated_names_without_losing_valid_siblings()
    -> Result<()> {
        let utterances = vec![
            source("a", "The courier Neris arrived."),
            source("b", "The headmaster opened the gate."),
        ];

        let names = parse_detected_names(
            "1\tPERSON=Invented | PERSON=neris | PERSON=The courier Neris arrived.\n2\tPERSON=The",
            &utterances,
        )?;

        assert_eq!(
            names
                .iter()
                .map(|candidate| candidate.name.source_english.as_str())
                .collect::<Vec<_>>(),
            ["Neris"]
        );
        Ok(())
    }

    #[test]
    fn omitted_name_lines_do_not_discard_valid_sibling_candidates() -> Result<()> {
        let utterances = vec![
            source("a", "Sable opened the gate."),
            source("b", "Ilyan followed her."),
        ];

        let names = parse_detected_names("1\tPERSON=Sable", &utterances)?;

        assert_eq!(
            names
                .iter()
                .map(|candidate| candidate.name.source_english.as_str())
                .collect::<Vec<_>>(),
            ["Sable"]
        );
        Ok(())
    }

    #[test]
    fn untyped_exact_name_candidates_still_require_contextual_adjudication() -> Result<()> {
        let utterances = vec![
            source("a", "MAYSA opened the gate."),
            source("b", "THIS WHOLE SENTENCE IS NOT A NAME."),
        ];
        let names = parse_detected_names(
            "1\tMAYSA\n2\tTHIS WHOLE SENTENCE IS NOT A NAME.",
            &utterances,
        )?;

        assert_eq!(names.len(), 1);
        assert_eq!(names[0].name.source_english, "MAYSA");
        assert_eq!(names[0].entity_type, "UNKNOWN");
        Ok(())
    }

    #[test]
    fn typed_name_candidates_accept_field_separators_and_reversed_pairs() -> Result<()> {
        let utterances = vec![source("a", "JAMEL asked JOHAN.")];
        let names = parse_detected_names("1\tJAMEL=PERSON\tJOHAN=PERSON", &utterances)?;

        assert_eq!(
            names
                .iter()
                .map(|candidate| (
                    candidate.name.source_english.as_str(),
                    candidate.entity_type.as_str()
                ))
                .collect::<Vec<_>>(),
            [("JAMEL", "PERSON"), ("JOHAN", "PERSON")]
        );
        Ok(())
    }

    #[test]
    fn transparent_unique_epithets_are_translated_not_preserved() {
        let utterances = vec![
            source("a", "THE SILVER CAPTAIN entered."),
            source("b", "We followed THE SILVER CAPTAIN."),
        ];

        let names = parse_detected_names("1\t-\n2\t-", &utterances).unwrap();

        assert!(names.is_empty());
    }

    #[test]
    fn typed_name_candidates_apply_only_safe_local_syntax_gates() -> Result<()> {
        let utterances = vec![
            source("a", "YOUR INSTINCTS"),
            source("b", "AREN'T TELLING YOU TO RUN AWAY?"),
            source("c", "It would be TALARIS of the Ice Palace."),
        ];

        let names = parse_detected_names(
            "1\tPERSON=YOUR | PLACE=INSTINCTS\n2\tPERSON=AREN'T | PLACE=TELLING | ORGANIZATION=YOU\n3\tPERSON=TALARIS",
            &utterances,
        )?;

        assert_eq!(
            names
                .iter()
                .map(|candidate| candidate.name.source_english.as_str())
                .collect::<Vec<_>>(),
            ["TELLING", "TALARIS"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn contextual_adjudication_rejects_ordinary_predicates_without_losing_names() -> Result<()>
    {
        let generator = FakeGenerator::new([
            "1\tPLACE=TELLING\n2\tPERSON=TALARIS",
            "1\t12\tTAB\tDESCRIPTIVE\n2\t12\tTAB\tOPAQUE",
        ]);
        let utterances = vec![
            source("a", "AREN'T TELLING YOU TO RUN AWAY?"),
            source("b", "It would be TALARIS of the Ice Palace."),
        ];

        let names =
            detect_proper_names_with(&generator, &utterances, &AtomicBool::new(false)).await?;

        assert_eq!(
            names
                .iter()
                .map(|name| name.source_english.as_str())
                .collect::<Vec<_>>(),
            ["TALARIS"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn typed_name_analysis_uses_generic_discovery_and_batched_adjudication() -> Result<()> {
        let generator = FakeGenerator::new([
            "1\tPERSON=Sable\n2\tPLACE=Ilyan Keep | PERSON=Invented",
            "1\tOPAQUE\n2\tOPAQUE",
        ]);
        let utterances = vec![
            source("a", "The quartermaster Sable opened the gate."),
            source("b", "Ilyan Keep welcomed the headmaster."),
        ];

        let names =
            detect_proper_names_with(&generator, &utterances, &AtomicBool::new(false)).await?;

        assert_eq!(
            names
                .iter()
                .map(|name| name.source_english.as_str())
                .collect::<Vec<_>>(),
            ["Sable", "Ilyan Keep"]
        );
        let prompt = generator.system_prompts.lock().unwrap()[0].to_ascii_lowercase();
        assert!(prompt.contains("shortest exact source span"));
        assert!(prompt.contains("do not output a complete sentence or clause"));
        assert!(prompt.contains("relationships, honorifics, occupations, roles, ranks, titles"));
        assert!(!prompt.contains("wife"));
        assert!(!prompt.contains("academy headmaster"));
        let verifier = generator.system_prompts.lock().unwrap()[1].to_ascii_lowercase();
        assert!(verifier.contains("capitalization and uniqueness alone never make a name"));
        assert!(verifier.contains("opaque or descriptive"));
        assert_eq!(generator.calls.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[test]
    fn semantic_classification_parser_keeps_only_exact_source_name_spans() -> Result<()> {
        let utterances = vec![source("a", "JAMEL, WHAT DO YOU THINK HAPPENED TO JOHAN?")];
        let names = parse_detected_names(
            "1\tPERSON=JAMEL | PERSON=JOHAN | PERSON=INVENTED",
            &utterances,
        )?;
        assert_eq!(
            names
                .iter()
                .map(|name| name.name.source_english.as_str())
                .collect::<Vec<_>>(),
            ["JAMEL", "JOHAN"]
        );
        Ok(())
    }

    #[test]
    fn semantic_parsers_accept_standard_numbered_list_separators() -> Result<()> {
        let utterances = vec![
            source("a", "JAMEL opened the gate."),
            source("b", "The headmaster followed him."),
        ];
        let names = parse_detected_names("1. PERSON=JAMEL\n2) -", &utterances)?;
        let classified =
            parse_semantic_classification("PAGE\tFURNITURE\n1. STORY\n2) FURNITURE", &utterances);
        assert!(classified.page_is_furniture);
        assert_eq!(
            classified.regions,
            [("b".to_owned(), HskTranslationDisposition::ExcludeNonStory)]
        );
        assert_eq!(
            names
                .iter()
                .map(|name| name.name.source_english.as_str())
                .collect::<Vec<_>>(),
            ["JAMEL"]
        );
        Ok(())
    }

    #[tokio::test]
    async fn semantic_region_classification_is_model_driven_and_fail_soft() -> Result<()> {
        let generator = FakeGenerator::new([
            "PAGE\tSTORY\n1\tSFX\n2\tSTORY\n3\tFURNITURE\n4\tARTWORK\n99\tSFX",
            "1\tPRESERVE",
        ]);
        let mut utterances = vec![
            source("a", "THUD"),
            source("b", "Neris heard a thud outside."),
            source("c", "EXAMPLESCANS.COM"),
            source("d", "Arcane Spear: Fifth Form"),
        ];
        utterances[0].semantic_layout = Some(HskSemanticLayout {
            detector_enclosed: true,
            measured_font_height_millionths: 32_000,
            appearance_band_count: 2,
            distinct_text_color_count: 2,
            has_outline: true,
            x0_millionths: 100_000,
            y0_millionths: 200_000,
            x1_millionths: 400_000,
            y1_millionths: 300_000,
            page_width: 1200,
            page_height: 800,
        });
        utterances[3].semantic_layout = Some(HskSemanticLayout {
            detector_enclosed: true,
            measured_font_height_millionths: 180_000,
            appearance_band_count: 2,
            distinct_text_color_count: 2,
            has_outline: true,
            x0_millionths: 100_000,
            y0_millionths: 200_000,
            x1_millionths: 400_000,
            y1_millionths: 300_000,
            page_width: 1200,
            page_height: 800,
        });

        let classified =
            classify_semantic_regions_with(&generator, &utterances, &AtomicBool::new(false))
                .await?;

        assert!(!classified.page_is_furniture);
        assert_eq!(
            classified.regions,
            [
                (
                    "a".to_owned(),
                    HskTranslationDisposition::ExcludeSoundEffect
                ),
                ("c".to_owned(), HskTranslationDisposition::ExcludeNonStory),
                ("d".to_owned(), HskTranslationDisposition::PreserveArtwork),
            ]
        );
        let prompt = generator.system_prompts.lock().unwrap()[0].to_ascii_lowercase();
        assert!(prompt.contains("semantic function"));
        assert!(prompt.contains("ordinary language"));
        assert!(prompt.contains("publisher/site credit"));
        assert!(prompt.contains("work/series title or logo"));
        assert!(prompt.contains("irrecoverable ocr letter soup"));
        assert!(prompt.contains("artwork"));
        assert!(prompt.contains("finite clause"));
        assert!(prompt.contains("when uncertain between artwork and story"));
        assert!(prompt.contains("entire section"));
        let artwork_prompt = generator.system_prompts.lock().unwrap()[1].to_ascii_lowercase();
        assert!(artwork_prompt.contains("item names or descriptions"));
        assert!(artwork_prompt.contains("missions"));
        assert!(artwork_prompt.contains("complete ordered section context"));
        let user_prompt = generator.user_prompts.lock().unwrap()[0].to_ascii_lowercase();
        assert!(user_prompt.contains("detector=enclosed"));
        assert!(user_prompt.contains("box=10.0%,20.0%-40.0%,30.0%"));
        assert!(user_prompt.contains("page=1200x800"));
        let artwork_user_prompt = generator.user_prompts.lock().unwrap()[1].to_ascii_lowercase();
        assert!(artwork_user_prompt.contains("complete ordered section context"));
        assert!(artwork_user_prompt.contains("page-position=4"));
        Ok(())
    }

    #[tokio::test]
    async fn semantic_role_classification_keeps_regions_beyond_translation_boundary_together()
    -> Result<()> {
        let generator = FakeGenerator::new([concat!(
            "PAGE\tFURNITURE\n",
            "1\tARTWORK\n",
            "2\tFURNITURE\n",
            "3\tFURNITURE\n",
            "4\tFURNITURE\n",
            "5\tFURNITURE\n",
            "6\tFURNITURE\n",
            "7\tFURNITURE\n",
            "8\tFURNITURE"
        )]);
        let utterances = (0..8)
            .map(|index| source(&format!("credit-{index}"), &format!("Staff credit {index}")))
            .collect::<Vec<_>>();

        let classified =
            classify_semantic_regions_with(&generator, &utterances, &AtomicBool::new(false))
                .await?;

        assert!(classified.page_is_furniture);
        assert_eq!(generator.calls.load(Ordering::Relaxed), 1);
        let prompt = &generator.user_prompts.lock().unwrap()[0];
        assert!(prompt.contains("8\ttext=Staff credit 7"));
        Ok(())
    }

    #[test]
    fn semantic_artwork_claims_fail_safe_for_story_syntax() {
        let mut utterances = vec![
            source("a", "YOUR INSTINCTS"),
            source("b", "AREN'T TELLING YOU TO RUN AWAY?"),
            source("c", "MYUNGWANG SWORD AUTHORITY"),
        ];
        utterances[2].semantic_layout = Some(HskSemanticLayout {
            detector_enclosed: true,
            measured_font_height_millionths: 180_000,
            appearance_band_count: 1,
            distinct_text_color_count: 1,
            has_outline: false,
            x0_millionths: 100_000,
            y0_millionths: 200_000,
            x1_millionths: 400_000,
            y1_millionths: 300_000,
            page_width: 900,
            page_height: 16_000,
        });

        let classified = parse_semantic_classification(
            "PAGE\tSTORY\n1\tARTWORK\n2\tARTWORK\n3\tARTWORK",
            &utterances,
        );

        assert_eq!(
            classified.regions,
            [("c".to_owned(), HskTranslationDisposition::PreserveArtwork)]
        );
    }

    #[test]
    fn complete_non_story_artwork_sections_are_page_furniture_but_mixed_sections_are_not() {
        let utterances = vec![
            source("a", "MY FANTASY STORY"),
            source("b", "EXAMPLE PUBLISHING"),
        ];
        let cover =
            parse_semantic_classification("PAGE\tARTWORK\n1\tARTWORK\n2\tFURNITURE", &utterances);
        assert!(cover.page_is_furniture);

        let mixed =
            parse_semantic_classification("PAGE\tARTWORK\n1\tARTWORK\n2\tSTORY", &utterances);
        assert!(!mixed.page_is_furniture);

        let truncated = parse_semantic_classification("PAGE\tARTWORK\n1\tARTWORK", &utterances);
        assert!(!truncated.page_is_furniture);
    }

    #[test]
    fn artwork_requires_independent_visual_lettering_evidence() {
        let mut utterances = vec![
            source("a", "THIRTY OF THEM!!! X3"),
            source("b", "THIRD SWORD"),
        ];
        utterances[0].semantic_layout = Some(HskSemanticLayout {
            detector_enclosed: true,
            measured_font_height_millionths: 80_000,
            appearance_band_count: 1,
            distinct_text_color_count: 1,
            has_outline: false,
            x0_millionths: 100_000,
            y0_millionths: 100_000,
            x1_millionths: 500_000,
            y1_millionths: 200_000,
            page_width: 900,
            page_height: 16_000,
        });
        utterances[1].semantic_layout = Some(HskSemanticLayout {
            detector_enclosed: true,
            measured_font_height_millionths: 180_000,
            appearance_band_count: 1,
            distinct_text_color_count: 1,
            has_outline: false,
            x0_millionths: 100_000,
            y0_millionths: 300_000,
            x1_millionths: 500_000,
            y1_millionths: 500_000,
            page_width: 900,
            page_height: 16_000,
        });

        let classified =
            parse_semantic_classification("PAGE\tARTWORK\n1\tARTWORK\n2\tARTWORK", &utterances);
        assert!(!classified.page_is_furniture);
        assert_eq!(
            classified.regions,
            [("b".to_owned(), HskTranslationDisposition::PreserveArtwork)]
        );
    }

    #[tokio::test]
    async fn every_visually_admissible_region_requires_focused_artwork_adjudication() -> Result<()>
    {
        let generator = FakeGenerator::new([
            "PAGE\tSTORY\n1\tSTORY\n2\tSTORY",
            "1\tTRANSLATE\n2\tPRESERVE",
        ]);
        let mut utterances = vec![
            source("a", "FORTY REWARD TOKENS"),
            source("b", "CELESTIAL BLADE: THIRD FORM"),
        ];
        for utterance in &mut utterances {
            utterance.semantic_layout = Some(HskSemanticLayout {
                detector_enclosed: true,
                measured_font_height_millionths: 180_000,
                appearance_band_count: 2,
                distinct_text_color_count: 2,
                has_outline: true,
                x0_millionths: 100_000,
                y0_millionths: 200_000,
                x1_millionths: 500_000,
                y1_millionths: 400_000,
                page_width: 900,
                page_height: 16_000,
            });
        }

        let classified =
            classify_semantic_regions_with(&generator, &utterances, &AtomicBool::new(false))
                .await?;
        assert_eq!(
            classified.regions,
            [("b".to_owned(), HskTranslationDisposition::PreserveArtwork)]
        );
        assert_eq!(generator.calls.load(Ordering::Relaxed), 2);
        let artwork_user_prompt = &generator.user_prompts.lock().unwrap()[1];
        assert!(artwork_user_prompt.contains("1\ttext=FORTY REWARD TOKENS"));
        assert!(artwork_user_prompt.contains("2\ttext=CELESTIAL BLADE: THIRD FORM"));
        assert!(artwork_user_prompt.contains("1\tpage-position=1"));
        assert!(artwork_user_prompt.contains("2\tpage-position=2"));
        Ok(())
    }

    #[test]
    fn name_adjudication_receives_generic_syntactic_attachment_evidence() {
        let utterances = vec![
            source("a", "We met the Crimson Captain."),
            source("b", "Crimson Captain arrived."),
        ];
        assert_eq!(
            candidate_syntax(&utterances, "Crimson Captain"),
            "definite-article-before"
        );
        assert_eq!(candidate_syntax(&utterances, "arrived"), "neutral");
    }

    #[test]
    fn name_markup_cannot_create_semantic_name_authority() {
        let source = "Tarin Voss met the senior administrator.";
        let (text, issues) = validate_and_strip_name_markup(
            source,
            "⟦Tarin Voss⟧见到了高级管理员。",
            HskNameHandling::KeepOriginal,
            &[],
        );
        assert_eq!(text, "Tarin Voss见到了高级管理员。");
        assert_eq!(
            issues,
            [
                HskTranslationIssue::InvalidNameMarkup,
                HskTranslationIssue::UnmarkedLatinText,
            ]
        );

        let (_, altered) = validate_and_strip_name_markup(
            source,
            "我见到了⟦TARIN VOSS⟧。",
            HskNameHandling::KeepOriginal,
            &[],
        );
        assert_eq!(
            altered,
            [
                HskTranslationIssue::InvalidNameMarkup,
                HskTranslationIssue::UnmarkedLatinText,
            ]
        );

        let (_, unmarked) = validate_and_strip_name_markup(
            source,
            "Tarin Voss见到了高级管理员。",
            HskNameHandling::KeepOriginal,
            &[],
        );
        assert_eq!(unmarked, [HskTranslationIssue::UnmarkedLatinText]);
    }

    #[test]
    fn approved_keep_original_names_do_not_require_fragile_model_markup() {
        let protected = [HskProtectedName {
            source_english: "Tarin Voss".to_owned(),
            chinese: "Tarin Voss".to_owned(),
        }];
        let (text, issues) = validate_and_strip_name_markup(
            "Tarin Voss met the administrator.",
            "Tarin Voss见到了管理员。",
            HskNameHandling::KeepOriginal,
            &protected,
        );

        assert_eq!(text, "Tarin Voss见到了管理员。");
        assert!(issues.is_empty());
    }

    fn source(id: &str, source_english: &str) -> HskSourceUtterance {
        HskSourceUtterance {
            id: id.to_owned(),
            kind: HskUtteranceKind::Dialogue,
            source_english: source_english.to_owned(),
            semantic_layout: None,
        }
    }

    fn request() -> HskTranslationBatchRequest {
        HskTranslationBatchRequest {
            requested_level: 2,
            learning_mode: HskLearningMode::Strict,
            name_handling: HskNameHandling::Chinese,
            translate_sound_effects: true,
            utterances: vec![
                source("private-bubble-a", "Alice does not have 2 tickets."),
                source("private-bubble-b", "Are you ready?"),
                source("private-bubble-c", "Let's go!"),
            ],
            preceding_utterances: (0..8)
                .map(|index| HskPrecedingUtterance {
                    source_english: format!("english-context-{index}"),
                    chinese: format!("chinese-context-{index}"),
                })
                .collect(),
            protected_names: vec![HskProtectedName {
                source_english: "Alice".to_owned(),
                chinese: "爱丽丝".to_owned(),
            }],
        }
    }

    #[tokio::test]
    async fn direct_batch_is_one_compact_greedy_generation_with_six_context_items() -> Result<()> {
        let generator = FakeGenerator::new([concat!(
            "1\t爱丽丝没有2张票。\n",
            "2\t你准备好了吗？\n",
            "3\t我们走吧！"
        )]);
        let result = translate_with(&generator, &request(), &AtomicBool::new(false)).await?;

        assert!(result.items.iter().all(HskTranslationOutcome::is_valid));
        assert_eq!(
            result
                .items
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["private-bubble-a", "private-bubble-b", "private-bubble-c"]
        );
        assert_eq!(generator.calls.load(Ordering::Relaxed), 1);

        let options = generator.options.lock().unwrap();
        assert!(options[0].max_tokens < 512);
        assert_eq!(options[0].temperature, 0.0);
        assert_eq!(options[0].top_k, None);
        assert_eq!(options[0].top_p, None);
        assert_eq!(options[0].min_p, None);
        assert_eq!(options[0].repeat_penalty, 1.0);
        assert_eq!(options[0].repeat_last_n, 0);
        assert_eq!(options[0].presence_penalty, 0.0);
        assert_eq!(options[0].grammar, None);

        let prompt = &generator.user_prompts.lock().unwrap()[0];
        assert!(prompt.contains("Previous translations (reference only; do not output):"));
        assert!(prompt.contains("english-context-2"));
        assert!(prompt.contains("english-context-7"));
        assert!(!prompt.contains("english-context-0"));
        assert!(!prompt.contains("english-context-1"));
        assert!(prompt.contains("- english-context-2 => chinese-context-2"));
        assert!(!prompt.contains("1\tenglish-context-2"));
        assert!(!prompt.contains("N\tAlice\t"));
        assert!(
            prompt.contains("- 1: approved glossary \"Alice\" => \"爱丽丝\" (use this exact form)")
        );
        assert!(prompt.contains("1\tAlice does not have 2 tickets."));
        assert!(!prompt.contains("INPUT\t"));
        assert!(!prompt.contains("\tD\t"));
        assert!(!prompt.contains("private-bubble"));
        assert!(generator.system_prompts.lock().unwrap()[0].contains("HSK 2.0 level 2"));
        assert!(generator.system_prompts.lock().unwrap()[0].contains("exactly 3 non-empty lines"));
        assert!(generator.system_prompts.lock().unwrap()[0].contains("start with `1\t`"));
        Ok(())
    }

    #[tokio::test]
    async fn oversized_translation_batches_are_partitioned_before_generation() -> Result<()> {
        let generator = MaxTwoUtteranceGenerator::new([
            concat!("1\t爱丽丝没有2张票。\n", "2\t你准备好了吗？"),
            "1\t我们走吧！",
        ]);

        let result = translate_with(&generator, &request(), &AtomicBool::new(false)).await?;

        assert!(result.items.iter().all(HskTranslationOutcome::is_valid));
        assert_eq!(generator.inner.calls.load(Ordering::Relaxed), 2);
        let prompts = generator.inner.user_prompts.lock().unwrap();
        assert_eq!(
            prompts
                .iter()
                .map(|prompt| {
                    prompt
                        .lines()
                        .filter(|line| {
                            line.split_once('\t')
                                .is_some_and(|(position, _)| position.parse::<usize>().is_ok())
                        })
                        .count()
                })
                .collect::<Vec<_>>(),
            [2, 1]
        );
        assert!(
            prompts
                .iter()
                .all(|prompt| !prompt.contains("Previous translations"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn completed_numbered_lines_stream_in_application_order() -> Result<()> {
        let generator = FakeGenerator::new(["1\t\u{4f60}\u{597d}\n2\t\u{597d}\n"]);
        let mut input = request();
        input.utterances = vec![source("bubble-a", "Hello"), source("bubble-b", "Ready")];
        input.preceding_utterances.clear();
        input.protected_names.clear();
        let mut streamed = Vec::new();

        let result = translate_with_streaming(
            &generator,
            &input,
            &AtomicBool::new(false),
            &mut |outcome| {
                streamed.push((outcome.id.clone(), outcome.text.clone()));
                Ok(())
            },
        )
        .await?;

        assert_eq!(
            streamed,
            vec![
                ("bubble-a".to_owned(), Some("\u{4f60}\u{597d}".to_owned())),
                ("bubble-b".to_owned(), Some("\u{597d}".to_owned())),
            ]
        );
        assert!(result.items.iter().all(HskTranslationOutcome::is_valid));
        Ok(())
    }

    #[tokio::test]
    async fn context_uses_real_token_budget_after_six_item_bound() -> Result<()> {
        let generator = FakeGenerator::new([]);
        let context = (0..8)
            .map(|index| HskPrecedingUtterance {
                source_english: format!("{index}-{}", "x".repeat(90)),
                chinese: "y".repeat(30),
            })
            .collect::<Vec<_>>();

        let bounded = bounded_context(&generator, &context).await?;
        let rendered = render_context_for_budget(&bounded);

        assert!(bounded.len() <= MAX_HSK_PRECEDING_UTTERANCES);
        assert!(rendered.chars().count() <= MAX_HSK_CONTEXT_TOKENS);
        assert_eq!(
            bounded.last().map(|item| item.source_english.as_str()),
            context.last().map(|item| item.source_english.as_str())
        );
        Ok(())
    }

    #[tokio::test]
    async fn malformed_items_do_not_discard_valid_siblings_or_trigger_a_retry() -> Result<()> {
        let generator = FakeGenerator::new([concat!(
            "commentary that is ignored\n",
            "1\t爱丽丝有3张票。\n",
            "2. 你准备好了吗？\n",
            "3\t\n",
            "99\tunexpected"
        )]);
        let result = translate_with(&generator, &request(), &AtomicBool::new(false)).await?;

        assert_eq!(generator.calls.load(Ordering::Relaxed), 1);
        assert!(!result.items[0].is_valid());
        assert_eq!(result.items[0].text.as_deref(), Some("爱丽丝有3张票。"));
        assert!(
            result.items[0]
                .issues
                .iter()
                .any(|issue| matches!(issue, HskTranslationIssue::NumberMismatch { .. }))
        );
        assert!(result.items[1].issues.is_empty());
        assert_eq!(
            result.items[2].issues,
            vec![HskTranslationIssue::EmptyTranslation]
        );
        Ok(())
    }

    #[test]
    fn parser_maps_out_of_order_lines_and_isolates_duplicate_and_missing_positions() {
        let input = request();
        let expected = input
            .utterances
            .iter()
            .map(|utterance| ExpectedUtterance {
                id: &utterance.id,
                source_english: &utterance.source_english,
            })
            .collect::<Vec<_>>();
        let result = parse_numbered_output(
            "2\t你准备好了吗？\r\n1\t爱丽丝没有２张票。\r\n1\t重复",
            &expected,
            &input.protected_names,
        );

        assert_eq!(
            result.items[0].issues,
            vec![HskTranslationIssue::DuplicateLine]
        );
        assert!(result.items[1].is_valid());
        assert_eq!(
            result.items[2].issues,
            vec![HskTranslationIssue::MissingLine]
        );
    }

    #[test]
    fn parser_returns_explicit_non_story_disposition_without_translation() {
        let input = request();
        let expected = input
            .utterances
            .iter()
            .map(|utterance| ExpectedUtterance {
                id: &utterance.id,
                source_english: &utterance.source_english,
            })
            .collect::<Vec<_>>();
        let result = parse_numbered_output(
            "1\t[NON-STORY]\n2\t你准备好了吗？\n3\t我们走吧！",
            &expected,
            &input.protected_names,
        );

        assert!(result.items[0].is_non_story());
        assert!(!result.items[0].is_valid());
        assert!(
            result.items[1..]
                .iter()
                .all(HskTranslationOutcome::is_valid)
        );
    }

    #[test]
    fn sound_effect_disposition_is_controlled_by_the_user_policy() {
        let expected = [ExpectedUtterance {
            id: "effect",
            source_english: "KLANG!",
        }];
        let excluded = parse_numbered_output_with_name_handling(
            "1\t[SFX]",
            &expected,
            &[],
            HskNameHandling::Chinese,
            false,
        );
        assert_eq!(
            excluded.items[0].disposition,
            HskTranslationDisposition::ExcludeSoundEffect
        );
        assert!(excluded.items[0].issues.is_empty());

        let translated = parse_numbered_output_with_name_handling(
            "1\t[SFX]",
            &expected,
            &[],
            HskNameHandling::Chinese,
            true,
        );
        assert_eq!(
            translated.items[0].disposition,
            HskTranslationDisposition::Translate
        );
        assert_eq!(
            translated.items[0].issues,
            vec![HskTranslationIssue::MalformedLine]
        );
    }

    #[test]
    fn repair_can_confirm_a_sound_effect_exclusion_only_when_disabled() {
        let expected = ExpectedUtterance {
            id: "effect",
            source_english: "THUD!",
        };
        let excluded = parse_repair_output_with_name_handling(
            SOUND_EFFECT_MARKER,
            &expected,
            &[],
            HskNameHandling::Chinese,
            false,
        );
        assert_eq!(
            excluded.disposition,
            HskTranslationDisposition::ExcludeSoundEffect
        );
        assert!(excluded.issues.is_empty());

        let invalid = parse_repair_output_with_name_handling(
            SOUND_EFFECT_MARKER,
            &expected,
            &[],
            HskNameHandling::Chinese,
            true,
        );
        assert_eq!(invalid.disposition, HskTranslationDisposition::Translate);
        assert_eq!(invalid.issues, vec![HskTranslationIssue::MalformedLine]);
    }

    #[test]
    fn repair_parser_rejects_non_story_disposition() {
        let expected = ExpectedUtterance {
            id: "story-region",
            source_english: "A real story line.",
        };
        let outcome = parse_repair_output(NON_STORY_MARKER, &expected, &[]);

        assert_eq!(outcome.issues, vec![HskTranslationIssue::MalformedLine]);
    }

    #[test]
    fn one_item_repair_batch_uses_the_single_line_protocol() {
        let expected = [ExpectedUtterance {
            id: "line-1",
            source_english: "CHEERS!",
        }];
        let parsed = parse_repair_batch_output(
            "干杯！",
            &expected,
            &[],
            HskNameHandling::KeepOriginal,
            true,
        );

        assert_eq!(parsed.items.len(), 1);
        assert_eq!(parsed.items[0].id, "line-1");
        assert_eq!(parsed.items[0].text.as_deref(), Some("干杯！"));
        assert!(parsed.items[0].issues.is_empty());
    }

    #[tokio::test]
    async fn direct_source_echo_is_ignored_instead_of_creating_a_duplicate_position() -> Result<()>
    {
        let generator = FakeGenerator::new([concat!(
            "1\tAlice does not have 2 tickets.\n",
            "1\t爱丽丝没有2张票。\n",
            "2\t你准备好了吗？\n",
            "3\t我们走吧！"
        )]);
        let result = translate_with(&generator, &request(), &AtomicBool::new(false)).await?;

        assert!(result.items.iter().all(HskTranslationOutcome::is_valid));
        assert_eq!(result.items[0].text.as_deref(), Some("爱丽丝没有2张票。"));
        Ok(())
    }

    #[tokio::test]
    async fn repair_is_one_chinese_only_call_for_one_application_owned_id() -> Result<()> {
        let generator = FakeGenerator::new(["1\t她有票。\n2\t你准备好了吗？", "爱丽丝没有2张票。"]);
        let input = request();
        let initial = translate_with(&generator, &input, &AtomicBool::new(false)).await?;
        assert!(initial.items[1].is_valid());
        assert!(!initial.items[0].is_valid());
        assert!(!initial.items[2].is_valid());

        let repair = HskTranslationRepairRequest {
            requested_level: input.requested_level,
            learning_mode: input.learning_mode,
            name_handling: input.name_handling,
            translate_sound_effects: input.translate_sound_effects,
            utterance: HskRepairUtterance {
                id: input.utterances[0].id.clone(),
                kind: input.utterances[0].kind,
                source_english: input.utterances[0].source_english.clone(),
                rejected_chinese: initial.items[0].text.clone(),
                avoid_chinese: Vec::new(),
                problems: initial.items[0].repair_problems(),
            },
            preceding_utterances: input.preceding_utterances.clone(),
            protected_names: input.protected_names.clone(),
        };
        let repaired = repair_with(&generator, &repair, &AtomicBool::new(false)).await?;

        assert_eq!(generator.calls.load(Ordering::Relaxed), 2);
        assert!(repaired.is_valid());
        assert_eq!(repaired.id, "private-bubble-a");

        let prompts = generator.user_prompts.lock().unwrap();
        let repair_prompt = &prompts[1];
        assert!(repair_prompt.contains("爱丽丝 does not have 2 tickets."));
        assert!(!repair_prompt.contains("Are you ready?"));
        assert!(!repair_prompt.contains("Let's go!"));
        assert!(!repair_prompt.contains("Previous translations"));
        assert!(!repair_prompt.contains("english-context-"));
        assert!(!repair_prompt.contains("chinese-context-"));
        assert!(!repair_prompt.contains("private-bubble"));
        assert!(!repair_prompt.lines().any(|line| line.starts_with("1\t")));
        assert!(generator.system_prompts.lock().unwrap()[1].contains("this one"));
        assert!(generator.system_prompts.lock().unwrap()[1].contains("no position"));
        Ok(())
    }

    #[test]
    fn repair_parser_rejects_source_and_prior_numbered_line_echoes() {
        let input = request();
        let expected = ExpectedUtterance {
            id: &input.utterances[0].id,
            source_english: &input.utterances[0].source_english,
        };
        let repaired =
            parse_repair_output("1\t2\t爱丽丝没有2张票。", &expected, &input.protected_names);

        assert_eq!(repaired.id, "private-bubble-a");
        assert_eq!(repaired.text, None);
        assert_eq!(repaired.issues, vec![HskTranslationIssue::MalformedLine]);

        let source_echo = parse_repair_output(
            &input.utterances[0].source_english,
            &expected,
            &input.protected_names,
        );
        assert_eq!(source_echo.text, None);
        assert_eq!(source_echo.issues, vec![HskTranslationIssue::SourceEcho]);

        let prior_lines = parse_repair_output(
            "1\t你准备好了吗？\n2\t我们走吧！",
            &expected,
            &input.protected_names,
        );
        assert_eq!(prior_lines.text, None);
        assert_eq!(prior_lines.issues, vec![HskTranslationIssue::MalformedLine]);
    }

    #[test]
    fn deterministic_validator_preserves_question_intent_with_names_and_numbers() {
        let input = request();
        assert!(
            preservation_issues(
                "Does Alice not have 2 tickets?",
                "爱丽丝没有2张票。",
                &input.protected_names,
                true,
            )
            .contains(&HskTranslationIssue::QuestionIntentMissing)
        );
        assert!(
            preservation_issues(
                "Does Alice not have 2 tickets?",
                "爱丽丝没有2张票？",
                &input.protected_names,
                true,
            )
            .is_empty()
        );
    }

    #[test]
    fn parser_restores_unambiguous_question_punctuation_without_regeneration() {
        let expected = [ExpectedUtterance {
            id: "question",
            source_english: "Where does your loyalty lie?",
        }];
        let result = parse_numbered_output("1\t你的忠诚在哪里。", &expected, &[]);

        assert_eq!(result.items[0].text.as_deref(), Some("你的忠诚在哪里？"));
        assert!(
            !result.items[0]
                .issues
                .contains(&HskTranslationIssue::QuestionIntentMissing)
        );
    }

    #[test]
    fn negative_do_imperative_is_not_misclassified_as_a_question() {
        assert!(!has_question_intent(
            "do not mourn those who left before us."
        ));
        assert!(has_question_intent("do you know who left?"));
        assert!(!has_question_intent(
            "has it been seven years since we faced each other in person,"
        ));
        assert!(!has_question_intent("what on earth..."));
    }

    #[test]
    fn deterministic_validator_preserves_ascii_protected_names_and_numbers_exactly() {
        let names = vec![HskProtectedName {
            source_english: "Jin".to_owned(),
            chinese: "JIN".to_owned(),
        }];
        assert!(
            preservation_issues(
                "Does Jin not have 27 keys?",
                "JIN没有27把钥匙？",
                &names,
                true,
            )
            .is_empty()
        );
        let issues = preservation_issues(
            "Does Jin not have 27 keys?",
            "Jin没有二十七把钥匙？",
            &names,
            true,
        );
        assert!(
            issues
                .iter()
                .any(|issue| matches!(issue, HskTranslationIssue::ProtectedNameMissing { .. }))
        );
        assert!(
            !issues
                .iter()
                .any(|issue| matches!(issue, HskTranslationIssue::NumberMismatch { .. }))
        );
    }

    #[test]
    fn deterministic_validator_ignores_digits_embedded_in_latin_ocr_tokens() {
        assert_eq!(
            ascii_numbers("IDENTIT4, WH4, M4, but 7 years and 120 people."),
            vec!["7", "120"]
        );
        assert!(
            preservation_issues(
                "I found your IDENTIT4 and returned after 7 years.",
                "我找到你的身份，7年后回来了。",
                &[],
                false,
            )
            .is_empty()
        );
    }

    #[test]
    fn deterministic_validator_preserves_multiplier_notation_without_treating_ocr_noise_as_numbers()
    {
        assert_eq!(
            ascii_numbers("THIRTY OF THEM!!! X3; another 3x, but IDENTIT4 and M4"),
            vec!["3", "3"]
        );
        assert!(preservation_issues("THIRTY OF THEM!!! X3", "三十个！×3", &[], false,).is_empty());
    }

    #[test]
    fn deterministic_validator_rejects_cross_item_expansion_of_short_fragments() {
        let issues = preservation_issues(
            "\"ASSASSINATION REQUESTS.\"",
            "以及那个策划了肃清小组的阴影之刃。",
            &[],
            true,
        );
        assert!(issues.iter().any(|issue| {
            matches!(
                issue,
                HskTranslationIssue::ExcessiveExpansion {
                    source_words: 2,
                    chinese_characters: 16
                }
            )
        }));
        assert!(
            preservation_issues("\"ASSASSINATION REQUESTS.\"", "“暗杀请求。”", &[], true,)
                .is_empty()
        );
        assert_eq!(english_word_count("No, wait—I meant this."), 5);
        assert_eq!(chinese_character_count("不是，等等——我是说这个。"), 9);
    }

    #[test]
    fn parser_accepts_numbered_spaces_and_normalizes_source_numbers() {
        let input = request();
        let expected = input
            .utterances
            .iter()
            .map(|utterance| ExpectedUtterance {
                id: &utterance.id,
                source_english: &utterance.source_english,
            })
            .collect::<Vec<_>>();
        let result = parse_numbered_output(
            "1 爱丽丝没有二张票。\n2 你准备好了吗？\n3 我们走吧！",
            &expected,
            &input.protected_names,
        );

        assert!(!result.items[0].is_valid());
        assert!(
            result.items[1..]
                .iter()
                .all(HskTranslationOutcome::is_valid)
        );
        assert_eq!(result.items[0].text.as_deref(), Some("爱丽丝没有二张票。"));
        let repaired =
            parse_repair_output("爱丽丝没有二张票。", &expected[0], &input.protected_names);
        assert!(repaired.is_valid());
        assert_eq!(chinese_integer_below_10_000(27), "二十七");
        assert_eq!(chinese_integer_below_10_000(2_006), "二千零六");
    }

    #[test]
    fn one_item_primary_accepts_a_single_direct_chinese_line() {
        let expected = [ExpectedUtterance {
            id: "item",
            source_english: "GOLD CANDIES.",
        }];
        let result = parse_numbered_output("金色糖果。", &expected, &[]);
        assert_eq!(result.items[0].text.as_deref(), Some("金色糖果。"));
        assert!(result.items[0].issues.is_empty());
    }

    #[test]
    fn source_multiplier_notation_is_normalized_before_latin_validation() {
        let expected = [ExpectedUtterance {
            id: "item",
            source_english: "THIRTY OF THEM!!! X3",
        }];
        let result = parse_numbered_output("1\t他们三十个！！！X3", &expected, &[]);
        assert_eq!(result.items[0].text.as_deref(), Some("他们三十个！！！×3"));
        assert!(result.items[0].issues.is_empty());
    }

    #[tokio::test]
    async fn empty_batches_and_pre_cancelled_calls_do_not_generate() -> Result<()> {
        let generator = FakeGenerator::new([]);
        let mut input = request();
        input.utterances.clear();
        assert!(
            translate_with(&generator, &input, &AtomicBool::new(false))
                .await?
                .items
                .is_empty()
        );

        input.utterances.push(source("id", "Hello"));
        let error = translate_with(&generator, &input, &AtomicBool::new(true))
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "cancelled");
        assert_eq!(generator.calls.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn output_budgets_are_tight_and_bounded_without_a_512_floor() {
        assert_eq!(output_token_budget(["Hi"].into_iter(), 1), 24);
        assert!(output_token_budget(["A short sentence."].into_iter(), 1) < 512);
        assert_eq!(
            output_token_budget(["x".repeat(242).as_str()].into_iter(), 6),
            177
        );
        let long = "x".repeat(10_000);
        assert_eq!(
            output_token_budget([long.as_str()].into_iter(), 1),
            MAX_OUTPUT_TOKENS
        );
    }

    #[test]
    fn name_handling_changes_primary_and_repair_instructions() {
        let original = translation_system_prompt(
            3,
            1,
            HskNameHandling::KeepOriginal,
            true,
            HskLearningMode::Strict,
        );
        let chinese = translation_system_prompt(
            3,
            1,
            HskNameHandling::Chinese,
            true,
            HskLearningMode::Strict,
        );
        let repair = repair_system_prompt(
            3,
            HskNameHandling::KeepOriginal,
            true,
            HskLearningMode::Strict,
        );

        assert!(original.contains("complete approved proper-name set"));
        assert!(original.contains("Never invent another name"));
        assert!(original.contains("Translate every other Latin word"));
        assert!(repair.contains("complete approved proper-name set"));
        assert!(repair.contains("opaque approved-name placeholder"));
        assert!(chinese.contains("phonetic Chinese transliteration"));
    }

    #[test]
    fn cache_metadata_helpers_expose_protocol_owned_identities() {
        assert_eq!(direct_hsk_prompt_hash(), HSK_TRANSLATION_PROMPT_HASH);
        assert_eq!(direct_hsk_validator_hash(), HSK_TRANSLATION_VALIDATOR_HASH);
        assert!(direct_hsk_prompt_hash().starts_with("sha256:"));
        assert!(direct_hsk_validator_hash().starts_with("sha256:"));
        assert_ne!(direct_hsk_prompt_hash(), direct_hsk_validator_hash());
        assert_eq!(HSK_TRANSLATION_MODEL, ModelId::Qwen3_5_4b);
        assert_eq!(
            HSK_TRANSLATION_MODEL_REVISION,
            "unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523:Qwen3.5-4B-Q4_K_M.gguf:sha256=00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
        );
    }

    #[test]
    fn request_validation_rejects_bad_levels_ids_names_and_repair_feedback() {
        for size in 1..=MAX_HSK_TRANSLATION_BATCH {
            let mut input = request();
            input.utterances = (0..size)
                .map(|index| source(&format!("bubble-{index}"), "Hello"))
                .collect();
            validate_translation_request(&input).unwrap();
        }

        let mut input = request();
        input.requested_level = 0;
        assert!(
            validate_translation_request(&input)
                .unwrap_err()
                .to_string()
                .contains("1 through 6")
        );

        input.requested_level = 2;
        input.utterances[1].id = input.utterances[0].id.clone();
        assert!(
            validate_translation_request(&input)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );

        input = request();
        input.protected_names.push(HskProtectedName {
            source_english: "alice".to_owned(),
            chinese: "艾丽斯".to_owned(),
        });
        assert!(
            validate_translation_request(&input)
                .unwrap_err()
                .to_string()
                .contains("conflicting")
        );

        let repair = HskTranslationRepairRequest {
            requested_level: 2,
            learning_mode: HskLearningMode::Strict,
            name_handling: HskNameHandling::Chinese,
            translate_sound_effects: true,
            utterance: HskRepairUtterance {
                id: "id".to_owned(),
                kind: HskUtteranceKind::Dialogue,
                source_english: "Hello".to_owned(),
                rejected_chinese: None,
                avoid_chinese: Vec::new(),
                problems: Vec::new(),
            },
            preceding_utterances: Vec::new(),
            protected_names: Vec::new(),
        };
        assert!(
            validate_repair_request(&repair)
                .unwrap_err()
                .to_string()
                .contains("requires non-empty problems")
        );

        input = request();
        input.utterances.extend([
            source("private-bubble-d", "Four"),
            source("private-bubble-e", "Five"),
            source("private-bubble-f", "Six"),
            source("private-bubble-g", "Seven"),
        ]);
        assert!(
            validate_translation_request(&input)
                .unwrap_err()
                .to_string()
                .contains("at most 6")
        );
    }
}
