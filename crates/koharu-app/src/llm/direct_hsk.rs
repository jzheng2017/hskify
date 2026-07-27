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
    DirectHskContext, DirectHskName, DirectHskNameStyle, context_budget_text,
    primary_system_prompt_with_name_style, primary_user_prompt,
    repair_system_prompt_with_name_style, repair_user_prompt,
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
// Composite cache identity: repository@commit, filename, and exact file digest.
pub const HSK_TRANSLATION_MODEL_REVISION: &str = "unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523:Qwen3.5-4B-Q4_K_M.gguf:sha256=00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4";
pub const MAX_HSK_PRECEDING_UTTERANCES: usize = 6;
pub const MAX_HSK_CONTEXT_TOKENS: usize = 256;
pub const MAX_HSK_TRANSLATION_BATCH: usize = 6;

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskTranslationBatchRequest {
    pub requested_level: u8,
    #[serde(default)]
    pub name_handling: HskNameHandling,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskTranslationOutcome {
    pub id: String,
    #[serde(default)]
    pub disposition: HskTranslationDisposition,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub declared_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<HskTranslationIssue>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HskTranslationDisposition {
    #[default]
    Translate,
    ExcludeNonStory,
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
        self.disposition == HskTranslationDisposition::ExcludeNonStory
            && self.issues.is_empty()
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
                "wrap each retained proper name once as `⟦exact source spelling⟧`".to_owned()
            }
            Self::UnmarkedLatinText => {
                "translate ordinary English; retain only proper names wrapped as `⟦name⟧`"
                    .to_owned()
            }
        }
    }
}

/// A repair request contains exactly one candidate rejected by parsing,
/// preservation checks, or the caller's deterministic HSK vocabulary
/// validator. The caller may schedule at most one such request per bubble.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskTranslationRepairRequest {
    pub requested_level: u8,
    #[serde(default)]
    pub name_handling: HskNameHandling,
    pub utterance: HskRepairUtterance,
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
    pub problems: Vec<String>,
}

trait Generator {
    async fn token_count(&self, text: &str) -> Result<usize>;

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

    /// Perform the sole targeted repair operation for one rejected bubble.
    pub async fn repair_invalid_item(
        &self,
        request: &HskTranslationRepairRequest,
        cancel: &AtomicBool,
    ) -> Result<HskTranslationOutcome> {
        repair_with(self.model, request, cancel).await
    }
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

    let mut bounded_request = request.clone();
    bounded_request.preceding_utterances =
        bounded_context(generator, &request.preceding_utterances).await?;
    let prompt = build_translation_prompt(&bounded_request);
    let options = GenerateOptions::greedy(output_token_budget(
        bounded_request
            .utterances
            .iter()
            .map(|utterance| utterance.source_english.as_str()),
        bounded_request.utterances.len(),
    ));
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

fn translation_system_prompt(level: u8, count: usize, name_handling: HskNameHandling) -> String {
    primary_system_prompt_with_name_style(level, count, name_handling.into())
}

fn repair_system_prompt(level: u8, name_handling: HskNameHandling) -> String {
    repair_system_prompt_with_name_style(level, name_handling.into())
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
    primary_user_prompt(&context, &names, &sources)
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
    repair_user_prompt(
        &utterance.source_english,
        utterance.rejected_chinese.as_deref(),
        &problems,
        &names,
    )
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
    let utterance = &request.utterance;
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
    )
}

fn parse_numbered_output_with_name_handling(
    output: &str,
    expected: &[ExpectedUtterance<'_>],
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
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

    let items = expected
        .iter()
        .zip(slots)
        .map(|(expected, lines)| {
            outcome_from_lines(expected, lines, protected_names, name_handling)
        })
        .collect();
    HskTranslationBatchResult { items }
}

fn parse_streamed_line(
    line: &str,
    expected: &[ExpectedUtterance<'_>],
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
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
    ))
}

fn parse_repair_output_with_name_handling(
    output: &str,
    expected: &ExpectedUtterance<'_>,
    protected_names: &[HskProtectedName],
    name_handling: HskNameHandling,
) -> HskTranslationOutcome {
    let mut lines = output
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .filter(|line| !line.trim().is_empty());
    let Some(line) = lines.next() else {
        return outcome_from_lines(expected, Vec::new(), protected_names, name_handling);
    };
    if lines.next().is_some() || line.contains('\t') {
        return outcome_from_lines(
            expected,
            vec![ParsedLine::Malformed],
            protected_names,
            name_handling,
        );
    }
    if compact_field(line) == compact_field(expected.source_english) {
        return HskTranslationOutcome {
            id: expected.id.to_owned(),
            disposition: HskTranslationDisposition::Translate,
            text: None,
            declared_names: Vec::new(),
            issues: vec![HskTranslationIssue::SourceEcho],
        };
    }
    if line.trim().eq_ignore_ascii_case(NON_STORY_MARKER) {
        return outcome_from_lines(
            expected,
            vec![ParsedLine::Malformed],
            protected_names,
            name_handling,
        );
    }
    let mut outcome = outcome_from_lines(
        expected,
        vec![ParsedLine::Candidate {
            text: line.to_owned(),
        }],
        protected_names,
        name_handling,
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
) -> HskTranslationOutcome {
    if lines.is_empty() {
        return HskTranslationOutcome {
            id: expected.id.to_owned(),
            disposition: HskTranslationDisposition::Translate,
            text: None,
            declared_names: Vec::new(),
            issues: vec![HskTranslationIssue::MissingLine],
        };
    }
    if lines.len() > 1 {
        let text = lines.drain(..).find_map(|line| match line {
            ParsedLine::Candidate { text, .. } if !text.trim().is_empty() => Some(text),
            ParsedLine::Candidate { .. } | ParsedLine::ExcludeNonStory | ParsedLine::Malformed => {
                None
            }
        });
        return HskTranslationOutcome {
            id: expected.id.to_owned(),
            disposition: HskTranslationDisposition::Translate,
            text,
            declared_names: Vec::new(),
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
                declared_names: Vec::new(),
                issues: Vec::new(),
            };
        }
        ParsedLine::Malformed => {
            return HskTranslationOutcome {
                id: expected.id.to_owned(),
                disposition: HskTranslationDisposition::Translate,
                text: None,
                declared_names: Vec::new(),
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
            declared_names: Vec::new(),
            issues: vec![HskTranslationIssue::EmptyTranslation],
        };
    }

    let (mut text, declared_names, mut markup_issues) =
        validate_and_strip_name_markup(expected.source_english, &text, name_handling);
    normalize_full_width_digits(&mut text);
    normalize_question_punctuation(expected.source_english, &mut text);
    let mut issues = preservation_issues(expected.source_english, &text, protected_names, false);
    issues.append(&mut markup_issues);
    HskTranslationOutcome {
        id: expected.id.to_owned(),
        disposition: HskTranslationDisposition::Translate,
        text: Some(text),
        declared_names,
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

    let text_start = match line.as_bytes().get(digit_count) {
        Some(b'\t') => digit_count + 1,
        // Qwen occasionally emits the requested compact numbered protocol
        // with an ASCII space instead of a tab. The position is still
        // unambiguous, so accept it without spending the sole repair.
        Some(b' ') => {
            line.as_bytes()[digit_count..]
                .iter()
                .take_while(|byte| **byte == b' ')
                .count()
                + digit_count
        }
        _ => return Some((position, ParsedLine::Malformed)),
    };
    let text = &line[text_start..];
    if text.contains('\t') {
        return Some((position, ParsedLine::Malformed));
    }
    if text.trim().eq_ignore_ascii_case(NON_STORY_MARKER) {
        return Some((position, ParsedLine::ExcludeNonStory));
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
) -> (String, Vec<String>, Vec<HskTranslationIssue>) {
    const OPEN: char = '⟦';
    const CLOSE: char = '⟧';

    let mut output = String::with_capacity(translation.len());
    let mut current_name = None::<String>;
    let mut names = Vec::<String>::new();
    let mut invalid_markup = name_handling == HskNameHandling::Chinese
        && translation
            .chars()
            .any(|character| matches!(character, OPEN | CLOSE));
    let mut unmarked_latin = false;

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
                {
                    invalid_markup = true;
                } else if !names.contains(&name) {
                    names.push(name.clone());
                }
                output.push_str(&name);
            }
            _ => {
                if let Some(name) = current_name.as_mut() {
                    name.push(character);
                } else {
                    unmarked_latin |= character.is_ascii_alphabetic();
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
    if name_handling == HskNameHandling::KeepOriginal && unmarked_latin {
        issues.push(HskTranslationIssue::UnmarkedLatinText);
    }
    if name_handling == HskNameHandling::Chinese {
        names.clear();
    }
    (output, names, issues)
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
            let touches_latin_ocr = number_start
                .checked_sub(1)
                .is_some_and(|before| bytes[before].is_ascii_alphabetic())
                || byte.is_ascii_alphabetic();
            if !touches_latin_ocr {
                numbers.push(text[number_start..index].to_owned());
            }
        }
    }
    if let Some(number_start) = start {
        let touches_latin_ocr = number_start
            .checked_sub(1)
            .is_some_and(|before| bytes[before].is_ascii_alphabetic());
        if !touches_latin_ocr {
            numbers.push(text[number_start..].to_owned());
        }
    }
    numbers
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
    async fn keep_original_names_are_decided_inside_the_single_translation_call() -> Result<()> {
        let generator = FakeGenerator::new(["1\t我昨天见到了⟦Tarin Voss⟧。"]);
        let input = HskTranslationBatchRequest {
            requested_level: 3,
            name_handling: HskNameHandling::KeepOriginal,
            utterances: vec![source("dialogue", "I saw Tarin Voss yesterday.")],
            preceding_utterances: Vec::new(),
            protected_names: Vec::new(),
        };

        let result = translate_with(&generator, &input, &AtomicBool::new(false)).await?;

        assert!(result.items[0].is_valid());
        assert_eq!(
            result.items[0].text.as_deref(),
            Some("我昨天见到了Tarin Voss。")
        );
        assert_eq!(result.items[0].declared_names, ["Tarin Voss"]);
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
            name_handling: HskNameHandling::KeepOriginal,
            utterances: vec![source("dialogue", "THE SENIOR ADMINISTRATOR ARRIVED.")],
            preceding_utterances: Vec::new(),
            protected_names: Vec::new(),
        };

        let result = translate_with(&generator, &input, &AtomicBool::new(false)).await?;

        assert!(result.items[0].is_valid());
        assert!(result.items[0].declared_names.is_empty());
        Ok(())
    }

    #[test]
    fn name_markup_is_mechanical_exact_and_never_semantic_code() {
        let source = "Tarin Voss met the senior administrator.";
        let (text, names, issues) = validate_and_strip_name_markup(
            source,
            "⟦Tarin Voss⟧见到了高级管理员。",
            HskNameHandling::KeepOriginal,
        );
        assert_eq!(text, "Tarin Voss见到了高级管理员。");
        assert_eq!(names, ["Tarin Voss"]);
        assert!(issues.is_empty());

        let (_, _, altered) = validate_and_strip_name_markup(
            source,
            "我见到了⟦TARIN VOSS⟧。",
            HskNameHandling::KeepOriginal,
        );
        assert_eq!(altered, [HskTranslationIssue::InvalidNameMarkup]);

        let (_, _, unmarked) = validate_and_strip_name_markup(
            source,
            "Tarin Voss见到了高级管理员。",
            HskNameHandling::KeepOriginal,
        );
        assert_eq!(unmarked, [HskTranslationIssue::UnmarkedLatinText]);
    }

    fn source(id: &str, source_english: &str) -> HskSourceUtterance {
        HskSourceUtterance {
            id: id.to_owned(),
            kind: HskUtteranceKind::Dialogue,
            source_english: source_english.to_owned(),
        }
    }

    fn request() -> HskTranslationBatchRequest {
        HskTranslationBatchRequest {
            requested_level: 2,
            name_handling: HskNameHandling::Chinese,
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
        assert_eq!(
            result.items[1].issues,
            vec![HskTranslationIssue::MalformedLine]
        );
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
    fn repair_parser_rejects_non_story_disposition() {
        let expected = ExpectedUtterance {
            id: "story-region",
            source_english: "A real story line.",
        };
        let outcome = parse_repair_output(NON_STORY_MARKER, &expected, &[]);

        assert_eq!(outcome.issues, vec![HskTranslationIssue::MalformedLine]);
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
            name_handling: input.name_handling,
            utterance: HskRepairUtterance {
                id: input.utterances[0].id.clone(),
                kind: input.utterances[0].kind,
                source_english: input.utterances[0].source_english.clone(),
                rejected_chinese: initial.items[0].text.clone(),
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
        let original = translation_system_prompt(3, 1, HskNameHandling::KeepOriginal);
        let chinese = translation_system_prompt(3, 1, HskNameHandling::Chinese);
        let repair = repair_system_prompt(3, HskNameHandling::KeepOriginal);

        assert!(original.contains("exactly as written in the English source"));
        assert!(repair.contains("Decide proper names from the complete source meaning"));
        assert!(repair.contains("⟦exact source spelling⟧"));
        assert!(chinese.contains("phonetic Chinese transliteration"));
    }

    #[test]
    fn cache_metadata_helpers_are_stable_and_layered() {
        assert_eq!(
            direct_hsk_prompt_hash(),
            "sha256:55190968a85b2619aca2d48087d9a52e22c48a881aee959aa69cbe25904dc558"
        );
        assert_eq!(
            direct_hsk_validator_hash(),
            "sha256:1c23256323cef94f965c4d1c093392a3515f61249eba5d79cd73aa6689a4a1b1"
        );
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
            name_handling: HskNameHandling::Chinese,
            utterance: HskRepairUtterance {
                id: "id".to_owned(),
                kind: HskUtteranceKind::Dialogue,
                source_english: "Hello".to_owned(),
                rejected_chinese: None,
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
