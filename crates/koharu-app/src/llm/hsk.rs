//! HSK-targeted page rewrite through the already loaded local translation model.
//!
//! Vocabulary compliance is intentionally not decided here. The caller supplies
//! deterministic validator feedback, and `hsk-control` remains the authority
//! that accepts or rejects every candidate.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use koharu_llm::{GenerateOptions, Grammar, Language};
use serde::{Deserialize, Serialize};

use super::{FAITHFUL_TRANSLATION_MODEL, Model, State};

pub const HSK_REWRITE_PROMPT_REVISION: &str = "hsk-rewrite-en-zh-v1";

const MIN_OUTPUT_TOKENS: usize = 512;
const MAX_OUTPUT_TOKENS: usize = 4096;
const OUTPUT_TOKENS_PER_SOURCE_CHAR: usize = 2;

const SYSTEM_PROMPT: &str = "\
You rewrite faithful Simplified-Chinese manga dialogue using vocabulary at or below the requested \
HSK 2.0 level. Preserve meaning, polarity and every negation unit, numbers, protected names, \
relationships, speaker consistency, tone, and region order. Grammar is only targeted to the \
requested level; do not claim that grammar is deterministically strict. On correction requests, \
fix every supplied validator and preservation issue. When finalAttempt is true, prefer a short \
plain paraphrase. Return only a JSON array in the same order as the regions. Every item must \
contain exactly the keys regionId and text. Include every requested regionId exactly once, add no \
IDs or fields, and add no commentary.";

const JSON_ARRAY_GBNF: &str = r#"
root ::= arr
value ::= object | array | string | number | ("true" | "false" | "null") ws
arr ::= "[\n" ws (value (",\n" ws value)*)? "]"
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [1-9] [0-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"#;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskRewritePageRequest {
    pub requested_level: u8,
    /// Zero for the initial rewrite, then one or two for validator corrections.
    pub correction_attempt: u8,
    pub final_attempt: bool,
    pub regions: Vec<HskRewriteRegion>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskRewriteRegion {
    pub id: String,
    pub reading_order: u32,
    pub source_english: String,
    pub faithful_chinese: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_chinese: Option<String>,
    #[serde(default)]
    pub protected_names: Vec<String>,
    #[serde(default)]
    pub validator_feedback: Vec<HskValidatorFeedback>,
    #[serde(default)]
    pub preservation_feedback: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskValidatorFeedback {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub reason: String,
    pub suggested_words: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HskRewrite {
    pub region_id: String,
    pub text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PromptPayload<'a> {
    prompt_revision: &'static str,
    vocabulary_policy: String,
    grammar_policy: String,
    sentence_length_target_chinese_chars: usize,
    correction_attempt: u8,
    final_attempt: bool,
    regions: &'a [HskRewriteRegion],
}

trait Generator {
    async fn generate(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        grammar: &str,
        max_tokens: usize,
        cancel: &AtomicBool,
    ) -> Result<String>;
}

impl Generator for Model {
    async fn generate(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        grammar: &str,
        max_tokens: usize,
        cancel: &AtomicBool,
    ) -> Result<String> {
        let mut state = self.state.write().await;
        let llm = match &mut *state {
            State::ReadyLocal(llm) if llm.id() == FAITHFUL_TRANSLATION_MODEL => llm,
            State::ReadyLocal(llm) => bail!(
                "HSK rewrite requires local model `{}`, but `{}` is loaded",
                FAITHFUL_TRANSLATION_MODEL,
                llm.id()
            ),
            State::ReadyProvider { .. } => {
                bail!("HSK rewrite is local-only; cloud and remote providers are disabled")
            }
            State::Loading { .. } => bail!("HSK rewrite model is still loading"),
            State::Failed { error, .. } => bail!("HSK rewrite model failed to load: {error}"),
            State::Empty => {
                bail!("HSK rewrite model `{FAITHFUL_TRANSLATION_MODEL}` is not loaded")
            }
        };

        let options = GenerateOptions {
            max_tokens,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            grammar: Some(Grammar {
                source: grammar.to_owned(),
                root: "root".to_owned(),
            }),
            ..FAITHFUL_TRANSLATION_MODEL.default_generate_options()
        };
        llm.generate_constrained(
            user_prompt,
            &options,
            Language::ChineseSimplified,
            system_prompt,
            cancel,
        )
    }
}

impl Model {
    /// Perform one HSK rewrite or validator-correction request.
    ///
    /// This method deliberately performs exactly one model generation. The
    /// companion owns the initial-plus-two correction bound.
    pub async fn rewrite_hsk_page(
        &self,
        request: &HskRewritePageRequest,
        cancel: &AtomicBool,
    ) -> Result<Vec<HskRewrite>> {
        rewrite_with(self, request, cancel).await
    }
}

async fn rewrite_with<G>(
    generator: &G,
    request: &HskRewritePageRequest,
    cancel: &AtomicBool,
) -> Result<Vec<HskRewrite>>
where
    G: Generator + ?Sized,
{
    check_cancelled(cancel)?;
    validate_request(request)?;
    if request.regions.is_empty() {
        return Ok(Vec::new());
    }

    let prompt = build_prompt(request)?;
    let raw = generator
        .generate(
            SYSTEM_PROMPT,
            &prompt,
            JSON_ARRAY_GBNF,
            output_token_budget(request),
            cancel,
        )
        .await?;
    check_cancelled(cancel)?;
    let rewrites =
        serde_json::from_str(&raw).context("HSK rewrite output is not the required JSON array")?;
    validate_output(request, rewrites)
}

fn build_prompt(request: &HskRewritePageRequest) -> Result<String> {
    serde_json::to_string(&PromptPayload {
        prompt_revision: HSK_REWRITE_PROMPT_REVISION,
        vocabulary_policy: format!(
            "Vocabulary: restricted to cumulative HSK 1–{}, except explicit protected names",
            request.requested_level
        ),
        grammar_policy: format!(
            "Grammar: targeted to HSK {} (advisory, not deterministically strict)",
            request.requested_level
        ),
        sentence_length_target_chinese_chars: sentence_length_target(request.requested_level),
        correction_attempt: request.correction_attempt,
        final_attempt: request.final_attempt,
        regions: &request.regions,
    })
    .context("failed to serialize HSK rewrite prompt")
}

fn sentence_length_target(level: u8) -> usize {
    match level {
        1 => 12,
        2 => 16,
        3 => 20,
        4 => 24,
        5 => 30,
        _ => 36,
    }
}

fn output_token_budget(request: &HskRewritePageRequest) -> usize {
    request
        .regions
        .iter()
        .map(|region| {
            region
                .source_english
                .chars()
                .count()
                .saturating_add(region.faithful_chinese.chars().count())
        })
        .sum::<usize>()
        .saturating_mul(OUTPUT_TOKENS_PER_SOURCE_CHAR)
        .saturating_add(request.regions.len().saturating_mul(48))
        .clamp(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
}

fn validate_request(request: &HskRewritePageRequest) -> Result<()> {
    if !(1..=6).contains(&request.requested_level) {
        bail!("HSK rewrite level must be from 1 through 6");
    }
    if request.correction_attempt > 2 {
        bail!("HSK rewrite correction attempt exceeds the bounded maximum");
    }
    if request.final_attempt != (request.correction_attempt == 2) {
        bail!("HSK rewrite finalAttempt must be true exactly for correction attempt 2");
    }

    let mut ids = HashSet::with_capacity(request.regions.len());
    let mut previous_order = None;
    for region in &request.regions {
        if region.id.trim().is_empty() {
            bail!("HSK rewrite region ID must not be empty");
        }
        if !ids.insert(region.id.as_str()) {
            bail!("duplicate HSK rewrite request ID `{}`", region.id);
        }
        if region.source_english.trim().is_empty() || region.faithful_chinese.trim().is_empty() {
            bail!(
                "HSK rewrite region `{}` requires source and faithful text",
                region.id
            );
        }
        if previous_order.is_some_and(|order| region.reading_order <= order) {
            bail!("HSK rewrite regions must be in strictly increasing reading order");
        }
        previous_order = Some(region.reading_order);
        if request.correction_attempt > 0
            && region
                .current_chinese
                .as_deref()
                .is_none_or(|text| text.trim().is_empty())
        {
            bail!(
                "HSK correction region `{}` requires the rejected current text",
                region.id
            );
        }
        if region
            .protected_names
            .iter()
            .any(|name| name.trim().is_empty())
        {
            bail!("HSK rewrite protected names must not be empty");
        }
    }
    Ok(())
}

fn validate_output(
    request: &HskRewritePageRequest,
    rewrites: Vec<HskRewrite>,
) -> Result<Vec<HskRewrite>> {
    if rewrites.len() != request.regions.len() {
        bail!(
            "HSK rewrite returned {} regions; expected {}",
            rewrites.len(),
            request.regions.len()
        );
    }
    let mut seen = HashSet::with_capacity(rewrites.len());
    for (index, (region, rewrite)) in request.regions.iter().zip(&rewrites).enumerate() {
        if !seen.insert(rewrite.region_id.as_str()) {
            bail!("duplicate HSK rewrite output ID `{}`", rewrite.region_id);
        }
        if rewrite.region_id != region.id {
            bail!(
                "HSK rewrite output order mismatch at index {index}: expected `{}`, got `{}`",
                region.id,
                rewrite.region_id
            );
        }
        if rewrite.text.trim().is_empty() {
            bail!("HSK rewrite for `{}` is empty", region.id);
        }
    }
    Ok(rewrites)
}

fn check_cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use super::*;

    struct FakeGenerator {
        output: Mutex<Option<String>>,
        prompt: Mutex<Option<String>>,
        calls: AtomicUsize,
    }

    impl FakeGenerator {
        fn new(output: &str) -> Self {
            Self {
                output: Mutex::new(Some(output.to_owned())),
                prompt: Mutex::new(None),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Generator for FakeGenerator {
        async fn generate(
            &self,
            system_prompt: &str,
            user_prompt: &str,
            grammar: &str,
            _max_tokens: usize,
            _cancel: &AtomicBool,
        ) -> Result<String> {
            assert!(system_prompt.contains("Grammar is only targeted"));
            assert!(grammar.contains("root ::= arr"));
            self.calls.fetch_add(1, Ordering::Relaxed);
            *self.prompt.lock().unwrap() = Some(user_prompt.to_owned());
            self.output
                .lock()
                .unwrap()
                .take()
                .context("fake output exhausted")
        }
    }

    fn request(correction_attempt: u8) -> HskRewritePageRequest {
        HskRewritePageRequest {
            requested_level: 2,
            correction_attempt,
            final_attempt: correction_attempt == 2,
            regions: vec![HskRewriteRegion {
                id: "region-1".to_owned(),
                reading_order: 0,
                source_english: "We must not leave 2 people.".to_owned(),
                faithful_chinese: "我们不能离开2个人。".to_owned(),
                current_chinese: (correction_attempt > 0).then(|| "我们立即离开。".to_owned()),
                protected_names: Vec::new(),
                validator_feedback: vec![HskValidatorFeedback {
                    text: "立即".to_owned(),
                    start_char: 2,
                    end_char: 4,
                    reason: "above-selected-hsk-level:5".to_owned(),
                    suggested_words: vec!["马上".to_owned()],
                }],
                preservation_feedback: vec![
                    "numbers changed: expected [\"2\"], actual []".to_owned(),
                    "negation markers changed: expected [\"不\"], actual []".to_owned(),
                ],
            }],
        }
    }

    #[tokio::test]
    async fn sends_exact_feedback_and_accurate_grammar_wording() -> Result<()> {
        let generator =
            FakeGenerator::new(r#"[{"regionId":"region-1","text":"我们不能离开2个人。"}]"#);
        let result = rewrite_with(&generator, &request(1), &AtomicBool::new(false)).await?;

        assert_eq!(result[0].region_id, "region-1");
        assert_eq!(generator.calls.load(Ordering::Relaxed), 1);
        let prompt = generator.prompt.lock().unwrap().clone().unwrap();
        let payload: serde_json::Value = serde_json::from_str(&prompt)?;
        assert_eq!(
            payload["grammarPolicy"],
            "Grammar: targeted to HSK 2 (advisory, not deterministically strict)"
        );
        assert_eq!(
            payload["regions"][0]["validatorFeedback"][0]["text"],
            "立即"
        );
        assert_eq!(payload["regions"][0]["currentChinese"], "我们立即离开。");
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_happens_before_generation() {
        let generator = FakeGenerator::new("unused");
        let error = rewrite_with(&generator, &request(0), &AtomicBool::new(true))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(generator.calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn rejects_out_of_order_or_unknown_output_ids() {
        let generator =
            FakeGenerator::new(r#"[{"regionId":"unknown","text":"我们不能离开2个人。"}]"#);
        let error = rewrite_with(&generator, &request(0), &AtomicBool::new(false))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("order mismatch"));
    }
}
