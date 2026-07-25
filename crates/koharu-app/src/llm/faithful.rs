//! Full-page, local-only faithful manga translation.
//!
//! This is the narrow reusable boundary consumed after OCR: callers provide
//! already ordered regions with stable IDs plus page-session continuity data,
//! and receive one faithful Simplified Chinese string for every ID.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use koharu_llm::{GenerateOptions, Grammar, Language, ModelId};
use serde::{Deserialize, Serialize};

use super::{Model, State};

pub const FAITHFUL_TRANSLATION_MODEL: ModelId = ModelId::Qwen3_5_4b;
pub const FAITHFUL_PROMPT_REVISION: &str = "faithful-en-zh-v3";

const MAX_PRECEDING_CONTEXT: usize = 12;
const MAX_MALFORMED_OUTPUT_RETRIES: usize = 2;
const MIN_OUTPUT_TOKENS: usize = 512;
const MAX_OUTPUT_TOKENS: usize = 4096;
const OUTPUT_TOKENS_PER_SOURCE_CHAR: usize = 2;

const SYSTEM_PROMPT: &str = "\
You are a professional English-to-Simplified-Chinese manga translator. \
Translate every current-page region together into concise, natural Simplified Chinese. \
Preserve meaning, polarity and negation, numbers exactly as written, names, relationships, \
speaker consistency, tone, and reading order. Preceding context is reference context only; \
do not include it in the output. Whenever a protected English name appears, use its supplied \
Chinese form exactly. Keep every Arabic digit as the same ASCII digit; never spell a digit \
with a Chinese numeral (for example, source 6 must remain 6, not 六). Return only a JSON array \
in the same order as the regions. Every item \
must contain exactly the keys regionId and text. Include every requested regionId exactly once, \
add no IDs or fields, and add no commentary.";

/// llama.cpp's bundled JSON-array grammar, kept here as the constrained
/// decoding envelope.
///
/// Serde then enforces the exact two-field object schema, and exact region
/// coverage and order remain deterministic postconditions because stable IDs
/// are request data rather than model-chosen schema keys.
const FAITHFUL_JSON_GBNF: &str = r#"
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
pub struct FaithfulPageRequest {
    pub regions: Vec<FaithfulOcrRegion>,
    #[serde(default)]
    pub preceding_context: Vec<PrecedingPageContext>,
    #[serde(default)]
    pub protected_names: Vec<ProtectedName>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FaithfulOcrRegion {
    pub id: String,
    pub kind: FaithfulRegionKind,
    pub reading_order: u32,
    pub source_english: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FaithfulRegionKind {
    Dialogue,
    Caption,
    Thought,
    Sfx,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PrecedingPageContext {
    pub source_english: String,
    pub chinese: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProtectedName {
    pub source_english: String,
    pub chinese: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FaithfulTranslation {
    pub region_id: String,
    pub text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PromptPayload<'a> {
    prompt_revision: &'static str,
    target_language: &'static str,
    preceding_context: &'a [PrecedingPageContext],
    protected_names: &'a [ProtectedName],
    regions: &'a [FaithfulOcrRegion],
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_output_problem: Option<&'a str>,
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

trait Parser {
    fn parse(&self, output: &str) -> Result<Vec<FaithfulTranslation>>;
}

struct JsonParser;

impl Parser for JsonParser {
    fn parse(&self, output: &str) -> Result<Vec<FaithfulTranslation>> {
        serde_json::from_str(output).context("model output is not the required JSON array")
    }
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
                "faithful translation requires local model `{}`, but `{}` is loaded",
                FAITHFUL_TRANSLATION_MODEL,
                llm.id()
            ),
            State::ReadyProvider { .. } => {
                bail!("faithful translation is local-only; cloud and remote providers are disabled")
            }
            State::Loading { .. } => bail!("faithful translation model is still loading"),
            State::Failed { error, .. } => {
                bail!("faithful translation model failed to load: {error}")
            }
            State::Empty => {
                bail!("faithful translation model `{FAITHFUL_TRANSLATION_MODEL}` is not loaded")
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
                source: grammar.to_string(),
                root: "root".to_string(),
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
    /// Translate one page's ordered OCR regions together with the selected
    /// local Qwen3.5 4B model.
    ///
    /// The call is local-only, JSON-grammar constrained, cancellation-aware,
    /// and accepts output only when IDs have exact coverage and order.
    pub async fn translate_faithful_page(
        &self,
        request: &FaithfulPageRequest,
        cancel: &AtomicBool,
    ) -> Result<Vec<FaithfulTranslation>> {
        translate_with(self, &JsonParser, request, cancel).await
    }
}

async fn translate_with<G, P>(
    generator: &G,
    parser: &P,
    request: &FaithfulPageRequest,
    cancel: &AtomicBool,
) -> Result<Vec<FaithfulTranslation>>
where
    G: Generator + ?Sized,
    P: Parser + ?Sized,
{
    check_cancelled(cancel)?;
    validate_request(request)?;
    if request.regions.is_empty() {
        return Ok(Vec::new());
    }

    let total_attempts = MAX_MALFORMED_OUTPUT_RETRIES + 1;
    let mut previous_problem = None;
    for attempt in 0..total_attempts {
        check_cancelled(cancel)?;
        let user_prompt = build_prompt(request, previous_problem.as_deref())?;
        let raw = generator
            .generate(
                SYSTEM_PROMPT,
                &user_prompt,
                FAITHFUL_JSON_GBNF,
                output_token_budget(request),
                cancel,
            )
            .await?;
        check_cancelled(cancel)?;

        match parser.parse(&raw).and_then(|mut translations| {
            repair_ascii_number_preservation(request, &mut translations);
            validate_output(request, translations)
        }) {
            Ok(translations) => return Ok(translations),
            Err(error) => {
                previous_problem = Some(format!("{error:#}"));
                if attempt + 1 == total_attempts {
                    break;
                }
            }
        }
    }

    bail!(
        "faithful translation output remained invalid after {total_attempts} attempts: {}",
        previous_problem
            .as_deref()
            .unwrap_or("unknown output error")
    )
}

fn build_prompt(request: &FaithfulPageRequest, previous_problem: Option<&str>) -> Result<String> {
    let context_start = request
        .preceding_context
        .len()
        .saturating_sub(MAX_PRECEDING_CONTEXT);
    serde_json::to_string(&PromptPayload {
        prompt_revision: FAITHFUL_PROMPT_REVISION,
        target_language: "zh-CN",
        preceding_context: &request.preceding_context[context_start..],
        protected_names: &request.protected_names,
        regions: &request.regions,
        previous_output_problem: previous_problem,
    })
    .context("failed to serialize faithful translation prompt")
}

fn output_token_budget(request: &FaithfulPageRequest) -> usize {
    request
        .regions
        .iter()
        .map(|region| region.source_english.chars().count())
        .sum::<usize>()
        .saturating_mul(OUTPUT_TOKENS_PER_SOURCE_CHAR)
        .saturating_add(request.regions.len().saturating_mul(32))
        .clamp(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
}

fn validate_request(request: &FaithfulPageRequest) -> Result<()> {
    let mut ids = HashSet::with_capacity(request.regions.len());
    let mut previous_order = None;
    for region in &request.regions {
        if region.id.trim().is_empty() {
            bail!("faithful translation region ID must not be empty");
        }
        if !ids.insert(region.id.as_str()) {
            bail!("duplicate faithful translation request ID `{}`", region.id);
        }
        if region.source_english.trim().is_empty() {
            bail!("region `{}` has empty OCR text", region.id);
        }
        if previous_order.is_some_and(|order| region.reading_order <= order) {
            bail!("faithful translation regions must be in strictly increasing reading order");
        }
        previous_order = Some(region.reading_order);
    }

    let mut names = HashMap::with_capacity(request.protected_names.len());
    for name in &request.protected_names {
        let source = name.source_english.trim();
        let chinese = name.chinese.trim();
        if source.is_empty() || chinese.is_empty() {
            bail!("protected names require non-empty English and Chinese forms");
        }
        let key = source.to_ascii_lowercase();
        if let Some(previous) = names.insert(key, chinese)
            && previous != chinese
        {
            bail!("protected name `{source}` has conflicting Chinese forms");
        }
    }
    Ok(())
}

fn validate_output(
    request: &FaithfulPageRequest,
    translations: Vec<FaithfulTranslation>,
) -> Result<Vec<FaithfulTranslation>> {
    let expected_ids: HashSet<&str> = request
        .regions
        .iter()
        .map(|region| region.id.as_str())
        .collect();
    let mut seen = HashSet::with_capacity(translations.len());
    for translation in &translations {
        if !seen.insert(translation.region_id.as_str()) {
            bail!(
                "duplicate faithful translation output ID `{}`",
                translation.region_id
            );
        }
        if !expected_ids.contains(translation.region_id.as_str()) {
            bail!(
                "unexpected faithful translation output ID `{}`",
                translation.region_id
            );
        }
    }
    if let Some(missing) = request
        .regions
        .iter()
        .find(|region| !seen.contains(region.id.as_str()))
    {
        bail!("missing faithful translation output ID `{}`", missing.id);
    }
    for (index, (region, translation)) in request.regions.iter().zip(&translations).enumerate() {
        if translation.region_id != region.id {
            bail!(
                "faithful translation output order mismatch at index {index}: expected `{}`, got `{}`",
                region.id,
                translation.region_id
            );
        }
        if translation.text.trim().is_empty() {
            bail!("faithful translation for `{}` is empty", region.id);
        }
        validate_preservation(region, &translation.text, &request.protected_names)?;
    }
    Ok(translations)
}

fn repair_ascii_number_preservation(
    request: &FaithfulPageRequest,
    translations: &mut [FaithfulTranslation],
) {
    for translation in translations {
        let Some(region) = request
            .regions
            .iter()
            .find(|region| region.id == translation.region_id)
        else {
            continue;
        };
        normalize_full_width_ascii_digits(&mut translation.text);
        let expected = ascii_numbers(&region.source_english);
        if ascii_numbers(&translation.text) == expected {
            continue;
        }

        let actual = ascii_numbers(&translation.text);
        if actual.is_empty() {
            translation.text.push_str("（原文数字：");
            for number in expected {
                if !translation.text.ends_with('：') {
                    translation.text.push('、');
                }
                translation.text.push_str(number);
            }
            translation.text.push('）');
        }
    }
}

fn normalize_full_width_ascii_digits(text: &mut String) {
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

fn validate_preservation(
    region: &FaithfulOcrRegion,
    chinese: &str,
    protected_names: &[ProtectedName],
) -> Result<()> {
    let expected_numbers = ascii_numbers(&region.source_english);
    let actual_numbers = ascii_numbers(chinese);
    if actual_numbers != expected_numbers {
        bail!(
            "faithful translation for `{}` did not preserve numbers exactly: expected {:?}, got {:?}",
            region.id,
            expected_numbers,
            actual_numbers
        );
    }

    let source_lower = region.source_english.to_ascii_lowercase();
    for name in protected_names {
        if source_lower.contains(&name.source_english.to_ascii_lowercase())
            && !chinese.contains(&name.chinese)
        {
            bail!(
                "faithful translation for `{}` did not preserve protected name `{}` as `{}`",
                region.id,
                name.source_english,
                name.chinese
            );
        }
    }

    if has_english_negation(&source_lower) && !has_chinese_negation(chinese) {
        bail!(
            "faithful translation for `{}` did not preserve negation",
            region.id
        );
    }
    Ok(())
}

fn ascii_numbers(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut numbers = Vec::new();
    let mut start = None;
    for (index, byte) in bytes.iter().copied().enumerate() {
        if byte.is_ascii_digit() {
            start.get_or_insert(index);
        } else if let Some(number_start) = start.take() {
            numbers.push(&text[number_start..index]);
        }
    }
    if let Some(number_start) = start {
        numbers.push(&text[number_start..]);
    }
    numbers
}

fn has_english_negation(source_lower: &str) -> bool {
    source_lower.contains("n't")
        || source_lower
            .split(|character: char| !character.is_ascii_alphabetic())
            .any(|word| {
                matches!(
                    word,
                    "no" | "not" | "never" | "nothing" | "nobody" | "neither" | "without"
                )
            })
}

fn has_chinese_negation(text: &str) -> bool {
    text.chars().any(|character| {
        matches!(
            character,
            '不' | '没' | '無' | '无' | '別' | '别' | '未' | '非' | '莫' | '甭'
        )
    })
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
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use anyhow::anyhow;

    use super::*;

    struct FakeGenerator {
        outputs: Mutex<VecDeque<String>>,
        prompts: Mutex<Vec<String>>,
        grammars: Mutex<Vec<String>>,
        calls: AtomicUsize,
    }

    impl FakeGenerator {
        fn new(outputs: impl IntoIterator<Item = &'static str>) -> Self {
            Self {
                outputs: Mutex::new(outputs.into_iter().map(str::to_string).collect()),
                prompts: Mutex::new(Vec::new()),
                grammars: Mutex::new(Vec::new()),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Generator for FakeGenerator {
        async fn generate(
            &self,
            _system_prompt: &str,
            user_prompt: &str,
            grammar: &str,
            _max_tokens: usize,
            _cancel: &AtomicBool,
        ) -> Result<String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.prompts.lock().unwrap().push(user_prompt.to_string());
            self.grammars.lock().unwrap().push(grammar.to_string());
            self.outputs
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| anyhow!("fake generator exhausted"))
        }
    }

    enum FakeParseResult {
        Ok(Vec<FaithfulTranslation>),
        Err(&'static str),
    }

    struct FakeParser {
        results: Mutex<VecDeque<FakeParseResult>>,
        calls: AtomicUsize,
    }

    impl FakeParser {
        fn new(results: impl IntoIterator<Item = FakeParseResult>) -> Self {
            Self {
                results: Mutex::new(results.into_iter().collect()),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Parser for FakeParser {
        fn parse(&self, _output: &str) -> Result<Vec<FaithfulTranslation>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            match self.results.lock().unwrap().pop_front() {
                Some(FakeParseResult::Ok(value)) => Ok(value),
                Some(FakeParseResult::Err(message)) => bail!("{message}"),
                None => bail!("fake parser exhausted"),
            }
        }
    }

    fn region(id: &str, reading_order: u32, source: &str) -> FaithfulOcrRegion {
        FaithfulOcrRegion {
            id: id.to_string(),
            kind: FaithfulRegionKind::Dialogue,
            reading_order,
            source_english: source.to_string(),
        }
    }

    fn translation(id: &str, text: &str) -> FaithfulTranslation {
        FaithfulTranslation {
            region_id: id.to_string(),
            text: text.to_string(),
        }
    }

    fn request() -> FaithfulPageRequest {
        FaithfulPageRequest {
            regions: vec![
                region("page-r0", 0, "We have to leave now!"),
                region("page-r1", 1, "Are you ready?"),
            ],
            preceding_context: Vec::new(),
            protected_names: Vec::new(),
        }
    }

    #[tokio::test]
    async fn sends_all_regions_context_and_names_in_one_constrained_request() -> Result<()> {
        let generator = FakeGenerator::new(["ignored"]);
        let parser = FakeParser::new([FakeParseResult::Ok(vec![
            translation("page-r0", "我们得马上离开！"),
            translation("page-r1", "你准备好了吗？"),
        ])]);
        let mut input = request();
        input.protected_names.push(ProtectedName {
            source_english: "Alice".to_string(),
            chinese: "爱丽丝".to_string(),
        });
        input.preceding_context = (0..13)
            .map(|index| PrecedingPageContext {
                source_english: format!("context-{index}"),
                chinese: format!("上下文-{index}"),
            })
            .collect();

        let result = translate_with(&generator, &parser, &input, &AtomicBool::new(false)).await?;

        assert_eq!(result.len(), 2);
        assert_eq!(generator.calls.load(Ordering::Relaxed), 1);
        assert_eq!(parser.calls.load(Ordering::Relaxed), 1);
        let prompts = generator.prompts.lock().unwrap();
        let payload: serde_json::Value = serde_json::from_str(&prompts[0])?;
        assert_eq!(payload["promptRevision"], "faithful-en-zh-v3");
        assert_eq!(payload["regions"].as_array().unwrap().len(), 2);
        assert_eq!(payload["precedingContext"].as_array().unwrap().len(), 12);
        assert_eq!(payload["precedingContext"][0]["sourceEnglish"], "context-1");
        assert_eq!(payload["protectedNames"][0]["chinese"], "爱丽丝");
        assert!(payload.get("previousOutputProblem").is_none());
        assert!(generator.grammars.lock().unwrap()[0].contains("root ::= arr"));
        Ok(())
    }

    #[tokio::test]
    async fn retries_parser_failure_then_accepts_a_complete_result() -> Result<()> {
        let generator = FakeGenerator::new(["bad", "good"]);
        let parser = FakeParser::new([
            FakeParseResult::Err("invalid JSON"),
            FakeParseResult::Ok(vec![
                translation("page-r0", "我们得走了！"),
                translation("page-r1", "你准备好了吗？"),
            ]),
        ]);

        let result =
            translate_with(&generator, &parser, &request(), &AtomicBool::new(false)).await?;

        assert_eq!(result.len(), 2);
        assert_eq!(generator.calls.load(Ordering::Relaxed), 2);
        let prompts = generator.prompts.lock().unwrap();
        let retry: serde_json::Value = serde_json::from_str(&prompts[1])?;
        assert_eq!(retry["previousOutputProblem"], "invalid JSON");
        Ok(())
    }

    #[tokio::test]
    async fn malformed_output_retries_are_bounded() {
        let generator = FakeGenerator::new(["bad-1", "bad-2", "bad-3", "unused"]);
        let parser = FakeParser::new([
            FakeParseResult::Err("bad one"),
            FakeParseResult::Err("bad two"),
            FakeParseResult::Err("bad three"),
        ]);

        let error = translate_with(&generator, &parser, &request(), &AtomicBool::new(false))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("after 3 attempts"));
        assert_eq!(generator.calls.load(Ordering::Relaxed), 3);
        assert_eq!(parser.calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn validates_duplicate_extra_missing_and_out_of_order_ids() {
        let input = request();
        let cases = [
            (
                vec![translation("page-r0", "甲"), translation("page-r0", "乙")],
                "duplicate",
            ),
            (
                vec![
                    translation("page-r0", "甲"),
                    translation("page-r1", "乙"),
                    translation("extra", "丙"),
                ],
                "unexpected",
            ),
            (vec![translation("page-r0", "甲")], "missing"),
            (
                vec![translation("page-r1", "乙"), translation("page-r0", "甲")],
                "order mismatch",
            ),
        ];

        for (output, expected_error) in cases {
            let error = validate_output(&input, output).unwrap_err();
            assert!(
                error.to_string().contains(expected_error),
                "expected `{expected_error}`, got `{error}`"
            );
        }
    }

    #[test]
    fn strict_json_parser_rejects_extra_fields() {
        let error = JsonParser
            .parse(r#"[{"regionId":"page-r0","text":"好","note":"extra"}]"#)
            .unwrap_err();
        assert!(error.to_string().contains("required JSON array"));
    }

    #[test]
    fn checks_numbers_protected_names_and_negation() -> Result<()> {
        let input = FaithfulPageRequest {
            regions: vec![region("page-r0", 0, "Alice isn't taking the 12 tickets.")],
            preceding_context: Vec::new(),
            protected_names: vec![ProtectedName {
                source_english: "Alice".to_string(),
                chinese: "爱丽丝".to_string(),
            }],
        };

        validate_output(&input, vec![translation("page-r0", "爱丽丝不拿那12张票。")])?;
        for (text, expected_error) in [
            ("爱丽丝不拿那些票。", "number"),
            ("她不拿那12张票。", "protected name"),
            ("爱丽丝拿那12张票。", "negation"),
        ] {
            let error = validate_output(&input, vec![translation("page-r0", text)]).unwrap_err();
            assert!(error.to_string().contains(expected_error));
        }
        Ok(())
    }

    #[test]
    fn repairs_model_spelled_numbers_without_hiding_unknown_number_tokens() -> Result<()> {
        let input = FaithfulPageRequest {
            regions: vec![
                region("r0", 0, "THE 6 MAIN CLANS."),
                region("r1", 1, "LEVEL 12"),
                region("r2", 2, "ROOM 3"),
                region("r3", 3, "6 OF 6"),
            ],
            preceding_context: Vec::new(),
            protected_names: Vec::new(),
        };
        let mut output = vec![
            translation("r0", "六大宗门。"),
            translation("r1", "等级"),
            translation("r2", "房间３"),
            translation("r3", "六个中的六个"),
        ];

        repair_ascii_number_preservation(&input, &mut output);

        assert_eq!(output[0].text, "六大宗门。（原文数字：6）");
        assert_eq!(output[1].text, "等级（原文数字：12）");
        assert_eq!(output[2].text, "房间3");
        assert_eq!(output[3].text, "六个中的六个（原文数字：6、6）");
        validate_output(&input, output)?;
        Ok(())
    }

    #[test]
    fn number_validation_uses_exact_order_boundaries_and_multiplicity() {
        for (source, chinese) in [
            ("1 AND 10", "只有10"),
            ("6 AND 6", "只有6"),
            ("1 THEN 2", "先2后1"),
            ("6", "数字16"),
            ("10", "数字110"),
        ] {
            let error = validate_preservation(&region("r0", 0, source), chinese, &[]).unwrap_err();
            assert!(
                error.to_string().contains("numbers exactly"),
                "unexpected error for `{source}` -> `{chinese}`: {error}"
            );
        }
    }

    #[test]
    fn number_repair_never_rewrites_chinese_words_or_larger_number_tokens() {
        let input = FaithfulPageRequest {
            regions: vec![
                region("r0", 0, "VALUE 16"),
                region("r1", 1, "VALUE 6"),
                region("r2", 2, "VALUES 1 AND 10"),
                region("r3", 3, "LET'S GO TOGETHER AT 1"),
            ],
            preceding_context: Vec::new(),
            protected_names: Vec::new(),
        };
        let mut output = vec![
            translation("r0", "数值１６"),
            translation("r1", "数值十六"),
            translation("r2", "只有10"),
            translation("r3", "我们一起走"),
        ];

        repair_ascii_number_preservation(&input, &mut output);

        assert_eq!(output[0].text, "数值16");
        assert_eq!(output[1].text, "数值十六（原文数字：6）");
        assert_eq!(output[2].text, "只有10");
        assert_eq!(output[3].text, "我们一起走（原文数字：1）");
        validate_preservation(&input.regions[0], &output[0].text, &[]).unwrap();
        validate_preservation(&input.regions[1], &output[1].text, &[]).unwrap();
        assert!(validate_preservation(&input.regions[2], &output[2].text, &[]).is_err());
        validate_preservation(&input.regions[3], &output[3].text, &[]).unwrap();
    }

    #[tokio::test]
    async fn cancellation_stops_before_generation() {
        let generator = FakeGenerator::new(["unused"]);
        let parser = FakeParser::new([]);
        let cancel = AtomicBool::new(true);

        let error = translate_with(&generator, &parser, &request(), &cancel)
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "cancelled");
        assert_eq!(generator.calls.load(Ordering::Relaxed), 0);
    }

    struct CancellingGenerator {
        calls: AtomicUsize,
    }

    impl Generator for CancellingGenerator {
        async fn generate(
            &self,
            _system_prompt: &str,
            _user_prompt: &str,
            _grammar: &str,
            _max_tokens: usize,
            cancel: &AtomicBool,
        ) -> Result<String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            cancel.store(true, Ordering::Relaxed);
            Ok("[]".to_string())
        }
    }

    #[tokio::test]
    async fn cancellation_after_generation_does_not_retry_or_parse() {
        let generator = CancellingGenerator {
            calls: AtomicUsize::new(0),
        };
        let parser = FakeParser::new([]);
        let cancel = AtomicBool::new(false);

        let error = translate_with(&generator, &parser, &request(), &cancel)
            .await
            .unwrap_err();

        assert_eq!(error.to_string(), "cancelled");
        assert_eq!(generator.calls.load(Ordering::Relaxed), 1);
        assert_eq!(parser.calls.load(Ordering::Relaxed), 0);
    }

    #[cfg(target_os = "windows")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "opt-in smoke requiring the local Qwen3.5 4B GGUF and llama.cpp runtime"]
    async fn qwen3_5_4b_local_faithful_translation_smoke() -> Result<()> {
        use koharu_llm::Llm;
        use koharu_llm::safe::llama_backend::LlamaBackend;
        use koharu_runtime::{ComputePolicy, RuntimeManager, default_app_data_root};

        let model_path = PathBuf::from(
            r"C:\Users\Jiankai\Documents\hskify\.cache\model-benchmark\Qwen3.5-4B-Q4_K_M.gguf",
        );
        if !model_path.is_file() {
            bail!("smoke model is missing at `{}`", model_path.display());
        }

        let runtime = RuntimeManager::new(default_app_data_root(), ComputePolicy::PreferGpu)?;
        runtime.prepare().await?;
        koharu_llm::sys::initialize(&runtime)?;
        let backend = Arc::new(LlamaBackend::init()?);
        let llm = Llm::load_file(
            &runtime,
            FAITHFUL_TRANSLATION_MODEL,
            false,
            model_path,
            Arc::clone(&backend),
        )
        .await?;
        let model = Model::new(runtime, false, backend);
        *model.state.write().await = State::ReadyLocal(llm);

        let input = FaithfulPageRequest {
            regions: vec![
                region("smoke-r0", 0, "We have to leave now!"),
                region("smoke-r1", 1, "Are you ready?"),
                region("smoke-r2", 2, "Yes. Let's go!"),
            ],
            preceding_context: Vec::new(),
            protected_names: Vec::new(),
        };
        let output = model
            .translate_faithful_page(&input, &AtomicBool::new(false))
            .await?;

        assert_eq!(
            output
                .iter()
                .map(|item| item.region_id.as_str())
                .collect::<Vec<_>>(),
            vec!["smoke-r0", "smoke-r1", "smoke-r2"]
        );
        assert!(output.iter().all(|item| !item.text.trim().is_empty()));
        Ok(())
    }
}
