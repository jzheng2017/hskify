//! Chapter-page understanding with a multimodal Qwen3.5 projector.
//!
//! The browser pipeline has reliable pixel and OCR evidence before it asks a
//! language model to decide roles, continuations, and entity types.  This
//! module is the typed boundary for that hand-off.  The resident resource pack
//! ships the matching Qwen3.5 projector and the browser companion attaches it
//! to the already loaded translation model.  Callers still probe the pair at
//! the resource boundary; a missing or mismatched projector never gets
//! silently replaced with text-only visual evidence.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use image::DynamicImage;
use koharu_runtime::RuntimeManager;
use serde::{Deserialize, Serialize};

use crate::paddleocr_vl::{PaddleOcrVl, PaddleOcrVlGenerateOptions, QWEN3_5_IMAGE_MARKER};
use crate::safe::llama_backend::LlamaBackend;
use crate::safe::model::LlamaModel;

/// Maximum number of evidence regions in one page-understanding call.  The
/// limit keeps the numbered contract bounded and gives the resident model a
/// deterministic context budget.
pub const MAX_PAGE_REGIONS: usize = 64;
/// Maximum number of chapter context lines carried into a page window.
pub const MAX_PAGE_CONTEXT_LINES: usize = 8;
const PAGE_MAX_NEW_TOKENS: usize = 768;

/// This is the published Qwen3.5-4B multimodal projector file name.
pub const QWEN3_5_PROJECTOR_FILENAME: &str = "mmproj-BF16.gguf";
pub const QWEN3_5_PROJECTOR_REPOSITORY: &str = "unsloth/Qwen3.5-4B-GGUF";

/// Capability probe result used by setup/status code.  A missing projector is
/// a normal unavailable capability, not an instruction to fall back to
/// heuristic semantic classification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "status")]
pub enum PageUnderstandingCapability {
    Available {
        model_path: String,
        projector_path: String,
    },
    Unavailable {
        reason: String,
    },
}

impl PageUnderstandingCapability {
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

/// Probe an explicit model/projector pair without loading native model state.
///
/// Both files are required.  A text-only Qwen model, a projector without its
/// matching model, a directory, or a missing path all produce an unavailable
/// capability.  The caller can expose this reason to setup UI and keep the
/// deterministic OCR pipeline active without pretending it saw page pixels.
#[must_use]
pub fn probe_qwen_page_understanding(
    model_path: impl AsRef<Path>,
    projector_path: impl AsRef<Path>,
) -> PageUnderstandingCapability {
    let model_path = model_path.as_ref();
    let projector_path = projector_path.as_ref();
    if !model_path.is_file() {
        return PageUnderstandingCapability::Unavailable {
            reason: format!(
                "Qwen3.5 page model is unavailable: `{}`",
                model_path.display()
            ),
        };
    }
    if !projector_path.is_file() {
        return PageUnderstandingCapability::Unavailable {
            reason: format!(
                "Qwen3.5 vision projector is unavailable: `{}`",
                projector_path.display()
            ),
        };
    }

    PageUnderstandingCapability::Available {
        model_path: model_path.display().to_string(),
        projector_path: projector_path.display().to_string(),
    }
}

/// A normalized point in page coordinates.  Coordinates are normalized at the
/// browser boundary so an image can be resized without changing evidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PagePoint {
    pub x: f32,
    pub y: f32,
}

/// OCR/layout evidence for one independent region.  The model may correct a
/// transcript or link the region to its continuation.  The browser may send
/// either the complete page surface or a bounded evidence viewport; polygon
/// coordinates are always normalized to the attached pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PageRegionEvidence {
    pub id: String,
    pub source_english: String,
    #[serde(default)]
    pub transcript_hypotheses: Vec<String>,
    pub polygon: Vec<PagePoint>,
    pub confidence: f32,
    pub reading_order: usize,
    #[serde(default)]
    pub bubble_id: Option<String>,
    #[serde(default)]
    pub connected_region_ids: Vec<String>,
}

/// Page-level input. `image` is an immutable browser-captured surface or a
/// geometry-derived evidence viewport; all other fields are explicit machine
/// evidence included in the text part of the same multimodal request.
#[derive(Debug, Clone)]
pub struct PageUnderstandingRequest {
    pub image: Arc<DynamicImage>,
    pub regions: Vec<PageRegionEvidence>,
    pub preceding_chinese: Vec<String>,
    pub following_english: Vec<String>,
}

impl PageUnderstandingRequest {
    pub fn validate(&self) -> Result<()> {
        if self.image.width() == 0 || self.image.height() == 0 {
            bail!("page-understanding image has no pixels");
        }
        if self.regions.len() > MAX_PAGE_REGIONS {
            bail!(
                "page-understanding request contains {} regions; maximum is {MAX_PAGE_REGIONS}",
                self.regions.len()
            );
        }
        if self.preceding_chinese.len() > MAX_PAGE_CONTEXT_LINES
            || self.following_english.len() > MAX_PAGE_CONTEXT_LINES
        {
            bail!("page-understanding chapter context exceeds the bounded window");
        }

        let mut ids = HashSet::with_capacity(self.regions.len());
        let mut reading_orders = HashSet::with_capacity(self.regions.len());
        for region in &self.regions {
            if region.id.trim().is_empty() {
                bail!("page-understanding region id is empty");
            }
            if !ids.insert(region.id.as_str()) {
                bail!("page-understanding region id is duplicated: {}", region.id);
            }
            if !reading_orders.insert(region.reading_order) {
                bail!(
                    "page-understanding region reading order is duplicated: {}",
                    region.reading_order
                );
            }
            if region.source_english.trim().is_empty() {
                bail!("page-understanding region {} has empty OCR text", region.id);
            }
            if !region.confidence.is_finite() || !(0.0..=1.0).contains(&region.confidence) {
                bail!(
                    "page-understanding region {} has invalid OCR confidence",
                    region.id
                );
            }
            if region.polygon.len() < 3 {
                bail!(
                    "page-understanding region {} polygon needs at least three points",
                    region.id
                );
            }
            for point in &region.polygon {
                if !point.x.is_finite()
                    || !point.y.is_finite()
                    || !(0.0..=1.0).contains(&point.x)
                    || !(0.0..=1.0).contains(&point.y)
                {
                    bail!(
                        "page-understanding region {} contains an out-of-bounds polygon point",
                        region.id
                    );
                }
            }
            for hypothesis in &region.transcript_hypotheses {
                if hypothesis.trim().is_empty() {
                    bail!(
                        "page-understanding region {} contains an empty OCR hypothesis",
                        region.id
                    );
                }
            }
        }
        Ok(())
    }

    /// Render only bounded, validated evidence.  Pixels are carried by MTMD;
    /// this text accompanies the image and never attempts to describe pixels
    /// with a synthetic caption.
    pub fn render_evidence_prompt(&self) -> Result<String> {
        self.validate()?;
        let evidence = serde_json::json!({
            "page": {
                "width": self.image.width(),
                "height": self.image.height(),
            },
            "regions": &self.regions,
            "precedingChinese": &self.preceding_chinese,
            "followingEnglish": &self.following_english,
        });
        Ok(format!(
            "Use the attached comic page pixels plus this numbered OCR/layout evidence.\n{}\n\nReturn exactly one JSON object matching the requested schema. Do not include markdown fences or commentary.",
            serde_json::to_string(&evidence).context("serialize page evidence")?
        ))
    }
}

/// Semantic role selected by the page model. Geometry and language checks in
/// the browser daemon validate this value; they do not infer a role from
/// capitalization or lexical heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageRole {
    Story,
    Furniture,
    Unreadable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageRegionRole {
    Story,
    Sfx,
    Furniture,
    Artwork,
    Unreadable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PageRegionDecision {
    pub id: String,
    pub role: PageRegionRole,
    pub transcript: String,
    /// Final Chinese wording for source-preserving artwork/SFX. Story
    /// regions remain under the HSK authority after this adjudication call.
    #[serde(default)]
    pub translated_chinese: Option<String>,
    #[serde(default)]
    pub continuation_of: Option<String>,
    #[serde(default)]
    pub entity_spans: Vec<PageEntitySpan>,
    /// Optional visual typography evidence returned by the same page call.
    /// When omitted, the browser uses measured OCR appearance as its fallback.
    #[serde(default)]
    pub style: Option<PageStyleEvidence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageFontCategory {
    Sans,
    Serif,
    Handwritten,
    Display,
    Brush,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageTextAlignment {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PageWritingMode {
    HorizontalTb,
    VerticalRl,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PageStyleEvidence {
    pub font_category: PageFontCategory,
    pub weight: u16,
    #[serde(default)]
    pub italic_degrees: f32,
    pub writing_mode: PageWritingMode,
    pub alignment: PageTextAlignment,
    pub line_height: f32,
    #[serde(default)]
    pub letter_spacing_em: f32,
    #[serde(default)]
    pub shadow_color: Option<[u8; 3]>,
    #[serde(default)]
    pub shadow_x_ratio: f32,
    #[serde(default)]
    pub shadow_y_ratio: f32,
}

impl PageStyleEvidence {
    fn validate(&self, region_id: &str) -> Result<()> {
        if !(100..=900).contains(&self.weight) {
            bail!(
                "page-understanding style weight for region {region_id} must be from 100 through 900"
            );
        }
        for (name, value, min, max) in [
            ("italicDegrees", self.italic_degrees, -30.0, 30.0),
            ("lineHeight", self.line_height, 0.8, 2.2),
            ("letterSpacingEm", self.letter_spacing_em, -0.08, 0.3),
            ("shadowXRatio", self.shadow_x_ratio, -0.3, 0.3),
            ("shadowYRatio", self.shadow_y_ratio, -0.3, 0.3),
        ] {
            if !value.is_finite() || !(min..=max).contains(&value) {
                bail!(
                    "page-understanding style {name} for region {region_id} is outside its bounded range"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PageEntitySpan {
    pub source: String,
    pub entity_type: PageEntityType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageEntityType {
    Person,
    Place,
    Organization,
    Event,
    CoinedEntity,
    Relationship,
    Occupation,
    Rank,
    Title,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PageUnderstandingResult {
    pub page_role: PageRole,
    pub regions: Vec<PageRegionDecision>,
}

impl PageUnderstandingResult {
    pub fn parse_and_validate(raw: &str, request: &PageUnderstandingRequest) -> Result<Self> {
        request.validate()?;
        let result: Self = serde_json::from_str(raw.trim())
            .context("page-understanding model did not return the required JSON object")?;
        if result.regions.len() != request.regions.len() {
            bail!(
                "page-understanding returned {} regions for {} evidence regions",
                result.regions.len(),
                request.regions.len()
            );
        }
        let expected = request
            .regions
            .iter()
            .map(|region| region.id.as_str())
            .collect::<HashSet<_>>();
        let reading_order = request
            .regions
            .iter()
            .map(|region| (region.id.as_str(), region.reading_order))
            .collect::<std::collections::HashMap<_, _>>();
        let mut seen = HashSet::with_capacity(result.regions.len());
        for decision in &result.regions {
            if !expected.contains(decision.id.as_str()) || !seen.insert(decision.id.as_str()) {
                bail!(
                    "page-understanding returned an unknown or duplicate region id: {}",
                    decision.id
                );
            }
            if decision.transcript.trim().is_empty() {
                bail!(
                    "page-understanding returned an empty transcript for region {}",
                    decision.id
                );
            }
            if !source_language_transcript_is_valid(&decision.transcript) {
                bail!(
                    "page-understanding returned a non-source-language transcript for region {}",
                    decision.id
                );
            }
            let evidence = request
                .regions
                .iter()
                .find(|region| region.id == decision.id)
                .expect("validated region id must have matching evidence");
            if !transcript_is_bounded_correction(decision, evidence) {
                bail!(
                    "page-understanding transcript for region {} is not supported by its OCR evidence",
                    decision.id
                );
            }
            if let Some(chinese) = decision.translated_chinese.as_deref()
                && (chinese.trim().is_empty() || !contains_han(chinese))
            {
                bail!(
                    "page-understanding returned an invalid Chinese translation for region {}",
                    decision.id
                );
            }
            if decision.translated_chinese.is_some()
                && matches!(
                    decision.role,
                    PageRegionRole::Story | PageRegionRole::Furniture | PageRegionRole::Unreadable
                )
            {
                bail!(
                    "page-understanding returned artwork translation for non-artwork region {}",
                    decision.id
                );
            }
            if let Some(parent) = &decision.continuation_of {
                if parent == &decision.id || !expected.contains(parent.as_str()) {
                    bail!(
                        "page-understanding continuation for {} references an invalid region {}",
                        decision.id,
                        parent
                    );
                }
                if reading_order
                    .get(parent.as_str())
                    .copied()
                    .zip(reading_order.get(decision.id.as_str()).copied())
                    .is_none_or(|(parent_order, child_order)| parent_order >= child_order)
                {
                    bail!(
                        "page-understanding continuation for {} must point to an earlier reading-order region {}",
                        decision.id,
                        parent
                    );
                }
            }
            for span in &decision.entity_spans {
                if span.source.trim().is_empty() {
                    bail!(
                        "page-understanding entity span is empty for region {}",
                        decision.id
                    );
                }
                // Entity spans are offsets into the corrected transcript, not
                // necessarily the raw OCR string. A genuine OCR correction
                // (e.g. `Enriqne` -> `Enrique`) must remain eligible while
                // still being required to occur in the model's final source
                // language text.
                if !source_span_exists(&decision.transcript, &span.source) {
                    bail!(
                        "page-understanding entity span `{}` does not occur in region {}",
                        span.source,
                        decision.id
                    );
                }
            }
            if let Some(style) = &decision.style {
                style.validate(&decision.id)?;
            }
        }
        validate_page_role_consistency(result.page_role, &result.regions)?;
        let by_id = result
            .regions
            .iter()
            .map(|decision| (decision.id.as_str(), decision))
            .collect::<std::collections::HashMap<_, _>>();
        for decision in &result.regions {
            let mut current = decision.id.as_str();
            let mut chain = HashSet::new();
            while let Some(next) = by_id
                .get(current)
                .and_then(|entry| entry.continuation_of.as_deref())
            {
                if !chain.insert(current) {
                    bail!(
                        "page-understanding continuation graph contains a cycle at {}",
                        current
                    );
                }
                current = next;
            }
        }
        Ok(result)
    }
}

fn source_span_exists(source: &str, candidate: &str) -> bool {
    let source = source.trim();
    let candidate = candidate.trim();
    if source.is_empty() || candidate.is_empty() {
        return false;
    }
    let source_folded = source.to_ascii_lowercase();
    let candidate_folded = candidate.to_ascii_lowercase();
    source_folded
        .match_indices(&candidate_folded)
        .any(|(start, matched)| {
            let end = start + matched.len();
            let starts_at_boundary =
                start == 0 || !source_folded.as_bytes()[start - 1].is_ascii_alphanumeric();
            let ends_at_boundary = end == source_folded.len()
                || !source_folded.as_bytes()[end].is_ascii_alphanumeric();
            starts_at_boundary && ends_at_boundary
        })
}

/// The multimodal model may correct a recognition typo, but it must not be
/// allowed to replace a region with an unrelated sentence.  This gate is
/// deliberately evidence-based: it compares the returned transcript with the
/// source OCR and its alternate hypotheses, without a lexical allow-list or a
/// chapter-specific spelling table.  Low-confidence OCR is allowed a wider
/// correction budget, while high-confidence evidence must retain a meaningful
/// character signal.
fn transcript_is_bounded_correction(
    decision: &PageRegionDecision,
    evidence: &PageRegionEvidence,
) -> bool {
    let transcript = source_signature(&decision.transcript);
    if transcript.is_empty() {
        return false;
    }
    let mut candidates = Vec::with_capacity(1 + evidence.transcript_hypotheses.len());
    candidates.push(source_signature(&evidence.source_english));
    candidates.extend(
        evidence
            .transcript_hypotheses
            .iter()
            .map(|hypothesis| source_signature(hypothesis)),
    );
    candidates.retain(|candidate| !candidate.is_empty());
    let Some((best_similarity, best_shared, best_source_length)) = candidates
        .iter()
        .map(|candidate| {
            (
                transcript_similarity(&transcript, candidate),
                common_subsequence_length(&transcript, candidate),
                candidate.len(),
            )
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
    else {
        return false;
    };
    let confidence = evidence.confidence.clamp(0.0, 1.0);
    // The page model may repair an OCR typo, but it must not be able to turn
    // a one-character overlap into an unrelated sentence.  These floors are
    // evidence bounds, not a word list: the allowed correction is determined
    // by the measured OCR confidence and the amount of source signal that
    // survives in the model transcript.
    // Very short OCR snippets are common in small labels and clipped bubble
    // lines.  A corrector may supply a missing name or inflection around the
    // one surviving token (for example, “Wife” -> “Enrique's wife”), so the
    // global similarity floor must be lower for this bounded case.  The
    // source-coverage and length limits below still require the returned text
    // to contain the complete observed token and prevent unrelated prose.
    let short_source = best_source_length <= 4;
    let minimum_similarity = if short_source {
        0.30
    } else if confidence >= 0.80 {
        0.45
    } else if confidence >= 0.55 {
        0.32
    } else {
        0.22
    };
    let shortest_length = best_source_length.min(transcript.len());
    let minimum_shared = if best_source_length <= 4 {
        1
    } else {
        (shortest_length as f32 * 0.30).ceil().max(2.0) as usize
    };
    let minimum_source_coverage = if short_source {
        0.75
    } else if confidence >= 0.80 {
        0.45
    } else if confidence >= 0.55 {
        0.32
    } else {
        0.22
    };
    let minimum_transcript_coverage = if short_source {
        0.20
    } else {
        minimum_source_coverage * 0.65
    };
    let source_coverage = best_shared as f32 / best_source_length.max(1) as f32;
    let transcript_coverage = best_shared as f32 / transcript.len().max(1) as f32;
    let transcript_length_bounded = transcript.len() <= best_source_length.saturating_mul(2) + 8;
    best_similarity >= minimum_similarity
        && best_shared >= minimum_shared
        && source_coverage >= minimum_source_coverage
        && transcript_coverage >= minimum_transcript_coverage
        && transcript_length_bounded
}

fn source_signature(text: &str) -> Vec<char> {
    text.chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn transcript_similarity(left: &[char], right: &[char]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let mut previous = vec![0_u16; right.len() + 1];
    let mut best = 0_u16;
    for left_char in left {
        let mut current = vec![0_u16; right.len() + 1];
        for (index, right_char) in right.iter().enumerate() {
            if left_char == right_char {
                current[index + 1] = previous[index].saturating_add(1);
                best = best.max(current[index + 1]);
            }
        }
        previous = current;
    }
    let denominator = left.len().max(right.len()) as f32;
    f32::from(best) / denominator
}

fn common_subsequence_length(left: &[char], right: &[char]) -> usize {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    let mut previous = vec![0_u16; right.len() + 1];
    for left_char in left {
        let mut current = vec![0_u16; right.len() + 1];
        for (index, right_char) in right.iter().enumerate() {
            current[index + 1] = if left_char == right_char {
                previous[index].saturating_add(1)
            } else {
                current[index].max(previous[index + 1])
            };
        }
        previous = current;
    }
    usize::from(previous[right.len()])
}

fn validate_page_role_consistency(
    page_role: PageRole,
    regions: &[PageRegionDecision],
) -> Result<()> {
    let invalid = regions.iter().find(|region| match page_role {
        PageRole::Story => false,
        PageRole::Furniture => matches!(
            region.role,
            PageRegionRole::Story | PageRegionRole::Sfx | PageRegionRole::Unreadable
        ),
        PageRole::Unreadable => matches!(region.role, PageRegionRole::Story | PageRegionRole::Sfx),
    });
    if let Some(region) = invalid {
        bail!(
            "page-understanding page role {:?} conflicts with region {} role {:?}",
            page_role,
            region.id,
            region.role
        );
    }
    Ok(())
}

/// The page adjudicator is instructed to correct OCR while retaining the
/// source language.  Keep that boundary deterministic: a translated (Han)
/// transcript cannot be fed back into the English translator as if it were
/// source evidence.  Accented Latin text, numbers, punctuation, and symbols
/// remain valid because out-of-sample readers commonly contain names and
/// stylized notation outside ASCII.
fn source_language_transcript_is_valid(transcript: &str) -> bool {
    let mut has_source_letter = false;
    let mut has_source_digit = false;
    for character in transcript.chars() {
        if matches!(
            character as u32,
            0x3400..=0x4dbf
                | 0x4e00..=0x9fff
                | 0xf900..=0xfaff
                | 0x3040..=0x30ff
                | 0xac00..=0xd7af
        ) {
            return false;
        }
        if character.is_ascii_alphabetic()
            || matches!(character as u32, 0x00c0..=0x024f | 0x1e00..=0x1eff)
        {
            has_source_letter = true;
        }
        if character.is_ascii_digit() {
            has_source_digit = true;
        }
    }
    has_source_letter || has_source_digit
}

fn contains_han(text: &str) -> bool {
    text.chars().any(|character| {
        matches!(
            character as u32,
            0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff
        )
    })
}

/// Qwen3.5 page-understanding backend. It is constructed only with an
/// explicit model/projector pair; no browser path can accidentally call this
/// backend without the matching projector.
pub struct QwenPageUnderstanding {
    model: PaddleOcrVl,
}

impl QwenPageUnderstanding {
    pub fn load_from_paths(
        runtime: &RuntimeManager,
        model_path: impl AsRef<Path>,
        projector_path: impl AsRef<Path>,
        cpu: bool,
        backend: Arc<LlamaBackend>,
    ) -> Result<Self> {
        let capability = probe_qwen_page_understanding(&model_path, &projector_path);
        if let PageUnderstandingCapability::Unavailable { reason } = capability {
            bail!("page-understanding unavailable: {reason}");
        }
        let model = PaddleOcrVl::load_from_paths(
            runtime,
            model_path,
            projector_path,
            cpu,
            backend,
            QWEN3_5_IMAGE_MARKER,
        )
        .context("load Qwen3.5 page-understanding model/projector")?;
        Ok(Self { model })
    }

    /// Attach the page projector to the already resident translation model.
    ///
    /// This is the normal product constructor.  It shares the model weights
    /// instead of loading a second Qwen GGUF, while retaining the explicit
    /// projector capability check at the boundary.
    pub fn load_from_shared_model(
        runtime: &RuntimeManager,
        model: Arc<LlamaModel>,
        projector_path: impl AsRef<Path>,
        cpu: bool,
        backend: Arc<LlamaBackend>,
    ) -> Result<Self> {
        let projector_path = projector_path.as_ref();
        if !projector_path.is_file() {
            bail!(
                "page-understanding unavailable: Qwen3.5 vision projector is unavailable: `{}`",
                projector_path.display()
            );
        }
        let model = PaddleOcrVl::load_from_model(
            runtime,
            model,
            projector_path,
            cpu,
            backend,
            QWEN3_5_IMAGE_MARKER,
        )
        .context("attach Qwen3.5 page-understanding projector to resident model")?;
        Ok(Self { model })
    }

    pub fn analyze(
        &mut self,
        request: &PageUnderstandingRequest,
    ) -> Result<PageUnderstandingResult> {
        let prompt = request.render_evidence_prompt()?;
        let output = self
            .model
            .inference_with_prompt(
                &request.image,
                &format!(
                    "You are the chapter page adjudicator. Decide roles, corrected source-language transcripts, continuation links, typed entity spans, final Chinese wording for preserved artwork/SFX, and visual typography evidence for every evidence region. The transcript must remain in the source language (do not translate it). Return translatedChinese only for artwork or SFX whose original lettering remains pixel-identical; use null for story and furniture because story text is translated by the HSK authority after this call. Return style only when the pixels support it; otherwise use null. The JSON schema is: {{\"pageRole\":\"story|furniture|unreadable\",\"regions\":[{{\"id\":string,\"role\":\"story|sfx|furniture|artwork|unreadable\",\"transcript\":string,\"translatedChinese\":string|null,\"continuationOf\":string|null,\"entitySpans\":[{{\"source\":string,\"entityType\":\"person|place|organization|event|coined_entity|relationship|occupation|rank|title\"}}],\"style\":{{\"fontCategory\":\"sans|serif|handwritten|display|brush\",\"weight\":number,\"italicDegrees\":number,\"writingMode\":\"horizontal-tb|vertical-rl\",\"alignment\":\"left|center|right\",\"lineHeight\":number,\"letterSpacingEm\":number,\"shadowColor\":[number,number,number]|null,\"shadowXRatio\":number,\"shadowYRatio\":number}}|null}}]}}. {prompt}"
                ),
                &PaddleOcrVlGenerateOptions {
                    max_new_tokens: PAGE_MAX_NEW_TOKENS,
                    ..Default::default()
                },
            )
            .context("run Qwen3.5 page-understanding inference")?;
        PageUnderstandingResult::parse_and_validate(&output.text, request)
    }

    /// Prime the multimodal execution path without making startup depend on
    /// a model following the page JSON contract.  Warm-up is an execution
    /// probe, not a semantic decision: a tiny rendered response is enough to
    /// initialise MTMD, the projector, CUDA kernels, and the resident model
    /// allocator.  Real pages still go through [`Self::analyze`] and its
    /// fail-closed parser.
    pub fn warm_up(&mut self) -> Result<()> {
        let image = DynamicImage::new_rgb8(64, 64);
        self.model
            .inference_with_prompt(
                &image,
                "Warm up the page-understanding model. Return one short token.",
                &PaddleOcrVlGenerateOptions {
                    max_new_tokens: 1,
                    ..Default::default()
                },
            )
            .map(|_| ())
            .context("prime Qwen3.5 page-understanding inference")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn request() -> PageUnderstandingRequest {
        PageUnderstandingRequest {
            image: Arc::new(DynamicImage::ImageRgb8(RgbImage::new(100, 80))),
            regions: vec![PageRegionEvidence {
                id: "p1-r1".to_owned(),
                source_english: "Wife".to_owned(),
                transcript_hypotheses: vec!["Wife".to_owned()],
                polygon: vec![
                    PagePoint { x: 0.1, y: 0.1 },
                    PagePoint { x: 0.4, y: 0.1 },
                    PagePoint { x: 0.4, y: 0.2 },
                    PagePoint { x: 0.1, y: 0.2 },
                ],
                confidence: 0.9,
                reading_order: 0,
                bubble_id: Some("b1".to_owned()),
                connected_region_ids: Vec::new(),
            }],
            preceding_chinese: vec!["她回来了。".to_owned()],
            following_english: vec!["Wait for me.".to_owned()],
        }
    }

    #[test]
    fn evidence_prompt_contains_pixels_dimensions_and_layout_without_crop_probes() {
        let prompt = request().render_evidence_prompt().unwrap();
        assert!(prompt.contains("\"width\":100"));
        assert!(prompt.contains("\"sourceEnglish\":\"Wife\""));
        assert!(prompt.contains("\"polygon\""));
        assert!(!prompt.contains("upper"));
    }

    #[test]
    fn validation_rejects_duplicate_ids_and_out_of_bounds_points() {
        let mut duplicate = request();
        duplicate.regions.push(duplicate.regions[0].clone());
        assert!(duplicate.validate().is_err());

        let mut out_of_bounds = request();
        out_of_bounds.regions[0].polygon[0].x = 1.1;
        assert!(out_of_bounds.validate().is_err());
    }

    #[test]
    fn result_parser_requires_exact_region_coverage() {
        let request = request();
        let valid = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"story","transcript":"Wife","continuationOf":null,"entitySpans":[{"source":"Wife","entityType":"relationship"}]}]}"#;
        let parsed = PageUnderstandingResult::parse_and_validate(valid, &request).unwrap();
        assert_eq!(parsed.regions[0].role, PageRegionRole::Story);

        let missing = r#"{"pageRole":"story","regions":[]}"#;
        assert!(PageUnderstandingResult::parse_and_validate(missing, &request).is_err());

        let invalid_span = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"story","transcript":"Wife","continuationOf":null,"entitySpans":[{"source":"Enrique","entityType":"person"}]}]}"#;
        assert!(PageUnderstandingResult::parse_and_validate(invalid_span, &request).is_err());

        let corrected_transcript = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"story","transcript":"Enrique's wife","continuationOf":null,"entitySpans":[{"source":"Enrique","entityType":"person"}]}]}"#;
        assert!(
            PageUnderstandingResult::parse_and_validate(corrected_transcript, &request).is_ok()
        );

        let translated_transcript = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"story","transcript":"妻子","continuationOf":null,"entitySpans":[]}] }"#;
        assert!(
            PageUnderstandingResult::parse_and_validate(translated_transcript, &request).is_err()
        );
    }

    #[test]
    fn result_parser_rejects_unrelated_model_transcripts() {
        let request = request();
        let unrelated = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"story","transcript":"volcanic thunderstorm","continuationOf":null,"entitySpans":[]}] }"#;
        assert!(PageUnderstandingResult::parse_and_validate(unrelated, &request).is_err());
    }

    #[test]
    fn result_parser_rejects_tiny_overlap_and_unbounded_expansion() {
        let mut request = request();
        request.regions[0].source_english = "Wife".to_owned();
        request.regions[0].transcript_hypotheses = vec!["Wife".to_owned()];

        let tiny_overlap = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"story","transcript":"volcanic thunderstorm","continuationOf":null,"entitySpans":[]}] }"#;
        assert!(PageUnderstandingResult::parse_and_validate(tiny_overlap, &request).is_err());

        let unbounded = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"story","transcript":"Wife is the person who arrived at the academy after the storm and spoke for a long time","continuationOf":null,"entitySpans":[]}] }"#;
        assert!(PageUnderstandingResult::parse_and_validate(unbounded, &request).is_err());
    }

    #[test]
    fn page_role_cannot_claim_furniture_while_returning_story_regions() {
        let request = request();
        let contradictory = r#"{"pageRole":"furniture","regions":[{"id":"p1-r1","role":"story","transcript":"Wife","continuationOf":null,"entitySpans":[]}] }"#;
        assert!(PageUnderstandingResult::parse_and_validate(contradictory, &request).is_err());
    }

    #[test]
    fn result_parser_accepts_bounded_style_evidence_and_rejects_unbounded_style() {
        let request = request();
        let valid = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"story","transcript":"Wife","continuationOf":null,"entitySpans":[],"style":{"fontCategory":"handwritten","weight":700,"italicDegrees":5,"writingMode":"horizontal-tb","alignment":"center","lineHeight":1.1,"letterSpacingEm":0,"shadowColor":[0,0,0],"shadowXRatio":0.02,"shadowYRatio":0.02}}]}"#;
        assert!(PageUnderstandingResult::parse_and_validate(valid, &request).is_ok());

        let invalid = valid.replace("\"weight\":700", "\"weight\":999");
        assert!(PageUnderstandingResult::parse_and_validate(&invalid, &request).is_err());
    }

    #[test]
    fn artwork_translation_is_chinese_and_story_translation_stays_with_hsk_authority() {
        let request = request();
        let artwork = r#"{"pageRole":"story","regions":[{"id":"p1-r1","role":"artwork","transcript":"Wife","translatedChinese":"妻子","continuationOf":null,"entitySpans":[]}] }"#;
        assert!(PageUnderstandingResult::parse_and_validate(artwork, &request).is_ok());

        let story_translation = artwork.replace("\"artwork\"", "\"story\"");
        assert!(PageUnderstandingResult::parse_and_validate(&story_translation, &request).is_err());

        let latin_translation = artwork.replace("妻子", "wife");
        assert!(PageUnderstandingResult::parse_and_validate(&latin_translation, &request).is_err());
    }

    #[test]
    fn source_language_gate_accepts_latin_names_and_numeric_notation() {
        assert!(source_language_transcript_is_valid("Énrique 2"));
        assert!(source_language_transcript_is_valid("R2D2"));
        assert!(!source_language_transcript_is_valid("妻子"));
        assert!(!source_language_transcript_is_valid("…?!"));
    }

    #[test]
    fn capability_probe_fails_closed_when_projector_is_missing() {
        let capability = probe_qwen_page_understanding(
            "C:/does-not-exist/Qwen3.5-4B-Q4_K_M.gguf",
            "C:/does-not-exist/mmproj-BF16.gguf",
        );
        assert!(!capability.is_available());
        assert!(matches!(
            capability,
            PageUnderstandingCapability::Unavailable { .. }
        ));
    }

    #[test]
    fn qwen_marker_is_native_vision_placeholder() {
        assert_eq!(
            QWEN3_5_IMAGE_MARKER,
            "<|vision_start|><|image_pad|><|vision_end|>"
        );
    }
}
