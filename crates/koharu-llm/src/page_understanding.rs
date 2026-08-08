//! Chapter-page understanding with a multimodal Qwen3.5 projector.
//!
//! The browser pipeline has reliable pixel and OCR evidence before it asks a
//! language model to decide roles and continuations.  This
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
pub const MAX_PAGE_REGIONS: usize = 12;
const PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION: &str = r#"Classify each numbered comic OCR region by what its words do in the attached page, not by font, size, isolation, or position. OCR already established that every supplied transcript is readable. sourceEnglish is immutable and each output line describes only that same numbered source region.

Choose roles in this order:
1. story: speech/thought balloons; dialogue; messages between characters; or narration that tells fictional events, actions, state, time, or location. Narrative captions remain story when unboxed, hand-lettered, isolated, or drawn over art. Phrases such as "some time later", "years later", "it began", or a prose account of a war are story.
2. sfx: an audible sound effect, including impacts, footsteps, and short action sounds such as "BAM", "KICK", or "STEP" even when they are isolated or highly stylized.
3. furniture: text depicted on an object or interface, including signs, billboards, posters, clothing, books/scripture, products, device controls or interface feedback visually attached to a depicted device (even when the lettering sits beside its outline), credits, handles, branding, watermarks, and scanlation notices. Preserve it even when plot-relevant.
4. artwork: text that explicitly identifies this comic/page (series, chapter, or episode title/number; cover logo/byline) or addresses the audience (creator end card; thanks/follow/subscribe/update promotion). A line beginning with "Chapter" or "Episode" and a number is artwork. Mere top-of-page placement, isolation, capitals, or stylized lettering is not artwork; dialogue and action sounds remain story/sfx unless their words explicitly identify the publication or address its audience.
Readability is not a semantic role: never suppress a supplied OCR region because its role is uncertain. Choose the best of the four roles above.
The optional bubbleId is detector geometry, not a role: a null value can still be unboxed story narration or sfx, while text visibly printed on a device or object remains furniture.

continuationOf is null or the 1-based number of an earlier story region whose sentence or connected dialogue this story region continues. It is null for sfx, furniture, and artwork. Return exactly the requested numbered JSON lines, with one ASCII space between each number and object and no prose or Markdown. Object schema: {"role":"story|sfx|furniture|artwork","continuationOf":number|null}"#;

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

/// OCR/layout evidence for one independent region. OCR is the sole transcript
/// authority; the model may classify the region and link it to a
/// continuation. The browser may send either the
/// complete page surface or a bounded evidence viewport; polygon coordinates
/// are always normalized to the attached pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PageRegionEvidence {
    pub id: String,
    pub source_english: String,
    pub polygon: Vec<PagePoint>,
    pub confidence: f32,
    pub reading_order: usize,
    /// Present only when the comic-text detector assigned this region to a
    /// speech/thought bubble. Unboxed narration and SFX intentionally have no
    /// bubble id and still may be story content.
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
        });
        Ok(format!(
            "Use the attached comic page pixels plus this numbered OCR/layout evidence.\n{}\n\nReturn exactly {} lines in evidence order. Each line must be `<1-based number><SPACE><one JSON object>`. Do not include markdown fences or commentary.",
            serde_json::to_string(&evidence).context("serialize page evidence")?,
            self.regions.len(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageRegionRole {
    Story,
    Sfx,
    Furniture,
    Artwork,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PageRegionDecision {
    pub id: String,
    pub role: PageRegionRole,
    /// Exact OCR-owned source transcript copied from the request after the
    /// semantic record validates. The model never emits or revises this field.
    pub transcript: String,
    #[serde(default)]
    pub continuation_of: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageUnderstandingResult {
    pub regions: Vec<PageRegionDecision>,
    /// A malformed, missing, or unsupported record fails only that region.
    /// Callers preserve those source pixels and continue with valid siblings.
    pub failed_region_ids: Vec<String>,
}

impl PageUnderstandingResult {
    pub fn parse_and_validate(raw: &str, request: &PageUnderstandingRequest) -> Result<Self> {
        request.validate()?;
        let mut records = std::iter::repeat_with(|| None)
            .take(request.regions.len())
            .collect::<Vec<Option<RawPageRegionDecision>>>();
        let mut duplicated = HashSet::new();
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            let Some((number, json)) = line.trim().split_once(' ') else {
                continue;
            };
            let Ok(number) = number.trim().parse::<usize>() else {
                continue;
            };
            if number == 0 || number > records.len() {
                continue;
            }
            let index = number - 1;
            if records[index].is_some() {
                duplicated.insert(index);
                records[index] = None;
                continue;
            }
            records[index] = serde_json::from_str::<RawPageRegionDecision>(json.trim()).ok();
        }

        let mut regions = Vec::with_capacity(request.regions.len());
        let mut failed_region_ids = Vec::new();
        for (index, (evidence, raw_decision)) in request.regions.iter().zip(records).enumerate() {
            let decision = if duplicated.contains(&index) {
                None
            } else {
                raw_decision
                    .and_then(|decision| validate_record(index, decision, evidence, request).ok())
            };
            if let Some(decision) = decision {
                regions.push(decision);
            } else {
                failed_region_ids.push(evidence.id.clone());
            }
        }
        Ok(Self {
            regions,
            failed_region_ids,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawPageRegionDecision {
    role: PageRegionRole,
    continuation_of: Option<usize>,
}

fn validate_record(
    index: usize,
    raw: RawPageRegionDecision,
    evidence: &PageRegionEvidence,
    request: &PageUnderstandingRequest,
) -> Result<PageRegionDecision> {
    // A bad continuation link cannot invalidate otherwise usable language.
    let continuation_of = (raw.role == PageRegionRole::Story)
        .then_some(raw.continuation_of)
        .flatten()
        .and_then(|parent| {
            (parent > 0 && parent <= index).then(|| request.regions[parent - 1].id.clone())
        });
    Ok(PageRegionDecision {
        id: evidence.id.clone(),
        role: raw.role,
        transcript: evidence.source_english.clone(),
        continuation_of,
    })
}

fn generation_budget(request: &PageUnderstandingRequest) -> usize {
    (64 + request.regions.len() * 64).clamp(128, 768)
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
                &format!("{PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION}\n{prompt}"),
                &PaddleOcrVlGenerateOptions {
                    max_new_tokens: generation_budget(request),
                    repetition_penalty: 1.0,
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
        }
    }

    #[test]
    fn evidence_prompt_contains_pixels_dimensions_and_layout_without_crop_probes() {
        let prompt = request().render_evidence_prompt().unwrap();
        assert!(prompt.contains("\"width\":100"));
        assert!(prompt.contains("\"sourceEnglish\":\"Wife\""));
        assert!(prompt.contains("\"polygon\""));
        assert!(!prompt.contains("precedingChinese"));
        assert!(!prompt.contains("followingEnglish"));
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
    fn result_parser_maps_numbered_role_records() {
        let request = request();
        let valid = "1 {\"role\":\"story\",\"continuationOf\":null}";
        let parsed = PageUnderstandingResult::parse_and_validate(valid, &request).unwrap();
        assert!(parsed.failed_region_ids.is_empty());
        assert_eq!(parsed.regions[0].id, "p1-r1");
        assert_eq!(parsed.regions[0].transcript, "Wife");

        let obsolete_tab = "1\t{\"role\":\"story\",\"continuationOf\":null}";
        let parsed = PageUnderstandingResult::parse_and_validate(obsolete_tab, &request).unwrap();
        assert!(parsed.regions.is_empty());
        assert_eq!(parsed.failed_region_ids, vec!["p1-r1"]);

        let legacy_translation =
            "1 {\"role\":\"story\",\"faithfulChinese\":\"妻子\",\"continuationOf\":null}";
        let parsed =
            PageUnderstandingResult::parse_and_validate(legacy_translation, &request).unwrap();
        assert!(parsed.regions.is_empty());
        assert_eq!(parsed.failed_region_ids, vec!["p1-r1"]);
    }

    #[test]
    fn invalid_continuation_cannot_discard_a_valid_story_role() {
        let request = request();
        let output = "1 {\"role\":\"story\",\"continuationOf\":99}";

        let parsed = PageUnderstandingResult::parse_and_validate(output, &request).unwrap();

        assert!(parsed.failed_region_ids.is_empty());
        assert_eq!(parsed.regions.len(), 1);
        assert_eq!(parsed.regions[0].continuation_of, None);
    }

    #[test]
    fn semantic_prompt_distinguishes_embedded_furniture_from_story_text() {
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("signs, billboards"));
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("Preserve it even"));
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("series, chapter, or episode"));
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("creator end card"));
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("device controls"));
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("some time later"));
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("Mere top-of-page placement"));
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("bubbleId is detector geometry"));
        assert!(
            PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("Readability is not a semantic role")
        );
        assert!(PAGE_UNDERSTANDING_SYSTEM_INSTRUCTION.contains("four roles"));
    }

    #[test]
    fn semantic_output_cannot_override_the_ocr_transcript() {
        let request = request();
        let expanded =
            "1 {\"role\":\"story\",\"transcript\":\"Enrique's wife\",\"continuationOf\":null}";
        let parsed = PageUnderstandingResult::parse_and_validate(expanded, &request).unwrap();
        assert!(parsed.regions.is_empty());
        assert_eq!(parsed.failed_region_ids, vec!["p1-r1"]);
    }

    #[test]
    fn malformed_record_does_not_discard_valid_siblings() {
        let mut request = request();
        let mut second = request.regions[0].clone();
        second.id = "p1-r2".to_owned();
        second.source_english = "Wait for me.".to_owned();
        second.reading_order = 1;
        request.regions.push(second);
        let output = "1 {\"role\":\"story\",\"continuationOf\":null}\n2 {malformed";
        let parsed = PageUnderstandingResult::parse_and_validate(output, &request).unwrap();
        assert_eq!(parsed.regions.len(), 1);
        assert_eq!(parsed.regions[0].id, "p1-r1");
        assert_eq!(parsed.failed_region_ids, vec!["p1-r2"]);
    }

    #[test]
    fn continuation_uses_an_earlier_number_not_an_opaque_id() {
        let mut request = request();
        let mut second = request.regions[0].clone();
        second.id = "p1-r2".to_owned();
        second.source_english = "Wait for me.".to_owned();
        second.reading_order = 1;
        request.regions.push(second);
        let output = "1 {\"role\":\"story\",\"continuationOf\":null}\n2 {\"role\":\"story\",\"continuationOf\":1}";
        let parsed = PageUnderstandingResult::parse_and_validate(output, &request).unwrap();
        assert_eq!(parsed.regions[1].continuation_of.as_deref(), Some("p1-r1"));
    }

    #[test]
    fn non_story_roles_cannot_create_dialogue_continuations() {
        let request = request();
        let output = "1 {\"role\":\"furniture\",\"continuationOf\":1}";

        let parsed = PageUnderstandingResult::parse_and_validate(output, &request).unwrap();

        assert!(parsed.failed_region_ids.is_empty());
        assert_eq!(parsed.regions[0].continuation_of, None);
    }

    #[test]
    fn role_only_budget_does_not_scale_with_transcript_length() {
        let mut short = request();
        let short_budget = generation_budget(&short);
        short.regions[0].source_english = "x".repeat(10_000);

        assert_eq!(generation_budget(&short), short_budget);
        assert!(short_budget <= 768);
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
