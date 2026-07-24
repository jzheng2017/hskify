//! Thin browser adapter over Koharu's production cleaning pipeline.
//!
//! Gate 3 deliberately stops after detection, masks, OCR, and inpainting.
//! The frozen result shape still contains translation fields, so until Gate 4
//! they carry the OCR source text as an identity fallback and vocabulary is
//! explicitly marked non-strict.

use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use camino::Utf8PathBuf;
use image::{GrayImage, ImageFormat};
use koharu_app::pipeline::{self, Artifact, PipelineSpec, Scope};
use koharu_app::{App, AppConfig, PipelineRunOptions, ProjectSession};
use koharu_core::{
    ImageData, ImageRole, MaskRole, Node, NodeId, NodeKind, Op, Page, PageId, PipelineStep,
    ReadingOrder, TextData, TextDirection, Transform,
};
use koharu_runtime::{ComputePolicy, RuntimeManager};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::contracts::{
    BrowserCacheStatus, BrowserJobRequest, BrowserJobStage, BrowserRegion, BrowserTextLayout,
    BrowserTextStyle, BrowserWarning, BrowserWarningCode, CleanImageMimeType, FontCategory, Point,
    ReadingDirection, RegionKind, TextAlignment, VocabularyStatus, WritingMode,
};
use crate::crypto::sha256_hex;

const CACHE_MARKER_VERSION: u8 = 1;
const PIPELINE_FINGERPRINT: &str = "gate3-v1:pp-doclayout-v3+comic-text-detector-seg+speech-bubble-segmentation+paddle-ocr-vl-1.6+lama-manga";
const LOW_OCR_CONFIDENCE: f32 = 0.60;

pub(crate) type CleaningProgressSink = Arc<dyn Fn(CleaningProgress) + Send + Sync>;

#[derive(Debug, Clone)]
pub(crate) struct CleaningInput {
    pub source_bytes: Arc<[u8]>,
    pub request: BrowserJobRequest,
}

#[derive(Debug, Clone)]
pub(crate) struct CleaningOutput {
    pub clean_image: Vec<u8>,
    pub clean_image_mime_type: CleanImageMimeType,
    pub regions: Vec<BrowserRegion>,
    pub warnings: Vec<BrowserWarning>,
    pub cache: BrowserCacheStatus,
}

#[derive(Debug, Clone)]
pub(crate) struct CleaningProgress {
    pub stage: BrowserJobStage,
    pub overall_progress: Option<f32>,
    pub current: Option<u32>,
    pub total: Option<u32>,
    pub message: String,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub(crate) struct CleaningError {
    pub code: &'static str,
    pub message: String,
}

impl CleaningError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    fn pipeline(error: anyhow::Error) -> Self {
        Self::new(
            "PIPELINE_FAILED",
            format!("Koharu cleaning pipeline failed: {error:#}"),
        )
    }
}

#[async_trait]
pub(crate) trait CleaningPipeline: Send + Sync {
    async fn run(
        &self,
        input: CleaningInput,
        cancel: Arc<AtomicBool>,
        progress: CleaningProgressSink,
    ) -> std::result::Result<CleaningOutput, CleaningError>;
}

pub(crate) struct KoharuPipeline {
    cache_root: PathBuf,
    app: OnceCell<Arc<App>>,
}

impl KoharuPipeline {
    pub(crate) fn new(cache_root: PathBuf) -> Self {
        Self {
            cache_root,
            app: OnceCell::new(),
        }
    }

    async fn app(&self) -> Result<&Arc<App>> {
        self.app
            .get_or_try_init(|| async {
                let data_root = self.cache_root.join("koharu-data");
                std::fs::create_dir_all(&data_root)
                    .with_context(|| format!("create Koharu data root {}", data_root.display()))?;
                let runtime = Arc::new(RuntimeManager::new(&data_root, ComputePolicy::PreferGpu)?);
                runtime.prepare().await.context("prepare Koharu runtime")?;
                let mut config = AppConfig::default();
                config.data.path = utf8_path(data_root)?;
                let app = App::new(config, runtime, false, env!("CARGO_PKG_VERSION"))
                    .context("initialize Koharu application services")?;
                Ok::<_, anyhow::Error>(Arc::new(app))
            })
            .await
    }

    fn project_path(&self, source_sha256: &str) -> PathBuf {
        self.cache_root
            .join("cleaning-projects-v1")
            .join(format!("{source_sha256}.khrproj"))
    }
}

#[async_trait]
impl CleaningPipeline for KoharuPipeline {
    async fn run(
        &self,
        input: CleaningInput,
        cancel: Arc<AtomicBool>,
        progress: CleaningProgressSink,
    ) -> std::result::Result<CleaningOutput, CleaningError> {
        if cancel.load(Ordering::Acquire) {
            return Err(CleaningError::new("CANCELLED", "Cleaning was cancelled."));
        }

        let project_path = self.project_path(&input.request.source_sha256);
        let (session, page_id) =
            open_or_create_project(&project_path, &input).map_err(CleaningError::pipeline)?;

        if cache_marker_matches(&project_path, &input.request)
            && cached_artifacts_ready(&session, page_id)
        {
            progress(CleaningProgress {
                stage: BrowserJobStage::Packaging,
                overall_progress: Some(0.98),
                current: None,
                total: None,
                message: "Packaging cached Koharu cleaning result".to_owned(),
            });
            return extract_output(&session, page_id, &input.request, true);
        }

        let app = self.app().await.map_err(CleaningError::pipeline)?;
        let config = app.config.load();
        let steps = cleaning_steps(&config);
        drop(config);

        let progress_bridge = {
            let progress = progress.clone();
            Arc::new(move |tick: pipeline::ProgressTick| {
                let Some((stage, message)) = tick.step.and_then(progress_stage) else {
                    return;
                };
                progress(CleaningProgress {
                    stage,
                    overall_progress: Some(f32::from(tick.overall_percent) / 100.0),
                    current: u32::try_from(tick.step_index + 1).ok(),
                    total: u32::try_from(tick.total_steps).ok(),
                    message: message.to_owned(),
                });
            }) as pipeline::ProgressSink
        };
        let warnings = Arc::new(Mutex::new(Vec::<String>::new()));
        let warning_bridge = {
            let warnings = warnings.clone();
            Arc::new(move |tick: pipeline::WarningTick| {
                warnings
                    .lock()
                    .expect("pipeline warning lock poisoned")
                    .push(format!("{}: {}", tick.step_id, tick.message));
            }) as pipeline::WarningSink
        };
        let spec = PipelineSpec {
            scope: Scope::Pages(vec![page_id]),
            steps,
            options: PipelineRunOptions {
                target_language: None,
                system_prompt: None,
                default_font: None,
                text_node_ids: None,
                region: None,
                reading_order: Some(reading_order(input.request.settings.reading_direction)),
            },
        };

        let outcome = pipeline::run(
            session.clone(),
            app.registry.clone(),
            app.runtime.clone(),
            app.cpu_only(),
            app.llm.clone(),
            app.renderer.clone(),
            spec,
            cancel.clone(),
            Some(progress_bridge),
            Some(warning_bridge),
        )
        .await
        .map_err(CleaningError::pipeline)?;
        if cancel.load(Ordering::Acquire) {
            return Err(CleaningError::new("CANCELLED", "Cleaning was cancelled."));
        }
        if outcome.warning_count > 0 {
            let details = warnings
                .lock()
                .expect("pipeline warning lock poisoned")
                .join("; ");
            return Err(CleaningError::new(
                "PIPELINE_STEP_FAILED",
                if details.is_empty() {
                    "One or more Koharu cleaning stages failed.".to_owned()
                } else {
                    format!("One or more Koharu cleaning stages failed: {details}")
                },
            ));
        }

        progress(CleaningProgress {
            stage: BrowserJobStage::Packaging,
            overall_progress: Some(0.98),
            current: None,
            total: None,
            message: "Packaging Koharu cleaning result".to_owned(),
        });
        let output = extract_output(&session, page_id, &input.request, false)?;
        session.compact().map_err(CleaningError::pipeline)?;
        write_cache_marker(&project_path, &input.request).map_err(CleaningError::pipeline)?;
        Ok(output)
    }
}

fn progress_stage(step: PipelineStep) -> Option<(BrowserJobStage, &'static str)> {
    match step {
        PipelineStep::Detect => Some((BrowserJobStage::Detecting, "Detecting text and masks")),
        PipelineStep::Ocr => Some((BrowserJobStage::Ocr, "Reading English text")),
        PipelineStep::Inpaint => Some((BrowserJobStage::Inpainting, "Cleaning source lettering")),
        PipelineStep::LlmGenerate | PipelineStep::Render => None,
    }
}

fn cleaning_steps(config: &AppConfig) -> Vec<String> {
    let pipeline = &config.pipeline;
    [
        pipeline.detector.as_str(),
        pipeline.segmenter.as_str(),
        pipeline.bubble_segmenter.as_str(),
        pipeline.ocr.as_str(),
        pipeline.inpainter.as_str(),
    ]
    .into_iter()
    .filter(|step| !step.trim().is_empty())
    .map(ToOwned::to_owned)
    .collect()
}

fn reading_order(direction: ReadingDirection) -> ReadingOrder {
    match direction {
        ReadingDirection::Ltr => ReadingOrder::Ltr,
        ReadingDirection::Auto | ReadingDirection::Rtl => ReadingOrder::Rtl,
    }
}

fn utf8_path(path: PathBuf) -> Result<Utf8PathBuf> {
    Utf8PathBuf::from_path_buf(path)
        .map_err(|path| anyhow!("browser cache path is not valid UTF-8: {}", path.display()))
}

fn open_or_create_project(
    project_path: &Path,
    input: &CleaningInput,
) -> Result<(Arc<ProjectSession>, PageId)> {
    if let Some(parent) = project_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create browser project cache {}", parent.display()))?;
    }
    let project_path = utf8_path(project_path.to_path_buf())?;
    if project_path.is_dir() {
        let session = ProjectSession::open(&project_path)
            .with_context(|| format!("open cached browser project {project_path}"))?;
        let page_id = validate_cached_source(&session, input)?;
        return Ok((session, page_id));
    }

    let session = ProjectSession::create(&project_path, "Browser cleaning cache")
        .with_context(|| format!("create cached browser project {project_path}"))?;
    let source_blob = session
        .blobs
        .put_bytes(input.source_bytes.as_ref())
        .context("store browser source image in Koharu blob storage")?;
    let mut page = Page::new(
        &input.request.client_image_id,
        input.request.natural_width,
        input.request.natural_height,
    );
    let page_id = page.id;
    let source_id = NodeId::new();
    page.nodes.insert(
        source_id,
        Node {
            id: source_id,
            transform: Transform::default(),
            visible: true,
            kind: NodeKind::Image(ImageData {
                role: ImageRole::Source,
                blob: source_blob,
                opacity: 1.0,
                natural_width: input.request.natural_width,
                natural_height: input.request.natural_height,
                name: Some(input.request.client_image_id.clone()),
            }),
        },
    );
    session
        .apply(Op::AddPage { page, at: 0 })
        .context("import browser source page into Koharu")?;
    Ok((session, page_id))
}

fn validate_cached_source(session: &ProjectSession, input: &CleaningInput) -> Result<PageId> {
    let scene = session.scene.read();
    if scene.pages.len() != 1 {
        bail!("cached browser project must contain exactly one page");
    }
    let (page_id, page) = scene.pages.iter().next().expect("one cached page");
    if page.width != input.request.natural_width || page.height != input.request.natural_height {
        bail!("cached browser project dimensions do not match the upload");
    }
    let source = page
        .nodes
        .values()
        .find_map(|node| match &node.kind {
            NodeKind::Image(image) if image.role == ImageRole::Source => Some(image),
            _ => None,
        })
        .context("cached browser project has no source image")?;
    let source_bytes = session.blobs.get_bytes(&source.blob)?;
    if !sha256_hex(&source_bytes).eq_ignore_ascii_case(&input.request.source_sha256) {
        bail!("cached browser project source hash does not match the upload");
    }
    Ok(*page_id)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CacheMarker {
    version: u8,
    pipeline_fingerprint: String,
    source_sha256: String,
    source_width: u32,
    source_height: u32,
}

fn cache_marker_path(project_path: &Path) -> PathBuf {
    project_path.join("browser-cleaning-v1.json")
}

fn cache_marker_matches(project_path: &Path, request: &BrowserJobRequest) -> bool {
    let Ok(bytes) = std::fs::read(cache_marker_path(project_path)) else {
        return false;
    };
    let Ok(marker) = serde_json::from_slice::<CacheMarker>(&bytes) else {
        return false;
    };
    marker.version == CACHE_MARKER_VERSION
        && marker.pipeline_fingerprint == PIPELINE_FINGERPRINT
        && marker
            .source_sha256
            .eq_ignore_ascii_case(&request.source_sha256)
        && marker.source_width == request.natural_width
        && marker.source_height == request.natural_height
}

fn write_cache_marker(project_path: &Path, request: &BrowserJobRequest) -> Result<()> {
    let marker = CacheMarker {
        version: CACHE_MARKER_VERSION,
        pipeline_fingerprint: PIPELINE_FINGERPRINT.to_owned(),
        source_sha256: request.source_sha256.clone(),
        source_width: request.natural_width,
        source_height: request.natural_height,
    };
    let bytes = serde_json::to_vec(&marker)?;
    std::fs::write(cache_marker_path(project_path), bytes)
        .with_context(|| format!("write cleaning cache marker {}", project_path.display()))
}

fn cached_artifacts_ready(session: &ProjectSession, page_id: PageId) -> bool {
    let scene = session.scene.read();
    let Some(page) = scene.pages.get(&page_id) else {
        return false;
    };
    [
        Artifact::TextBoxes,
        Artifact::OcrText,
        Artifact::SegmentMask,
        Artifact::BubbleMask,
        Artifact::Inpainted,
    ]
    .into_iter()
    .all(|artifact| artifact.ready(page))
}

fn extract_output(
    session: &ProjectSession,
    page_id: PageId,
    request: &BrowserJobRequest,
    cache_hit: bool,
) -> std::result::Result<CleaningOutput, CleaningError> {
    let scene = session.scene_snapshot();
    let page = scene
        .pages
        .get(&page_id)
        .ok_or_else(|| CleaningError::new("PIPELINE_FAILED", "Koharu page disappeared."))?;
    let text_nodes = page
        .nodes
        .values()
        .filter_map(|node| match &node.kind {
            NodeKind::Text(text) => Some((node.transform, text)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if text_nodes.is_empty() {
        return Err(CleaningError::new(
            "NO_TEXT_DETECTED",
            "Koharu did not detect any text regions.",
        ));
    }
    if !Artifact::OcrText.ready(page) {
        return Err(CleaningError::new(
            "OCR_INCOMPLETE",
            "Koharu did not return English OCR for every detected region.",
        ));
    }
    if !Artifact::SegmentMask.ready(page) || !Artifact::BubbleMask.ready(page) {
        return Err(CleaningError::new(
            "MASK_GENERATION_FAILED",
            "Koharu did not produce the required text and bubble masks.",
        ));
    }

    let bubble_mask = page
        .nodes
        .values()
        .find_map(|node| match &node.kind {
            NodeKind::Mask(mask) if mask.role == MaskRole::Bubble => Some(&mask.blob),
            _ => None,
        })
        .ok_or_else(|| {
            CleaningError::new("MASK_GENERATION_FAILED", "Koharu bubble mask is missing.")
        })
        .and_then(|blob| {
            session.blobs.load_image(blob).map_err(|error| {
                CleaningError::new(
                    "MASK_GENERATION_FAILED",
                    format!("Koharu bubble mask could not be loaded: {error:#}"),
                )
            })
        })?
        .to_luma8();
    let bubble_polygons = bubble_polygons(&bubble_mask, page.width, page.height);

    let mut regions = Vec::with_capacity(text_nodes.len());
    let mut warnings = Vec::new();
    let mut id_counts = HashMap::<String, usize>::new();
    for (reading_order, (transform, text)) in text_nodes.into_iter().enumerate() {
        let source_english = text
            .text
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                CleaningError::new(
                    "OCR_INCOMPLETE",
                    "Koharu returned an empty OCR text region.",
                )
            })?
            .to_owned();
        let text_polygon = text_polygon(transform, text, page.width, page.height);
        let base_id = stable_region_id(&request.source_sha256, &text_polygon);
        let count = id_counts.entry(base_id.clone()).or_default();
        *count += 1;
        let id = if *count == 1 {
            base_id
        } else {
            format!("{base_id}-{}", *count)
        };
        let bubble_polygon = bubble_label_for(&bubble_mask, transform)
            .and_then(|label| bubble_polygons.get(&label).cloned());
        let confidence = finite_unit(text.confidence);
        if confidence < LOW_OCR_CONFIDENCE {
            warnings.push(BrowserWarning {
                code: BrowserWarningCode::LowOcrConfidence,
                region_id: Some(id.clone()),
                message: format!("Koharu OCR confidence is low ({confidence:.2}) for this region."),
            });
        }
        let (style, layout) = browser_style_and_layout(
            transform,
            text,
            page.width,
            &source_english,
            bubble_polygon.clone(),
        );
        regions.push(BrowserRegion {
            id,
            kind: RegionKind::Dialogue,
            text_polygon,
            bubble_polygon,
            rotation_degrees: finite_or(text.rotation_deg.unwrap_or(transform.rotation_deg), 0.0),
            source_english: source_english.clone(),
            faithful_chinese: source_english.clone(),
            displayed_chinese: source_english,
            pinyin: String::new(),
            ocr_confidence: confidence,
            reading_order: u32::try_from(reading_order).unwrap_or(u32::MAX),
            vocabulary: VocabularyStatus {
                requested_hsk_level: request.settings.hsk_level,
                strictly_valid: false,
                exceptions: Vec::new(),
            },
            style,
            layout,
        });
    }

    let inpainted = page
        .nodes
        .values()
        .find_map(|node| match &node.kind {
            NodeKind::Image(image) if image.role == ImageRole::Inpainted => Some(&image.blob),
            _ => None,
        })
        .ok_or_else(|| {
            CleaningError::new(
                "INPAINTING_FAILED",
                "Koharu did not produce a cleaned image.",
            )
        })?;
    let mut clean_image = session.blobs.get_bytes(inpainted).map_err(|error| {
        CleaningError::new(
            "INPAINTING_FAILED",
            format!("Koharu cleaned image could not be loaded: {error:#}"),
        )
    })?;
    let clean_image_mime_type = match image::guess_format(&clean_image) {
        Ok(ImageFormat::Png) => CleanImageMimeType::Png,
        Ok(ImageFormat::WebP) => CleanImageMimeType::Webp,
        _ => {
            let image = session.blobs.load_image(inpainted).map_err(|error| {
                CleaningError::new(
                    "INPAINTING_FAILED",
                    format!("Koharu cleaned image could not be decoded: {error:#}"),
                )
            })?;
            let mut encoded = Cursor::new(Vec::new());
            image
                .write_to(&mut encoded, ImageFormat::Png)
                .map_err(|error| {
                    CleaningError::new(
                        "INPAINTING_FAILED",
                        format!("Koharu cleaned image could not be encoded: {error:#}"),
                    )
                })?;
            clean_image = encoded.into_inner();
            CleanImageMimeType::Png
        }
    };

    Ok(CleaningOutput {
        clean_image,
        clean_image_mime_type,
        regions,
        warnings,
        cache: BrowserCacheStatus {
            detection_hit: cache_hit,
            ocr_hit: cache_hit,
            inpaint_hit: cache_hit,
            translation_hit: false,
        },
    })
}

fn text_polygon(transform: Transform, text: &TextData, width: u32, height: u32) -> Vec<Point> {
    let mut pixels = text
        .line_polygons
        .as_ref()
        .into_iter()
        .flatten()
        .flat_map(|polygon| polygon.iter().copied())
        .filter(|point| point[0].is_finite() && point[1].is_finite())
        .collect::<Vec<_>>();
    if pixels.len() < 3 {
        pixels = rotated_rectangle(transform);
    }
    let hull = convex_hull(pixels);
    let pixels = if hull.len() >= 3 {
        hull
    } else {
        rotated_rectangle(transform)
    };
    normalize_polygon(&pixels, width, height)
}

fn rotated_rectangle(transform: Transform) -> Vec<[f32; 2]> {
    let x0 = finite_or(transform.x, 0.0);
    let y0 = finite_or(transform.y, 0.0);
    let width = finite_or(transform.width, 1.0).max(1.0);
    let height = finite_or(transform.height, 1.0).max(1.0);
    let center_x = x0 + width / 2.0;
    let center_y = y0 + height / 2.0;
    let radians = finite_or(transform.rotation_deg, 0.0).to_radians();
    let (sin, cos) = radians.sin_cos();
    [
        [x0, y0],
        [x0 + width, y0],
        [x0 + width, y0 + height],
        [x0, y0 + height],
    ]
    .into_iter()
    .map(|[x, y]| {
        let dx = x - center_x;
        let dy = y - center_y;
        [
            center_x + dx * cos - dy * sin,
            center_y + dx * sin + dy * cos,
        ]
    })
    .collect()
}

fn normalize_polygon(points: &[[f32; 2]], width: u32, height: u32) -> Vec<Point> {
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    points
        .iter()
        .map(|point| Point {
            x: (finite_or(point[0], 0.0) / width).clamp(0.0, 1.0),
            y: (finite_or(point[1], 0.0) / height).clamp(0.0, 1.0),
        })
        .collect()
}

fn convex_hull(mut points: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    points.sort_by(|left, right| {
        left[0]
            .partial_cmp(&right[0])
            .unwrap_or(CmpOrdering::Equal)
            .then_with(|| left[1].partial_cmp(&right[1]).unwrap_or(CmpOrdering::Equal))
    });
    points.dedup_by(|left, right| {
        (left[0] - right[0]).abs() < f32::EPSILON && (left[1] - right[1]).abs() < f32::EPSILON
    });
    if points.len() <= 2 {
        return points;
    }

    fn cross(origin: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
        (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
    }

    let mut lower = Vec::new();
    for point in &points {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }
    let mut upper = Vec::new();
    for point in points.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
        {
            upper.pop();
        }
        upper.push(*point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn bubble_label_for(mask: &GrayImage, transform: Transform) -> Option<u8> {
    let (width, height) = mask.dimensions();
    let x0 = finite_or(transform.x, 0.0).floor().max(0.0) as u32;
    let y0 = finite_or(transform.y, 0.0).floor().max(0.0) as u32;
    let x1 = (finite_or(transform.x + transform.width, 0.0)
        .ceil()
        .max(0.0) as u32)
        .min(width);
    let y1 = (finite_or(transform.y + transform.height, 0.0)
        .ceil()
        .max(0.0) as u32)
        .min(height);
    let mut counts = [0_u32; 256];
    for y in y0.min(height)..y1 {
        for x in x0.min(width)..x1 {
            let label = mask.get_pixel(x, y).0[0];
            if label != 0 {
                counts[usize::from(label)] = counts[usize::from(label)].saturating_add(1);
            }
        }
    }
    counts
        .iter()
        .enumerate()
        .skip(1)
        .max_by_key(|(_, count)| *count)
        .and_then(|(label, count)| (*count > 0).then_some(label as u8))
}

fn bubble_polygons(mask: &GrayImage, width: u32, height: u32) -> HashMap<u8, Vec<Point>> {
    let mut candidates = HashMap::<u8, Vec<[f32; 2]>>::new();
    for y in 0..mask.height() {
        let mut spans = HashMap::<u8, (u32, u32)>::new();
        for x in 0..mask.width() {
            let label = mask.get_pixel(x, y).0[0];
            if label == 0 {
                continue;
            }
            spans
                .entry(label)
                .and_modify(|span| span.1 = x)
                .or_insert((x, x));
        }
        for (label, (min_x, max_x)) in spans {
            let points = candidates.entry(label).or_default();
            points.extend([
                [min_x as f32, y as f32],
                [(max_x + 1) as f32, y as f32],
                [(max_x + 1) as f32, (y + 1) as f32],
                [min_x as f32, (y + 1) as f32],
            ]);
        }
    }
    candidates
        .into_iter()
        .filter_map(|(label, points)| {
            let polygon = normalize_polygon(&convex_hull(points), width, height);
            (polygon.len() >= 3).then_some((label, polygon))
        })
        .collect()
}

fn stable_region_id(source_sha256: &str, polygon: &[Point]) -> String {
    let mut canonical = String::from("gate3-region-v1|");
    canonical.push_str(source_sha256);
    for point in polygon {
        canonical.push('|');
        canonical.push_str(&format!("{:.6},{:.6}", point.x, point.y));
    }
    let digest = sha256_hex(canonical.as_bytes());
    format!(
        "{}-region-{}",
        &source_sha256[..8.min(source_sha256.len())],
        &digest[..16]
    )
}

fn browser_style_and_layout(
    transform: Transform,
    text: &TextData,
    image_width: u32,
    source_english: &str,
    safe_polygon: Option<Vec<Point>>,
) -> (BrowserTextStyle, BrowserTextLayout) {
    let prediction = text.font_prediction.as_ref();
    let style = text.style.as_ref();
    let foreground = prediction
        .map(|prediction| rgb(prediction.text_color))
        .unwrap_or_else(|| {
            style
                .map(|style| rgba(style.color))
                .unwrap_or_else(|| "#000000".to_owned())
        });
    let stroke_width = prediction
        .map(|prediction| prediction.stroke_width_px)
        .or_else(|| {
            style
                .and_then(|style| style.stroke.as_ref())
                .and_then(|stroke| stroke.width_px)
        })
        .filter(|width| width.is_finite() && *width > 0.0);
    let outline_color = stroke_width.map(|_| {
        prediction
            .map(|prediction| rgb(prediction.stroke_color))
            .or_else(|| {
                style
                    .and_then(|style| style.stroke.as_ref())
                    .map(|stroke| rgba(stroke.color))
            })
            .unwrap_or_else(|| "#ffffff".to_owned())
    });
    let bold = style
        .and_then(|style| style.effect)
        .is_some_and(|effect| effect.bold);
    let italic = style
        .and_then(|style| style.effect)
        .is_some_and(|effect| effect.italic);
    let serif = prediction
        .and_then(|prediction| {
            prediction.named_fonts.iter().max_by(|left, right| {
                left.probability
                    .partial_cmp(&right.probability)
                    .unwrap_or(CmpOrdering::Equal)
            })
        })
        .is_some_and(|font| font.serif);
    let vertical = text.source_direction == Some(TextDirection::Vertical)
        || prediction.is_some_and(|prediction| prediction.direction == TextDirection::Vertical);
    let font_size = prediction
        .map(|prediction| prediction.font_size_px)
        .or(text.detected_font_size_px)
        .or_else(|| style.and_then(|style| style.font_size))
        .filter(|size| size.is_finite() && *size > 0.0)
        .unwrap_or_else(|| finite_or(transform.height, 1.0).max(1.0) * 0.35);
    let line_height = prediction
        .map(|prediction| prediction.line_height)
        .filter(|line_height| line_height.is_finite() && *line_height > 0.0)
        .unwrap_or(1.2)
        .clamp(0.5, 3.0);
    let alignment = match style.and_then(|style| style.text_align) {
        Some(koharu_core::TextAlign::Left) => TextAlignment::Left,
        Some(koharu_core::TextAlign::Right) => TextAlignment::Right,
        Some(koharu_core::TextAlign::Center) | None => TextAlignment::Center,
    };
    let browser_style = BrowserTextStyle {
        font_id: "fixture-sans".to_owned(),
        category: if serif {
            FontCategory::Serif
        } else {
            FontCategory::Sans
        },
        foreground,
        weight: if bold { 700 } else { 400 },
        italic_degrees: if italic { 12.0 } else { 0.0 },
        outline_color,
        outline_width_ratio: stroke_width
            .map(|width| width / finite_or(transform.width, 1.0).max(1.0))
            .unwrap_or(0.0)
            .clamp(0.0, 1.0),
        shadow_color: None,
        shadow_x_ratio: 0.0,
        shadow_y_ratio: 0.0,
        alignment,
        writing_mode: if vertical {
            WritingMode::VerticalRl
        } else {
            WritingMode::HorizontalTb
        },
        line_height,
        letter_spacing_em: 0.0,
    };
    let layout = BrowserTextLayout {
        suggested_lines: source_english.lines().map(ToOwned::to_owned).collect(),
        font_size_to_image_width: (font_size / image_width.max(1) as f32).clamp(f32::EPSILON, 1.0),
        safe_polygon,
    };
    (browser_style, layout)
}

fn finite_unit(value: f32) -> f32 {
    finite_or(value, 0.0).clamp(0.0, 1.0)
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn rgb(color: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", color[0], color[1], color[2])
}

fn rgba(color: [u8; 4]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}{:02x}",
        color[0], color[1], color[2], color[3]
    )
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgba, RgbaImage};
    use koharu_core::{BlobRef, MaskData};
    use tempfile::tempdir;

    use super::*;
    use crate::contracts::{HskLevel, Validate};

    fn png(image: DynamicImage) -> Vec<u8> {
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, ImageFormat::Png).unwrap();
        bytes.into_inner()
    }

    fn add_image_node(page: &mut Page, role: ImageRole, blob: BlobRef, width: u32, height: u32) {
        let id = NodeId::new();
        page.nodes.insert(
            id,
            Node {
                id,
                transform: Transform::default(),
                visible: true,
                kind: NodeKind::Image(ImageData {
                    role,
                    blob,
                    opacity: 1.0,
                    natural_width: width,
                    natural_height: height,
                    name: None,
                }),
            },
        );
    }

    fn add_mask_node(page: &mut Page, role: MaskRole, blob: BlobRef) {
        let id = NodeId::new();
        page.nodes.insert(
            id,
            Node {
                id,
                transform: Transform::default(),
                visible: false,
                kind: NodeKind::Mask(MaskData { role, blob }),
            },
        );
    }

    fn add_text_node(
        page: &mut Page,
        transform: Transform,
        confidence: f32,
        text: &str,
        polygon: [[f32; 2]; 4],
    ) {
        let id = NodeId::new();
        page.nodes.insert(
            id,
            Node {
                id,
                transform,
                visible: true,
                kind: NodeKind::Text(TextData {
                    confidence,
                    source_lang: Some("en".to_owned()),
                    source_direction: Some(TextDirection::Horizontal),
                    line_polygons: Some(vec![polygon]),
                    rotation_deg: Some(transform.rotation_deg),
                    detected_font_size_px: Some(24.0),
                    text: Some(text.to_owned()),
                    ..Default::default()
                }),
            },
        );
    }

    #[test]
    fn tall_webtoon_scene_packages_stable_regions_masks_and_clean_blob() {
        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 4_096;
        let temp = tempdir().unwrap();
        let project = Utf8PathBuf::from_path_buf(temp.path().join("tall.khrproj")).unwrap();
        let session = ProjectSession::create(&project, "tall synthetic").unwrap();

        let mut source = RgbaImage::from_pixel(WIDTH, HEIGHT, Rgba([238, 238, 238, 255]));
        for y in 200..520 {
            for x in 40..216 {
                source.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        for y in 2_700..3_060 {
            for x in 28..228 {
                source.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        let source_bytes = png(DynamicImage::ImageRgba8(source.clone()));
        let source_blob = session.blobs.put_bytes(&source_bytes).unwrap();
        let clean_blob = session
            .blobs
            .put_bytes(&png(DynamicImage::ImageRgba8(source)))
            .unwrap();

        let mut segment = GrayImage::new(WIDTH, HEIGHT);
        for y in 310..360 {
            for x in 70..190 {
                segment.put_pixel(x, y, Luma([255]));
            }
        }
        let segment_blob = session
            .blobs
            .put_bytes(&png(DynamicImage::ImageLuma8(segment)))
            .unwrap();
        let mut bubbles = GrayImage::new(WIDTH, HEIGHT);
        for (label, center_y, radius_x, radius_y) in [
            (1_u8, 360_i32, 96_i32, 150_i32),
            (2_u8, 2_880_i32, 108_i32, 170_i32),
        ] {
            for y in (center_y - radius_y)..=(center_y + radius_y) {
                for x in (128 - radius_x)..=(128 + radius_x) {
                    let dx = (x - 128) as f32 / radius_x as f32;
                    let dy = (y - center_y) as f32 / radius_y as f32;
                    if dx * dx + dy * dy <= 1.0 {
                        bubbles.put_pixel(x as u32, y as u32, Luma([label]));
                    }
                }
            }
        }
        let bubble_blob = session
            .blobs
            .put_bytes(&png(DynamicImage::ImageLuma8(bubbles)))
            .unwrap();

        let mut page = Page::new("synthetic-webtoon.png", WIDTH, HEIGHT);
        let page_id = page.id;
        add_image_node(&mut page, ImageRole::Source, source_blob, WIDTH, HEIGHT);
        add_image_node(&mut page, ImageRole::Inpainted, clean_blob, WIDTH, HEIGHT);
        add_mask_node(&mut page, MaskRole::Segment, segment_blob);
        add_mask_node(&mut page, MaskRole::Bubble, bubble_blob);
        add_text_node(
            &mut page,
            Transform {
                x: 68.0,
                y: 300.0,
                width: 124.0,
                height: 72.0,
                rotation_deg: -2.0,
            },
            0.42,
            "WE SHOULD GO NOW!",
            [[70.0, 305.0], [190.0, 301.0], [192.0, 368.0], [72.0, 372.0]],
        );
        add_text_node(
            &mut page,
            Transform {
                x: 58.0,
                y: 2_815.0,
                width: 142.0,
                height: 88.0,
                rotation_deg: 0.0,
            },
            0.96,
            "I KNOW.",
            [
                [60.0, 2_820.0],
                [198.0, 2_820.0],
                [198.0, 2_900.0],
                [60.0, 2_900.0],
            ],
        );
        session.apply(Op::AddPage { page, at: 0 }).unwrap();

        let request = BrowserJobRequest {
            protocol_version: crate::contracts::PROTOCOL_VERSION,
            client_image_id: "tall-webtoon".to_owned(),
            source_sha256: sha256_hex(&source_bytes),
            source_mime_type: "image/png".to_owned(),
            natural_width: WIDTH,
            natural_height: HEIGHT,
            page_session_id: "page".to_owned(),
            page_index: 0,
            settings: crate::contracts::BrowserJobSettings {
                source_language: "en".to_owned(),
                target_language: "zh-CN".to_owned(),
                hsk_standard: "2.0".to_owned(),
                hsk_level: HskLevel::Three,
                reading_direction: ReadingDirection::Auto,
                translate_sound_effects: false,
            },
            preceding_context: Some(Vec::new()),
        };
        let first = extract_output(&session, page_id, &request, false).unwrap();
        let second = extract_output(&session, page_id, &request, true).unwrap();

        assert_eq!(first.regions.len(), 2);
        assert_eq!(
            first
                .regions
                .iter()
                .map(|region| region.id.as_str())
                .collect::<Vec<_>>(),
            second
                .regions
                .iter()
                .map(|region| region.id.as_str())
                .collect::<Vec<_>>()
        );
        assert!(first.regions.iter().all(|region| {
            region
                .text_polygon
                .iter()
                .all(|point| (0.0..=1.0).contains(&point.x) && (0.0..=1.0).contains(&point.y))
        }));
        assert!(
            first
                .regions
                .iter()
                .all(|region| region.bubble_polygon.as_ref().is_some_and(|p| p.len() > 4))
        );
        assert_eq!(first.regions[0].faithful_chinese, "WE SHOULD GO NOW!");
        assert!(!first.regions[0].vocabulary.strictly_valid);
        assert_eq!(first.warnings.len(), 1);
        assert_eq!(first.warnings[0].code, BrowserWarningCode::LowOcrConfidence);
        assert!(!first.cache.detection_hit);
        assert!(second.cache.detection_hit);
        assert_eq!(
            image::load_from_memory(&first.clean_image)
                .unwrap()
                .dimensions(),
            (WIDTH, HEIGHT)
        );

        let result = crate::contracts::BrowserJobResult {
            protocol_version: crate::contracts::PROTOCOL_VERSION,
            job_id: "job-tall".to_owned(),
            source_sha256: request.source_sha256,
            source_width: WIDTH,
            source_height: HEIGHT,
            clean_image_blob_id: "blob-tall".to_owned(),
            clean_image_mime_type: first.clean_image_mime_type,
            regions: first.regions,
            warnings: first.warnings,
            cache: first.cache,
        };
        result.validate().unwrap();
    }
}
