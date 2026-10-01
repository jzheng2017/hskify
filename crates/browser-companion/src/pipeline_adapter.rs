//! Direct, chapter-aware browser pipeline.
//!
//! The browser path deliberately does not create Koharu projects. It decodes
//! the upload once, runs resident CUDA models over overlapping detector tiles,
//! restores accepted text regions in one image-level semantic inpainting pass,
//! and publishes one transparent cleanup patch per translated dialogue region.

mod geometry;
mod ocr;
mod patch;
#[path = "pipeline_adapter/ppocr_small.rs"]
mod ppocr;
#[path = "pipeline_adapter/ppocr_detector.rs"]
mod ppocr_detector;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use camino::Utf8PathBuf;
use hsk_control::{
    HskControl, HskLevel as ControlHskLevel, LookupItemContext as ControlLookupItem, ProperName,
    ValidationReport, ViolationReason,
};
use icu_segmenter::{SentenceSegmenter, options::SentenceBreakInvariantOptions};
use image::{
    DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage,
    imageops::{FilterType, crop_imm},
};
use imageproc::geometric_transformations::{Border, Interpolation, rotate_about_center};
use koharu_app::llm::{
    DirectSourceProvenance, FaithfulSourceUtterance, FaithfulTranslationBatchRequest,
    HSK_TRANSLATION_MODEL, HskLayoutConstraints, HskLearningMode, HskPrecedingUtterance,
    HskRepairUtterance, HskSourceUtterance, HskTranslationBatchRequest, HskTranslationOutcome,
    HskTranslationRepairBatchRequest, HskUtteranceKind, MAX_HSK_LAYOUT_CHARACTERS,
    MAX_HSK_LAYOUT_LINES, MAX_HSK_PRECEDING_UTTERANCES, MIN_HSK_LAYOUT_CHARACTERS,
};
use koharu_app::{App, AppConfig};
use koharu_llm::page_understanding::{
    PagePoint, PageRegionEvidence, PageRegionRole, PageUnderstandingRequest,
    PageUnderstandingResult, QwenPageUnderstanding, probe_qwen_page_understanding,
};
use koharu_llm::safe::model::AddBos;
use koharu_ml::comic_text_bubble_detector::{ComicTextBubbleDetector, DETECTOR_TILE_BATCH_SIZE};
use koharu_ml::inpainting::expand_mask_for_inpainting;
use koharu_ml::lama::Lama;
use koharu_ml::manga_text_segmentation_2025::{DEFAULT_TEXT_MASK_THRESHOLD, MangaTextSegmentation};
use koharu_ml::probability_map::ProbabilityMap;
use koharu_ml::speech_bubble_segmentation::SpeechBubbleSegmentation;
use koharu_ml::types::TextRegion;
use koharu_runtime::{ComputePolicy, RuntimeManager};
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as AsyncMutex, OnceCell, oneshot};

use self::geometry::{
    Candidate, CandidateKind, PixelBounds, PixelRect, Tile, bubbles_for_tile,
    candidates_for_comic_text_boxes, candidates_for_text_boxes, next_detector_batch_count,
    ocr_crop_rect, overlapping_tiles, prioritize_tiles, reading_order_ranks, spatially_dedupe,
    take_finalized_lines, take_finalized_rejected_lines, text_candidate_is_confirmed,
};
use self::patch::{
    CleanupMask, CleanupQuality, PatchPng, bubble_component_bounds, bubble_id_for_rect,
    bubble_id_mask, compact_cleanup_mask, crop_probability_map, label_bubble_components,
    make_inpainted_patch, merge_binary_mask, merge_cleanup_mask, merge_probability_map,
    merge_source_guided_glyph_probabilities, protected_pixels_match, region_polygons,
    score_cleanup_candidate_local, verified_text_mask_for_regions_local,
};
use self::ppocr::{
    MAX_LINE_BATCH_SIZE, PpOcrAppearanceBand, PpOcrLine, PpOcrPrediction, PpOcrSmallRecognizer,
};
use self::ppocr_detector::PpOcrSmallDetector;
use crate::chapter_session::{
    ChapterContextStore, ChapterContextUnit, ChapterSessionStore, PageAnalysis, PageSurface,
    PageSurfaceKind, RegionPlan, RegionRole,
};
use crate::contracts::{
    BrowserJobStage, BrowserSurfaceKind, BrowserTextColorBand, BrowserTextLayout, BrowserTextStyle,
    ChapterKind, DocumentBlockPreserved, DocumentBlockReady, DocumentJobRequest,
    DocumentSourceBlock, FontCategory, HskLevel, HskRepairState, ImagePipelineInput,
    ImageRegionPreserved, ImageRegionReady, ImageRegionRole, LearningMode, LookupItem,
    LookupResult, LookupToken, NormalizedRect, Point, RegionConfidenceEvidence, SourceProvenance,
    SourceSpanKind, TeachingTerm, TeachingTermReason, TextAlignment, TranslatedHskStatus,
    TranslatedText, WritingMode,
};
use crate::crypto::sha256_hex;
use crate::cuda_scheduler::{
    CudaAdmissionError, CudaPriority, CudaScheduler, CudaWorkload, global_cuda_scheduler,
};
use crate::server::{JobUpdateDraft, JobUpdateSink};
use crate::setup::{
    BUBBLE_SEGMENTER_CONFIG_ID, BUBBLE_SEGMENTER_WEIGHTS_ID, DETECTOR_CONFIG_ID,
    DETECTOR_PREPROCESSOR_ID, DETECTOR_WEIGHTS_ID, INPAINTER_WEIGHTS_ID, OCR_CONFIG_ID,
    OCR_DETECTOR_CONFIG_ID, OCR_DETECTOR_MODEL_ID, OCR_MODEL_ID, PAGE_PROJECTOR_ID,
    ResidentResourcePaths, TEXT_SEGMENTER_WEIGHTS_ID, TRANSLATION_MODEL_ID,
};

const OCR_REGION_BATCH_SIZE: usize = MAX_LINE_BATCH_SIZE;
// Keep recovery inference batched without holding the vision permit for an
// entire page. This matches the detector's bounded CUDA batch size, so a
// visible detector/OCR phase can run between recovery batches.
const TRANSLATION_BATCH_MAX: usize = 6;
const TRANSLATION_BATCH_MIN: usize = 3;
const DOCUMENT_BATCH_MAX: usize = 6;
const DOCUMENT_PIECE_TOKEN_BUDGET: usize = 320;
const DOCUMENT_BATCH_TOKEN_BUDGET: usize = 1_800;
const DOCUMENT_CONTEXT_TOKEN_BUDGET: usize = 512;
const BROWSER_QWEN_INFERENCE_THREADS: i32 = 6;
// Cleanup is a bounded, optional stage. A stalled inpainting/quality task must
// never hold the ordered language stream indefinitely: the source pixels stay
// intact and the region is published as unreadable when this deadline expires.
const CLEANUP_RESULT_TIMEOUT: Duration = Duration::from_secs(90);
const TRANSLATION_CACHE_SCHEMA: &str = "hskify-source-context-policy-v6-2026-10-01";
// Multimodal inference should see enough artwork to classify a region, but a
// continuous reader strip must not be sent to the projector at its full
// height for every bounded language window.  The evidence viewport is
// derived from the accepted OCR/bubble geometry, never from a chapter or
// reader-specific crop rule.
const PAGE_EVIDENCE_MAX_PIXELS: u64 = 6_000_000;
const PAGE_EVIDENCE_CROP_RATIO: f32 = 0.82;
const PAGE_EVIDENCE_MIN_MARGIN: f32 = 96.0;
const PAGE_EVIDENCE_MAX_MARGIN: f32 = 512.0;
const PAGE_ROLE_MAX_LONG_EDGE: u32 = 768;

pub(crate) use crate::contracts::LookupContext as ItemLookupContext;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LookupInput {
    Selection(String),
    Hover {
        displayed_text: String,
        character_offset: usize,
    },
}

const TRANSLATION_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;
const PREPROCESSING_THREADS: usize = 6;

static PREPROCESSING_POOL: OnceLock<std::result::Result<Arc<PreprocessingPool>, String>> =
    OnceLock::new();

#[derive(Debug, Clone)]
pub(crate) struct ImageJobInput {
    pub source: Arc<DynamicImage>,
    pub request: ImagePipelineInput,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub(crate) struct PipelineError {
    pub code: &'static str,
    pub message: String,
}

impl PipelineError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    fn pipeline(error: anyhow::Error) -> Self {
        Self::new(
            "PIPELINE_FAILED",
            format!("Direct browser pipeline failed: {error:#}"),
        )
    }

    pub(crate) fn cancelled() -> Self {
        Self::new("CANCELLED", "Chapter processing was cancelled.")
    }

    /// A browser retry is safe only for failures whose evidence is still
    /// valid and whose cause is external to the page analysis.  OCR,
    /// semantic, HSK, cleanup, and model-contract failures are terminal for
    /// this job; retrying them would rerun the identical image pipeline and
    /// create the false progress/retry loop the chapter contract forbids.
    pub(crate) fn is_transient(&self) -> bool {
        matches!(self.code, "CUDA_QUEUE_FULL" | "UPDATE_PUBLISH_FAILED")
    }
}

#[async_trait]
pub(crate) trait ChapterPipeline: Send + Sync {
    async fn warm_up(&self, kind: ChapterKind) -> std::result::Result<(), PipelineError>;

    async fn run_image(
        &self,
        input: ImageJobInput,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), PipelineError>;

    async fn run_document(
        &self,
        request: DocumentJobRequest,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), PipelineError>;

    async fn lookup(
        &self,
        input: LookupInput,
        item: Option<ItemLookupContext>,
    ) -> std::result::Result<LookupResult, PipelineError>;

    /// Rebuild chapter ordering state before a terminal result-cache replay.
    /// A cached page skips model execution, but it must still participate in
    /// the same analysis/language barriers and dialogue/entity graph as a
    /// freshly processed page. Test pipelines may use the default no-op.
    fn restore_cached_context(
        &self,
        _request: &ImagePipelineInput,
        _regions: &[ImageRegionReady],
        _preserved_regions: &[ImageRegionPreserved],
    ) -> std::result::Result<(), PipelineError> {
        Ok(())
    }

    fn image_cache_context(&self, _request: &ImagePipelineInput) -> Vec<ChapterContextUnit> {
        Vec::new()
    }

    /// Record a non-retryable pre-pipeline failure in the chapter graph.  A
    /// page that fails before [`run`] starts still occupies a canonical page
    /// slot; leaving that slot open would make later pages wait forever for a
    /// predecessor that can no longer contribute context.  Lightweight test
    /// pipelines do not need a graph, so the default is intentionally a
    /// no-op.
    fn mark_page_terminal(
        &self,
        _request: &ImagePipelineInput,
    ) -> std::result::Result<(), PipelineError> {
        Ok(())
    }

    /// Release all chapter-owned dialogue/entity state once the browser has
    /// sealed or cancelled a chapter.  The default keeps lightweight test
    /// pipelines source-compatible while the production pipeline owns the
    /// actual session graph.
    fn close_chapter(&self, _page_session_id: &str) {}

    fn record_font_invocation(&self) {}

    fn resources_ready(&self) -> bool;
}

#[derive(Clone, Copy)]
enum RuntimeComponent {
    Vision = 0,
    Ocr = 1,
    Projector = 2,
    Segmentation = 3,
    Inpainting = 4,
    Patch = 5,
    Font = 6,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeCounters {
    vision: u64,
    ocr: u64,
    projector: u64,
    segmentation: u64,
    inpainting: u64,
    patch: u64,
    font: u64,
}

impl RuntimeCounters {
    fn between(start: Self, end: Self) -> Self {
        Self {
            vision: end.vision.saturating_sub(start.vision),
            ocr: end.ocr.saturating_sub(start.ocr),
            projector: end.projector.saturating_sub(start.projector),
            segmentation: end.segmentation.saturating_sub(start.segmentation),
            inpainting: end.inpainting.saturating_sub(start.inpainting),
            patch: end.patch.saturating_sub(start.patch),
            font: end.font.saturating_sub(start.font),
        }
    }
}

struct RuntimeInstrumentation {
    initializations: [AtomicU64; 7],
    invocations: [AtomicU64; 7],
}

impl Default for RuntimeInstrumentation {
    fn default() -> Self {
        Self {
            initializations: std::array::from_fn(|_| AtomicU64::new(0)),
            invocations: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl RuntimeInstrumentation {
    fn initialized(&self, component: RuntimeComponent) {
        self.initializations[component as usize].fetch_add(1, Ordering::Relaxed);
    }

    fn invoked(&self, component: RuntimeComponent) {
        self.invocations[component as usize].fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (RuntimeCounters, RuntimeCounters) {
        let read = |values: &[AtomicU64; 7]| RuntimeCounters {
            vision: values[RuntimeComponent::Vision as usize].load(Ordering::Relaxed),
            ocr: values[RuntimeComponent::Ocr as usize].load(Ordering::Relaxed),
            projector: values[RuntimeComponent::Projector as usize].load(Ordering::Relaxed),
            segmentation: values[RuntimeComponent::Segmentation as usize].load(Ordering::Relaxed),
            inpainting: values[RuntimeComponent::Inpainting as usize].load(Ordering::Relaxed),
            patch: values[RuntimeComponent::Patch as usize].load(Ordering::Relaxed),
            font: values[RuntimeComponent::Font as usize].load(Ordering::Relaxed),
        };
        (read(&self.initializations), read(&self.invocations))
    }
}

pub(crate) struct HskifyPipeline {
    cache_root: PathBuf,
    cuda_scheduler: Arc<CudaScheduler>,
    language: OnceCell<Arc<LanguageRuntime>>,
    vision: OnceCell<Arc<VisionRuntime>>,
    language_ready: OnceCell<()>,
    vision_ready: OnceCell<()>,
    chapter_sessions: Mutex<ChapterSessionStore>,
    chapter_contexts: Mutex<ChapterContextStore>,
    instrumentation: Arc<RuntimeInstrumentation>,
    analysis_cache: Mutex<TranslationCache<Arc<AnalyzedPage>>>,
}

struct ImagePipeline<'a> {
    shared: &'a HskifyPipeline,
}

impl std::ops::Deref for ImagePipeline<'_> {
    type Target = HskifyPipeline;

    fn deref(&self) -> &Self::Target {
        self.shared
    }
}

struct DocumentPipeline<'a> {
    shared: &'a HskifyPipeline,
}

impl std::ops::Deref for DocumentPipeline<'_> {
    type Target = HskifyPipeline;

    fn deref(&self) -> &Self::Target {
        self.shared
    }
}

impl HskifyPipeline {
    pub(crate) fn new(cache_root: PathBuf) -> Self {
        Self {
            cache_root,
            cuda_scheduler: global_cuda_scheduler(),
            language: OnceCell::new(),
            vision: OnceCell::new(),
            language_ready: OnceCell::new(),
            vision_ready: OnceCell::new(),
            chapter_sessions: Mutex::new(ChapterSessionStore::default()),
            chapter_contexts: Mutex::new(ChapterContextStore::default()),
            instrumentation: Arc::new(RuntimeInstrumentation::default()),
            analysis_cache: Mutex::new(TranslationCache::default()),
        }
    }

    fn resource_paths(&self) -> Result<ResidentResourcePaths> {
        ResidentResourcePaths::discover()
    }

    fn image_pipeline(&self) -> ImagePipeline<'_> {
        ImagePipeline { shared: self }
    }

    fn document_pipeline(&self) -> DocumentPipeline<'_> {
        DocumentPipeline { shared: self }
    }

    async fn vision_runtime(&self) -> Result<&Arc<VisionRuntime>> {
        let resources = self.resource_paths()?;
        let language = Arc::clone(self.language_runtime().await?);
        self.vision
            .get_or_try_init(|| async move {
                VisionRuntime::load(language, resources, Arc::clone(&self.instrumentation))
                    .await
                    .map(Arc::new)
            })
            .await
    }

    async fn language_runtime(&self) -> Result<&Arc<LanguageRuntime>> {
        let resources = self.resource_paths()?;
        let runtime_root = resources.runtime_root().to_path_buf();
        let app_state_root = self.cache_root.join("browser-runtime").join("app-state");
        self.language
            .get_or_try_init(|| async move {
                LanguageRuntime::load(runtime_root, app_state_root, resources)
                    .await
                    .map(Arc::new)
            })
            .await
    }

    async fn hsk_control(&self) -> Result<&Arc<HskControl>> {
        Ok(&self.language_runtime().await?.hsk_control)
    }

    async fn ready_language(&self) -> Result<(&Arc<LanguageRuntime>, &Arc<HskControl>)> {
        let language = self.language_runtime().await?;
        self.language_ready
            .get_or_try_init(|| {
                let language = Arc::clone(language);
                async move {
                    tokio::task::spawn_blocking(move || language.prime())
                        .await
                        .context("join resident language inference warm-up")??;
                    Ok::<(), anyhow::Error>(())
                }
            })
            .await?;
        Ok((language, &language.hsk_control))
    }

    async fn ready_models(&self) -> Result<(&Arc<VisionRuntime>, &Arc<HskControl>)> {
        let (_, control) = self.ready_language().await?;
        let resident = self.vision_runtime().await?;
        self.vision_ready
            .get_or_try_init(|| {
                let resident = Arc::clone(resident);
                async move {
                    tokio::task::spawn_blocking(move || {
                        resident.prime_non_language_inference()?;
                        resident.prime_page_understanding()
                    })
                    .await
                    .context("join resident full-pipeline inference warm-up")??;
                    Ok::<(), anyhow::Error>(())
                }
            })
            .await?;
        Ok((resident, control))
    }
}

impl DocumentPipeline<'_> {
    async fn run(
        &self,
        request: DocumentJobRequest,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), PipelineError> {
        let runtime_start = self.instrumentation.snapshot();
        cancellation_boundary(cancel.as_ref())?;
        publish_progress(
            &sink,
            BrowserJobStage::Registering,
            None,
            Some(0.01),
            Some(0),
            Some(request.blocks.len().min(u32::MAX as usize) as u32),
            "Registering the complete ordered document context",
        )?;
        self.chapter_contexts
            .lock()
            .map_err(|_| {
                PipelineError::new("CHAPTER_CONTEXT_FAILED", "Chapter context lock poisoned.")
            })?
            .register(
                &request.page_session_id,
                request.blocks.iter().map(|block| ChapterContextUnit {
                    item_id: block.item_id.clone(),
                    source_index: block.source_index,
                    item_order: block.item_order,
                    source_text: block.text.clone(),
                    displayed_text: None,
                }),
            );
        publish_progress(
            &sink,
            BrowserJobStage::Warming,
            None,
            Some(0.02),
            None,
            None,
            "Preparing the shared language runtime",
        )?;
        let (language, control) = self
            .ready_language()
            .await
            .map_err(PipelineError::pipeline)?;
        let translator = language.app.llm.direct_hsk_translator();
        let mut document_evidence = DocumentJobEvidence {
            job_id: sink.job_id().to_owned(),
            tokenizer_identity: format!(
                "{}@{}:native-llama-tokenizer",
                translator.model_id(),
                translator.model_revision()
            ),
            dispatches: Vec::new(),
            runtime_initializations: RuntimeCounters::default(),
            runtime_invocations: RuntimeCounters::default(),
        };
        let model = language
            .app
            .llm
            .local_model_handle()
            .await
            .map_err(PipelineError::pipeline)?;

        // Focus updates are coalesced by the browser. Give the first update a
        // bounded chance to arrive without delaying a caller that supplied no
        // visible blocks.
        if sink.focus().visible_block_ids.is_empty() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        cancellation_boundary(cancel.as_ref())?;
        let mut planned = Vec::<DocumentPiece>::new();
        let mut context_plan = Vec::<DocumentContextSpan>::new();
        let mut preserved = HashMap::<usize, String>::new();
        for (block_index, block) in request.blocks.iter().enumerate() {
            match split_document_block(model.as_ref(), block_index, block) {
                Ok(pieces) => {
                    context_plan.extend(pieces.iter().map(DocumentContextSpan::from));
                    planned.extend(pieces);
                }
                Err(reason) => {
                    // A source-preserved block still occupies its canonical
                    // context position. If tokenization itself failed, keep
                    // an over-budget barrier so context selection cannot jump
                    // across it to a farther block.
                    let token_count = model_token_count(model.as_ref(), &block.text)
                        .unwrap_or(DOCUMENT_CONTEXT_TOKEN_BUDGET.saturating_add(1));
                    context_plan.push(DocumentContextSpan {
                        piece_id: None,
                        source_text: block.text.clone(),
                        separator_after: String::new(),
                        token_count,
                    });
                    preserved.insert(block_index, reason);
                }
            }
        }

        let targeted = !request.retry_item_ids.is_empty();
        let retry_ids = request
            .retry_item_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let mut remaining = planned
            .iter()
            .filter(|piece| {
                !targeted || retry_ids.contains(request.blocks[piece.block_index].item_id.as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut accumulators = request
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                let piece_count = planned
                    .iter()
                    .filter(|piece| piece.block_index == index)
                    .count();
                let mut accumulator = DocumentAccumulator::new(block.clone(), piece_count);
                accumulator.published = targeted && !retry_ids.contains(block.item_id.as_str());
                accumulator
            })
            .collect::<Vec<_>>();
        for (index, reason) in preserved.drain() {
            accumulators[index].reject(reason);
        }
        let mut published_count = 0usize;
        publish_ready_document_blocks(
            &sink,
            control,
            request.settings.hsk_level,
            request.settings.learning_mode,
            &mut accumulators,
            &mut published_count,
            &self.chapter_contexts,
            &request.page_session_id,
        )?;
        let total = request.blocks.len().max(1);
        let mut first_dispatch = true;

        while !remaining.is_empty() {
            cancellation_boundary(cancel.as_ref())?;
            let selected_focus = wait_for_active_document(&sink, cancel.as_ref()).await?;
            let visible = selected_focus
                .visible_block_ids
                .iter()
                .cloned()
                .collect::<HashSet<_>>();
            let (batch, dispatch_reason) =
                take_next_document_batch(&mut remaining, &request.blocks, &visible, first_dispatch);
            first_dispatch = false;
            if batch.is_empty() {
                return Err(PipelineError::new(
                    "DOCUMENT_SCHEDULER_FAILED",
                    "The document scheduler could not select a remaining source piece.",
                ));
            }
            document_evidence.dispatches.push(DocumentDispatchEvidence {
                reason: dispatch_reason.to_owned(),
                item_ids: batch
                    .iter()
                    .map(|piece| request.blocks[piece.block_index].item_id.clone())
                    .fold(Vec::new(), |mut ids, item_id| {
                        if !ids.contains(&item_id) {
                            ids.push(item_id);
                        }
                        ids
                    }),
                token_count: batch.iter().map(|piece| piece.token_count).sum(),
            });
            refresh_document_runtime_evidence(
                &mut document_evidence,
                runtime_start,
                self.instrumentation.as_ref(),
            );
            emit_document_evidence(&document_evidence).map_err(PipelineError::pipeline)?;
            let mut queued = batch.into_iter();
            while let Some(piece) = queued.next() {
                let current_focus = sink.focus();
                if !current_focus.active
                    || current_focus.visible_block_ids != selected_focus.visible_block_ids
                {
                    remaining.push(piece);
                    remaining.extend(queued);
                    break;
                }
                let translated = self
                    .translation_service(language, control)
                    .translate_document_piece(
                        &piece,
                        &request,
                        &context_plan,
                        visible.contains(&request.blocks[piece.block_index].item_id),
                        cancel.clone(),
                    )
                    .await?;
                match translated.items.get(&piece.piece_id) {
                    Some(Ok(value)) => accumulators[piece.block_index].record(
                        piece.piece_order,
                        value.clone(),
                        piece.separator_after,
                    ),
                    Some(Err(reason)) => accumulators[piece.block_index].reject(reason.clone()),
                    None => accumulators[piece.block_index]
                        .reject("the language model omitted this document piece".to_owned()),
                }
                publish_ready_document_blocks(
                    &sink,
                    control,
                    request.settings.hsk_level,
                    request.settings.learning_mode,
                    &mut accumulators,
                    &mut published_count,
                    &self.chapter_contexts,
                    &request.page_session_id,
                )?;
            }
            let finished = published_count;
            publish_progress(
                &sink,
                BrowserJobStage::Translating,
                Some(finished as f32 / total as f32),
                Some(0.05 + 0.9 * finished as f32 / total as f32),
                Some(finished.min(u32::MAX as usize) as u32),
                Some(total.min(u32::MAX as usize) as u32),
                "Publishing final document blocks",
            )?;
        }

        for index in 0..accumulators.len() {
            if !accumulators[index].published && !accumulators[index].is_complete() {
                accumulators[index].reject("document block translation was incomplete".to_owned());
            }
        }
        publish_ready_document_blocks(
            &sink,
            control,
            request.settings.hsk_level,
            request.settings.learning_mode,
            &mut accumulators,
            &mut published_count,
            &self.chapter_contexts,
            &request.page_session_id,
        )?;
        refresh_document_runtime_evidence(
            &mut document_evidence,
            runtime_start,
            self.instrumentation.as_ref(),
        );
        emit_document_evidence(&document_evidence).map_err(PipelineError::pipeline)?;
        Ok(())
    }
}

impl HskifyPipeline {
    fn translation_service<'a>(
        &'a self,
        language: &'a LanguageRuntime,
        control: &'a HskControl,
    ) -> TranslationService<'a> {
        TranslationService {
            language,
            control,
            cuda_scheduler: &self.cuda_scheduler,
        }
    }
}

impl ImagePipeline<'_> {
    fn record_page_analysis(
        &self,
        request: &ImagePipelineInput,
        width: u32,
        height: u32,
        kind: PageSurfaceKind,
        regions: &[RegionPlan],
        complete: bool,
    ) -> std::result::Result<(), PipelineError> {
        let mut sessions = self.chapter_sessions.lock().map_err(|_| {
            PipelineError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
        })?;
        if sessions
            .session(&request.page_session_id)
            .and_then(|chapter| chapter.surfaces.get(&request.source_index))
            .is_some_and(|surface| surface.source_sha256 != request.source_sha256)
        {
            return Err(PipelineError::new(
                "SOURCE_REVISION_CHANGED",
                "The chapter has replaced this page source.",
            ));
        }
        sessions
            .session_mut(&request.page_session_id)
            .record_analysis(PageAnalysis {
                surface: PageSurface {
                    session_id: request.page_session_id.clone(),
                    page_index: request.source_index,
                    source_sha256: request.source_sha256.clone(),
                    width,
                    height,
                    kind,
                },
                regions: regions.to_vec(),
                complete,
            });
        drop(sessions);
        self.chapter_contexts
            .lock()
            .map_err(|_| {
                PipelineError::new("CHAPTER_CONTEXT_FAILED", "Chapter context lock poisoned.")
            })?
            .register(
                &request.page_session_id,
                regions.iter().filter_map(|region| {
                    (!matches!(
                        region.role,
                        RegionRole::Exclusion
                            | RegionRole::TechniqueArtwork
                            | RegionRole::Unreadable
                    ))
                    .then(|| ChapterContextUnit {
                        item_id: region.id.clone(),
                        source_index: request.source_index,
                        item_order: region.reading_order,
                        source_text: region.source_english.clone(),
                        displayed_text: None,
                    })
                }),
            );
        Ok(())
    }

    async fn run(
        &self,
        input: ImageJobInput,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), PipelineError> {
        cancellation_boundary(cancel.as_ref())?;
        publish_progress(
            &sink,
            BrowserJobStage::Decoding,
            None,
            Some(0.01),
            None,
            None,
            "Decoding the source image once",
        )?;
        let source = input.source;
        let (image_width, image_height) = source.dimensions();
        if image_width != input.request.natural_width
            || image_height != input.request.natural_height
        {
            return Err(PipelineError::new(
                "IMAGE_DIMENSION_MISMATCH",
                "Decoded dimensions do not match the submitted job metadata.",
            ));
        }
        let surface_kind = page_surface_kind(input.request.surface_kind, image_width, image_height);
        // Keep the non-Send std mutex guard inside a synchronous scope.  The
        // pipeline is a `Send` future because the daemon may move it between
        // Tokio workers while resident model work is awaited below.
        {
            let mut chapter_sessions = self.chapter_sessions.lock().map_err(|_| {
                PipelineError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
            })?;
            let chapter = chapter_sessions.session_mut(&input.request.page_session_id);
            let changed = chapter.register_surface(PageSurface {
                session_id: input.request.page_session_id.clone(),
                page_index: input.request.source_index,
                source_sha256: input.request.source_sha256.clone(),
                width: image_width,
                height: image_height,
                kind: surface_kind.clone(),
            });
            drop(chapter_sessions);
            if changed {
                self.chapter_contexts
                    .lock()
                    .map_err(|_| {
                        PipelineError::new(
                            "CHAPTER_CONTEXT_FAILED",
                            "Chapter context lock poisoned.",
                        )
                    })?
                    .remove_source(&input.request.page_session_id, input.request.source_index);
            }
        }
        cancellation_boundary(cancel.as_ref())?;
        publish_progress(
            &sink,
            BrowserJobStage::Detecting,
            None,
            Some(0.02),
            None,
            None,
            "Loading resident CUDA detector, OCR, and translation models",
        )?;
        let (resident, control) = self.ready_models().await.map_err(PipelineError::pipeline)?;
        let analysis_key = image_analysis_key(&input.request);
        let cached_analysis = self
            .analysis_cache
            .lock()
            .map_err(|_| PipelineError::new("CACHE_FAILED", "Page analysis cache lock poisoned."))?
            .get(&analysis_key);
        if let Some(analysis) = cached_analysis.filter(|analysis| {
            analysis.preserved.iter().all(|region| {
                region.disposition == crate::contracts::PreservationDisposition::Excluded
            }) && input
                .request
                .retry_item_ids
                .iter()
                .all(|id| analysis.regions.iter().any(|region| &region.id == id))
        }) {
            self.record_page_analysis(
                &input.request,
                image_width,
                image_height,
                surface_kind,
                &analysis.plans,
                true,
            )?;
            for region in &analysis.preserved {
                if input.request.retry_item_ids.is_empty()
                    || input.request.retry_item_ids.contains(&region.item_id)
                {
                    sink.publish(JobUpdateDraft::ImageRegionPreserved {
                        region: region.clone(),
                    })
                    .map_err(|error| publish_error(error, &sink))?;
                }
            }
            let viewport = sink.focus();
            let mut regions = analysis.regions.clone();
            for region in &mut regions {
                region.visible = viewport.active
                    && region.candidate.bubble_rect.intersects_viewport(
                        &viewport.visible_rects,
                        image_width,
                        image_height,
                    );
                region.translation_queued_at = tokio::time::Instant::now();
            }
            self.flush_translation_queue(
                resident,
                control,
                &input.request,
                &mut regions,
                cancel,
                &sink,
                0.80,
                image_width,
                image_height,
                &mut Vec::new(),
                &mut TranslationLatencyPhase::AwaitingFirstVisibleRegion,
                true,
            )
            .await?;
            return Ok(());
        }
        let preprocessing = global_preprocessing_pool().map_err(PipelineError::pipeline)?;
        cancellation_boundary(cancel.as_ref())?;

        let mut tiles = overlapping_tiles(image_width, image_height);
        let total_tiles = tiles.len();
        let total_tiles_u32 = u32::try_from(total_tiles).unwrap_or(u32::MAX);
        let mut processed_tiles = 0usize;
        let mut seen_text_blocks = Vec::<PixelRect>::new();
        let mut recognized_lines = Vec::<RecognizedLine>::new();
        let mut text_probabilities = ProbabilityMap::zeros(image_width, image_height);
        let mut pending_translation = Vec::<PreparedRegion>::new();
        let mut analyzed_regions = Vec::<PreparedRegion>::new();
        let mut translation_latency_phase = TranslationLatencyPhase::AwaitingFirstVisibleRegion;
        // The daemon chapter session supplies preceding dialogue immediately
        // before each translation window. Never seed it from browser fields.
        let mut dialogue_context = Vec::new();
        let mut prepared_next_tiles: Option<TileBatchTask> = None;
        // OCR proposals that do not survive the two-view consensus gate are
        // retained as evidence until the page reaches its terminal commit.
        // They must become source-preserving unreadable regions rather than
        // disappearing from the chapter coverage graph.
        let mut rejected_ocr_lines = Vec::<RejectedOcrLine>::new();
        let mut page_region_plans = Vec::<RegionPlan>::new();
        // A tall reader page often yields several detector frontiers. Keep
        // non-visible lines together so semantic classification, name
        // adjudication, and inpainting run once for the page tail instead of
        // once per frontier. The currently visible frontier still takes the
        // fast path below.
        let mut deferred_page_lines = Vec::<RecognizedLine>::new();
        let mut bubble_masks = BubbleMaskCache::new(image_width, image_height);
        let mut text_mask_completed_tiles = HashSet::<usize>::new();

        while !tiles.is_empty() {
            cancellation_boundary(cancel.as_ref())?;
            if sink.is_cancelled() {
                return Err(PipelineError::cancelled());
            }
            let viewport = sink.focus();
            prioritize_tiles(
                &mut tiles,
                &viewport.visible_rects,
                viewport.active,
                image_width,
                image_height,
                input.request.reading_direction,
            );
            let take = next_detector_batch_count(
                &tiles,
                &viewport.visible_rects,
                viewport.active,
                image_width,
                image_height,
                DETECTOR_TILE_BATCH_SIZE,
            );
            let use_prepared = prepared_next_tiles
                .as_ref()
                .is_some_and(|task| tiles_start_with(&tiles, &task.tiles));
            let (tile_batch, tile_images) = if use_prepared {
                let task = prepared_next_tiles
                    .take()
                    .expect("prepared tile task was just inspected");
                tiles.drain(..task.tiles.len());
                task.finish()
                    .await
                    .context("finish speculative detector tile crops")
                    .map_err(PipelineError::pipeline)?
            } else {
                // A new viewport can invalidate the speculative offscreen
                // choice. Dropping its receiver lets visible work overtake it
                // at this tile boundary without waiting for the stale crop.
                prepared_next_tiles.take();
                let tile_batch = tiles.drain(..take).collect::<Vec<_>>();
                TileBatchTask::start(preprocessing.as_ref(), source.clone(), tile_batch)
                    .finish()
                    .await
                    .context("prepare detector tile crops on the browser preprocessing pool")
                    .map_err(PipelineError::pipeline)?
            };
            let overall = batch_overall_progress(processed_tiles, total_tiles);
            publish_progress(
                &sink,
                BrowserJobStage::Detecting,
                Some(processed_tiles as f32 / total_tiles.max(1) as f32),
                Some(overall),
                Some(u32::try_from(processed_tiles).unwrap_or(u32::MAX)),
                Some(total_tiles_u32),
                "Detecting English story text in the next tile batch",
            )?;

            cancellation_boundary(cancel.as_ref())?;
            let admission_viewport = sink.focus();
            prioritize_tiles(
                &mut tiles,
                &admission_viewport.visible_rects,
                admission_viewport.active,
                image_width,
                image_height,
                input.request.reading_direction,
            );
            if !tiles.is_empty() {
                let next_count = next_detector_batch_count(
                    &tiles,
                    &admission_viewport.visible_rects,
                    admission_viewport.active,
                    image_width,
                    image_height,
                    DETECTOR_TILE_BATCH_SIZE,
                );
                prepared_next_tiles = Some(TileBatchTask::start(
                    preprocessing.as_ref(),
                    source.clone(),
                    tiles[..next_count].to_vec(),
                ));
            }
            let batch_is_visible = admission_viewport.active
                && tile_batch.iter().any(|tile| {
                    let tile_rect = NormalizedRect {
                        x: tile.x as f32 / image_width.max(1) as f32,
                        y: tile.y as f32 / image_height.max(1) as f32,
                        width: tile.width as f32 / image_width.max(1) as f32,
                        height: tile.height as f32 / image_height.max(1) as f32,
                    };
                    admission_viewport
                        .visible_rects
                        .iter()
                        .any(|visible| normalized_rects_intersect(&tile_rect, visible))
                });
            let detector_priority = if batch_is_visible {
                CudaPriority::Visible
            } else {
                CudaPriority::Offscreen
            };
            let detector_started = Instant::now();
            let (detections, ocr_detections) = {
                // Lexical ownership makes the detector phase incapable of
                // retaining CUDA admission across downstream dispatch.
                let _detector_permit = self
                    .cuda_scheduler
                    .acquire(CudaWorkload::Vision, detector_priority, cancel.clone())
                    .await
                    .map_err(cuda_admission_error)?;
                resident.instrumentation.invoked(RuntimeComponent::Vision);
                let detector = resident.detector.lock().map_err(|_| {
                    PipelineError::new("MODEL_STATE_FAILED", "Detector lock poisoned.")
                })?;
                let detections = detector
                    .inference_tiles(&tile_images)
                    .context("run true-batched CUDA comic text detection")
                    .map_err(PipelineError::pipeline)?;
                resident.instrumentation.invoked(RuntimeComponent::Ocr);
                let ocr_detections = resident
                    .ocr_detector
                    .lock()
                    .map_err(|_| {
                        PipelineError::new("MODEL_STATE_FAILED", "OCR detector lock poisoned.")
                    })?
                    .detect_tiles(&tile_images)
                    .context("run true-batched CUDA PP-OCR text detection")
                    .map_err(PipelineError::pipeline)?;
                (detections, ocr_detections)
            };
            let detector_elapsed = detector_started.elapsed();
            cancellation_boundary(cancel.as_ref())?;
            if detections.len() != tile_batch.len() || ocr_detections.len() != tile_batch.len() {
                return Err(PipelineError::new(
                    "DETECTION_FAILED",
                    "Detector returned an incomplete tile batch.",
                ));
            }
            // Reconsider queued translation immediately after the detector CUDA
            // batch, before CPU postprocessing or any offscreen OCR admission.
            self.flush_translation_queue(
                resident,
                control,
                &input.request,
                &mut pending_translation,
                cancel.clone(),
                &sink,
                overall,
                image_width,
                image_height,
                &mut dialogue_context,
                &mut translation_latency_phase,
                false,
            )
            .await?;
            let comic_bubbles = detections
                .iter()
                .zip(&tile_batch)
                .map(|(detection, tile)| bubbles_for_tile(detection, tile))
                .collect::<Vec<_>>();
            // PP-OCR supplies preferred line geometry. The already-resident
            // comic detector supplies an independent recovery stream for text
            // that PP-OCR misses; its wider object boxes never replace a
            // covered PP-OCR line.
            let ppocr_candidates = ocr_detections
                .iter()
                .zip(&comic_bubbles)
                .zip(&tile_batch)
                .flat_map(|((detection, bubbles), tile)| {
                    candidates_for_text_boxes(detection, bubbles, tile, image_width, image_height)
                })
                .collect::<Vec<_>>();
            let mut ppocr_candidates = spatially_dedupe(ppocr_candidates, &seen_text_blocks);
            ppocr_candidates.retain(text_candidate_is_confirmed);
            let preferred_rects = seen_text_blocks
                .iter()
                .copied()
                .chain(ppocr_candidates.iter().map(|candidate| candidate.text_rect))
                .collect::<Vec<_>>();
            let comic_candidates = detections
                .iter()
                .zip(&comic_bubbles)
                .zip(&tile_batch)
                .flat_map(|((detection, bubbles), tile)| {
                    candidates_for_comic_text_boxes(
                        detection,
                        bubbles,
                        tile,
                        image_width,
                        image_height,
                    )
                })
                .collect::<Vec<_>>();
            let mut comic_candidates = spatially_dedupe(comic_candidates, &preferred_rects);
            comic_candidates.retain(text_candidate_is_confirmed);
            let mut candidates = ppocr_candidates;
            candidates.extend(comic_candidates);
            let mask_started = Instant::now();
            let regions = candidates
                .iter()
                .map(|candidate| candidate.text_rect)
                .collect::<Vec<_>>();
            merge_source_guided_glyph_probabilities(
                source
                    .as_rgb8()
                    .expect("browser source images are canonical RGB"),
                &mut text_probabilities,
                &regions,
            );
            if std::env::var_os("HSKIFY_TRACE_PIPELINE_TIMING").is_some_and(|value| value == "1") {
                eprintln!(
                    "hskify-vision-timing detector_ms={} mask_ms={} tiles={} mask=source-consensus",
                    detector_elapsed.as_millis(),
                    mask_started.elapsed().as_millis(),
                    tile_batch.len(),
                );
            }
            if rejected_ocr_tracing_enabled() {
                for candidate in &candidates {
                    eprintln!(
                        "hskify-detector-candidate source={} kind={:?} rect={:.1},{:.1},{:.1},{:.1} confidence={:.4}",
                        &input.request.source_sha256[..8],
                        candidate.kind,
                        candidate.text_rect.x0,
                        candidate.text_rect.y0,
                        candidate.text_rect.x1,
                        candidate.text_rect.y1,
                        candidate.detector_confidence,
                    );
                }
            }
            if !candidates.is_empty() {
                publish_progress(
                    &sink,
                    BrowserJobStage::Ocr,
                    None,
                    Some(overall),
                    None,
                    None,
                    "Reading English story text in OCR batches of eight",
                )?;
            }
            let mut masked_candidates = candidates;
            while !masked_candidates.is_empty() {
                let ocr_result = ocr_batch(
                    resident,
                    source.clone(),
                    &mut masked_candidates,
                    OcrProposalSource::Detector,
                    &input.request,
                    &sink,
                    cancel.clone(),
                    &self.cuda_scheduler,
                    &preprocessing,
                    &text_probabilities,
                )
                .await?;
                for line in ocr_result.accepted {
                    seen_text_blocks.push(line.candidate.text_rect);
                    recognized_lines.push(line);
                }
                rejected_ocr_lines.extend(ocr_result.rejected);
            }
            processed_tiles += tile_batch.len();
            if !tiles.is_empty() {
                let finalized_rejected = take_finalized_rejected_lines(
                    &mut rejected_ocr_lines,
                    &tiles,
                    image_width,
                    image_height,
                );
                page_region_plans.extend(publish_rejected_ocr_regions(
                    &finalized_rejected,
                    &seen_text_blocks,
                    &input.request,
                    image_width,
                    image_height,
                    &sink,
                )?);
                let finalized_lines =
                    take_finalized_lines(&mut recognized_lines, &tiles, image_width, image_height);
                if finalized_lines.is_empty() {
                    continue;
                }
                let viewport = sink.focus();
                let mut immediate_lines = Vec::new();
                for line in finalized_lines {
                    let immediate = viewport.active
                        && line.candidate.bubble_rect.intersects_viewport(
                            &viewport.visible_rects,
                            image_width,
                            image_height,
                        );
                    if immediate {
                        immediate_lines.push(line);
                    } else {
                        deferred_page_lines.push(line);
                    }
                }
                // Without an active viewport (for example, a background
                // import), defer everything and perform one page-level pass
                // after detection. Interactive readers always have a visible
                // frontier and therefore retain the low-latency path.
                if immediate_lines.is_empty() {
                    continue;
                }
                let (prepared_regions, probabilities, region_plans) = prepare_grouped_regions(
                    Arc::clone(resident),
                    source.clone(),
                    immediate_lines,
                    &input.request,
                    &dialogue_context,
                    &sink,
                    cancel.clone(),
                    &self.cuda_scheduler,
                    &preprocessing,
                    &mut bubble_masks,
                    &mut text_mask_completed_tiles,
                    text_probabilities,
                    overall,
                )
                .await?;
                text_probabilities = probabilities;
                page_region_plans.extend(region_plans);
                self.record_page_analysis(
                    &input.request,
                    image_width,
                    image_height,
                    surface_kind.clone(),
                    &page_region_plans,
                    false,
                )?;
                analyzed_regions.extend(prepared_regions.iter().cloned());
                pending_translation.extend(prepared_regions);
                // Multi-tile pages remain independently analyzable. The
                // final page pass joins lines whose bubble ownership was not
                // yet closed by the canonical tile frontier.
                self.flush_translation_queue(
                    resident,
                    control,
                    &input.request,
                    &mut pending_translation,
                    cancel.clone(),
                    &sink,
                    overall,
                    image_width,
                    image_height,
                    &mut dialogue_context,
                    &mut translation_latency_phase,
                    false,
                )
                .await?;
            }
            cancellation_boundary(cancel.as_ref())?;
        }

        recognized_lines.extend(deferred_page_lines);
        if !recognized_lines.is_empty() {
            let finalized_lines = std::mem::take(&mut recognized_lines);
            let (prepared_regions, _, region_plans) = prepare_grouped_regions(
                Arc::clone(resident),
                source.clone(),
                finalized_lines,
                &input.request,
                &dialogue_context,
                &sink,
                cancel.clone(),
                &self.cuda_scheduler,
                &preprocessing,
                &mut bubble_masks,
                &mut text_mask_completed_tiles,
                text_probabilities,
                0.78,
            )
            .await?;
            page_region_plans.extend(region_plans);
            self.record_page_analysis(
                &input.request,
                image_width,
                image_height,
                surface_kind.clone(),
                &page_region_plans,
                false,
            )?;
            analyzed_regions.extend(prepared_regions.iter().cloned());
            pending_translation.extend(prepared_regions);
        }

        self.flush_translation_queue(
            resident,
            control,
            &input.request,
            &mut pending_translation,
            cancel.clone(),
            &sink,
            0.80,
            image_width,
            image_height,
            &mut dialogue_context,
            &mut translation_latency_phase,
            true,
        )
        .await?;
        let unreadable_ocr_plans = publish_rejected_ocr_regions(
            &rejected_ocr_lines,
            &seen_text_blocks,
            &input.request,
            image_width,
            image_height,
            &sink,
        )?;
        page_region_plans.extend(unreadable_ocr_plans);
        let mut region_plans = BTreeMap::<String, RegionPlan>::new();
        for plan in page_region_plans {
            region_plans.entry(plan.id.clone()).or_insert(plan);
        }
        let region_plans = region_plans.into_values().collect::<Vec<_>>();
        self.record_page_analysis(
            &input.request,
            image_width,
            image_height,
            surface_kind,
            &region_plans,
            true,
        )?;
        if input.request.retry_item_ids.is_empty()
            && analyzed_regions.iter().all(|region| {
                region.cleanup.result.get().is_some_and(|result| {
                    result
                        .decisions
                        .get(&region.id)
                        .is_some_and(|decision| decision.patch.is_some())
                })
            })
        {
            let preserved = sink
                .preserved_regions()
                .into_iter()
                .filter(|region| {
                    !analyzed_regions
                        .iter()
                        .any(|prepared| prepared.id == region.item_id)
                })
                .collect();
            self.analysis_cache
                .lock()
                .map_err(|_| {
                    PipelineError::new("CACHE_FAILED", "Page analysis cache lock poisoned.")
                })?
                .insert(
                    analysis_key,
                    Arc::new(AnalyzedPage {
                        regions: analyzed_regions,
                        plans: region_plans,
                        preserved,
                    }),
                );
        }
        publish_progress(
            &sink,
            BrowserJobStage::Packaging,
            Some(1.0),
            Some(0.98),
            None,
            None,
            "All region-local patches and translations are published",
        )?;
        Ok(())
    }
}

impl ImagePipeline<'_> {
    #[allow(clippy::too_many_arguments)]
    async fn flush_translation_queue(
        &self,
        resident: &VisionRuntime,
        control: &HskControl,
        request: &ImagePipelineInput,
        pending: &mut Vec<PreparedRegion>,
        cancel: Arc<AtomicBool>,
        sink: &JobUpdateSink,
        overall_progress: f32,
        image_width: u32,
        image_height: u32,
        context: &mut Vec<HskPrecedingUtterance>,
        latency_phase: &mut TranslationLatencyPhase,
        force: bool,
    ) -> std::result::Result<(), PipelineError> {
        while !pending.is_empty() {
            prioritize_pending_translation(pending, sink, image_width, image_height);
            let count = match translation_boundary_action(
                pending,
                force,
                cancel.load(Ordering::Acquire) || sink.is_cancelled(),
                *latency_phase == TranslationLatencyPhase::AwaitingFirstVisibleRegion
                    && pending.first().is_some_and(|region| region.visible),
            ) {
                TranslationBoundaryAction::ContinueUpstream => {
                    // Sparse offscreen work waits for more regions or the
                    // final flush. Visible work never takes this branch.
                    return Ok(());
                }
                TranslationBoundaryAction::Dispatch(count) => count,
                TranslationBoundaryAction::Cancelled => {
                    return Err(PipelineError::cancelled());
                }
            };
            if !(1..=TRANSLATION_BATCH_MAX).contains(&count) {
                return Err(PipelineError::new(
                    "TRANSLATION_BATCH_FAILED",
                    "Translation batching produced an invalid microbatch size.",
                ));
            }
            let mut batch = pending.drain(..count).collect::<Vec<_>>();
            let mut following_english = pending
                .iter()
                .map(|region| (region.reading_order, region.source_english.clone()))
                .collect::<Vec<_>>();
            following_english.sort_by_key(|(reading_order, _)| *reading_order);
            let mut following_english = following_english
                .into_iter()
                .take(MAX_HSK_PRECEDING_UTTERANCES)
                .map(|(_, source)| source)
                .collect::<Vec<_>>();
            // A page can be analyzed ahead of the ordered language stream.
            // Add its canonical source-language look-ahead after the current
            // microbatch so connected bubbles on the next page have context,
            // while never leaking a future translation/entity decision.
            let last_reading_order = batch
                .iter()
                .map(|region| region.reading_order)
                .max()
                .unwrap_or_default();
            let chapter_following = self
                .chapter_contexts
                .lock()
                .map_err(|_| {
                    PipelineError::new("CHAPTER_CONTEXT_FAILED", "Chapter context lock poisoned.")
                })?
                .following_source(
                    &request.page_session_id,
                    (request.source_index, last_reading_order),
                    MAX_HSK_PRECEDING_UTTERANCES,
                );
            for source in chapter_following {
                if following_english.len() >= MAX_HSK_PRECEDING_UTTERANCES {
                    break;
                }
                if !following_english
                    .iter()
                    .any(|existing| existing.eq_ignore_ascii_case(&source))
                {
                    following_english.push(source);
                }
            }
            // Visibility determines which window is admitted first, never the
            // order of language context inside that window. Once admitted,
            // every generation and terminal graph commit is canonical page /
            // reading order so connected bubbles cannot inherit a completion-
            // race or a viewport-priority ordering.
            batch.sort_by_key(|region| region.reading_order);
            let primary_published_visible = self
                .translate_and_publish_shared(
                    resident,
                    control,
                    request,
                    batch,
                    cancel.clone(),
                    sink,
                    overall_progress,
                    image_width,
                    image_height,
                    context,
                    &following_english,
                )
                .await?;
            // Final-only rendering may withhold a rejected or malformed
            // primary. Interactive admission remains reserved until either
            // the primary or its terminal repair has actually published a
            // visible final region; merely attempting generation is not a
            // user-visible milestone.
            complete_translation_batch(latency_phase, primary_published_visible);
        }
        Ok(())
    }

    fn remember_terminal_context(
        &self,
        request: &ImagePipelineInput,
        region: &PreparedRegion,
        chinese: &str,
    ) -> std::result::Result<(), PipelineError> {
        if region.source_english.trim().is_empty() || chinese.trim().is_empty() {
            return Ok(());
        }
        self.chapter_contexts
            .lock()
            .map_err(|_| {
                PipelineError::new("CHAPTER_CONTEXT_FAILED", "Chapter context lock poisoned.")
            })?
            .publish(
                &request.page_session_id,
                (request.source_index, region.reading_order),
                &region.id,
                chinese.to_owned(),
            );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn translate_and_publish_shared(
        &self,
        resident: &VisionRuntime,
        control: &HskControl,
        request: &ImagePipelineInput,
        regions: Vec<PreparedRegion>,
        cancel: Arc<AtomicBool>,
        sink: &JobUpdateSink,
        overall_progress: f32,
        image_width: u32,
        image_height: u32,
        context: &mut Vec<HskPrecedingUtterance>,
        _following_english: &[String],
    ) -> std::result::Result<bool, PipelineError> {
        let regions = regions
            .into_iter()
            .filter(|region| {
                request.retry_item_ids.is_empty() || request.retry_item_ids.contains(&region.id)
            })
            .collect::<Vec<_>>();
        if regions.is_empty() {
            return Ok(false);
        }
        if regions.len() > TRANSLATION_BATCH_MAX {
            return Err(PipelineError::new(
                "TRANSLATION_BATCH_FAILED",
                "Language microbatches are limited to six source spans.",
            ));
        }
        publish_progress(
            sink,
            BrowserJobStage::Translating,
            None,
            Some(overall_progress),
            None,
            None,
            "Producing final Chinese for verified image text",
        )?;
        let priority = prepared_region_priority(&regions, sink, image_width, image_height);
        let units = regions
            .iter()
            .filter(|region| {
                request.retry_item_ids.is_empty() || request.retry_item_ids.contains(&region.id)
            })
            .map(|region| {
                let (max_characters, max_lines) =
                    layout_budget_for_region(region, image_width, image_height);
                TranslationUnit {
                    id: region.id.clone(),
                    kind: hsk_utterance_kind_for_region(region),
                    source_text: region.source_english.clone(),
                    faithful_chinese: region.faithful_chinese.clone(),
                    provenance: DirectSourceProvenance::Ocr,
                    layout: Some(HskLayoutConstraints {
                        max_characters,
                        max_lines,
                    }),
                    source_context: region.source_context.clone(),
                }
            })
            .collect::<Vec<_>>();
        let translation_batch = self
            .translation_service(&resident.language, control)
            .translate_units(
                &units,
                request.settings.hsk_level,
                request.settings.learning_mode,
                priority,
                cancel.clone(),
            )
            .await?;
        let translator = resident.language.app.llm.direct_hsk_translator();
        record_image_language_evidence(
            sink.job_id(),
            format!(
                "{}@{}:native-llama-tokenizer",
                translator.model_id(),
                translator.model_revision()
            ),
            0,
            translation_batch.generation_duration,
        )
        .map_err(PipelineError::pipeline)?;
        let translated = translation_batch.items;
        cancellation_boundary(cancel.as_ref())?;
        let mut published_visible = false;
        for region in regions {
            let translation = match translated.get(&region.id) {
                Some(Ok(value)) => value.clone(),
                Some(Err(reason)) => {
                    publish_unreadable_prepared(
                        sink,
                        &region,
                        request,
                        image_width,
                        image_height,
                        reason,
                    )?;
                    continue;
                }
                None => {
                    publish_unreadable_prepared(
                        sink,
                        &region,
                        request,
                        image_width,
                        image_height,
                        "The language service omitted this source span; source pixels were preserved.",
                    )?;
                    continue;
                }
            };
            let cleanup = region.cleanup.result().await;
            let Some(decision) = cleanup.decisions.get(&region.id) else {
                publish_unreadable_prepared(
                    sink,
                    &region,
                    request,
                    image_width,
                    image_height,
                    "Cleanup did not produce a verified patch; source pixels were preserved.",
                )?;
                continue;
            };
            if decision.patch.is_none() {
                publish_unreadable_prepared(
                    sink,
                    &region,
                    request,
                    image_width,
                    image_height,
                    decision.reason.as_deref().unwrap_or(
                        "Cleanup verification did not pass; source pixels were preserved.",
                    ),
                )?;
                continue;
            }
            publish_region(
                sink,
                &region,
                decision,
                translation.clone(),
                request.settings.hsk_level,
                request.settings.learning_mode,
                control,
                image_width,
                image_height,
            )?;
            published_visible |= region.visible;
            self.remember_terminal_context(request, &region, &translation.displayed_chinese)?;
            append_terminal_context(
                context,
                &region.source_english,
                &translation.displayed_chinese,
            );
        }
        Ok(published_visible)
    }
}

fn page_surface_kind(kind: BrowserSurfaceKind, width: u32, height: u32) -> PageSurfaceKind {
    match kind {
        BrowserSurfaceKind::Image if (height as f64) > (width as f64) * 2.5 => {
            PageSurfaceKind::ContinuousStrip
        }
        BrowserSurfaceKind::Image => PageSurfaceKind::Image,
        BrowserSurfaceKind::Background => PageSurfaceKind::Image,
        BrowserSurfaceKind::Canvas => PageSurfaceKind::Canvas,
        BrowserSurfaceKind::Webgl => PageSurfaceKind::WebGl,
        BrowserSurfaceKind::Frame => PageSurfaceKind::Frame,
    }
}

#[async_trait]
impl ChapterPipeline for HskifyPipeline {
    async fn warm_up(&self, kind: ChapterKind) -> std::result::Result<(), PipelineError> {
        match kind {
            ChapterKind::Document => self
                .ready_language()
                .await
                .map(|_| ())
                .map_err(PipelineError::pipeline),
            ChapterKind::Image => self
                .ready_models()
                .await
                .map(|_| ())
                .map_err(PipelineError::pipeline),
        }
    }

    async fn run_image(
        &self,
        input: ImageJobInput,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), PipelineError> {
        let request = input.request.clone();
        let result = self.image_pipeline().run(input, cancel, sink).await;
        if result.is_err() {
            // A failed/cancelled page is terminal for chapter ordering too;
            // later admitted pages must not wait forever for analysis or
            // language context that can never be produced. Successful pages
            // keep the richer complete analysis written by run_direct.
            let _ = self.mark_page_terminal(&request);
        }
        result
    }

    async fn run_document(
        &self,
        request: DocumentJobRequest,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), PipelineError> {
        self.document_pipeline().run(request, cancel, sink).await
    }

    fn mark_page_terminal(
        &self,
        request: &ImagePipelineInput,
    ) -> std::result::Result<(), PipelineError> {
        let kind = page_surface_kind(
            request.surface_kind,
            request.natural_width,
            request.natural_height,
        );
        self.image_pipeline().record_page_analysis(
            request,
            request.natural_width,
            request.natural_height,
            kind,
            &[],
            true,
        )?;
        Ok(())
    }

    fn image_cache_context(&self, request: &ImagePipelineInput) -> Vec<ChapterContextUnit> {
        self.chapter_contexts
            .lock()
            .map(|contexts| {
                contexts
                    .snapshot_excluding_source(
                        &request.page_session_id,
                        request.source_index,
                        &request.chapter_source_order,
                    )
                    .into_iter()
                    .map(|mut unit| {
                        unit.displayed_text = None;
                        unit
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn restore_cached_context(
        &self,
        request: &ImagePipelineInput,
        regions: &[ImageRegionReady],
        preserved_regions: &[ImageRegionPreserved],
    ) -> std::result::Result<(), PipelineError> {
        let mut sessions = self.chapter_sessions.lock().map_err(|_| {
            PipelineError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
        })?;
        let session = sessions.session_mut(&request.page_session_id);
        let changed = session.register_surface(PageSurface {
            session_id: request.page_session_id.clone(),
            page_index: request.source_index,
            source_sha256: request.source_sha256.clone(),
            width: request.natural_width,
            height: request.natural_height,
            kind: page_surface_kind(
                request.surface_kind,
                request.natural_width,
                request.natural_height,
            ),
        });

        let mut plans = BTreeMap::<String, RegionPlan>::new();
        for region in regions {
            plans.insert(
                region.item_id.clone(),
                RegionPlan {
                    id: region.item_id.clone(),
                    reading_order: region.item_order,
                    role: match region.kind {
                        SourceSpanKind::Dialogue => RegionRole::Dialogue,
                        SourceSpanKind::Sfx => RegionRole::System,
                        SourceSpanKind::Prose
                        | SourceSpanKind::Heading
                        | SourceSpanKind::Caption
                        | SourceSpanKind::Thought => RegionRole::Narration,
                    },
                    source_english: region.text.source_text.clone(),
                    continuation_group: region.context_group.clone(),
                },
            );
        }
        for region in preserved_regions {
            plans.insert(
                region.item_id.clone(),
                RegionPlan {
                    id: region.item_id.clone(),
                    reading_order: region.item_order,
                    role: if region.reason == "artwork-preserved" {
                        RegionRole::TechniqueArtwork
                    } else {
                        RegionRole::Unreadable
                    },
                    source_english: region.source_text.clone(),
                    continuation_group: None,
                },
            );
        }
        let surface = session
            .surfaces
            .get(&request.source_index)
            .cloned()
            .expect("cached page surface was registered above");
        session.record_analysis(PageAnalysis {
            surface,
            regions: plans.into_values().collect(),
            complete: true,
        });
        drop(sessions);
        let mut contexts = self.chapter_contexts.lock().map_err(|_| {
            PipelineError::new("CHAPTER_CONTEXT_FAILED", "Chapter context lock poisoned.")
        })?;
        if changed {
            contexts.remove_source(&request.page_session_id, request.source_index);
        }
        contexts.register(
            &request.page_session_id,
            regions.iter().map(|region| ChapterContextUnit {
                item_id: region.item_id.clone(),
                source_index: request.source_index,
                item_order: region.item_order,
                source_text: region.text.source_text.clone(),
                displayed_text: Some(region.text.displayed_chinese.clone()),
            }),
        );
        Ok(())
    }

    fn close_chapter(&self, page_session_id: &str) {
        if let Ok(mut sessions) = self.chapter_sessions.lock() {
            sessions.remove(page_session_id);
        }
        if let Ok(mut contexts) = self.chapter_contexts.lock() {
            contexts.remove(page_session_id);
        }
    }

    async fn lookup(
        &self,
        input: LookupInput,
        item: Option<ItemLookupContext>,
    ) -> std::result::Result<LookupResult, PipelineError> {
        let control = self
            .hsk_control()
            .await
            .map_err(|error| PipelineError::new("RESOURCES_NOT_READY", format!("{error:#}")))?;
        let (proper_names, context) = match item {
            Some(item) => (
                item.proper_names,
                Some(ControlLookupItem {
                    displayed_chinese: item.displayed_chinese,
                    base_chinese: item.base_chinese,
                    source_text: item.source_text,
                }),
            ),
            None => (Vec::new(), None),
        };
        let result = match input {
            LookupInput::Selection(selected_text) => {
                control.lookup_with_item_context(&selected_text, &proper_names, context)
            }
            LookupInput::Hover {
                displayed_text,
                character_offset,
            } => {
                let hovered_character = displayed_text
                    .chars()
                    .nth(character_offset)
                    .expect("server validates hover offsets")
                    .to_string();
                control
                    .lookup_at_with_item_context(
                        &displayed_text,
                        character_offset,
                        &proper_names,
                        context.clone(),
                    )
                    .unwrap_or(hsk_control::LookupResult {
                        selected_text: hovered_character,
                        tokens: Vec::new(),
                        item: context,
                    })
            }
        };
        Ok(browser_lookup_result(result))
    }

    fn record_font_invocation(&self) {
        self.instrumentation.invoked(RuntimeComponent::Font);
    }

    fn resources_ready(&self) -> bool {
        self.language.get().is_some()
    }
}

/// Language-only resident state. Document jobs stop here: no detector, OCR,
/// projector, segmenter, inpainter, patch, or font path is touched.
struct LanguageRuntime {
    runtime: Arc<RuntimeManager>,
    app: Arc<App>,
    hsk_control: Arc<HskControl>,
    translation_cache: Mutex<TranslationCache>,
    faithful_cache: Mutex<TranslationCache<FaithfulText>>,
}

impl LanguageRuntime {
    async fn load(
        runtime_root: PathBuf,
        app_state_root: PathBuf,
        resources: ResidentResourcePaths,
    ) -> Result<Self> {
        koharu_runtime::require_hskify_cuda_target()
            .context("language model load requires the exact Hskify CUDA target")?;
        let runtime = Arc::new(
            RuntimeManager::new(&runtime_root, ComputePolicy::CudaRequired)
                .context("initialize language runtime")?,
        );
        runtime
            .prepare()
            .await
            .context("prepare language runtime")?;
        let mut config = AppConfig::default();
        config.data.path = utf8_path(app_state_root)?;
        let app = Arc::new(
            App::new(config, runtime.clone(), false, env!("CARGO_PKG_VERSION"))
                .context("initialize language application state")?,
        );
        let translation_model = resources.path(TRANSLATION_MODEL_ID)?.to_path_buf();
        let hsk_path = resources.hsk.clone();
        let dictionary_path = resources.dictionary.clone();
        let hsk_future = tokio::task::spawn_blocking(move || {
            let hsk_json = std::fs::read_to_string(&hsk_path)
                .with_context(|| format!("read HSK data {}", hsk_path.display()))?;
            let dictionary_json = std::fs::read_to_string(&dictionary_path)
                .with_context(|| format!("read dictionary data {}", dictionary_path.display()))?;
            HskControl::from_json(&hsk_json, &dictionary_json)
                .context("load deterministic HSK control data")
                .map(Arc::new)
        });
        let model_future = app.llm.load_local_file_with_threads(
            HSK_TRANSLATION_MODEL,
            translation_model,
            BROWSER_QWEN_INFERENCE_THREADS,
        );
        let ((), hsk_control) =
            tokio::try_join!(async { model_future.await.map(|_| ()) }, async {
                hsk_future.await.context("join HSK data loader")?
            },)?;
        Ok(Self {
            runtime,
            app,
            hsk_control,
            translation_cache: Mutex::new(TranslationCache::default()),
            faithful_cache: Mutex::new(TranslationCache::default()),
        })
    }

    fn prime(&self) -> Result<()> {
        let cancel = AtomicBool::new(false);
        let translator = self.app.llm.direct_hsk_translator();
        tokio::runtime::Handle::current().block_on(async {
            translator
                .warm_up(&cancel)
                .await
                .context("prime direct HSK translation inference")
        })
    }
}

struct VisionRuntime {
    language: Arc<LanguageRuntime>,
    instrumentation: Arc<RuntimeInstrumentation>,
    detector: Mutex<ComicTextBubbleDetector>,
    ocr_detector: Mutex<PpOcrSmallDetector>,
    ocr: Mutex<PpOcrSmallRecognizer>,
    text_segmenter: Mutex<MangaTextSegmentation>,
    bubble_segmenter: Mutex<SpeechBubbleSegmentation>,
    inpainter: Mutex<Lama>,
    page_understanding: Mutex<QwenPageUnderstanding>,
}

impl VisionRuntime {
    async fn load(
        language: Arc<LanguageRuntime>,
        resources: ResidentResourcePaths,
        instrumentation: Arc<RuntimeInstrumentation>,
    ) -> Result<Self> {
        let runtime = Arc::clone(&language.runtime);
        let app = Arc::clone(&language.app);
        let detector_config = resources.path(DETECTOR_CONFIG_ID)?.to_path_buf();
        let detector_preprocessor = resources.path(DETECTOR_PREPROCESSOR_ID)?.to_path_buf();
        let detector_weights = resources.path(DETECTOR_WEIGHTS_ID)?.to_path_buf();
        let ocr_config = resources.path(OCR_CONFIG_ID)?.to_path_buf();
        let ocr_model = resources.path(OCR_MODEL_ID)?.to_path_buf();
        let ocr_detector_config = resources.path(OCR_DETECTOR_CONFIG_ID)?.to_path_buf();
        let ocr_detector_model = resources.path(OCR_DETECTOR_MODEL_ID)?.to_path_buf();
        let text_segmenter_weights = resources.path(TEXT_SEGMENTER_WEIGHTS_ID)?.to_path_buf();
        let bubble_segmenter_config = resources.path(BUBBLE_SEGMENTER_CONFIG_ID)?.to_path_buf();
        let bubble_segmenter_weights = resources.path(BUBBLE_SEGMENTER_WEIGHTS_ID)?.to_path_buf();
        let inpainter_weights = resources.path(INPAINTER_WEIGHTS_ID)?.to_path_buf();
        let translation_model = resources.path(TRANSLATION_MODEL_ID)?.to_path_buf();
        let page_projector_path = resources.path(PAGE_PROJECTOR_ID)?.to_path_buf();
        let page_capability =
            probe_qwen_page_understanding(&translation_model, &page_projector_path);
        let detector_future = async move {
            ComicTextBubbleDetector::load_from_paths(
                detector_config,
                detector_preprocessor,
                detector_weights,
                false,
            )
            .await
            .context("load resident comic text detector")
        };
        let ocr_future = async move {
            tokio::task::spawn_blocking(move || PpOcrSmallRecognizer::load(&ocr_model, &ocr_config))
                .await
                .context("join resident PP-OCR small recognizer loader")?
        };
        let ocr_detector_future = async move {
            tokio::task::spawn_blocking(move || {
                PpOcrSmallDetector::load(&ocr_detector_model, &ocr_detector_config)
            })
            .await
            .context("join resident PP-OCR small detector loader")?
        };
        let cleanup_models_future = async move {
            tokio::task::spawn_blocking(move || {
                let text_segmenter =
                    MangaTextSegmentation::load_from_path(text_segmenter_weights, false)
                        .context("load resident manga text segmenter")?;
                let bubble_segmenter = SpeechBubbleSegmentation::load_from_paths(
                    bubble_segmenter_config,
                    bubble_segmenter_weights,
                    false,
                )
                .context("load resident speech bubble segmenter")?;
                let inpainter = Lama::load_from_path(inpainter_weights, false)
                    .context("load resident manga inpainter")?;
                Ok::<_, anyhow::Error>((text_segmenter, bubble_segmenter, inpainter))
            })
            .await
            .context("join resident cleanup model loader")?
        };
        let page_model_runtime = runtime.clone();
        let page_model_backend = app.llm.backend();
        let page_capability_for_load = page_capability.clone();
        let page_app = app.clone();
        let page_future = async move {
            if !page_capability_for_load.is_available() {
                return Err(anyhow!(
                    "resident Qwen3.5 page-understanding capability is unavailable: {:?}",
                    page_capability_for_load
                ));
            }
            let resident_model = page_app.llm.local_model_handle().await?;
            let loaded = tokio::task::spawn_blocking(move || {
                QwenPageUnderstanding::load_from_shared_model(
                    &page_model_runtime,
                    resident_model,
                    page_projector_path,
                    false,
                    page_model_backend,
                )
            })
            .await;
            match loaded {
                Ok(Ok(model)) => Ok(model),
                Ok(Err(error)) => Err(error),
                Err(error) => Err(anyhow!(
                    "Qwen3.5 page-understanding loader task failed: {error}"
                )),
            }
        };
        let (
            detector,
            ocr_detector,
            ocr,
            (text_segmenter, bubble_segmenter, inpainter),
            page_understanding,
        ) = tokio::try_join!(
            detector_future,
            ocr_detector_future,
            ocr_future,
            cleanup_models_future,
            page_future
        )
        .context("load resident CUDA models")?;
        instrumentation.initialized(RuntimeComponent::Vision);
        instrumentation.initialized(RuntimeComponent::Ocr);
        instrumentation.initialized(RuntimeComponent::Projector);
        instrumentation.initialized(RuntimeComponent::Segmentation);
        instrumentation.initialized(RuntimeComponent::Inpainting);
        Ok(Self {
            language,
            instrumentation,
            detector: Mutex::new(detector),
            ocr_detector: Mutex::new(ocr_detector),
            ocr: Mutex::new(ocr),
            text_segmenter: Mutex::new(text_segmenter),
            bubble_segmenter: Mutex::new(bubble_segmenter),
            inpainter: Mutex::new(inpainter),
            page_understanding: Mutex::new(page_understanding),
        })
    }

    fn prime_non_language_inference(&self) -> Result<()> {
        // Prime every non-language model on the actual interactive shapes.
        // Loading weights does not initialize Candle/ORT CUDA kernels or the
        // dynamic OCR output allocator.  The recovery segmenter is included
        // here as well: its graph is used for every page and warming it once
        // before the first request prevents a hidden multi-second stall in
        // the middle of the chapter pipeline.
        let sample = DynamicImage::new_rgb8(1_024, 1_024);
        self.instrumentation.invoked(RuntimeComponent::Vision);
        self.detector
            .lock()
            .map_err(|_| anyhow!("detector lock poisoned during inference warm-up"))?
            .inference_tiles(std::slice::from_ref(&sample))
            .context("prime comic text detector inference")?;
        self.instrumentation.invoked(RuntimeComponent::Ocr);
        self.ocr_detector
            .lock()
            .map_err(|_| anyhow!("OCR detector lock poisoned during inference warm-up"))?
            .detect_tiles(std::slice::from_ref(&sample))
            .context("prime PP-OCR small text detector inference")?;
        let segmentation_sample = DynamicImage::new_rgb8(2_048, 2_048);
        self.instrumentation.invoked(RuntimeComponent::Segmentation);
        self.text_segmenter
            .lock()
            .map_err(|_| anyhow!("text segmenter lock poisoned during inference warm-up"))?
            .inference_batch(std::slice::from_ref(&segmentation_sample))
            .context("prime manga text segmentation inference")?;

        self.instrumentation.invoked(RuntimeComponent::Segmentation);
        self.bubble_segmenter
            .lock()
            .map_err(|_| anyhow!("bubble segmenter lock poisoned during inference warm-up"))?
            .inference(&sample)
            .context("prime speech bubble segmentation inference")?;

        let mut ocr_pixels = RgbImage::from_pixel(320, 64, Rgb([255, 255, 255]));
        let mut ocr_probabilities = ProbabilityMap::zeros(320, 64);
        for y in 18..46 {
            for x in 24..296 {
                ocr_pixels.put_pixel(x, y, Rgb([0, 0, 0]));
                ocr_probabilities.values[(y * 320 + x) as usize] = 1.0;
            }
        }
        self.instrumentation.invoked(RuntimeComponent::Ocr);
        self.ocr
            .lock()
            .map_err(|_| anyhow!("OCR lock poisoned during inference warm-up"))?
            .recognize_regions(&[DynamicImage::ImageRgb8(ocr_pixels)], &[ocr_probabilities])
            .context("prime PP-OCR small CUDA inference and dynamic output allocation")?;

        let inpaint_image = RgbImage::from_pixel(512, 512, Rgb([255, 255, 255]));
        let mut inpaint_mask = GrayImage::new(512, 512);
        for y in 220..292 {
            for x in 176..336 {
                inpaint_mask.put_pixel(x, y, Luma([255]));
            }
        }
        let inpaint_bubble = GrayImage::from_pixel(512, 512, Luma([255]));
        self.instrumentation.invoked(RuntimeComponent::Inpainting);
        self.inpainter
            .lock()
            .map_err(|_| anyhow!("inpainter lock poisoned during inference warm-up"))?
            .inference_rgb_with_blocks(
                &inpaint_image,
                &inpaint_mask,
                &inpaint_bubble,
                &[TextRegion {
                    x: 176.0,
                    y: 220.0,
                    width: 160.0,
                    height: 72.0,
                    confidence: 1.0,
                    detected_font_size_px: Some(36.0),
                    detector: Some("resident-warm-up".to_owned()),
                    ..TextRegion::default()
                }],
            )
            .context("prime LaMa manga inpainting inference")?;
        Ok(())
    }

    fn prime_page_understanding(&self) -> Result<()> {
        self.instrumentation.invoked(RuntimeComponent::Vision);
        self.instrumentation.invoked(RuntimeComponent::Projector);
        tokio::runtime::Handle::current().block_on(async {
            self.page_understanding
                .lock()
                .map_err(|_| anyhow!("page-understanding lock poisoned during warm-up"))?
                .warm_up()
                .context("prime multimodal page understanding")
        })
    }
}

fn utf8_path(path: PathBuf) -> Result<Utf8PathBuf> {
    Utf8PathBuf::from_path_buf(path).map_err(|path| anyhow!("path is not valid UTF-8: {path:?}"))
}

#[derive(Clone)]
struct DocumentPiece {
    piece_id: String,
    block_index: usize,
    piece_order: usize,
    source_text: String,
    separator_after: String,
    kind: SourceSpanKind,
    token_count: usize,
}

#[derive(Clone)]
struct DocumentContextSpan {
    piece_id: Option<String>,
    source_text: String,
    separator_after: String,
    token_count: usize,
}

impl From<&DocumentPiece> for DocumentContextSpan {
    fn from(piece: &DocumentPiece) -> Self {
        Self {
            piece_id: Some(piece.piece_id.clone()),
            source_text: piece.source_text.clone(),
            separator_after: piece.separator_after.clone(),
            token_count: piece.token_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DocumentDispatchEvidence {
    reason: String,
    item_ids: Vec<String>,
    token_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DocumentJobEvidence {
    job_id: String,
    tokenizer_identity: String,
    dispatches: Vec<DocumentDispatchEvidence>,
    runtime_initializations: RuntimeCounters,
    runtime_invocations: RuntimeCounters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImageJobEvidence {
    job_id: String,
    tokenizer_identity: String,
    language_unit_count: u64,
    language_generation_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum BenchmarkSample {
    Document(DocumentJobEvidence),
    Image(ImageJobEvidence),
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BenchmarkEvidenceFile {
    samples: Vec<BenchmarkSample>,
}

fn refresh_document_runtime_evidence(
    evidence: &mut DocumentJobEvidence,
    start: (RuntimeCounters, RuntimeCounters),
    instrumentation: &RuntimeInstrumentation,
) {
    let end = instrumentation.snapshot();
    evidence.runtime_initializations = RuntimeCounters::between(start.0, end.0);
    evidence.runtime_invocations = RuntimeCounters::between(start.1, end.1);
}

fn emit_document_evidence(evidence: &DocumentJobEvidence) -> Result<()> {
    update_benchmark_evidence(|file| {
        if let Some(existing) = file.samples.iter_mut().find(|sample| {
            matches!(sample, BenchmarkSample::Document(value) if value.job_id == evidence.job_id)
        }) {
            *existing = BenchmarkSample::Document(evidence.clone());
        } else {
            file.samples
                .push(BenchmarkSample::Document(evidence.clone()));
        }
    })
}

fn record_image_language_evidence(
    job_id: &str,
    tokenizer_identity: String,
    unit_count: usize,
    duration: Duration,
) -> Result<()> {
    if unit_count == 0 && duration.is_zero() {
        return Ok(());
    }
    let duration_ms = if duration.is_zero() {
        0
    } else {
        duration.as_millis().max(1).min(u128::from(u64::MAX)) as u64
    };
    update_benchmark_evidence(|file| {
        accumulate_image_language_evidence(
            file,
            job_id,
            tokenizer_identity,
            unit_count,
            duration_ms,
        );
    })
}

fn accumulate_image_language_evidence(
    file: &mut BenchmarkEvidenceFile,
    job_id: &str,
    tokenizer_identity: String,
    unit_count: usize,
    duration_ms: u64,
) {
    if let Some(BenchmarkSample::Image(existing)) = file
        .samples
        .iter_mut()
        .find(|sample| matches!(sample, BenchmarkSample::Image(value) if value.job_id == job_id))
    {
        existing.language_unit_count = existing
            .language_unit_count
            .saturating_add(unit_count.min(u64::MAX as usize) as u64);
        existing.language_generation_duration_ms = existing
            .language_generation_duration_ms
            .saturating_add(duration_ms);
        existing.tokenizer_identity = tokenizer_identity;
    } else {
        file.samples.push(BenchmarkSample::Image(ImageJobEvidence {
            job_id: job_id.to_owned(),
            tokenizer_identity,
            language_unit_count: unit_count.min(u64::MAX as usize) as u64,
            language_generation_duration_ms: duration_ms,
        }));
    }
}

fn update_benchmark_evidence(update: impl FnOnce(&mut BenchmarkEvidenceFile)) -> Result<()> {
    let Some(path) = std::env::var_os("HSKIFY_BENCH_EVIDENCE_PATH") else {
        return Ok(());
    };
    let path = PathBuf::from(path);
    if path.as_os_str().is_empty() {
        bail!("HSKIFY_BENCH_EVIDENCE_PATH cannot be empty");
    }
    static EVIDENCE_WRITE: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = EVIDENCE_WRITE
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| anyhow!("benchmark evidence lock poisoned"))?;
    let mut file = if path.exists() {
        serde_json::from_slice::<BenchmarkEvidenceFile>(
            &std::fs::read(&path)
                .with_context(|| format!("read benchmark evidence {}", path.display()))?,
        )
        .with_context(|| format!("parse benchmark evidence {}", path.display()))?
    } else {
        BenchmarkEvidenceFile::default()
    };
    update(&mut file);
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&file).context("serialize benchmark evidence")?,
    )
    .with_context(|| format!("write benchmark evidence {}", path.display()))?;
    Ok(())
}

fn model_token_count(model: &koharu_llm::safe::model::LlamaModel, text: &str) -> Result<usize> {
    model
        .str_to_token(text, AddBos::Never)
        .map(|tokens| tokens.len())
        .context("count document source tokens")
}

fn document_piece_neighbor_context(
    context_plan: &[DocumentContextSpan],
    batch: &[DocumentPiece],
) -> Result<(Vec<String>, Vec<String>)> {
    if batch.is_empty() {
        bail!("document context requires a non-empty batch");
    }
    let selected = batch
        .iter()
        .map(|piece| piece.piece_id.as_str())
        .collect::<HashSet<_>>();
    let positions = context_plan
        .iter()
        .enumerate()
        .filter_map(|(index, span)| {
            span.piece_id
                .as_deref()
                .is_some_and(|piece_id| selected.contains(piece_id))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if positions.len() != batch.len()
        || positions
            .windows(2)
            .any(|pair| pair[1] != pair[0].saturating_add(1))
    {
        bail!("document scheduler selected a non-canonical piece window");
    }
    let first = positions[0];
    let last = *positions.last().expect("non-empty positions");
    let mut preceding = Vec::new();
    let mut following = Vec::new();
    let mut used_tokens = 0usize;
    let mut before = first.checked_sub(1);
    let mut after = (last + 1 < context_plan.len()).then_some(last + 1);
    while preceding.len() + following.len() < MAX_HSK_PRECEDING_UTTERANCES
        && (before.is_some() || after.is_some())
    {
        if let Some(index) = before.take() {
            let span = &context_plan[index];
            if used_tokens.saturating_add(span.token_count) <= DOCUMENT_CONTEXT_TOKEN_BUDGET {
                preceding.push(format!("{}{}", span.source_text, span.separator_after));
                used_tokens = used_tokens.saturating_add(span.token_count);
                before = index.checked_sub(1);
            }
        }
        if preceding.len() + following.len() >= MAX_HSK_PRECEDING_UTTERANCES {
            break;
        }
        if let Some(index) = after.take() {
            let span = &context_plan[index];
            if used_tokens.saturating_add(span.token_count) <= DOCUMENT_CONTEXT_TOKEN_BUDGET {
                following.push(format!("{}{}", span.source_text, span.separator_after));
                used_tokens = used_tokens.saturating_add(span.token_count);
                after = (index + 1 < context_plan.len()).then_some(index + 1);
            }
        }
    }
    preceding.reverse();
    Ok((preceding, following))
}

fn split_document_block(
    model: &koharu_llm::safe::model::LlamaModel,
    block_index: usize,
    block: &DocumentSourceBlock,
) -> std::result::Result<Vec<DocumentPiece>, String> {
    let block_token_count =
        model_token_count(model, &block.text).map_err(|error| format!("{error:#}"))?;
    if block_token_count <= DOCUMENT_PIECE_TOKEN_BUDGET {
        return Ok(vec![DocumentPiece {
            piece_id: format!("{}::0", block.item_id),
            block_index,
            piece_order: 0,
            source_text: block.text.clone(),
            separator_after: String::new(),
            kind: block.kind,
            token_count: block_token_count,
        }]);
    }

    let segmenter = SentenceSegmenter::new(SentenceBreakInvariantOptions::default());
    let boundaries = segmenter.segment_str(&block.text).collect::<Vec<_>>();
    let mut sentences = Vec::<&str>::new();
    for pair in boundaries.windows(2) {
        let sentence = &block.text[pair[0]..pair[1]];
        if !sentence.trim().is_empty() {
            if model_token_count(model, sentence).map_err(|error| format!("{error:#}"))?
                > DOCUMENT_PIECE_TOKEN_BUDGET
            {
                return Err(
                    "one sentence exceeds the resident language context and cannot be split safely"
                        .to_owned(),
                );
            }
            sentences.push(sentence);
        }
    }
    if sentences.len() < 2 {
        return Err(
            "the oversized block has no safe ICU sentence boundary for translation".to_owned(),
        );
    }

    let mut chunks = Vec::<String>::new();
    let mut current = String::new();
    for sentence in sentences {
        let candidate = format!("{current}{sentence}");
        if !current.is_empty()
            && model_token_count(model, &candidate).map_err(|error| format!("{error:#}"))?
                > DOCUMENT_PIECE_TOKEN_BUDGET
        {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(sentence);
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
        .into_iter()
        .enumerate()
        .map(|(piece_order, source)| {
            let trimmed = source.trim_end_matches(char::is_whitespace);
            let separator_after = source[trimmed.len()..]
                .chars()
                .filter(|character| *character == '\n')
                .collect::<String>();
            let source_text = trimmed.to_owned();
            let token_count =
                model_token_count(model, &source_text).map_err(|error| format!("{error:#}"))?;
            Ok(DocumentPiece {
                piece_id: format!("{}::{piece_order}", block.item_id),
                block_index,
                piece_order,
                source_text,
                separator_after,
                kind: block.kind,
                token_count,
            })
        })
        .collect()
}

pub(crate) async fn wait_for_active_document(
    sink: &JobUpdateSink,
    cancel: &AtomicBool,
) -> std::result::Result<crate::server::JobFocus, PipelineError> {
    loop {
        cancellation_boundary(cancel)?;
        let focus = sink.focus();
        if focus.active {
            return Ok(focus);
        }
        sink.wait_for_focus_change(focus.revision, Duration::from_millis(250))
            .await;
    }
}

fn take_next_document_batch(
    remaining: &mut Vec<DocumentPiece>,
    blocks: &[DocumentSourceBlock],
    visible_ids: &HashSet<String>,
    isolate_first_visible: bool,
) -> (Vec<DocumentPiece>, &'static str) {
    remaining.sort_unstable_by_key(|piece| (piece.block_index, piece.piece_order));
    let priority_block = blocks.iter().enumerate().find_map(|(index, block)| {
        (visible_ids.contains(&block.item_id)
            && remaining.iter().any(|piece| piece.block_index == index))
        .then_some(index)
    });
    let start = priority_block
        .and_then(|index| {
            remaining
                .iter()
                .position(|piece| piece.block_index == index)
        })
        .unwrap_or(0);
    let mut selected = Vec::<DocumentPiece>::new();
    let mut tokens = 0_usize;
    for piece in remaining.iter().skip(start) {
        if priority_block.is_some() && !visible_ids.contains(&blocks[piece.block_index].item_id) {
            break;
        }
        if isolate_first_visible && priority_block.is_some_and(|index| piece.block_index != index) {
            break;
        }
        if let Some(previous) = selected.last() {
            let contiguous_piece = piece.block_index == previous.block_index
                && piece.piece_order == previous.piece_order.saturating_add(1);
            let contiguous_block = piece.block_index == previous.block_index.saturating_add(1)
                && piece.piece_order == 0;
            if !contiguous_piece && !contiguous_block {
                break;
            }
        }
        let piece_tokens = piece.token_count;
        if !selected.is_empty()
            && (selected.len() >= DOCUMENT_BATCH_MAX
                || tokens.saturating_add(piece_tokens) > DOCUMENT_BATCH_TOKEN_BUDGET)
        {
            break;
        }
        tokens = tokens.saturating_add(piece_tokens);
        selected.push(piece.clone());
    }
    let selected_ids = selected
        .iter()
        .map(|piece| piece.piece_id.as_str())
        .collect::<HashSet<_>>();
    remaining.retain(|piece| !selected_ids.contains(piece.piece_id.as_str()));
    let reason = if priority_block.is_some() {
        "visible"
    } else {
        "ordered"
    };
    (selected, reason)
}

struct TranslationService<'a> {
    language: &'a LanguageRuntime,
    control: &'a HskControl,
    cuda_scheduler: &'a Arc<CudaScheduler>,
}

#[derive(Clone)]
struct TranslationUnit {
    id: String,
    kind: HskUtteranceKind,
    source_text: String,
    faithful_chinese: Option<FaithfulText>,
    provenance: DirectSourceProvenance,
    layout: Option<HskLayoutConstraints>,
    source_context: SourceContext,
}

#[derive(Debug, Clone, Default)]
struct SourceContext {
    preceding: Vec<String>,
    following: Vec<String>,
}

#[derive(Debug, Clone)]
struct FaithfulText {
    text: String,
    protected_names: Vec<koharu_app::llm::ProtectedName>,
}
impl std::ops::Deref for FaithfulText {
    type Target = str;
    fn deref(&self) -> &str {
        &self.text
    }
}
impl From<String> for FaithfulText {
    fn from(text: String) -> Self {
        Self {
            text,
            protected_names: Vec::new(),
        }
    }
}
impl FaithfulText {
    fn proper_names(&self) -> Vec<ProperName> {
        use koharu_app::llm::ProtectedNameReason as R;
        self.protected_names
            .iter()
            .map(|name| ProperName {
                text: name.chinese_text.clone(),
                reason: match name.reason {
                    R::PersonName => hsk_control::ProperNameReason::PersonName,
                    R::PlaceName => hsk_control::ProperNameReason::PlaceName,
                    R::Title => hsk_control::ProperNameReason::Title,
                    R::UnavoidableProperNoun => {
                        hsk_control::ProperNameReason::UnavoidableProperNoun
                    }
                },
            })
            .collect()
    }
}
impl CacheValue for FaithfulText {
    fn retained_bytes(&self) -> usize {
        self.text.len()
            + self
                .protected_names
                .iter()
                .map(|n| n.source_text.len() + n.chinese_text.len() + 64)
                .sum::<usize>()
    }
}

struct FaithfulBatchResult {
    items: HashMap<String, FaithfulText>,
    generation_duration: Duration,
}

struct TranslationBatchResult {
    items: HashMap<String, std::result::Result<CachedTranslation, String>>,
    generation_duration: Duration,
}

impl TranslationService<'_> {
    #[allow(clippy::too_many_arguments)]
    async fn establish_faithful(
        &self,
        utterances: Vec<FaithfulSourceUtterance>,
        preceding_utterances: Vec<HskPrecedingUtterance>,
        preceding_english: Vec<String>,
        following_english: Vec<String>,
        provenance: DirectSourceProvenance,
        priority: CudaPriority,
        cancel: Arc<AtomicBool>,
    ) -> std::result::Result<FaithfulBatchResult, PipelineError> {
        if utterances.is_empty() {
            return Ok(FaithfulBatchResult {
                items: HashMap::new(),
                generation_duration: Duration::ZERO,
            });
        }
        let translator = self.language.app.llm.direct_hsk_translator();
        let mut keys = HashMap::with_capacity(utterances.len());
        let mut items = HashMap::with_capacity(utterances.len());
        let mut missing = Vec::new();
        {
            let mut cache =
                self.language.faithful_cache.lock().map_err(|_| {
                    PipelineError::new("CACHE_FAILED", "Faithful cache lock poisoned.")
                })?;
            for utterance in utterances {
                let material = serde_json::to_vec(&(
                    "faithful-source-names-v3",
                    &utterance.source_english,
                    utterance.kind,
                    match provenance {
                        DirectSourceProvenance::Dom => "dom",
                        DirectSourceProvenance::Ocr => "ocr",
                    },
                    &preceding_utterances,
                    &preceding_english,
                    &following_english,
                    translator.model_revision(),
                    translator.prompt_hash(),
                    translator.validator_hash(),
                    self.control.cache_revision(),
                ))
                .map_err(|error| PipelineError::pipeline(error.into()))?;
                let key = sha256_hex(&material);
                if let Some(text) = cache.get(&key) {
                    items.insert(utterance.id.clone(), text);
                } else {
                    missing.push(utterance.clone());
                }
                keys.insert(utterance.id, key);
            }
        }
        if missing.is_empty() {
            return Ok(FaithfulBatchResult {
                items,
                generation_duration: Duration::ZERO,
            });
        }
        let permit = self
            .cuda_scheduler
            .acquire(CudaWorkload::Language, priority, cancel.clone())
            .await
            .map_err(cuda_admission_error)?;
        let translator = self.language.app.llm.direct_hsk_translator();
        let generation_started = Instant::now();
        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(
                translator.translate_faithful_batch_for_source(
                    &FaithfulTranslationBatchRequest {
                        utterances: missing,
                        preceding_utterances,
                        preceding_english,
                        following_english,
                    },
                    provenance,
                    cancel.as_ref(),
                ),
            )
        })
        .map_err(PipelineError::pipeline)?;
        let generation_duration = generation_started.elapsed();
        drop(permit);
        cancellation_boundary(cancel.as_ref())?;
        let mut cache = self
            .language
            .faithful_cache
            .lock()
            .map_err(|_| PipelineError::new("CACHE_FAILED", "Faithful cache lock poisoned."))?;
        for outcome in result.items {
            if outcome.is_valid() {
                if let Some(text) = outcome.text {
                    let value = FaithfulText {
                        text,
                        protected_names: outcome.protected_names,
                    };
                    cache.insert(keys[&outcome.id].clone(), value.clone());
                    items.insert(outcome.id, value);
                }
            }
        }
        Ok(FaithfulBatchResult {
            items,
            generation_duration,
        })
    }

    async fn translate_document_piece(
        &self,
        piece: &DocumentPiece,
        request: &DocumentJobRequest,
        context_plan: &[DocumentContextSpan],
        visible: bool,
        cancel: Arc<AtomicBool>,
    ) -> std::result::Result<TranslationBatchResult, PipelineError> {
        let (preceding, following) =
            document_piece_neighbor_context(context_plan, std::slice::from_ref(piece))
                .map_err(PipelineError::pipeline)?;
        let unit = TranslationUnit {
            id: piece.piece_id.clone(),
            kind: document_hsk_kind(piece.kind),
            source_text: piece.source_text.clone(),
            faithful_chinese: None,
            provenance: DirectSourceProvenance::Dom,
            layout: None,
            source_context: SourceContext {
                preceding,
                following,
            },
        };
        self.translate_units(
            std::slice::from_ref(&unit),
            request.settings.hsk_level,
            request.settings.learning_mode,
            if visible {
                CudaPriority::Visible
            } else {
                CudaPriority::Offscreen
            },
            cancel,
        )
        .await
    }

    async fn translate_units(
        &self,
        units: &[TranslationUnit],
        level: HskLevel,
        mode: LearningMode,
        priority: CudaPriority,
        cancel: Arc<AtomicBool>,
    ) -> std::result::Result<TranslationBatchResult, PipelineError> {
        let mut items = HashMap::with_capacity(units.len());
        let mut generation_duration = Duration::ZERO;
        for unit in units {
            cancellation_boundary(cancel.as_ref())?;
            let result = self
                .translate_unit(
                    std::slice::from_ref(unit),
                    level,
                    mode,
                    Vec::new(),
                    unit.source_context.preceding.clone(),
                    unit.source_context.following.clone(),
                    priority,
                    cancel.clone(),
                )
                .await?;
            items.extend(result.items);
            generation_duration += result.generation_duration;
        }
        Ok(TranslationBatchResult {
            items,
            generation_duration,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn translate_unit(
        &self,
        units: &[TranslationUnit],
        requested_level: HskLevel,
        learning_mode: LearningMode,
        preceding_utterances: Vec<HskPrecedingUtterance>,
        preceding_english: Vec<String>,
        following_english: Vec<String>,
        priority: CudaPriority,
        cancel: Arc<AtomicBool>,
    ) -> std::result::Result<TranslationBatchResult, PipelineError> {
        cancellation_boundary(cancel.as_ref())?;
        let provenance = units.first().map(|unit| unit.provenance).ok_or_else(|| {
            PipelineError::new("TRANSLATION_BATCH_FAILED", "The language batch is empty.")
        })?;
        if units.iter().any(|unit| unit.provenance != provenance) {
            return Err(PipelineError::new(
                "TRANSLATION_MODALITY_MISMATCH",
                "One language batch cannot mix DOM and OCR provenance.",
            ));
        }
        let translator = self.language.app.llm.direct_hsk_translator();
        let model_id = translator.model_id().to_string();
        let batch_source_texts = units
            .iter()
            .map(|unit| unit.source_text.clone())
            .collect::<Vec<_>>();
        let keys = units
            .iter()
            .map(|unit| {
                (
                    unit.id.clone(),
                    translation_cache_key(
                        &unit.source_text,
                        unit.kind,
                        unit.layout,
                        &batch_source_texts,
                        &preceding_utterances,
                        &preceding_english,
                        &following_english,
                        provenance,
                        learning_mode,
                        u8::from(requested_level),
                        &model_id,
                        translator.model_revision(),
                        translator.prompt_hash(),
                        translator.validator_hash(),
                        self.control.cache_revision(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut results = HashMap::with_capacity(units.len());
        let units = {
            let mut cache = self.language.translation_cache.lock().map_err(|_| {
                PipelineError::new("CACHE_FAILED", "Translation cache lock poisoned.")
            })?;
            units
                .iter()
                .filter(|unit| {
                    if let Some(value) = cache.get(&keys[&unit.id]).filter(translation_is_final) {
                        results.insert(unit.id.clone(), Ok(value));
                        false
                    } else {
                        true
                    }
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        if units.is_empty() {
            return Ok(TranslationBatchResult {
                items: results,
                generation_duration: Duration::ZERO,
            });
        }
        let mut faithful_by_id = units
            .iter()
            .filter_map(|unit| {
                unit.faithful_chinese
                    .clone()
                    .map(|text| (unit.id.clone(), text))
            })
            .collect::<HashMap<_, _>>();
        let missing_faithful = units
            .iter()
            .filter(|unit| !faithful_by_id.contains_key(&unit.id))
            .map(|unit| FaithfulSourceUtterance {
                id: unit.id.clone(),
                kind: unit.kind,
                source_english: unit.source_text.clone(),
            })
            .collect::<Vec<_>>();
        let mut generation_duration = Duration::ZERO;
        if !missing_faithful.is_empty() {
            let faithful = self
                .establish_faithful(
                    missing_faithful,
                    preceding_utterances.clone(),
                    preceding_english.clone(),
                    following_english.clone(),
                    provenance,
                    priority,
                    cancel.clone(),
                )
                .await?;
            generation_duration += faithful.generation_duration;
            faithful_by_id.extend(faithful.items);
        }
        cancellation_boundary(cancel.as_ref())?;

        let level = ControlHskLevel::new(u8::from(requested_level))
            .map_err(|error| PipelineError::new("INVALID_HSK_LEVEL", error.to_string()))?;
        let mut pending = Vec::new();
        {
            let mut cache = self.language.translation_cache.lock().map_err(|_| {
                PipelineError::new("CACHE_FAILED", "Translation cache lock poisoned.")
            })?;
            for unit in &units {
                let Some(faithful_chinese) = faithful_by_id.get(&unit.id) else {
                    results.insert(
                        unit.id.clone(),
                        Err("faithful Chinese could not be established".to_owned()),
                    );
                    continue;
                };
                let key = &keys[&unit.id];
                {
                    let report = self.control.validate(
                        faithful_chinese,
                        level,
                        &faithful_chinese.proper_names(),
                    );
                    if learning_mode == LearningMode::Strict
                        && report.strictly_valid
                        && unit.layout.is_none_or(|layout| {
                            faithful_chinese.chars().count() <= usize::from(layout.max_characters)
                                && faithful_chinese.lines().count() <= usize::from(layout.max_lines)
                        })
                    {
                        let mut value = natural_translation(faithful_chinese.clone(), report);
                        populate_pinyin(self.control, &mut value);
                        cache.insert(key.clone(), value.clone());
                        results.insert(unit.id.clone(), Ok(value));
                    } else {
                        pending.push(unit.clone());
                    }
                }
            }
        }

        if learning_mode == LearningMode::Natural {
            for unit in pending {
                let faithful = faithful_by_id
                    .get(&unit.id)
                    .expect("pending units have faithful Chinese")
                    .clone();
                let report = self
                    .control
                    .validate(&faithful, level, &faithful.proper_names());
                let mut value = natural_translation(faithful, report);
                populate_pinyin(self.control, &mut value);
                self.language
                    .translation_cache
                    .lock()
                    .map_err(|_| {
                        PipelineError::new("CACHE_FAILED", "Translation cache lock poisoned.")
                    })?
                    .insert(keys[&unit.id].clone(), value.clone());
                results.insert(unit.id, Ok(value));
            }
            return Ok(TranslationBatchResult {
                items: results,
                generation_duration,
            });
        }

        if pending.is_empty() {
            return Ok(TranslationBatchResult {
                items: results,
                generation_duration,
            });
        }
        let utterances = pending
            .iter()
            .map(|unit| HskSourceUtterance {
                id: unit.id.clone(),
                kind: unit.kind,
                source_english: unit.source_text.clone(),
                faithful_chinese: faithful_by_id[&unit.id].text.clone(),
                layout: unit.layout,
            })
            .collect::<Vec<_>>();
        let permit = self
            .cuda_scheduler
            .acquire(CudaWorkload::Language, priority, cancel.clone())
            .await
            .map_err(cuda_admission_error)?;
        let primary_started = Instant::now();
        let primary = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(translator.translate_batch_for_source(
                &HskTranslationBatchRequest {
                    requested_level: u8::from(requested_level),
                    learning_mode: HskLearningMode::Strict,
                    utterances,
                    preceding_utterances: preceding_utterances.clone(),
                    preceding_english: preceding_english.clone(),
                    following_english: following_english.clone(),
                },
                provenance,
                cancel.as_ref(),
            ))
        })
        .map_err(PipelineError::pipeline)?;
        generation_duration += primary_started.elapsed();
        drop(permit);
        cancellation_boundary(cancel.as_ref())?;
        let mut outcomes = primary
            .items
            .into_iter()
            .map(|outcome| (outcome.id.clone(), outcome))
            .collect::<HashMap<_, _>>();
        let mut states = pending
            .iter()
            .map(|unit| {
                let outcome = outcomes
                    .remove(&unit.id)
                    .unwrap_or_else(|| missing_translation_outcome(&unit.id));
                (
                    unit.id.clone(),
                    TranslationState::from_initial(
                        outcome,
                        self.control,
                        level,
                        &faithful_by_id[&unit.id].proper_names(),
                        LearningMode::Strict,
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        for unit in &pending {
            if let Some(state) = states.get_mut(&unit.id) {
                for name in &faithful_by_id[&unit.id].protected_names {
                    if !state
                        .displayed_chinese
                        .as_deref()
                        .or(state.base_chinese.as_deref())
                        .is_some_and(|text| name.is_anchored(&unit.source_text, text))
                    {
                        state.problems.push(format!(
                            "Preserve the exact proper name: {} => {}",
                            name.source_text, name.chinese_text
                        ));
                    }
                }
            }
        }
        let repair_utterances = pending
            .iter()
            .filter_map(|unit| {
                let state = states.get(&unit.id)?;
                (!state.problems.is_empty()).then(|| HskRepairUtterance {
                    id: unit.id.clone(),
                    kind: unit.kind,
                    source_english: unit.source_text.clone(),
                    faithful_chinese: faithful_by_id[&unit.id].text.clone(),
                    layout: unit.layout,
                    rejected_chinese: state.base_chinese.clone(),
                    avoid_chinese: state.avoid_chinese(),
                    problems: state.problems.clone(),
                })
            })
            .collect::<Vec<_>>();
        if !repair_utterances.is_empty() {
            let permit = self
                .cuda_scheduler
                .acquire(CudaWorkload::Language, priority, cancel.clone())
                .await
                .map_err(cuda_admission_error)?;
            let repair_started = Instant::now();
            let repaired = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(
                    translator.repair_invalid_batch_for_source(
                        &HskTranslationRepairBatchRequest {
                            requested_level: u8::from(requested_level),
                            learning_mode: HskLearningMode::Strict,
                            utterances: repair_utterances,
                            preceding_utterances,
                            preceding_english,
                            following_english,
                        },
                        provenance,
                        cancel.as_ref(),
                    ),
                )
            })
            .map_err(PipelineError::pipeline)?;
            generation_duration += repair_started.elapsed();
            drop(permit);
            for outcome in repaired.items {
                if let Some(state) = states.get_mut(&outcome.id) {
                    state.apply_repair(
                        outcome.clone(),
                        self.control,
                        level,
                        &faithful_by_id[&outcome.id].proper_names(),
                    );
                }
            }
        }

        for unit in pending {
            let value = match states.remove(&unit.id) {
                Some(state) if state.problems.is_empty() => state
                    .finish()
                    .map(|mut value| {
                        value.protected_names = faithful_by_id[&unit.id].protected_names.clone();
                        populate_pinyin(self.control, &mut value);
                        value
                    })
                    .map_err(|error| format!("{error:#}")),
                Some(_) => {
                    Err("strict HSK validation remained invalid after one repair".to_owned())
                }
                None => Err("no complete strict translation was returned".to_owned()),
            };
            let value = value.and_then(|value| {
                if value
                    .protected_names
                    .iter()
                    .all(|name| name.is_anchored(&unit.source_text, &value.displayed_chinese))
                {
                    Ok(value)
                } else {
                    Err("a protected proper name was changed or omitted".to_owned())
                }
            });
            if let Ok(final_value) = &value {
                self.language
                    .translation_cache
                    .lock()
                    .map_err(|_| {
                        PipelineError::new("CACHE_FAILED", "Translation cache lock poisoned.")
                    })?
                    .insert(keys[&unit.id].clone(), final_value.clone());
            }
            results.insert(unit.id, value);
        }
        Ok(TranslationBatchResult {
            items: results,
            generation_duration,
        })
    }
}

fn document_hsk_kind(kind: SourceSpanKind) -> HskUtteranceKind {
    match kind {
        SourceSpanKind::Prose => HskUtteranceKind::Prose,
        SourceSpanKind::Heading => HskUtteranceKind::Heading,
        SourceSpanKind::Dialogue => HskUtteranceKind::Dialogue,
        SourceSpanKind::Caption => HskUtteranceKind::Caption,
        SourceSpanKind::Thought => HskUtteranceKind::Thought,
        SourceSpanKind::Sfx => HskUtteranceKind::Sfx,
    }
}

struct DocumentAccumulator {
    block: DocumentSourceBlock,
    expected_pieces: usize,
    pieces: BTreeMap<usize, (CachedTranslation, String)>,
    rejection: Option<String>,
    published: bool,
}

impl DocumentAccumulator {
    fn new(block: DocumentSourceBlock, expected_pieces: usize) -> Self {
        Self {
            block,
            expected_pieces,
            pieces: BTreeMap::new(),
            rejection: None,
            published: false,
        }
    }

    fn record(&mut self, order: usize, value: CachedTranslation, separator: String) {
        if self.rejection.is_none() {
            self.pieces.insert(order, (value, separator));
        }
    }

    fn reject(&mut self, reason: String) {
        if self.rejection.is_none() {
            self.rejection = Some(reason);
        }
    }

    fn is_complete(&self) -> bool {
        self.rejection.is_some() || self.pieces.len() == self.expected_pieces
    }
}

#[allow(clippy::too_many_arguments)]
fn publish_ready_document_blocks(
    sink: &JobUpdateSink,
    control: &HskControl,
    requested_level: HskLevel,
    learning_mode: LearningMode,
    accumulators: &mut [DocumentAccumulator],
    next_index: &mut usize,
    chapter_contexts: &Mutex<ChapterContextStore>,
    chapter_id: &str,
) -> std::result::Result<(), PipelineError> {
    for accumulator in accumulators
        .iter_mut()
        .filter(|item| !item.published && item.is_complete())
    {
        publish_document_accumulator(
            sink,
            control,
            requested_level,
            learning_mode,
            accumulator,
            chapter_contexts,
            chapter_id,
        )?;
        *next_index += 1;
    }
    Ok(())
}

fn publish_document_accumulator(
    sink: &JobUpdateSink,
    control: &HskControl,
    requested_level: HskLevel,
    learning_mode: LearningMode,
    accumulator: &mut DocumentAccumulator,
    chapter_contexts: &Mutex<ChapterContextStore>,
    chapter_id: &str,
) -> std::result::Result<(), PipelineError> {
    if accumulator.published {
        return Ok(());
    }
    if let Some(reason) = accumulator.rejection.take() {
        sink.publish(JobUpdateDraft::DocumentBlockPreserved {
            block: DocumentBlockPreserved {
                parent_block_id: accumulator.block.parent_block_id.clone(),
                sub_item_order: accumulator.block.sub_item_order,
                item_id: accumulator.block.item_id.clone(),
                source_index: accumulator.block.source_index,
                item_order: accumulator.block.item_order,
                kind: accumulator.block.kind,
                source_text: accumulator.block.text.clone(),
                reason,
            },
        })
        .map_err(|error| publish_error(error, sink))?;
        accumulator.published = true;
        return Ok(());
    }
    if accumulator.pieces.len() != accumulator.expected_pieces {
        return Ok(());
    }
    let mut base_chinese = String::new();
    let mut displayed_chinese = String::new();
    let mut repair_state = HskRepairState::NotNeeded;
    for (piece, separator) in accumulator.pieces.values() {
        base_chinese.push_str(&piece.base_chinese);
        base_chinese.push_str(separator);
        displayed_chinese.push_str(&piece.displayed_chinese);
        displayed_chinese.push_str(separator);
        if piece.repair_state == HskRepairState::Accepted {
            repair_state = HskRepairState::Accepted;
        }
    }
    let level = ControlHskLevel::new(u8::from(requested_level))
        .map_err(|error| PipelineError::new("INVALID_HSK_LEVEL", error.to_string()))?;
    let protected_names = accumulator
        .pieces
        .values()
        .flat_map(|(piece, _)| piece.protected_names.iter().cloned())
        .collect::<Vec<_>>();
    let names = FaithfulText {
        text: displayed_chinese.clone(),
        protected_names: protected_names.clone(),
    }
    .proper_names();
    let report = control.validate(&displayed_chinese, level, &names);
    if learning_policy_requires_repair(&report, learning_mode) {
        accumulator.reject("joined document block failed whole-block HSK validation".to_owned());
        return publish_document_accumulator(
            sink,
            control,
            requested_level,
            learning_mode,
            accumulator,
            chapter_contexts,
            chapter_id,
        );
    }
    let mut joined = CachedTranslation {
        protected_names,
        base_chinese,
        displayed_chinese: report.normalized_text.clone(),
        pinyin: String::new(),
        report,
        repair_state,
    };
    populate_pinyin(control, &mut joined);
    let hsk = TranslatedHskStatus {
        requested_level,
        learning_mode,
        strictly_valid: joined.report.strictly_valid,
        level_coverage: level_coverage(&joined.report),
        above_level_tokens: above_level_tokens(&joined.report),
        teaching_terms: teaching_terms(control, &joined.report),
        repair_state: joined.repair_state,
    };

    sink.publish(JobUpdateDraft::DocumentBlockReady {
        block: DocumentBlockReady {
            parent_block_id: accumulator.block.parent_block_id.clone(),
            sub_item_order: accumulator.block.sub_item_order,
            item_id: accumulator.block.item_id.clone(),
            source_index: accumulator.block.source_index,
            item_order: accumulator.block.item_order,
            kind: accumulator.block.kind,
            text: TranslatedText {
                source_text: accumulator.block.text.clone(),
                termination: koharu_llm::GenerationTermination::Stop,
                protected_names: joined.protected_names.clone(),
                base_chinese: joined.base_chinese,
                displayed_chinese: joined.displayed_chinese.clone(),
                pinyin: joined.pinyin,
                hsk,
            },
        },
    })
    .map_err(|error| publish_error(error, sink))?;
    chapter_contexts
        .lock()
        .map_err(|_| {
            PipelineError::new("CHAPTER_CONTEXT_FAILED", "Chapter context lock poisoned.")
        })?
        .publish(
            chapter_id,
            (accumulator.block.source_index, accumulator.block.item_order),
            &accumulator.block.item_id,
            joined.displayed_chinese,
        );
    accumulator.published = true;
    Ok(())
}

#[derive(Clone)]
struct PreparedRegion {
    id: String,
    candidate: Candidate,
    source_english: String,
    /// Established by the text-only semantic stage after visual role
    /// classification, while verified cleanup runs independently.
    faithful_chinese: Option<FaithfulText>,
    ocr_confidence: f32,
    reading_order: u32,
    /// Canonical chapter graph link assigned by page understanding.  This is
    /// The page model supplies this link; deterministic geometry never
    /// invents continuation groups from timing or completion order.
    continuation_group: Option<String>,
    role: ImageRegionRole,
    source_line_count: usize,
    prediction: PpOcrPrediction,
    appearance_bands: Vec<SourceAppearanceBand>,
    measured_font_height: f32,
    bubble_polygon: Vec<Point>,
    layout_polygon: Vec<Point>,
    /// Cleanup is intentionally a chapter-page task rather than part of the
    /// detector critical path. Translation can run while the verified patch
    /// is being produced; publication awaits this handle so an unverified
    /// source is never overwritten.
    cleanup: Arc<CleanupBatchTask>,
    visible: bool,
    translation_queued_at: tokio::time::Instant,
    source_context: SourceContext,
}

struct AnalyzedPage {
    regions: Vec<PreparedRegion>,
    plans: Vec<RegionPlan>,
    preserved: Vec<ImageRegionPreserved>,
}
impl CacheValue for Arc<AnalyzedPage> {
    fn retained_bytes(&self) -> usize {
        self.regions
            .iter()
            .map(|region| {
                let patches = region
                    .cleanup
                    .result
                    .get()
                    .map(|result| {
                        result
                            .decisions
                            .values()
                            .map(|decision| {
                                decision.patch.as_ref().map_or(0, |patch| patch.bytes.len())
                            })
                            .sum::<usize>()
                    })
                    .unwrap_or(usize::MAX / 2);
                patches
                    .saturating_add(region.source_english.len())
                    .saturating_add(
                        region
                            .faithful_chinese
                            .as_ref()
                            .map_or(0, CacheValue::retained_bytes),
                    )
                    .saturating_add(4_096)
            })
            .fold(0usize, usize::saturating_add)
            .saturating_add(self.plans.len().saturating_mul(2_048))
            .saturating_add(self.preserved.len().saturating_mul(2_048))
    }
}
fn image_analysis_key(request: &ImagePipelineInput) -> String {
    sha256_hex(
        &serde_json::to_vec(&(
            "verified-page-analysis-v2",
            crate::contracts::BUILD_FINGERPRINT,
            &request.source_sha256,
            request.natural_width,
            request.natural_height,
            request.reading_direction,
            &request.surrounding_context,
        ))
        .expect("source snapshot is serializable"),
    )
}

struct CleanupBatchTask {
    receiver: AsyncMutex<Option<oneshot::Receiver<Arc<CleanupBatchResult>>>>,
    result: OnceCell<Arc<CleanupBatchResult>>,
    // Verified cleanup overlaps HSK realization. If the bounded publication
    // wait expires, abort the detached task instead of letting an orphaned
    // inpaint job occupy the CUDA scheduler after the region is terminal.
    abort: Option<tokio::task::AbortHandle>,
}

#[derive(Debug)]
struct CleanupBatchResult {
    decisions: HashMap<String, CleanupDecision>,
}

#[derive(Debug, Clone)]
struct CleanupDecision {
    patch: Option<PatchPng>,
    reason: Option<String>,
    quality: Option<CleanupQuality>,
}

impl CleanupBatchTask {
    #[cfg(test)]
    fn ready(result: CleanupBatchResult) -> Arc<Self> {
        let result_cell = OnceCell::new();
        let _ = result_cell.set(Arc::new(result));
        Arc::new(Self {
            receiver: AsyncMutex::new(None),
            result: result_cell,
            abort: None,
        })
    }

    fn spawn<F>(task: F) -> Arc<Self>
    where
        F: std::future::Future<Output = CleanupBatchResult> + Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _ = sender.send(Arc::new(task.await));
        });
        let handle = Arc::new(Self {
            receiver: AsyncMutex::new(Some(receiver)),
            result: OnceCell::new(),
            abort: Some(task.abort_handle()),
        });
        handle
    }

    fn cancel(&self) {
        if let Some(abort) = &self.abort {
            abort.abort();
        }
    }

    async fn result(&self) -> Arc<CleanupBatchResult> {
        self.result
            .get_or_init(|| async {
                let receiver = self.receiver.lock().await.take();
                match receiver {
                    Some(receiver) => {
                        match tokio::time::timeout(CLEANUP_RESULT_TIMEOUT, receiver).await {
                            Ok(Ok(result)) => result,
                            // A timeout or a dropped producer is terminal for this
                            // cleanup candidate.  Stop the detached task before
                            // returning the empty decision set so it cannot keep
                            // consuming GPU/CPU capacity after the caller moves
                            // on to the next page.
                            Ok(Err(_)) | Err(_) => {
                                self.cancel();
                                Arc::new(CleanupBatchResult {
                                    decisions: HashMap::new(),
                                })
                            }
                        }
                    }
                    None => Arc::new(CleanupBatchResult {
                        decisions: HashMap::new(),
                    }),
                }
            })
            .await
            .clone()
    }
}

impl Drop for CleanupBatchTask {
    fn drop(&mut self) {
        if let Some(abort) = &self.abort {
            abort.abort();
        }
    }
}

#[derive(Debug)]
struct RecognizedLine {
    candidate: Candidate,
    prediction: PpOcrPrediction,
    crop_bounds: PixelBounds,
}

#[derive(Debug)]
struct RejectedOcrLine {
    candidate: Candidate,
    prediction: PpOcrPrediction,
}

#[derive(Debug, Default)]
struct OcrBatchResult {
    accepted: Vec<RecognizedLine>,
    rejected: Vec<RejectedOcrLine>,
}

struct BubbleMaskCache {
    completed_tiles: HashSet<usize>,
    union: image::GrayImage,
    /// Labels are derived from the accumulated union. Progressive pages can
    /// invoke preparation repeatedly without changing that union, so retain
    /// the full-page connected-component result until new tiles are merged.
    labels: Option<Arc<image::GrayImage>>,
    component_bounds: Option<Arc<BTreeMap<u8, PixelRect>>>,
}

impl BubbleMaskCache {
    fn new(image_width: u32, image_height: u32) -> Self {
        Self {
            completed_tiles: HashSet::new(),
            union: image::GrayImage::new(image_width, image_height),
            labels: None,
            component_bounds: None,
        }
    }

    fn invalidate_labels(&mut self) {
        self.labels = None;
        self.component_bounds = None;
    }

    fn labels(&mut self) -> Arc<image::GrayImage> {
        if self.labels.is_none() {
            self.labels = Some(Arc::new(label_bubble_components(&self.union)));
        }
        self.labels
            .as_ref()
            .expect("bubble labels initialized above")
            .clone()
    }

    fn component_bounds(&mut self) -> Arc<BTreeMap<u8, PixelRect>> {
        if self.component_bounds.is_none() {
            let labels = self.labels();
            self.component_bounds = Some(Arc::new(bubble_component_bounds(labels.as_ref())));
        }
        self.component_bounds
            .as_ref()
            .expect("bubble component bounds initialized above")
            .clone()
    }
}

fn text_rects_represent_same_block(left: PixelRect, right: PixelRect) -> bool {
    left.iou(right) >= 0.35 || left.overlap_over_smaller(right) >= 0.60
}

fn recognized_line_quality(line: &RecognizedLine) -> (u32, u32) {
    // Overlapping detector tiles can produce two equivalent transcripts for
    // one line. Choose the measured model evidence first; never let a
    // longer alphabetic string outrank a shorter but more reliable sequence
    // (the old rule admitted plausible letter soup). Structural evidence only
    // breaks a confidence tie.
    (
        (line.prediction.confidence.clamp(0.0, 1.0) * 1_000_000.0).round() as u32,
        (line.candidate.detector_confidence.clamp(0.0, 1.0) * 1_000_000.0).round() as u32,
    )
}

#[derive(Debug, Clone)]
struct GroupedRegion {
    candidate: Candidate,
    reading_order: u32,
    source_english: String,
    faithful_chinese: Option<FaithfulText>,
    ocr_confidence: f32,
    continuation_group: Option<String>,
    role: ImageRegionRole,
    source_line_count: usize,
    prediction: PpOcrPrediction,
    appearance_bands: Vec<SourceAppearanceBand>,
    measured_font_height: f32,
    cleanup_blocks: Vec<TextRegion>,
}

#[derive(Debug, Clone)]
struct CleanedGroupedRegion {
    group: GroupedRegion,
    cleanup_mask: CleanupMask,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceAppearanceBand {
    position_millionths: u32,
    text_color: [u8; 3],
    stroke_color: [u8; 3],
    has_stroke_color: bool,
}

struct PreprocessingPool {
    pool: ThreadPool,
}

impl PreprocessingPool {
    fn new() -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(PREPROCESSING_THREADS)
            .thread_name(|index| format!("hsk-browser-preprocess-{index}"))
            .build()
            .context("create dedicated six-thread browser preprocessing pool")?;
        Ok(Self { pool })
    }

    fn start<T, F>(&self, task: F) -> oneshot::Receiver<Result<T>>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let (send, receive) = oneshot::channel();
        self.pool.spawn(move || {
            let _ = send.send(task());
        });
        receive
    }

    async fn run<T, F>(&self, task: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        self.start(task)
            .await
            .context("browser preprocessing worker stopped before returning its result")?
    }

    #[cfg(test)]
    fn thread_count(&self) -> usize {
        self.pool.current_num_threads()
    }
}

struct TileBatchTask {
    tiles: Vec<Tile>,
    receive: oneshot::Receiver<Result<Vec<DynamicImage>>>,
}

impl TileBatchTask {
    fn start(
        preprocessing: &PreprocessingPool,
        source: Arc<DynamicImage>,
        tiles: Vec<Tile>,
    ) -> Self {
        let tiles_for_crops = tiles.clone();
        let receive = preprocessing.start(move || {
            Ok(tiles_for_crops
                .iter()
                .map(|tile| source.crop_imm(tile.x, tile.y, tile.width, tile.height))
                .collect::<Vec<_>>())
        });
        Self { tiles, receive }
    }

    async fn finish(self) -> Result<(Vec<Tile>, Vec<DynamicImage>)> {
        let images = self
            .receive
            .await
            .context("browser preprocessing worker stopped before returning detector crops")??;
        Ok((self.tiles, images))
    }
}

fn tiles_start_with(remaining: &[Tile], prepared: &[Tile]) -> bool {
    !prepared.is_empty()
        && prepared.len() <= remaining.len()
        && remaining
            .iter()
            .zip(prepared)
            .all(|(left, right)| left.id == right.id)
}

fn global_preprocessing_pool() -> Result<Arc<PreprocessingPool>> {
    match PREPROCESSING_POOL.get_or_init(|| {
        PreprocessingPool::new()
            .map(Arc::new)
            .map_err(|error| format!("{error:#}"))
    }) {
        Ok(pool) => Ok(pool.clone()),
        Err(error) => bail!("{error}"),
    }
}

async fn ocr_batch(
    resident: &VisionRuntime,
    source: Arc<DynamicImage>,
    candidates: &mut Vec<Candidate>,
    proposal_source: OcrProposalSource,
    request: &ImagePipelineInput,
    sink: &JobUpdateSink,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    preprocessing: &Arc<PreprocessingPool>,
    text_probabilities: &ProbabilityMap,
) -> std::result::Result<OcrBatchResult, PipelineError> {
    let (image_width, image_height) = source.dimensions();
    if candidates.is_empty() {
        return Ok(OcrBatchResult::default());
    }
    cancellation_boundary(cancel.as_ref())?;
    if sink.is_cancelled() {
        return Err(PipelineError::cancelled());
    }
    let viewport = sink.focus();
    let ranks = reading_order_ranks(
        &candidates
            .iter()
            .map(|candidate| candidate.text_rect)
            .collect::<Vec<_>>(),
        request.reading_direction,
    );
    let mut ranked_candidates = std::mem::take(candidates)
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    ranked_candidates.sort_by_key(|(index, candidate)| {
        let visible = viewport.active
            && candidate.bubble_rect.intersects_viewport(
                &viewport.visible_rects,
                image_width,
                image_height,
            );
        (!visible, ranks[*index])
    });
    *candidates = ranked_candidates
        .into_iter()
        .map(|(_, candidate)| candidate)
        .collect();
    let count = OCR_REGION_BATCH_SIZE.min(candidates.len());
    let candidate_chunk = candidates.drain(..count).collect::<Vec<_>>();
    let admission_viewport = sink.focus();
    let cuda_priority = if admission_viewport.active
        && candidate_chunk.iter().any(|candidate| {
            candidate.bubble_rect.intersects_viewport(
                &admission_viewport.visible_rects,
                image_width,
                image_height,
            )
        }) {
        CudaPriority::Visible
    } else {
        CudaPriority::Offscreen
    };
    let source_for_crops = source.clone();
    let candidates_for_crops = candidate_chunk.clone();
    let prepared_crops = preprocessing
        .run(move || {
            Ok(candidates_for_crops
                .iter()
                .map(|candidate| {
                    let bounds = ocr_crop_rect(candidate, image_width, image_height)
                        .pixel_bounds(image_width, image_height);
                    (
                        rectify_ocr_crop(
                            source_for_crops.crop_imm(
                                bounds.x,
                                bounds.y,
                                bounds.width,
                                bounds.height,
                            ),
                            candidate.rotation_radians,
                        ),
                        bounds,
                    )
                })
                .collect::<Vec<_>>())
        })
        .await
        .context("prepare OCR crops on the browser preprocessing pool")
        .map_err(PipelineError::pipeline)?;
    let (crops, crop_bounds): (Vec<_>, Vec<_>) = prepared_crops.into_iter().unzip();
    let crop_text_probabilities = crop_bounds
        .iter()
        .map(|bounds| crop_probability_map(text_probabilities, *bounds))
        .collect::<Vec<_>>();
    cancellation_boundary(cancel.as_ref())?;
    let cuda_permit = cuda_scheduler
        .acquire(CudaWorkload::Vision, cuda_priority, cancel.clone())
        .await
        .map_err(cuda_admission_error)?;
    resident.instrumentation.invoked(RuntimeComponent::Ocr);
    let predictions = {
        let mut ocr = resident
            .ocr
            .lock()
            .map_err(|_| PipelineError::new("MODEL_STATE_FAILED", "OCR model lock poisoned."))?;
        ocr.recognize_regions_with_consensus(&crops, &crop_text_probabilities)
            .context("run batched CUDA PP-OCR small recognition with calibrated consensus")
            .map_err(PipelineError::pipeline)?
    };
    drop(cuda_permit);
    cancellation_boundary(cancel.as_ref())?;
    if predictions.len() != candidate_chunk.len() {
        return Err(PipelineError::new(
            "OCR_FAILED",
            "OCR returned an incomplete region batch.",
        ));
    }
    let mut result = OcrBatchResult::default();
    for ((candidate, prediction), crop_bounds) in candidate_chunk
        .into_iter()
        .zip(predictions)
        .zip(crop_bounds)
    {
        let accepted =
            accept_english_ocr_line(prediction.confidence, &prediction.text, proposal_source);
        if !accepted {
            if rejected_ocr_tracing_enabled() {
                eprintln!(
                    "hskify-ocr-rejected-line source={} rect={:.1},{:.1},{:.1},{:.1} confidence={:.4} text={:?}",
                    &request.source_sha256[..8],
                    candidate.text_rect.x0,
                    candidate.text_rect.y0,
                    candidate.text_rect.x1,
                    candidate.text_rect.y1,
                    prediction.confidence,
                    prediction.text,
                );
            }
            result.rejected.push(RejectedOcrLine {
                candidate,
                prediction,
            });
            continue;
        }
        // PP-OCR owns the line polygon. Keep one immutable recognition record
        // per detector proposal; punctuation or casing must never invent a
        // second region from one crop. Multiple hypotheses, when present,
        // remain evidence for the page adjudicator.
        result.accepted.push(RecognizedLine {
            candidate: candidate_with_ocr_extent(candidate, &prediction, crop_bounds),
            prediction,
            crop_bounds,
        });
    }
    Ok(result)
}

/// Rectify the line view using the independent detector's measured principal
/// axis. The source image itself is never rotated; this is a recognizer-only
/// view, so cleanup and layout continue to use the original pixel geometry.
fn rectify_ocr_crop(crop: DynamicImage, rotation_radians: f32) -> DynamicImage {
    if !rotation_radians.is_finite() || rotation_radians.abs() < 0.08 {
        return crop;
    }
    let angle = rotation_radians.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
    let rgb = crop.to_rgb8();
    DynamicImage::ImageRgb8(rotate_about_center(
        &rgb,
        -angle,
        Interpolation::Bilinear,
        Border::Replicate,
    ))
}

fn candidate_with_ocr_extent(
    mut candidate: Candidate,
    prediction: &PpOcrPrediction,
    crop_bounds: PixelBounds,
) -> Candidate {
    // Appearance bands are measured in the rectified recognizer view. Their
    // top/bottom coordinates cannot be projected back onto page geometry
    // without the detector polygon, so keep the original detector bounds for
    // rotated text instead of applying a misleading axis-aligned expansion.
    if prediction.appearance_bands.is_empty() || candidate.rotation_radians.abs() >= 0.08 {
        return candidate;
    }
    let crop_top = crop_bounds.y as f32;
    let crop_height = crop_bounds.height.max(1) as f32;
    let recovered_top = prediction
        .appearance_bands
        .iter()
        .enumerate()
        .filter(|(index, band)| {
            appearance_band_is_owned_by_candidate(
                &candidate,
                crop_bounds,
                Some(band),
                prediction.ocr_lines.get(*index),
            )
        })
        .map(|(_, band)| crop_top + band.top_ratio.clamp(0.0, 1.0) * crop_height)
        .fold(candidate.text_rect.y0, f32::min);
    let recovered_bottom = prediction
        .appearance_bands
        .iter()
        .enumerate()
        .filter(|(index, band)| {
            appearance_band_is_owned_by_candidate(
                &candidate,
                crop_bounds,
                Some(band),
                prediction.ocr_lines.get(*index),
            )
        })
        .map(|(_, band)| crop_top + band.bottom_ratio.clamp(0.0, 1.0) * crop_height)
        .fold(candidate.text_rect.y1, f32::max);
    if recovered_top < candidate.text_rect.y0 || recovered_bottom > candidate.text_rect.y1 {
        candidate.text_rect.y0 = recovered_top.min(candidate.text_rect.y0);
        candidate.text_rect.y1 = recovered_bottom.max(candidate.text_rect.y1);
        candidate.bubble_rect = candidate.bubble_rect.union(candidate.text_rect);
    }
    candidate
}

/// Run independent bounded page-understanding windows in canonical order.
/// Windows do not overlap and decisions are never merged or revised: each
/// source region has exactly one semantic owner. A malformed model record is
/// returned as a failed region while valid siblings remain usable.
async fn adjudicate_grouped_page(
    resident: Arc<VisionRuntime>,
    source: Arc<DynamicImage>,
    grouped: &[GroupedRegion],
    request: &ImagePipelineInput,
    priority: CudaPriority,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    image_width: u32,
    image_height: u32,
) -> std::result::Result<PageUnderstandingResult, PipelineError> {
    let window_size = koharu_llm::page_understanding::MAX_PAGE_REGIONS;
    let mut regions = Vec::with_capacity(grouped.len());
    let mut failed_region_ids = Vec::new();
    for window in grouped.chunks(window_size) {
        cancellation_boundary(cancel.as_ref())?;
        let result = adjudicate_page_window(
            Arc::clone(&resident),
            source.clone(),
            window,
            request,
            priority,
            cancel.clone(),
            cuda_scheduler,
            image_width,
            image_height,
        )
        .await;
        match result {
            Ok(result) => {
                regions.extend(result.regions);
                failed_region_ids.extend(result.failed_region_ids);
            }
            Err(_) => {
                cancellation_boundary(cancel.as_ref())?;
                failed_region_ids.extend(window.iter().map(|group| {
                    stable_region_id(&request.source_sha256, group.candidate.text_rect)
                }));
            }
        }
    }
    Ok(PageUnderstandingResult {
        regions,
        failed_region_ids,
    })
}

/// Translate only the OCR-accepted candidates whose visual roles are story or
/// SFX. The caller starts the independent cleanup task first, so Language work
/// overlaps that one Vision pass without spending tokens on furniture,
/// artwork, covers, or malformed role records.
async fn translate_faithful_candidates(
    resident: Arc<VisionRuntime>,
    grouped: &[GroupedRegion],
    request: &ImagePipelineInput,
    _preceding_context: &[HskPrecedingUtterance],
    priority: CudaPriority,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    sink: &JobUpdateSink,
) -> std::result::Result<HashMap<String, FaithfulText>, PipelineError> {
    let service = TranslationService {
        language: resident.language.as_ref(),
        control: resident.language.hsk_control.as_ref(),
        cuda_scheduler,
    };
    let mut translations = HashMap::with_capacity(grouped.len());
    for (index, group) in grouped.iter().enumerate() {
        cancellation_boundary(cancel.as_ref())?;
        let source_context = image_source_context(grouped, index, request);
        let id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
        let faithful = service
            .establish_faithful(
                vec![FaithfulSourceUtterance {
                    id: id.clone(),
                    kind: match group.role {
                        ImageRegionRole::System => HskUtteranceKind::Sfx,
                        ImageRegionRole::Narration => HskUtteranceKind::Caption,
                        ImageRegionRole::Dialogue => HskUtteranceKind::Dialogue,
                    },
                    source_english: group.source_english.clone(),
                }],
                Vec::new(),
                source_context.preceding,
                source_context.following,
                DirectSourceProvenance::Ocr,
                priority,
                cancel.clone(),
            )
            .await?;
        let translator = resident.language.app.llm.direct_hsk_translator();
        record_image_language_evidence(
            sink.job_id(),
            format!(
                "{}@{}:native-llama-tokenizer",
                translator.model_id(),
                translator.model_revision()
            ),
            1,
            faithful.generation_duration,
        )
        .map_err(PipelineError::pipeline)?;
        translations.extend(faithful.items);
    }
    Ok(translations)
}

/// The same immutable English snapshot supplies faithful and HSK generation and their cache identities.
fn image_source_context(
    grouped: &[GroupedRegion],
    index: usize,
    request: &ImagePipelineInput,
) -> SourceContext {
    let position = request
        .chapter_source_order
        .iter()
        .position(|index| *index == request.source_index)
        .unwrap_or(0);
    let before = |unit: &&ChapterContextUnit| {
        request
            .chapter_source_order
            .iter()
            .position(|index| *index == unit.source_index)
            .is_some_and(|index| index < position)
    };
    let after = |unit: &&ChapterContextUnit| {
        request
            .chapter_source_order
            .iter()
            .position(|index| *index == unit.source_index)
            .is_some_and(|index| index > position)
    };
    let preceding = request
        .surrounding_context
        .iter()
        .filter(before)
        .map(|unit| &unit.source_text)
        .chain(grouped[..index].iter().map(|group| &group.source_english))
        .rev()
        .take(MAX_HSK_PRECEDING_UTTERANCES)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let following = grouped[index + 1..]
        .iter()
        .map(|group| &group.source_english)
        .chain(
            request
                .surrounding_context
                .iter()
                .filter(after)
                .map(|unit| &unit.source_text),
        )
        .take(MAX_HSK_PRECEDING_UTTERANCES)
        .cloned()
        .collect();
    SourceContext {
        preceding,
        following,
    }
}

/// Execute one bounded page-understanding request.
#[allow(clippy::too_many_arguments)]
async fn adjudicate_page_window(
    resident: Arc<VisionRuntime>,
    source: Arc<DynamicImage>,
    grouped: &[GroupedRegion],
    request: &ImagePipelineInput,
    priority: CudaPriority,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    image_width: u32,
    image_height: u32,
) -> std::result::Result<PageUnderstandingResult, PipelineError> {
    let (evidence_surface, evidence_viewport) =
        page_evidence_surface(&source, grouped, image_width, image_height);
    let evidence = PageUnderstandingRequest {
        image: evidence_surface,
        regions: grouped
            .iter()
            .map(|group| {
                page_region_evidence(group, grouped, &request.source_sha256, evidence_viewport)
            })
            .collect(),
    };
    cancellation_boundary(cancel.as_ref())?;
    let permit = cuda_scheduler
        .acquire(CudaWorkload::Vision, priority, cancel.clone())
        .await
        .map_err(cuda_admission_error)?;
    resident.instrumentation.invoked(RuntimeComponent::Vision);
    resident
        .instrumentation
        .invoked(RuntimeComponent::Projector);
    let result =
        tokio::task::spawn_blocking(move || {
            let mut model = resident.page_understanding.lock().map_err(|_| {
                PipelineError::new(
                    "MODEL_STATE_FAILED",
                    "Qwen3.5 page-understanding model lock poisoned.",
                )
            })?;
            model.analyze(&evidence).map_err(|error| {
                PipelineError::pipeline(error.context(
                    "Qwen3.5 page understanding did not return a complete validated decision",
                ))
            })
        })
        .await
        .map_err(|error| {
            PipelineError::pipeline(anyhow!(
                "Qwen3.5 page-understanding worker did not complete: {error}"
            ))
        })?;
    drop(permit);
    result
}

fn page_region_evidence(
    group: &GroupedRegion,
    grouped: &[GroupedRegion],
    source_sha256: &str,
    evidence_viewport: PixelRect,
) -> PageRegionEvidence {
    let has_confirmed_bubble = group.candidate.has_detector_core;
    PageRegionEvidence {
        id: stable_region_id(source_sha256, group.candidate.text_rect),
        source_english: group.source_english.clone(),
        polygon: polygon_in_evidence_viewport(group.candidate.text_rect, evidence_viewport)
            .into_iter()
            .map(|point| PagePoint {
                x: point.x,
                y: point.y,
            })
            .collect(),
        confidence: group.ocr_confidence.clamp(0.0, 1.0),
        reading_order: group.reading_order as usize,
        // Bubble ownership requires an actual detector bubble containing the
        // text. A detector's dialogue class by itself is not a bubble: using
        // its padded text rectangle as one biases short device labels and
        // signs toward dialogue.
        bubble_id: has_confirmed_bubble
            .then(|| stable_region_id(source_sha256, group.candidate.confirmed_bubble_rect)),
        connected_region_ids: has_confirmed_bubble
            .then(|| {
                grouped
                    .iter()
                    .filter(|other| {
                        !std::ptr::eq(*other, group)
                            && other.candidate.has_detector_core
                            && group
                                .candidate
                                .confirmed_bubble_rect
                                .overlap_over_smaller(other.candidate.confirmed_bubble_rect)
                                >= 0.50
                    })
                    .map(|other| stable_region_id(source_sha256, other.candidate.text_rect))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
    }
}

/// Select a bounded visual evidence surface for one ordered language window.
///
/// OCR and bubble geometry are already independently accepted before this
/// function runs.  Their union is therefore the only source of the crop; no
/// title, page, or reader-specific crop is possible.  Small ordinary pages
/// continue to use the complete surface.  Tall strips and sparse windows use
/// an expanded local viewport. OCR already supplies the immutable transcript,
/// so the projector receives a bounded-resolution role surface that retains
/// bubble borders, objects, and nearby artwork without paying to reread every
/// source glyph at full resolution.
fn page_evidence_surface(
    source: &Arc<DynamicImage>,
    grouped: &[GroupedRegion],
    image_width: u32,
    image_height: u32,
) -> (Arc<DynamicImage>, PixelRect) {
    let full = PixelRect::new(0.0, 0.0, image_width as f32, image_height as f32)
        .expect("decoded page surface must have non-zero dimensions");
    let Some(first) = grouped.first() else {
        return (bounded_page_role_surface(Arc::clone(source)), full);
    };
    let evidence = grouped.iter().skip(1).fold(
        first
            .candidate
            .confirmed_bubble_rect
            .union(first.candidate.text_rect),
        |bounds, group| {
            bounds.union(
                group
                    .candidate
                    .confirmed_bubble_rect
                    .union(group.candidate.text_rect),
            )
        },
    );
    let largest_source_glyph = grouped
        .iter()
        .map(|group| group.measured_font_height.max(1.0))
        .fold(1.0, f32::max);
    let margin =
        (largest_source_glyph * 8.0).clamp(PAGE_EVIDENCE_MIN_MARGIN, PAGE_EVIDENCE_MAX_MARGIN);
    let viewport = evidence.expand(margin, image_width, image_height);
    let bounds = viewport.pixel_bounds(image_width, image_height);
    let full_pixels = u64::from(image_width) * u64::from(image_height);
    let viewport_pixels = u64::from(bounds.width) * u64::from(bounds.height);
    let sparse_enough = (viewport_pixels as f32) < full_pixels as f32 * PAGE_EVIDENCE_CROP_RATIO;
    let over_budget = full_pixels > PAGE_EVIDENCE_MAX_PIXELS;
    if bounds.width == 0
        || bounds.height == 0
        || (!sparse_enough && !over_budget)
        || (bounds.width == image_width && bounds.height == image_height)
    {
        return (bounded_page_role_surface(Arc::clone(source)), full);
    }
    let cropped = source.crop_imm(bounds.x, bounds.y, bounds.width, bounds.height);
    let crop_viewport = PixelRect::new(
        bounds.x as f32,
        bounds.y as f32,
        (bounds.x + bounds.width) as f32,
        (bounds.y + bounds.height) as f32,
    )
    .expect("non-empty crop bounds must form a valid viewport");
    (bounded_page_role_surface(Arc::new(cropped)), crop_viewport)
}

fn bounded_page_role_surface(surface: Arc<DynamicImage>) -> Arc<DynamicImage> {
    let longest = surface.width().max(surface.height());
    if longest <= PAGE_ROLE_MAX_LONG_EDGE {
        return surface;
    }
    let scale = PAGE_ROLE_MAX_LONG_EDGE as f64 / longest as f64;
    let width = (surface.width() as f64 * scale).round().max(1.0) as u32;
    let height = (surface.height() as f64 * scale).round().max(1.0) as u32;
    Arc::new(surface.resize_exact(width, height, FilterType::Triangle))
}

fn polygon_in_evidence_viewport(rect: PixelRect, viewport: PixelRect) -> Vec<Point> {
    let local = PixelRect::new(
        rect.x0 - viewport.x0,
        rect.y0 - viewport.y0,
        rect.x1 - viewport.x0,
        rect.y1 - viewport.y0,
    )
    .and_then(|candidate| {
        candidate.intersection(PixelRect::new(
            0.0,
            0.0,
            viewport.width(),
            viewport.height(),
        )?)
    })
    .unwrap_or_else(|| {
        PixelRect::new(
            0.0,
            0.0,
            viewport.width().max(1.0),
            viewport.height().max(1.0),
        )
        .expect("evidence viewport has non-zero dimensions")
    });
    local.polygon(
        viewport.width().round().max(1.0) as u32,
        viewport.height().round().max(1.0) as u32,
    )
}

fn apply_page_adjudication_transcripts(
    grouped: &mut [GroupedRegion],
    request: &ImagePipelineInput,
    result: &PageUnderstandingResult,
) -> HashMap<String, PageRegionRole> {
    let decisions = result
        .regions
        .iter()
        .map(|decision| (decision.id.as_str(), decision))
        .collect::<HashMap<_, _>>();
    let mut roles = HashMap::with_capacity(result.regions.len());
    for group in grouped {
        let id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
        let Some(decision) = decisions.get(id.as_str()) else {
            continue;
        };
        // OCR is the transcript authority. The semantic model receives but
        // cannot emit or revise this text; the validated result carries back
        // the exact request transcript for a single typed hand-off.
        debug_assert_eq!(group.source_english, decision.transcript);
        // Keep continuation topology as typed model evidence.  It is later
        // committed to DialogueGraph in document order, so connected bubbles
        // remain connected even when page work finishes out of order.
        group.continuation_group = decision.continuation_of.clone();
        group.role = match decision.role {
            PageRegionRole::Sfx => ImageRegionRole::System,
            PageRegionRole::Story => group.role,
            PageRegionRole::Furniture | PageRegionRole::Artwork => group.role,
        };
        roles.insert(id, decision.role);
    }
    roles
}

async fn prepare_grouped_regions(
    resident: Arc<VisionRuntime>,
    source: Arc<DynamicImage>,
    lines: Vec<RecognizedLine>,
    request: &ImagePipelineInput,
    preceding_context: &[HskPrecedingUtterance],
    sink: &JobUpdateSink,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    preprocessing: &Arc<PreprocessingPool>,
    bubble_masks: &mut BubbleMaskCache,
    text_mask_completed_tiles: &mut HashSet<usize>,
    mut text_probabilities: ProbabilityMap,
    overall_progress: f32,
) -> std::result::Result<(Vec<PreparedRegion>, ProbabilityMap, Vec<RegionPlan>), PipelineError> {
    if lines.is_empty() {
        return Ok((Vec::new(), text_probabilities, Vec::new()));
    }
    let prepare_started = Instant::now();
    let (image_width, image_height) = source.dimensions();
    let bubble_supports = lines
        .iter()
        .map(|line| {
            line.candidate
                .confirmed_bubble_rect
                .union(line.candidate.text_rect)
                .expand(
                    (line.candidate.text_rect.height() * 2.0).clamp(48.0, 192.0),
                    image_width,
                    image_height,
                )
        })
        .collect::<Vec<_>>();
    let bubble_tiles = overlapping_tiles(image_width, image_height)
        .into_iter()
        .filter(|tile| {
            !bubble_masks.completed_tiles.contains(&tile.id)
                && bubble_supports
                    .iter()
                    .any(|support| tile.rect().intersection(*support).is_some())
        })
        .collect::<Vec<_>>();
    let tiles_for_crops = bubble_tiles.clone();
    let source_for_crops = source.clone();
    let bubble_crops = preprocessing
        .run(move || {
            Ok(tiles_for_crops
                .iter()
                .map(|tile| source_for_crops.crop_imm(tile.x, tile.y, tile.width, tile.height))
                .collect::<Vec<_>>())
        })
        .await
        .context("prepare semantic cleanup tiles")
        .map_err(PipelineError::pipeline)?;
    cancellation_boundary(cancel.as_ref())?;
    let viewport = sink.focus();
    let bubble_priority = if viewport.active
        && lines.iter().any(|line| {
            line.candidate.bubble_rect.intersects_viewport(
                &viewport.visible_rects,
                image_width,
                image_height,
            )
        }) {
        CudaPriority::Visible
    } else {
        CudaPriority::Offscreen
    };
    let bubble_started = Instant::now();
    if !bubble_tiles.is_empty() {
        if bubble_crops.len() != bubble_tiles.len() {
            return Err(PipelineError::new(
                "BUBBLE_SEGMENTATION_FAILED",
                "Speech bubble cleanup prepared an incomplete tile batch.",
            ));
        }
        // Keep cleanup admission bounded just like detector work. A large
        // tail can require many contour tiles;
        // holding Vision for the entire batch would prevent visible work from
        // overtaking it until every offscreen contour is decoded.
        for (tile_batch, crop_batch) in bubble_tiles
            .chunks(DETECTOR_TILE_BATCH_SIZE)
            .zip(bubble_crops.chunks(DETECTOR_TILE_BATCH_SIZE))
        {
            cancellation_boundary(cancel.as_ref())?;
            let bubble_permit = cuda_scheduler
                .acquire(CudaWorkload::Vision, bubble_priority, cancel.clone())
                .await
                .map_err(cuda_admission_error)?;
            resident
                .instrumentation
                .invoked(RuntimeComponent::Segmentation);
            let results = {
                let bubble_segmenter = resident.bubble_segmenter.lock().map_err(|_| {
                    PipelineError::new("MODEL_STATE_FAILED", "Bubble segmenter lock poisoned.")
                })?;
                bubble_segmenter
                    .inference_batch(crop_batch)
                    .context("batch-segment speech bubble contours")
                    .map_err(PipelineError::pipeline)?
            };
            drop(bubble_permit);
            if results.len() != tile_batch.len() {
                return Err(PipelineError::new(
                    "BUBBLE_SEGMENTATION_FAILED",
                    "Speech bubble segmentation returned an incomplete tile batch.",
                ));
            }
            for (tile, result) in tile_batch.iter().zip(results) {
                merge_binary_mask(
                    &mut bubble_masks.union,
                    &bubble_id_mask(&result),
                    tile.x,
                    tile.y,
                );
                bubble_masks.completed_tiles.insert(tile.id);
            }
        }
        bubble_masks.invalidate_labels();
    }
    let bubble_mask = bubble_masks.labels();
    let bubble_components = bubble_masks.component_bounds();
    let bubble_elapsed = bubble_started.elapsed();
    let groups = group_recognized_lines(lines, bubble_mask.as_ref());
    let mut grouped = preprocessing
        .run(move || {
            Ok(groups
                .into_iter()
                .filter_map(|group| {
                    let source_english = grouped_source_english(&group);
                    if source_english.is_empty() {
                        return None;
                    }
                    let candidate = merge_group_candidate(&group, image_width, image_height);
                    let appearance_bands = grouped_appearance_bands(&group, candidate.text_rect);
                    let cleanup_blocks = cleanup_blocks_for_group(&group);
                    let total_weight = group
                        .iter()
                        .map(|line| line.prediction.text.chars().count().max(1))
                        .sum::<usize>();
                    let ocr_confidence = group
                        .iter()
                        .map(|line| {
                            line.prediction.confidence.clamp(0.0, 1.0)
                                * line.prediction.text.chars().count().max(1) as f32
                        })
                        .sum::<f32>()
                        / total_weight.max(1) as f32;
                    let mut prediction = group
                        .iter()
                        .max_by_key(|line| line.prediction.text.chars().count())
                        .expect("recognized group is non-empty")
                        .prediction
                        .clone();
                    let measured_font_height = group
                        .iter()
                        .map(|line| line.candidate.text_rect.height())
                        .fold(1.0_f32, f32::max);
                    prediction.text.clone_from(&source_english);
                    prediction.confidence = ocr_confidence;
                    Some(GroupedRegion {
                        candidate,
                        reading_order: 0,
                        source_english,
                        faithful_chinese: None,
                        ocr_confidence,
                        continuation_group: None,
                        role: match candidate.kind {
                            CandidateKind::StoryText => ImageRegionRole::Dialogue,
                            CandidateKind::FreeText => ImageRegionRole::Narration,
                        },
                        source_line_count: cleanup_blocks.len().max(1),
                        prediction,
                        appearance_bands,
                        measured_font_height,
                        cleanup_blocks,
                    })
                })
                .collect::<Vec<_>>())
        })
        .await
        .context("group recognized dialogue on the browser preprocessing pool")
        .map_err(PipelineError::pipeline)?;

    let ranks = reading_order_ranks(
        &grouped
            .iter()
            .map(|group| group.candidate.text_rect)
            .collect::<Vec<_>>(),
        request.reading_direction,
    );
    let mut ranked_groups = grouped.into_iter().enumerate().collect::<Vec<_>>();
    ranked_groups.sort_by_key(|(index, _)| ranks[*index]);
    grouped = ranked_groups
        .into_iter()
        .enumerate()
        .map(|(rank, (_, mut group))| {
            group.reading_order = rank.min(u32::MAX as usize) as u32;
            group
        })
        .collect();

    cancellation_boundary(cancel.as_ref())?;
    let priority = if sink.focus().active
        && grouped.iter().any(|group| {
            group.candidate.bubble_rect.intersects_viewport(
                &sink.focus().visible_rects,
                image_width,
                image_height,
            )
        }) {
        CudaPriority::Visible
    } else {
        CudaPriority::Offscreen
    };
    let semantic_started = Instant::now();
    let page_adjudication = match adjudicate_grouped_page(
        Arc::clone(&resident),
        source.clone(),
        &grouped,
        request,
        priority,
        cancel.clone(),
        cuda_scheduler,
        image_width,
        image_height,
    )
    .await
    {
        Ok(result) => result,
        Err(_error) => {
            // A malformed or visually unsupported page decision is an
            // evidence failure, not permission to paint a guessed cleanup or
            // to restart the same image indefinitely.  Preserve every source
            // region and let the browser expose one bounded hover explanation
            // per region while the rest of the chapter continues.
            for group in &grouped {
                publish_unreadable_group(
                    sink,
                    group,
                    request,
                    image_width,
                    image_height,
                    "Page text could not be verified; source pixels were preserved. Hover it for help.",
                )?;
            }
            let all_ids = grouped
                .iter()
                .map(|group| stable_region_id(&request.source_sha256, group.candidate.text_rect))
                .collect::<HashSet<_>>();
            let region_plans =
                grouped_region_plans(&grouped, &all_ids, &HashSet::new(), &all_ids, request);
            return Ok((Vec::new(), text_probabilities, region_plans));
        }
    };
    let multimodal_roles =
        apply_page_adjudication_transcripts(&mut grouped, request, &page_adjudication);
    let mut excluded_ids = HashSet::<String>::new();
    let mut preserved_artwork_ids = HashSet::<String>::new();
    let mut unreadable_ids = page_adjudication
        .failed_region_ids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    for id in &page_adjudication.failed_region_ids {
        if let Some(group) = grouped.iter().find(|group| {
            stable_region_id(&request.source_sha256, group.candidate.text_rect) == *id
        }) {
            publish_unreadable_group(
                sink,
                group,
                request,
                image_width,
                image_height,
                "Page understanding could not establish readable story text; source pixels were preserved.",
            )?;
        }
        excluded_ids.insert(id.clone());
    }
    for group in &grouped {
        let id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
        let Some(role) = multimodal_roles.get(&id) else {
            continue;
        };
        match role {
            PageRegionRole::Furniture => {
                excluded_ids.insert(id);
            }
            PageRegionRole::Artwork => {
                preserved_artwork_ids.insert(id);
            }
            PageRegionRole::Story | PageRegionRole::Sfx => {}
        }
    }
    let semantic_elapsed = semantic_started.elapsed();
    for group in grouped.iter().filter(|group| {
        preserved_artwork_ids.contains(&stable_region_id(
            &request.source_sha256,
            group.candidate.text_rect,
        ))
    }) {
        publish_preserved_group(sink, group, request, image_width, image_height)?;
    }
    let all_grouped = grouped.clone();
    grouped.retain(|group| {
        let id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
        !excluded_ids.contains(&id) && !preserved_artwork_ids.contains(&id)
    });
    if grouped.is_empty() {
        let region_plans = grouped_region_plans(
            &all_grouped,
            &excluded_ids,
            &preserved_artwork_ids,
            &unreadable_ids,
            request,
        );
        return Ok((Vec::new(), text_probabilities, region_plans));
    }

    // The learned glyph segmenter is cleanup-only. Run it only after semantic
    // ownership has proved that this batch has translation consumers; a page
    // of credits, branding, or decorative lettering must spend no glyph-mask
    // inference at all.
    let text_supports = grouped
        .iter()
        .map(|group| {
            group
                .candidate
                .confirmed_bubble_rect
                .union(group.candidate.text_rect)
                .expand(
                    (group.candidate.text_rect.height() * 2.0).clamp(48.0, 192.0),
                    image_width,
                    image_height,
                )
        })
        .collect::<Vec<_>>();
    let text_tiles = overlapping_tiles(image_width, image_height)
        .into_iter()
        .filter(|tile| {
            !text_mask_completed_tiles.contains(&tile.id)
                && text_supports
                    .iter()
                    .any(|support| tile.rect().intersection(*support).is_some())
        })
        .collect::<Vec<_>>();
    let source_for_text_crops = source.clone();
    let text_tiles_for_crops = text_tiles.clone();
    let text_crops = preprocessing
        .run(move || {
            Ok(text_tiles_for_crops
                .iter()
                .map(|tile| source_for_text_crops.crop_imm(tile.x, tile.y, tile.width, tile.height))
                .collect::<Vec<_>>())
        })
        .await
        .context("prepare glyph segmentation tiles")
        .map_err(PipelineError::pipeline)?;
    if !text_tiles.is_empty() {
        if text_crops.len() != text_tiles.len() {
            return Err(PipelineError::new(
                "TEXT_SEGMENTATION_FAILED",
                "Glyph-mask cleanup prepared an incomplete tile batch.",
            ));
        }
        for (tile_batch, crop_batch) in text_tiles
            .chunks(DETECTOR_TILE_BATCH_SIZE)
            .zip(text_crops.chunks(DETECTOR_TILE_BATCH_SIZE))
        {
            cancellation_boundary(cancel.as_ref())?;
            let permit = cuda_scheduler
                .acquire(CudaWorkload::Vision, bubble_priority, cancel.clone())
                .await
                .map_err(cuda_admission_error)?;
            resident
                .instrumentation
                .invoked(RuntimeComponent::Segmentation);
            let results = {
                let segmenter = resident.text_segmenter.lock().map_err(|_| {
                    PipelineError::new("MODEL_STATE_FAILED", "Text segmenter lock poisoned.")
                })?;
                segmenter
                    .inference_batch(crop_batch)
                    .context("segment recognized source glyph mattes")
                    .map_err(PipelineError::pipeline)?
            };
            drop(permit);
            if results.len() != tile_batch.len() {
                return Err(PipelineError::new(
                    "TEXT_SEGMENTATION_FAILED",
                    "Glyph segmentation returned an incomplete tile batch.",
                ));
            }
            for (tile, result) in tile_batch.iter().zip(results) {
                merge_probability_map(&mut text_probabilities, &result, tile.x, tile.y);
                text_mask_completed_tiles.insert(tile.id);
            }
        }
    }

    // Only visually verified story regions are cleaned. The one inpaint task
    // uses Vision while faithful translation and any strict HSK rewrite use
    // Language, so cleanup never extends the language critical path.
    let cleanup = spawn_cleanup_batch(
        Arc::clone(&resident),
        source.clone(),
        grouped.clone(),
        bubble_mask.clone(),
        text_probabilities.clone(),
        request.source_sha256.clone(),
        cancel.clone(),
        Arc::clone(cuda_scheduler),
        Arc::clone(preprocessing),
        priority,
        image_width,
        image_height,
    );

    publish_progress(
        sink,
        BrowserJobStage::Inpainting,
        None,
        Some(overall_progress),
        None,
        None,
        "Restoring the artwork behind the original text",
    )?;
    cancellation_boundary(cancel.as_ref())?;
    publish_progress(
        sink,
        BrowserJobStage::Translating,
        None,
        Some(overall_progress),
        None,
        None,
        "Translating visually verified story text into Chinese",
    )?;
    let faithful_started = Instant::now();
    let mut faithful_translations = translate_faithful_candidates(
        Arc::clone(&resident),
        &grouped,
        request,
        preceding_context,
        priority,
        cancel.clone(),
        cuda_scheduler,
        sink,
    )
    .await?;
    for group in &mut grouped {
        let id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
        group.faithful_chinese = faithful_translations.remove(&id);
        if group.faithful_chinese.is_some() {
            continue;
        }
        publish_unreadable_group(
            sink,
            group,
            request,
            image_width,
            image_height,
            "A verified Chinese translation could not be established; source pixels were preserved.",
        )?;
        unreadable_ids.insert(id.clone());
        excluded_ids.insert(id);
    }
    let faithful_elapsed = faithful_started.elapsed();
    grouped.retain(|group| {
        let id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
        !excluded_ids.contains(&id)
    });
    let region_plans = grouped_region_plans(
        &all_grouped,
        &excluded_ids,
        &preserved_artwork_ids,
        &unreadable_ids,
        request,
    );
    if grouped.is_empty() {
        return Ok((Vec::new(), text_probabilities, region_plans));
    }
    let latest_viewport = sink.focus();
    let translation_queued_at = tokio::time::Instant::now();
    if std::env::var_os("HSKIFY_TRACE_PIPELINE_TIMING").is_some_and(|value| value == "1") {
        eprintln!(
            "hskify-prepare-timing groups={} bubble_ms={} role_ms={} faithful_ms={} cleanup=overlapped total_ms={}",
            grouped.len(),
            bubble_elapsed.as_millis(),
            semantic_elapsed.as_millis(),
            faithful_elapsed.as_millis(),
            prepare_started.elapsed().as_millis(),
        );
    }
    let source_contexts = (0..grouped.len())
        .map(|index| image_source_context(&grouped, index, request))
        .collect::<Vec<_>>();
    let prepared_regions = grouped
        .into_iter()
        .zip(source_contexts)
        .map(|(group, source_context)| {
            let candidate = group.candidate;
            let source_english = group.source_english;
            let faithful_chinese = group.faithful_chinese;
            let ocr_confidence = group.ocr_confidence;
            let continuation_group = group.continuation_group;
            let role = group.role;
            let source_line_count = group.source_line_count;
            let prediction = group.prediction;
            let appearance_bands = group.appearance_bands;
            let measured_font_height = group.measured_font_height;
            let (bubble_polygon, layout_polygon) = region_polygons(
                bubble_mask.as_ref(),
                bubble_components.as_ref(),
                candidate.text_rect,
                candidate.confirmed_bubble_rect,
                measured_font_height,
            );
            let visible = latest_viewport.active
                && candidate.bubble_rect.intersects_viewport(
                    &latest_viewport.visible_rects,
                    image_width,
                    image_height,
                );
            let reading_order = group.reading_order;
            PreparedRegion {
                id: stable_region_id(&request.source_sha256, candidate.text_rect),
                candidate,
                source_english,
                faithful_chinese,
                ocr_confidence,
                reading_order,
                continuation_group,
                role,
                source_line_count,
                prediction,
                appearance_bands,
                measured_font_height,
                bubble_polygon,
                layout_polygon,
                cleanup: cleanup.clone(),
                visible,
                translation_queued_at,
                source_context,
            }
        })
        .collect::<Vec<_>>();
    Ok((prepared_regions, text_probabilities, region_plans))
}

fn cleanup_failure_result(
    grouped: &[GroupedRegion],
    source_sha256: &str,
    message: &str,
) -> CleanupBatchResult {
    let decisions = grouped
        .iter()
        .map(|group| {
            (
                stable_region_id(source_sha256, group.candidate.text_rect),
                CleanupDecision {
                    patch: None,
                    reason: Some(message.to_owned()),
                    quality: None,
                },
            )
        })
        .collect();
    CleanupBatchResult { decisions }
}

async fn run_cleanup_inpaint(
    resident: &VisionRuntime,
    source: &DynamicImage,
    erase_mask: &image::GrayImage,
    bubble_mask: &image::GrayImage,
    text_blocks: &[TextRegion],
    cancel: &Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    priority: CudaPriority,
) -> Result<DynamicImage> {
    cancellation_boundary(cancel.as_ref()).map_err(|error| anyhow!(error.to_string()))?;
    let permit = cuda_scheduler
        .acquire(CudaWorkload::Vision, priority, Arc::clone(cancel))
        .await
        .map_err(|error| anyhow!(error.to_string()))?;
    resident
        .instrumentation
        .invoked(RuntimeComponent::Inpainting);
    let result = resident
        .inpainter
        .lock()
        .map_err(|_| anyhow!("Inpainter lock poisoned."))
        .and_then(|inpainter| {
            inpainter
                .inference_rgb_with_blocks(
                    source
                        .as_rgb8()
                        .expect("browser source images are canonical RGB"),
                    erase_mask,
                    bubble_mask,
                    text_blocks,
                )
                .context("restore artwork with the manga inpainter")
        });
    drop(permit);
    result.map(DynamicImage::ImageRgb8)
}

fn cleanup_decisions_for_image(
    source: &image::RgbImage,
    inpainted: &DynamicImage,
    cleaned_groups: &[CleanedGroupedRegion],
    source_sha256: &str,
    instrumentation: &RuntimeInstrumentation,
) -> Result<HashMap<String, CleanupDecision>> {
    let inpainted = inpainted
        .as_rgb8()
        .ok_or_else(|| anyhow!("cleanup inpaint result is not RGB"))?;
    // Verify the page-wide protected-pixel invariant once.  The inpainting
    // candidate is shared by every region in this batch; checking the full
    // image inside each region's quality score made tall pages quadratic in
    // the number of bubbles.
    let mut changed_mask = GrayImage::new(source.width(), source.height());
    for cleaned in cleaned_groups {
        merge_cleanup_mask(&mut changed_mask, &cleaned.cleanup_mask);
    }
    if !protected_pixels_match(source, inpainted, &changed_mask) {
        bail!("cleanup changed protected source pixels");
    }
    let mut decisions = HashMap::new();
    for cleaned in cleaned_groups {
        let group = &cleaned.group;
        let id = stable_region_id(source_sha256, group.candidate.text_rect);
        let decision = match score_cleanup_candidate_local(source, inpainted, &cleaned.cleanup_mask)
        {
            Some(quality) if quality.passes() => {
                instrumentation.invoked(RuntimeComponent::Patch);
                match make_inpainted_patch(inpainted, &cleaned.cleanup_mask) {
                    Ok(patch) => CleanupDecision {
                        patch: Some(patch),
                        reason: None,
                        quality: Some(quality),
                    },
                    Err(error) => CleanupDecision {
                        patch: None,
                        reason: Some(format!("Cleanup patch encoding failed: {error:#}")),
                        quality: Some(quality),
                    },
                }
            }
            Some(quality) => CleanupDecision {
                patch: None,
                reason: Some(
                    "Cleanup verification did not pass; source pixels were preserved.".to_owned(),
                ),
                quality: Some(quality),
            },
            None => CleanupDecision {
                patch: None,
                reason: Some(
                    "Cleanup quality check failed; source pixels were preserved.".to_owned(),
                ),
                quality: None,
            },
        };
        if std::env::var_os("HSKIFY_TRACE_PIPELINE_TIMING").is_some_and(|value| value == "1") {
            eprintln!(
                "hskify-cleanup-quality id={id} source={:?} accepted={} quality={:?}",
                group.source_english,
                decision.patch.is_some(),
                decision.quality,
            );
        }
        decisions.insert(id, decision);
    }
    Ok(decisions)
}

/// Start the page-level cleanup transaction without making the detector and
/// language paths wait for it. The result is immutable and shared by every
/// region in this analysis batch; each publication awaits only at its final
/// commit point.
#[allow(clippy::too_many_arguments)]
fn spawn_cleanup_batch(
    resident: Arc<VisionRuntime>,
    source: Arc<DynamicImage>,
    grouped: Vec<GroupedRegion>,
    bubble_mask: Arc<GrayImage>,
    text_probabilities: ProbabilityMap,
    source_sha256: String,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: Arc<CudaScheduler>,
    preprocessing: Arc<PreprocessingPool>,
    priority: CudaPriority,
    image_width: u32,
    image_height: u32,
) -> Arc<CleanupBatchTask> {
    CleanupBatchTask::spawn(async move {
        if cancel.load(Ordering::Acquire) {
            return cleanup_failure_result(
                &grouped,
                &source_sha256,
                "Cleanup was cancelled; source pixels were preserved.",
            );
        }
        let source_for_cleanup = source.clone();
        let masks_for_cleanup = bubble_mask.clone();
        let groups_for_masks = grouped.clone();
        let mask_result = preprocessing
            .run(move || {
                let source_rgb = source_for_cleanup
                    .as_rgb8()
                    .expect("browser source images are canonical RGB");
                let mut erase_mask = image::GrayImage::new(image_width, image_height);
                let mut all_text_blocks = Vec::new();
                let mut cleaned_groups = Vec::with_capacity(groups_for_masks.len());
                for group in groups_for_masks {
                    let support = group
                        .candidate
                        .confirmed_bubble_rect
                        .union(group.candidate.text_rect);
                    let learned_mask = verified_text_mask_for_regions_local(
                        source_rgb,
                        &text_probabilities,
                        masks_for_cleanup.as_ref(),
                        &group.cleanup_blocks,
                        support,
                        DEFAULT_TEXT_MASK_THRESHOLD,
                    )
                    .with_context(|| {
                        format!(
                            "learned text mask did not cover every OCR line in {:?}",
                            group.source_english
                        )
                    })?;
                    let local_bubble_mask = crop_imm(
                        masks_for_cleanup.as_ref(),
                        learned_mask.bounds.x,
                        learned_mask.bounds.y,
                        learned_mask.bounds.width,
                        learned_mask.bounds.height,
                    )
                    .to_image();
                    let local_blocks = group
                        .cleanup_blocks
                        .iter()
                        .map(|block| TextRegion {
                            x: block.x - learned_mask.bounds.x as f32,
                            y: block.y - learned_mask.bounds.y as f32,
                            ..block.clone()
                        })
                        .collect::<Vec<_>>();
                    let expanded_local = expand_mask_for_inpainting(
                        &learned_mask.mask,
                        &local_bubble_mask,
                        &local_blocks,
                    );
                    let local_support = PixelRect {
                        x0: support.x0 - learned_mask.bounds.x as f32,
                        y0: support.y0 - learned_mask.bounds.y as f32,
                        x1: support.x1 - learned_mask.bounds.x as f32,
                        y1: support.y1 - learned_mask.bounds.y as f32,
                    };
                    let mut cleanup_mask = compact_cleanup_mask(&expanded_local, local_support)
                        .with_context(|| {
                            format!(
                                "expanded cleanup mask was empty for OCR-confirmed dialogue {:?}",
                                group.source_english
                            )
                        })?;
                    cleanup_mask.bounds.x =
                        cleanup_mask.bounds.x.saturating_add(learned_mask.bounds.x);
                    cleanup_mask.bounds.y =
                        cleanup_mask.bounds.y.saturating_add(learned_mask.bounds.y);
                    merge_cleanup_mask(&mut erase_mask, &cleanup_mask);
                    all_text_blocks.extend(group.cleanup_blocks.iter().cloned());
                    cleaned_groups.push(CleanedGroupedRegion {
                        group,
                        cleanup_mask,
                    });
                }
                Ok::<_, anyhow::Error>((cleaned_groups, erase_mask, all_text_blocks))
            })
            .await;
        let (cleaned_groups, erase_mask, text_blocks) = match mask_result {
            Ok(result) => result,
            Err(error) => {
                return cleanup_failure_result(
                    &grouped,
                    &source_sha256,
                    &format!("Cleanup mask verification failed: {error:#}"),
                );
            }
        };
        if cancel.load(Ordering::Acquire) {
            return cleanup_failure_result(
                &grouped,
                &source_sha256,
                "Cleanup was cancelled; source pixels were preserved.",
            );
        }
        let inpainted = match run_cleanup_inpaint(
            resident.as_ref(),
            source.as_ref(),
            &erase_mask,
            bubble_mask.as_ref(),
            &text_blocks,
            &cancel,
            &cuda_scheduler,
            priority,
        )
        .await
        {
            Ok(image) => image,
            Err(error) => {
                return cleanup_failure_result(
                    &grouped,
                    &source_sha256,
                    &format!("Cleanup inpainting failed: {error:#}"),
                );
            }
        };
        let source_rgb = source
            .as_rgb8()
            .expect("browser source images are canonical RGB")
            .clone();
        let cleaned_for_scoring = cleaned_groups.clone();
        let source_sha256_for_cleanup = source_sha256.clone();
        let source_for_first = source_rgb.clone();
        let instrumentation = Arc::clone(&resident.instrumentation);
        let first_decisions = match preprocessing
            .run(move || {
                cleanup_decisions_for_image(
                    &source_for_first,
                    &inpainted,
                    &cleaned_for_scoring,
                    &source_sha256_for_cleanup,
                    instrumentation.as_ref(),
                )
            })
            .await
        {
            Ok(decisions) => decisions,
            Err(error) => {
                return cleanup_failure_result(
                    &grouped,
                    &source_sha256,
                    &format!("Cleanup patch preparation failed: {error:#}"),
                );
            }
        };
        CleanupBatchResult {
            decisions: first_decisions,
        }
    })
}

fn group_recognized_lines(
    mut lines: Vec<RecognizedLine>,
    bubble_mask: &image::GrayImage,
) -> Vec<Vec<RecognizedLine>> {
    lines.sort_by(|left, right| {
        left.candidate
            .text_rect
            .y0
            .total_cmp(&right.candidate.text_rect.y0)
            .then_with(|| {
                left.candidate
                    .text_rect
                    .x0
                    .total_cmp(&right.candidate.text_rect.x0)
            })
    });
    let mut groups = Vec::<(Option<u8>, Vec<RecognizedLine>)>::new();
    for line in lines {
        let bubble_id = (line.candidate.kind == CandidateKind::StoryText)
            .then(|| bubble_id_for_rect(bubble_mask, line.candidate.text_rect))
            .flatten();
        let matching = groups.iter().position(|(group_id, members)| {
            semantic_bubble_ids_are_compatible(*group_id, bubble_id)
                && members
                    .iter()
                    .all(|member| detector_bubble_cores_are_equivalent(member, &line))
        });
        if let Some(index) = matching {
            if groups[index].0.is_none() {
                groups[index].0 = bubble_id;
            }
            groups[index].1.push(line);
        } else {
            groups.push((bubble_id, vec![line]));
        }
    }
    groups
        .into_iter()
        .map(|(_, lines)| dedupe_recognized_line_group(lines))
        .collect()
}

fn dedupe_recognized_line_group(mut lines: Vec<RecognizedLine>) -> Vec<RecognizedLine> {
    let mut deduped = Vec::<RecognizedLine>::with_capacity(lines.len());
    for line in lines.drain(..) {
        let duplicate = deduped
            .iter()
            .position(|existing| recognized_lines_are_duplicate(existing, &line));
        if let Some(index) = duplicate {
            if recognized_line_quality(&line) > recognized_line_quality(&deduped[index]) {
                deduped[index] = line;
            }
        } else {
            deduped.push(line);
        }
    }
    deduped.sort_by(|left, right| {
        left.candidate
            .text_rect
            .y0
            .total_cmp(&right.candidate.text_rect.y0)
            .then_with(|| {
                left.candidate
                    .text_rect
                    .x0
                    .total_cmp(&right.candidate.text_rect.x0)
            })
    });
    deduped
}

fn recognized_lines_are_duplicate(left: &RecognizedLine, right: &RecognizedLine) -> bool {
    if left.candidate.kind != right.candidate.kind {
        return false;
    }
    let left_text = recognized_line_source_english(left);
    let right_text = recognized_line_source_english(right);
    if !ocr_texts_are_equivalent(&left_text, &right_text) {
        return false;
    }
    let left_rect = left.candidate.text_rect;
    let right_rect = right.candidate.text_rect;
    if left_rect.overlap_over_smaller(right_rect) >= 0.30 {
        return true;
    }
    // Distinct lines in one bubble may legitimately repeat a short token
    // (for example, two separate "line"/"No" utterances).  Adjacency alone
    // is not duplicate evidence; only materially overlapping geometry proves
    // that two OCR proposals describe the same source glyphs.
    false
}

fn ocr_texts_are_equivalent(left: &str, right: &str) -> bool {
    let left = left
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(|character| character.to_lowercase())
        .collect::<String>();
    let right = right
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(|character| character.to_lowercase())
        .collect::<String>();
    let shorter = left.len().min(right.len());
    if shorter < 4 {
        return false;
    }
    left.contains(&right)
        || right.contains(&left)
        || ascii_edit_distance_at_most(&left, &right, (shorter / 5).max(1).min(3))
}

fn ascii_edit_distance_at_most(left: &str, right: &str, limit: usize) -> bool {
    if left.len().abs_diff(right.len()) > limit {
        return false;
    }
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_byte) in left.bytes().enumerate() {
        let mut current = vec![left_index + 1; right.len() + 1];
        for (right_index, right_byte) in right.bytes().enumerate() {
            current[right_index + 1] = (previous[right_index]
                + usize::from(left_byte != right_byte))
            .min(current[right_index] + 1)
            .min(previous[right_index + 1] + 1);
        }
        if current.iter().copied().min().unwrap_or(limit + 1) > limit {
            return false;
        }
        previous = current;
    }
    previous[right.len()] <= limit
}

fn semantic_bubble_ids_are_compatible(left: Option<u8>, right: Option<u8>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => true,
    }
}

fn detector_bubble_cores_are_equivalent(left: &RecognizedLine, right: &RecognizedLine) -> bool {
    if left.candidate.kind != right.candidate.kind {
        return false;
    }
    match (
        left.candidate.has_detector_core,
        right.candidate.has_detector_core,
    ) {
        (true, false) => {
            return detector_core_contains_external_text(&left.candidate, &right.candidate);
        }
        (false, true) => {
            return detector_core_contains_external_text(&right.candidate, &left.candidate);
        }
        // Detector-free proposals do not carry a shared bubble identity. They
        // still need a local geometric relationship before they can be joined;
        // otherwise every free-form label on a tall page becomes one semantic
        // utterance (for example a site watermark plus an in-story effect).
        (false, false) => return detector_free_lines_are_compatible(left, right),
        (true, true) => {}
    }
    // OCR can split one speech bubble into several line proposals. When the
    // detector supplied the exact same confirmed bubble core, that identity
    // is stronger than the gap between individual line boxes; keep the whole
    // bubble as one semantic region while still separating neighbouring cores.
    if left.candidate.confirmed_bubble_rect == right.candidate.confirmed_bubble_rect {
        return true;
    }
    if !detector_text_lines_are_locally_adjacent(left, right) {
        return false;
    }
    let left_core = left.candidate.confirmed_bubble_rect;
    let right_core = right.candidate.confirmed_bubble_rect;
    left_core.intersection(right_core).is_some()
        && left_core.contains_point(right_core.center())
        && right_core.contains_point(left_core.center())
        && left_core.contains_point(right.candidate.text_rect.center())
        && right_core.contains_point(left.candidate.text_rect.center())
}

fn detector_core_contains_external_text(detector: &Candidate, external: &Candidate) -> bool {
    detector
        .confirmed_bubble_rect
        .contains_point(external.text_rect.center())
        && detector
            .confirmed_bubble_rect
            .overlap_over_smaller(external.text_rect)
            >= 0.25
}

fn detector_text_lines_are_locally_adjacent(left: &RecognizedLine, right: &RecognizedLine) -> bool {
    let left_rect = left.candidate.text_rect;
    let right_rect = right.candidate.text_rect;
    let smaller_height = left_rect.height().min(right_rect.height()).max(1.0);
    let larger_height = left_rect.height().max(right_rect.height()).max(1.0);
    if larger_height > smaller_height * 3.0 {
        return false;
    }
    let Some(intersection) = left_rect.intersection(right_rect) else {
        let vertical_gap = if left_rect.y1 < right_rect.y0 {
            right_rect.y0 - left_rect.y1
        } else if right_rect.y1 < left_rect.y0 {
            left_rect.y0 - right_rect.y1
        } else {
            0.0
        };
        let horizontal_gap = if left_rect.x1 < right_rect.x0 {
            right_rect.x0 - left_rect.x1
        } else if right_rect.x1 < left_rect.x0 {
            left_rect.x0 - right_rect.x1
        } else {
            0.0
        };
        return (horizontal_gap <= (smaller_height * 1.75).max(24.0)
            && vertical_gap <= (smaller_height * 2.5).max(32.0))
            || (vertical_gap <= (smaller_height * 1.75).max(24.0)
                && horizontal_gap <= (smaller_height * 2.5).max(32.0));
    };
    let horizontal_overlap =
        intersection.width() / left_rect.width().min(right_rect.width()).max(1.0);
    let vertical_overlap =
        intersection.height() / left_rect.height().min(right_rect.height()).max(1.0);
    horizontal_overlap >= 0.20 || vertical_overlap >= 0.20
}

fn detector_free_lines_are_compatible(left: &RecognizedLine, right: &RecognizedLine) -> bool {
    let left_rect = left.candidate.text_rect;
    let right_rect = right.candidate.text_rect;
    let left_height = left_rect.height().max(1.0);
    let right_height = right_rect.height().max(1.0);
    let smaller_height = left_height.min(right_height);
    let larger_height = left_height.max(right_height);

    // A headline and a watermark can be only a few pixels apart, but their
    // rendered scales are materially different. Treat that scale discontinuity
    // as a group boundary while allowing ordinary multi-line captions, whose
    // glyph heights are normally within roughly 2.5x of one another.
    if larger_height > smaller_height * 2.5 {
        return false;
    }

    let vertical_gap = if left_rect.y1 < right_rect.y0 {
        right_rect.y0 - left_rect.y1
    } else if right_rect.y1 < left_rect.y0 {
        left_rect.y0 - right_rect.y1
    } else {
        0.0
    };
    let horizontal_gap = if left_rect.x1 < right_rect.x0 {
        right_rect.x0 - left_rect.x1
    } else if right_rect.x1 < left_rect.x0 {
        left_rect.x0 - right_rect.x1
    } else {
        0.0
    };
    // Measure each axis independently. `PixelRect::intersection` is empty
    // when two caption lines are vertically separated, even though their
    // columns are perfectly aligned; using the 2-D intersection here would
    // incorrectly split every multi-line free-text caption.
    let horizontal_overlap =
        (left_rect.x1.min(right_rect.x1) - left_rect.x0.max(right_rect.x0)).max(0.0);
    let vertical_overlap =
        (left_rect.y1.min(right_rect.y1) - left_rect.y0.max(right_rect.y0)).max(0.0);
    let shared_width = left_rect.width().min(right_rect.width()).max(1.0);
    let shared_height = left_rect.height().min(right_rect.height()).max(1.0);

    // Same-line words/effects are joined when their glyph bands overlap in Y
    // and the horizontal gap is local. Multi-line captions/effects are joined
    // when their columns overlap and the vertical gap is no larger than a
    // normal line spacing interval. Both checks are scale-relative so they
    // work across desktop-sized and narrow comic panels.
    let compatible = (vertical_overlap / shared_height >= 0.20
        && horizontal_gap <= (smaller_height * 1.75).max(24.0))
        || (horizontal_overlap / shared_width >= 0.20
            && vertical_gap <= (smaller_height * 1.75).max(24.0));
    compatible
}

fn grouped_source_english(group: &[RecognizedLine]) -> String {
    let joined = group
        .iter()
        .map(recognized_line_source_english)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    compact_ocr_text(&joined)
}

fn recognized_line_source_english(line: &RecognizedLine) -> String {
    let prediction = &line.prediction;
    if prediction.ocr_lines.is_empty() {
        return compact_ocr_text(&prediction.text);
    }
    let owned = prediction
        .ocr_lines
        .iter()
        .enumerate()
        .filter(|(index, ocr_line)| {
            appearance_band_is_owned_by_candidate(
                &line.candidate,
                line.crop_bounds,
                prediction.appearance_bands.get(*index),
                Some(ocr_line),
            )
        })
        .map(|(_, ocr_line)| compact_ocr_text(&ocr_line.text))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    if owned.is_empty() {
        compact_ocr_text(&prediction.text)
    } else {
        owned.join(" ")
    }
}

fn appearance_band_is_owned_by_candidate(
    candidate: &Candidate,
    crop_bounds: PixelBounds,
    appearance_band: Option<&PpOcrAppearanceBand>,
    ocr_bounds: Option<&PpOcrLine>,
) -> bool {
    if !candidate.has_detector_core {
        return true;
    }
    let bubble = candidate.confirmed_bubble_rect;
    let line_rect = ocr_bounds
        .and_then(|line| {
            PixelRect::new(
                (crop_bounds.x + line.bounds.left) as f32,
                (crop_bounds.y + line.bounds.top) as f32,
                (crop_bounds.x + line.bounds.right) as f32,
                (crop_bounds.y + line.bounds.bottom) as f32,
            )
        })
        .or_else(|| {
            appearance_band.and_then(|band| {
                PixelRect::new(
                    crop_bounds.x as f32,
                    crop_bounds.y as f32
                        + band.top_ratio.clamp(0.0, 1.0) * crop_bounds.height as f32,
                    (crop_bounds.x + crop_bounds.width) as f32,
                    crop_bounds.y as f32
                        + band.bottom_ratio.clamp(0.0, 1.0) * crop_bounds.height as f32,
                )
            })
        });
    let Some(line_rect) = line_rect else {
        return true;
    };
    line_rect.overlap_over_smaller(bubble) >= 0.50 || bubble.contains_point(line_rect.center())
}

fn cleanup_blocks_for_group(group: &[RecognizedLine]) -> Vec<TextRegion> {
    group.iter().flat_map(cleanup_blocks_for_line).collect()
}

fn cleanup_blocks_for_line(line: &RecognizedLine) -> Vec<TextRegion> {
    let candidate = line.candidate;
    let make_block = |rect: PixelRect| TextRegion {
        x: rect.x0,
        y: rect.y0,
        width: rect.width(),
        height: rect.height(),
        confidence: candidate.detector_confidence,
        detected_font_size_px: Some(rect.height().max(1.0)),
        detector: Some("browser-comic-text-bubble-detector".to_owned()),
        ..TextRegion::default()
    };
    if line.prediction.appearance_bands.is_empty() {
        return vec![make_block(candidate.text_rect)];
    }
    let crop_top = line.crop_bounds.y as f32;
    let crop_height = line.crop_bounds.height.max(1) as f32;
    let blocks = line
        .prediction
        .appearance_bands
        .iter()
        .enumerate()
        .filter_map(|(index, band)| {
            if !appearance_band_is_owned_by_candidate(
                &line.candidate,
                line.crop_bounds,
                Some(band),
                line.prediction.ocr_lines.get(index),
            ) {
                return None;
            }
            PixelRect::new(
                candidate.text_rect.x0,
                (crop_top + band.top_ratio.clamp(0.0, 1.0) * crop_height)
                    .max(candidate.text_rect.y0),
                candidate.text_rect.x1,
                (crop_top + band.bottom_ratio.clamp(0.0, 1.0) * crop_height)
                    .min(candidate.text_rect.y1),
            )
            .map(make_block)
        })
        .collect::<Vec<_>>();
    if blocks.is_empty() {
        vec![make_block(candidate.text_rect)]
    } else {
        blocks
    }
}

fn grouped_appearance_bands(
    group: &[RecognizedLine],
    group_text_rect: PixelRect,
) -> Vec<SourceAppearanceBand> {
    let mut bands = group
        .iter()
        .flat_map(|line| {
            line.prediction
                .appearance_bands
                .iter()
                .enumerate()
                .filter(move |(index, band)| {
                    appearance_band_is_owned_by_candidate(
                        &line.candidate,
                        line.crop_bounds,
                        Some(band),
                        line.prediction.ocr_lines.get(*index),
                    )
                })
                .map(move |(_, band)| {
                    source_appearance_band(line.crop_bounds, band, group_text_rect)
                })
        })
        .collect::<Vec<_>>();
    bands.sort_by_key(|band| band.position_millionths);
    // Foreground palette changes are the source's intentional emphasis
    // structure. Outline pixels are lower-confidence boundaries in the same
    // learned text field and naturally vary with antialiasing/background, so
    // they enrich a retained band but never manufacture extra translated lines.
    bands.dedup_by(|right, left| same_palette_color(right.text_color, left.text_color));
    bands
}

fn same_palette_color(left: [u8; 3], right: [u8; 3]) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| left >> 5 == right >> 5)
}

fn source_appearance_band(
    crop_bounds: PixelBounds,
    band: &PpOcrAppearanceBand,
    group_text_rect: PixelRect,
) -> SourceAppearanceBand {
    let source_center = crop_bounds.y as f32
        + ((band.top_ratio + band.bottom_ratio) * 0.5) * crop_bounds.height as f32;
    let position =
        ((source_center - group_text_rect.y0) / group_text_rect.height().max(1.0)).clamp(0.0, 1.0);
    SourceAppearanceBand {
        position_millionths: (position * 1_000_000.0).round() as u32,
        text_color: band.text_color,
        stroke_color: band.stroke_color,
        has_stroke_color: band.has_stroke_color,
    }
}

fn merge_group_candidate(
    group: &[RecognizedLine],
    _image_width: u32,
    _image_height: u32,
) -> Candidate {
    let first = group.first().expect("recognized group is non-empty");
    let text_rect = group
        .iter()
        .skip(1)
        .fold(first.candidate.text_rect, |rect, line| {
            rect.union(line.candidate.text_rect)
        });
    let confirmed_bubble_rect = group
        .iter()
        .skip(1)
        .fold(first.candidate.confirmed_bubble_rect, |rect, line| {
            rect.union(line.candidate.confirmed_bubble_rect)
        });
    let layout_rect = confirmed_bubble_rect.union(text_rect);
    Candidate {
        kind: first.candidate.kind,
        text_rect,
        bubble_rect: layout_rect,
        confirmed_bubble_rect,
        detector_confidence: group
            .iter()
            .map(|line| line.candidate.detector_confidence)
            .fold(0.0, f32::max),
        has_detector_core: group.iter().all(|line| line.candidate.has_detector_core),
        rotation_radians: group
            .iter()
            .map(|line| line.candidate.rotation_radians)
            .sum::<f32>()
            / group.len() as f32,
    }
}

fn prioritize_pending_translation(
    regions: &mut [PreparedRegion],
    sink: &JobUpdateSink,
    image_width: u32,
    image_height: u32,
) {
    let viewport = sink.focus();
    for region in regions.iter_mut() {
        region.visible = viewport.active
            && region.candidate.bubble_rect.intersects_viewport(
                &viewport.visible_rects,
                image_width,
                image_height,
            );
    }
    sort_pending_translation(regions);
}

fn sort_pending_translation(regions: &mut [PreparedRegion]) {
    regions.sort_by(|left, right| {
        right
            .visible
            .cmp(&left.visible)
            .then_with(|| left.reading_order.cmp(&right.reading_order))
            .then_with(|| left.translation_queued_at.cmp(&right.translation_queued_at))
    });
}

fn prepared_region_priority(
    regions: &[PreparedRegion],
    sink: &JobUpdateSink,
    image_width: u32,
    image_height: u32,
) -> CudaPriority {
    let viewport = sink.focus();
    if viewport.active
        && regions.iter().any(|region| {
            region.candidate.bubble_rect.intersects_viewport(
                &viewport.visible_rects,
                image_width,
                image_height,
            )
        })
    {
        CudaPriority::Visible
    } else {
        CudaPriority::Offscreen
    }
}

fn normalized_rects_intersect(left: &NormalizedRect, right: &NormalizedRect) -> bool {
    left.x < right.x + right.width
        && right.x < left.x + left.width
        && left.y < right.y + right.height
        && right.y < left.y + left.height
}

fn cuda_admission_error(error: CudaAdmissionError) -> PipelineError {
    match error {
        CudaAdmissionError::Cancelled => PipelineError::cancelled(),
        queue_error @ CudaAdmissionError::QueueFull { .. } => {
            PipelineError::new("CUDA_QUEUE_FULL", queue_error.to_string())
        }
    }
}

fn compact_ocr_text(text: &str) -> String {
    // OCR output is evidence, not prose to be repaired by string heuristics.
    // Preserve recognizer punctuation and token boundaries exactly; semantic
    // correction belongs to the chapter-level vision/translation pass.
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn append_terminal_context(
    context: &mut Vec<HskPrecedingUtterance>,
    source_english: &str,
    chinese: &str,
) {
    let source_english = source_english.trim();
    let chinese = chinese.trim();
    if source_english.is_empty() || chinese.is_empty() {
        return;
    }
    if context.iter().any(|utterance| {
        utterance
            .source_english
            .eq_ignore_ascii_case(source_english)
            && utterance.chinese == chinese
    }) {
        return;
    }
    context.push(HskPrecedingUtterance {
        source_english: source_english.to_owned(),
        chinese: chinese.to_owned(),
    });
    if context.len() > MAX_HSK_PRECEDING_UTTERANCES {
        context.drain(..context.len() - MAX_HSK_PRECEDING_UTTERANCES);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OcrProposalSource {
    Detector,
}

fn accept_english_ocr_line(
    confidence: f32,
    text: &str,
    _proposal_source: OcrProposalSource,
) -> bool {
    if !confidence.is_finite() || confidence < self::ocr::BROWSER_OCR_MIN_CONFIDENCE {
        return false;
    }
    let text = text.trim();
    if text.is_empty()
        || text.contains('\u{fffd}')
        || text.to_ascii_uppercase().contains("<UNK>")
        || text.chars().any(char::is_control)
    {
        return false;
    }
    let alphabetic = text
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect::<Vec<_>>();
    !alphabetic.is_empty()
        && alphabetic
            .iter()
            .all(|character| is_latin_letter(*character))
}

fn rejected_ocr_tracing_enabled() -> bool {
    std::env::var_os("HSKIFY_TRACE_REJECTED_OCR").is_some_and(|value| value == "1")
}

fn hsk_utterance_kind(kind: CandidateKind) -> HskUtteranceKind {
    match kind {
        CandidateKind::StoryText => HskUtteranceKind::Dialogue,
        CandidateKind::FreeText => HskUtteranceKind::Caption,
    }
}

fn hsk_utterance_kind_for_region(region: &PreparedRegion) -> HskUtteranceKind {
    if region.role == ImageRegionRole::System {
        HskUtteranceKind::Sfx
    } else {
        hsk_utterance_kind(region.candidate.kind)
    }
}

fn grouped_region_plans(
    regions: &[GroupedRegion],
    excluded_ids: &HashSet<String>,
    preserved_artwork_ids: &HashSet<String>,
    unreadable_ids: &HashSet<String>,
    request: &ImagePipelineInput,
) -> Vec<RegionPlan> {
    regions
        .iter()
        .map(|region| {
            let id = stable_region_id(&request.source_sha256, region.candidate.text_rect);
            let role = if unreadable_ids.contains(&id) {
                RegionRole::Unreadable
            } else if preserved_artwork_ids.contains(&id) {
                RegionRole::TechniqueArtwork
            } else if excluded_ids.contains(&id) {
                RegionRole::Exclusion
            } else {
                match region.role {
                    ImageRegionRole::Dialogue => RegionRole::Dialogue,
                    ImageRegionRole::Narration => RegionRole::Narration,
                    ImageRegionRole::System => RegionRole::System,
                }
            };
            RegionPlan {
                id,
                reading_order: region.reading_order,
                role,
                source_english: region.source_english.clone(),
                continuation_group: region.continuation_group.clone(),
            }
        })
        .collect()
}

/// Derive a bounded generation budget from the measured inset polygon and the
/// source glyph height.  The renderer uses the same polygon and source-relative
/// minimum, so the model gets a physical constraint instead of a prose guess
/// about how much Chinese might fit.
fn layout_budget_for_region(
    region: &PreparedRegion,
    image_width: u32,
    image_height: u32,
) -> (u16, u8) {
    let (min_x, max_x, min_y, max_y) = region.layout_polygon.iter().fold(
        (1.0_f32, 0.0_f32, 1.0_f32, 0.0_f32),
        |(min_x, max_x, min_y, max_y), point| {
            (
                min_x.min(point.x),
                max_x.max(point.x),
                min_y.min(point.y),
                max_y.max(point.y),
            )
        },
    );
    let width = ((max_x - min_x).max(0.0) * image_width as f32).max(1.0);
    let height = ((max_y - min_y).max(0.0) * image_height as f32).max(1.0);
    let source_height = region.measured_font_height.max(1.0);
    let geometric_lines = (height / (source_height * 1.25)).floor() as usize;
    let max_lines = geometric_lines
        .max(region.source_line_count.max(1))
        .clamp(1, usize::from(MAX_HSK_LAYOUT_LINES)) as u8;
    let characters_per_line = (width / (source_height * 0.86)).floor() as usize;
    let max_characters = characters_per_line
        .saturating_mul(usize::from(max_lines))
        .clamp(
            usize::from(MIN_HSK_LAYOUT_CHARACTERS),
            usize::from(MAX_HSK_LAYOUT_CHARACTERS),
        ) as u16;
    (max_characters, max_lines)
}

fn publish_preserved_group(
    sink: &JobUpdateSink,
    group: &GroupedRegion,
    request: &ImagePipelineInput,
    image_width: u32,
    image_height: u32,
) -> std::result::Result<(), PipelineError> {
    let region_id = stable_region_id(&request.source_sha256, group.candidate.text_rect);

    sink.publish(JobUpdateDraft::ImageRegionPreserved {
        region: ImageRegionPreserved {
            disposition: crate::contracts::PreservationDisposition::Excluded,
            item_id: region_id,
            text_polygon: group.candidate.text_rect.polygon(image_width, image_height),
            source_text: group.source_english.clone(),
            confidence: group.ocr_confidence,
            item_order: group.reading_order,
            reason: "artwork-preserved".to_owned(),
        },
    })
    .map_err(|error| publish_error(error, sink))?;
    Ok(())
}

fn publish_unreadable_group(
    sink: &JobUpdateSink,
    group: &GroupedRegion,
    request: &ImagePipelineInput,
    image_width: u32,
    image_height: u32,
    reason: &str,
) -> std::result::Result<(), PipelineError> {
    let region_id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
    // Keep a source-preserving lookup context even when no patch is safe. The
    // browser can expose the OCR transcript and failure reason on hover
    // without pretending that an unverified Chinese overlay exists.
    if !request.retry_item_ids.is_empty()
        && !request.retry_item_ids.contains(&stable_region_id(
            &request.source_sha256,
            group.candidate.text_rect,
        ))
    {
        return Ok(());
    }

    sink.publish(JobUpdateDraft::ImageRegionPreserved {
        region: ImageRegionPreserved {
            disposition: crate::contracts::PreservationDisposition::Failed,
            item_id: region_id,
            text_polygon: group.candidate.text_rect.polygon(image_width, image_height),
            source_text: group.source_english.clone(),
            confidence: group.ocr_confidence,
            item_order: group.reading_order,
            reason: reason.to_owned(),
        },
    })
    .map_err(|error| publish_error(error, sink))?;
    Ok(())
}

fn is_latin_letter(character: char) -> bool {
    character.is_ascii_alphabetic()
        || matches!(
            character as u32,
            0x00c0..=0x00ff | 0x0100..=0x017f | 0x0180..=0x024f | 0x1e00..=0x1eff
        )
}

fn stable_region_id(source_sha256: &str, rect: PixelRect) -> String {
    let canonical = format!(
        "hskify-region|{source_sha256}|{:.2}|{:.2}|{:.2}|{:.2}",
        rect.x0, rect.y0, rect.x1, rect.y1
    );
    let digest = sha256_hex(canonical.as_bytes());
    format!(
        "{}-region-{}",
        &source_sha256[..source_sha256.len().min(8)],
        &digest[..16]
    )
}

#[derive(Debug, Clone)]
struct CachedTranslation {
    protected_names: Vec<koharu_app::llm::ProtectedName>,
    base_chinese: String,
    displayed_chinese: String,
    pinyin: String,
    report: ValidationReport,
    repair_state: HskRepairState,
}

fn natural_translation(
    faithful_chinese: impl Into<FaithfulText>,
    report: ValidationReport,
) -> CachedTranslation {
    let faithful = faithful_chinese.into();
    CachedTranslation {
        protected_names: faithful.protected_names,
        base_chinese: faithful.text,
        displayed_chinese: report.normalized_text.clone(),
        pinyin: String::new(),
        report,
        repair_state: HskRepairState::NotNeeded,
    }
}

struct TranslationCacheEntry<T> {
    value: T,
    bytes: usize,
    last_used: u64,
}

struct TranslationCache<T = CachedTranslation> {
    entries: HashMap<String, TranslationCacheEntry<T>>,
    retained_bytes: usize,
    clock: u64,
    max_bytes: usize,
}

impl<T> Default for TranslationCache<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            retained_bytes: 0,
            clock: 0,
            max_bytes: TRANSLATION_CACHE_MAX_BYTES,
        }
    }
}

impl<T: Clone + CacheValue> TranslationCache<T> {
    fn get(&mut self, key: &str) -> Option<T> {
        self.clock = self.clock.saturating_add(1);
        let entry = self.entries.get_mut(key)?;
        entry.last_used = self.clock;
        Some(entry.value.clone())
    }

    fn insert(&mut self, key: String, value: T) {
        let bytes = key.len().saturating_add(value.retained_bytes());
        if bytes > self.max_bytes {
            return;
        }
        if let Some(previous) = self.entries.remove(&key) {
            self.retained_bytes = self.retained_bytes.saturating_sub(previous.bytes);
        }
        self.clock = self.clock.saturating_add(1);
        self.retained_bytes = self.retained_bytes.saturating_add(bytes);
        self.entries.insert(
            key,
            TranslationCacheEntry {
                value,
                bytes,
                last_used: self.clock,
            },
        );
        while self.retained_bytes > self.max_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.retained_bytes = self.retained_bytes.saturating_sub(removed.bytes);
            }
        }
    }
}

trait CacheValue {
    fn retained_bytes(&self) -> usize;
}
impl CacheValue for String {
    fn retained_bytes(&self) -> usize {
        self.len()
    }
}
impl CacheValue for CachedTranslation {
    fn retained_bytes(&self) -> usize {
        translation_cache_bytes("", self)
    }
}

fn translation_cache_bytes(key: &str, value: &CachedTranslation) -> usize {
    let report = &value.report;
    key.len()
        .saturating_add(value.base_chinese.len())
        .saturating_add(value.displayed_chinese.len())
        .saturating_add(value.pinyin.len())
        .saturating_add(
            value
                .protected_names
                .iter()
                .map(|n| n.source_text.len() + n.chinese_text.len() + 64)
                .sum::<usize>(),
        )
        .saturating_add(report.normalized_text.len())
        .saturating_add(report.cache_revision.len())
        .saturating_add(
            report
                .violations
                .iter()
                .map(|violation| {
                    violation.text.len().saturating_add(
                        violation
                            .suggested_words
                            .iter()
                            .map(String::len)
                            .sum::<usize>(),
                    )
                })
                .sum::<usize>(),
        )
        .saturating_add(
            report
                .exceptions
                .iter()
                .map(|exception| exception.text.len())
                .sum::<usize>(),
        )
        .saturating_add(std::mem::size_of::<TranslationCacheEntry<CachedTranslation>>())
}

struct TranslationState {
    base_chinese: Option<String>,
    displayed_chinese: Option<String>,
    latest_rejected_chinese: Option<String>,
    latest_rejected_report: Option<ValidationReport>,
    report: Option<ValidationReport>,
    problems: Vec<String>,
    structurally_valid: bool,
    learning_mode: LearningMode,
    repair_state: HskRepairState,
}

impl TranslationState {
    fn from_initial(
        outcome: HskTranslationOutcome,
        control: &HskControl,
        level: ControlHskLevel,
        proper_names: &[ProperName],
        learning_mode: LearningMode,
    ) -> Self {
        let mut problems = outcome.repair_problems();
        let structurally_valid = outcome.is_valid();
        let base_chinese = nonempty_translation(outcome.text);
        let report = base_chinese
            .as_deref()
            .map(|text| control.validate(text, level, proper_names));
        if let Some(report) = &report
            && learning_policy_requires_repair(report, learning_mode)
        {
            append_validation_problems(&mut problems, report);
        }
        let valid = problems.is_empty() && report.is_some();
        Self {
            displayed_chinese: valid.then(|| {
                report
                    .as_ref()
                    .expect("valid translation has a validation report")
                    .normalized_text
                    .clone()
            }),
            base_chinese,
            latest_rejected_chinese: None,
            latest_rejected_report: None,
            report,
            problems,
            structurally_valid,
            learning_mode,
            repair_state: HskRepairState::NotNeeded,
        }
    }

    fn can_publish(&self) -> bool {
        self.structurally_valid && self.base_chinese.is_some() && self.report.is_some()
    }

    fn avoid_chinese(&self) -> Vec<String> {
        let report = self
            .latest_rejected_report
            .as_ref()
            .or(self.report.as_ref());
        let mut terms = Vec::new();
        if let Some(report) = report {
            for violation in &report.violations {
                let term = violation.text.trim();
                if !term.is_empty() && !terms.iter().any(|existing| existing == term) {
                    terms.push(term.to_owned());
                }
            }
        }
        terms
    }

    fn apply_repair(
        &mut self,
        outcome: HskTranslationOutcome,
        control: &HskControl,
        level: ControlHskLevel,
        proper_names: &[ProperName],
    ) -> bool {
        let mut problems = outcome.repair_problems();
        let repaired_structurally_valid = outcome.is_valid();
        let repaired = nonempty_translation(outcome.text);
        let report = repaired
            .as_deref()
            .filter(|_| repaired_structurally_valid)
            .map(|repaired| control.validate(repaired, level, proper_names));
        if let Some(report) = &report
            && learning_policy_requires_repair(report, self.learning_mode)
        {
            append_validation_problems(&mut problems, report);
        }
        self.apply_evaluated_repair(
            repaired.filter(|_| repaired_structurally_valid),
            report,
            problems,
        )
    }

    fn apply_evaluated_repair(
        &mut self,
        repaired: Option<String>,
        report: Option<ValidationReport>,
        problems: Vec<String>,
    ) -> bool {
        let had_usable_primary = self.can_publish();
        let accepted = repaired.is_some() && report.is_some() && problems.is_empty();
        self.latest_rejected_chinese = if accepted { None } else { repaired.clone() };
        if accepted {
            self.latest_rejected_report = None;
        } else if report.is_some() {
            self.latest_rejected_report = report.clone();
        }
        if let (Some(repaired), Some(report)) = (repaired, report)
            && (accepted || !had_usable_primary)
        {
            if !had_usable_primary {
                self.base_chinese = Some(repaired);
            }
            self.displayed_chinese = Some(report.normalized_text.clone());
            self.report = Some(report);
            self.structurally_valid = true;
        }
        self.problems = problems;
        self.repair_state = if accepted && self.can_publish() {
            HskRepairState::Accepted
        } else {
            HskRepairState::Rejected
        };
        accepted && self.can_publish()
    }

    fn finish(mut self) -> Result<CachedTranslation> {
        if !self.can_publish()
            || matches!(
                self.repair_state,
                HskRepairState::Pending | HskRepairState::Rejected
            )
        {
            bail!("direct translation and its repair are not safe to publish");
        }
        let displayed_chinese = self
            .displayed_chinese
            .take()
            .or_else(|| {
                self.report
                    .as_ref()
                    .map(|report| report.normalized_text.clone())
            })
            .filter(|text| !text.trim().is_empty())
            .context("direct translation and its sole repair produced no Chinese text")?;
        let base_chinese = self
            .base_chinese
            .take()
            .unwrap_or_else(|| displayed_chinese.clone());
        let report = self
            .report
            .take()
            .context("translated Chinese is missing deterministic HSK validation")?;
        Ok(CachedTranslation {
            protected_names: Vec::new(),
            base_chinese,
            displayed_chinese,
            pinyin: String::new(),
            report,
            repair_state: self.repair_state,
        })
    }
}

fn nonempty_translation(text: Option<String>) -> Option<String> {
    text.map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
}

fn missing_translation_outcome(id: &str) -> HskTranslationOutcome {
    use koharu_app::llm::HskTranslationIssue;
    HskTranslationOutcome {
        termination: koharu_llm::GenerationTermination::Stop,
        protected_names: Vec::new(),
        id: id.to_owned(),
        text: None,
        issues: vec![HskTranslationIssue::MissingLine],
    }
}

fn append_validation_problems(problems: &mut Vec<String>, report: &ValidationReport) {
    let mut violations = Vec::<&str>::new();
    for violation in &report.violations {
        let token = violation.text.trim();
        if !token.is_empty() && !violations.contains(&token) {
            violations.push(token);
        }
    }
    if violations.is_empty() {
        return;
    }
    violations.sort_unstable();
    let guidance = violations
        .into_iter()
        .take(16)
        .map(|token| format!("`{token}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let problem = format!(
        "rewrite these detected above-level spans with contextually natural easier words and grammar: {guidance}; never substitute an unrelated dictionary word"
    );
    if !problems.contains(&problem) {
        problems.push(problem);
    }
}

fn level_coverage(report: &ValidationReport) -> f32 {
    if report.lexical_token_count == 0 {
        return 1.0;
    }
    let accepted = report.lexical_token_count.saturating_sub(
        report
            .above_level_token_count
            .min(report.lexical_token_count),
    );
    accepted as f32 / report.lexical_token_count as f32
}

fn learning_policy_requires_repair(report: &ValidationReport, mode: LearningMode) -> bool {
    // Natural mode is a best-effort simplify-preserve-teach policy. Its
    // deterministic report drives teaching metadata, but vocabulary must not
    // delay or suppress a faithful translation: an advanced term can be the
    // shortest natural way to preserve meaning. Strict mode is the explicit
    // opt-in that treats every non-name above-level term as a repair gate.
    mode == LearningMode::Strict && !report.strictly_valid
}

fn translation_is_final(translation: &CachedTranslation) -> bool {
    matches!(
        translation.repair_state,
        HskRepairState::NotNeeded | HskRepairState::Accepted
    )
}

#[allow(clippy::too_many_arguments)]
fn translation_cache_key(
    source_english: &str,
    kind: HskUtteranceKind,
    layout: Option<HskLayoutConstraints>,
    batch_source_texts: &[String],
    context: &[HskPrecedingUtterance],
    preceding_english: &[String],
    following_english: &[String],
    provenance: DirectSourceProvenance,
    learning_mode: LearningMode,
    hsk_level: u8,
    model_id: &str,
    model_revision: &str,
    prompt_hash: &str,
    validator_hash: &str,
    hsk_control_revision: &str,
) -> String {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct KeyMaterial<'a> {
        schema: &'static str,
        source_text: &'a str,
        kind: HskUtteranceKind,
        layout: Option<HskLayoutConstraints>,
        batch_source_texts: &'a [String],
        provenance: &'static str,
        context: &'a [HskPrecedingUtterance],
        preceding_english: &'a [String],
        following_english: &'a [String],
        learning_mode: LearningMode,
        hsk_level: u8,
        model_id: &'a str,
        model_revision: &'a str,
        prompt_hash: &'a str,
        validator_hash: &'a str,
        hsk_control_revision: &'a str,
    }
    let start = context.len().saturating_sub(MAX_HSK_PRECEDING_UTTERANCES);
    let normalize_source = |text: &str| match provenance {
        DirectSourceProvenance::Dom => text.to_owned(),
        DirectSourceProvenance::Ocr => compact_ocr_text(text),
    };
    let source_text = normalize_source(source_english);
    let normalized_batch_source_texts = batch_source_texts
        .iter()
        .map(|text| normalize_source(text))
        .collect::<Vec<_>>();
    let material = KeyMaterial {
        schema: TRANSLATION_CACHE_SCHEMA,
        source_text: &source_text,
        kind,
        layout,
        batch_source_texts: &normalized_batch_source_texts,
        provenance: match provenance {
            DirectSourceProvenance::Dom => "dom",
            DirectSourceProvenance::Ocr => "ocr",
        },
        context: &context[start..],
        preceding_english,
        following_english,
        learning_mode,
        hsk_level,
        model_id,
        model_revision,
        prompt_hash,
        validator_hash,
        hsk_control_revision,
    };
    let bytes = serde_json::to_vec(&material).expect("cache key material is serializable");
    format!("sha256:{}", sha256_hex(&bytes))
}

fn publish_region(
    sink: &JobUpdateSink,
    region: &PreparedRegion,
    cleanup: &CleanupDecision,
    translation: CachedTranslation,
    requested_level: HskLevel,
    learning_mode: LearningMode,
    control: &HskControl,
    image_width: u32,
    image_height: u32,
) -> std::result::Result<(), PipelineError> {
    let text_polygon = region
        .candidate
        .text_rect
        .polygon(image_width, image_height);
    let (style, layout) = style_and_layout(
        &region,
        &translation.displayed_chinese,
        image_width,
        region.layout_polygon.clone(),
    );
    let patch = cleanup.patch.as_ref().ok_or_else(|| {
        PipelineError::new("CLEANUP_UNVERIFIED", "Cleanup patch was not verified.")
    })?;
    let patch_rect = patch.bounds.normalized(image_width, image_height);
    let patch = sink
        .store_generated_patch_png(patch_rect, patch.bytes.clone())
        .map_err(|error| publish_error(error, sink))?;
    let above_level_tokens = above_level_tokens(&translation.report);
    let teaching_terms = teaching_terms(control, &translation.report);
    let translated = ImageRegionReady {
        item_id: region.id.clone(),
        text_polygon,
        bubble_polygon: Some(region.bubble_polygon.clone()),
        patch,
        text: TranslatedText {
            source_text: region.source_english.clone(),
            termination: koharu_llm::GenerationTermination::Stop,
            protected_names: translation.protected_names.clone(),
            base_chinese: translation.base_chinese.clone(),
            displayed_chinese: translation.displayed_chinese.clone(),
            pinyin: translation.pinyin.clone(),
            hsk: TranslatedHskStatus {
                requested_level,
                learning_mode,
                strictly_valid: translation.report.strictly_valid,
                level_coverage: level_coverage(&translation.report),
                above_level_tokens,
                teaching_terms,
                repair_state: translation.repair_state,
            },
        },
        provenance: SourceProvenance::Ocr,
        kind: match region.role {
            ImageRegionRole::Dialogue => SourceSpanKind::Dialogue,
            ImageRegionRole::Narration => SourceSpanKind::Caption,
            ImageRegionRole::System => SourceSpanKind::Sfx,
        },
        confidence: region.ocr_confidence,
        item_order: region.reading_order,
        context_group: region.continuation_group.clone(),
        confidence_evidence: Some(RegionConfidenceEvidence {
            ocr_consensus: region.ocr_confidence,
            geometry_coverage: geometry_coverage(region),
            context_consistency: if region.continuation_group.is_some() {
                1.0
            } else {
                0.8
            },
            cleanup_score: cleanup.quality.map_or(0.0, CleanupQuality::score),
        }),
        style,
        layout,
    };

    sink.publish(JobUpdateDraft::ImageRegionReady {
        region: Box::new(translated),
    })
    .map_err(|error| publish_error(error, sink))?;
    Ok(())
}

fn geometry_coverage(region: &PreparedRegion) -> f32 {
    region
        .candidate
        .text_rect
        .overlap_over_smaller(region.candidate.confirmed_bubble_rect)
        .clamp(0.0, 1.0)
}

fn publish_unreadable_prepared(
    sink: &JobUpdateSink,
    region: &PreparedRegion,
    request: &ImagePipelineInput,
    image_width: u32,
    image_height: u32,
    reason: &str,
) -> std::result::Result<(), PipelineError> {
    if !request.retry_item_ids.is_empty() && !request.retry_item_ids.contains(&region.id) {
        return Ok(());
    }

    sink.publish(JobUpdateDraft::ImageRegionPreserved {
        region: ImageRegionPreserved {
            disposition: crate::contracts::PreservationDisposition::Failed,
            item_id: region.id.clone(),
            text_polygon: region
                .candidate
                .text_rect
                .polygon(image_width, image_height),
            source_text: region.source_english.clone(),
            confidence: region.ocr_confidence,
            item_order: region.reading_order,
            reason: reason.to_owned(),
        },
    })
    .map_err(|error| publish_error(error, sink))?;
    let _ = request;
    Ok(())
}

/// Convert OCR detector proposals that failed both recognition
/// views into terminal, source-preserving regions.  The detector is allowed
/// to find text that the recognizer cannot read; silently dropping that
/// proposal would make a coverage metric look green while leaving an English
/// bubble untouched.  The browser receives no patch, only a stable hover/tap
/// target and the reason for the evidence failure.
fn publish_rejected_ocr_regions(
    rejected: &[RejectedOcrLine],
    accepted_rects: &[PixelRect],
    request: &ImagePipelineInput,
    image_width: u32,
    image_height: u32,
    sink: &JobUpdateSink,
) -> std::result::Result<Vec<RegionPlan>, PipelineError> {
    let mut selected = Vec::<&RejectedOcrLine>::new();
    for line in rejected {
        if accepted_rects
            .iter()
            .any(|accepted| text_rects_represent_same_block(line.candidate.text_rect, *accepted))
        {
            continue;
        }
        let duplicate = selected.iter().position(|existing| {
            text_rects_represent_same_block(existing.candidate.text_rect, line.candidate.text_rect)
        });
        if let Some(index) = duplicate {
            if rejected_ocr_quality(line) > rejected_ocr_quality(selected[index]) {
                selected[index] = line;
            }
        } else {
            selected.push(line);
        }
    }
    let ranks = reading_order_ranks(
        &selected
            .iter()
            .map(|line| line.candidate.text_rect)
            .collect::<Vec<_>>(),
        request.reading_direction,
    );
    let mut ranked = selected.into_iter().enumerate().collect::<Vec<_>>();
    ranked.sort_by_key(|(index, _)| ranks[*index]);

    let mut plans = Vec::with_capacity(ranked.len());
    for (reading_order, (_, line)) in ranked.into_iter().enumerate() {
        let reading_order = reading_order.min(u32::MAX as usize) as u32;
        let id = stable_region_id(&request.source_sha256, line.candidate.text_rect);
        if !request.retry_item_ids.is_empty() && !request.retry_item_ids.contains(&id) {
            continue;
        }
        let source_english = rejected_ocr_source(&line.prediction);

        sink.publish(JobUpdateDraft::ImageRegionPreserved {
            region: ImageRegionPreserved {
            disposition: crate::contracts::PreservationDisposition::Failed,
                item_id: id.clone(),
                text_polygon: line
                    .candidate
                    .text_rect
                    .polygon(image_width, image_height),
                source_text: source_english.clone(),
                confidence: line.prediction.confidence.clamp(0.0, 1.0),
                item_order: reading_order,
                reason: "OCR consensus failed after two preprocessing views; source pixels were preserved. Hover it for help.".to_owned(),
            },
        })
        .map_err(|error| publish_error(error, sink))?;
        plans.push(RegionPlan {
            id,
            reading_order,
            role: RegionRole::Unreadable,
            source_english,
            continuation_group: None,
        });
    }
    Ok(plans)
}

fn rejected_ocr_quality(line: &RejectedOcrLine) -> (u32, u32) {
    (
        (line.prediction.confidence.clamp(0.0, 1.0) * 1_000_000.0).round() as u32,
        line.prediction.text.chars().count().min(u32::MAX as usize) as u32,
    )
}

fn rejected_ocr_source(prediction: &PpOcrPrediction) -> String {
    // A rejected proposal has failed the two-view OCR agreement check.
    // Its transcript is therefore not trusted source evidence and must not
    // enter lookup, chapter context, or the browser's hover metadata. The
    // source pixels remain visible; the stable notice tells the reader why no
    // translation was painted without presenting letter soup as fact.
    let _ = prediction;
    "Unrecognized text".to_owned()
}

fn above_level_tokens(report: &ValidationReport) -> Vec<String> {
    let mut tokens = Vec::new();
    for violation in &report.violations {
        if !tokens.contains(&violation.text) {
            tokens.push(violation.text.clone());
        }
    }
    tokens
}

fn teaching_terms(control: &HskControl, report: &ValidationReport) -> Vec<TeachingTerm> {
    let mut terms = Vec::new();
    let mut previous_end = 0;
    for violation in &report.violations {
        if violation.start_char < previous_end || violation.start_char >= violation.end_char {
            continue;
        }
        let lookup = control.lookup(&violation.text, &[]);
        let mut pinyin = Vec::new();
        let mut definitions = Vec::new();
        for token in lookup.tokens {
            if !token.pinyin.trim().is_empty() {
                pinyin.push(token.pinyin);
            }
            for definition in token.definitions {
                if !definition.trim().is_empty() && !definitions.contains(&definition) {
                    definitions.push(definition);
                }
            }
        }
        if definitions.is_empty() {
            definitions
                .push("Story term kept because a simpler wording would be less clear.".to_owned());
        }
        let (required_level, reason) = match violation.reason {
            ViolationReason::AboveSelectedHskLevel { required_level } => (
                HskLevel::try_from(required_level.get()).ok(),
                TeachingTermReason::AboveLevel,
            ),
            _ => (None, TeachingTermReason::OutsideList),
        };
        terms.push(TeachingTerm {
            text: violation.text.clone(),
            start_char: violation.start_char,
            end_char: violation.end_char,
            pinyin: if pinyin.is_empty() {
                violation.text.clone()
            } else {
                pinyin.join(" ")
            },
            definitions,
            required_level,
            reason,
        });
        previous_end = violation.end_char;
    }
    terms
}

fn populate_pinyin(control: &HskControl, translation: &mut CachedTranslation) {
    let proper_names = translation
        .report
        .exceptions
        .iter()
        .map(|exception| ProperName {
            text: exception.text.clone(),
            reason: exception.reason,
        })
        .collect::<Vec<_>>();
    translation.pinyin = control
        .lookup(&translation.displayed_chinese, &proper_names)
        .tokens
        .into_iter()
        .map(|token| {
            if token.pinyin.trim().is_empty() {
                token.simplified
            } else {
                token.pinyin
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    if translation.pinyin.trim().is_empty() {
        translation
            .pinyin
            .clone_from(&translation.displayed_chinese);
    }
}

fn style_and_layout(
    region: &PreparedRegion,
    displayed_chinese: &str,
    image_width: u32,
    bubble_polygon: Vec<crate::contracts::Point>,
) -> (BrowserTextStyle, BrowserTextLayout) {
    let foreground = rgb(region.prediction.text_color);
    let outline_color = region
        .prediction
        .has_stroke_color
        .then(|| rgb(region.prediction.stroke_color));
    let outline_width_ratio = if outline_color.is_some() {
        (2.0 / region.candidate.text_rect.width().max(1.0)).clamp(0.002, 0.08)
    } else {
        0.0
    };
    let color_bands = region
        .appearance_bands
        .iter()
        .map(|band| BrowserTextColorBand {
            position: band.position_millionths as f32 / 1_000_000.0,
            foreground: rgb(band.text_color),
            outline_color: band.has_stroke_color.then(|| rgb(band.stroke_color)),
        })
        .collect::<Vec<_>>();
    // Color bands describe paint runs, not line breaks.  Keep line geometry
    // tied to the independently recognized source lines so a two-color bubble
    // is not silently split into two arbitrary Chinese lines.
    let suggested_line_count = region.source_line_count.max(1);
    let category = inferred_font_category(region);
    let font_id = match category {
        FontCategory::Serif => "hskify-serif",
        FontCategory::Handwritten => "hskify-handwritten",
        FontCategory::Display => "hskify-display",
        FontCategory::Brush => "hskify-brush",
        FontCategory::Sans => "hskify-sans",
    };
    let weight = inferred_font_weight(region);
    let writing_mode = inferred_writing_mode(region);
    let alignment = inferred_alignment(region);
    let line_height = inferred_line_height(region, suggested_line_count);
    let italic_degrees = 0.0;
    let letter_spacing_em = 0.0;
    let shadow_color = None;
    let shadow_x_ratio = 0.0;
    let shadow_y_ratio = 0.0;
    (
        BrowserTextStyle {
            font_id: font_id.to_owned(),
            category,
            foreground,
            weight,
            italic_degrees,
            outline_color,
            outline_width_ratio,
            shadow_color,
            shadow_x_ratio,
            shadow_y_ratio,
            alignment,
            writing_mode,
            line_height,
            letter_spacing_em,
            color_bands,
        },
        BrowserTextLayout {
            suggested_lines: suggested_lines(displayed_chinese, suggested_line_count),
            font_size_to_image_width: (region.measured_font_height / image_width.max(1) as f32)
                .clamp(0.002, 0.25),
            safe_polygon: bubble_polygon,
        },
    )
}

/// Infer typography from measured source-line evidence rather than selecting
/// one global replacement style.  These are deliberately visual/layout
/// signals (stroke, color runs, aspect, and measured spacing), never title or
/// word lists, so the same rules apply to unfamiliar readers.
fn inferred_font_category(region: &PreparedRegion) -> FontCategory {
    let color_runs = region.appearance_bands.len();
    let outlined = region.prediction.has_stroke_color;
    if region.role == ImageRegionRole::System && (outlined || color_runs > 1) {
        FontCategory::Display
    } else if outlined && color_runs > 1 {
        FontCategory::Brush
    } else if outlined {
        FontCategory::Handwritten
    } else if region.source_line_count > 2 {
        FontCategory::Serif
    } else {
        FontCategory::Sans
    }
}

fn inferred_font_weight(region: &PreparedRegion) -> u16 {
    if region.prediction.has_stroke_color {
        700
    } else if region.source_line_count > 2 {
        500
    } else {
        600
    }
}

fn inferred_writing_mode(region: &PreparedRegion) -> WritingMode {
    let rect = region.candidate.text_rect;
    if region.source_line_count == 1
        && region.prediction.ocr_lines.len() == 1
        && rect.height() > rect.width() * 1.8
    {
        WritingMode::VerticalRl
    } else {
        WritingMode::HorizontalTb
    }
}

fn inferred_alignment(region: &PreparedRegion) -> TextAlignment {
    let text_center = region.candidate.text_rect.center().0;
    let bubble_center = region.candidate.confirmed_bubble_rect.center().0;
    let offset = text_center - bubble_center;
    let threshold = region.candidate.confirmed_bubble_rect.width() * 0.18;
    if offset < -threshold {
        TextAlignment::Left
    } else if offset > threshold {
        TextAlignment::Right
    } else {
        TextAlignment::Center
    }
}

fn inferred_line_height(region: &PreparedRegion, line_count: usize) -> f32 {
    let source_height = region.candidate.text_rect.height().max(1.0);
    let measured = region.measured_font_height.max(1.0);
    (source_height / (measured * line_count.max(1) as f32)).clamp(0.9, 1.5)
}

fn suggested_lines(text: &str, preferred_line_count: usize) -> Vec<String> {
    let characters = text.chars().collect::<Vec<_>>();
    if preferred_line_count > 1 {
        let line_count = preferred_line_count.min(characters.len().max(1));
        return (0..line_count)
            .map(|index| {
                let start = index * characters.len() / line_count;
                let end = (index + 1) * characters.len() / line_count;
                characters[start..end].iter().collect()
            })
            .filter(|line: &String| !line.is_empty())
            .collect();
    }
    if characters.len() <= 10 {
        return vec![text.to_owned()];
    }
    let line_length = (characters.len() as f32).sqrt().ceil().max(6.0) as usize;
    characters
        .chunks(line_length)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn rgb(color: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", color[0], color[1], color[2])
}

fn batch_overall_progress(processed: usize, total: usize) -> f32 {
    0.04 + (processed as f32 / total.max(1) as f32) * 0.84
}

fn translation_queue_ready_len(pending: &[PreparedRegion], force: bool) -> usize {
    if pending.is_empty() {
        return 0;
    }
    let visible = pending.iter().take_while(|region| region.visible).count();
    if visible > 0 {
        return visible;
    }
    if force || pending.len() >= TRANSLATION_BATCH_MIN {
        return pending.len();
    }
    0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TranslationBoundaryAction {
    ContinueUpstream,
    Dispatch(usize),
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TranslationLatencyPhase {
    AwaitingFirstVisibleRegion,
    Throughput,
}

fn complete_translation_batch(phase: &mut TranslationLatencyPhase, published_visible_final: bool) {
    if published_visible_final {
        *phase = TranslationLatencyPhase::Throughput;
    }
}

fn translation_boundary_action(
    pending: &[PreparedRegion],
    force: bool,
    cancelled: bool,
    first_visible_region: bool,
) -> TranslationBoundaryAction {
    if cancelled {
        return TranslationBoundaryAction::Cancelled;
    }
    if first_visible_region {
        return TranslationBoundaryAction::Dispatch(1);
    }
    let eligible = translation_queue_ready_len(pending, force);
    if eligible == 0 {
        TranslationBoundaryAction::ContinueUpstream
    } else {
        TranslationBoundaryAction::Dispatch(translation_batch_len(eligible))
    }
}

fn translation_batch_len(available: usize) -> usize {
    debug_assert!(available > 0);
    if available <= TRANSLATION_BATCH_MAX {
        return available;
    }
    let tail = available - TRANSLATION_BATCH_MAX;
    if tail < TRANSLATION_BATCH_MIN {
        TRANSLATION_BATCH_MAX - (TRANSLATION_BATCH_MIN - tail)
    } else {
        TRANSLATION_BATCH_MAX
    }
}

#[allow(clippy::too_many_arguments)]
fn publish_progress(
    sink: &JobUpdateSink,
    stage: BrowserJobStage,
    stage_progress: Option<f32>,
    overall_progress: Option<f32>,
    current: Option<u32>,
    total: Option<u32>,
    message: impl Into<String>,
) -> std::result::Result<(), PipelineError> {
    sink.publish(JobUpdateDraft::Progress {
        stage,
        stage_progress,
        overall_progress,
        current,
        total,
        message: message.into(),
    })
    .map_err(|error| publish_error(error, sink))?;
    Ok(())
}

fn publish_error(error: crate::server::PublishError, sink: &JobUpdateSink) -> PipelineError {
    if sink.is_cancelled() {
        PipelineError::cancelled()
    } else {
        PipelineError::new("UPDATE_PUBLISH_FAILED", error.to_string())
    }
}

fn cancellation_boundary(cancel: &AtomicBool) -> std::result::Result<(), PipelineError> {
    if cancel.load(Ordering::Acquire) {
        Err(PipelineError::cancelled())
    } else {
        Ok(())
    }
}

pub(crate) fn browser_lookup_result(result: hsk_control::LookupResult) -> LookupResult {
    LookupResult {
        selected_text: result.selected_text,
        tokens: result
            .tokens
            .into_iter()
            .map(|token| LookupToken {
                simplified: token.simplified,
                pinyin: token.pinyin,
                definitions: token.definitions,
                hsk_level: token
                    .hsk_level
                    .and_then(|level| HskLevel::try_from(level.get()).ok()),
                proper_name: token.proper_name,
            })
            .collect(),
        item: result.item.map(|item| LookupItem {
            displayed_chinese: item.displayed_chinese,
            base_chinese: item.base_chinese,
            source_text: item.source_text,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hsk_control::{HskViolation, ViolationReason};

    fn validation_report(text: &str, violations: Vec<HskViolation>) -> ValidationReport {
        let above_level_token_count = violations.len();
        ValidationReport {
            normalized_text: text.to_owned(),
            requested_level: ControlHskLevel::new(1).unwrap(),
            strictly_valid: violations.is_empty(),
            lexical_token_count: above_level_token_count.max(1),
            above_level_token_count,
            violations,
            exceptions: Vec::new(),
            cache_revision: "test-control-r1".to_owned(),
        }
    }

    fn above_level_violation(text: &str) -> HskViolation {
        HskViolation {
            text: text.to_owned(),
            start_char: 0,
            end_char: text.chars().count(),
            reason: ViolationReason::AboveSelectedHskLevel {
                required_level: ControlHskLevel::new(2).unwrap(),
            },
            suggested_words: vec!["学生".to_owned()],
        }
    }

    #[test]
    fn multimodal_evidence_uses_geometry_derived_crop_for_sparse_tall_surfaces() {
        let source = Arc::new(DynamicImage::new_rgb8(1_000, 12_000));
        let text_rect = PixelRect::new(240.0, 4_800.0, 760.0, 5_200.0).unwrap();
        let bubble_rect = PixelRect::new(160.0, 4_500.0, 840.0, 5_500.0).unwrap();
        let grouped = vec![GroupedRegion {
            candidate: Candidate {
                kind: CandidateKind::StoryText,
                text_rect,
                bubble_rect,
                confirmed_bubble_rect: bubble_rect,
                detector_confidence: 0.9,
                has_detector_core: true,
                rotation_radians: 0.0,
            },
            reading_order: 0,
            source_english: "A readable sentence.".to_owned(),
            faithful_chinese: Some("一句可读的话。".to_owned().into()),
            ocr_confidence: 0.9,
            continuation_group: None,
            role: ImageRegionRole::Dialogue,
            source_line_count: 1,
            prediction: PpOcrPrediction {
                text: "A readable sentence.".to_owned(),
                confidence: 0.9,
                text_color: [0, 0, 0],
                stroke_color: [255, 255, 255],
                has_stroke_color: false,
                appearance_bands: Vec::new(),
                ocr_lines: Vec::new(),
            },
            appearance_bands: Vec::new(),
            measured_font_height: 72.0,
            cleanup_blocks: Vec::new(),
        }];

        let (cropped, viewport) = page_evidence_surface(&source, &grouped, 1_000, 12_000);

        assert!(cropped.height() < source.height());
        assert_eq!(
            cropped.width().max(cropped.height()),
            PAGE_ROLE_MAX_LONG_EDGE
        );
        assert!(viewport.y0 > 0.0);
        let polygon = polygon_in_evidence_viewport(text_rect, viewport);
        assert!(
            polygon
                .iter()
                .all(|point| { (0.0..=1.0).contains(&point.x) && (0.0..=1.0).contains(&point.y) })
        );
        assert!(polygon.iter().any(|point| point.y > 0.1 && point.y < 0.9));

        let bubble_evidence = page_region_evidence(&grouped[0], &grouped, "source", viewport);
        assert!(bubble_evidence.bubble_id.is_some());
        let mut uncontained_text = grouped.clone();
        uncontained_text[0].candidate.has_detector_core = false;
        let uncontained_evidence =
            page_region_evidence(&uncontained_text[0], &uncontained_text, "source", viewport);
        assert_eq!(uncontained_evidence.bubble_id, None);
        assert!(uncontained_evidence.connected_region_ids.is_empty());
    }

    #[test]
    fn repair_feedback_groups_spans_without_context_free_dictionary_substitutions() {
        let report = validation_report(
            "高级高级词",
            vec![
                above_level_violation("高级"),
                above_level_violation("高级"),
                above_level_violation("词"),
            ],
        );
        let mut problems = Vec::new();

        append_validation_problems(&mut problems, &report);

        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("contextually natural easier words"));
        assert_eq!(problems[0].matches("高级").count(), 1);
        assert!(!problems[0].contains("学生"));
        assert!(problems[0].contains("never substitute an unrelated dictionary word"));
    }

    #[test]
    fn natural_learning_reports_teachable_terms_without_queuing_a_repair() {
        let mut report = validation_report("侦探看了看。", vec![above_level_violation("侦探")]);
        report.lexical_token_count = 4;
        report.above_level_token_count = 1;

        assert!(!learning_policy_requires_repair(
            &report,
            LearningMode::Natural
        ));
        assert!(learning_policy_requires_repair(
            &report,
            LearningMode::Strict
        ));
    }

    #[test]
    fn natural_learning_publishes_the_faithful_reference_without_a_model_rewrite() {
        let report = validation_report("呼叫支援吗？", vec![above_level_violation("支援")]);

        let translation = natural_translation("呼叫支援吗？".to_owned(), report);

        assert_eq!(translation.base_chinese, "呼叫支援吗？");
        assert_eq!(translation.displayed_chinese, "呼叫支援吗？");
        assert_eq!(translation.repair_state, HskRepairState::NotNeeded);
    }

    #[test]
    fn natural_learning_never_turns_vocabulary_metadata_into_a_publication_gate() {
        let mut second = above_level_violation("证据");
        second.start_char = 2;
        second.end_char = 4;
        let mut report = validation_report("侦探证据", vec![above_level_violation("侦探"), second]);
        report.lexical_token_count = 5;
        report.above_level_token_count = 2;

        assert!(!learning_policy_requires_repair(
            &report,
            LearningMode::Natural
        ));
        assert!(learning_policy_requires_repair(
            &report,
            LearningMode::Strict
        ));
    }

    #[test]
    fn ocr_acceptance_checks_model_confidence_and_decodable_latin_text_only() {
        assert!(accept_english_ocr_line(
            0.91,
            "FORGOTTEN,",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.91,
            "-ALDRIN-",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "R2D2",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "m2y.",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "X3",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "Ahem.!..",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "CHAPTER 30",
            OcrProposalSource::Detector
        ));
        assert!(!accept_english_ocr_line(
            0.44,
            "Too uncertain",
            OcrProposalSource::Detector
        ));
        assert!(!accept_english_ocr_line(
            0.99,
            "<UNK>",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "I",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "A.",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "a",
            OcrProposalSource::Detector
        ));
        assert!(accept_english_ocr_line(
            0.99,
            "h.",
            OcrProposalSource::Detector
        ));
    }

    #[test]
    fn compact_ocr_text_only_normalizes_whitespace() {
        assert_eq!(
            compact_ocr_text("  BY   NOW,  HE'S  HERE. "),
            "BY NOW, HE'S HERE."
        );
        assert_eq!(compact_ocr_text("NO, NO! DON'T GO."), "NO, NO! DON'T GO.");
    }

    #[test]
    fn rejected_ocr_hover_source_never_exposes_gibberish_as_story_text() {
        let gibberish = PpOcrPrediction {
            text: "<UNK> 123 !!!".to_owned(),
            confidence: 0.12,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
            ocr_lines: Vec::new(),
        };
        assert_eq!(rejected_ocr_source(&gibberish), "Unrecognized text");

        let rejected_even_when_alphabetic = PpOcrPrediction {
            text: "  THE   DOOR  ".to_owned(),
            confidence: 0.48,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
            ocr_lines: Vec::new(),
        };
        assert_eq!(
            rejected_ocr_source(&rejected_even_when_alphabetic),
            "Unrecognized text"
        );
    }

    #[test]
    fn grouping_uses_segmented_bubble_identity_not_detector_box_similarity() {
        let shared = PixelRect::new(20.0, 20.0, 180.0, 140.0).unwrap();
        let other = PixelRect::new(200.0, 20.0, 360.0, 140.0).unwrap();
        let candidate = |text_rect, bubble_rect| Candidate {
            kind: CandidateKind::StoryText,
            text_rect,
            bubble_rect,
            confirmed_bubble_rect: bubble_rect,
            detector_confidence: 0.95,
            has_detector_core: true,
            rotation_radians: 0.0,
        };
        let prediction = || PpOcrPrediction {
            text: "line".to_owned(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
            ocr_lines: Vec::new(),
        };
        let lines = vec![
            RecognizedLine {
                candidate: candidate(PixelRect::new(40.0, 50.0, 160.0, 70.0).unwrap(), shared),
                prediction: prediction(),
                crop_bounds: PixelBounds {
                    x: 40,
                    y: 50,
                    width: 120,
                    height: 20,
                },
            },
            RecognizedLine {
                candidate: candidate(
                    PixelRect::new(45.0, 80.0, 155.0, 100.0).unwrap(),
                    PixelRect::new(18.0, 18.0, 182.0, 142.0).unwrap(),
                ),
                prediction: prediction(),
                crop_bounds: PixelBounds {
                    x: 45,
                    y: 80,
                    width: 110,
                    height: 20,
                },
            },
            RecognizedLine {
                candidate: candidate(PixelRect::new(220.0, 50.0, 340.0, 70.0).unwrap(), other),
                prediction: prediction(),
                crop_bounds: PixelBounds {
                    x: 220,
                    y: 50,
                    width: 120,
                    height: 20,
                },
            },
        ];
        let mut bubble_mask = image::GrayImage::new(400, 200);
        for y in 20..140 {
            for x in 20..180 {
                bubble_mask.put_pixel(x, y, image::Luma([1]));
            }
            for x in 200..360 {
                bubble_mask.put_pixel(x, y, image::Luma([2]));
            }
        }

        let groups = group_recognized_lines(lines, &bubble_mask);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 1);
    }

    #[test]
    fn grouping_splits_distinct_detector_cores_inside_one_connected_bubble_component() {
        let left_core = PixelRect::new(20.0, 20.0, 190.0, 140.0).unwrap();
        let right_core = PixelRect::new(170.0, 20.0, 340.0, 140.0).unwrap();
        let prediction = || PpOcrPrediction {
            text: "line".to_owned(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
            ocr_lines: Vec::new(),
        };
        let make_line = |text_rect, core, crop_x| RecognizedLine {
            candidate: Candidate {
                kind: CandidateKind::StoryText,
                text_rect,
                bubble_rect: core,
                confirmed_bubble_rect: core,
                detector_confidence: 0.95,
                has_detector_core: true,
                rotation_radians: 0.0,
            },
            prediction: prediction(),
            crop_bounds: PixelBounds {
                x: crop_x,
                y: 50,
                width: 100,
                height: 20,
            },
        };
        let lines = vec![
            make_line(
                PixelRect::new(50.0, 50.0, 150.0, 70.0).unwrap(),
                left_core,
                50,
            ),
            make_line(
                PixelRect::new(210.0, 50.0, 310.0, 70.0).unwrap(),
                right_core,
                210,
            ),
        ];
        // The learned bubble contour is deliberately connected, as happens
        // when overlapping balloons touch at their outlines.
        let bubble_mask = image::GrayImage::from_pixel(380, 180, image::Luma([1]));

        let groups = group_recognized_lines(lines, &bubble_mask);

        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|group| group.len() == 1));
    }

    #[test]
    fn grouping_drops_partial_ocr_duplicate_inside_the_same_bubble() {
        let bubble = PixelRect::new(20.0, 20.0, 360.0, 220.0).unwrap();
        let candidate = |text_rect| Candidate {
            kind: CandidateKind::StoryText,
            text_rect,
            bubble_rect: bubble,
            confirmed_bubble_rect: bubble,
            detector_confidence: 0.95,
            has_detector_core: true,
            rotation_radians: 0.0,
        };
        let prediction = |text: &str| PpOcrPrediction {
            text: text.to_owned(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
            ocr_lines: Vec::new(),
        };
        let lines = vec![
            RecognizedLine {
                candidate: candidate(PixelRect::new(80.0, 80.0, 230.0, 116.0).unwrap()),
                prediction: prediction("YEAH, THAT'S WHAT I MEAN!"),
                crop_bounds: PixelBounds {
                    x: 80,
                    y: 80,
                    width: 150,
                    height: 36,
                },
            },
            RecognizedLine {
                candidate: candidate(PixelRect::new(90.0, 84.0, 160.0, 112.0).unwrap()),
                prediction: prediction("YEAH,"),
                crop_bounds: PixelBounds {
                    x: 90,
                    y: 84,
                    width: 70,
                    height: 28,
                },
            },
        ];
        let groups = group_recognized_lines(lines, &image::GrayImage::new(400, 260));

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 1);
        assert_eq!(groups[0][0].prediction.text, "YEAH, THAT'S WHAT I MEAN!");
    }

    #[test]
    fn grouping_keeps_detector_free_lines_local_and_scale_consistent() {
        let prediction = || PpOcrPrediction {
            text: "line".to_owned(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
            ocr_lines: Vec::new(),
        };
        let make_line = |text_rect| RecognizedLine {
            candidate: Candidate {
                kind: CandidateKind::FreeText,
                text_rect,
                bubble_rect: text_rect,
                confirmed_bubble_rect: text_rect,
                detector_confidence: 0.95,
                has_detector_core: false,
                rotation_radians: 0.0,
            },
            prediction: prediction(),
            crop_bounds: text_rect.pixel_bounds(800, 1200),
        };
        let lines = vec![
            // Two caption lines: same scale, aligned columns, normal line gap.
            make_line(PixelRect::new(120.0, 100.0, 680.0, 140.0).unwrap()),
            make_line(PixelRect::new(130.0, 154.0, 670.0, 194.0).unwrap()),
            // A nearby but much smaller publisher watermark: it must not be
            // merged into the caption above.
            make_line(PixelRect::new(20.0, 202.0, 180.0, 216.0).unwrap()),
            // A distant free-form label is a separate semantic region.
            make_line(PixelRect::new(120.0, 700.0, 680.0, 740.0).unwrap()),
        ];
        let groups = group_recognized_lines(lines, &image::GrayImage::new(800, 1200));

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 1);
        assert_eq!(groups[2].len(), 1);
    }

    #[test]
    fn grouping_recovers_lines_from_one_detector_core_when_segmentation_is_missing() {
        let core = PixelRect::new(20.0, 20.0, 190.0, 140.0).unwrap();
        let make_line = |text_rect, crop_y| RecognizedLine {
            candidate: Candidate {
                kind: CandidateKind::StoryText,
                text_rect,
                bubble_rect: core,
                confirmed_bubble_rect: core,
                detector_confidence: 0.95,
                has_detector_core: true,
                rotation_radians: 0.0,
            },
            prediction: PpOcrPrediction {
                text: "line".to_owned(),
                confidence: 0.95,
                text_color: [0, 0, 0],
                stroke_color: [255, 255, 255],
                has_stroke_color: false,
                appearance_bands: Vec::new(),
                ocr_lines: Vec::new(),
            },
            crop_bounds: PixelBounds {
                x: 40,
                y: crop_y,
                width: 120,
                height: 20,
            },
        };
        let lines = vec![
            make_line(PixelRect::new(40.0, 50.0, 160.0, 70.0).unwrap(), 50),
            make_line(PixelRect::new(45.0, 80.0, 155.0, 100.0).unwrap(), 80),
        ];
        let bubble_mask = image::GrayImage::new(220, 180);

        let groups = group_recognized_lines(lines, &bubble_mask);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn grouping_attaches_external_text_only_when_it_is_inside_the_detector_bubble() {
        let core = PixelRect::new(20.0, 20.0, 190.0, 140.0).unwrap();
        let prediction = || PpOcrPrediction {
            text: "line".to_owned(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
            ocr_lines: Vec::new(),
        };
        let make_line = |text_rect, has_detector_core| RecognizedLine {
            candidate: Candidate {
                kind: CandidateKind::StoryText,
                text_rect,
                bubble_rect: if has_detector_core {
                    core
                } else {
                    text_rect.expand(8.0, 240, 300)
                },
                confirmed_bubble_rect: if has_detector_core {
                    core
                } else {
                    text_rect.expand(8.0, 240, 300)
                },
                detector_confidence: 0.95,
                has_detector_core,
                rotation_radians: 0.0,
            },
            prediction: prediction(),
            crop_bounds: text_rect.pixel_bounds(240, 300),
        };
        let lines = vec![
            make_line(PixelRect::new(40.0, 50.0, 160.0, 70.0).unwrap(), true),
            make_line(PixelRect::new(45.0, 82.0, 155.0, 102.0).unwrap(), false),
            make_line(PixelRect::new(45.0, 200.0, 155.0, 220.0).unwrap(), false),
        ];
        // Deliberately one connected learned component, as on white gutters.
        let bubble_mask = image::GrayImage::from_pixel(240, 300, image::Luma([1]));

        let groups = group_recognized_lines(lines, &bubble_mask);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 1);
    }

    #[test]
    fn cleanup_blocks_retain_each_ocr_line_band_for_mask_coverage() {
        let text_rect = PixelRect::new(20.0, 20.0, 180.0, 100.0).unwrap();
        let line = RecognizedLine {
            candidate: Candidate {
                kind: CandidateKind::StoryText,
                text_rect,
                bubble_rect: text_rect,
                confirmed_bubble_rect: text_rect,
                detector_confidence: 0.95,
                has_detector_core: true,
                rotation_radians: 0.0,
            },
            prediction: PpOcrPrediction {
                text: "two lines".to_owned(),
                confidence: 0.95,
                text_color: [0, 0, 0],
                stroke_color: [255, 255, 255],
                has_stroke_color: false,
                appearance_bands: vec![
                    PpOcrAppearanceBand {
                        top_ratio: 0.10,
                        bottom_ratio: 0.35,
                        text_color: [0, 0, 0],
                        stroke_color: [255, 255, 255],
                        has_stroke_color: false,
                    },
                    PpOcrAppearanceBand {
                        top_ratio: 0.60,
                        bottom_ratio: 0.90,
                        text_color: [0, 0, 0],
                        stroke_color: [255, 255, 255],
                        has_stroke_color: false,
                    },
                ],
                ocr_lines: Vec::new(),
            },
            crop_bounds: PixelBounds {
                x: 17,
                y: 17,
                width: 166,
                height: 86,
            },
        };

        let blocks = cleanup_blocks_for_line(&line);

        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].y + blocks[0].height < blocks[1].y);
    }

    #[test]
    fn grouped_ocr_text_never_expands_the_confirmed_bubble_used_for_layout() {
        let bubble = PixelRect::new(20.0, 20.0, 180.0, 140.0).unwrap();
        let candidate = |text_rect| Candidate {
            kind: CandidateKind::StoryText,
            text_rect,
            bubble_rect: bubble.union(text_rect),
            confirmed_bubble_rect: bubble,
            detector_confidence: 0.95,
            has_detector_core: true,
            rotation_radians: 0.0,
        };
        let prediction = || PpOcrPrediction {
            text: String::new(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
            ocr_lines: Vec::new(),
        };
        let lines = vec![
            RecognizedLine {
                candidate: candidate(PixelRect::new(40.0, 50.0, 160.0, 70.0).unwrap()),
                prediction: prediction(),
                crop_bounds: PixelBounds {
                    x: 40,
                    y: 50,
                    width: 120,
                    height: 20,
                },
            },
            RecognizedLine {
                candidate: candidate(PixelRect::new(45.0, 80.0, 205.0, 100.0).unwrap()),
                prediction: prediction(),
                crop_bounds: PixelBounds {
                    x: 45,
                    y: 80,
                    width: 160,
                    height: 20,
                },
            },
        ];

        let merged = merge_group_candidate(&lines, 400, 300);

        assert_eq!(merged.confirmed_bubble_rect, bubble);
        assert!(merged.bubble_rect.x1 > bubble.x1);
    }

    #[test]
    fn grouped_source_appearance_retains_real_color_changes_in_reading_order() {
        let candidate = |text_rect| Candidate {
            kind: CandidateKind::StoryText,
            text_rect,
            bubble_rect: PixelRect::new(10.0, 10.0, 190.0, 110.0).unwrap(),
            confirmed_bubble_rect: PixelRect::new(10.0, 10.0, 190.0, 110.0).unwrap(),
            detector_confidence: 0.95,
            has_detector_core: true,
            rotation_radians: 0.0,
        };
        let prediction = |color, slight_variation, stroke_color| PpOcrPrediction {
            text: "line".to_owned(),
            confidence: 0.95,
            text_color: color,
            stroke_color,
            has_stroke_color: true,
            appearance_bands: vec![PpOcrAppearanceBand {
                top_ratio: 0.0,
                bottom_ratio: 1.0,
                text_color: [
                    color[0] + slight_variation,
                    color[1] + slight_variation,
                    color[2] + slight_variation,
                ],
                stroke_color,
                has_stroke_color: true,
            }],
            ocr_lines: Vec::new(),
        };
        let lines = vec![
            RecognizedLine {
                candidate: candidate(PixelRect::new(30.0, 20.0, 170.0, 40.0).unwrap()),
                prediction: prediction([0, 0, 0], 0, [160, 160, 160]),
                crop_bounds: PixelBounds {
                    x: 30,
                    y: 20,
                    width: 140,
                    height: 20,
                },
            },
            RecognizedLine {
                candidate: candidate(PixelRect::new(30.0, 45.0, 170.0, 65.0).unwrap()),
                prediction: prediction([0, 0, 0], 5, [224, 224, 224]),
                crop_bounds: PixelBounds {
                    x: 30,
                    y: 45,
                    width: 140,
                    height: 20,
                },
            },
            RecognizedLine {
                candidate: candidate(PixelRect::new(30.0, 70.0, 170.0, 90.0).unwrap()),
                prediction: prediction([32, 96, 224], 0, [0, 0, 0]),
                crop_bounds: PixelBounds {
                    x: 30,
                    y: 70,
                    width: 140,
                    height: 20,
                },
            },
        ];

        let bands =
            grouped_appearance_bands(&lines, PixelRect::new(30.0, 20.0, 170.0, 90.0).unwrap());

        assert_eq!(
            bands.len(),
            2,
            "near-identical black lines share one palette"
        );
        assert!(bands[0].position_millionths < bands[1].position_millionths);
        assert_eq!(bands[1].text_color, [32, 96, 224]);
    }

    fn cache_key(
        source_text: &str,
        batch_source_texts: &[String],
        layout: Option<HskLayoutConstraints>,
        preceding_english: &[String],
        following_english: &[String],
        provenance: DirectSourceProvenance,
    ) -> String {
        translation_cache_key(
            source_text,
            HskUtteranceKind::Dialogue,
            layout,
            batch_source_texts,
            &[],
            preceding_english,
            following_english,
            provenance,
            LearningMode::Natural,
            2,
            "qwen",
            "model-r1",
            "prompt-r1",
            "validator-r1",
            "control-r1",
        )
    }

    #[test]
    fn cache_key_separates_every_cobatched_source_span() {
        let first = vec!["First".to_owned(), "Neighbor A".to_owned()];
        let second = vec!["First".to_owned(), "Neighbor B".to_owned()];
        assert_ne!(
            cache_key("First", &first, None, &[], &[], DirectSourceProvenance::Dom,),
            cache_key(
                "First",
                &second,
                None,
                &[],
                &[],
                DirectSourceProvenance::Dom,
            )
        );
    }

    #[test]
    fn cache_key_separates_directional_source_context() {
        let neighbor = vec!["Aldrin had already left.".to_owned()];
        assert_ne!(
            cache_key(
                "He closed the door.",
                &["He closed the door.".to_owned()],
                None,
                &neighbor,
                &[],
                DirectSourceProvenance::Dom,
            ),
            cache_key(
                "He closed the door.",
                &["He closed the door.".to_owned()],
                None,
                &[],
                &neighbor,
                DirectSourceProvenance::Dom,
            )
        );
    }

    #[test]
    fn cache_key_separates_image_layout_constraints() {
        let batch = vec!["Leave now".to_owned()];
        let compact = Some(HskLayoutConstraints {
            max_characters: 24,
            max_lines: 2,
        });
        let roomy = Some(HskLayoutConstraints {
            max_characters: 48,
            max_lines: 4,
        });
        assert_ne!(
            cache_key(
                "Leave now",
                &batch,
                compact,
                &[],
                &[],
                DirectSourceProvenance::Ocr,
            ),
            cache_key(
                "Leave now",
                &batch,
                roomy,
                &[],
                &[],
                DirectSourceProvenance::Ocr,
            )
        );
    }

    #[test]
    fn dom_cache_identity_preserves_authoritative_line_breaks() {
        assert_ne!(
            cache_key(
                "First line\nSecond line",
                &["First line\nSecond line".to_owned()],
                None,
                &[],
                &[],
                DirectSourceProvenance::Dom,
            ),
            cache_key(
                "First line Second line",
                &["First line Second line".to_owned()],
                None,
                &[],
                &[],
                DirectSourceProvenance::Dom,
            )
        );
        assert_eq!(
            cache_key(
                "  LEAVE \n NOW ",
                &["  LEAVE \n NOW ".to_owned()],
                None,
                &[],
                &[],
                DirectSourceProvenance::Ocr,
            ),
            cache_key(
                "LEAVE NOW",
                &["LEAVE NOW".to_owned()],
                None,
                &[],
                &[],
                DirectSourceProvenance::Ocr,
            )
        );
    }

    #[test]
    fn cache_key_covers_policy_model_validator_and_bounded_translated_context() {
        let context = (0..7)
            .map(|index| HskPrecedingUtterance {
                source_english: format!("source-{index}"),
                chinese: format!("中文-{index}"),
            })
            .collect::<Vec<_>>();
        let make = |mode,
                    level,
                    model_revision: &str,
                    prompt_hash: &str,
                    validator_hash: &str,
                    control_revision: &str,
                    context: &[HskPrecedingUtterance]| {
            translation_cache_key(
                "Leave now",
                HskUtteranceKind::Dialogue,
                None,
                &["Leave now".to_owned()],
                context,
                &[],
                &[],
                DirectSourceProvenance::Ocr,
                mode,
                level,
                "qwen",
                model_revision,
                prompt_hash,
                validator_hash,
                control_revision,
            )
        };
        let base = make(
            LearningMode::Natural,
            2,
            "model-r1",
            "prompt-r1",
            "validator-r1",
            "control-r1",
            &context,
        );
        assert_ne!(
            base,
            make(
                LearningMode::Strict,
                2,
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
                &context,
            )
        );
        assert_ne!(
            base,
            make(
                LearningMode::Natural,
                3,
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
                &context,
            )
        );
        assert_ne!(
            base,
            make(
                LearningMode::Natural,
                2,
                "model-r2",
                "prompt-r1",
                "validator-r1",
                "control-r1",
                &context,
            )
        );
        assert_ne!(
            base,
            make(
                LearningMode::Natural,
                2,
                "model-r1",
                "prompt-r2",
                "validator-r1",
                "control-r1",
                &context,
            )
        );
        assert_ne!(
            base,
            make(
                LearningMode::Natural,
                2,
                "model-r1",
                "prompt-r1",
                "validator-r2",
                "control-r1",
                &context,
            )
        );
        assert_ne!(
            base,
            make(
                LearningMode::Natural,
                2,
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r2",
                &context,
            )
        );
        assert_eq!(
            base,
            make(
                LearningMode::Natural,
                2,
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
                &context[1..],
            )
        );
    }

    fn test_document_block(index: usize) -> DocumentSourceBlock {
        DocumentSourceBlock {
            parent_block_id: format!("parent-{index}"),
            sub_item_order: 0,
            item_id: format!("block-{index}"),
            source_index: index as u32,
            item_order: 0,
            kind: SourceSpanKind::Prose,
            provenance: SourceProvenance::Dom,
            text: format!("Source block {index}."),
        }
    }

    fn test_document_piece(block_index: usize, piece_order: usize) -> DocumentPiece {
        DocumentPiece {
            piece_id: format!("block-{block_index}::{piece_order}"),
            block_index,
            piece_order,
            source_text: format!("piece-{block_index}-{piece_order}"),
            separator_after: "\n".to_owned(),
            kind: SourceSpanKind::Prose,
            token_count: 10,
        }
    }

    fn preserved_context_span(source_text: &str, token_count: usize) -> DocumentContextSpan {
        DocumentContextSpan {
            piece_id: None,
            source_text: source_text.to_owned(),
            separator_after: String::new(),
            token_count,
        }
    }

    #[test]
    fn document_scheduler_reprioritizes_focus_without_mixing_canonical_gaps() {
        let blocks = (0..10).map(test_document_block).collect::<Vec<_>>();
        let mut remaining = (0..10)
            .map(|index| test_document_piece(index, 0))
            .collect::<Vec<_>>();

        let visible = HashSet::from(["block-5".to_owned()]);
        let (first, reason) = take_next_document_batch(&mut remaining, &blocks, &visible, true);
        assert_eq!(reason, "visible");
        assert_eq!(
            first
                .iter()
                .map(|piece| piece.block_index)
                .collect::<Vec<_>>(),
            [5]
        );

        let jumped = HashSet::from(["block-8".to_owned()]);
        let (second, reason) = take_next_document_batch(&mut remaining, &blocks, &jumped, false);
        assert_eq!(reason, "visible");
        assert_eq!(
            second
                .iter()
                .map(|piece| piece.block_index)
                .collect::<Vec<_>>(),
            [8]
        );

        let (ordered, reason) =
            take_next_document_batch(&mut remaining, &blocks, &HashSet::new(), false);
        assert_eq!(reason, "ordered");
        assert_eq!(
            ordered
                .iter()
                .map(|piece| piece.block_index)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn oversized_visible_block_keeps_adjacent_piece_context_across_batches() {
        let blocks = vec![test_document_block(0)];
        let planned = (0..8)
            .map(|piece_order| test_document_piece(0, piece_order))
            .collect::<Vec<_>>();
        let mut remaining = planned.clone();
        let visible = HashSet::from(["block-0".to_owned()]);
        let (first, _) = take_next_document_batch(&mut remaining, &blocks, &visible, true);
        assert_eq!(first.len(), DOCUMENT_BATCH_MAX);
        let (second, _) = take_next_document_batch(&mut remaining, &blocks, &visible, false);
        assert_eq!(
            second
                .iter()
                .map(|piece| piece.piece_order)
                .collect::<Vec<_>>(),
            [6, 7]
        );
        let context_plan = planned
            .iter()
            .map(DocumentContextSpan::from)
            .collect::<Vec<_>>();
        let (preceding, following) =
            document_piece_neighbor_context(&context_plan, &second).unwrap();
        assert_eq!(preceding.len(), MAX_HSK_PRECEDING_UTTERANCES);
        assert_eq!(preceding.first().unwrap(), "piece-0-0\n");
        assert_eq!(preceding.last().unwrap(), "piece-0-5\n");
        assert!(following.is_empty());
    }

    #[test]
    fn visible_jump_keeps_an_adjacent_source_preserved_block_in_raw_context() {
        let blocks = (0..4).map(test_document_block).collect::<Vec<_>>();
        let pieces = [
            test_document_piece(0, 0),
            test_document_piece(2, 0),
            test_document_piece(3, 0),
        ];
        let mut ordered_remaining = pieces.to_vec();
        let (before_barrier, _) =
            take_next_document_batch(&mut ordered_remaining, &blocks, &HashSet::new(), false);
        assert_eq!(
            before_barrier
                .iter()
                .map(|piece| piece.block_index)
                .collect::<Vec<_>>(),
            [0]
        );
        let (after_barrier, _) =
            take_next_document_batch(&mut ordered_remaining, &blocks, &HashSet::new(), false);
        assert_eq!(
            after_barrier
                .iter()
                .map(|piece| piece.block_index)
                .collect::<Vec<_>>(),
            [2, 3]
        );

        let mut remaining = pieces.to_vec();
        let visible = HashSet::from(["block-2".to_owned()]);
        let (batch, reason) = take_next_document_batch(&mut remaining, &blocks, &visible, true);
        assert_eq!(reason, "visible");
        assert_eq!(batch[0].block_index, 2);

        let context_plan = vec![
            DocumentContextSpan::from(&pieces[0]),
            preserved_context_span("preserved source block 1", 20),
            DocumentContextSpan::from(&pieces[1]),
            DocumentContextSpan::from(&pieces[2]),
        ];
        let (preceding, following) =
            document_piece_neighbor_context(&context_plan, &batch).unwrap();
        assert_eq!(
            preceding,
            [
                "piece-0-0\n".to_owned(),
                "preserved source block 1".to_owned()
            ]
        );
        assert_eq!(following, ["piece-3-0\n".to_owned()]);
    }

    #[test]
    fn raw_context_never_skips_past_an_immediate_over_budget_neighbor() {
        let target = test_document_piece(2, 0);
        let context_plan = vec![
            DocumentContextSpan::from(&test_document_piece(0, 0)),
            preserved_context_span("over-budget predecessor", DOCUMENT_CONTEXT_TOKEN_BUDGET + 1),
            DocumentContextSpan::from(&target),
            preserved_context_span("over-budget follower", DOCUMENT_CONTEXT_TOKEN_BUDGET + 1),
            DocumentContextSpan::from(&test_document_piece(4, 0)),
        ];
        let (preceding, following) =
            document_piece_neighbor_context(&context_plan, &[target]).unwrap();
        assert!(preceding.is_empty());
        assert!(following.is_empty());
    }

    #[test]
    fn image_evidence_counts_source_units_once_and_sums_all_generation_stages() {
        let mut file = BenchmarkEvidenceFile::default();
        accumulate_image_language_evidence(&mut file, "job-image", "qwen@test".to_owned(), 3, 40);
        accumulate_image_language_evidence(&mut file, "job-image", "qwen@test".to_owned(), 0, 15);
        let BenchmarkSample::Image(sample) = &file.samples[0] else {
            panic!("expected image evidence");
        };
        assert_eq!(sample.language_unit_count, 3);
        assert_eq!(sample.language_generation_duration_ms, 55);
        let json = serde_json::to_value(file).unwrap();
        assert_eq!(json["samples"][0]["kind"], "image");
        assert_eq!(json["samples"][0]["languageUnitCount"], 3);
    }

    fn pending_region(id: &str) -> PreparedRegion {
        let rect = PixelRect::new(1.0, 1.0, 9.0, 9.0).unwrap();
        PreparedRegion {
            id: id.to_owned(),
            candidate: Candidate {
                kind: CandidateKind::StoryText,
                text_rect: rect,
                bubble_rect: rect,
                confirmed_bubble_rect: rect,
                detector_confidence: 0.99,
                has_detector_core: true,
                rotation_radians: 0.0,
            },
            source_english: "Graduate student".to_owned(),
            faithful_chinese: Some("研究生".to_owned().into()),
            ocr_confidence: 0.99,
            reading_order: 0,
            continuation_group: None,
            role: ImageRegionRole::Dialogue,
            source_line_count: 1,
            prediction: PpOcrPrediction {
                text: "Graduate student".to_owned(),
                confidence: 0.99,
                text_color: [0, 0, 0],
                stroke_color: [255, 255, 255],
                has_stroke_color: false,
                appearance_bands: Vec::new(),
                ocr_lines: Vec::new(),
            },
            appearance_bands: Vec::new(),
            measured_font_height: rect.height(),
            bubble_polygon: rect.polygon(10, 10),
            layout_polygon: rect.polygon(10, 10),
            cleanup: CleanupBatchTask::ready(CleanupBatchResult {
                decisions: HashMap::new(),
            }),
            visible: false,
            translation_queued_at: tokio::time::Instant::now(),
            source_context: SourceContext::default(),
        }
    }

    #[test]
    fn cancellation_is_observed_at_batch_boundaries() {
        let cancel = AtomicBool::new(false);
        assert!(cancellation_boundary(&cancel).is_ok());
        cancel.store(true, Ordering::Release);
        let error = cancellation_boundary(&cancel).unwrap_err();
        assert_eq!(error.code, "CANCELLED");
    }

    #[test]
    fn pending_translation_priority_uses_reading_order_before_enqueue_time_offscreen() {
        let now = tokio::time::Instant::now();
        let mut first_in_reading_order = pending_region("first");
        first_in_reading_order.reading_order = 1;
        first_in_reading_order.translation_queued_at = now + Duration::from_millis(20);

        let mut second_in_reading_order = pending_region("second");
        second_in_reading_order.reading_order = 2;
        second_in_reading_order.translation_queued_at = now;

        let mut pending = vec![second_in_reading_order, first_in_reading_order];
        sort_pending_translation(&mut pending);

        assert_eq!(
            pending
                .iter()
                .map(|region| region.id.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
    }

    #[test]
    fn pending_translation_priority_allows_visible_to_overtake_offscreen() {
        let now = tokio::time::Instant::now();
        let mut offscreen = pending_region("offscreen");
        offscreen.visible = false;
        offscreen.reading_order = 1;
        offscreen.translation_queued_at = now;

        let mut visible = pending_region("visible");
        visible.visible = true;
        visible.reading_order = 99;
        visible.translation_queued_at = now + Duration::from_millis(20);

        let mut pending = vec![offscreen, visible];
        sort_pending_translation(&mut pending);

        assert_eq!(pending[0].id, "visible");
        assert_eq!(pending[1].id, "offscreen");
    }

    #[test]
    fn sparse_offscreen_translation_waits_for_batch_or_final_flush() {
        let mut one = vec![pending_region("bubble-a")];
        one[0].visible = false;
        assert_eq!(
            translation_boundary_action(&one, false, false, false),
            TranslationBoundaryAction::ContinueUpstream
        );
        assert_eq!(
            translation_boundary_action(&one, true, false, false),
            TranslationBoundaryAction::Dispatch(1)
        );
    }

    #[test]
    fn first_visible_translation_is_dispatched_alone_before_throughput_batching() {
        let now = tokio::time::Instant::now();
        let mut pending = (0..TRANSLATION_BATCH_MAX)
            .map(|index| pending_region(&format!("bubble-{index}")))
            .collect::<Vec<_>>();
        for region in &mut pending {
            region.translation_queued_at = now;
        }

        assert_eq!(
            translation_boundary_action(&pending, true, false, true),
            TranslationBoundaryAction::Dispatch(1)
        );
        assert_eq!(
            translation_boundary_action(&pending, true, false, false),
            TranslationBoundaryAction::Dispatch(TRANSLATION_BATCH_MAX)
        );
    }

    #[test]
    fn throughput_starts_only_after_a_visible_final_is_published() {
        let mut phase = TranslationLatencyPhase::AwaitingFirstVisibleRegion;

        complete_translation_batch(&mut phase, false);
        assert_eq!(phase, TranslationLatencyPhase::AwaitingFirstVisibleRegion);

        complete_translation_batch(&mut phase, true);

        assert_eq!(phase, TranslationLatencyPhase::Throughput);
    }

    #[test]
    fn visible_translation_dispatches_without_a_timer_or_full_batch() {
        let mut visible = vec![pending_region("bubble-a"), pending_region("bubble-b")];
        visible[0].visible = true;
        visible[1].visible = true;
        assert_eq!(
            translation_boundary_action(&visible, false, false, false),
            TranslationBoundaryAction::Dispatch(2)
        );

        visible[1].visible = false;
        assert_eq!(
            translation_boundary_action(&visible, false, false, false),
            TranslationBoundaryAction::Dispatch(1)
        );
    }

    #[tokio::test]
    async fn translation_batches_cap_at_six_and_cancellation_wins_at_boundaries() {
        for available in 1..32 {
            let count = translation_batch_len(available);
            assert!((1..=TRANSLATION_BATCH_MAX).contains(&count));
            assert!(count <= available);
            let remaining = available - count;
            assert!(remaining == 0 || remaining >= TRANSLATION_BATCH_MIN);
        }

        let now = tokio::time::Instant::now();
        let mut full = (0..7)
            .map(|index| pending_region(&format!("bubble-{index}")))
            .collect::<Vec<_>>();
        for region in &mut full {
            region.translation_queued_at = now;
        }
        assert_eq!(
            translation_boundary_action(&full[..6], false, false, false),
            TranslationBoundaryAction::Dispatch(TRANSLATION_BATCH_MAX)
        );
        assert_eq!(
            translation_boundary_action(&full, false, false, false),
            TranslationBoundaryAction::Dispatch(4)
        );

        let cancel = AtomicBool::new(false);
        assert!(cancellation_boundary(&cancel).is_ok());
        cancel.store(true, Ordering::Release);
        assert_eq!(
            translation_boundary_action(&full, false, cancel.load(Ordering::Acquire), false,),
            TranslationBoundaryAction::Cancelled
        );
        assert_eq!(
            cancellation_boundary(&cancel).unwrap_err().code,
            "CANCELLED"
        );
    }

    #[test]
    fn browser_preprocessing_pool_has_exactly_six_threads() {
        let pool = global_preprocessing_pool().unwrap();
        assert_eq!(pool.thread_count(), PREPROCESSING_THREADS);
        assert_eq!(PREPROCESSING_THREADS, 6);
    }
}
