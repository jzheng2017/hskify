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

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use camino::Utf8PathBuf;
use hsk_control::{
    HskControl, HskLevel as ControlHskLevel, LookupRegionContext as ControlLookupRegion,
    ProperName, ProperNameReason, ValidationReport, ViolationReason,
};
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage, imageops::crop_imm};
use imageproc::geometric_transformations::{Border, Interpolation, rotate_about_center};
use koharu_app::llm::{
    HSK_TRANSLATION_MODEL, HskLearningMode, HskNameHandling, HskPrecedingUtterance,
    HskProtectedName, HskRepairUtterance, HskSourceUtterance, HskTranslationBatchRequest,
    HskTranslationOutcome, HskTranslationRepairBatchRequest, HskUtteranceKind,
    MAX_HSK_LAYOUT_CHARACTERS, MAX_HSK_LAYOUT_LINES, MAX_HSK_PRECEDING_UTTERANCES,
    MIN_HSK_LAYOUT_CHARACTERS,
};
use koharu_app::{App, AppConfig};
use koharu_llm::page_understanding::{
    PageEntityType, PageFontCategory, PagePoint, PageRegionDecision, PageRegionEvidence,
    PageRegionRole, PageRole, PageStyleEvidence, PageTextAlignment, PageUnderstandingRequest,
    PageUnderstandingResult, PageWritingMode, QwenPageUnderstanding, probe_qwen_page_understanding,
};
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
use tokio::sync::{Mutex as AsyncMutex, Notify, OnceCell, oneshot};

use self::geometry::{
    Candidate, CandidateKind, PixelBounds, PixelRect, Tile, bubbles_for_tile,
    candidates_for_text_boxes, next_detector_batch_count, ocr_crop_rect, overlapping_tiles,
    prioritize_tiles, reading_order_key, spatially_dedupe, take_finalized_lines,
    text_candidate_is_confirmed,
};
use self::patch::{
    CleanupMask, CleanupQuality, PatchPng, broaden_cleanup_mask, bubble_component_bounds,
    bubble_id_for_rect, bubble_id_mask, compact_cleanup_mask, crop_probability_map,
    label_bubble_components, make_inpainted_patch, merge_binary_mask, merge_cleanup_mask,
    merge_probability_map, merge_source_guided_glyph_probabilities, protected_pixels_match,
    region_polygons, score_cleanup_candidate_local, verified_text_mask_for_regions_local,
};
use self::ppocr::{
    MAX_LINE_BATCH_SIZE, PpOcrAppearanceBand, PpOcrLine, PpOcrPrediction, PpOcrSmallRecognizer,
};
use self::ppocr_detector::PpOcrSmallDetector;
use crate::chapter_session::{
    ChapterEntity, ChapterEntityType, ChapterSessionStore, DialogueNode, PageAnalysis, PageSurface,
    PageSurfaceKind, RegionPlan, RegionRole,
};
use crate::contracts::{
    BrowserJobRequest, BrowserJobStage, BrowserSurfaceKind, BrowserTextColorBand,
    BrowserTextLayout, BrowserTextStyle, FontCategory, HskLevel, HskRepairState, LearningMode,
    LookupRegion, LookupResult, LookupToken, NameTranslation, NormalizedRect, Point,
    PreservedArtworkRegion, RegionConfidenceEvidence, RegionEntitySpan, RegionEntityType,
    TeachingTerm, TeachingTermReason, TextAlignment, TranslatedHskStatus, TranslatedRegion,
    TranslatedRegionRole, WritingMode,
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
const TRANSLATION_MAX_FLUSH_DELAY: Duration = Duration::from_millis(75);
const BROWSER_QWEN_INFERENCE_THREADS: i32 = 6;
const MAX_HSK_REPAIR_ATTEMPTS: u8 = 2;
// Cleanup is a bounded, optional stage. A stalled inpainting/quality task must
// never hold the ordered language stream indefinitely: the source pixels stay
// intact and the region is published as unreadable when this deadline expires.
const CLEANUP_RESULT_TIMEOUT: Duration = Duration::from_secs(90);
const TRANSLATION_CACHE_SCHEMA: &str = "hskify-chapter-session-translation-v3-2026-08-02";
const PAGE_WINDOW_OVERLAP: usize = 8;
// Multimodal inference should see enough artwork to classify a region, but a
// continuous reader strip must not be sent to the projector at its full
// height for every bounded language window.  The evidence viewport is
// derived from the accepted OCR/bubble geometry, never from a chapter or
// reader-specific crop rule.
const PAGE_EVIDENCE_MAX_PIXELS: u64 = 6_000_000;
const PAGE_EVIDENCE_CROP_RATIO: f32 = 0.82;
const PAGE_EVIDENCE_MIN_MARGIN: f32 = 96.0;
const PAGE_EVIDENCE_MAX_MARGIN: f32 = 512.0;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RegionLookupContext {
    pub(crate) source_english: String,
    pub(crate) base_chinese: String,
    pub(crate) displayed_chinese: String,
    pub(crate) proper_names: Vec<ProperName>,
}

impl RegionLookupContext {
    /// Cached dictionary context is part of the terminal region identity. A
    /// stale context must never survive a result-cache replay because it can
    /// explain a different sentence or reintroduce an entity from another
    /// page. Keep this validation structural: it checks exact field identity
    /// and bounded name records, not capitalization or lexical wordlists.
    pub(crate) fn validate_against(&self, region: &TranslatedRegion) -> Result<()> {
        if self.source_english != region.source_english
            || self.base_chinese != region.base_chinese
            || self.displayed_chinese != region.displayed_chinese
        {
            bail!("cached lookup context does not match its translated region");
        }
        let mut names = HashSet::with_capacity(self.proper_names.len());
        for name in &self.proper_names {
            if name.text.trim().is_empty() || !names.insert(name.text.as_str()) {
                bail!("cached lookup context contains an empty or duplicate proper name");
            }
        }
        Ok(())
    }
}

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
pub(crate) struct CleaningInput {
    pub source: Arc<DynamicImage>,
    pub request: BrowserJobRequest,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub(crate) struct CleaningError {
    pub code: &'static str,
    pub message: String,
}

impl CleaningError {
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
        Self::new("CANCELLED", "Cleaning was cancelled.")
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
pub(crate) trait CleaningPipeline: Send + Sync {
    async fn warm_up(&self) -> std::result::Result<(), CleaningError>;

    async fn run(
        &self,
        input: CleaningInput,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), CleaningError>;

    async fn lookup(
        &self,
        input: LookupInput,
        region: Option<RegionLookupContext>,
    ) -> std::result::Result<LookupResult, CleaningError>;

    /// Rebuild chapter ordering state before a terminal result-cache replay.
    /// A cached page skips model execution, but it must still participate in
    /// the same analysis/language barriers and dialogue/entity graph as a
    /// freshly processed page. Test pipelines may use the default no-op.
    fn restore_cached_context(
        &self,
        _request: &BrowserJobRequest,
        _regions: &[TranslatedRegion],
        _preserved_artwork: &[PreservedArtworkRegion],
        _unreadable_regions: &[crate::contracts::UnreadableRegion],
    ) -> std::result::Result<(), CleaningError> {
        Ok(())
    }

    /// Record a non-retryable pre-pipeline failure in the chapter graph.  A
    /// page that fails before [`run`] starts still occupies a canonical page
    /// slot; leaving that slot open would make later pages wait forever for a
    /// predecessor that can no longer contribute context.  Lightweight test
    /// pipelines do not need a graph, so the default is intentionally a
    /// no-op.
    fn mark_page_terminal(
        &self,
        _request: &BrowserJobRequest,
    ) -> std::result::Result<(), CleaningError> {
        Ok(())
    }

    /// Release all chapter-owned dialogue/entity state once the browser has
    /// sealed or cancelled a chapter.  The default keeps lightweight test
    /// pipelines source-compatible while the production pipeline owns the
    /// actual session graph.
    fn close_chapter(&self, _page_session_id: &str) {}

    fn resources_ready(&self) -> bool;
}

pub(crate) struct KoharuPipeline {
    cache_root: PathBuf,
    cuda_scheduler: Arc<CudaScheduler>,
    resident: OnceCell<Arc<ResidentState>>,
    hsk_control: OnceCell<Arc<HskControl>>,
    inference_ready: OnceCell<()>,
    translation_cache: Mutex<TranslationCache>,
    chapter_sessions: Mutex<ChapterSessionStore>,
    chapter_progress_notify: Arc<Notify>,
}

impl KoharuPipeline {
    pub(crate) fn new(cache_root: PathBuf) -> Self {
        Self {
            cache_root,
            cuda_scheduler: global_cuda_scheduler(),
            resident: OnceCell::new(),
            hsk_control: OnceCell::new(),
            inference_ready: OnceCell::new(),
            translation_cache: Mutex::new(TranslationCache::default()),
            chapter_sessions: Mutex::new(ChapterSessionStore::default()),
            chapter_progress_notify: Arc::new(Notify::new()),
        }
    }

    fn resource_paths(&self) -> Result<ResidentResourcePaths> {
        ResidentResourcePaths::discover()
    }

    fn record_page_analysis(
        &self,
        request: &BrowserJobRequest,
        width: u32,
        height: u32,
        kind: PageSurfaceKind,
        regions: &[RegionPlan],
        complete: bool,
    ) -> std::result::Result<(), CleaningError> {
        let mut sessions = self.chapter_sessions.lock().map_err(|_| {
            CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
        })?;
        sessions
            .session_mut(&request.page_session_id)
            .record_analysis(PageAnalysis {
                surface: PageSurface {
                    session_id: request.page_session_id.clone(),
                    page_index: request.page_index,
                    source_sha256: request.source_sha256.clone(),
                    width,
                    height,
                    kind,
                },
                regions: regions.to_vec(),
                complete,
            });
        drop(sessions);
        self.chapter_progress_notify.notify_waiters();
        Ok(())
    }

    async fn resident(&self) -> Result<&Arc<ResidentState>> {
        let resources = self.resource_paths()?;
        let runtime_root = resources.runtime_root().to_path_buf();
        let app_state_root = self.cache_root.join("browser-runtime").join("app-state");
        self.resident
            .get_or_try_init(|| async move {
                ResidentState::load(runtime_root, app_state_root, resources)
                    .await
                    .map(Arc::new)
            })
            .await
    }

    async fn hsk_control(&self) -> Result<&Arc<HskControl>> {
        let resources = self.resource_paths()?;
        self.hsk_control
            .get_or_try_init(|| async move {
                tokio::task::spawn_blocking(move || {
                    let hsk_json = std::fs::read_to_string(&resources.hsk)
                        .with_context(|| format!("read HSK data {}", resources.hsk.display()))?;
                    let dictionary_json = std::fs::read_to_string(&resources.dictionary)
                        .with_context(|| {
                            format!("read dictionary data {}", resources.dictionary.display())
                        })?;
                    HskControl::from_json(&hsk_json, &dictionary_json)
                        .context("load deterministic HSK control data")
                        .map(Arc::new)
                })
                .await
                .context("join HSK data loader")?
            })
            .await
    }

    /// Keep the analysis frontier in canonical chapter order while allowing
    /// detector/OCR/cleanup work for later pages to run ahead. A page that is
    /// already admitted to this daemon waits for every earlier page in the
    /// browser's immutable chapter order to expose a page-understanding
    /// decision. The order is registered before model work starts, so a later
    /// upload cannot outrun an earlier upload that is still being admitted.
    async fn wait_for_preceding_page_analysis(
        &self,
        request: &BrowserJobRequest,
        cancel: &AtomicBool,
        sink: &JobUpdateSink,
    ) -> std::result::Result<(), CleaningError> {
        loop {
            cancellation_boundary(cancel)?;
            if sink.is_cancelled() {
                return Err(CleaningError::cancelled());
            }
            let waiting = self
                .chapter_sessions
                .lock()
                .map_err(|_| {
                    CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
                })?
                .session(&request.page_session_id)
                .is_some_and(|session| {
                    session
                        .expected_pages
                        .iter()
                        .copied()
                        .filter(|page_index| *page_index < request.page_index)
                        .any(|page_index| !session.analysis_complete(page_index))
                });
            if !waiting {
                return Ok(());
            }
            let notified = self.chapter_progress_notify.notified();
            // The state check precedes registration of the notification
            // future, so re-check after registration to close the notify race
            // between a preceding page committing its analysis and this page
            // beginning to wait.
            let still_waiting = self
                .chapter_sessions
                .lock()
                .map_err(|_| {
                    CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
                })?
                .session(&request.page_session_id)
                .is_some_and(|session| {
                    session
                        .expected_pages
                        .iter()
                        .copied()
                        .filter(|page_index| *page_index < request.page_index)
                        .any(|page_index| !session.analysis_complete(page_index))
                });
            if still_waiting {
                tokio::select! {
                    _ = notified => {}
                    _ = tokio::time::sleep(Duration::from_millis(250)) => {}
                }
            }
        }
    }

    /// The language stream is ordered separately from analysis. This barrier
    /// is deliberately reached only when translating a window, after the
    /// current page has already performed detection, OCR, semantic adjudication
    /// and restoration preparation. Future pages can therefore do expensive
    /// vision work while the preceding page finishes its terminal language
    /// publication, without exposing completion-order context to the model.
    async fn wait_for_preceding_page_language(
        &self,
        request: &BrowserJobRequest,
        cancel: &AtomicBool,
        sink: &JobUpdateSink,
    ) -> std::result::Result<(), CleaningError> {
        loop {
            cancellation_boundary(cancel)?;
            if sink.is_cancelled() {
                return Err(CleaningError::cancelled());
            }
            let waiting = self
                .chapter_sessions
                .lock()
                .map_err(|_| {
                    CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
                })?
                .session(&request.page_session_id)
                .is_some_and(|session| {
                    session
                        .expected_pages
                        .iter()
                        .copied()
                        .filter(|page_index| *page_index < request.page_index)
                        .any(|page_index| !session.language_complete(page_index))
                });
            if !waiting {
                return Ok(());
            }
            let notified = self.chapter_progress_notify.notified();
            let still_waiting = self
                .chapter_sessions
                .lock()
                .map_err(|_| {
                    CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
                })?
                .session(&request.page_session_id)
                .is_some_and(|session| {
                    session
                        .expected_pages
                        .iter()
                        .copied()
                        .filter(|page_index| *page_index < request.page_index)
                        .any(|page_index| !session.language_complete(page_index))
                });
            if still_waiting {
                tokio::select! {
                    _ = notified => {}
                    _ = tokio::time::sleep(Duration::from_millis(250)) => {}
                }
            }
        }
    }

    async fn ready_models(&self) -> Result<(&Arc<ResidentState>, &Arc<HskControl>)> {
        let (resident, control) = tokio::try_join!(self.resident(), self.hsk_control())?;
        self.inference_ready
            .get_or_try_init(|| {
                let resident = Arc::clone(resident);
                async move {
                    tokio::task::spawn_blocking(move || {
                        resident.prime_non_language_inference()?;
                        resident.prime_language_inference()
                    })
                    .await
                    .context("join resident full-pipeline inference warm-up")??;
                    Ok::<(), anyhow::Error>(())
                }
            })
            .await?;
        Ok((resident, control))
    }

    async fn run_direct(
        &self,
        input: CleaningInput,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), CleaningError> {
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
            return Err(CleaningError::new(
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
                CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
            })?;
            let chapter = chapter_sessions.session_mut(&input.request.page_session_id);
            chapter.register_expected_pages(&input.request.chapter_page_order);
            chapter.register_surface(PageSurface {
                session_id: input.request.page_session_id.clone(),
                page_index: input.request.page_index,
                source_sha256: input.request.source_sha256.clone(),
                width: image_width,
                height: image_height,
                kind: surface_kind.clone(),
            });
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
        let (resident, control) = self.ready_models().await.map_err(CleaningError::pipeline)?;
        let preprocessing = global_preprocessing_pool().map_err(CleaningError::pipeline)?;
        cancellation_boundary(cancel.as_ref())?;

        let mut tiles = overlapping_tiles(image_width, image_height);
        let total_tiles = tiles.len();
        let total_tiles_u32 = u32::try_from(total_tiles).unwrap_or(u32::MAX);
        let mut processed_tiles = 0usize;
        let mut seen_text_blocks = Vec::<PixelRect>::new();
        let mut recognized_lines = Vec::<RecognizedLine>::new();
        let mut text_probabilities = ProbabilityMap::zeros(image_width, image_height);
        let mut pending_translation = Vec::<PreparedRegion>::new();
        let mut translation_latency_phase = TranslationLatencyPhase::AwaitingFirstVisibleRegion;
        let mut repair_queue = RepairQueue::default();
        // The daemon chapter session supplies preceding dialogue immediately
        // before each translation window. Never seed it from browser fields.
        let mut dialogue_context = Vec::new();
        let mut prepared_next_tiles: Option<TileBatchTask> = None;
        let mut deferred_detector_candidates = Vec::<Candidate>::new();
        let mut deferred_detector_lines = Vec::<RecognizedLine>::new();
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

        while !tiles.is_empty() {
            cancellation_boundary(cancel.as_ref())?;
            if sink.is_cancelled() {
                return Err(CleaningError::cancelled());
            }
            let viewport = sink.viewport();
            prioritize_tiles(
                &mut tiles,
                &viewport.visible_rects,
                viewport.active,
                image_width,
                image_height,
                input.request.settings.reading_direction,
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
                    .map_err(CleaningError::pipeline)?
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
                    .map_err(CleaningError::pipeline)?
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
            let admission_viewport = sink.viewport();
            prioritize_tiles(
                &mut tiles,
                &admission_viewport.visible_rects,
                admission_viewport.active,
                image_width,
                image_height,
                input.request.settings.reading_direction,
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
                let detector = resident.detector.lock().map_err(|_| {
                    CleaningError::new("MODEL_STATE_FAILED", "Detector lock poisoned.")
                })?;
                let detections = detector
                    .inference_tiles(&tile_images)
                    .context("run true-batched CUDA comic text detection")
                    .map_err(CleaningError::pipeline)?;
                let ocr_detections = resident
                    .ocr_detector
                    .lock()
                    .map_err(|_| {
                        CleaningError::new("MODEL_STATE_FAILED", "OCR detector lock poisoned.")
                    })?
                    .detect_tiles(&tile_images)
                    .context("run true-batched CUDA PP-OCR text detection")
                    .map_err(CleaningError::pipeline)?;
                (detections, ocr_detections)
            };
            let detector_elapsed = detector_started.elapsed();
            cancellation_boundary(cancel.as_ref())?;
            if detections.len() != tile_batch.len() || ocr_detections.len() != tile_batch.len() {
                return Err(CleaningError::new(
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
                &mut repair_queue,
                &mut translation_latency_phase,
                false,
                false,
            )
            .await?;
            let comic_bubbles = detections
                .iter()
                .zip(&tile_batch)
                .map(|(detection, tile)| bubbles_for_tile(detection, tile))
                .collect::<Vec<_>>();
            // PP-OCRv6-small owns every text-line proposal. The comic model
            // contributes bubble topology only; its text-class detections are
            // intentionally not promoted to OCR candidates because they are
            // object boxes, not calibrated line geometry. This keeps the
            // recognizer input independent of the legacy bubble heuristic.
            let candidates = ocr_detections
                .iter()
                .zip(&comic_bubbles)
                .zip(&tile_batch)
                .flat_map(|((detection, bubbles), tile)| {
                    candidates_for_text_boxes(detection, bubbles, tile, image_width, image_height)
                })
                .collect::<Vec<_>>();
            let mut candidates = spatially_dedupe(candidates, &seen_text_blocks);
            candidates.retain(text_candidate_is_confirmed);
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
            let original_candidates = candidates.clone();
            let mut masked_candidates = candidates;
            let mut masked_lines = Vec::new();
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
                masked_lines.extend(ocr_result.accepted);
                rejected_ocr_lines.extend(ocr_result.rejected);
            }
            let (accepted, deferred, disputed) =
                verified_source_guided_ocr_lines(original_candidates, masked_lines);
            deferred_detector_lines.extend(deferred);
            deferred_detector_candidates.extend(disputed);
            for line in accepted {
                seen_text_blocks.push(line.candidate.text_rect);
                recognized_lines.push(line);
            }
            processed_tiles += tile_batch.len();
            if !tiles.is_empty() {
                let finalized_lines =
                    take_finalized_lines(&mut recognized_lines, &tiles, image_width, image_height);
                if finalized_lines.is_empty() {
                    continue;
                }
                let viewport = sink.viewport();
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
                    control,
                    &dialogue_context,
                    &sink,
                    cancel.clone(),
                    &self.cuda_scheduler,
                    &preprocessing,
                    &mut bubble_masks,
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
                    &mut repair_queue,
                    &mut translation_latency_phase,
                    false,
                    false,
                )
                .await?;
            }
            cancellation_boundary(cancel.as_ref())?;
        }

        recognized_lines.extend(deferred_page_lines);
        if !recognized_lines.is_empty() {
            let finalized_lines = std::mem::take(&mut recognized_lines);
            let (prepared_regions, source_guided_probabilities, region_plans) =
                prepare_grouped_regions(
                    Arc::clone(resident),
                    source.clone(),
                    finalized_lines,
                    &input.request,
                    control,
                    &dialogue_context,
                    &sink,
                    cancel.clone(),
                    &self.cuda_scheduler,
                    &preprocessing,
                    &mut bubble_masks,
                    text_probabilities,
                    0.78,
                )
                .await?;
            text_probabilities = source_guided_probabilities;
            page_region_plans.extend(region_plans);
            self.record_page_analysis(
                &input.request,
                image_width,
                image_height,
                surface_kind.clone(),
                &page_region_plans,
                false,
            )?;
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
            &mut repair_queue,
            &mut translation_latency_phase,
            true,
            false,
        )
        .await?;
        // The detector frontier is complete. Finish every terminal repair
        // already known before publishing the remaining page regions.
        repair_queue.finish_primary_phase();
        self.process_queued_repairs(
            resident,
            control,
            &input.request,
            &mut repair_queue,
            cancel.clone(),
            &sink,
            image_width,
            image_height,
            0.81,
            false,
        )
        .await?;
        while !deferred_detector_candidates.is_empty() {
            let ocr_result = ocr_batch(
                resident,
                source.clone(),
                &mut deferred_detector_candidates,
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
                merge_best_recognized_line(&mut deferred_detector_lines, line);
            }
            rejected_ocr_lines.extend(ocr_result.rejected);
        }
        // Deferred detector OCR still belongs to the same page-wide text
        // identity set as the fast visible lines. Keep one spatial identity
        // instead of publishing duplicate patches.
        for line in deferred_detector_lines {
            if seen_text_blocks
                .iter()
                .any(|known| text_rects_represent_same_block(line.candidate.text_rect, *known))
            {
                continue;
            }
            seen_text_blocks.push(line.candidate.text_rect);
            recognized_lines.push(line);
        }

        let (prepared_regions, _text_probabilities, region_plans) = prepare_grouped_regions(
            Arc::clone(resident),
            source.clone(),
            recognized_lines,
            &input.request,
            control,
            &dialogue_context,
            &sink,
            cancel.clone(),
            &self.cuda_scheduler,
            &preprocessing,
            &mut bubble_masks,
            text_probabilities,
            0.88,
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
        pending_translation.extend(prepared_regions);
        self.flush_translation_queue(
            resident,
            control,
            &input.request,
            &mut pending_translation,
            cancel.clone(),
            &sink,
            0.90,
            image_width,
            image_height,
            &mut dialogue_context,
            &mut repair_queue,
            &mut translation_latency_phase,
            true,
            true,
        )
        .await?;
        cancellation_boundary(cancel.as_ref())?;
        repair_queue.finish_primary_phase();
        self.process_queued_repairs(
            resident,
            control,
            &input.request,
            &mut repair_queue,
            cancel.clone(),
            &sink,
            image_width,
            image_height,
            0.94,
            false,
        )
        .await?;
        cancellation_boundary(cancel.as_ref())?;
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
        self.record_page_analysis(
            &input.request,
            image_width,
            image_height,
            surface_kind,
            &region_plans.into_values().collect::<Vec<_>>(),
            true,
        )?;
        self.chapter_sessions
            .lock()
            .map_err(|_| {
                CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
            })?
            .session_mut(&input.request.page_session_id)
            .mark_language_complete(input.request.page_index);
        self.chapter_progress_notify.notify_waiters();
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

    #[allow(clippy::too_many_arguments)]
    async fn flush_translation_queue(
        &self,
        resident: &ResidentState,
        control: &HskControl,
        request: &BrowserJobRequest,
        pending: &mut Vec<PreparedRegion>,
        cancel: Arc<AtomicBool>,
        sink: &JobUpdateSink,
        overall_progress: f32,
        image_width: u32,
        image_height: u32,
        context: &mut Vec<HskPrecedingUtterance>,
        repair_queue: &mut RepairQueue,
        latency_phase: &mut TranslationLatencyPhase,
        force: bool,
        page_semantics_complete: bool,
    ) -> std::result::Result<(), CleaningError> {
        while !pending.is_empty() {
            prioritize_pending_translation(pending, sink, image_width, image_height);
            let count = match translation_boundary_action(
                pending,
                force,
                tokio::time::Instant::now(),
                cancel.load(Ordering::Acquire) || sink.is_cancelled(),
                !page_semantics_complete
                    && *latency_phase == TranslationLatencyPhase::AwaitingFirstVisibleRegion
                    && pending.first().is_some_and(|region| region.visible),
            ) {
                TranslationBoundaryAction::ContinueUpstream => {
                    // This is a CUDA scheduling boundary, not a timer owner.
                    // Let the caller submit available detector/OCR work and
                    // reconsider the tail instead of idling CUDA here.
                    return Ok(());
                }
                TranslationBoundaryAction::Dispatch(count) => count,
                TranslationBoundaryAction::Cancelled => {
                    return Err(CleaningError::cancelled());
                }
            };
            if !(1..=TRANSLATION_BATCH_MAX).contains(&count) {
                return Err(CleaningError::new(
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
                .chapter_sessions
                .lock()
                .map_err(|_| {
                    CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
                })?
                .session(&request.page_session_id)
                .map(|session| {
                    session.following_source(
                        request.page_index,
                        last_reading_order,
                        MAX_HSK_PRECEDING_UTTERANCES,
                    )
                })
                .unwrap_or_default();
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
            let batch_contains_visible = batch.iter().any(|region| region.visible);
            let was_awaiting_first_visible =
                *latency_phase == TranslationLatencyPhase::AwaitingFirstVisibleRegion;
            let primary_published_visible = self
                .translate_and_publish(
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
                    repair_queue,
                    &following_english,
                )
                .await?;
            let repair_published_visible = if was_awaiting_first_visible && batch_contains_visible {
                self.process_queued_repairs(
                    resident,
                    control,
                    request,
                    repair_queue,
                    cancel.clone(),
                    sink,
                    image_width,
                    image_height,
                    overall_progress,
                    true,
                )
                .await?
            } else {
                false
            };
            // Final-only rendering may withhold a rejected or malformed
            // primary. Interactive admission remains reserved until either
            // the primary or its terminal repair has actually published a
            // visible final region; merely attempting generation is not a
            // user-visible milestone.
            complete_translation_batch(
                latency_phase,
                primary_published_visible || repair_published_visible,
            );
        }
        Ok(())
    }

    fn remember_terminal_region_names(
        &self,
        request: &BrowserJobRequest,
        region: &PreparedRegion,
    ) -> std::result::Result<(), CleaningError> {
        if request.settings.name_translation != NameTranslation::KeepOriginal {
            return Ok(());
        }
        let mut persistable = region.proper_names.iter().cloned().collect::<Vec<_>>();
        for entity in &region.entities {
            if !matches!(
                entity.entity_type,
                RegionEntityType::Person
                    | RegionEntityType::Place
                    | RegionEntityType::Organization
                    | RegionEntityType::Coined
            ) {
                continue;
            }
            let Some(chinese) = entity.translated.as_ref() else {
                continue;
            };
            let candidate = HskProtectedName {
                source_english: entity.source.clone(),
                chinese: chinese.clone(),
            };
            if !persistable.iter().any(|name| {
                name.source_english
                    .eq_ignore_ascii_case(&candidate.source_english)
            }) {
                persistable.push(candidate);
            }
        }
        if persistable.is_empty() {
            return Ok(());
        }
        let mut sessions = self.chapter_sessions.lock().map_err(|_| {
            CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
        })?;
        let session = sessions.session_mut(&request.page_session_id);
        for name in persistable {
            let source_english = name.source_english;
            let chinese = name.chinese;
            let entity_type = region
                .entities
                .iter()
                .find(|entity| entity.source.eq_ignore_ascii_case(&source_english))
                .map(|entity| Self::chapter_entity_type(entity.entity_type))
                .unwrap_or(ChapterEntityType::Unknown);
            session.remember_entity(ChapterEntity {
                source_english,
                entity_type,
                chinese: Some(chinese),
                first_page: request.page_index,
                first_reading_order: region.reading_order,
                pages: [request.page_index].into_iter().collect(),
            });
        }
        Ok(())
    }

    fn chapter_entity_type(entity_type: RegionEntityType) -> ChapterEntityType {
        match entity_type {
            RegionEntityType::Person => ChapterEntityType::Person,
            RegionEntityType::Place => ChapterEntityType::Place,
            RegionEntityType::Organization => ChapterEntityType::Organization,
            RegionEntityType::Coined => ChapterEntityType::CoinedEntity,
            RegionEntityType::Relationship => ChapterEntityType::Relationship,
            RegionEntityType::Occupation => ChapterEntityType::Occupation,
            RegionEntityType::Rank => ChapterEntityType::Rank,
            RegionEntityType::Title => ChapterEntityType::Title,
        }
    }

    fn remember_terminal_dialogue(
        &self,
        request: &BrowserJobRequest,
        region: &PreparedRegion,
        chinese: &str,
    ) -> std::result::Result<(), CleaningError> {
        if region.source_english.trim().is_empty() || chinese.trim().is_empty() {
            return Ok(());
        }
        let mut sessions = self.chapter_sessions.lock().map_err(|_| {
            CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
        })?;
        sessions
            .session_mut(&request.page_session_id)
            .record_dialogue(
                request.page_index,
                vec![DialogueNode {
                    page_index: request.page_index,
                    reading_order: region.reading_order,
                    region_id: region.id.clone(),
                    source_english: region.source_english.clone(),
                    chinese: chinese.to_owned(),
                    // Continuation links are produced by the chapter-level
                    // adjudicator and carried all the way to the terminal
                    // dialogue graph.  Never infer them from job completion
                    // order: the graph is the canonical source of context.
                    continuation_group: region.continuation_group.clone(),
                }],
            );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn translate_and_publish(
        &self,
        resident: &ResidentState,
        control: &HskControl,
        request: &BrowserJobRequest,
        regions: Vec<PreparedRegion>,
        cancel: Arc<AtomicBool>,
        sink: &JobUpdateSink,
        overall_progress: f32,
        image_width: u32,
        image_height: u32,
        context: &mut Vec<HskPrecedingUtterance>,
        repair_queue: &mut RepairQueue,
        following_english: &[String],
    ) -> std::result::Result<bool, CleaningError> {
        if regions.is_empty() {
            return Ok(false);
        }
        if regions.len() > TRANSLATION_BATCH_MAX {
            return Err(CleaningError::new(
                "TRANSLATION_BATCH_FAILED",
                "Page-wide translation is disabled; microbatches are limited to six regions.",
            ));
        }
        cancellation_boundary(cancel.as_ref())?;
        self.wait_for_preceding_page_analysis(request, cancel.as_ref(), sink)
            .await?;
        self.wait_for_preceding_page_language(request, cancel.as_ref(), sink)
            .await?;
        publish_progress(
            sink,
            BrowserJobStage::Translating,
            None,
            Some(overall_progress),
            None,
            None,
            "Translating English directly into HSK-targeted Chinese",
        )?;
        // Chapter context is owned by the daemon session. The language barrier
        // above guarantees that all earlier pages have reached a terminal
        // state, while the graph still filters to the exact preceding reading
        // position for connected bubbles and overlapping windows.
        cancellation_boundary(cancel.as_ref())?;
        let first_reading_order = regions
            .iter()
            .map(|region| region.reading_order)
            .min()
            .unwrap_or(u32::MAX);
        *context = self
            .chapter_sessions
            .lock()
            .map_err(|_| {
                CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
            })?
            .before_position(
                &request.page_session_id,
                request.page_index,
                first_reading_order,
            );
        let translator = resident.app.llm.direct_hsk_translator();
        let batch_context = context.clone();
        // Entity memory and the multimodal page adjudicator are the sole
        // sources of protected names. Browser-provided glossaries cannot
        // override chapter evidence or leak completion-order state.
        let mut all_protected_names = Vec::new();
        if request.settings.name_translation == NameTranslation::KeepOriginal {
            let remembered = self
                .chapter_sessions
                .lock()
                .map_err(|_| {
                    CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
                })?
                .session(&request.page_session_id)
                .map(|session| {
                    session
                        .entities_before_position(request.page_index, first_reading_order)
                        .filter_map(|entity| {
                            entity.chinese.as_ref().map(|chinese| HskProtectedName {
                                source_english: entity.source_english.clone(),
                                chinese: chinese.clone(),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            merge_protected_names(&mut all_protected_names, remembered);
        }
        if request.settings.name_translation == NameTranslation::KeepOriginal {
            for region in &regions {
                merge_protected_names(&mut all_protected_names, region.proper_names.clone());
            }
        }
        let protected_names =
            relevant_protected_names(&regions, &batch_context, &all_protected_names);
        if rejected_ocr_tracing_enabled() {
            eprintln!(
                "hskify-translation-batch-names sources={:?} names={:?}",
                regions
                    .iter()
                    .map(|region| region.source_english.as_str())
                    .collect::<Vec<_>>(),
                protected_names
                    .iter()
                    .map(|name| name.source_english.as_str())
                    .collect::<Vec<_>>(),
            );
        }
        let cuda_priority = prepared_region_priority(&regions, sink, image_width, image_height);
        cancellation_boundary(cancel.as_ref())?;
        let validator_names = control_proper_names(&protected_names);
        let name_handling = hsk_name_handling(request.settings.name_translation);
        let level = u8::from(request.settings.hsk_level);
        let control_level = ControlHskLevel::new(level)
            .map_err(|error| CleaningError::new("INVALID_HSK_LEVEL", error.to_string()))?;
        let mut keys = Vec::with_capacity(regions.len());
        let mut translated = vec![None::<CachedTranslation>; regions.len()];
        let mut missing_indices = Vec::new();
        let mut published_visible = false;
        let mut states = std::iter::repeat_with(|| None::<TranslationState>)
            .take(regions.len())
            .collect::<Vec<_>>();
        let model_id = translator.model_id().to_string();

        {
            let mut cache = self.translation_cache.lock().map_err(|_| {
                CleaningError::new("CACHE_FAILED", "Translation cache lock poisoned.")
            })?;
            for index in 0..regions.len() {
                let key = translation_cache_key(
                    &regions[index].source_english,
                    hsk_utterance_kind_for_region(&regions[index]),
                    &batch_context,
                    following_english,
                    &protected_names,
                    request.settings.name_translation,
                    request.settings.learning_mode,
                    level,
                    &model_id,
                    translator.model_revision(),
                    translator.prompt_hash(),
                    translator.validator_hash(),
                    control.cache_revision(),
                );
                let cached = cache.get(&key);
                if cached
                    .as_ref()
                    .is_some_and(|translation| translation.repair_state == HskRepairState::Pending)
                {
                    states[index] = cached.clone().map(|translation| {
                        TranslationState::from_cached(translation, request.settings.learning_mode)
                    });
                }
                translated[index] = cached;
                if translated[index].is_none() {
                    missing_indices.push(index);
                }
                keys.push(key);
            }
        }
        let generation_indices = primary_generation_indices(&translated, &states);

        for (index, translation) in translated.iter().enumerate() {
            let Some(translation) = translation.clone() else {
                continue;
            };
            if !translation_is_final(&translation) {
                continue;
            }
            // Cached entries have already passed the terminal validation and
            // cleanup gates. They are safe to publish immediately; generated
            // entries below are deliberately withheld until their validation
            // and repair state is terminal.
            let region = &regions[index];
            let cleanup = region.cleanup.result().await;
            let Some(decision) = cleanup.decisions.get(&region.id) else {
                publish_unreadable_prepared(
                    sink,
                    region,
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
                    region,
                    request,
                    image_width,
                    image_height,
                    decision.reason.as_deref().unwrap_or(
                        "Cleanup verification did not pass; source pixels were preserved.",
                    ),
                )?;
                continue;
            }
            let displayed_chinese = translation.displayed_chinese.clone();
            publish_region(
                sink,
                region,
                decision,
                translation,
                request.settings.hsk_level,
                request.settings.learning_mode,
                control,
                image_width,
                image_height,
            )?;
            published_visible |= regions[index].visible;
            self.remember_terminal_dialogue(request, &regions[index], &displayed_chinese)?;
            append_terminal_context(context, &regions[index].source_english, &displayed_chinese);
            self.remember_terminal_region_names(request, &regions[index])?;
        }

        if !generation_indices.is_empty() {
            let utterances = generation_indices
                .iter()
                .map(|index| {
                    let (max_characters, max_lines) =
                        layout_budget_for_region(&regions[*index], image_width, image_height);
                    HskSourceUtterance {
                        id: regions[*index].id.clone(),
                        kind: hsk_utterance_kind_for_region(&regions[*index]),
                        source_english: regions[*index].source_english.clone(),
                        max_characters,
                        max_lines,
                    }
                })
                .collect::<Vec<_>>();
            let index_by_id = generation_indices
                .iter()
                .map(|index| (regions[*index].id.clone(), *index))
                .collect::<HashMap<_, _>>();
            cancellation_boundary(cancel.as_ref())?;
            let cuda_permit = self
                .cuda_scheduler
                .acquire(CudaWorkload::Language, cuda_priority, cancel.clone())
                .await
                .map_err(cuda_admission_error)?;
            let mut publish_streamed = |outcome: &HskTranslationOutcome| -> Result<()> {
                cancellation_boundary(cancel.as_ref()).map_err(anyhow::Error::new)?;
                let index = *index_by_id
                    .get(&outcome.id)
                    .with_context(|| format!("unknown streamed translation id {}", outcome.id))?;
                if translated[index].is_some() || states[index].is_some() {
                    return Ok(());
                }
                let state = TranslationState::from_initial(
                    outcome.clone(),
                    control,
                    control_level,
                    &validator_names,
                    request.settings.learning_mode,
                );
                if let Some(translation) = state.initial_translation() {
                    // Streaming is an internal generation mechanism only. Do
                    // not expose this candidate to the browser: a later HSK
                    // repair must never visibly revise text already shown.
                    self.translation_cache
                        .lock()
                        .map_err(|_| anyhow!("translation cache lock poisoned"))?
                        .insert(keys[index].clone(), translation.clone());
                    translated[index] = Some(translation);
                }
                states[index] = Some(state);
                Ok(())
            };
            let initial = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(translator.translate_batch_streaming(
                    &HskTranslationBatchRequest {
                        requested_level: level,
                        learning_mode: hsk_learning_mode(request.settings.learning_mode),
                        name_handling,
                        translate_sound_effects: request.settings.translate_sound_effects,
                        utterances,
                        preceding_utterances: batch_context.clone(),
                        following_english: following_english.to_vec(),
                        protected_names: protected_names.clone(),
                    },
                    cancel.as_ref(),
                    &mut publish_streamed,
                ))
            })
            .context("run direct HSK translation batch")
            .map_err(CleaningError::pipeline)?;
            drop(publish_streamed);
            drop(cuda_permit);
            cancellation_boundary(cancel.as_ref())?;

            for outcome in initial.items {
                let Some(&index) = index_by_id.get(&outcome.id) else {
                    continue;
                };
                if translated[index].is_none() && states[index].is_none() {
                    states[index] = Some(TranslationState::from_initial(
                        outcome,
                        control,
                        control_level,
                        &validator_names,
                        request.settings.learning_mode,
                    ));
                }
            }
            for &index in &missing_indices {
                if states[index].is_none() {
                    states[index] = Some(TranslationState::from_initial(
                        missing_translation_outcome(&regions[index].id),
                        control,
                        control_level,
                        &validator_names,
                        request.settings.learning_mode,
                    ));
                }
            }
        }

        for &index in &missing_indices {
            if translated[index].is_some() {
                continue;
            }
            let Some(mut primary) = states[index]
                .as_ref()
                .and_then(TranslationState::initial_translation)
            else {
                continue;
            };
            populate_pinyin(control, &mut primary);
            self.translation_cache
                .lock()
                .map_err(|_| {
                    CleaningError::new("CACHE_FAILED", "Translation cache lock poisoned.")
                })?
                .insert(keys[index].clone(), primary.clone());
            translated[index] = Some(primary);
        }

        cancellation_boundary(cancel.as_ref())?;
        for (index, region) in regions.into_iter().enumerate() {
            let Some(state) = states[index].take() else {
                continue;
            };
            if state.problems.is_empty() {
                // This is the sole publication point for newly generated
                // translations. Everything that reaches the browser has
                // passed the complete validation path above, so no later
                // repair can replace visible text.
                let mut result = state.finish().map_err(CleaningError::pipeline)?;
                populate_pinyin(control, &mut result);
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
                    result.clone(),
                    request.settings.hsk_level,
                    request.settings.learning_mode,
                    control,
                    image_width,
                    image_height,
                )?;
                published_visible |= region.visible;
                self.remember_terminal_dialogue(request, &region, &result.displayed_chinese)?;
                append_terminal_context(context, &region.source_english, &result.displayed_chinese);
                self.remember_terminal_region_names(request, &region)?;
                self.translation_cache
                    .lock()
                    .map_err(|_| {
                        CleaningError::new("CACHE_FAILED", "Translation cache lock poisoned.")
                    })?
                    .insert(keys[index].clone(), result);
                continue;
            }
            let mut problems = state.problems.clone();
            problems.push(
                "pre-translation semantic analysis already classified this as story content; return a complete Simplified Chinese translation"
                    .to_owned(),
            );
            if rejected_ocr_tracing_enabled() {
                eprintln!(
                    "hskify-repair-queued source={:?} kind={:?} detector_core={} problems={:?}",
                    region.source_english,
                    region.candidate.kind,
                    region.candidate.has_detector_core,
                    problems,
                );
            }
            let (max_characters, max_lines) =
                layout_budget_for_region(&region, image_width, image_height);
            let utterance = HskRepairUtterance {
                id: region.id.clone(),
                kind: hsk_utterance_kind_for_region(&region),
                source_english: region.source_english.clone(),
                max_characters,
                max_lines,
                rejected_chinese: state.base_chinese.clone(),
                avoid_chinese: state.avoid_chinese(),
                problems,
            };
            repair_queue.enqueue(PendingRepair {
                cache_key: keys[index].clone(),
                region,
                utterance,
                protected_names: protected_names.clone(),
                state,
                attempts: 0,
            });
        }
        Ok(published_visible)
    }

    #[allow(clippy::too_many_arguments)]
    async fn process_queued_repairs(
        &self,
        resident: &ResidentState,
        control: &HskControl,
        request: &BrowserJobRequest,
        repair_queue: &mut RepairQueue,
        cancel: Arc<AtomicBool>,
        sink: &JobUpdateSink,
        image_width: u32,
        image_height: u32,
        overall_progress: f32,
        visible_only: bool,
    ) -> std::result::Result<bool, CleaningError> {
        if repair_queue.is_empty() {
            return Ok(false);
        }
        cancellation_boundary(cancel.as_ref())?;
        if !visible_only {
            publish_progress(
                sink,
                BrowserJobStage::HskValidating,
                None,
                Some(overall_progress),
                None,
                None,
                "Finishing this chapter's translations",
            )?;
        }

        let translator = resident.app.llm.direct_hsk_translator();
        let name_handling = hsk_name_handling(request.settings.name_translation);
        let level = u8::from(request.settings.hsk_level);
        let control_level = ControlHskLevel::new(level)
            .map_err(|error| CleaningError::new("INVALID_HSK_LEVEL", error.to_string()))?;
        let mut published_visible = false;

        while !repair_queue.is_empty() {
            cancellation_boundary(cancel.as_ref())?;
            if sink.is_cancelled() {
                return Err(CleaningError::cancelled());
            }
            let mut jobs = if visible_only {
                repair_queue.take_visible_batch(TRANSLATION_BATCH_MAX)
            } else {
                repair_queue.take_batch(TRANSLATION_BATCH_MAX)
            };
            if jobs.is_empty() {
                break;
            }
            let active_indices = jobs
                .iter()
                .enumerate()
                .filter_map(|(index, job)| (!job.state.repair_succeeded()).then_some(index))
                .collect::<Vec<_>>();
            // Repairs use the same daemon-owned, canonical chapter context as
            // primary generation.  Never send an empty context merely because
            // this stage runs after the primary batch: that would make a
            // continuation lose its speaker/subject and would reintroduce the
            // completion-order bug the chapter graph exists to prevent.
            let repair_context = if let Some(reading_order) =
                jobs.iter().map(|job| job.region.reading_order).min()
            {
                let sessions = self.chapter_sessions.lock().map_err(|_| {
                    CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
                })?;
                sessions.before_position(
                    &request.page_session_id,
                    request.page_index,
                    reading_order,
                )
            } else {
                Vec::new()
            };
            let repair_following = if let Some(reading_order) =
                jobs.iter().map(|job| job.region.reading_order).max()
            {
                self.chapter_sessions
                    .lock()
                    .map_err(|_| {
                        CleaningError::new(
                            "CHAPTER_SESSION_FAILED",
                            "Chapter session lock poisoned.",
                        )
                    })?
                    .session(&request.page_session_id)
                    .map(|session| {
                        session.following_source(
                            request.page_index,
                            reading_order,
                            MAX_HSK_PRECEDING_UTTERANCES,
                        )
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let mut batch_names = Vec::<HskProtectedName>::new();
            let mut utterances = Vec::<HskRepairUtterance>::with_capacity(active_indices.len());
            for &index in &active_indices {
                let job = &jobs[index];
                merge_protected_names(&mut batch_names, job.protected_names.clone());
                utterances.push(job.utterance.clone());
            }
            let mut retry_indices = HashSet::new();
            // Repairs are bounded transactions. The browser never receives a
            // rejected draft; a failed transaction is requeued with its exact
            // validator evidence until the small retry budget is exhausted.
            let cuda_permit = self
                .cuda_scheduler
                .acquire(
                    CudaWorkload::Language,
                    if visible_only {
                        CudaPriority::Visible
                    } else {
                        CudaPriority::Offscreen
                    },
                    cancel.clone(),
                )
                .await
                .map_err(cuda_admission_error)?;
            cancellation_boundary(cancel.as_ref())?;
            let repair_result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(translator.repair_invalid_batch(
                    &HskTranslationRepairBatchRequest {
                        requested_level: level,
                        learning_mode: hsk_learning_mode(request.settings.learning_mode),
                        name_handling,
                        // The browser contract keeps decorative sound effects
                        // as source artwork.  Repairs are still story regions,
                        // but they must inherit the same policy instead of
                        // silently enabling a second translation mode.
                        translate_sound_effects: request.settings.translate_sound_effects,
                        utterances,
                        preceding_utterances: repair_context.clone(),
                        following_english: repair_following.clone(),
                        protected_names: batch_names,
                    },
                    cancel.as_ref(),
                ))
            });
            drop(cuda_permit);
            match repair_result {
                Ok(repaired) => {
                    let mut by_id = repaired
                        .items
                        .into_iter()
                        .map(|outcome| (outcome.id.clone(), outcome))
                        .collect::<HashMap<_, _>>();
                    for index in active_indices {
                        let job = &mut jobs[index];
                        let outcome = by_id
                            .remove(&job.region.id)
                            .unwrap_or_else(|| missing_translation_outcome(&job.region.id));
                        let accepted = job.state.apply_repair(
                            outcome,
                            control,
                            control_level,
                            &control_proper_names(&job.protected_names),
                        );
                        if should_retry_repair(job, accepted) {
                            prepare_repair_retry(job);
                            retry_indices.insert(index);
                        }
                    }
                }
                Err(_) if cancel.load(Ordering::Acquire) || sink.is_cancelled() => {
                    return Err(CleaningError::cancelled());
                }
                Err(error) => {
                    eprintln!(
                        "hskify: batched HSK repair failed for {} regions: {error:#}",
                        active_indices.len(),
                    );
                    for index in active_indices {
                        jobs[index].state.reject_failed_repair();
                        if should_retry_repair(&jobs[index], false) {
                            prepare_repair_retry(&mut jobs[index]);
                            retry_indices.insert(index);
                        }
                    }
                }
            }
            cancellation_boundary(cancel.as_ref())?;
            for (index, job) in jobs.into_iter().enumerate() {
                if retry_indices.contains(&index) {
                    repair_queue.requeue(job);
                    continue;
                }
                let rejected_chinese = job.state.base_chinese.clone();
                let rejection_problems = job.state.problems.clone();
                let mut result = match job.state.finish() {
                    Ok(result) => result,
                    Err(error) => {
                        eprintln!(
                            "hskify: preserving original pixels for terminally unpublishable story OCR {:?}: {error:#}; rejected={:?}; problems={:?}",
                            job.region.source_english, rejected_chinese, rejection_problems,
                        );
                        continue;
                    }
                };
                populate_pinyin(control, &mut result);
                cancellation_boundary(cancel.as_ref())?;

                let cleanup = job.region.cleanup.result().await;
                let Some(decision) = cleanup.decisions.get(&job.region.id) else {
                    publish_unreadable_prepared(
                        sink,
                        &job.region,
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
                        &job.region,
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
                    &job.region,
                    decision,
                    result.clone(),
                    request.settings.hsk_level,
                    request.settings.learning_mode,
                    control,
                    image_width,
                    image_height,
                )?;
                published_visible |= job.region.visible;
                self.remember_terminal_dialogue(request, &job.region, &result.displayed_chinese)?;
                self.remember_terminal_region_names(request, &job.region)?;

                self.translation_cache
                    .lock()
                    .map_err(|_| {
                        CleaningError::new("CACHE_FAILED", "Translation cache lock poisoned.")
                    })?
                    .insert(job.cache_key, result);
            }
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
impl CleaningPipeline for KoharuPipeline {
    async fn warm_up(&self) -> std::result::Result<(), CleaningError> {
        self.ready_models()
            .await
            .map(|_| ())
            .map_err(CleaningError::pipeline)
    }

    async fn run(
        &self,
        input: CleaningInput,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), CleaningError> {
        let request = input.request.clone();
        let result = self.run_direct(input, cancel, sink).await;
        if result.is_err() {
            // A failed/cancelled page is terminal for chapter ordering too;
            // later admitted pages must not wait forever for analysis or
            // language context that can never be produced. Successful pages
            // keep the richer complete analysis written by run_direct.
            let _ = self.mark_page_terminal(&request);
        }
        result
    }

    fn mark_page_terminal(
        &self,
        request: &BrowserJobRequest,
    ) -> std::result::Result<(), CleaningError> {
        let kind = page_surface_kind(
            request.surface_kind,
            request.natural_width,
            request.natural_height,
        );
        self.record_page_analysis(
            request,
            request.natural_width,
            request.natural_height,
            kind,
            &[],
            true,
        )?;
        self.chapter_sessions
            .lock()
            .map_err(|_| {
                CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
            })?
            .session_mut(&request.page_session_id)
            .mark_language_complete(request.page_index);
        self.chapter_progress_notify.notify_waiters();
        Ok(())
    }

    fn restore_cached_context(
        &self,
        request: &BrowserJobRequest,
        regions: &[TranslatedRegion],
        preserved_artwork: &[PreservedArtworkRegion],
        unreadable_regions: &[crate::contracts::UnreadableRegion],
    ) -> std::result::Result<(), CleaningError> {
        let mut sessions = self.chapter_sessions.lock().map_err(|_| {
            CleaningError::new("CHAPTER_SESSION_FAILED", "Chapter session lock poisoned.")
        })?;
        let session = sessions.session_mut(&request.page_session_id);
        session.register_expected_pages(&request.chapter_page_order);
        session.register_surface(PageSurface {
            session_id: request.page_session_id.clone(),
            page_index: request.page_index,
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
                region.id.clone(),
                RegionPlan {
                    id: region.id.clone(),
                    reading_order: region.reading_order,
                    role: match region.role {
                        Some(TranslatedRegionRole::Dialogue) => RegionRole::Dialogue,
                        Some(TranslatedRegionRole::Narration) => RegionRole::Narration,
                        Some(TranslatedRegionRole::System) => RegionRole::System,
                        None => RegionRole::Unknown,
                    },
                    source_english: region.source_english.clone(),
                    continuation_group: region.context_group.clone(),
                },
            );
            session.record_dialogue(
                request.page_index,
                vec![DialogueNode {
                    page_index: request.page_index,
                    reading_order: region.reading_order,
                    region_id: region.id.clone(),
                    source_english: region.source_english.clone(),
                    chinese: region.displayed_chinese.clone(),
                    continuation_group: region.context_group.clone(),
                }],
            );
            for entity in &region.entities {
                let Some(chinese) = entity.translated.as_ref() else {
                    continue;
                };
                if !matches!(
                    entity.entity_type,
                    RegionEntityType::Person
                        | RegionEntityType::Place
                        | RegionEntityType::Organization
                        | RegionEntityType::Coined
                ) {
                    continue;
                }
                session.remember_entity(ChapterEntity {
                    source_english: entity.source.clone(),
                    entity_type: Self::chapter_entity_type(entity.entity_type),
                    chinese: Some(chinese.clone()),
                    first_page: request.page_index,
                    first_reading_order: region.reading_order,
                    pages: [request.page_index].into_iter().collect(),
                });
            }
        }
        for region in preserved_artwork {
            plans.insert(
                region.id.clone(),
                RegionPlan {
                    id: region.id.clone(),
                    reading_order: region.reading_order,
                    role: RegionRole::TechniqueArtwork,
                    source_english: region.source_english.clone(),
                    continuation_group: None,
                },
            );
        }
        for region in unreadable_regions {
            plans.insert(
                region.id.clone(),
                RegionPlan {
                    id: region.id.clone(),
                    reading_order: region.reading_order,
                    role: RegionRole::Unreadable,
                    source_english: region.source_english.clone(),
                    continuation_group: None,
                },
            );
        }
        let surface = session
            .surfaces
            .get(&request.page_index)
            .cloned()
            .expect("cached page surface was registered above");
        session.record_analysis(PageAnalysis {
            surface,
            regions: plans.into_values().collect(),
            complete: true,
        });
        session.mark_language_complete(request.page_index);
        drop(sessions);
        self.chapter_progress_notify.notify_waiters();
        Ok(())
    }

    fn close_chapter(&self, page_session_id: &str) {
        if let Ok(mut sessions) = self.chapter_sessions.lock() {
            sessions.remove(page_session_id);
        }
        self.chapter_progress_notify.notify_waiters();
    }

    async fn lookup(
        &self,
        input: LookupInput,
        region: Option<RegionLookupContext>,
    ) -> std::result::Result<LookupResult, CleaningError> {
        let control = self
            .hsk_control()
            .await
            .map_err(|error| CleaningError::new("RESOURCES_NOT_READY", format!("{error:#}")))?;
        let (proper_names, context) = match region {
            Some(region) => (
                region.proper_names,
                Some(ControlLookupRegion {
                    displayed_chinese: region.displayed_chinese,
                    base_chinese: region.base_chinese,
                    source_english: region.source_english,
                }),
            ),
            None => (Vec::new(), None),
        };
        let result = match input {
            LookupInput::Selection(selected_text) => {
                control.lookup_with_region_context(&selected_text, &proper_names, context)
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
                    .lookup_at_with_region_context(
                        &displayed_text,
                        character_offset,
                        &proper_names,
                        context.clone(),
                    )
                    .unwrap_or(hsk_control::LookupResult {
                        selected_text: hovered_character,
                        tokens: Vec::new(),
                        region: context,
                    })
            }
        };
        Ok(browser_lookup_result(result))
    }

    fn resources_ready(&self) -> bool {
        self.resident.get().is_some() && self.hsk_control.get().is_some()
    }
}

struct ResidentState {
    app: Arc<App>,
    detector: Mutex<ComicTextBubbleDetector>,
    ocr_detector: Mutex<PpOcrSmallDetector>,
    ocr: Mutex<PpOcrSmallRecognizer>,
    text_segmenter: Mutex<MangaTextSegmentation>,
    bubble_segmenter: Mutex<SpeechBubbleSegmentation>,
    inpainter: Mutex<Lama>,
    page_understanding: Mutex<QwenPageUnderstanding>,
}

impl ResidentState {
    async fn load(
        runtime_root: PathBuf,
        app_state_root: PathBuf,
        resources: ResidentResourcePaths,
    ) -> Result<Self> {
        koharu_runtime::require_hskify_cuda_target()
            .context("resident model load requires the exact Hskify CUDA target")?;

        let runtime = Arc::new(
            RuntimeManager::new(&runtime_root, ComputePolicy::CudaRequired)
                .context("initialize resident model runtime")?,
        );
        runtime
            .prepare()
            .await
            .context("prepare resident model runtime")?;
        let mut config = AppConfig::default();
        config.data.path = utf8_path(app_state_root)?;
        let app = Arc::new(
            App::new(config, runtime.clone(), false, env!("CARGO_PKG_VERSION"))
                .context("initialize resident application model state")?,
        );
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
        let llm_future = app.llm.load_local_file_with_threads(
            HSK_TRANSLATION_MODEL,
            translation_model.clone(),
            BROWSER_QWEN_INFERENCE_THREADS,
        );
        let page_model_runtime = runtime.clone();
        let page_model_backend = app.llm.backend();
        let page_capability_for_load = page_capability.clone();
        let page_app = app.clone();
        let page_future = async move {
            llm_future.await?;
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
        Ok(Self {
            app,
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
        self.detector
            .lock()
            .map_err(|_| anyhow!("detector lock poisoned during inference warm-up"))?
            .inference_tiles(std::slice::from_ref(&sample))
            .context("prime comic text detector inference")?;
        self.ocr_detector
            .lock()
            .map_err(|_| anyhow!("OCR detector lock poisoned during inference warm-up"))?
            .detect_tiles(std::slice::from_ref(&sample))
            .context("prime PP-OCR small text detector inference")?;
        let segmentation_sample = DynamicImage::new_rgb8(2_048, 2_048);
        self.text_segmenter
            .lock()
            .map_err(|_| anyhow!("text segmenter lock poisoned during inference warm-up"))?
            .inference_batch(std::slice::from_ref(&segmentation_sample))
            .context("prime manga text segmentation inference")?;

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

    fn prime_language_inference(&self) -> Result<()> {
        // Exercise the resident multimodal and translation execution paths
        // before setup reports ready. Warm-up deliberately uses raw inference
        // rather than a fake page/translation response, so startup never
        // depends on a model obeying a production wire contract.
        let cancel = AtomicBool::new(false);
        let translator = self.app.llm.direct_hsk_translator();
        tokio::runtime::Handle::current().block_on(async {
            self.page_understanding
                .lock()
                .map_err(|_| anyhow!("page-understanding lock poisoned during warm-up"))?
                .warm_up()
                .context("prime multimodal page understanding")?;
            translator
                .warm_up(&cancel)
                .await
                .context("prime direct HSK translation inference")?;
            Ok::<(), anyhow::Error>(())
        })?;
        Ok(())
    }
}

fn utf8_path(path: PathBuf) -> Result<Utf8PathBuf> {
    Utf8PathBuf::from_path_buf(path).map_err(|path| anyhow!("path is not valid UTF-8: {path:?}"))
}

struct PreparedRegion {
    id: String,
    candidate: Candidate,
    source_english: String,
    ocr_confidence: f32,
    reading_order: u32,
    /// Canonical chapter graph link assigned by page understanding.  This is
    /// The page model supplies this link; deterministic geometry never
    /// invents continuation groups from timing or completion order.
    continuation_group: Option<String>,
    entities: Vec<RegionEntitySpan>,
    role: TranslatedRegionRole,
    source_line_count: usize,
    prediction: PpOcrPrediction,
    /// Optional model-returned typography evidence. The visual OCR evidence
    /// remains the fallback when the page model cannot confidently identify a
    /// source style.
    style: Option<PageStyleEvidence>,
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
    /// Names discovered for this exact source region. They remain local to
    /// the translation transaction until a terminal publication succeeds;
    /// only terminal page decisions enter chapter entity memory.
    proper_names: Vec<HskProtectedName>,
    translation_queued_at: tokio::time::Instant,
}

struct CleanupBatchTask {
    receiver: AsyncMutex<Option<oneshot::Receiver<Arc<CleanupBatchResult>>>>,
    result: OnceCell<Arc<CleanupBatchResult>>,
    // Cleanup is speculative work that can overlap page understanding.  If
    // the semantic decision rejects the page (or the bounded wait expires),
    // abort the detached task instead of letting an orphaned inpaint job keep
    // occupying the CUDA scheduler after the page has already gone terminal.
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

fn verified_source_guided_ocr_lines(
    candidates: Vec<Candidate>,
    mut lines: Vec<RecognizedLine>,
) -> (Vec<RecognizedLine>, Vec<RecognizedLine>, Vec<Candidate>) {
    let mut accepted = Vec::new();
    let mut deferred = Vec::new();
    let mut disputed = Vec::new();
    for candidate in candidates {
        let Some(index) = lines.iter().position(|line| {
            line.candidate
                .text_rect
                .overlap_over_smaller(candidate.text_rect)
                >= 0.80
        }) else {
            disputed.push(candidate);
            continue;
        };
        let line = lines.swap_remove(index);
        if candidate.kind != CandidateKind::StoryText
            || line.prediction.confidence < self::ocr::BROWSER_OCR_MIN_CONFIDENCE
        {
            deferred.push(line);
            disputed.push(candidate);
            continue;
        }
        accepted.push(line);
    }
    (accepted, deferred, disputed)
}

fn merge_best_recognized_line(lines: &mut Vec<RecognizedLine>, candidate: RecognizedLine) {
    let Some(existing) = lines.iter_mut().find(|line| {
        line.candidate
            .text_rect
            .overlap_over_smaller(candidate.candidate.text_rect)
            >= 0.80
    }) else {
        lines.push(candidate);
        return;
    };
    if recognized_line_quality(&candidate) > recognized_line_quality(existing) {
        *existing = candidate;
    }
}

fn text_rects_represent_same_block(left: PixelRect, right: PixelRect) -> bool {
    left.iou(right) >= 0.35 || left.overlap_over_smaller(right) >= 0.60
}

fn recognized_line_quality(line: &RecognizedLine) -> (u32, u32) {
    // Overlapping detector tiles can produce two equivalent transcripts for
    // one line. Choose the calibrated model evidence first; never let a
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
    source_english: String,
    translated_chinese: Option<String>,
    ocr_confidence: f32,
    continuation_group: Option<String>,
    entities: Vec<RegionEntitySpan>,
    role: TranslatedRegionRole,
    source_line_count: usize,
    prediction: PpOcrPrediction,
    style: Option<PageStyleEvidence>,
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

struct PendingRepair {
    cache_key: String,
    region: PreparedRegion,
    utterance: HskRepairUtterance,
    protected_names: Vec<HskProtectedName>,
    state: TranslationState,
    attempts: u8,
}

fn should_retry_repair(job: &PendingRepair, accepted: bool) -> bool {
    !accepted && !job.state.can_publish() && job.attempts + 1 < MAX_HSK_REPAIR_ATTEMPTS
}

fn prepare_repair_retry(job: &mut PendingRepair) {
    job.attempts = job.attempts.saturating_add(1);
    let rejected = job
        .state
        .latest_rejected_chinese
        .clone()
        .or_else(|| job.state.base_chinese.clone());
    let mut problems = job.state.problems.clone();
    if problems.is_empty() {
        problems.push(
            "return a complete Simplified Chinese translation for this story region".to_owned(),
        );
    }
    for name in &job.protected_names {
        let chinese = name.chinese.trim();
        if chinese.is_empty()
            || rejected
                .as_deref()
                .is_some_and(|text| text.contains(chinese))
        {
            continue;
        }
        append_repair_problem(
            &mut problems,
            format!("copy protected name `{chinese}` exactly; do not omit it"),
        );
    }
    job.utterance.rejected_chinese = rejected;
    job.utterance.avoid_chinese = job.state.avoid_chinese();
    job.utterance.problems = problems;
}

fn append_repair_problem(problems: &mut Vec<String>, problem: String) {
    if !problems.iter().any(|existing| existing == &problem) {
        problems.push(problem);
    }
}

#[derive(Default)]
struct RepairQueue {
    jobs: VecDeque<PendingRepair>,
    region_ids: HashSet<String>,
    primary_phase_complete: bool,
}

impl RepairQueue {
    fn enqueue(&mut self, job: PendingRepair) {
        if self.region_ids.insert(job.region.id.clone()) {
            self.jobs.push_back(job);
        }
    }

    fn requeue(&mut self, job: PendingRepair) {
        debug_assert!(self.region_ids.contains(&job.region.id));
        self.jobs.push_back(job);
    }

    fn finish_primary_phase(&mut self) {
        self.primary_phase_complete = true;
    }

    fn take_batch(&mut self, maximum: usize) -> Vec<PendingRepair> {
        if !self.primary_phase_complete || maximum == 0 {
            return Vec::new();
        }
        let mut selected = Vec::with_capacity(maximum.min(self.jobs.len()));
        while selected.len() < maximum {
            match self.jobs.pop_front() {
                Some(job) => selected.push(job),
                None => break,
            }
        }
        selected
    }

    fn take_visible_batch(&mut self, maximum: usize) -> Vec<PendingRepair> {
        if self.primary_phase_complete || maximum == 0 {
            return Vec::new();
        }
        let mut selected = Vec::with_capacity(maximum.min(self.jobs.len()));
        let mut remaining = VecDeque::with_capacity(self.jobs.len());
        while let Some(job) = self.jobs.pop_front() {
            if selected.len() < maximum && job.region.visible {
                selected.push(job);
            } else {
                remaining.push_back(job);
            }
        }
        self.jobs = remaining;
        selected
    }

    fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
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
    resident: &ResidentState,
    source: Arc<DynamicImage>,
    candidates: &mut Vec<Candidate>,
    proposal_source: OcrProposalSource,
    request: &BrowserJobRequest,
    sink: &JobUpdateSink,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    preprocessing: &Arc<PreprocessingPool>,
    text_probabilities: &ProbabilityMap,
) -> std::result::Result<OcrBatchResult, CleaningError> {
    let (image_width, image_height) = source.dimensions();
    if candidates.is_empty() {
        return Ok(OcrBatchResult::default());
    }
    cancellation_boundary(cancel.as_ref())?;
    if sink.is_cancelled() {
        return Err(CleaningError::cancelled());
    }
    let viewport = sink.viewport();
    candidates.sort_by(|left, right| {
        let left_visible = viewport.active
            && left.bubble_rect.intersects_viewport(
                &viewport.visible_rects,
                image_width,
                image_height,
            );
        let right_visible = viewport.active
            && right.bubble_rect.intersects_viewport(
                &viewport.visible_rects,
                image_width,
                image_height,
            );
        right_visible.cmp(&left_visible).then_with(|| {
            reading_order_key(
                left.text_rect,
                image_width,
                image_height,
                request.settings.reading_direction,
            )
            .cmp(&reading_order_key(
                right.text_rect,
                image_width,
                image_height,
                request.settings.reading_direction,
            ))
        })
    });
    let count = OCR_REGION_BATCH_SIZE.min(candidates.len());
    let candidate_chunk = candidates.drain(..count).collect::<Vec<_>>();
    let admission_viewport = sink.viewport();
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
        .map_err(CleaningError::pipeline)?;
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
    let predictions = {
        let mut ocr = resident
            .ocr
            .lock()
            .map_err(|_| CleaningError::new("MODEL_STATE_FAILED", "OCR model lock poisoned."))?;
        ocr.recognize_regions_with_consensus(&crops, &crop_text_probabilities)
            .context("run batched CUDA PP-OCR small recognition with calibrated consensus")
            .map_err(CleaningError::pipeline)?
    };
    drop(cuda_permit);
    cancellation_boundary(cancel.as_ref())?;
    if predictions.len() != candidate_chunk.len() {
        return Err(CleaningError::new(
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

/// Run the page-level adjudication for a grouped page. The model receives the
/// immutable page surface and the exact OCR/layout evidence used by the
/// deterministic pipeline. Long continuous strips can contain more evidence
/// regions than one multimodal context window can safely carry, so the page is
/// partitioned into bounded, canonical windows. This is a context boundary,
/// not a second semantic pipeline: every window uses the same model contract
/// and the caller merges only the terminal, validated decisions.
async fn adjudicate_grouped_page(
    resident: &ResidentState,
    source: Arc<DynamicImage>,
    grouped: &[GroupedRegion],
    request: &BrowserJobRequest,
    preceding_context: &[HskPrecedingUtterance],
    priority: CudaPriority,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    image_width: u32,
    image_height: u32,
) -> std::result::Result<PageUnderstandingResult, CleaningError> {
    if grouped.len() <= koharu_llm::page_understanding::MAX_PAGE_REGIONS {
        let following = grouped
            .iter()
            .skip(1)
            .take(koharu_llm::page_understanding::MAX_PAGE_CONTEXT_LINES)
            .map(|group| group.source_english.clone())
            .collect::<Vec<_>>();
        return adjudicate_page_window(
            resident,
            source,
            grouped,
            request,
            preceding_context,
            &following,
            0,
            priority,
            cancel,
            cuda_scheduler,
            image_width,
            image_height,
        )
        .await;
    }

    let window_size = koharu_llm::page_understanding::MAX_PAGE_REGIONS;
    let overlap = PAGE_WINDOW_OVERLAP.min(window_size.saturating_sub(1));
    let stride = window_size.saturating_sub(overlap).max(1);
    let mut merged_regions = BTreeMap::<String, PageRegionDecision>::new();
    let mut merged_role = PageRole::Furniture;
    let mut offset = 0usize;
    while offset < grouped.len() {
        cancellation_boundary(cancel.as_ref())?;
        let end = (offset + window_size).min(grouped.len());
        let window = &grouped[offset..end];
        let following = grouped
            .iter()
            .skip(end)
            .take(koharu_llm::page_understanding::MAX_PAGE_CONTEXT_LINES)
            .map(|group| group.source_english.clone())
            .collect::<Vec<_>>();
        let result = adjudicate_page_window(
            resident,
            source.clone(),
            window,
            request,
            preceding_context,
            &following,
            offset,
            priority,
            cancel.clone(),
            cuda_scheduler,
            image_width,
            image_height,
        )
        .await?;
        // A page containing any story evidence is a story page. Unreadable is
        // retained only when every bounded window is unreadable; this prevents
        // one malformed tail from hiding valid dialogue earlier in a strip.
        merged_role = match (merged_role, result.page_role) {
            (PageRole::Story, _) | (_, PageRole::Story) => PageRole::Story,
            (PageRole::Unreadable, PageRole::Furniture)
            | (PageRole::Furniture, PageRole::Unreadable) => PageRole::Unreadable,
            (left, _) => left,
        };
        for decision in result.regions {
            merge_overlapping_page_decision(&mut merged_regions, decision);
        }
        if end == grouped.len() {
            break;
        }
        offset += stride;
    }
    Ok(PageUnderstandingResult {
        page_role: merged_role,
        regions: merged_regions.into_values().collect(),
    })
}

/// Overlap gives the page model enough shared topology to link a continuation
/// across a bounded multimodal window. The first (canonical) decision remains
/// authoritative for transcript/role/style; a later overlapping decision may
/// only add a continuation link or entity evidence that the first window did
/// not see. This makes the merge deterministic and prevents duplicate region
/// publication after the overlap is removed.
fn merge_overlapping_page_decision(
    decisions: &mut BTreeMap<String, PageRegionDecision>,
    incoming: PageRegionDecision,
) {
    let Some(existing) = decisions.get_mut(&incoming.id) else {
        decisions.insert(incoming.id.clone(), incoming);
        return;
    };
    if existing.continuation_of.is_none() {
        existing.continuation_of = incoming.continuation_of;
    }
    for entity in incoming.entity_spans {
        if !existing.entity_spans.contains(&entity) {
            existing.entity_spans.push(entity);
        }
    }
    if existing.style.is_none() {
        existing.style = incoming.style;
    }
    if existing.translated_chinese.is_none() {
        existing.translated_chinese = incoming.translated_chinese;
    }
}

/// Execute one bounded page-understanding request. `reading_order_offset` is
/// global to the page, so a continuation can never be made valid merely by a
/// chunk-local reindexing. Adjacent windows overlap; shared regions let the
/// page model express a continuation across the context boundary before the
/// caller merges duplicate terminal decisions by region identity.
#[allow(clippy::too_many_arguments)]
async fn adjudicate_page_window(
    resident: &ResidentState,
    source: Arc<DynamicImage>,
    grouped: &[GroupedRegion],
    request: &BrowserJobRequest,
    preceding_context: &[HskPrecedingUtterance],
    following_english: &[String],
    reading_order_offset: usize,
    priority: CudaPriority,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    image_width: u32,
    image_height: u32,
) -> std::result::Result<PageUnderstandingResult, CleaningError> {
    let (evidence_surface, evidence_viewport) =
        page_evidence_surface(&source, grouped, image_width, image_height);
    let evidence_width = evidence_surface.width();
    let evidence_height = evidence_surface.height();
    let evidence = PageUnderstandingRequest {
        image: evidence_surface,
        regions: grouped
            .iter()
            .enumerate()
            .map(|(local_reading_order, group)| PageRegionEvidence {
                id: stable_region_id(&request.source_sha256, group.candidate.text_rect),
                source_english: group.source_english.clone(),
                transcript_hypotheses: std::iter::once(group.source_english.clone())
                    .chain(
                        group
                            .prediction
                            .ocr_lines
                            .iter()
                            .map(|line| line.text.clone())
                            .filter(|text| text != &group.source_english),
                    )
                    .take(4)
                    .collect(),
                polygon: polygon_in_evidence_viewport(
                    group.candidate.text_rect,
                    evidence_viewport,
                    evidence_width,
                    evidence_height,
                )
                .into_iter()
                .map(|point| PagePoint {
                    x: point.x,
                    y: point.y,
                })
                .collect(),
                confidence: group.ocr_confidence.clamp(0.0, 1.0),
                reading_order: reading_order_offset + local_reading_order,
                bubble_id: Some(stable_region_id(
                    &request.source_sha256,
                    group.candidate.confirmed_bubble_rect,
                )),
                connected_region_ids: grouped
                    .iter()
                    .filter(|other| {
                        !std::ptr::eq(*other, group)
                            && group
                                .candidate
                                .confirmed_bubble_rect
                                .overlap_over_smaller(other.candidate.confirmed_bubble_rect)
                                >= 0.50
                    })
                    .map(|other| {
                        stable_region_id(&request.source_sha256, other.candidate.text_rect)
                    })
                    .collect(),
            })
            .collect(),
        preceding_chinese: preceding_context
            .iter()
            .map(|utterance| utterance.chinese.clone())
            .collect(),
        // The page adjudicator also sees the next untranslated lines in the
        // same canonical reading window.  This lets it join continuation
        // bubbles and disambiguate short pronouns without waiting for a
        // completion-racy later job.
        following_english: following_english.to_vec(),
    };
    cancellation_boundary(cancel.as_ref())?;
    let permit = cuda_scheduler
        .acquire(CudaWorkload::Vision, priority, cancel.clone())
        .await
        .map_err(cuda_admission_error)?;
    let result = {
        let mut model = resident.page_understanding.lock().map_err(|_| {
            CleaningError::new(
                "MODEL_STATE_FAILED",
                "Qwen3.5 page-understanding model lock poisoned.",
            )
        })?;
        model.analyze(&evidence)
    };
    drop(permit);
    result.map_err(|error| {
        CleaningError::pipeline(
            error
                .context("Qwen3.5 page understanding did not return a complete validated decision"),
        )
    })
}

/// Select a bounded visual evidence surface for one ordered language window.
///
/// OCR and bubble geometry are already independently accepted before this
/// function runs.  Their union is therefore the only source of the crop; no
/// title, page, or reader-specific crop is possible.  Small ordinary pages
/// continue to use the complete surface.  Tall strips and sparse windows use
/// an expanded local viewport so the multimodal projector receives larger,
/// more legible glyphs while retaining bubble borders and nearby artwork.
fn page_evidence_surface(
    source: &Arc<DynamicImage>,
    grouped: &[GroupedRegion],
    image_width: u32,
    image_height: u32,
) -> (Arc<DynamicImage>, PixelRect) {
    let full = PixelRect::new(0.0, 0.0, image_width as f32, image_height as f32)
        .expect("decoded page surface must have non-zero dimensions");
    let Some(first) = grouped.first() else {
        return (Arc::clone(source), full);
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
        return (Arc::clone(source), full);
    }
    let cropped = source.crop_imm(bounds.x, bounds.y, bounds.width, bounds.height);
    let crop_viewport = PixelRect::new(
        bounds.x as f32,
        bounds.y as f32,
        (bounds.x + bounds.width) as f32,
        (bounds.y + bounds.height) as f32,
    )
    .expect("non-empty crop bounds must form a valid viewport");
    (Arc::new(cropped), crop_viewport)
}

fn polygon_in_evidence_viewport(
    rect: PixelRect,
    viewport: PixelRect,
    image_width: u32,
    image_height: u32,
) -> Vec<Point> {
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
    local.polygon(image_width, image_height)
}

fn apply_page_adjudication_transcripts(
    grouped: &mut [GroupedRegion],
    request: &BrowserJobRequest,
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
        // The result parser has already checked non-empty transcripts and exact
        // IDs. Preserve that corrected transcript for translation while the
        // OCR geometry and cleanup mask remain tied to source pixels.
        group.source_english = decision.transcript.clone();
        group.prediction.text = decision.transcript.clone();
        group.translated_chinese = decision.translated_chinese.clone();
        // Keep continuation topology as typed model evidence.  It is later
        // committed to DialogueGraph in document order, so connected bubbles
        // remain connected even when page work finishes out of order.
        group.continuation_group = decision.continuation_of.clone();
        group.style = decision.style.clone();
        group.role = match decision.role {
            PageRegionRole::Sfx => TranslatedRegionRole::System,
            PageRegionRole::Story => group.role,
            PageRegionRole::Furniture | PageRegionRole::Artwork | PageRegionRole::Unreadable => {
                group.role
            }
        };
        group.entities = decision
            .entity_spans
            .iter()
            .filter_map(|span| {
                let (start_char, end_char) = source_char_span(&group.source_english, &span.source)?;
                Some(RegionEntitySpan {
                    source: span.source.clone(),
                    start_char,
                    end_char,
                    entity_type: page_entity_type(span.entity_type),
                    translated: (matches!(
                        span.entity_type,
                        PageEntityType::Person
                            | PageEntityType::Place
                            | PageEntityType::Organization
                            | PageEntityType::Event
                            | PageEntityType::CoinedEntity
                    ) && request.settings.name_translation
                        == NameTranslation::KeepOriginal)
                        .then(|| span.source.clone()),
                })
            })
            .collect();
        roles.insert(id, decision.role);
    }
    roles
}

fn source_char_span(source: &str, candidate: &str) -> Option<(usize, usize)> {
    let source_chars = source.chars().collect::<Vec<_>>();
    let candidate_chars = candidate.chars().collect::<Vec<_>>();
    if candidate_chars.is_empty() || candidate_chars.len() > source_chars.len() {
        return None;
    }
    source_chars
        .windows(candidate_chars.len())
        .position(|window| {
            window
                .iter()
                .zip(&candidate_chars)
                .all(|(left, right)| left.eq_ignore_ascii_case(right))
        })
        .map(|start| (start, start + candidate_chars.len()))
}

fn page_entity_type(entity_type: PageEntityType) -> RegionEntityType {
    match entity_type {
        PageEntityType::Person => RegionEntityType::Person,
        PageEntityType::Place => RegionEntityType::Place,
        PageEntityType::Organization => RegionEntityType::Organization,
        PageEntityType::Event | PageEntityType::CoinedEntity => RegionEntityType::Coined,
        PageEntityType::Relationship => RegionEntityType::Relationship,
        PageEntityType::Occupation => RegionEntityType::Occupation,
        PageEntityType::Rank => RegionEntityType::Rank,
        PageEntityType::Title => RegionEntityType::Title,
    }
}

fn protected_names_from_page_adjudication(
    grouped: &[GroupedRegion],
    request: &BrowserJobRequest,
    result: &PageUnderstandingResult,
) -> Vec<HskProtectedName> {
    let known_ids = grouped
        .iter()
        .map(|group| stable_region_id(&request.source_sha256, group.candidate.text_rect))
        .collect::<HashSet<_>>();
    let mut names = Vec::new();
    for decision in &result.regions {
        if !known_ids.contains(&decision.id) {
            continue;
        }
        // Entity spans are validated against the adjudicator's corrected
        // source-language transcript.  Using the stale OCR string here would
        // discard a genuine name whenever the model fixes a recognition typo.
        let source = decision.transcript.as_str();
        for span in &decision.entity_spans {
            if !matches!(
                span.entity_type,
                PageEntityType::Person
                    | PageEntityType::Place
                    | PageEntityType::Organization
                    | PageEntityType::Event
                    | PageEntityType::CoinedEntity
            ) {
                continue;
            }
            let candidate = span.source.trim();
            if source_contains_name_span(source, candidate)
                && !names.iter().any(|name: &HskProtectedName| {
                    name.source_english.eq_ignore_ascii_case(candidate)
                })
            {
                // Keep-original mode uses source spelling as the protected
                // output. Relationships, occupations, ranks, and titles are
                // intentionally excluded above and remain translatable.
                names.push(HskProtectedName {
                    source_english: candidate.to_owned(),
                    chinese: candidate.to_owned(),
                });
            }
        }
    }
    names
}

async fn prepare_grouped_regions(
    resident: Arc<ResidentState>,
    source: Arc<DynamicImage>,
    lines: Vec<RecognizedLine>,
    request: &BrowserJobRequest,
    control: &HskControl,
    preceding_context: &[HskPrecedingUtterance],
    sink: &JobUpdateSink,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    preprocessing: &Arc<PreprocessingPool>,
    bubble_masks: &mut BubbleMaskCache,
    mut text_probabilities: ProbabilityMap,
    overall_progress: f32,
) -> std::result::Result<(Vec<PreparedRegion>, ProbabilityMap, Vec<RegionPlan>), CleaningError> {
    if lines.is_empty() {
        return Ok((Vec::new(), text_probabilities, Vec::new()));
    }
    let prepare_started = Instant::now();
    let (image_width, image_height) = source.dimensions();
    let cleanup_supports = lines
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
    let cleanup_tiles = overlapping_tiles(image_width, image_height)
        .into_iter()
        .filter(|tile| {
            !bubble_masks.completed_tiles.contains(&tile.id)
                && cleanup_supports
                    .iter()
                    .any(|support| tile.rect().intersection(*support).is_some())
        })
        .collect::<Vec<_>>();
    let tiles_for_crops = cleanup_tiles.clone();
    let source_for_crops = source.clone();
    let cleanup_crops = preprocessing
        .run(move || {
            Ok(tiles_for_crops
                .iter()
                .map(|tile| source_for_crops.crop_imm(tile.x, tile.y, tile.width, tile.height))
                .collect::<Vec<_>>())
        })
        .await
        .context("prepare semantic cleanup tiles")
        .map_err(CleaningError::pipeline)?;
    cancellation_boundary(cancel.as_ref())?;
    let viewport = sink.viewport();
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
    // The learned segmentation model is a glyph matte source for already
    // recognized OCR regions. It is intentionally scoped to cleanup support
    // tiles and can never invent a text candidate or a translation region.
    if !cleanup_tiles.is_empty() {
        if cleanup_crops.len() != cleanup_tiles.len() {
            return Err(CleaningError::new(
                "TEXT_SEGMENTATION_FAILED",
                "Glyph-mask cleanup prepared an incomplete tile batch.",
            ));
        }
        for (tile_batch, crop_batch) in cleanup_tiles
            .chunks(DETECTOR_TILE_BATCH_SIZE)
            .zip(cleanup_crops.chunks(DETECTOR_TILE_BATCH_SIZE))
        {
            cancellation_boundary(cancel.as_ref())?;
            let permit = cuda_scheduler
                .acquire(CudaWorkload::Vision, bubble_priority, cancel.clone())
                .await
                .map_err(cuda_admission_error)?;
            let results = {
                let segmenter = resident.text_segmenter.lock().map_err(|_| {
                    CleaningError::new("MODEL_STATE_FAILED", "Text segmenter lock poisoned.")
                })?;
                segmenter
                    .inference_batch(crop_batch)
                    .context("segment recognized source glyph mattes")
                    .map_err(CleaningError::pipeline)?
            };
            drop(permit);
            if results.len() != tile_batch.len() {
                return Err(CleaningError::new(
                    "TEXT_SEGMENTATION_FAILED",
                    "Glyph segmentation returned an incomplete tile batch.",
                ));
            }
            for (tile, result) in tile_batch.iter().zip(results) {
                merge_probability_map(&mut text_probabilities, &result, tile.x, tile.y);
            }
        }
    }
    let bubble_started = Instant::now();
    if !cleanup_tiles.is_empty() {
        if cleanup_crops.len() != cleanup_tiles.len() {
            return Err(CleaningError::new(
                "BUBBLE_SEGMENTATION_FAILED",
                "Speech bubble cleanup prepared an incomplete tile batch.",
            ));
        }
        // Keep cleanup admission bounded just like detector work. A large
        // tail can require many contour tiles;
        // holding Vision for the entire batch would prevent visible work from
        // overtaking it until every offscreen contour is decoded.
        for (tile_batch, crop_batch) in cleanup_tiles
            .chunks(DETECTOR_TILE_BATCH_SIZE)
            .zip(cleanup_crops.chunks(DETECTOR_TILE_BATCH_SIZE))
        {
            cancellation_boundary(cancel.as_ref())?;
            let bubble_permit = cuda_scheduler
                .acquire(CudaWorkload::Vision, bubble_priority, cancel.clone())
                .await
                .map_err(cuda_admission_error)?;
            let results = {
                let bubble_segmenter = resident.bubble_segmenter.lock().map_err(|_| {
                    CleaningError::new("MODEL_STATE_FAILED", "Bubble segmenter lock poisoned.")
                })?;
                bubble_segmenter
                    .inference_batch(crop_batch)
                    .context("batch-segment speech bubble contours")
                    .map_err(CleaningError::pipeline)?
            };
            drop(bubble_permit);
            if results.len() != tile_batch.len() {
                return Err(CleaningError::new(
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
                        source_english,
                        translated_chinese: None,
                        ocr_confidence,
                        continuation_group: None,
                        entities: Vec::new(),
                        style: None,
                        role: match candidate.kind {
                            CandidateKind::StoryText => TranslatedRegionRole::Dialogue,
                            CandidateKind::FreeText => TranslatedRegionRole::Narration,
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
        .map_err(CleaningError::pipeline)?;

    cancellation_boundary(cancel.as_ref())?;
    let priority = if sink.viewport().active
        && grouped.iter().any(|group| {
            group.candidate.bubble_rect.intersects_viewport(
                &sink.viewport().visible_rects,
                image_width,
                image_height,
            )
        }) {
        CudaPriority::Visible
    } else {
        CudaPriority::Offscreen
    };
    // Start page cleanup before the multimodal adjudicator.  Both consume the
    // same immutable OCR geometry, and the cleanup task has its own bounded
    // CUDA admission, so the expensive restoration work can overlap the page
    // understanding call instead of extending the first-visible critical
    // path.  Regions later classified as furniture/artwork simply discard
    // their verified decision; no patch is published for them.
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
    let semantic_started = Instant::now();
    let page_adjudication = match adjudicate_grouped_page(
        &resident,
        source.clone(),
        &grouped,
        request,
        preceding_context,
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
            cleanup.cancel();
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
            let region_plans = grouped_region_plans(
                &grouped,
                &all_ids,
                &HashSet::new(),
                &all_ids,
                request,
                image_width,
                image_height,
            );
            return Ok((Vec::new(), text_probabilities, region_plans));
        }
    };
    let multimodal_names =
        protected_names_from_page_adjudication(&grouped, request, &page_adjudication);
    let multimodal_roles =
        apply_page_adjudication_transcripts(&mut grouped, request, &page_adjudication);
    let mut excluded_ids = HashSet::<String>::new();
    let mut preserved_artwork_ids = HashSet::<String>::new();
    let mut unreadable_ids = HashSet::<String>::new();
    if matches!(page_adjudication.page_role, PageRole::Furniture) {
        excluded_ids.extend(
            grouped
                .iter()
                .map(|group| stable_region_id(&request.source_sha256, group.candidate.text_rect)),
        );
    } else if matches!(page_adjudication.page_role, PageRole::Unreadable) {
        for group in grouped.iter() {
            publish_unreadable_group(
                sink,
                group,
                request,
                image_width,
                image_height,
                "Page understanding could not establish readable story text; source pixels were preserved.",
            )?;
            excluded_ids.insert(stable_region_id(
                &request.source_sha256,
                group.candidate.text_rect,
            ));
            unreadable_ids.insert(stable_region_id(
                &request.source_sha256,
                group.candidate.text_rect,
            ));
        }
    } else {
        for (id, role) in multimodal_roles {
            match role {
                PageRegionRole::Furniture => {
                    excluded_ids.insert(id);
                }
                PageRegionRole::Unreadable => {
                    if let Some(group) = grouped.iter().find(|group| {
                        stable_region_id(&request.source_sha256, group.candidate.text_rect) == id
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
                    unreadable_ids.insert(id.clone());
                    excluded_ids.insert(id);
                }
                PageRegionRole::Artwork => {
                    preserved_artwork_ids.insert(id);
                }
                PageRegionRole::Sfx if !request.settings.translate_sound_effects => {
                    // Sound effects stay pixel-identical by default.  Keep
                    // them in the preserved-source channel so the browser
                    // still receives the model's optional Chinese/pinyin
                    // teaching metadata for hover/tap lookup; silently
                    // excluding them would make the text disappear from the
                    // chapter graph and from the learning surface.
                    preserved_artwork_ids.insert(id);
                }
                PageRegionRole::Story | PageRegionRole::Sfx => {}
            }
        }
    }
    let semantic_elapsed = semantic_started.elapsed();
    for group in grouped.iter().filter(|group| {
        preserved_artwork_ids.contains(&stable_region_id(
            &request.source_sha256,
            group.candidate.text_rect,
        ))
    }) {
        publish_preserved_group(sink, group, request, control, image_width, image_height)?;
    }
    let region_plans = grouped_region_plans(
        &grouped,
        &excluded_ids,
        &preserved_artwork_ids,
        &unreadable_ids,
        request,
        image_width,
        image_height,
    );
    grouped.retain(|group| {
        let id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
        !excluded_ids.contains(&id) && !preserved_artwork_ids.contains(&id)
    });
    if grouped.is_empty() {
        // Furniture, artwork, SFX, and unreadable regions have no translated
        // patch consumer.  Do not leave the speculative cleanup task running
        // after the semantic stage has deliberately excluded every group.
        cleanup.cancel();
        return Ok((Vec::new(), text_probabilities, region_plans));
    }

    publish_progress(
        sink,
        BrowserJobStage::Inpainting,
        None,
        Some(overall_progress),
        None,
        None,
        "Restoring the artwork behind the original text",
    )?;
    // Typed entity spans from the page model are the only name authority. No
    // deterministic lexical pass runs after the page decision, so titles such
    // as “Wife” and “Academy Headmaster” remain translatable.
    let semantic_names = multimodal_names;
    cancellation_boundary(cancel.as_ref())?;
    let latest_viewport = sink.viewport();
    let translation_queued_at = tokio::time::Instant::now();
    if std::env::var_os("HSKIFY_TRACE_PIPELINE_TIMING").is_some_and(|value| value == "1") {
        eprintln!(
            "hskify-prepare-timing groups={} bubble_ms={} semantic_ms={} cleanup=overlapped total_ms={}",
            grouped.len(),
            bubble_elapsed.as_millis(),
            semantic_elapsed.as_millis(),
            prepare_started.elapsed().as_millis(),
        );
    }
    let mut prepared_regions = grouped
        .into_iter()
        .map(|group| {
            let candidate = group.candidate;
            let source_english = group.source_english;
            let ocr_confidence = group.ocr_confidence;
            let continuation_group = group.continuation_group;
            let entities = group.entities;
            let role = group.role;
            let source_line_count = group.source_line_count;
            let prediction = group.prediction;
            let style = group.style;
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
            let reading_order = reading_order_key(
                candidate.text_rect,
                image_width,
                image_height,
                request.settings.reading_direction,
            );
            PreparedRegion {
                id: stable_region_id(&request.source_sha256, candidate.text_rect),
                candidate,
                source_english,
                ocr_confidence,
                reading_order,
                continuation_group,
                entities,
                role,
                source_line_count,
                prediction,
                style,
                appearance_bands,
                measured_font_height,
                bubble_polygon,
                layout_polygon,
                cleanup: cleanup.clone(),
                visible,
                proper_names: Vec::new(),
                translation_queued_at,
            }
        })
        .collect::<Vec<_>>();
    if request.settings.name_translation == NameTranslation::KeepOriginal
        && !semantic_names.is_empty()
    {
        for region in &mut prepared_regions {
            region.proper_names = semantic_names
                .iter()
                .filter(|name| {
                    source_contains_name_span(&region.source_english, &name.source_english)
                })
                .cloned()
                .collect();
        }
    }
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
    resident: &ResidentState,
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
        decisions.insert(id, decision);
    }
    Ok(decisions)
}

fn broadened_cleanup_evidence(
    cleaned_groups: &[CleanedGroupedRegion],
    image_width: u32,
    image_height: u32,
) -> (image::GrayImage, Vec<CleanedGroupedRegion>) {
    let mut erase_mask = image::GrayImage::new(image_width, image_height);
    let mut groups = Vec::with_capacity(cleaned_groups.len());
    for cleaned in cleaned_groups {
        let mut broadened = cleaned.clone();
        broadened.cleanup_mask.mask = broaden_cleanup_mask(&cleaned.cleanup_mask.mask);
        merge_cleanup_mask(&mut erase_mask, &broadened.cleanup_mask);
        groups.push(broadened);
    }
    (erase_mask, groups)
}

/// Start the page-level cleanup transaction without making the detector and
/// language paths wait for it. The result is immutable and shared by every
/// region in this analysis batch; each publication awaits only at its final
/// commit point.
#[allow(clippy::too_many_arguments)]
fn spawn_cleanup_batch(
    resident: Arc<ResidentState>,
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
        let first_decisions = match preprocessing
            .run(move || {
                cleanup_decisions_for_image(
                    &source_for_first,
                    &inpainted,
                    &cleaned_for_scoring,
                    &source_sha256_for_cleanup,
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
        let needs_retry = first_decisions
            .values()
            .any(|decision| decision.patch.is_none());
        let decisions = if needs_retry && !cancel.load(Ordering::Acquire) {
            // A rejected candidate gets exactly one cleanup-stage retry using
            // a broadened, mask-derived halo. OCR, page detection, and
            // translation are not restarted with identical evidence.
            let (retry_mask, retry_groups) =
                broadened_cleanup_evidence(&cleaned_groups, image_width, image_height);
            match run_cleanup_inpaint(
                resident.as_ref(),
                source.as_ref(),
                &retry_mask,
                bubble_mask.as_ref(),
                &text_blocks,
                &cancel,
                &cuda_scheduler,
                priority,
            )
            .await
            {
                Ok(retry_image) => {
                    let source_for_retry = source_rgb.clone();
                    let groups_for_retry = retry_groups;
                    let sha_for_retry = source_sha256.clone();
                    let retry_decisions = preprocessing
                        .run(move || {
                            cleanup_decisions_for_image(
                                &source_for_retry,
                                &retry_image,
                                &groups_for_retry,
                                &sha_for_retry,
                            )
                        })
                        .await
                        .unwrap_or_default();
                    first_decisions
                        .into_iter()
                        .map(|(id, first)| {
                            let selected = if first.patch.is_some() {
                                first
                            } else {
                                retry_decisions
                                    .get(&id)
                                    .filter(|retry| retry.patch.is_some())
                                    .cloned()
                                    .unwrap_or(first)
                            };
                            (id, selected)
                        })
                        .collect()
                }
                Err(_) => first_decisions,
            }
        } else {
            first_decisions
        };
        CleanupBatchResult { decisions }
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
    let viewport = sink.viewport();
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
    let viewport = sink.viewport();
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

fn cuda_admission_error(error: CudaAdmissionError) -> CleaningError {
    match error {
        CudaAdmissionError::Cancelled => CleaningError::cancelled(),
        queue_error @ CudaAdmissionError::QueueFull { .. } => {
            CleaningError::new("CUDA_QUEUE_FULL", queue_error.to_string())
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
    if region.role == TranslatedRegionRole::System {
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
    request: &BrowserJobRequest,
    image_width: u32,
    image_height: u32,
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
                    TranslatedRegionRole::Dialogue => RegionRole::Dialogue,
                    TranslatedRegionRole::Narration => RegionRole::Narration,
                    TranslatedRegionRole::System => RegionRole::System,
                }
            };
            RegionPlan {
                id,
                reading_order: reading_order_key(
                    region.candidate.text_rect,
                    image_width,
                    image_height,
                    request.settings.reading_direction,
                ),
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

fn source_contains_name_span(source: &str, name: &str) -> bool {
    let source = source.to_ascii_uppercase();
    let name = name.trim().to_ascii_uppercase();
    if name.is_empty() {
        return false;
    }
    source.match_indices(&name).any(|(start, matched)| {
        let end = start + matched.len();
        let starts_at_boundary =
            start == 0 || !source.as_bytes()[start - 1].is_ascii_alphanumeric();
        let ends_at_boundary =
            end == source.len() || !source.as_bytes()[end].is_ascii_alphanumeric();
        starts_at_boundary && ends_at_boundary
    })
}

fn relevant_protected_names(
    regions: &[PreparedRegion],
    context: &[HskPrecedingUtterance],
    names: &[HskProtectedName],
) -> Vec<HskProtectedName> {
    let mut seen = HashSet::new();
    names
        .iter()
        .filter(|name| {
            (regions.iter().any(|region| {
                source_contains_name_span(&region.source_english, &name.source_english)
            }) || context.iter().any(|utterance| {
                source_contains_name_span(&utterance.source_english, &name.source_english)
            })) && seen.insert(name.source_english.to_ascii_uppercase())
        })
        .cloned()
        .collect()
}

fn publish_preserved_group(
    sink: &JobUpdateSink,
    group: &GroupedRegion,
    request: &BrowserJobRequest,
    control: &HskControl,
    image_width: u32,
    image_height: u32,
) -> std::result::Result<(), CleaningError> {
    let region_id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
    let translated_chinese = group
        .translated_chinese
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned);
    let pinyin = translated_chinese.as_deref().map(|chinese| {
        let pinyin = control
            .lookup(chinese, &[])
            .tokens
            .into_iter()
            .map(|token| {
                if token.pinyin.trim().is_empty() {
                    token.simplified
                } else {
                    token.pinyin
                }
            })
            .filter(|token| !token.trim().is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if pinyin.is_empty() {
            chinese.to_owned()
        } else {
            pinyin
        }
    });
    sink.remember_region_for_lookup(
        region_id.clone(),
        RegionLookupContext {
            source_english: group.source_english.clone(),
            base_chinese: translated_chinese
                .clone()
                .unwrap_or_else(|| group.source_english.clone()),
            displayed_chinese: translated_chinese
                .clone()
                .unwrap_or_else(|| group.source_english.clone()),
            proper_names: Vec::new(),
        },
    );
    sink.publish(JobUpdateDraft::ArtworkPreserved {
        region: PreservedArtworkRegion {
            id: region_id,
            text_polygon: group.candidate.text_rect.polygon(image_width, image_height),
            // Preserved artwork is never replaced or overlaid. The canonical
            // OCR spelling is metadata only: it makes the learning/diagnostic
            // surface useful when a stylized glyph run is recognized with
            // harmless letter substitutions (for example LIGHTNING), while
            // the pixels remain untouched.
            source_english: group.source_english.clone(),
            ocr_confidence: group.ocr_confidence,
            reading_order: reading_order_key(
                group.candidate.text_rect,
                image_width,
                image_height,
                request.settings.reading_direction,
            ),
            translated_chinese,
            pinyin,
            teaching_terms: Vec::new(),
        },
    })
    .map_err(|error| publish_error(error, sink))?;
    Ok(())
}

fn publish_unreadable_group(
    sink: &JobUpdateSink,
    group: &GroupedRegion,
    request: &BrowserJobRequest,
    image_width: u32,
    image_height: u32,
    reason: &str,
) -> std::result::Result<(), CleaningError> {
    let region_id = stable_region_id(&request.source_sha256, group.candidate.text_rect);
    // Keep a source-preserving lookup context even when no patch is safe. The
    // browser can expose the OCR transcript and failure reason on hover
    // without pretending that an unverified Chinese overlay exists.
    sink.remember_region_for_lookup(
        region_id.clone(),
        RegionLookupContext {
            source_english: group.source_english.clone(),
            base_chinese: group.source_english.clone(),
            displayed_chinese: group.source_english.clone(),
            proper_names: Vec::new(),
        },
    );
    sink.publish(JobUpdateDraft::Unreadable {
        region: crate::contracts::UnreadableRegion {
            id: region_id,
            text_polygon: group.candidate.text_rect.polygon(image_width, image_height),
            source_english: group.source_english.clone(),
            ocr_confidence: group.ocr_confidence,
            reading_order: reading_order_key(
                group.candidate.text_rect,
                image_width,
                image_height,
                request.settings.reading_direction,
            ),
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
    base_chinese: String,
    displayed_chinese: String,
    pinyin: String,
    report: ValidationReport,
    repair_state: HskRepairState,
}

struct TranslationCacheEntry {
    value: CachedTranslation,
    bytes: usize,
    last_used: u64,
}

struct TranslationCache {
    entries: HashMap<String, TranslationCacheEntry>,
    retained_bytes: usize,
    clock: u64,
    max_bytes: usize,
}

impl Default for TranslationCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            retained_bytes: 0,
            clock: 0,
            max_bytes: TRANSLATION_CACHE_MAX_BYTES,
        }
    }
}

impl TranslationCache {
    fn get(&mut self, key: &str) -> Option<CachedTranslation> {
        self.clock = self.clock.saturating_add(1);
        let entry = self.entries.get_mut(key)?;
        entry.last_used = self.clock;
        Some(entry.value.clone())
    }

    fn insert(&mut self, key: String, value: CachedTranslation) {
        let bytes = translation_cache_bytes(&key, &value);
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

fn translation_cache_bytes(key: &str, value: &CachedTranslation) -> usize {
    let report = &value.report;
    key.len()
        .saturating_add(value.base_chinese.len())
        .saturating_add(value.displayed_chinese.len())
        .saturating_add(value.pinyin.len())
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
        .saturating_add(std::mem::size_of::<TranslationCacheEntry>())
}

struct TranslationState {
    base_chinese: Option<String>,
    displayed_chinese: Option<String>,
    latest_rejected_chinese: Option<String>,
    latest_rejected_report: Option<ValidationReport>,
    report: Option<ValidationReport>,
    problems: Vec<String>,
    meaning_valid: bool,
    learning_mode: LearningMode,
    repair_state: HskRepairState,
}

impl TranslationState {
    fn excluded() -> Self {
        Self {
            base_chinese: None,
            displayed_chinese: None,
            latest_rejected_chinese: None,
            latest_rejected_report: None,
            report: None,
            problems: Vec::new(),
            meaning_valid: true,
            learning_mode: LearningMode::Natural,
            repair_state: HskRepairState::NotNeeded,
        }
    }

    fn from_initial(
        outcome: HskTranslationOutcome,
        control: &HskControl,
        level: ControlHskLevel,
        proper_names: &[ProperName],
        learning_mode: LearningMode,
    ) -> Self {
        if outcome.is_non_story() {
            return Self::excluded();
        }
        let mut problems = outcome.repair_problems();
        let meaning_valid = outcome.issues.is_empty();
        let base_chinese = nonempty_translation(outcome.text);
        let report = base_chinese
            .as_deref()
            .map(|text| control.validate(text, level, proper_names));
        if let Some(report) = &report
            && !learning_policy_satisfied(report, learning_mode)
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
            meaning_valid,
            learning_mode,
            repair_state: HskRepairState::NotNeeded,
        }
    }

    fn from_cached(translation: CachedTranslation, learning_mode: LearningMode) -> Self {
        let mut problems = Vec::new();
        if !learning_policy_satisfied(&translation.report, learning_mode) {
            append_validation_problems(&mut problems, &translation.report);
        }
        Self {
            base_chinese: Some(translation.base_chinese),
            displayed_chinese: Some(translation.displayed_chinese),
            latest_rejected_chinese: None,
            latest_rejected_report: None,
            report: Some(translation.report),
            problems,
            meaning_valid: true,
            learning_mode,
            repair_state: translation.repair_state,
        }
    }

    fn can_publish(&self) -> bool {
        self.meaning_valid && self.base_chinese.is_some() && self.report.is_some()
    }

    fn repair_succeeded(&self) -> bool {
        self.problems.is_empty() && self.can_publish()
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

    fn initial_translation(&self) -> Option<CachedTranslation> {
        if !self.can_publish() {
            return None;
        }
        let report = self.report.clone()?;
        Some(CachedTranslation {
            base_chinese: self.base_chinese.clone()?,
            displayed_chinese: report.normalized_text.clone(),
            pinyin: String::new(),
            repair_state: if self.problems.is_empty() {
                HskRepairState::NotNeeded
            } else {
                HskRepairState::Pending
            },
            report,
        })
    }

    fn apply_repair(
        &mut self,
        outcome: HskTranslationOutcome,
        control: &HskControl,
        level: ControlHskLevel,
        proper_names: &[ProperName],
    ) -> bool {
        let mut problems = outcome.repair_problems();
        let repaired_meaning_valid = outcome.issues.is_empty();
        let repaired = nonempty_translation(outcome.text);
        let report = repaired
            .as_deref()
            .filter(|_| repaired_meaning_valid)
            .map(|repaired| control.validate(repaired, level, proper_names));
        if let Some(report) = &report
            && !learning_policy_satisfied(report, self.learning_mode)
        {
            append_validation_problems(&mut problems, report);
        }
        self.apply_evaluated_repair(
            repaired.filter(|_| repaired_meaning_valid),
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
            self.meaning_valid = true;
        }
        self.problems = problems;
        self.repair_state = if accepted && self.can_publish() {
            HskRepairState::Accepted
        } else {
            HskRepairState::Rejected
        };
        accepted && self.can_publish()
    }

    fn reject_failed_repair(&mut self) {
        self.problems = vec!["automatic repair could not be completed".to_owned()];
        self.latest_rejected_chinese = None;
        self.repair_state = HskRepairState::Rejected;
    }

    fn finish(mut self) -> Result<CachedTranslation> {
        if !self.can_publish() {
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
        id: id.to_owned(),
        disposition: Default::default(),
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

fn natural_learning_term_budget(report: &ValidationReport) -> usize {
    let absolute_budget = match report.requested_level.get() {
        1..=3 => 1,
        4..=5 => 2,
        _ => 3,
    };
    let percentage_budget = report.lexical_token_count.div_ceil(20);
    absolute_budget.max(percentage_budget)
}

fn learning_policy_satisfied(report: &ValidationReport, mode: LearningMode) -> bool {
    if report.strictly_valid {
        return true;
    }
    if mode == LearningMode::Strict {
        return false;
    }
    let target = match report.requested_level.get() {
        1..=3 => 0.90,
        4 => 0.93,
        _ => 0.95,
    };
    report.above_level_token_count <= natural_learning_term_budget(report)
        && (level_coverage(report) >= target || report.lexical_token_count <= 10)
}

fn translation_is_final(translation: &CachedTranslation) -> bool {
    translation.repair_state != HskRepairState::Pending
}

fn merge_protected_names(
    target: &mut Vec<HskProtectedName>,
    candidates: impl IntoIterator<Item = HskProtectedName>,
) {
    for candidate in candidates {
        if target.iter().any(|existing| {
            existing
                .source_english
                .eq_ignore_ascii_case(&candidate.source_english)
        }) {
            continue;
        }
        target.push(candidate);
    }
    target.sort_by(|left, right| {
        left.source_english
            .to_ascii_lowercase()
            .cmp(&right.source_english.to_ascii_lowercase())
            .then_with(|| left.source_english.cmp(&right.source_english))
    });
}

fn hsk_name_handling(preference: NameTranslation) -> HskNameHandling {
    match preference {
        NameTranslation::KeepOriginal => HskNameHandling::KeepOriginal,
        NameTranslation::Chinese => HskNameHandling::Chinese,
    }
}

fn hsk_learning_mode(mode: LearningMode) -> HskLearningMode {
    match mode {
        LearningMode::Natural => HskLearningMode::Natural,
        LearningMode::Strict => HskLearningMode::Strict,
    }
}

fn control_proper_names(names: &[HskProtectedName]) -> Vec<ProperName> {
    names
        .iter()
        .map(|name| ProperName {
            text: name.chinese.clone(),
            reason: ProperNameReason::UnavoidableProperNoun,
        })
        .collect()
}

// Only an all-hit replay can skip the model. A partial hit regenerates every
// numbered input so uncached siblings see the identical deterministic prompt.
fn primary_generation_indices<T, U>(
    cache_results: &[Option<T>],
    preclassified: &[Option<U>],
) -> Vec<usize> {
    if cache_results.len() != preclassified.len() {
        return Vec::new();
    }
    if cache_results
        .iter()
        .zip(preclassified)
        .all(|(cached, classified)| cached.is_some() || classified.is_some())
    {
        Vec::new()
    } else {
        preclassified
            .iter()
            .enumerate()
            .filter_map(|(index, classified)| classified.is_none().then_some(index))
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn translation_cache_key(
    source_english: &str,
    kind: HskUtteranceKind,
    context: &[HskPrecedingUtterance],
    following_english: &[String],
    protected_names: &[HskProtectedName],
    name_translation: NameTranslation,
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
        ocr_text: &'a str,
        kind: HskUtteranceKind,
        context: &'a [HskPrecedingUtterance],
        following_english: &'a [String],
        protected_names: &'a [HskProtectedName],
        name_translation: NameTranslation,
        learning_mode: LearningMode,
        hsk_level: u8,
        model_id: &'a str,
        model_revision: &'a str,
        prompt_hash: &'a str,
        validator_hash: &'a str,
        hsk_control_revision: &'a str,
    }
    let start = context.len().saturating_sub(MAX_HSK_PRECEDING_UTTERANCES);
    let ocr_text = compact_ocr_text(source_english);
    let material = KeyMaterial {
        schema: TRANSLATION_CACHE_SCHEMA,
        ocr_text: &ocr_text,
        kind,
        context: &context[start..],
        following_english,
        protected_names,
        name_translation,
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
) -> std::result::Result<(), CleaningError> {
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
        CleaningError::new("CLEANUP_UNVERIFIED", "Cleanup patch was not verified.")
    })?;
    let patch_rect = patch.bounds.normalized(image_width, image_height);
    let patch = sink
        .store_generated_patch_png(patch_rect, patch.bytes.clone())
        .map_err(|error| publish_error(error, sink))?;
    let above_level_tokens = above_level_tokens(&translation.report);
    let teaching_terms = teaching_terms(control, &translation.report);
    let translated = TranslatedRegion {
        id: region.id.clone(),
        text_polygon,
        bubble_polygon: Some(region.bubble_polygon.clone()),
        patch,
        source_english: region.source_english.clone(),
        base_chinese: translation.base_chinese.clone(),
        displayed_chinese: translation.displayed_chinese.clone(),
        pinyin: translation.pinyin.clone(),
        ocr_confidence: region.ocr_confidence,
        reading_order: region.reading_order,
        role: Some(region.role),
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
        entities: region.entities.clone(),
        style,
        layout,
        hsk: TranslatedHskStatus {
            requested_level,
            learning_mode,
            strictly_valid: translation.report.strictly_valid,
            level_coverage: level_coverage(&translation.report),
            above_level_tokens,
            teaching_terms,
            repair_state: translation.repair_state,
        },
    };
    sink.remember_region_for_lookup(
        region.id.clone(),
        RegionLookupContext {
            source_english: region.source_english.clone(),
            base_chinese: translation.base_chinese,
            displayed_chinese: translation.displayed_chinese,
            proper_names: translation
                .report
                .exceptions
                .into_iter()
                .map(|exception| ProperName {
                    text: exception.text,
                    reason: exception.reason,
                })
                .collect(),
        },
    );
    sink.publish(JobUpdateDraft::RegionReady {
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
    request: &BrowserJobRequest,
    image_width: u32,
    image_height: u32,
    reason: &str,
) -> std::result::Result<(), CleaningError> {
    sink.remember_region_for_lookup(
        region.id.clone(),
        RegionLookupContext {
            source_english: region.source_english.clone(),
            base_chinese: region.source_english.clone(),
            displayed_chinese: region.source_english.clone(),
            proper_names: Vec::new(),
        },
    );
    sink.publish(JobUpdateDraft::Unreadable {
        region: crate::contracts::UnreadableRegion {
            id: region.id.clone(),
            text_polygon: region
                .candidate
                .text_rect
                .polygon(image_width, image_height),
            source_english: region.source_english.clone(),
            ocr_confidence: region.ocr_confidence,
            reading_order: region.reading_order,
            reason: reason.to_owned(),
        },
    })
    .map_err(|error| publish_error(error, sink))?;
    let _ = request;
    Ok(())
}

/// Convert OCR detector proposals that failed both calibrated recognition
/// views into terminal, source-preserving regions.  The detector is allowed
/// to find text that the recognizer cannot read; silently dropping that
/// proposal would make a coverage metric look green while leaving an English
/// bubble untouched.  The browser receives no patch, only a stable hover/tap
/// target and the reason for the evidence failure.
fn publish_rejected_ocr_regions(
    rejected: &[RejectedOcrLine],
    accepted_rects: &[PixelRect],
    request: &BrowserJobRequest,
    image_width: u32,
    image_height: u32,
    sink: &JobUpdateSink,
) -> std::result::Result<Vec<RegionPlan>, CleaningError> {
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
    selected.sort_by(|left, right| {
        reading_order_key(
            left.candidate.text_rect,
            image_width,
            image_height,
            request.settings.reading_direction,
        )
        .cmp(&reading_order_key(
            right.candidate.text_rect,
            image_width,
            image_height,
            request.settings.reading_direction,
        ))
    });

    let mut plans = Vec::with_capacity(selected.len());
    for line in selected {
        let id = stable_region_id(&request.source_sha256, line.candidate.text_rect);
        let source_english = rejected_ocr_source(&line.prediction);
        sink.remember_region_for_lookup(
            id.clone(),
            RegionLookupContext {
                source_english: source_english.clone(),
                base_chinese: source_english.clone(),
                displayed_chinese: source_english.clone(),
                proper_names: Vec::new(),
            },
        );
        sink.publish(JobUpdateDraft::Unreadable {
            region: crate::contracts::UnreadableRegion {
                id: id.clone(),
                text_polygon: line
                    .candidate
                    .text_rect
                    .polygon(image_width, image_height),
                source_english: source_english.clone(),
                ocr_confidence: line.prediction.confidence.clamp(0.0, 1.0),
                reading_order: reading_order_key(
                    line.candidate.text_rect,
                    image_width,
                    image_height,
                    request.settings.reading_direction,
                ),
                reason: "OCR consensus failed after independent recovery views; source pixels were preserved. Hover it for help.".to_owned(),
            },
        })
        .map_err(|error| publish_error(error, sink))?;
        plans.push(RegionPlan {
            id,
            reading_order: reading_order_key(
                line.candidate.text_rect,
                image_width,
                image_height,
                request.settings.reading_direction,
            ),
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
    // A rejected proposal has failed the calibrated two-view OCR consensus.
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
    let model_style = region.style.as_ref();
    let category = model_style
        .map(style_font_category)
        .unwrap_or_else(|| inferred_font_category(region));
    let font_id = match category {
        FontCategory::Serif => "hmt-serif",
        FontCategory::Handwritten => "hmt-handwritten",
        FontCategory::Display => "hmt-display",
        FontCategory::Brush => "hmt-brush",
        FontCategory::Sans => "hmt-sans",
    };
    let weight = model_style
        .map(|style| style.weight)
        .unwrap_or_else(|| inferred_font_weight(region));
    let writing_mode = model_style
        .map(style_writing_mode)
        .unwrap_or_else(|| inferred_writing_mode(region));
    let alignment = model_style
        .map(style_alignment)
        .unwrap_or_else(|| inferred_alignment(region));
    let line_height = model_style.map_or_else(
        || inferred_line_height(region, suggested_line_count),
        |style| style.line_height,
    );
    let italic_degrees = model_style.map_or(0.0, |style| style.italic_degrees);
    let letter_spacing_em = model_style.map_or(0.0, |style| style.letter_spacing_em);
    let shadow_color = model_style.and_then(|style| style.shadow_color.map(rgb));
    let shadow_x_ratio = model_style.map_or(0.0, |style| style.shadow_x_ratio);
    let shadow_y_ratio = model_style.map_or(0.0, |style| style.shadow_y_ratio);
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

fn style_font_category(style: &PageStyleEvidence) -> FontCategory {
    match style.font_category {
        PageFontCategory::Sans => FontCategory::Sans,
        PageFontCategory::Serif => FontCategory::Serif,
        PageFontCategory::Handwritten => FontCategory::Handwritten,
        PageFontCategory::Display => FontCategory::Display,
        PageFontCategory::Brush => FontCategory::Brush,
    }
}

fn style_writing_mode(style: &PageStyleEvidence) -> WritingMode {
    match style.writing_mode {
        PageWritingMode::HorizontalTb => WritingMode::HorizontalTb,
        PageWritingMode::VerticalRl => WritingMode::VerticalRl,
    }
}

fn style_alignment(style: &PageStyleEvidence) -> TextAlignment {
    match style.alignment {
        PageTextAlignment::Left => TextAlignment::Left,
        PageTextAlignment::Center => TextAlignment::Center,
        PageTextAlignment::Right => TextAlignment::Right,
    }
}

/// Infer typography from measured source-line evidence rather than selecting
/// one global replacement style.  These are deliberately visual/layout
/// signals (stroke, color runs, aspect, and measured spacing), never title or
/// word lists, so the same rules apply to unfamiliar readers.
fn inferred_font_category(region: &PreparedRegion) -> FontCategory {
    let color_runs = region.appearance_bands.len();
    let outlined = region.prediction.has_stroke_color;
    if region.role == TranslatedRegionRole::System && (outlined || color_runs > 1) {
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

fn translation_queue_ready_len(
    pending: &[PreparedRegion],
    force: bool,
    now: tokio::time::Instant,
) -> usize {
    if pending.is_empty() {
        return 0;
    }
    if force || pending.len() >= TRANSLATION_BATCH_MIN {
        return pending.len();
    }
    translation_queue_deadline(pending)
        .is_some_and(|deadline| deadline <= now)
        .then_some(pending.len())
        .unwrap_or(0)
}

fn translation_queue_deadline(pending: &[PreparedRegion]) -> Option<tokio::time::Instant> {
    pending
        .iter()
        .map(|region| region.translation_queued_at + TRANSLATION_MAX_FLUSH_DELAY)
        .min()
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
    now: tokio::time::Instant,
    cancelled: bool,
    first_visible_region: bool,
) -> TranslationBoundaryAction {
    if cancelled {
        return TranslationBoundaryAction::Cancelled;
    }
    if first_visible_region {
        return TranslationBoundaryAction::Dispatch(1);
    }
    let eligible = translation_queue_ready_len(pending, force, now);
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
) -> std::result::Result<(), CleaningError> {
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

fn publish_error(error: crate::server::PublishError, sink: &JobUpdateSink) -> CleaningError {
    if sink.is_cancelled() {
        CleaningError::cancelled()
    } else {
        CleaningError::new("UPDATE_PUBLISH_FAILED", error.to_string())
    }
}

fn cancellation_boundary(cancel: &AtomicBool) -> std::result::Result<(), CleaningError> {
    if cancel.load(Ordering::Acquire) {
        Err(CleaningError::cancelled())
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
        region: result.region.map(|region| LookupRegion {
            displayed_chinese: region.displayed_chinese,
            base_chinese: region.base_chinese,
            source_english: region.source_english,
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
    fn source_guided_ocr_accepts_consensus_confidence_and_defers_non_story_text() {
        let rect = PixelRect::new(10.0, 10.0, 90.0, 40.0).unwrap();
        let candidate = |kind| Candidate {
            kind,
            text_rect: rect,
            bubble_rect: rect.expand(10.0, 100, 100),
            confirmed_bubble_rect: rect.expand(10.0, 100, 100),
            detector_confidence: 0.99,
            has_detector_core: true,
            rotation_radians: 0.0,
        };
        let line = |candidate, confidence| RecognizedLine {
            candidate,
            prediction: PpOcrPrediction {
                text: "ordinary dialogue".to_owned(),
                confidence,
                text_color: [0, 0, 0],
                stroke_color: [255, 255, 255],
                has_stroke_color: false,
                appearance_bands: Vec::new(),
                ocr_lines: Vec::new(),
            },
            crop_bounds: rect.pixel_bounds(100, 100),
        };
        let story = candidate(CandidateKind::StoryText);
        let free = candidate(CandidateKind::FreeText);
        let (accepted, deferred, disputed) = verified_source_guided_ocr_lines(
            vec![story, free],
            vec![line(story, 0.99), line(free, 0.99)],
        );
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].candidate.kind, CandidateKind::StoryText);
        assert_eq!(deferred.len(), 1);
        assert_eq!(deferred[0].candidate.kind, CandidateKind::FreeText);
        assert_eq!(disputed, vec![free]);

        let (accepted, deferred, disputed) =
            verified_source_guided_ocr_lines(vec![story], vec![line(story, 0.54)]);
        assert!(accepted.is_empty());
        assert_eq!(deferred.len(), 1);
        assert_eq!(disputed, vec![story]);
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
            source_english: "A readable sentence.".to_owned(),
            translated_chinese: None,
            ocr_confidence: 0.9,
            continuation_group: None,
            entities: Vec::new(),
            role: TranslatedRegionRole::Dialogue,
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
            style: None,
            appearance_bands: Vec::new(),
            measured_font_height: 72.0,
            cleanup_blocks: Vec::new(),
        }];

        let (cropped, viewport) = page_evidence_surface(&source, &grouped, 1_000, 12_000);

        assert!(cropped.height() < source.height());
        assert!(viewport.y0 > 0.0);
        let polygon =
            polygon_in_evidence_viewport(text_rect, viewport, cropped.width(), cropped.height());
        assert!(
            polygon
                .iter()
                .all(|point| { (0.0..=1.0).contains(&point.x) && (0.0..=1.0).contains(&point.y) })
        );
        assert!(polygon.iter().any(|point| point.y > 0.1 && point.y < 0.9));
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
    fn natural_learning_accepts_one_teachable_term_but_strict_mode_does_not() {
        let mut report = validation_report("侦探看了看。", vec![above_level_violation("侦探")]);
        report.lexical_token_count = 4;
        report.above_level_token_count = 1;

        assert!(learning_policy_satisfied(&report, LearningMode::Natural));
        assert!(!learning_policy_satisfied(&report, LearningMode::Strict));
        assert_eq!(level_coverage(&report), 0.75);
    }

    #[test]
    fn natural_learning_rejects_excess_advanced_vocabulary_in_a_short_line() {
        let mut second = above_level_violation("证据");
        second.start_char = 2;
        second.end_char = 4;
        let mut report = validation_report("侦探证据", vec![above_level_violation("侦探"), second]);
        report.lexical_token_count = 5;
        report.above_level_token_count = 2;

        assert!(!learning_policy_satisfied(&report, LearningMode::Natural));
    }

    fn usable_pending_state() -> TranslationState {
        let report = validation_report("研究生", vec![above_level_violation("研究生")]);
        TranslationState {
            base_chinese: Some("研究生".to_owned()),
            displayed_chinese: Some("研究生".to_owned()),
            latest_rejected_chinese: None,
            latest_rejected_report: None,
            report: Some(report),
            problems: vec!["replace invalid or above-level token `研究生` at 0..3".to_owned()],
            meaning_valid: true,
            learning_mode: LearningMode::Strict,
            repair_state: HskRepairState::Pending,
        }
    }

    fn pending_repair(id: &str) -> PendingRepair {
        let rect = PixelRect::new(1.0, 1.0, 9.0, 9.0).unwrap();
        PendingRepair {
            cache_key: format!("cache-{id}"),
            region: PreparedRegion {
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
                ocr_confidence: 0.99,
                reading_order: 0,
                continuation_group: None,
                entities: Vec::new(),
                role: TranslatedRegionRole::Dialogue,
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
                style: None,
                appearance_bands: Vec::new(),
                measured_font_height: rect.height(),
                bubble_polygon: rect.polygon(10, 10),
                layout_polygon: rect.polygon(10, 10),
                cleanup: CleanupBatchTask::ready(CleanupBatchResult {
                    decisions: HashMap::new(),
                }),
                visible: false,
                proper_names: Vec::new(),
                translation_queued_at: tokio::time::Instant::now(),
            },
            utterance: HskRepairUtterance {
                id: id.to_owned(),
                kind: HskUtteranceKind::Dialogue,
                source_english: "Graduate student".to_owned(),
                max_characters: 64,
                max_lines: 3,
                rejected_chinese: Some("研究生".to_owned()),
                avoid_chinese: vec!["研究生".to_owned()],
                problems: vec!["above level".to_owned()],
            },
            protected_names: Vec::new(),
            state: usable_pending_state(),
            attempts: 0,
        }
    }

    fn prepared_region_with_source(source: &str) -> PreparedRegion {
        let mut region = pending_repair("relevance").region;
        region.source_english = source.to_owned();
        region
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

    #[test]
    fn cache_key_is_stable_across_batch_regrouping_and_numbered_position() {
        let context = vec![HskPrecedingUtterance {
            source_english: "Earlier".to_owned(),
            chinese: "以前".to_owned(),
        }];
        let original_group = ["Wait here", "A sibling", "  Leave \n now  "];
        let regrouped = ["Leave now", "A different sibling"];

        assert_eq!(
            translation_cache_key(
                original_group[2],
                HskUtteranceKind::Dialogue,
                &context,
                &[],
                &[],
                NameTranslation::KeepOriginal,
                LearningMode::Natural,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            ),
            translation_cache_key(
                regrouped[0],
                HskUtteranceKind::Dialogue,
                &context,
                &[],
                &[],
                NameTranslation::KeepOriginal,
                LearningMode::Natural,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        );
    }

    #[test]
    fn cache_key_separates_following_source_context() {
        let first = vec!["Wait here.".to_owned()];
        let second = vec!["Leave now.".to_owned()];
        let key = |following: &[String]| {
            translation_cache_key(
                "We should go.",
                HskUtteranceKind::Dialogue,
                &[],
                following,
                &[],
                NameTranslation::KeepOriginal,
                LearningMode::Natural,
                3,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        };

        assert_ne!(key(&first), key(&second));
    }

    #[test]
    fn cache_key_separates_original_and_chinese_name_preferences() {
        let key = |name_translation| {
            translation_cache_key(
                "Alice is here",
                HskUtteranceKind::Dialogue,
                &[],
                &[],
                &[],
                name_translation,
                LearningMode::Natural,
                3,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        };

        assert_ne!(
            key(NameTranslation::KeepOriginal),
            key(NameTranslation::Chinese)
        );
    }

    #[test]
    fn cache_key_separates_natural_and_strict_learning_modes() {
        let key = |learning_mode| {
            translation_cache_key(
                "The detective checked the evidence.",
                HskUtteranceKind::Dialogue,
                &[],
                &[],
                &[],
                NameTranslation::KeepOriginal,
                learning_mode,
                3,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        };

        assert_ne!(key(LearningMode::Natural), key(LearningMode::Strict));
    }

    #[test]
    fn only_pretranslation_approved_names_become_hsk_exceptions() {
        let names = control_proper_names(&[
            HskProtectedName {
                source_english: "Alice".to_owned(),
                chinese: "Alice".to_owned(),
            },
            HskProtectedName {
                source_english: "Bob".to_owned(),
                chinese: "Bob".to_owned(),
            },
        ]);

        assert_eq!(
            names
                .iter()
                .map(|name| name.text.as_str())
                .collect::<Vec<_>>(),
            ["Alice", "Bob"]
        );
        assert!(control_proper_names(&[]).is_empty());
    }

    #[test]
    fn cache_key_covers_every_output_affecting_input_and_limits_context_to_six() {
        let key = |source_english: &str,
                   context: &[HskPrecedingUtterance],
                   protected_names: &[HskProtectedName],
                   hsk_level: u8,
                   model_id: &str,
                   model_revision: &str,
                   prompt_hash: &str,
                   validator_hash: &str,
                   control_revision: &str| {
            translation_cache_key(
                source_english,
                HskUtteranceKind::Dialogue,
                context,
                &[],
                protected_names,
                NameTranslation::KeepOriginal,
                LearningMode::Natural,
                hsk_level,
                model_id,
                model_revision,
                prompt_hash,
                validator_hash,
                control_revision,
            )
        };
        let context = vec![HskPrecedingUtterance {
            source_english: "Earlier".to_owned(),
            chinese: "以前".to_owned(),
        }];
        let no_names = Vec::<HskProtectedName>::new();
        let base = key(
            "Leave now",
            &context,
            &no_names,
            2,
            "qwen",
            "model-r1",
            "prompt-r1",
            "validator-r1",
            "control-r1",
        );
        assert_ne!(
            base,
            key(
                "Stay here",
                &context,
                &no_names,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        );
        assert_ne!(
            base,
            key(
                "Leave now",
                &[],
                &no_names,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        );
        assert_ne!(
            base,
            key(
                "Leave now",
                &context,
                &no_names,
                3,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        );
        assert_ne!(
            base,
            key(
                "Leave now",
                &context,
                &no_names,
                2,
                "other-qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        );
        assert_ne!(
            base,
            key(
                "Leave now",
                &context,
                &no_names,
                2,
                "qwen",
                "model-r2",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        );
        assert_ne!(
            base,
            key(
                "Leave now",
                &context,
                &no_names,
                2,
                "qwen",
                "model-r1",
                "prompt-r2",
                "validator-r1",
                "control-r1",
            )
        );
        assert_ne!(
            base,
            key(
                "Leave now",
                &context,
                &no_names,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r2",
                "control-r1",
            )
        );
        assert_ne!(
            base,
            key(
                "Leave now",
                &context,
                &no_names,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r2",
            )
        );
        assert_ne!(
            base,
            key(
                "Leave now",
                &context,
                &[HskProtectedName {
                    source_english: "Leave now".to_owned(),
                    chinese: "爱丽丝".to_owned(),
                }],
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        );

        let seven_context_items = (0..7)
            .map(|index| HskPrecedingUtterance {
                source_english: format!("source-{index}"),
                chinese: format!("中文-{index}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            key(
                "Leave now",
                &seven_context_items,
                &no_names,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            ),
            key(
                "Leave now",
                &seven_context_items[1..],
                &no_names,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        );
    }

    #[test]
    fn relevant_protected_names_keep_only_contextual_names_in_input_order() {
        let regions = vec![prepared_region_with_source("Alice met Bob")];
        let context = vec![HskPrecedingUtterance {
            source_english: "Carol spoke to Alice".to_owned(),
            chinese: "Carol".to_owned(),
        }];
        let names = vec![
            HskProtectedName {
                source_english: "Unrelated".to_owned(),
                chinese: "Unrelated".to_owned(),
            },
            HskProtectedName {
                source_english: "alice".to_owned(),
                chinese: "Alice first".to_owned(),
            },
            HskProtectedName {
                source_english: "CAROL".to_owned(),
                chinese: "Carol".to_owned(),
            },
            HskProtectedName {
                source_english: "lice".to_owned(),
                chinese: "Partial".to_owned(),
            },
            HskProtectedName {
                source_english: "ALICE".to_owned(),
                chinese: "Alice duplicate".to_owned(),
            },
            HskProtectedName {
                source_english: "BOB".to_owned(),
                chinese: "Bob".to_owned(),
            },
        ];
        let filtered = relevant_protected_names(&regions, &context, &names);

        assert_eq!(
            filtered
                .iter()
                .map(|name| name.source_english.as_str())
                .collect::<Vec<_>>(),
            ["alice", "CAROL", "BOB"]
        );
        assert_eq!(filtered[0].chinese, "Alice first");
        assert_eq!(
            control_proper_names(&filtered)[0].text.as_str(),
            "Alice first"
        );

        let key = |protected_names: &[HskProtectedName]| {
            translation_cache_key(
                "Alice met Bob",
                HskUtteranceKind::Dialogue,
                &context,
                &[],
                protected_names,
                NameTranslation::KeepOriginal,
                LearningMode::Natural,
                2,
                "qwen",
                "model-r1",
                "prompt-r1",
                "validator-r1",
                "control-r1",
            )
        };
        let unrelated = relevant_protected_names(
            &regions,
            &context,
            &[HskProtectedName {
                source_english: "Unrelated".to_owned(),
                chinese: "Unrelated".to_owned(),
            }],
        );
        assert!(unrelated.is_empty());
        assert_eq!(key(&[]), key(&unrelated));
        assert_ne!(key(&[]), key(&filtered));
    }

    #[test]
    fn partial_cache_hits_preserve_the_full_primary_prompt_batch() {
        let sources = ["First", "Second", "Third"];
        let partial_hits = [Some("cached-first"), None, Some("cached-third")];
        let generation_sources =
            primary_generation_indices(&partial_hits, &[None::<()>, None, None])
                .into_iter()
                .map(|index| sources[index])
                .collect::<Vec<_>>();

        assert_eq!(generation_sources, sources.to_vec());
        assert!(
            primary_generation_indices(
                &[
                    Some("cached-first"),
                    Some("cached-second"),
                    Some("cached-third"),
                ],
                &[None::<()>, None, None]
            )
            .is_empty()
        );
    }

    #[test]
    fn semantic_sound_effects_are_removed_without_shrinking_story_context() {
        let cache = [Some("cached-first"), None, None];
        let classified = [None, Some("sfx"), None];

        assert_eq!(primary_generation_indices(&cache, &classified), vec![0, 2]);
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
        let mut first_in_reading_order = pending_repair("first").region;
        first_in_reading_order.reading_order = 1;
        first_in_reading_order.translation_queued_at = now + Duration::from_millis(20);

        let mut second_in_reading_order = pending_repair("second").region;
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
        let mut offscreen = pending_repair("offscreen").region;
        offscreen.visible = false;
        offscreen.reading_order = 1;
        offscreen.translation_queued_at = now;

        let mut visible = pending_repair("visible").region;
        visible.visible = true;
        visible.reading_order = 99;
        visible.translation_queued_at = now + Duration::from_millis(20);

        let mut pending = vec![offscreen, visible];
        sort_pending_translation(&mut pending);

        assert_eq!(pending[0].id, "visible");
        assert_eq!(pending[1].id, "offscreen");
    }

    #[tokio::test]
    async fn sparse_translation_tail_is_not_dispatched_before_its_deadline() {
        let now = tokio::time::Instant::now();
        let mut one = vec![pending_repair("bubble-a").region];
        one[0].translation_queued_at = now;
        assert_eq!(
            translation_queue_deadline(&one),
            Some(now + TRANSLATION_MAX_FLUSH_DELAY)
        );
        assert_eq!(
            translation_boundary_action(&one, false, now, false, false),
            TranslationBoundaryAction::ContinueUpstream
        );
        assert_eq!(
            translation_boundary_action(
                &one,
                false,
                now + TRANSLATION_MAX_FLUSH_DELAY - Duration::from_nanos(1),
                false,
                false
            ),
            TranslationBoundaryAction::ContinueUpstream
        );
    }

    #[test]
    fn first_visible_translation_is_dispatched_alone_before_throughput_batching() {
        let now = tokio::time::Instant::now();
        let mut pending = (0..TRANSLATION_BATCH_MAX)
            .map(|index| pending_repair(&format!("bubble-{index}")).region)
            .collect::<Vec<_>>();
        for region in &mut pending {
            region.translation_queued_at = now;
        }

        assert_eq!(
            translation_boundary_action(&pending, true, now, false, true),
            TranslationBoundaryAction::Dispatch(1)
        );
        assert_eq!(
            translation_boundary_action(&pending, true, now, false, false),
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

    #[tokio::test]
    async fn unready_translation_tail_does_not_consume_tokio_time_or_block_upstream_work() {
        let now = tokio::time::Instant::now();
        let mut two = vec![
            pending_repair("bubble-a").region,
            pending_repair("bubble-b").region,
        ];
        two[0].translation_queued_at = now + Duration::from_millis(20);
        two[1].translation_queued_at = now;

        // The production boundary uses this zero result to return immediately,
        // allowing the caller's already-available OCR/detector batch to run.
        let checked_at = tokio::time::Instant::now();
        assert_eq!(
            translation_boundary_action(&two, false, now, false, false),
            TranslationBoundaryAction::ContinueUpstream
        );
        assert!(checked_at.elapsed() < Duration::from_millis(5));
    }

    #[tokio::test]
    async fn sparse_translation_tail_dispatches_at_the_deadline_boundary() {
        let now = tokio::time::Instant::now();
        let mut tail = vec![
            pending_repair("bubble-a").region,
            pending_repair("bubble-b").region,
        ];
        tail[0].translation_queued_at = now + Duration::from_millis(20);
        tail[1].translation_queued_at = now;

        assert_eq!(
            translation_boundary_action(
                &tail,
                false,
                now + TRANSLATION_MAX_FLUSH_DELAY - Duration::from_nanos(1),
                false,
                false
            ),
            TranslationBoundaryAction::ContinueUpstream
        );
        assert_eq!(
            translation_boundary_action(
                &tail,
                false,
                now + TRANSLATION_MAX_FLUSH_DELAY,
                false,
                false,
            ),
            TranslationBoundaryAction::Dispatch(2)
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
            .map(|index| pending_repair(&format!("bubble-{index}")).region)
            .collect::<Vec<_>>();
        for region in &mut full {
            region.translation_queued_at = now;
        }
        assert_eq!(
            translation_boundary_action(&full[..6], false, now, false, false),
            TranslationBoundaryAction::Dispatch(TRANSLATION_BATCH_MAX)
        );
        assert_eq!(
            translation_boundary_action(&full, false, now, false, false),
            TranslationBoundaryAction::Dispatch(4)
        );

        let cancel = AtomicBool::new(false);
        assert!(cancellation_boundary(&cancel).is_ok());
        cancel.store(true, Ordering::Release);
        assert_eq!(
            translation_boundary_action(&full, false, now, cancel.load(Ordering::Acquire), false,),
            TranslationBoundaryAction::Cancelled
        );
        assert_eq!(
            cancellation_boundary(&cancel).unwrap_err().code,
            "CANCELLED"
        );
    }

    #[test]
    fn repair_queue_deduplicates_and_stays_closed_until_primary_work_finishes() {
        let mut queue = RepairQueue::default();
        queue.enqueue(pending_repair("bubble-a"));
        queue.enqueue(pending_repair("bubble-a"));

        assert_eq!(queue.jobs.len(), 1);
        assert!(queue.take_batch(TRANSLATION_BATCH_MAX).is_empty());
        assert_eq!(queue.jobs.len(), 1);

        queue.finish_primary_phase();
        let batch = queue.take_batch(TRANSLATION_BATCH_MAX);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].region.id, "bubble-a");
        assert!(queue.take_batch(TRANSLATION_BATCH_MAX).is_empty());
    }

    #[test]
    fn first_visible_repair_can_finish_before_offscreen_primary_work() {
        let mut queue = RepairQueue::default();
        let offscreen = pending_repair("offscreen");
        let mut visible = pending_repair("visible");
        visible.region.visible = true;
        queue.enqueue(offscreen);
        queue.enqueue(visible);

        let early = queue.take_visible_batch(TRANSLATION_BATCH_MAX);

        assert_eq!(early.len(), 1);
        assert_eq!(early[0].region.id, "visible");
        assert_eq!(queue.jobs.len(), 1);
        assert_eq!(queue.jobs[0].region.id, "offscreen");
        queue.finish_primary_phase();
        assert_eq!(queue.take_batch(TRANSLATION_BATCH_MAX).len(), 1);
    }

    #[test]
    fn missing_visible_primary_is_retried_on_the_interactive_critical_path() {
        let mut queue = RepairQueue::default();
        let mut missing = pending_repair("missing");
        missing.region.visible = true;
        missing.utterance.rejected_chinese = None;
        queue.enqueue(missing);

        let early = queue.take_visible_batch(TRANSLATION_BATCH_MAX);
        assert_eq!(early.len(), 1);
        assert_eq!(early[0].region.id, "missing");
        assert!(queue.jobs.is_empty());
    }

    #[test]
    fn pending_primary_stays_internal_until_repair_is_terminal() {
        let primary = usable_pending_state().initial_translation().unwrap();
        assert_eq!(primary.displayed_chinese, "研究生");
        assert_eq!(primary.repair_state, HskRepairState::Pending);
        assert!(!translation_is_final(&primary));

        let mut cache = TranslationCache::default();
        cache.insert("primary-key".to_owned(), primary);
        let cached = cache.get("primary-key").unwrap();
        assert_eq!(cached.displayed_chinese, "研究生");
        assert_eq!(cached.repair_state, HskRepairState::Pending);
        assert!(!translation_is_final(&cached));
    }

    #[test]
    fn repair_retry_is_reserved_for_an_unpublishable_primary() {
        let usable = pending_repair("usable");
        assert!(!should_retry_repair(&usable, false));

        let mut unsafe_job = pending_repair("unsafe");
        unsafe_job.state.meaning_valid = false;
        assert!(should_retry_repair(&unsafe_job, false));
        unsafe_job.attempts = MAX_HSK_REPAIR_ATTEMPTS - 1;
        assert!(!should_retry_repair(&unsafe_job, false));
        assert!(!should_retry_repair(&unsafe_job, true));
    }

    #[test]
    fn only_a_valid_improvement_replaces_a_usable_primary() {
        let mut accepted_state = usable_pending_state();
        assert!(accepted_state.apply_evaluated_repair(
            Some("学生".to_owned()),
            Some(validation_report("学生", Vec::new())),
            Vec::new(),
        ));
        let accepted = accepted_state.finish().unwrap();
        assert_eq!(accepted.base_chinese, "研究生");
        assert_eq!(accepted.displayed_chinese, "学生");
        assert_eq!(accepted.repair_state, HskRepairState::Accepted);
        let mut cache = TranslationCache::default();
        cache.insert("refined-key".to_owned(), accepted);
        let refined = cache.get("refined-key").unwrap();
        assert_eq!(refined.displayed_chinese, "学生");
        assert_eq!(refined.repair_state, HskRepairState::Accepted);

        let mut worse_state = usable_pending_state();
        assert!(!worse_state.apply_evaluated_repair(
            Some("教授".to_owned()),
            Some(validation_report(
                "教授",
                vec![above_level_violation("教授")],
            )),
            vec!["still above level".to_owned()],
        ));
        assert!(!worse_state.repair_succeeded());
        assert_eq!(worse_state.latest_rejected_chinese.as_deref(), Some("教授"));
        assert_eq!(worse_state.avoid_chinese(), vec!["教授"]);
        let worse = worse_state.finish().unwrap();
        assert_eq!(worse.base_chinese, "研究生");
        assert_eq!(worse.displayed_chinese, "研究生");
        assert_eq!(worse.repair_state, HskRepairState::Rejected);

        let mut malformed_state = usable_pending_state();
        assert!(!malformed_state.apply_evaluated_repair(
            None,
            None,
            vec!["malformed repair".to_owned()],
        ));
        let malformed = malformed_state.finish().unwrap();
        assert_eq!(malformed.base_chinese, "研究生");
        assert_eq!(malformed.displayed_chinese, "研究生");
        assert_eq!(malformed.repair_state, HskRepairState::Rejected);
    }

    #[test]
    fn repair_transport_failure_terminalizes_a_publishable_primary() {
        let mut state = usable_pending_state();

        state.reject_failed_repair();

        assert!(state.can_publish());
        let terminal = state.finish().unwrap();
        assert_eq!(terminal.base_chinese, "研究生");
        assert_eq!(terminal.displayed_chinese, "研究生");
        assert_eq!(terminal.repair_state, HskRepairState::Rejected);
    }

    #[test]
    fn repair_transport_failure_does_not_publish_an_unsafe_primary() {
        let mut state = TranslationState {
            base_chinese: Some("索林来了。".to_owned()),
            displayed_chinese: None,
            latest_rejected_chinese: None,
            latest_rejected_report: None,
            report: Some(validation_report("索林来了。", Vec::new())),
            problems: vec!["protected name was not preserved".to_owned()],
            meaning_valid: false,
            learning_mode: LearningMode::Natural,
            repair_state: HskRepairState::Pending,
        };

        state.reject_failed_repair();

        assert!(!state.can_publish());
        assert!(state.finish().is_err());
    }

    #[test]
    fn rejected_name_preservation_never_becomes_publishable() {
        let mut state = TranslationState {
            base_chinese: Some("我昨天看见索林了。".to_owned()),
            displayed_chinese: None,
            latest_rejected_chinese: None,
            latest_rejected_report: None,
            report: Some(validation_report("我昨天看见索林了。", Vec::new())),
            problems: vec!["translate protected name `Neris` exactly as `Neris`".to_owned()],
            meaning_valid: false,
            learning_mode: LearningMode::Natural,
            repair_state: HskRepairState::Pending,
        };

        assert!(!state.apply_evaluated_repair(
            None,
            None,
            vec!["translate protected name `Neris` exactly as `Neris`".to_owned()],
        ));
        assert!(!state.can_publish());
        assert!(state.finish().is_err());
    }

    #[test]
    fn browser_preprocessing_pool_has_exactly_six_threads() {
        let pool = global_preprocessing_pool().unwrap();
        assert_eq!(pool.thread_count(), PREPROCESSING_THREADS);
        assert_eq!(PREPROCESSING_THREADS, 6);
    }
}
