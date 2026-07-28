//! Direct, progressive browser pipeline.
//!
//! The browser path deliberately does not create Koharu projects. It decodes
//! the upload once, runs resident CUDA models over overlapping detector tiles,
//! restores accepted text regions in one image-level semantic inpainting pass,
//! and publishes one transparent cleanup patch per translated dialogue region.

mod geometry;
mod patch;
mod ppocr_v5;

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use camino::Utf8PathBuf;
use hsk_control::{
    HskControl, HskLevel as ControlHskLevel, LookupRegionContext as ControlLookupRegion,
    ProperName, ProperNameReason, ValidationReport, ViolationReason,
};
use image::{DynamicImage, GenericImageView};
use koharu_app::llm::{
    HSK_TRANSLATION_MODEL, HskLearningMode, HskNameHandling, HskPrecedingUtterance,
    HskProtectedName, HskRepairUtterance, HskSemanticLayout, HskSourceUtterance,
    HskTranslationBatchRequest, HskTranslationDisposition, HskTranslationIssue,
    HskTranslationOutcome, HskTranslationRepairBatchRequest, HskUtteranceKind,
    MAX_HSK_PRECEDING_UTTERANCES,
};
use koharu_app::{App, AppConfig};
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
use tokio::sync::{OnceCell, oneshot};

use self::geometry::{
    Candidate, CandidateKind, DetectedBubble, PixelBounds, PixelRect, Tile,
    bubble_segmentation_fallback_candidates, bubbles_for_tile, candidates_for_tile,
    next_detector_batch_count, ocr_crop_rect, overlapping_tiles, prioritize_tiles,
    reading_order_key, segmentation_fallback_candidates, spatially_dedupe, take_finalized_lines,
    text_candidate_is_confirmed,
};
use self::patch::{
    CleanupMask, PatchPng, bubble_component_bounds, bubble_id_for_rect, bubble_id_mask,
    compact_cleanup_mask, crop_probability_map, label_bubble_components, make_inpainted_patch,
    merge_binary_mask, merge_cleanup_mask, merge_probability_map, region_polygons,
    verified_text_mask_for_regions,
};
use self::ppocr_v5::{EnglishPpOcrV5, MAX_LINE_BATCH_SIZE, PpOcrAppearanceBand, PpOcrPrediction};
use crate::contracts::{
    BrowserJobRequest, BrowserJobStage, BrowserTextColorBand, BrowserTextLayout, BrowserTextStyle,
    FontCategory, HskLevel, HskRepairState, LearningMode, LookupRegion, LookupResult, LookupToken,
    NameTranslation, NormalizedRect, Point, ProgressiveHskStatus, ProgressiveRegion, TeachingTerm,
    TeachingTermReason, TextAlignment, WritingMode,
};
use crate::crypto::sha256_hex;
use crate::cuda_scheduler::{
    CudaAdmissionError, CudaPriority, CudaScheduler, global_cuda_scheduler,
};
use crate::server::{JobUpdateDraft, JobUpdateSink};
use crate::setup::{
    BUBBLE_SEGMENTER_CONFIG_ID, BUBBLE_SEGMENTER_WEIGHTS_ID, DETECTOR_CONFIG_ID,
    DETECTOR_PREPROCESSOR_ID, DETECTOR_WEIGHTS_ID, INPAINTER_WEIGHTS_ID, OCR_CONFIG_ID,
    OCR_MODEL_ID, ResidentResourcePaths, TEXT_SEGMENTER_WEIGHTS_ID, TRANSLATION_MODEL_ID,
};

const OCR_REGION_BATCH_SIZE: usize = MAX_LINE_BATCH_SIZE;
const TRANSLATION_BATCH_MAX: usize = 6;
const TRANSLATION_BATCH_MIN: usize = 3;
const TRANSLATION_MAX_FLUSH_DELAY: Duration = Duration::from_millis(75);
const MAX_TARGETED_REPAIR_ATTEMPTS: usize = 4;
const BROWSER_QWEN_INFERENCE_THREADS: i32 = 6;
const MIN_OCR_CONFIDENCE: f32 = 0.45;
const TRANSLATION_CACHE_SCHEMA: &str = "hskify-direct-hsk-region-cache-v29";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RegionLookupContext {
    pub(crate) source_english: String,
    pub(crate) base_chinese: String,
    pub(crate) displayed_chinese: String,
    pub(crate) proper_names: Vec<ProperName>,
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
const ENTITY_MEMORY_MAX_SESSIONS: usize = 64;
const ENTITY_MEMORY_MAX_NAMES_PER_SESSION: usize = 256;
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

    fn resources_ready(&self) -> bool;
}

pub(crate) struct KoharuPipeline {
    cache_root: PathBuf,
    cuda_scheduler: Arc<CudaScheduler>,
    resident: OnceCell<Arc<ResidentState>>,
    hsk_control: OnceCell<Arc<HskControl>>,
    translation_cache: Mutex<TranslationCache>,
    entity_memory: Mutex<ChapterEntityMemory>,
}

impl KoharuPipeline {
    pub(crate) fn new(cache_root: PathBuf) -> Self {
        Self {
            cache_root,
            cuda_scheduler: global_cuda_scheduler(),
            resident: OnceCell::new(),
            hsk_control: OnceCell::new(),
            translation_cache: Mutex::new(TranslationCache::default()),
            entity_memory: Mutex::new(ChapterEntityMemory::default()),
        }
    }

    fn resource_paths(&self) -> Result<ResidentResourcePaths> {
        ResidentResourcePaths::discover()
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
        let (resident, control) = tokio::try_join!(self.resident(), self.hsk_control())
            .map_err(CleaningError::pipeline)?;
        let preprocessing = global_preprocessing_pool().map_err(CleaningError::pipeline)?;
        cancellation_boundary(cancel.as_ref())?;

        let mut tiles = overlapping_tiles(image_width, image_height);
        let total_tiles = tiles.len();
        let total_tiles_u32 = u32::try_from(total_tiles).unwrap_or(u32::MAX);
        let mut processed_tiles = 0usize;
        let mut seen_text_blocks = Vec::<PixelRect>::new();
        let mut detected_bubbles = Vec::<DetectedBubble>::new();
        let mut recognized_lines = Vec::<RecognizedLine>::new();
        let mut text_probabilities = ProbabilityMap::zeros(image_width, image_height);
        let mut pending_translation = Vec::<PreparedRegion>::new();
        let mut translation_latency_phase = TranslationLatencyPhase::AwaitingFirstVisibleRegion;
        let mut repair_queue = RepairQueue::default();
        let mut dialogue_context = translation_context(&input.request);
        let mut prepared_next_tiles: Option<TileBatchTask> = None;

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
            let detector_priority = if admission_viewport.active
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
                }) {
                CudaPriority::Visible
            } else {
                CudaPriority::Offscreen
            };
            let cuda_permit = self
                .cuda_scheduler
                .acquire(detector_priority, cancel.clone())
                .await
                .map_err(cuda_admission_error)?;
            let detections = {
                let detector = resident.detector.lock().map_err(|_| {
                    CleaningError::new("MODEL_STATE_FAILED", "Detector lock poisoned.")
                })?;
                detector
                    .inference_tiles(&tile_images)
                    .context("run true-batched CUDA comic text detection")
                    .map_err(CleaningError::pipeline)?
            };
            {
                let text_segmenter = resident.text_segmenter.lock().map_err(|_| {
                    CleaningError::new("MODEL_STATE_FAILED", "Text segmenter lock poisoned.")
                })?;
                for (tile, tile_image) in tile_batch.iter().zip(&tile_images) {
                    let tile_probabilities = text_segmenter
                        .inference(tile_image)
                        .context("segment source text glyphs")
                        .map_err(CleaningError::pipeline)?;
                    merge_probability_map(
                        &mut text_probabilities,
                        &tile_probabilities,
                        tile.x,
                        tile.y,
                    );
                }
            }
            drop(cuda_permit);
            cancellation_boundary(cancel.as_ref())?;
            if detections.len() != tile_batch.len() {
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

            let candidates = detections
                .iter()
                .zip(&tile_batch)
                .flat_map(|(detection, tile)| {
                    candidates_for_tile(detection, tile, image_width, image_height)
                })
                .collect::<Vec<_>>();
            detected_bubbles.extend(
                detections
                    .iter()
                    .zip(&tile_batch)
                    .flat_map(|(detection, tile)| bubbles_for_tile(detection, tile)),
            );
            let mut candidates = spatially_dedupe(candidates, &seen_text_blocks);
            candidates.retain(text_candidate_is_confirmed);
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
            while !candidates.is_empty() {
                let accepted = ocr_batch(
                    resident,
                    source.clone(),
                    &mut candidates,
                    OcrProposalSource::Detector,
                    &input.request,
                    &sink,
                    cancel.clone(),
                    &self.cuda_scheduler,
                    &preprocessing,
                    &text_probabilities,
                )
                .await?;
                for line in accepted {
                    seen_text_blocks.push(line.candidate.text_rect);
                    recognized_lines.push(line);
                }
            }
            processed_tiles += tile_batch.len();
            let finalized_lines =
                take_finalized_lines(&mut recognized_lines, &tiles, image_width, image_height);
            if !finalized_lines.is_empty() {
                let (prepared_regions, probabilities) = prepare_grouped_regions(
                    resident,
                    source.clone(),
                    finalized_lines,
                    &input.request,
                    &sink,
                    cancel.clone(),
                    &self.cuda_scheduler,
                    &preprocessing,
                    text_probabilities,
                    overall,
                )
                .await?;
                text_probabilities = probabilities;
                pending_translation.extend(prepared_regions);
                if !tiles.is_empty() {
                    // Multi-tile pages remain progressive. On the final
                    // detector batch, wait for the fast segmentation fallbacks
                    // below so the semantic pass sees the complete page
                    // section instead of classifying an isolated cover title.
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
                        true,
                        false,
                    )
                    .await?;
                }
            }
            cancellation_boundary(cancel.as_ref())?;
        }

        let mut bubble_fallback_candidates = bubble_segmentation_fallback_candidates(
            &text_probabilities,
            image_width,
            image_height,
            &detected_bubbles,
            &seen_text_blocks,
        );
        if !bubble_fallback_candidates.is_empty() {
            publish_progress(
                &sink,
                BrowserJobStage::Ocr,
                None,
                Some(0.84),
                None,
                None,
                "Reading text from detected speech bubbles",
            )?;
        }
        while !bubble_fallback_candidates.is_empty() {
            let accepted = ocr_batch(
                resident,
                source.clone(),
                &mut bubble_fallback_candidates,
                OcrProposalSource::SegmentationFallback,
                &input.request,
                &sink,
                cancel.clone(),
                &self.cuda_scheduler,
                &preprocessing,
                &text_probabilities,
            )
            .await?;
            for line in accepted {
                seen_text_blocks.push(line.candidate.text_rect);
                recognized_lines.push(line);
            }
        }

        let mut fallback_candidates = segmentation_fallback_candidates(
            &text_probabilities,
            image_width,
            image_height,
            &seen_text_blocks,
        );
        if !fallback_candidates.is_empty() {
            publish_progress(
                &sink,
                BrowserJobStage::Ocr,
                None,
                Some(0.86),
                None,
                None,
                "Checking learned text regions the page detector did not cover",
            )?;
        }
        while !fallback_candidates.is_empty() {
            let accepted = ocr_batch(
                resident,
                source.clone(),
                &mut fallback_candidates,
                OcrProposalSource::SegmentationFallback,
                &input.request,
                &sink,
                cancel.clone(),
                &self.cuda_scheduler,
                &preprocessing,
                &text_probabilities,
            )
            .await?;
            for line in accepted {
                seen_text_blocks.push(line.candidate.text_rect);
                recognized_lines.push(line);
            }
        }

        let (prepared_regions, _text_probabilities) = prepare_grouped_regions(
            resident,
            source.clone(),
            recognized_lines,
            &input.request,
            &sink,
            cancel.clone(),
            &self.cuda_scheduler,
            &preprocessing,
            text_probabilities,
            0.88,
        )
        .await?;
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
        )
        .await?;
        cancellation_boundary(cancel.as_ref())?;
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
            let batch = pending.drain(..count).collect::<Vec<_>>();
            let batch_contains_visible = batch.iter().any(|region| region.visible);
            let published = self
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
                )
                .await?;
            if published && batch_contains_visible {
                *latency_phase = TranslationLatencyPhase::Throughput;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn translate_and_publish(
        &self,
        resident: &ResidentState,
        control: &HskControl,
        request: &BrowserJobRequest,
        mut regions: Vec<PreparedRegion>,
        cancel: Arc<AtomicBool>,
        sink: &JobUpdateSink,
        overall_progress: f32,
        image_width: u32,
        image_height: u32,
        context: &mut Vec<HskPrecedingUtterance>,
        repair_queue: &mut RepairQueue,
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
        publish_progress(
            sink,
            BrowserJobStage::Translating,
            None,
            Some(overall_progress),
            None,
            None,
            "Translating English directly into HSK-targeted Chinese",
        )?;
        let translator = resident.app.llm.direct_hsk_translator();
        let batch_context = context.clone();
        let mut protected_names = translation_glossary(request);
        let needs_name_analysis =
            request.settings.name_translation == NameTranslation::KeepOriginal;
        let mut semantic_sources = regions
            .iter()
            .map(|region| HskSourceUtterance {
                id: region.id.clone(),
                kind: hsk_utterance_kind(region.candidate.kind),
                source_english: region.source_english.clone(),
                semantic_layout: Some(HskSemanticLayout {
                    detector_enclosed: region.candidate.has_detector_core,
                    x0_millionths: normalized_millionths(
                        region.candidate.text_rect.x0,
                        image_width,
                    ),
                    y0_millionths: normalized_millionths(
                        region.candidate.text_rect.y0,
                        image_height,
                    ),
                    x1_millionths: normalized_millionths(
                        region.candidate.text_rect.x1,
                        image_width,
                    ),
                    y1_millionths: normalized_millionths(
                        region.candidate.text_rect.y1,
                        image_height,
                    ),
                    page_width: image_width,
                    page_height: image_height,
                }),
            })
            .collect::<Vec<_>>();
        if needs_name_analysis {
            let remembered = self
                .entity_memory
                .lock()
                .map_err(|_| {
                    CleaningError::new("ENTITY_MEMORY_FAILED", "Entity memory lock poisoned.")
                })?
                .names_for(&request.page_session_id);
            merge_protected_names(&mut protected_names, remembered);
        }
        let cuda_priority = prepared_region_priority(&regions, sink, image_width, image_height);
        cancellation_boundary(cancel.as_ref())?;
        let mut cuda_permit = Some(
            self.cuda_scheduler
                .acquire(cuda_priority, cancel.clone())
                .await
                .map_err(cuda_admission_error)?,
        );
        let semantic_classification = match tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(translator.classify_semantic_regions(&semantic_sources, cancel.as_ref()))
        }) {
            Ok(classifications) => classifications,
            Err(_) if cancel.load(Ordering::Acquire) || sink.is_cancelled() => {
                return Err(CleaningError::cancelled());
            }
            Err(error) => {
                return Err(CleaningError::pipeline(
                    error.context("run pre-translation semantic analysis"),
                ));
            }
        };
        if semantic_classification.page_is_furniture {
            return Ok(false);
        }
        let mut semantic_exclusion_ids = HashSet::<String>::new();
        for (id, disposition) in semantic_classification.regions {
            let Some(region) = regions.iter().find(|region| region.id == id) else {
                continue;
            };
            match semantic_exclusion_action(
                &region.candidate,
                disposition,
                request.settings.translate_sound_effects,
            ) {
                SemanticExclusionAction::Exclude => {
                    semantic_exclusion_ids.insert(id);
                }
                SemanticExclusionAction::Translate => {}
                SemanticExclusionAction::VerifyExclusion => {
                    let target = semantic_sources
                        .iter()
                        .find(|source| source.id == region.id)
                        .expect("semantic source was built from every region");
                    let verified = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(
                            translator.verify_excludable_region(
                                target,
                                &semantic_sources,
                                !request.settings.translate_sound_effects,
                                cancel.as_ref(),
                            ),
                        )
                    });
                    match verified {
                        Ok(true) => {
                            semantic_exclusion_ids.insert(id);
                        }
                        Ok(false) => {}
                        Err(_) if cancel.load(Ordering::Acquire) || sink.is_cancelled() => {
                            return Err(CleaningError::cancelled());
                        }
                        Err(error) => {
                            return Err(CleaningError::pipeline(
                                error.context("verify disputed non-story region"),
                            ));
                        }
                    }
                }
            }
        }
        regions.retain(|region| !semantic_exclusion_ids.contains(&region.id));
        if regions.is_empty() {
            return Ok(false);
        }
        let retained_ids = regions
            .iter()
            .map(|region| region.id.as_str())
            .collect::<HashSet<_>>();
        semantic_sources.retain(|source| retained_ids.contains(source.id.as_str()));
        if needs_name_analysis {
            let detected = match tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(translator.detect_proper_names(
                    &semantic_sources,
                    &protected_names,
                    cancel.as_ref(),
                ))
            }) {
                Ok(detected) => detected,
                Err(_) if cancel.load(Ordering::Acquire) || sink.is_cancelled() => {
                    return Err(CleaningError::cancelled());
                }
                Err(error) => {
                    return Err(CleaningError::pipeline(
                        error.context("run pre-translation proper-name analysis"),
                    ));
                }
            };
            if !detected.is_empty() {
                merge_protected_names(&mut protected_names, detected.clone());
                self.entity_memory
                    .lock()
                    .map_err(|_| {
                        CleaningError::new("ENTITY_MEMORY_FAILED", "Entity memory lock poisoned.")
                    })?
                    .remember(&request.page_session_id, &detected);
            }
        }
        let validator_names = control_proper_names(&protected_names);
        let name_handling = hsk_name_handling(request.settings.name_translation);
        let level = u8::from(request.settings.hsk_level);
        let control_level = ControlHskLevel::new(level)
            .map_err(|error| CleaningError::new("INVALID_HSK_LEVEL", error.to_string()))?;
        let mut keys = Vec::with_capacity(regions.len());
        let mut translated = vec![None::<CachedTranslation>; regions.len()];
        let mut missing_indices = Vec::new();
        let mut published = vec![false; regions.len()];
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
                    hsk_utterance_kind(regions[index].candidate.kind),
                    &batch_context,
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
            publish_region(
                sink,
                &regions[index],
                translation,
                request.settings.hsk_level,
                request.settings.learning_mode,
                control,
                image_width,
                image_height,
            )?;
            published[index] = true;
        }

        if !generation_indices.is_empty() {
            let utterances = generation_indices
                .iter()
                .map(|index| HskSourceUtterance {
                    id: regions[*index].id.clone(),
                    kind: hsk_utterance_kind(regions[*index].candidate.kind),
                    source_english: regions[*index].source_english.clone(),
                    semantic_layout: None,
                })
                .collect::<Vec<_>>();
            let index_by_id = generation_indices
                .iter()
                .map(|index| (regions[*index].id.clone(), *index))
                .collect::<HashMap<_, _>>();
            cancellation_boundary(cancel.as_ref())?;
            let cuda_permit = match cuda_permit.take() {
                Some(permit) => permit,
                None => self
                    .cuda_scheduler
                    .acquire(cuda_priority, cancel.clone())
                    .await
                    .map_err(cuda_admission_error)?,
            };
            let mut publish_streamed = |outcome: &HskTranslationOutcome| -> Result<()> {
                cancellation_boundary(cancel.as_ref()).map_err(anyhow::Error::new)?;
                let index = *index_by_id
                    .get(&outcome.id)
                    .with_context(|| format!("unknown streamed translation id {}", outcome.id))?;
                if translated[index].is_some() || states[index].is_some() {
                    return Ok(());
                }
                let outcome = normalize_preclassified_story_outcome(outcome.clone());
                let state = TranslationState::from_initial(
                    outcome,
                    control,
                    control_level,
                    &validator_names,
                    request.settings.learning_mode,
                );
                if let Some(mut translation) = state.initial_translation() {
                    populate_pinyin(control, &mut translation);
                    publish_region(
                        sink,
                        &regions[index],
                        translation.clone(),
                        request.settings.hsk_level,
                        request.settings.learning_mode,
                        control,
                        image_width,
                        image_height,
                    )
                    .map_err(anyhow::Error::new)?;
                    published[index] = true;
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
                        normalize_preclassified_story_outcome(outcome),
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
                publish_region(
                    sink,
                    &regions[index],
                    primary.clone(),
                    request.settings.hsk_level,
                    request.settings.learning_mode,
                    control,
                    image_width,
                    image_height,
                )?;
                published[index] = true;
                self.translation_cache
                    .lock()
                    .map_err(|_| {
                        CleaningError::new("CACHE_FAILED", "Translation cache lock poisoned.")
                    })?
                    .insert(keys[index].clone(), primary.clone());
                translated[index] = Some(primary);
            }
        }

        cancellation_boundary(cancel.as_ref())?;
        for (index, region) in regions.iter().enumerate() {
            if !published[index] {
                continue;
            }
            let Some(translation) = translated[index].as_ref() else {
                continue;
            };
            context.push(HskPrecedingUtterance {
                source_english: region.source_english.clone(),
                chinese: translation.displayed_chinese.clone(),
            });
        }
        if context.len() > MAX_HSK_PRECEDING_UTTERANCES {
            context.drain(..context.len() - MAX_HSK_PRECEDING_UTTERANCES);
        }

        let published_any = published.iter().any(|published| *published);
        for (index, region) in regions.into_iter().enumerate() {
            let Some(state) = states[index].take() else {
                continue;
            };
            if state.problems.is_empty() {
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
            let utterance = HskRepairUtterance {
                id: region.id.clone(),
                kind: hsk_utterance_kind(region.candidate.kind),
                source_english: region.source_english.clone(),
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
                published: published[index],
            });
        }
        Ok(published_any)
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
    ) -> std::result::Result<(), CleaningError> {
        if repair_queue.is_empty() {
            return Ok(());
        }
        cancellation_boundary(cancel.as_ref())?;
        publish_progress(
            sink,
            BrowserJobStage::HskValidating,
            None,
            Some(0.94),
            None,
            None,
            "Refining queued translations after all primary regions are published",
        )?;

        let translator = resident.app.llm.direct_hsk_translator();
        let name_handling = hsk_name_handling(request.settings.name_translation);
        let level = u8::from(request.settings.hsk_level);
        let control_level = ControlHskLevel::new(level)
            .map_err(|error| CleaningError::new("INVALID_HSK_LEVEL", error.to_string()))?;

        while !repair_queue.is_empty() {
            cancellation_boundary(cancel.as_ref())?;
            if sink.is_cancelled() {
                return Err(CleaningError::cancelled());
            }
            let mut jobs = repair_queue.take_batch(TRANSLATION_BATCH_MAX);
            for attempt in 0..MAX_TARGETED_REPAIR_ATTEMPTS {
                let active_indices = jobs
                    .iter()
                    .enumerate()
                    .filter_map(|(index, job)| (!job.state.repair_succeeded()).then_some(index))
                    .collect::<Vec<_>>();
                if active_indices.is_empty() {
                    break;
                }
                let mut batch_names = Vec::<HskProtectedName>::new();
                let mut utterances = Vec::<HskRepairUtterance>::with_capacity(active_indices.len());
                for &index in &active_indices {
                    let job = &jobs[index];
                    merge_protected_names(&mut batch_names, repair_names_for_job(job));
                    utterances.push(if attempt == 0 {
                        job.utterance.clone()
                    } else {
                        let mut problems = if job.state.problems.is_empty() {
                            vec![
                                "the previous automatic repair was not safe to publish; return one complete Simplified Chinese line"
                                    .to_owned(),
                            ]
                        } else {
                            job.state.problems.clone()
                        };
                        problems.push(repair_convergence_instruction(
                            repair_generation_mode(request.settings.learning_mode, attempt),
                            request.settings.hsk_level,
                            attempt,
                        ));
                        HskRepairUtterance {
                            id: job.region.id.clone(),
                            kind: hsk_utterance_kind(job.region.candidate.kind),
                            source_english: job.region.source_english.clone(),
                            rejected_chinese: job.state.rejected_for_repair(),
                            avoid_chinese: job.state.avoid_chinese(),
                            problems,
                        }
                    });
                }
                // Primary work for this image has already drained. Offscreen
                // admission also lets visible primary work from another job pass.
                let cuda_permit = self
                    .cuda_scheduler
                    .acquire(CudaPriority::Offscreen, cancel.clone())
                    .await
                    .map_err(cuda_admission_error)?;
                cancellation_boundary(cancel.as_ref())?;
                let repair_result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(translator.repair_invalid_batch(
                        &HskTranslationRepairBatchRequest {
                            requested_level: level,
                            learning_mode: hsk_learning_mode(repair_generation_mode(
                                request.settings.learning_mode,
                                attempt,
                            )),
                            name_handling,
                            translate_sound_effects: true,
                            utterances,
                            preceding_utterances: Vec::new(),
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
                            let repair_names = repair_names_for_job(job);
                            job.state.apply_repair(
                                outcome,
                                control,
                                control_level,
                                &control_proper_names(&repair_names),
                            );
                        }
                    }
                    Err(_) if cancel.load(Ordering::Acquire) || sink.is_cancelled() => {
                        return Err(CleaningError::cancelled());
                    }
                    Err(error) => {
                        eprintln!(
                            "hskify: batched HSK repair attempt {} failed for {} regions: {error:#}",
                            attempt + 1,
                            active_indices.len(),
                        );
                        for index in active_indices {
                            jobs[index].state.reject_failed_repair();
                        }
                    }
                }
                cancellation_boundary(cancel.as_ref())?;
            }
            for job in jobs {
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

                if job.published {
                    // A published primary must always receive a terminal
                    // refinement. If the bounded repairs still contain only
                    // unavoidable HSK-level wording, `finish` keeps the meaningful
                    // primary and marks it rejected instead of leaving the browser
                    // indefinitely pending.
                    publish_refinement(
                        sink,
                        &job.region.id,
                        &result,
                        request.settings.hsk_level,
                        request.settings.learning_mode,
                        control,
                    )?;
                } else {
                    publish_region(
                        sink,
                        &job.region,
                        result.clone(),
                        request.settings.hsk_level,
                        request.settings.learning_mode,
                        control,
                        image_width,
                        image_height,
                    )?;
                }

                self.translation_cache
                    .lock()
                    .map_err(|_| {
                        CleaningError::new("CACHE_FAILED", "Translation cache lock poisoned.")
                    })?
                    .insert(job.cache_key, result);
            }
        }
        Ok(())
    }
}

#[async_trait]
impl CleaningPipeline for KoharuPipeline {
    async fn warm_up(&self) -> std::result::Result<(), CleaningError> {
        tokio::try_join!(self.resident(), self.hsk_control())
            .map(|_| ())
            .map_err(CleaningError::pipeline)
    }

    async fn run(
        &self,
        input: CleaningInput,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), CleaningError> {
        self.run_direct(input, cancel, sink).await
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
    ocr: Mutex<EnglishPpOcrV5>,
    text_segmenter: Mutex<MangaTextSegmentation>,
    bubble_segmenter: Mutex<SpeechBubbleSegmentation>,
    inpainter: Mutex<Lama>,
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
        let text_segmenter_weights = resources.path(TEXT_SEGMENTER_WEIGHTS_ID)?.to_path_buf();
        let bubble_segmenter_config = resources.path(BUBBLE_SEGMENTER_CONFIG_ID)?.to_path_buf();
        let bubble_segmenter_weights = resources.path(BUBBLE_SEGMENTER_WEIGHTS_ID)?.to_path_buf();
        let inpainter_weights = resources.path(INPAINTER_WEIGHTS_ID)?.to_path_buf();
        let translation_model = resources.path(TRANSLATION_MODEL_ID)?.to_path_buf();
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
            tokio::task::spawn_blocking(move || EnglishPpOcrV5::load(&ocr_model, &ocr_config))
                .await
                .context("join resident English PP-OCRv5 loader")?
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
            translation_model,
            BROWSER_QWEN_INFERENCE_THREADS,
        );
        let (detector, ocr, (text_segmenter, bubble_segmenter, inpainter), ()) = tokio::try_join!(
            detector_future,
            ocr_future,
            cleanup_models_future,
            llm_future
        )
        .context("load resident CUDA models")?;
        Ok(Self {
            app,
            detector: Mutex::new(detector),
            ocr: Mutex::new(ocr),
            text_segmenter: Mutex::new(text_segmenter),
            bubble_segmenter: Mutex::new(bubble_segmenter),
            inpainter: Mutex::new(inpainter),
        })
    }
}

fn utf8_path(path: PathBuf) -> Result<Utf8PathBuf> {
    Utf8PathBuf::from_path_buf(path).map_err(|path| anyhow!("path is not valid UTF-8: {path:?}"))
}

#[derive(Debug)]
struct PreparedRegion {
    id: String,
    candidate: Candidate,
    source_english: String,
    ocr_confidence: f32,
    reading_order: u32,
    prediction: PpOcrPrediction,
    appearance_bands: Vec<SourceAppearanceBand>,
    measured_font_height: f32,
    patch: PatchPng,
    bubble_polygon: Vec<Point>,
    layout_polygon: Vec<Point>,
    visible: bool,
    translation_queued_at: tokio::time::Instant,
}

#[derive(Debug)]
struct RecognizedLine {
    candidate: Candidate,
    prediction: PpOcrPrediction,
    crop_bounds: PixelBounds,
}

#[derive(Debug)]
struct GroupedRegion {
    candidate: Candidate,
    source_english: String,
    ocr_confidence: f32,
    prediction: PpOcrPrediction,
    appearance_bands: Vec<SourceAppearanceBand>,
    measured_font_height: f32,
    cleanup_blocks: Vec<TextRegion>,
}

#[derive(Debug)]
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
    published: bool,
}

fn repair_names_for_job(job: &PendingRepair) -> Vec<HskProtectedName> {
    job.protected_names.clone()
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
) -> std::result::Result<Vec<RecognizedLine>, CleaningError> {
    let (image_width, image_height) = source.dimensions();
    if candidates.is_empty() {
        return Ok(Vec::new());
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
                        source_for_crops.crop_imm(bounds.x, bounds.y, bounds.width, bounds.height),
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
        .acquire(cuda_priority, cancel.clone())
        .await
        .map_err(cuda_admission_error)?;
    let predictions = {
        let mut ocr = resident
            .ocr
            .lock()
            .map_err(|_| CleaningError::new("MODEL_STATE_FAILED", "OCR model lock poisoned."))?;
        ocr.recognize_regions(&crops, &crop_text_probabilities)
            .context("run batched CUDA English PP-OCRv5 recognition")
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
    Ok(candidate_chunk
        .into_iter()
        .zip(predictions)
        .zip(crop_bounds)
        .filter(|((candidate, prediction), _)| {
            let accepted = accept_english_ocr_line(
                prediction.confidence,
                &prediction.text,
                proposal_source,
            );
            if !accepted && rejected_ocr_tracing_enabled() {
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
            accepted
        })
        .map(|((candidate, prediction), crop_bounds)| RecognizedLine {
            candidate,
            prediction,
            crop_bounds,
        })
        .collect())
}

async fn prepare_grouped_regions(
    resident: &ResidentState,
    source: Arc<DynamicImage>,
    lines: Vec<RecognizedLine>,
    request: &BrowserJobRequest,
    sink: &JobUpdateSink,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    preprocessing: &Arc<PreprocessingPool>,
    text_probabilities: ProbabilityMap,
    overall_progress: f32,
) -> std::result::Result<(Vec<PreparedRegion>, ProbabilityMap), CleaningError> {
    if lines.is_empty() {
        return Ok((Vec::new(), text_probabilities));
    }
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
            cleanup_supports
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
    let bubble_permit = cuda_scheduler
        .acquire(bubble_priority, cancel.clone())
        .await
        .map_err(cuda_admission_error)?;
    let mut bubble_union = image::GrayImage::new(image_width, image_height);
    {
        let bubble_segmenter = resident.bubble_segmenter.lock().map_err(|_| {
            CleaningError::new("MODEL_STATE_FAILED", "Bubble segmenter lock poisoned.")
        })?;
        for (tile, crop) in cleanup_tiles.iter().zip(&cleanup_crops) {
            let result = bubble_segmenter
                .inference(crop)
                .context("segment speech bubble contours")
                .map_err(CleaningError::pipeline)?;
            merge_binary_mask(&mut bubble_union, &bubble_id_mask(&result), tile.x, tile.y);
        }
    }
    drop(bubble_permit);
    let bubble_mask = label_bubble_components(&bubble_union);
    let groups = group_recognized_lines(lines, &bubble_mask);
    let grouped = preprocessing
        .run(move || {
            Ok(groups
                .into_iter()
                .map(|group| {
                    let source_english = grouped_source_english(&group);
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
                    GroupedRegion {
                        candidate,
                        source_english,
                        ocr_confidence,
                        prediction,
                        appearance_bands,
                        measured_font_height,
                        cleanup_blocks,
                    }
                })
                .collect::<Vec<_>>())
        })
        .await
        .context("group recognized dialogue on the browser preprocessing pool")
        .map_err(CleaningError::pipeline)?;

    cancellation_boundary(cancel.as_ref())?;
    publish_progress(
        sink,
        BrowserJobStage::Inpainting,
        None,
        Some(overall_progress),
        None,
        None,
        "Restoring the artwork behind the original text",
    )?;
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
    drop(cleanup_crops);
    drop(cleanup_tiles);
    let source_for_cleanup = source.clone();
    let (cleaned_groups, erase_mask, text_blocks, bubble_mask, text_probabilities) = preprocessing
        .run(move || {
            let bubble_image = DynamicImage::ImageLuma8(bubble_mask.clone());
            let mut erase_mask = image::GrayImage::new(image_width, image_height);
            let mut all_text_blocks = Vec::new();
            let mut cleaned_groups = Vec::with_capacity(grouped.len());
            for group in grouped {
                let learned_mask = verified_text_mask_for_regions(
                    source_for_cleanup.as_ref(),
                    &text_probabilities,
                    &bubble_mask,
                    &group.cleanup_blocks,
                    DEFAULT_TEXT_MASK_THRESHOLD,
                )
                .with_context(|| {
                    format!(
                        "learned text mask did not cover every OCR line in {:?}",
                        group.source_english
                    )
                })?;
                let group_erase_mask = expand_mask_for_inpainting(
                    &DynamicImage::ImageLuma8(learned_mask),
                    &bubble_image,
                    &group.cleanup_blocks,
                );
                let support = group
                    .candidate
                    .confirmed_bubble_rect
                    .union(group.candidate.text_rect);
                let cleanup_mask =
                    compact_cleanup_mask(&group_erase_mask, support).with_context(|| {
                        format!(
                            "expanded cleanup mask was empty for OCR-confirmed dialogue {:?}",
                            group.source_english
                        )
                    })?;
                merge_cleanup_mask(&mut erase_mask, &cleanup_mask);
                all_text_blocks.extend(group.cleanup_blocks.iter().cloned());
                cleaned_groups.push(CleanedGroupedRegion {
                    group,
                    cleanup_mask,
                });
            }
            Ok((
                cleaned_groups,
                erase_mask,
                all_text_blocks,
                bubble_mask,
                text_probabilities,
            ))
        })
        .await
        .context("verify and isolate learned cleanup masks by bubble")
        .map_err(CleaningError::pipeline)?;
    cancellation_boundary(cancel.as_ref())?;
    let cuda_permit = cuda_scheduler
        .acquire(priority, cancel.clone())
        .await
        .map_err(cuda_admission_error)?;
    let inpainted = resident
        .inpainter
        .lock()
        .map_err(|_| CleaningError::new("MODEL_STATE_FAILED", "Inpainter lock poisoned."))?
        .inference_with_blocks(
            source.as_ref(),
            &DynamicImage::ImageLuma8(erase_mask.clone()),
            &DynamicImage::ImageLuma8(bubble_mask.clone()),
            &text_blocks,
        )
        .context("restore artwork with the manga inpainter")
        .map_err(CleaningError::pipeline)?;
    drop(cuda_permit);
    cancellation_boundary(cancel.as_ref())?;

    let prepared_groups = preprocessing
        .run(move || {
            let bubble_components = bubble_component_bounds(&bubble_mask);
            cleaned_groups
                .into_iter()
                .map(|cleaned| {
                    let group = cleaned.group;
                    let patch = make_inpainted_patch(&inpainted, &cleaned.cleanup_mask)?;
                    let (bubble_polygon, layout_polygon) = region_polygons(
                        &bubble_mask,
                        &bubble_components,
                        group.candidate.text_rect,
                        group.candidate.confirmed_bubble_rect,
                        group.measured_font_height,
                    );
                    Ok((
                        group.candidate,
                        group.source_english,
                        group.ocr_confidence,
                        group.prediction,
                        group.appearance_bands,
                        group.measured_font_height,
                        patch,
                        bubble_polygon,
                        layout_polygon,
                    ))
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .context("encode model-inpainted cleanup patches")
        .map_err(CleaningError::pipeline)?;
    let latest_viewport = sink.viewport();
    let translation_queued_at = tokio::time::Instant::now();
    Ok((
        prepared_groups
            .into_iter()
            .map(
                |(
                    candidate,
                    source_english,
                    ocr_confidence,
                    prediction,
                    appearance_bands,
                    measured_font_height,
                    patch,
                    bubble_polygon,
                    layout_polygon,
                )| {
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
                        prediction,
                        appearance_bands,
                        measured_font_height,
                        patch,
                        bubble_polygon,
                        layout_polygon,
                        visible,
                        translation_queued_at,
                    }
                },
            )
            .collect(),
        text_probabilities,
    ))
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
    groups.into_iter().map(|(_, lines)| lines).collect()
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
            return left
                .candidate
                .confirmed_bubble_rect
                .contains_point(right.candidate.text_rect.center());
        }
        (false, true) => {
            return right
                .candidate
                .confirmed_bubble_rect
                .contains_point(left.candidate.text_rect.center());
        }
        (false, false) => return true,
        (true, true) => {}
    }
    let left_core = left.candidate.confirmed_bubble_rect;
    let right_core = right.candidate.confirmed_bubble_rect;
    left_core.intersection(right_core).is_some()
        && left_core.contains_point(right_core.center())
        && right_core.contains_point(left_core.center())
        && left_core.contains_point(right.candidate.text_rect.center())
        && right_core.contains_point(left.candidate.text_rect.center())
}

fn grouped_source_english(group: &[RecognizedLine]) -> String {
    group
        .iter()
        .map(|line| compact_ocr_text(&line.prediction.text))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
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
        .filter_map(|band| {
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
                .map(move |band| source_appearance_band(line.crop_bounds, band, group_text_rect))
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

fn normalized_millionths(coordinate: f32, extent: u32) -> u32 {
    if extent == 0 {
        return 0;
    }
    ((coordinate / extent as f32).clamp(0.0, 1.0) * 1_000_000.0).round() as u32
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
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OcrProposalSource {
    Detector,
    SegmentationFallback,
}

fn accept_english_ocr_line(
    confidence: f32,
    text: &str,
    proposal_source: OcrProposalSource,
) -> bool {
    if !confidence.is_finite() || confidence < MIN_OCR_CONFIDENCE {
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
    let minimum_alphabetic = match proposal_source {
        // A detector-backed one-letter utterance such as "I" is valid story
        // text. A fallback-only one-letter proposal is too ambiguous with
        // punctuation glyphs and must have additional lexical evidence.
        OcrProposalSource::Detector => 1,
        OcrProposalSource::SegmentationFallback => 2,
    };
    alphabetic.len() >= minimum_alphabetic
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SemanticExclusionAction {
    Exclude,
    VerifyExclusion,
    Translate,
}

fn semantic_exclusion_action(
    candidate: &Candidate,
    disposition: HskTranslationDisposition,
    translate_sound_effects: bool,
) -> SemanticExclusionAction {
    match disposition {
        HskTranslationDisposition::ExcludeSoundEffect
            if !translate_sound_effects
                && candidate.kind == CandidateKind::FreeText
                && !candidate.has_detector_core =>
        {
            SemanticExclusionAction::Exclude
        }
        HskTranslationDisposition::ExcludeNonStory
            if candidate.kind == CandidateKind::FreeText && !candidate.has_detector_core =>
        {
            SemanticExclusionAction::Exclude
        }
        HskTranslationDisposition::ExcludeNonStory => SemanticExclusionAction::VerifyExclusion,
        HskTranslationDisposition::ExcludeSoundEffect if !translate_sound_effects => {
            SemanticExclusionAction::VerifyExclusion
        }
        HskTranslationDisposition::ExcludeSoundEffect | HskTranslationDisposition::Translate => {
            SemanticExclusionAction::Translate
        }
    }
}

fn normalize_preclassified_story_outcome(
    mut outcome: HskTranslationOutcome,
) -> HskTranslationOutcome {
    if !outcome.is_non_story() {
        return outcome;
    }

    // The dedicated pre-translation semantic pass is authoritative. Once a
    // region survives it, the translation model cannot independently remove
    // that story text; a conflicting marker becomes a repairable missing
    // translation.
    outcome.disposition = HskTranslationDisposition::Translate;
    outcome.issues.push(HskTranslationIssue::MissingLine);
    outcome
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

#[derive(Default)]
struct ChapterEntityMemory {
    sessions: VecDeque<(String, Vec<HskProtectedName>)>,
}

impl ChapterEntityMemory {
    fn names_for(&mut self, page_session_id: &str) -> Vec<HskProtectedName> {
        let Some(index) = self
            .sessions
            .iter()
            .position(|(session, _)| session == page_session_id)
        else {
            return Vec::new();
        };
        let entry = self
            .sessions
            .remove(index)
            .expect("known entity-memory index exists");
        let names = entry.1.clone();
        self.sessions.push_back(entry);
        names
    }

    fn remember(&mut self, page_session_id: &str, detected: &[HskProtectedName]) {
        let mut names = self.names_for(page_session_id);
        for candidate in detected {
            if names.iter().any(|existing| {
                existing
                    .source_english
                    .eq_ignore_ascii_case(&candidate.source_english)
            }) {
                continue;
            }
            if names.len() == ENTITY_MEMORY_MAX_NAMES_PER_SESSION {
                break;
            }
            names.push(candidate.clone());
        }
        if let Some(index) = self
            .sessions
            .iter()
            .position(|(session, _)| session == page_session_id)
        {
            self.sessions.remove(index);
        }
        self.sessions.push_back((page_session_id.to_owned(), names));
        while self.sessions.len() > ENTITY_MEMORY_MAX_SESSIONS {
            self.sessions.pop_front();
        }
    }
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

    fn rejected_for_repair(&self) -> Option<String> {
        self.latest_rejected_chinese
            .clone()
            .or_else(|| self.base_chinese.clone())
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
    let mut violations = Vec::<(&str, Vec<&str>)>::new();
    for violation in &report.violations {
        let token = violation.text.trim();
        if token.is_empty() {
            continue;
        }
        let entry = if let Some(entry) = violations.iter_mut().find(|entry| entry.0 == token) {
            entry
        } else {
            violations.push((token, Vec::new()));
            violations
                .last_mut()
                .expect("a violation entry was just inserted")
        };
        for suggestion in &violation.suggested_words {
            let suggestion = suggestion.trim();
            if !suggestion.is_empty() && !entry.1.contains(&suggestion) {
                entry.1.push(suggestion);
            }
        }
    }
    if violations.is_empty() {
        return;
    }
    violations.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let guidance = violations
        .into_iter()
        .take(16)
        .map(|(token, suggestions)| {
            if suggestions.is_empty() {
                format!("`{token}`: rewrite the surrounding phrase with allowed everyday words")
            } else {
                format!(
                    "`{token}`: possible level-safe words [{}]; choose only what preserves the source meaning, or rewrite the phrase",
                    suggestions.join(", ")
                )
            }
        })
        .collect::<Vec<_>>()
        .join("; ");
    let problem = format!(
        "rewrite all invalid or above-level wording with easier grammar and vocabulary; deterministic HSK guidance: {guidance}"
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

fn repair_convergence_instruction(mode: LearningMode, level: HskLevel, attempt: usize) -> String {
    let target = match u8::from(level) {
        1..=3 => 90,
        4 => 93,
        _ => 95,
    };
    let term_limit = match u8::from(level) {
        1..=3 => 1,
        4..=5 => 2,
        _ => 3,
    };
    let strategy = match attempt {
        1 => "Replace the listed terms by paraphrasing their meanings with everyday words.",
        2 => "Rebuild the answer as short, direct clauses instead of editing the previous wording.",
        _ => "Use the simplest complete clause structure that preserves every source detail.",
    };
    match mode {
        LearningMode::Natural => format!(
            "The previous answer still failed Natural learning. Reach at least {target}% \
            level-appropriate lexical occurrences and keep at most {term_limit} listed \
            above-level occurrence in the whole line. {strategy}"
        ),
        LearningMode::Strict => format!(
            "The previous answer still failed Strict HSK. Keep none of the listed invalid \
            vocabulary or grammar. {strategy}"
        ),
    }
}

fn repair_generation_mode(requested: LearningMode, _attempt: usize) -> LearningMode {
    requested
}

fn translation_context(request: &BrowserJobRequest) -> Vec<HskPrecedingUtterance> {
    let context = request.preceding_context.as_deref().unwrap_or_default();
    let start = context.len().saturating_sub(MAX_HSK_PRECEDING_UTTERANCES);
    context[start..]
        .iter()
        .map(|utterance| HskPrecedingUtterance {
            source_english: utterance.source_english.clone(),
            chinese: utterance.chinese.clone(),
        })
        .collect()
}

fn translation_glossary(request: &BrowserJobRequest) -> Vec<HskProtectedName> {
    request
        .proper_name_glossary
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|name| HskProtectedName {
            source_english: name.source_english.clone(),
            chinese: match request.settings.name_translation {
                NameTranslation::KeepOriginal => name.source_english.clone(),
                NameTranslation::Chinese => name.chinese.clone(),
            },
        })
        .collect()
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
    let patch_rect = region.patch.bounds.normalized(image_width, image_height);
    let patch = sink
        .store_patch_png(patch_rect, region.patch.bytes.clone())
        .map_err(|error| publish_error(error, sink))?;
    let above_level_tokens = above_level_tokens(&translation.report);
    let teaching_terms = teaching_terms(control, &translation.report);
    let progressive = ProgressiveRegion {
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
        style,
        layout,
        hsk: ProgressiveHskStatus {
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
        region: Box::new(progressive),
    })
    .map_err(|error| publish_error(error, sink))?;
    Ok(())
}

fn publish_refinement(
    sink: &JobUpdateSink,
    region_id: &str,
    translation: &CachedTranslation,
    requested_level: HskLevel,
    learning_mode: LearningMode,
    control: &HskControl,
) -> std::result::Result<(), CleaningError> {
    sink.publish(JobUpdateDraft::RegionRefined {
        region_id: region_id.to_owned(),
        displayed_chinese: translation.displayed_chinese.clone(),
        pinyin: translation.pinyin.clone(),
        hsk: ProgressiveHskStatus {
            requested_level,
            learning_mode,
            strictly_valid: translation.report.strictly_valid,
            level_coverage: level_coverage(&translation.report),
            above_level_tokens: above_level_tokens(&translation.report),
            teaching_terms: teaching_terms(control, &translation.report),
            repair_state: translation.repair_state,
        },
    })
    .map_err(|error| publish_error(error, sink))?;
    sink.refine_region_for_lookup(
        region_id,
        translation.displayed_chinese.clone(),
        translation
            .report
            .exceptions
            .iter()
            .map(|exception| ProperName {
                text: exception.text.clone(),
                reason: exception.reason,
            })
            .collect(),
    );
    Ok(())
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
    let suggested_line_count = color_bands.len().max(1);
    (
        BrowserTextStyle {
            font_id: "hmt-sans".to_owned(),
            category: FontCategory::Sans,
            foreground,
            weight: 600,
            italic_degrees: 0.0,
            outline_color,
            outline_width_ratio,
            shadow_color: None,
            shadow_x_ratio: 0.0,
            shadow_y_ratio: 0.0,
            alignment: TextAlignment::Center,
            writing_mode: WritingMode::HorizontalTb,
            line_height: 1.15,
            letter_spacing_em: 0.0,
            color_bands,
        },
        BrowserTextLayout {
            suggested_lines: suggested_lines(displayed_chinese, suggested_line_count),
            font_size_to_image_width: (region.measured_font_height / image_width.max(1) as f32)
                .clamp(0.002, 0.25),
            safe_polygon: Some(bubble_polygon),
        },
    )
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
    fn repair_feedback_groups_repeated_vocabulary_violations() {
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
        assert!(problems[0].contains("deterministic HSK guidance:"));
        assert_eq!(problems[0].matches("高级").count(), 1);
        assert!(problems[0].contains("possible level-safe words [学生]"));
        assert!(problems[0].contains("choose only what preserves the source meaning"));
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

    #[test]
    fn repair_convergence_changes_strategy_and_states_the_acceptance_boundary() {
        let first_retry = repair_convergence_instruction(LearningMode::Natural, HskLevel::Three, 1);
        let second_retry =
            repair_convergence_instruction(LearningMode::Natural, HskLevel::Three, 2);
        let strict = repair_convergence_instruction(LearningMode::Strict, HskLevel::Three, 1);

        assert!(first_retry.contains("at least 90%"));
        assert!(first_retry.contains("at most 1"));
        assert!(first_retry.contains("paraphrasing their meanings"));
        assert!(second_retry.contains("short, direct clauses"));
        assert_ne!(first_retry, second_retry);
        assert!(strict.contains("Keep none"));
        assert_eq!(
            repair_generation_mode(LearningMode::Natural, 0),
            LearningMode::Natural
        );
        assert_eq!(
            repair_generation_mode(LearningMode::Natural, 1),
            LearningMode::Natural
        );
        assert_eq!(
            repair_generation_mode(LearningMode::Strict, 2),
            LearningMode::Strict
        );
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
                },
                source_english: "Graduate student".to_owned(),
                ocr_confidence: 0.99,
                reading_order: 0,
                prediction: PpOcrPrediction {
                    text: "Graduate student".to_owned(),
                    confidence: 0.99,
                    text_color: [0, 0, 0],
                    stroke_color: [255, 255, 255],
                    has_stroke_color: false,
                    appearance_bands: Vec::new(),
                },
                appearance_bands: Vec::new(),
                measured_font_height: rect.height(),
                patch: PatchPng {
                    bounds: geometry::PixelBounds {
                        x: 1,
                        y: 1,
                        width: 8,
                        height: 8,
                    },
                    bytes: vec![1, 2, 3],
                },
                bubble_polygon: rect.polygon(10, 10),
                layout_polygon: rect.polygon(10, 10),
                visible: false,
                translation_queued_at: tokio::time::Instant::now(),
            },
            utterance: HskRepairUtterance {
                id: id.to_owned(),
                kind: HskUtteranceKind::Dialogue,
                source_english: "Graduate student".to_owned(),
                rejected_chinese: Some("研究生".to_owned()),
                avoid_chinese: vec!["研究生".to_owned()],
                problems: vec!["above level".to_owned()],
            },
            protected_names: Vec::new(),
            state: usable_pending_state(),
            published: true,
        }
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
        assert!(!accept_english_ocr_line(
            0.99,
            "I",
            OcrProposalSource::SegmentationFallback
        ));
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
        };
        let prediction = || PpOcrPrediction {
            text: "line".to_owned(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
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
        };
        let make_line = |text_rect, core, crop_x| RecognizedLine {
            candidate: Candidate {
                kind: CandidateKind::StoryText,
                text_rect,
                bubble_rect: core,
                confirmed_bubble_rect: core,
                detector_confidence: 0.95,
                has_detector_core: true,
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
            },
            prediction: PpOcrPrediction {
                text: "line".to_owned(),
                confidence: 0.95,
                text_color: [0, 0, 0],
                stroke_color: [255, 255, 255],
                has_stroke_color: false,
                appearance_bands: Vec::new(),
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
    fn grouping_attaches_fallback_text_only_when_it_is_inside_the_detector_bubble() {
        let core = PixelRect::new(20.0, 20.0, 190.0, 140.0).unwrap();
        let prediction = || PpOcrPrediction {
            text: "line".to_owned(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
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
        };
        let prediction = || PpOcrPrediction {
            text: String::new(),
            confidence: 0.95,
            text_color: [0, 0, 0],
            stroke_color: [255, 255, 255],
            has_stroke_color: false,
            appearance_bands: Vec::new(),
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
    fn cache_key_separates_original_and_chinese_name_preferences() {
        let key = |name_translation| {
            translation_cache_key(
                "Alice is here",
                HskUtteranceKind::Dialogue,
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
                    source_english: "Alice".to_owned(),
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
    fn semantic_region_disposition_requires_model_and_topology_agreement() {
        let candidate = |kind, has_detector_core| Candidate {
            kind,
            text_rect: PixelRect::new(10.0, 10.0, 100.0, 40.0).unwrap(),
            bubble_rect: PixelRect::new(5.0, 5.0, 105.0, 45.0).unwrap(),
            confirmed_bubble_rect: PixelRect::new(5.0, 5.0, 105.0, 45.0).unwrap(),
            detector_confidence: 0.95,
            has_detector_core,
        };

        assert_eq!(
            semantic_exclusion_action(
                &candidate(CandidateKind::FreeText, false),
                HskTranslationDisposition::ExcludeNonStory,
                false,
            ),
            SemanticExclusionAction::Exclude
        );
        assert_eq!(
            semantic_exclusion_action(
                &candidate(CandidateKind::StoryText, true),
                HskTranslationDisposition::ExcludeNonStory,
                false,
            ),
            SemanticExclusionAction::VerifyExclusion
        );
        assert_eq!(
            semantic_exclusion_action(
                &candidate(CandidateKind::StoryText, true),
                HskTranslationDisposition::ExcludeSoundEffect,
                false,
            ),
            SemanticExclusionAction::VerifyExclusion
        );
        assert_eq!(
            semantic_exclusion_action(
                &candidate(CandidateKind::FreeText, false),
                HskTranslationDisposition::ExcludeSoundEffect,
                false,
            ),
            SemanticExclusionAction::Exclude
        );
        assert_eq!(
            semantic_exclusion_action(
                &candidate(CandidateKind::FreeText, false),
                HskTranslationDisposition::ExcludeSoundEffect,
                true,
            ),
            SemanticExclusionAction::Translate
        );
        assert_eq!(
            semantic_exclusion_action(
                &candidate(CandidateKind::StoryText, true),
                HskTranslationDisposition::Translate,
                false,
            ),
            SemanticExclusionAction::Translate
        );
        let conflicted = normalize_preclassified_story_outcome(HskTranslationOutcome {
            id: "story".to_owned(),
            disposition: HskTranslationDisposition::ExcludeNonStory,
            text: None,
            issues: Vec::new(),
        });
        assert_eq!(conflicted.disposition, HskTranslationDisposition::Translate);
        assert_eq!(conflicted.issues, [HskTranslationIssue::MissingLine]);
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
    fn usable_above_level_primary_is_cacheable_while_its_repair_is_pending() {
        let primary = usable_pending_state().initial_translation().unwrap();
        assert_eq!(primary.displayed_chinese, "研究生");
        assert_eq!(primary.repair_state, HskRepairState::Pending);

        let mut cache = TranslationCache::default();
        cache.insert("primary-key".to_owned(), primary);
        let cached = cache.get("primary-key").unwrap();
        assert_eq!(cached.displayed_chinese, "研究生");
        assert_eq!(cached.repair_state, HskRepairState::Pending);
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
        assert_eq!(worse_state.rejected_for_repair().as_deref(), Some("教授"));
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
