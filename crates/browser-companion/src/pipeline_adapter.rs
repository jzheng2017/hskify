//! Direct, progressive browser pipeline.
//!
//! The browser path deliberately does not create Koharu projects or materialize
//! whole cleaned images. It decodes the upload once, runs resident CUDA models
//! over overlapping tiles, and publishes one transparent cleanup patch per
//! translated dialogue region.

mod geometry;
mod patch;
mod ppocr_v5;
mod ppocr_v5_detector;

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
    ProperName, ProperNameReason, ValidationReport,
};
use image::{DynamicImage, GenericImageView};
use koharu_app::llm::{
    HSK_TRANSLATION_MODEL, HskPrecedingUtterance, HskProtectedName, HskRepairUtterance,
    HskSourceUtterance, HskTranslationBatchRequest, HskTranslationOutcome,
    HskTranslationRepairRequest, HskUtteranceKind, MAX_HSK_PRECEDING_UTTERANCES,
};
use koharu_app::{App, AppConfig};
use koharu_runtime::{ComputePolicy, RuntimeManager};
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use tokio::sync::{OnceCell, oneshot};

use self::geometry::{
    Candidate, CandidateKind, PixelBounds, PixelRect, Tile, candidates_for_tile, ocr_crop_rect,
    overlapping_tiles, prioritize_tiles, reading_order_key, spatially_dedupe,
    text_candidate_is_confirmed,
};
use self::patch::{PatchPng, PlacedInkMask, make_cleanup_patch};
use self::ppocr_v5::{EnglishPpOcrV5, MAX_LINE_BATCH_SIZE, PpOcrPrediction};
use self::ppocr_v5_detector::{DETECTOR_TILE_BATCH_SIZE, PpOcrV5TextDetector};
use crate::contracts::{
    BrowserJobRequest, BrowserJobStage, BrowserTextLayout, BrowserTextStyle, FontCategory,
    HskLevel, HskRepairState, LookupRegion, LookupResult, LookupToken, NormalizedRect,
    ProgressiveHskStatus, ProgressiveRegion, ReadingDirection, TextAlignment, WritingMode,
};
use crate::crypto::sha256_hex;
use crate::cuda_scheduler::{
    CudaAdmissionError, CudaPriority, CudaScheduler, global_cuda_scheduler,
};
use crate::server::{JobUpdateDraft, JobUpdateSink};
use crate::setup::{
    DETECTOR_MODEL_ID, OCR_CONFIG_ID, OCR_MODEL_ID, ResidentResourcePaths, TRANSLATION_MODEL_ID,
};

const OCR_REGION_BATCH_SIZE: usize = MAX_LINE_BATCH_SIZE;
const TRANSLATION_BATCH_MAX: usize = 6;
const TRANSLATION_BATCH_MIN: usize = 3;
const TRANSLATION_MAX_FLUSH_DELAY: Duration = Duration::from_millis(75);
const BROWSER_QWEN_INFERENCE_THREADS: i32 = 6;
const MIN_OCR_CONFIDENCE: f32 = 0.45;
const TRANSLATION_CACHE_SCHEMA: &str = "hskify-direct-hsk-region-cache-v4";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RegionLookupContext {
    pub(crate) source_english: String,
    pub(crate) base_chinese: String,
    pub(crate) displayed_chinese: String,
    pub(crate) proper_names: Vec<ProperName>,
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
}

#[async_trait]
pub(crate) trait CleaningPipeline: Send + Sync {
    async fn run(
        &self,
        input: CleaningInput,
        cancel: Arc<AtomicBool>,
        sink: JobUpdateSink,
    ) -> std::result::Result<(), CleaningError>;

    async fn lookup(
        &self,
        selected_text: String,
        region: Option<RegionLookupContext>,
    ) -> std::result::Result<LookupResult, CleaningError>;

    fn resources_ready(&self) -> bool;
}

pub(crate) struct KoharuPipeline {
    cache_root: PathBuf,
    cuda_scheduler: Arc<CudaScheduler>,
    resident: OnceCell<Arc<ResidentState>>,
    hsk_control: OnceCell<Arc<HskControl>>,
    resources: std::result::Result<ResidentResourcePaths, String>,
    translation_cache: Mutex<TranslationCache>,
}

impl KoharuPipeline {
    pub(crate) fn new(cache_root: PathBuf) -> Self {
        Self {
            cache_root,
            cuda_scheduler: global_cuda_scheduler(),
            resident: OnceCell::new(),
            hsk_control: OnceCell::new(),
            resources: ResidentResourcePaths::discover().map_err(|error| format!("{error:#}")),
            translation_cache: Mutex::new(TranslationCache::default()),
        }
    }

    fn resource_paths(&self) -> Result<&ResidentResourcePaths> {
        self.resources.as_ref().map_err(|error| anyhow!("{error}"))
    }

    async fn resident(&self) -> Result<&Arc<ResidentState>> {
        let resources = self.resource_paths()?.clone();
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
        let resources = self.resource_paths()?.clone();
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
        let mut pending_translation = Vec::<PreparedRegion>::new();
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
            let take = DETECTOR_TILE_BATCH_SIZE.min(tiles.len());
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
                let next_count = DETECTOR_TILE_BATCH_SIZE.min(tiles.len());
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
                let mut detector = resident.detector.lock().map_err(|_| {
                    CleaningError::new("MODEL_STATE_FAILED", "Detector lock poisoned.")
                })?;
                detector
                    .detect_tiles(&tile_images)
                    .context("run true-batched CUDA PP-OCRv5 text detection")
                    .map_err(CleaningError::pipeline)?
            };
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
            let mut candidates = spatially_dedupe(candidates, &seen_text_blocks);
            candidates.retain(text_candidate_is_confirmed);
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
            let mut recognized_lines = Vec::new();
            while !candidates.is_empty() {
                let accepted = ocr_batch(
                    resident,
                    source.clone(),
                    &mut candidates,
                    &input.request,
                    &sink,
                    cancel.clone(),
                    &self.cuda_scheduler,
                    &preprocessing,
                )
                .await?;
                for line in accepted {
                    seen_text_blocks.push(line.candidate.text_rect);
                    recognized_lines.push(line);
                }
            }
            let prepared_regions = prepare_grouped_regions(
                source.clone(),
                recognized_lines,
                &input.request,
                &sink,
                &preprocessing,
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
                overall,
                image_width,
                image_height,
                &mut dialogue_context,
                &mut repair_queue,
                false,
            )
            .await?;
            processed_tiles += tile_batch.len();
            cancellation_boundary(cancel.as_ref())?;
        }

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
        force: bool,
    ) -> std::result::Result<(), CleaningError> {
        while !pending.is_empty() {
            prioritize_pending_translation(pending, sink, image_width, image_height);
            let count = match translation_boundary_action(
                pending,
                force,
                tokio::time::Instant::now(),
                cancel.load(Ordering::Acquire) || sink.is_cancelled(),
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
            self.translate_and_publish(
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
        }
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
    ) -> std::result::Result<(), CleaningError> {
        if regions.is_empty() {
            return Ok(());
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
        let protected_names = translation_glossary(request);
        let validator_names = control_proper_names(&protected_names);
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
                    states[index] = cached.clone().map(TranslationState::from_cached);
                }
                translated[index] = cached;
                if translated[index].is_none() {
                    missing_indices.push(index);
                }
                keys.push(key);
            }
        }
        let generation_indices = primary_generation_indices(&translated);

        for (index, translation) in translated.iter().enumerate() {
            let Some(translation) = translation.clone() else {
                continue;
            };
            publish_region(
                sink,
                &regions[index],
                translation,
                request.settings.hsk_level,
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
                })
                .collect::<Vec<_>>();
            let index_by_id = generation_indices
                .iter()
                .map(|index| (regions[*index].id.clone(), *index))
                .collect::<HashMap<_, _>>();
            cancellation_boundary(cancel.as_ref())?;
            let cuda_priority = prepared_region_priority(&regions, sink, image_width, image_height);
            let cuda_permit = self
                .cuda_scheduler
                .acquire(cuda_priority, cancel.clone())
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
                );
                if let Some(mut translation) = state.initial_translation() {
                    populate_pinyin(control, &mut translation);
                    publish_region(
                        sink,
                        &regions[index],
                        translation.clone(),
                        request.settings.hsk_level,
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
                        outcome,
                        control,
                        control_level,
                        &validator_names,
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

        for (index, region) in regions.into_iter().enumerate() {
            let Some(state) = states[index].take() else {
                continue;
            };
            if state.problems.is_empty() {
                continue;
            }
            let utterance = HskRepairUtterance {
                id: region.id.clone(),
                kind: hsk_utterance_kind(region.candidate.kind),
                source_english: region.source_english.clone(),
                rejected_chinese: state.base_chinese.clone(),
                problems: state.problems.clone(),
            };
            repair_queue.enqueue(PendingRepair {
                cache_key: keys[index].clone(),
                region,
                utterance,
                preceding_utterances: batch_context.clone(),
                state,
                published: published[index],
            });
        }
        Ok(())
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
        let protected_names = translation_glossary(request);
        let validator_names = control_proper_names(&protected_names);
        let level = u8::from(request.settings.hsk_level);
        let control_level = ControlHskLevel::new(level)
            .map_err(|error| CleaningError::new("INVALID_HSK_LEVEL", error.to_string()))?;

        while let Some(mut job) = repair_queue.next() {
            cancellation_boundary(cancel.as_ref())?;
            if sink.is_cancelled() {
                return Err(CleaningError::cancelled());
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
                tokio::runtime::Handle::current().block_on(translator.repair_invalid_item(
                    &HskTranslationRepairRequest {
                        requested_level: level,
                        utterance: job.utterance.clone(),
                        preceding_utterances: job.preceding_utterances.clone(),
                        protected_names: protected_names.clone(),
                    },
                    cancel.as_ref(),
                ))
            });
            drop(cuda_permit);
            let repaired = match repair_result {
                Ok(repaired) => repaired,
                Err(_) if cancel.load(Ordering::Acquire) || sink.is_cancelled() => {
                    return Err(CleaningError::cancelled());
                }
                Err(error) => {
                    return Err(CleaningError::pipeline(
                        error.context("run one targeted HSK bubble repair"),
                    ));
                }
            };
            cancellation_boundary(cancel.as_ref())?;

            let accepted =
                job.state
                    .apply_repair(repaired, control, control_level, &validator_names);
            let publishable = job.state.can_publish();
            let Ok(mut result) = job.state.finish() else {
                eprintln!(
                    "hskify: translation produced no publishable Chinese for region {} after one repair",
                    job.region.id
                );
                continue;
            };
            populate_pinyin(control, &mut result);
            cancellation_boundary(cancel.as_ref())?;

            if job.published {
                if accepted {
                    publish_refinement(sink, &job.region.id, &result, request.settings.hsk_level)?;
                }
            } else if publishable {
                publish_region(
                    sink,
                    &job.region,
                    result.clone(),
                    request.settings.hsk_level,
                    image_width,
                    image_height,
                )?;
            } else {
                eprintln!(
                    "hskify: translation rejected for region {} after one repair: {}",
                    job.region.id,
                    job.utterance.problems.join("; ")
                );
                continue;
            }

            self.translation_cache
                .lock()
                .map_err(|_| {
                    CleaningError::new("CACHE_FAILED", "Translation cache lock poisoned.")
                })?
                .insert(job.cache_key, result);
        }
        Ok(())
    }
}

#[async_trait]
impl CleaningPipeline for KoharuPipeline {
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
        selected_text: String,
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
        Ok(browser_lookup_result(control.lookup_with_region_context(
            &selected_text,
            &proper_names,
            context,
        )))
    }

    fn resources_ready(&self) -> bool {
        self.resident.get().is_some() && self.hsk_control.get().is_some()
    }
}

struct ResidentState {
    app: Arc<App>,
    detector: Mutex<PpOcrV5TextDetector>,
    ocr: Mutex<EnglishPpOcrV5>,
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
        let detector_model = resources.path(DETECTOR_MODEL_ID)?.to_path_buf();
        let ocr_config = resources.path(OCR_CONFIG_ID)?.to_path_buf();
        let ocr_model = resources.path(OCR_MODEL_ID)?.to_path_buf();
        let translation_model = resources.path(TRANSLATION_MODEL_ID)?.to_path_buf();
        let detector_future = async move {
            tokio::task::spawn_blocking(move || PpOcrV5TextDetector::load(&detector_model))
                .await
                .context("join resident PP-OCRv5 detector loader")?
        };
        let ocr_future = async move {
            tokio::task::spawn_blocking(move || EnglishPpOcrV5::load(&ocr_model, &ocr_config))
                .await
                .context("join resident English PP-OCRv5 loader")?
        };
        let llm_future = app.llm.load_local_file_with_threads(
            HSK_TRANSLATION_MODEL,
            translation_model,
            BROWSER_QWEN_INFERENCE_THREADS,
        );
        let (detector, ocr, ()) = tokio::try_join!(detector_future, ocr_future, llm_future)
            .context("load resident CUDA models")?;
        Ok(Self {
            app,
            detector: Mutex::new(detector),
            ocr: Mutex::new(ocr),
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
    patch: PatchPng,
    visible: bool,
    translation_queued_at: tokio::time::Instant,
}

#[derive(Debug)]
struct RecognizedLine {
    candidate: Candidate,
    prediction: PpOcrPrediction,
    crop_bounds: PixelBounds,
}

struct PendingRepair {
    cache_key: String,
    region: PreparedRegion,
    utterance: HskRepairUtterance,
    preceding_utterances: Vec<HskPrecedingUtterance>,
    state: TranslationState,
    published: bool,
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

    fn next(&mut self) -> Option<PendingRepair> {
        if self.primary_phase_complete {
            self.jobs.pop_front()
        } else {
            None
        }
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
    request: &BrowserJobRequest,
    sink: &JobUpdateSink,
    cancel: Arc<AtomicBool>,
    cuda_scheduler: &Arc<CudaScheduler>,
    preprocessing: &Arc<PreprocessingPool>,
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
        ocr.recognize_regions(&crops)
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
            let _ = candidate.kind;
            accept_english_ocr_line(prediction.confidence, &prediction.text)
        })
        .map(|((candidate, prediction), crop_bounds)| RecognizedLine {
            candidate,
            prediction,
            crop_bounds,
        })
        .collect())
}

async fn prepare_grouped_regions(
    source: Arc<DynamicImage>,
    lines: Vec<RecognizedLine>,
    request: &BrowserJobRequest,
    sink: &JobUpdateSink,
    preprocessing: &Arc<PreprocessingPool>,
) -> std::result::Result<Vec<PreparedRegion>, CleaningError> {
    if lines.is_empty() {
        return Ok(Vec::new());
    }
    let (image_width, image_height) = source.dimensions();
    let groups = group_recognized_lines(lines, request.settings.reading_direction);
    let source_for_patches = source.clone();
    let prepared_groups = preprocessing
        .run(move || {
            let grouped_sources = groups
                .iter()
                .map(|group| grouped_source_english(group))
                .collect::<Vec<_>>();
            let credit_context = grouped_sources
                .iter()
                .any(|text| looks_like_credit_context(text));
            groups
                .into_iter()
                .zip(grouped_sources)
                .map(|(group, source_english)| {
                    let candidate = merge_group_candidate(&group, image_width, image_height);
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
                    if !accept_english_ocr(group[0].candidate.kind, ocr_confidence, &source_english)
                        || (credit_context && looks_like_isolated_credit_name(&source_english))
                    {
                        return Ok(None);
                    }
                    let mut prediction = group
                        .iter()
                        .max_by_key(|line| line.prediction.text.chars().count())
                        .expect("recognized group is non-empty")
                        .prediction
                        .clone();
                    let inks = group
                        .iter()
                        .filter_map(|line| {
                            line.prediction.ink_mask.as_ref().map(|mask| PlacedInkMask {
                                mask,
                                crop_bounds: line.crop_bounds,
                            })
                        })
                        .collect::<Vec<_>>();
                    let patch = make_cleanup_patch(
                        source_for_patches.as_ref(),
                        candidate.text_rect,
                        candidate.confirmed_bubble_rect,
                        &inks,
                    )?;
                    prediction.text.clone_from(&source_english);
                    prediction.confidence = ocr_confidence;
                    prediction.ink_mask = None;
                    Ok(Some((
                        candidate,
                        source_english,
                        ocr_confidence,
                        prediction,
                        patch,
                    )))
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .context("pack cleanup patches on the browser preprocessing pool")
        .map_err(CleaningError::pipeline)?;
    let latest_viewport = sink.viewport();
    let translation_queued_at = tokio::time::Instant::now();
    Ok(prepared_groups
        .into_iter()
        .flatten()
        .map(
            |(candidate, source_english, ocr_confidence, prediction, patch)| {
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
                    patch,
                    visible,
                    translation_queued_at,
                }
            },
        )
        .collect())
}

fn grouped_source_english(group: &[RecognizedLine]) -> String {
    group
        .iter()
        .map(|line| compact_ocr_text(&line.prediction.text))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn group_recognized_lines(
    mut lines: Vec<RecognizedLine>,
    reading_direction: ReadingDirection,
) -> Vec<Vec<RecognizedLine>> {
    lines.sort_by(|left, right| {
        left.candidate
            .text_rect
            .y0
            .total_cmp(&right.candidate.text_rect.y0)
            .then_with(|| match reading_direction {
                ReadingDirection::Rtl => right
                    .candidate
                    .text_rect
                    .x0
                    .total_cmp(&left.candidate.text_rect.x0),
                ReadingDirection::Auto | ReadingDirection::Ltr => left
                    .candidate
                    .text_rect
                    .x0
                    .total_cmp(&right.candidate.text_rect.x0),
            })
    });
    let mut groups = Vec::<Vec<RecognizedLine>>::new();
    for line in lines {
        let destination = groups
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                group.len() < 8
                    && group
                        .last()
                        .is_some_and(|previous| lines_belong_to_same_block(previous, &line))
            })
            .min_by(|(_, left), (_, right)| {
                let left_gap = line.candidate.text_rect.y0
                    - left
                        .last()
                        .expect("filtered group is non-empty")
                        .candidate
                        .text_rect
                        .y1;
                let right_gap = line.candidate.text_rect.y0
                    - right
                        .last()
                        .expect("filtered group is non-empty")
                        .candidate
                        .text_rect
                        .y1;
                left_gap.total_cmp(&right_gap)
            })
            .map(|(index, _)| index);
        if let Some(index) = destination {
            groups[index].push(line);
        } else {
            groups.push(vec![line]);
        }
    }
    groups
}

fn lines_belong_to_same_block(previous: &RecognizedLine, next: &RecognizedLine) -> bool {
    let upper = previous.candidate.text_rect;
    let lower = next.candidate.text_rect;
    let smaller_height = upper.height().min(lower.height()).max(1.0);
    let larger_height = upper.height().max(lower.height()).max(1.0);
    let vertical_gap = lower.y0 - upper.y1;
    if vertical_gap < -smaller_height * 0.35 || vertical_gap > (larger_height * 2.2).max(24.0) {
        return false;
    }
    if previous
        .prediction
        .text
        .trim_end()
        .ends_with(['.', '!', '?', '…'])
        && vertical_gap > larger_height * 0.8
    {
        return false;
    }
    let horizontal_overlap = (upper.x1.min(lower.x1) - upper.x0.max(lower.x0)).max(0.0);
    let overlap_ratio = horizontal_overlap / upper.width().min(lower.width()).max(1.0);
    let (upper_center, _) = upper.center();
    let (lower_center, _) = lower.center();
    let center_distance = (upper_center - lower_center).abs();
    let horizontally_aligned =
        overlap_ratio >= 0.20 || center_distance <= upper.width().max(lower.width()) * 0.30;
    horizontally_aligned
        && color_distance(previous.prediction.text_color, next.prediction.text_color) <= 180
}

fn color_distance(left: [u8; 3], right: [u8; 3]) -> u16 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| u16::from(left.abs_diff(right)))
        .sum()
}

fn merge_group_candidate(
    group: &[RecognizedLine],
    image_width: u32,
    image_height: u32,
) -> Candidate {
    let first = group.first().expect("recognized group is non-empty");
    let text_rect = group
        .iter()
        .skip(1)
        .fold(first.candidate.text_rect, |rect, line| {
            rect.union(line.candidate.text_rect)
        });
    let bubble_rect = group
        .iter()
        .skip(1)
        .fold(first.candidate.bubble_rect, |rect, line| {
            rect.union(line.candidate.bubble_rect)
        });
    let layout_padding = (text_rect.height().min(text_rect.width()) * 0.08).clamp(4.0, 20.0);
    let layout_rect =
        bubble_rect.union(text_rect.expand(layout_padding, image_width, image_height));
    Candidate {
        kind: first.candidate.kind,
        text_rect,
        bubble_rect: layout_rect,
        confirmed_bubble_rect: layout_rect,
        detector_confidence: group
            .iter()
            .map(|line| line.candidate.detector_confidence)
            .fold(0.0, f32::max),
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
    regions.sort_by(|left, right| {
        right.visible.cmp(&left.visible).then_with(|| {
            if left.visible {
                left.reading_order.cmp(&right.reading_order)
            } else {
                left.translation_queued_at
                    .cmp(&right.translation_queued_at)
                    .then_with(|| left.reading_order.cmp(&right.reading_order))
            }
        })
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
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

const NARRATION_SFX_WORDS: &[&str] = &[
    "ah", "bam", "bang", "boom", "bump", "clang", "clank", "crack", "crash", "ding", "gasp", "ha",
    "haha", "hiss", "kaboom", "knock", "oh", "pow", "ring", "slam", "snap", "splash", "swoosh",
    "thud", "thump", "ugh", "wham", "whoosh", "zap",
];

fn accept_english_ocr(_kind: CandidateKind, confidence: f32, text: &str) -> bool {
    if !accept_english_ocr_line(confidence, text) {
        return false;
    }
    let text = text.trim();
    let lowercase = text.to_lowercase();
    let words = lowercase
        .split(|character: char| !is_latin_letter(character))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    !contains_credit_or_release_metadata(&words)
        && !looks_like_long_ocr_gibberish(text)
        && !looks_like_low_confidence_ocr_gibberish(confidence, text, &words)
        && is_confident_english_story_text(text)
}

fn accept_english_ocr_line(confidence: f32, text: &str) -> bool {
    if !confidence.is_finite() || confidence < MIN_OCR_CONFIDENCE {
        return false;
    }
    let text = text.trim();
    if text.is_empty()
        || text.contains('\u{fffd}')
        || text.to_ascii_uppercase().contains("<UNK>")
        || text.chars().any(char::is_control)
        || looks_like_compact_alphanumeric_noise(text)
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

fn looks_like_compact_alphanumeric_noise(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.chars().any(char::is_whitespace)
        || trimmed
            .chars()
            .any(|character| !character.is_ascii_alphanumeric())
    {
        return false;
    }
    let letters = trimmed
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .count();
    let digits = trimmed
        .chars()
        .filter(|character| character.is_ascii_digit())
        .count();
    trimmed.len() <= 10 && letters >= 3 && digits >= 2
}

fn looks_like_long_ocr_gibberish(text: &str) -> bool {
    const STRUCTURAL_ENGLISH_WORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did", "do", "does", "for",
        "from", "had", "has", "have", "he", "her", "him", "his", "i", "if", "in", "is", "it", "me",
        "my", "not", "of", "on", "or", "she", "that", "the", "their", "them", "they", "this", "to",
        "us", "was", "we", "were", "with", "you", "your",
    ];
    let lowercase = text.to_lowercase();
    let words = lowercase
        .split(|character: char| !is_latin_letter(character))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let structural_words = words
        .iter()
        .filter(|word| STRUCTURAL_ENGLISH_WORDS.contains(word))
        .count();
    words.len() >= 8 && structural_words.saturating_mul(5) < words.len()
}

fn looks_like_low_confidence_ocr_gibberish(confidence: f32, text: &str, words: &[&str]) -> bool {
    if confidence >= 0.75 {
        return false;
    }
    const STRUCTURAL_ENGLISH_WORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did", "do", "does", "for",
        "from", "had", "has", "have", "he", "her", "him", "his", "i", "if", "in", "is", "it", "me",
        "my", "not", "of", "on", "or", "she", "that", "the", "their", "them", "they", "this", "to",
        "us", "was", "we", "were", "with", "you", "your",
    ];
    let structural_words = words
        .iter()
        .filter(|word| STRUCTURAL_ENGLISH_WORDS.contains(word))
        .count();
    (words.len() >= 4 && structural_words == 0)
        || (confidence < 0.70
            && words.len() <= 2
            && words.iter().any(|word| word.chars().count() >= 6)
            && !text.trim_end().ends_with(['.', '!', '?', '…']))
}

fn is_confident_english_story_text(text: &str) -> bool {
    let lowercase = text.to_lowercase();
    if lowercase.contains("http")
        || lowercase.contains("www.")
        || lowercase.contains(".com")
        || lowercase.contains(".net")
        || lowercase.contains('@')
    {
        return false;
    }
    let compact_identifier = text
        .trim()
        .chars()
        .all(|character| character.is_ascii_alphanumeric())
        && text
            .chars()
            .filter(|character| character.is_ascii_alphabetic())
            .count()
            >= 2
        && text
            .chars()
            .filter(|character| character.is_ascii_digit())
            .count()
            >= 2;
    if compact_identifier {
        return true;
    }
    let words = lowercase
        .split(|character: char| !is_latin_letter(character))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if words.is_empty()
        || words.iter().all(|word| NARRATION_SFX_WORDS.contains(word))
        || looks_like_short_non_english_ocr(&words)
        || looks_like_fragmented_metadata_ocr(&words)
        || contains_credit_or_release_metadata(&words)
    {
        return false;
    }
    if words.len() == 1 {
        return text.trim_end().ends_with(['.', '!', '?', '…']);
    }
    is_short_quoted_narration(text, &words)
        || words.len() >= 2
        || text.trim_end().ends_with(['.', '!', '?', '…'])
}

fn looks_like_short_non_english_ocr(words: &[&str]) -> bool {
    const SHORT_ENGLISH_PHRASES: &[(&str, &str)] = &[
        ("do", "it"),
        ("go", "on"),
        ("i", "am"),
        ("i", "do"),
        ("i", "go"),
        ("is", "it"),
        ("it", "is"),
        ("no", "it"),
        ("to", "me"),
        ("we", "do"),
        ("we", "go"),
    ];
    words.len() == 2
        && words.iter().all(|word| word.chars().count() <= 2)
        && !SHORT_ENGLISH_PHRASES.contains(&(words[0], words[1]))
}

fn is_short_quoted_narration(text: &str, words: &[&str]) -> bool {
    if words.len() < 2 {
        return false;
    }
    let trimmed = text.trim();
    let Some(opening) = trimmed.chars().next() else {
        return false;
    };
    let closing = match opening {
        '"' => '"',
        '\'' => '\'',
        '“' => '”',
        '‘' => '’',
        _ => return false,
    };
    let Some(inner) = trimmed
        .strip_prefix(opening)
        .and_then(|value| value.strip_suffix(closing))
    else {
        return false;
    };
    inner
        .trim_end()
        .chars()
        .last()
        .is_some_and(|character| matches!(character, '.' | '!' | '?' | '…'))
}

fn contains_credit_or_release_metadata(words: &[&str]) -> bool {
    const STRONG_METADATA_WORDS: &[&str] = &[
        "chapter",
        "chapters",
        "copyright",
        "credits",
        "discord",
        "edited",
        "editor",
        "episode",
        "episodes",
        "lettered",
        "letterer",
        "patreon",
        "prologue",
        "proofread",
        "proofreader",
        "redraw",
        "redrawer",
        "release",
        "releases",
        "scanlated",
        "scanlation",
        "scans",
        "season",
        "translated",
        "translation",
        "translator",
        "typeset",
        "typesetter",
        "volume",
    ];
    const PRODUCTION_CREDIT_WORDS: &[&str] = &[
        "art",
        "background",
        "color",
        "colour",
        "directing",
        "editing",
        "effect",
        "effects",
        "line",
        "original",
        "sketch",
        "story",
        "text",
        "words",
    ];
    STRONG_METADATA_WORDS
        .iter()
        .any(|metadata| words.contains(metadata))
        || words
            .iter()
            .filter(|word| PRODUCTION_CREDIT_WORDS.contains(word))
            .take(2)
            .count()
            >= 2
        || words.windows(2).any(|pair| {
            matches!(
                pair,
                ["art" | "story" | "words", "by"] | ["fastest", "releases"] | ["read", "at"]
            )
        })
}

fn looks_like_credit_context(text: &str) -> bool {
    let lowercase = text.to_lowercase();
    let words = lowercase
        .split(|character: char| !is_latin_letter(character))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    contains_credit_or_release_metadata(&words)
        || lowercase.contains("discord")
        || lowercase.contains("www.")
        || lowercase.contains(".com")
        || lowercase.contains(".net")
}

fn looks_like_isolated_credit_name(text: &str) -> bool {
    const STRUCTURAL_ENGLISH_WORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "of", "on", "or",
        "the", "to", "with",
    ];
    let trimmed = text.trim();
    if trimmed.ends_with(['.', '!', '?', '…'])
        || trimmed.chars().any(char::is_lowercase)
        || trimmed
            .chars()
            .any(|character| !(character.is_alphabetic() || character.is_whitespace()))
    {
        return false;
    }
    let lowercase = trimmed.to_lowercase();
    let words = lowercase
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    (2..=4).contains(&words.len())
        && words.iter().all(|word| word.chars().count() >= 3)
        && !words
            .iter()
            .any(|word| STRUCTURAL_ENGLISH_WORDS.contains(word))
}

fn looks_like_fragmented_metadata_ocr(words: &[&str]) -> bool {
    words.len() >= 5
        && words
            .iter()
            .filter(|word| word.chars().count() <= 2)
            .count()
            * 5
            >= words.len() * 3
}

fn hsk_utterance_kind(kind: CandidateKind) -> HskUtteranceKind {
    match kind {
        CandidateKind::StoryText => HskUtteranceKind::Dialogue,
    }
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
    report: Option<ValidationReport>,
    problems: Vec<String>,
    meaning_valid: bool,
    repair_state: HskRepairState,
}

impl TranslationState {
    fn from_initial(
        outcome: HskTranslationOutcome,
        control: &HskControl,
        level: ControlHskLevel,
        proper_names: &[ProperName],
    ) -> Self {
        let excluded = outcome.is_excluded();
        let mut problems = outcome.repair_problems();
        let meaning_valid = outcome.issues.is_empty();
        let base_chinese = nonempty_translation(outcome.text);
        let report = base_chinese
            .as_deref()
            .map(|text| control.validate(text, level, proper_names));
        if let Some(report) = &report {
            append_validation_problems(&mut problems, report);
        }
        let valid = !excluded && problems.is_empty() && report.is_some();
        Self {
            displayed_chinese: valid.then(|| {
                report
                    .as_ref()
                    .expect("valid translation has a validation report")
                    .normalized_text
                    .clone()
            }),
            base_chinese,
            report,
            problems,
            meaning_valid,
            repair_state: HskRepairState::NotNeeded,
        }
    }

    fn from_cached(translation: CachedTranslation) -> Self {
        let mut problems = Vec::new();
        append_validation_problems(&mut problems, &translation.report);
        Self {
            base_chinese: Some(translation.base_chinese),
            displayed_chinese: Some(translation.displayed_chinese),
            report: Some(translation.report),
            problems,
            meaning_valid: true,
            repair_state: translation.repair_state,
        }
    }

    fn can_publish(&self) -> bool {
        self.meaning_valid && self.base_chinese.is_some() && self.report.is_some()
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
        if let Some(report) = &report {
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

    fn finish(mut self) -> Result<CachedTranslation> {
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
        text: None,
        issues: vec![HskTranslationIssue::MissingLine],
    }
}

fn append_validation_problems(problems: &mut Vec<String>, report: &ValidationReport) {
    for violation in &report.violations {
        let problem = format!(
            "replace invalid or above-level token `{}` at {}..{}",
            violation.text, violation.start_char, violation.end_char
        );
        if !problems.contains(&problem) {
            problems.push(problem);
        }
    }
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
            chinese: name.chinese.clone(),
        })
        .collect()
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
fn primary_generation_indices<T>(cache_results: &[Option<T>]) -> Vec<usize> {
    if cache_results.iter().all(Option::is_some) {
        Vec::new()
    } else {
        (0..cache_results.len()).collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn translation_cache_key(
    source_english: &str,
    kind: HskUtteranceKind,
    context: &[HskPrecedingUtterance],
    protected_names: &[HskProtectedName],
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
    image_width: u32,
    image_height: u32,
) -> std::result::Result<(), CleaningError> {
    let text_polygon = region
        .candidate
        .text_rect
        .polygon(image_width, image_height);
    let layout_polygon = region
        .candidate
        .bubble_rect
        .polygon(image_width, image_height);
    let bubble_polygon = None;
    let (style, layout) = style_and_layout(
        &region,
        &translation.displayed_chinese,
        image_width,
        layout_polygon,
    );
    let patch_rect = region.patch.bounds.normalized(image_width, image_height);
    let patch = sink
        .store_patch_png(patch_rect, region.patch.bytes.clone())
        .map_err(|error| publish_error(error, sink))?;
    let above_level_tokens = above_level_tokens(&translation.report);
    let progressive = ProgressiveRegion {
        id: region.id.clone(),
        text_polygon,
        bubble_polygon,
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
            strictly_valid: translation.report.strictly_valid,
            above_level_tokens,
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
) -> std::result::Result<(), CleaningError> {
    sink.publish(JobUpdateDraft::RegionRefined {
        region_id: region_id.to_owned(),
        displayed_chinese: translation.displayed_chinese.clone(),
        pinyin: translation.pinyin.clone(),
        hsk: ProgressiveHskStatus {
            requested_level,
            strictly_valid: translation.report.strictly_valid,
            above_level_tokens: above_level_tokens(&translation.report),
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
        },
        BrowserTextLayout {
            suggested_lines: suggested_lines(displayed_chinese),
            font_size_to_image_width: (region.candidate.text_rect.height() * 0.65
                / image_width.max(1) as f32)
                .clamp(0.002, 0.25),
            safe_polygon: Some(bubble_polygon),
        },
    )
}

fn suggested_lines(text: &str) -> Vec<String> {
    let characters = text.chars().collect::<Vec<_>>();
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

fn translation_boundary_action(
    pending: &[PreparedRegion],
    force: bool,
    now: tokio::time::Instant,
    cancelled: bool,
) -> TranslationBoundaryAction {
    if cancelled {
        return TranslationBoundaryAction::Cancelled;
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
        ValidationReport {
            normalized_text: text.to_owned(),
            requested_level: ControlHskLevel::new(1).unwrap(),
            strictly_valid: violations.is_empty(),
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

    fn usable_pending_state() -> TranslationState {
        let report = validation_report("研究生", vec![above_level_violation("研究生")]);
        TranslationState {
            base_chinese: Some("研究生".to_owned()),
            displayed_chinese: Some("研究生".to_owned()),
            report: Some(report),
            problems: vec!["replace invalid or above-level token `研究生` at 0..3".to_owned()],
            meaning_valid: true,
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
                },
                source_english: "Graduate student".to_owned(),
                ocr_confidence: 0.99,
                reading_order: 0,
                prediction: PpOcrPrediction {
                    text: "Graduate student".to_owned(),
                    confidence: 0.99,
                    ink_mask: None,
                    text_color: [0, 0, 0],
                    stroke_color: [255, 255, 255],
                    has_stroke_color: false,
                },
                patch: PatchPng {
                    bounds: geometry::PixelBounds {
                        x: 1,
                        y: 1,
                        width: 8,
                        height: 8,
                    },
                    bytes: vec![1, 2, 3],
                },
                visible: false,
                translation_queued_at: tokio::time::Instant::now(),
            },
            utterance: HskRepairUtterance {
                id: id.to_owned(),
                kind: HskUtteranceKind::Dialogue,
                source_english: "Graduate student".to_owned(),
                rejected_chinese: Some("研究生".to_owned()),
                problems: vec!["above level".to_owned()],
            },
            preceding_utterances: Vec::new(),
            state: usable_pending_state(),
            published: true,
        }
    }

    fn recognized_line(text: &str, rect: PixelRect, color: [u8; 3]) -> RecognizedLine {
        RecognizedLine {
            candidate: Candidate {
                kind: CandidateKind::StoryText,
                text_rect: rect,
                bubble_rect: rect,
                confirmed_bubble_rect: rect,
                detector_confidence: 0.99,
            },
            prediction: PpOcrPrediction {
                text: text.to_owned(),
                confidence: 0.99,
                ink_mask: None,
                text_color: color,
                stroke_color: [0, 0, 0],
                has_stroke_color: false,
            },
            crop_bounds: PixelBounds {
                x: rect.x0 as u32,
                y: rect.y0 as u32,
                width: rect.width() as u32,
                height: rect.height() as u32,
            },
        }
    }

    #[test]
    fn adjacent_same_color_lines_form_one_story_text_block() {
        let lines = vec![
            recognized_line(
                "AND BECAME THE HERO",
                PixelRect::new(100.0, 100.0, 500.0, 140.0).unwrap(),
                [250, 250, 250],
            ),
            recognized_line(
                "WHO BROUGHT AN END",
                PixelRect::new(130.0, 148.0, 470.0, 188.0).unwrap(),
                [248, 249, 250],
            ),
        ];

        let groups = group_recognized_lines(lines, ReadingDirection::Ltr);

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn colored_emphasis_and_side_by_side_bubbles_stay_separate() {
        let lines = vec![
            recognized_line(
                "THE PRETEXT OF",
                PixelRect::new(180.0, 100.0, 500.0, 140.0).unwrap(),
                [250, 250, 250],
            ),
            recognized_line(
                "ASSASSINATION REQUESTS",
                PixelRect::new(120.0, 148.0, 560.0, 190.0).unwrap(),
                [210, 30, 35],
            ),
            recognized_line(
                "LEFT BUBBLE",
                PixelRect::new(20.0, 240.0, 220.0, 280.0).unwrap(),
                [20, 20, 20],
            ),
            recognized_line(
                "RIGHT BUBBLE",
                PixelRect::new(350.0, 242.0, 560.0, 282.0).unwrap(),
                [20, 20, 20],
            ),
        ];

        let groups = group_recognized_lines(lines, ReadingDirection::Ltr);

        assert_eq!(groups.len(), 4);
        assert!(groups.iter().all(|group| group.len() == 1));
    }

    #[test]
    fn english_gate_accepts_confident_latin_bubble_text() {
        assert!(accept_english_ocr_line(0.91, "FORGOTTEN,"));
        assert!(accept_english_ocr_line(0.91, "-ENRIQUE-"));
        assert!(accept_english_ocr(
            CandidateKind::StoryText,
            0.91,
            "'LITTLE' MAN?"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.44,
            "Too uncertain"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "rgrodbedj e gbdue t js gb socews nggpodbgedg ectrseeed gbucbgebbr ege rbgtg"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "もう帰る"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "30 YEARS SINCE THE PROLOGUE"
        ));
        assert!(!accept_english_ocr(CandidateKind::StoryText, 0.99, "Ka Z"));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "30H0EXC"
        ));
        assert!(accept_english_ocr(CandidateKind::StoryText, 0.99, "Go on"));
        assert!(accept_english_ocr(CandidateKind::StoryText, 0.99, "R2D2"));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.64,
            "Egodbeab Ejgne ysugede deedj rgebjon."
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.66,
            "I Egsbgla"
        ));
    }

    #[test]
    fn narration_gate_accepts_prose_but_rejects_sfx_credits_logos_and_ambiguity() {
        assert!(accept_english_ocr(
            CandidateKind::StoryText,
            0.93,
            "And the shadow blade who devised the extermination unit."
        ));
        assert!(accept_english_ocr(
            CandidateKind::StoryText,
            0.93,
            "Thirty years later"
        ));
        assert!(accept_english_ocr(
            CandidateKind::StoryText,
            0.93,
            "NEXT IS..."
        ));
        assert!(accept_english_ocr(
            CandidateKind::StoryText,
            0.93,
            "“DANGEROUS ASSIGNMENTS.”"
        ));
        assert!(!accept_english_ocr(CandidateKind::StoryText, 0.99, "WHAM"));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "BANG BOOM CRASH WHAM!"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "“BANG BOOM!”"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "Translated by Moonlight Scans"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "READ-MANGA.COM"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "ORIGINAL CONTI | BACK GROUND | SKETCH | LINE ART | COLOR | EFFECT | TEXT | EDITING | DIRECTING"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "BACK GROUND | COLOR | EFFECT | EDITING"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "V d at a ea Read at.."
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "for the fastest releases"
        ));
        assert!(!accept_english_ocr(
            CandidateKind::StoryText,
            0.99,
            "30 YEARS SINCE THE PROLOGUE"
        ));
        assert!(accept_english_ocr(CandidateKind::StoryText, 0.99, "AND..."));
    }

    #[test]
    fn credit_context_only_rejects_isolated_uppercase_name_labels() {
        assert!(looks_like_credit_context("30 YEARS SINCE THE PROLOGUE"));
        assert!(looks_like_credit_context("for the fastest releases"));
        assert!(looks_like_isolated_credit_name("STEPH BLACK"));
        assert!(looks_like_isolated_credit_name("SHOUNEN BLACK BLACK"));

        assert!(!looks_like_isolated_credit_name("Southern Prizhenkaya"));
        assert!(!looks_like_isolated_credit_name(
            "THE NAMELESS HERO RETURNS"
        ));
        assert!(!looks_like_credit_context("SOUTHERN PRIZHENKAYA"));
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
        let generation_sources = primary_generation_indices(&partial_hits)
            .into_iter()
            .map(|index| sources[index])
            .collect::<Vec<_>>();

        assert_eq!(generation_sources, sources.to_vec());
        assert!(
            primary_generation_indices(&[
                Some("cached-first"),
                Some("cached-second"),
                Some("cached-third"),
            ])
            .is_empty()
        );
    }

    #[test]
    fn cancellation_is_observed_at_batch_boundaries() {
        let cancel = AtomicBool::new(false);
        assert!(cancellation_boundary(&cancel).is_ok());
        cancel.store(true, Ordering::Release);
        let error = cancellation_boundary(&cancel).unwrap_err();
        assert_eq!(error.code, "CANCELLED");
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
            translation_boundary_action(&one, false, now, false),
            TranslationBoundaryAction::ContinueUpstream
        );
        assert_eq!(
            translation_boundary_action(
                &one,
                false,
                now + TRANSLATION_MAX_FLUSH_DELAY - Duration::from_nanos(1),
                false
            ),
            TranslationBoundaryAction::ContinueUpstream
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
            translation_boundary_action(&two, false, now, false),
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
                false
            ),
            TranslationBoundaryAction::ContinueUpstream
        );
        assert_eq!(
            translation_boundary_action(&tail, false, now + TRANSLATION_MAX_FLUSH_DELAY, false),
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
            translation_boundary_action(&full[..6], false, now, false),
            TranslationBoundaryAction::Dispatch(TRANSLATION_BATCH_MAX)
        );
        assert_eq!(
            translation_boundary_action(&full, false, now, false),
            TranslationBoundaryAction::Dispatch(4)
        );

        let cancel = AtomicBool::new(false);
        assert!(cancellation_boundary(&cancel).is_ok());
        cancel.store(true, Ordering::Release);
        assert_eq!(
            translation_boundary_action(&full, false, now, cancel.load(Ordering::Acquire)),
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
        assert!(queue.next().is_none());
        assert_eq!(queue.jobs.len(), 1);

        queue.finish_primary_phase();
        assert_eq!(queue.next().unwrap().region.id, "bubble-a");
        assert!(queue.next().is_none());
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
    fn browser_preprocessing_pool_has_exactly_six_threads() {
        let pool = global_preprocessing_pool().unwrap();
        assert_eq!(pool.thread_count(), PREPROCESSING_THREADS);
        assert_eq!(PREPROCESSING_THREADS, 6);
    }
}
