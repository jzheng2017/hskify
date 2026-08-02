//! Secure unversioned loopback service with append-only progressive job updates.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes, HttpBody, to_bytes};
use axum::extract::multipart::{Field, MultipartRejection};
use axum::extract::{Extension, Multipart, Path, Query, Request, State};
use axum::http::header::{
    ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
    ACCESS_CONTROL_MAX_AGE, AUTHORIZATION, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE, HOST,
    ORIGIN, VARY,
};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use chrono::Utc;
use http_body::{Frame, SizeHint};
use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader, Limits};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::{sleep, timeout};
use uuid::Uuid;

use crate::contracts::{
    BUILD_FINGERPRINT, BrowserCapabilities, BrowserJobCreated, BrowserJobRequest, BrowserJobStage,
    BrowserSetupState, BrowserSetupStatus, CreateJobRequest, ErrorResponse, HealthResponse,
    HealthStatus, HskLevel, JobUpdate, JobUpdatesResponse, LookupInteraction, LookupRequest,
    NativeReadyResponse, NativeReadyType, NormalizedRect, PatchMimeType, PreservedArtworkRegion,
    ProgressiveRegion, RegionPatch, Validate, ViewportUpdateRequest,
};
use crate::crypto::{SECRET_BYTES, decode_secret, generate_secret, secrets_equal, sha256_hex};
use crate::decoded_cache::DecodedImageCache;
use crate::fixtures;
use crate::origin::validate_extension_origin;
use crate::pipeline_adapter::{
    CleaningError, CleaningInput, CleaningPipeline, KoharuPipeline, LookupInput,
    RegionLookupContext,
};
use crate::result_cache::{CachedJob, CachedRegion, ResultCache};
use crate::setup::{ManagedResourcePaths, ModelSetup};
use crate::{CONTROL_HEADER, EXTENSION_ORIGIN_HEADER};

const INTERNAL_SESSION_PATH: &str = "/browser-internal/session";
const MAX_INTERNAL_BODY_BYTES: usize = 4 * 1024;
const MAX_LOOKUP_BODY_BYTES: usize = 16 * 1024;
const MAX_VIEWPORT_BODY_BYTES: usize = 32 * 1024;
const MAX_FONT_BYTES: u64 = 32 * 1024 * 1024;
const MAX_SESSIONS: usize = 64;
const DEFAULT_MAX_RETAINED_JOBS: usize = 128;
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const DEFAULT_UPDATE_WAIT_MS: u64 = 20_000;
const MAX_UPDATE_WAIT_MS: u64 = 20_000;
const MAX_UPDATES_PER_JOB: usize = 10_000;
const WARMUP_NOT_STARTED: u8 = 0;
const WARMUP_RUNNING: u8 = 1;
const WARMUP_READY: u8 = 2;
const WARMUP_FAILED: u8 = 3;

#[derive(Debug, Clone)]
pub struct ServerLimits {
    pub max_upload_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_http_body_bytes: usize,
    pub max_pixels: u64,
    pub max_dimension: u32,
    pub max_decoded_bytes: u64,
    pub max_patch_blob_bytes: usize,
    pub max_retained_jobs: usize,
    pub max_stored_blob_bytes: usize,
    pub max_concurrent_requests: usize,
}

impl Default for ServerLimits {
    fn default() -> Self {
        const MIB: usize = 1024 * 1024;
        Self {
            max_upload_bytes: 20 * MIB,
            max_metadata_bytes: 64 * 1024,
            max_http_body_bytes: 21 * MIB,
            max_pixels: 25_000_000,
            max_dimension: 16_384,
            max_decoded_bytes: 128 * MIB as u64,
            max_patch_blob_bytes: 16 * MIB,
            max_retained_jobs: DEFAULT_MAX_RETAINED_JOBS,
            max_stored_blob_bytes: 256 * MIB,
            // Long-polling is bounded by session authentication and this
            // semaphore, but must not starve ordinary job/viewport requests.
            max_concurrent_requests: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub port: u16,
    pub session_ttl: Duration,
    pub idle_timeout: Duration,
    pub limits: ServerLimits,
}

impl BridgeConfig {
    pub fn for_port(port: u16) -> Self {
        Self {
            port,
            session_ttl: Duration::from_secs(15 * 60),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            limits: ServerLimits::default(),
        }
    }
}

#[derive(Debug)]
struct Session {
    token: [u8; SECRET_BYTES],
    origin: String,
    expires_at_unix_ms: u64,
}

#[derive(Debug, Clone)]
struct StoredBlob {
    bytes: Arc<[u8]>,
    content_type: &'static str,
    owner_job_id: String,
}

#[derive(Debug, Clone)]
pub struct JobViewport {
    pub revision: u64,
    pub visible_rects: Vec<NormalizedRect>,
    pub active: bool,
}

#[derive(Debug)]
struct JobLog {
    updates: Vec<JobUpdate>,
    terminal: bool,
    last_overall_progress: Option<f32>,
    published_regions: HashSet<String>,
    progressive_regions: HashMap<String, ProgressiveRegion>,
    preserved_artwork_regions: HashMap<String, PreservedArtworkRegion>,
    lookup_contexts: HashMap<String, RegionLookupContext>,
    viewport: JobViewport,
}

#[derive(Debug)]
struct JobRecord {
    order: u64,
    job_id: String,
    source: Mutex<Option<Arc<DynamicImage>>>,
    cancel: Arc<AtomicBool>,
    active: AtomicBool,
    log: Mutex<JobLog>,
    updates_notify: Notify,
    viewport_notify: Notify,
}

impl JobRecord {
    fn new(
        order: u64,
        job_id: String,
        source: Option<Arc<DynamicImage>>,
        visible_rects: Vec<NormalizedRect>,
    ) -> Self {
        Self {
            order,
            job_id,
            source: Mutex::new(source),
            cancel: Arc::new(AtomicBool::new(false)),
            active: AtomicBool::new(true),
            log: Mutex::new(JobLog {
                updates: Vec::new(),
                terminal: false,
                last_overall_progress: None,
                published_regions: HashSet::new(),
                progressive_regions: HashMap::new(),
                preserved_artwork_regions: HashMap::new(),
                lookup_contexts: HashMap::new(),
                viewport: JobViewport {
                    revision: 1,
                    visible_rects,
                    active: true,
                },
            }),
            updates_notify: Notify::new(),
            viewport_notify: Notify::new(),
        }
    }

    fn source(&self) -> Option<Arc<DynamicImage>> {
        self.source
            .lock()
            .expect("job source lock poisoned")
            .clone()
    }

    fn release_source(&self) {
        self.source.lock().expect("job source lock poisoned").take();
    }

    fn retained_source_bytes(&self) -> usize {
        self.source
            .lock()
            .expect("job source lock poisoned")
            .as_ref()
            .map_or(0, |image| {
                usize::try_from(
                    u64::from(image.width())
                        .saturating_mul(u64::from(image.height()))
                        .saturating_mul(3),
                )
                .unwrap_or(usize::MAX)
            })
    }

    fn is_terminal(&self) -> bool {
        self.log.lock().expect("job log lock poisoned").terminal
    }

    fn is_evictable(&self) -> bool {
        !self.active.load(Ordering::Acquire) && self.is_terminal()
    }

    fn append(&self, draft: JobUpdateDraft) -> Result<JobUpdate, PublishError> {
        let is_cancellation = matches!(&draft, JobUpdateDraft::Cancelled { .. });
        if self.cancel.load(Ordering::Acquire) && !is_cancellation {
            return Err(PublishError::Cancelled);
        }
        let mut log = self.log.lock().expect("job log lock poisoned");
        if log.terminal {
            return Err(PublishError::Terminal);
        }
        let is_terminal = matches!(
            &draft,
            JobUpdateDraft::Complete { .. }
                | JobUpdateDraft::Failed { .. }
                | JobUpdateDraft::Cancelled { .. }
        );
        if log.updates.len() >= MAX_UPDATES_PER_JOB && !is_terminal {
            return Err(PublishError::UpdateLimit);
        }
        if let JobUpdateDraft::Progress {
            overall_progress: Some(next),
            ..
        } = &draft
            && log
                .last_overall_progress
                .is_some_and(|previous| *next < previous)
        {
            return Err(PublishError::RegressiveProgress);
        }
        match &draft {
            JobUpdateDraft::RegionReady { region }
                if log.published_regions.contains(&region.id) =>
            {
                return Err(PublishError::DuplicateRegion(region.id.clone()));
            }
            JobUpdateDraft::ArtworkPreserved { region }
                if log.published_regions.contains(&region.id) =>
            {
                return Err(PublishError::DuplicateRegion(region.id.clone()));
            }
            _ => {}
        }
        let sequence = log
            .updates
            .last()
            .map(JobUpdate::sequence)
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(PublishError::SequenceExhausted)?;
        let update = draft.into_update(sequence);
        update.validate().map_err(PublishError::Contract)?;
        match &update {
            JobUpdate::RegionReady { region, .. } => {
                log.published_regions.insert(region.id.clone());
                log.progressive_regions
                    .insert(region.id.clone(), region.as_ref().clone());
            }
            JobUpdate::ArtworkPreserved { region, .. } => {
                log.published_regions.insert(region.id.clone());
                log.preserved_artwork_regions
                    .insert(region.id.clone(), region.clone());
            }
            JobUpdate::Progress {
                overall_progress: Some(value),
                ..
            } => {
                log.last_overall_progress = Some(*value);
            }
            _ => {}
        }
        if update.is_terminal() {
            log.terminal = true;
        }
        log.updates.push(update.clone());
        drop(log);
        self.updates_notify.notify_waiters();
        Ok(update)
    }

    fn replay_after(&self, after: u64) -> ReplaySnapshot {
        let log = self.log.lock().expect("job log lock poisoned");
        let latest = log.updates.last().map(JobUpdate::sequence).unwrap_or(0);
        let updates = log
            .updates
            .iter()
            .filter(|update| update.sequence() > after)
            .cloned()
            .collect();
        ReplaySnapshot {
            latest,
            terminal: log.terminal,
            updates,
        }
    }

    fn update_viewport(&self, request: ViewportUpdateRequest) {
        let mut log = self.log.lock().expect("job log lock poisoned");
        log.viewport.revision = log.viewport.revision.saturating_add(1);
        log.viewport.visible_rects = request.visible_rects;
        log.viewport.active = request.active;
        drop(log);
        self.viewport_notify.notify_waiters();
    }

    fn viewport(&self) -> JobViewport {
        self.log
            .lock()
            .expect("job log lock poisoned")
            .viewport
            .clone()
    }

    fn remember_lookup_context(&self, region_id: String, context: RegionLookupContext) {
        self.log
            .lock()
            .expect("job log lock poisoned")
            .lookup_contexts
            .insert(region_id, context);
    }

    fn lookup_context(&self, region_id: &str) -> Option<RegionLookupContext> {
        self.log
            .lock()
            .expect("job log lock poisoned")
            .lookup_contexts
            .get(region_id)
            .cloned()
    }

    fn progressive_regions(&self) -> Vec<ProgressiveRegion> {
        let mut regions = self
            .log
            .lock()
            .expect("job log lock poisoned")
            .progressive_regions
            .values()
            .cloned()
            .collect::<Vec<_>>();
        regions.sort_by(|left, right| {
            left.reading_order
                .cmp(&right.reading_order)
                .then_with(|| left.id.cmp(&right.id))
        });
        regions
    }

    fn preserved_artwork_regions(&self) -> Vec<PreservedArtworkRegion> {
        let mut regions = self
            .log
            .lock()
            .expect("job log lock poisoned")
            .preserved_artwork_regions
            .values()
            .cloned()
            .collect::<Vec<_>>();
        regions.sort_by(|left, right| {
            left.reading_order
                .cmp(&right.reading_order)
                .then_with(|| left.id.cmp(&right.id))
        });
        regions
    }
}

#[derive(Debug)]
struct ReplaySnapshot {
    latest: u64,
    terminal: bool,
    updates: Vec<JobUpdate>,
}

#[derive(Debug, Clone)]
pub enum JobUpdateDraft {
    Progress {
        stage: BrowserJobStage,
        stage_progress: Option<f32>,
        overall_progress: Option<f32>,
        current: Option<u32>,
        total: Option<u32>,
        message: String,
    },
    RegionReady {
        region: Box<ProgressiveRegion>,
    },
    ArtworkPreserved {
        region: PreservedArtworkRegion,
    },
    Complete {
        message: Option<String>,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
    Cancelled {
        message: Option<String>,
    },
}

impl JobUpdateDraft {
    fn into_update(self, sequence: u64) -> JobUpdate {
        match self {
            Self::Progress {
                stage,
                stage_progress,
                overall_progress,
                current,
                total,
                message,
            } => JobUpdate::Progress {
                sequence,
                stage,
                stage_progress,
                overall_progress,
                current,
                total,
                message,
            },
            Self::RegionReady { region } => JobUpdate::RegionReady { sequence, region },
            Self::ArtworkPreserved { region } => JobUpdate::ArtworkPreserved { sequence, region },
            Self::Complete { message } => JobUpdate::Complete { sequence, message },
            Self::Failed {
                code,
                message,
                retryable,
            } => JobUpdate::Failed {
                sequence,
                code,
                message,
                retryable,
            },
            Self::Cancelled { message } => JobUpdate::Cancelled { sequence, message },
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PublishError {
    #[error("job was cancelled")]
    Cancelled,
    #[error("job update log is terminal")]
    Terminal,
    #[error("region was published more than once: {0}")]
    DuplicateRegion(String),
    #[error("patch blob does not belong to this job: {0}")]
    UnknownPatch(String),
    #[error("job update sequence is exhausted")]
    SequenceExhausted,
    #[error("job update log reached its bounded limit")]
    UpdateLimit,
    #[error("overall progress must not move backwards")]
    RegressiveProgress,
    #[error("progressive contract validation failed: {0}")]
    Contract(crate::contracts::ContractError),
    #[error("patch must be a decodable PNG")]
    InvalidPatch,
    #[error("patch exceeds the per-blob limit")]
    PatchTooLarge,
    #[error("bounded patch storage is full")]
    StorageFull,
}

#[derive(Clone)]
pub struct JobUpdateSink {
    state: Arc<BridgeState>,
    record: Arc<JobRecord>,
}

impl JobUpdateSink {
    pub fn job_id(&self) -> &str {
        &self.record.job_id
    }

    /// Store one PNG patch and return the descriptor to place in a region
    /// update. Sequence assignment remains separate and atomic in `publish`.
    pub fn store_patch_png(
        &self,
        rect: NormalizedRect,
        bytes: Vec<u8>,
    ) -> Result<RegionPatch, PublishError> {
        self.store_generated_patch_png(rect, bytes)
    }

    pub(crate) fn store_generated_patch_png(
        &self,
        rect: NormalizedRect,
        bytes: Vec<u8>,
    ) -> Result<RegionPatch, PublishError> {
        self.validate_patch_storage_request(&rect)?;
        self.state
            .store_generated_patch_png(&self.record, rect, bytes)
    }

    pub(crate) fn store_cached_patch_png(
        &self,
        rect: NormalizedRect,
        bytes: Arc<[u8]>,
    ) -> Result<RegionPatch, PublishError> {
        self.validate_patch_storage_request(&rect)?;
        self.state.store_cached_patch_png(&self.record, rect, bytes)
    }

    fn validate_patch_storage_request(&self, rect: &NormalizedRect) -> Result<(), PublishError> {
        if self.record.cancel.load(Ordering::Acquire) {
            return Err(PublishError::Cancelled);
        }
        if self.record.is_terminal() {
            return Err(PublishError::Terminal);
        }
        rect.validate_at("patch.rect")
            .map_err(PublishError::Contract)
    }

    /// Append one replayable update. The sink assigns the next sequence and
    /// rejects publication after any terminal event.
    pub fn publish(&self, draft: JobUpdateDraft) -> Result<JobUpdate, PublishError> {
        if let JobUpdateDraft::RegionReady { region } = &draft
            && !self
                .state
                .patch_belongs_to_job(&self.record.job_id, &region.patch.blob_id)
        {
            return Err(PublishError::UnknownPatch(region.patch.blob_id.clone()));
        }
        let update = self.record.append(draft)?;
        self.state.touch();
        Ok(update)
    }

    pub fn viewport(&self) -> JobViewport {
        self.record.viewport()
    }

    pub async fn wait_for_viewport_change(
        &self,
        after_revision: u64,
        max_wait: Duration,
    ) -> JobViewport {
        loop {
            let notified = self.record.viewport_notify.notified();
            let viewport = self.record.viewport();
            if viewport.revision > after_revision {
                return viewport;
            }
            if timeout(max_wait, notified).await.is_err() {
                return self.record.viewport();
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.record.cancel.load(Ordering::Acquire)
    }

    /// Preserve only the language context used by the optional dictionary
    /// lookup route. Browser clients receive `ProgressiveRegion` updates.
    pub(crate) fn remember_region_for_lookup(
        &self,
        region_id: String,
        context: RegionLookupContext,
    ) {
        self.record.remember_lookup_context(region_id, context);
    }
}

#[derive(Debug, Default)]
struct Storage {
    jobs: HashMap<String, Arc<JobRecord>>,
    blobs: HashMap<String, StoredBlob>,
    next_job_order: u64,
}

#[derive(Debug, Clone, Copy)]
enum CapacityKind {
    Jobs,
    Blobs,
}

impl Storage {
    fn next_order(&mut self) -> u64 {
        let order = self.next_job_order;
        self.next_job_order = self.next_job_order.saturating_add(1);
        order
    }

    fn stored_blob_bytes(&self) -> usize {
        self.blobs
            .values()
            .try_fold(0_usize, |total, blob| total.checked_add(blob.bytes.len()))
            .unwrap_or(usize::MAX)
    }

    fn retained_bytes(&self) -> usize {
        let sources = self
            .jobs
            .values()
            .try_fold(0_usize, |total, job| {
                total.checked_add(job.retained_source_bytes())
            })
            .unwrap_or(usize::MAX);
        self.stored_blob_bytes().saturating_add(sources)
    }

    fn oldest_evictable_job_id(&self) -> Option<String> {
        self.jobs
            .iter()
            .filter(|(_, job)| job.is_evictable())
            .min_by(|(left_id, left), (right_id, right)| {
                left.order
                    .cmp(&right.order)
                    .then_with(|| left_id.cmp(right_id))
            })
            .map(|(job_id, _)| job_id.clone())
    }

    fn evict_job(&mut self, job_id: &str) {
        if self.jobs.remove(job_id).is_some() {
            self.blobs.retain(|_, blob| blob.owner_job_id != job_id);
        }
    }

    fn make_room_for_job(
        &mut self,
        limits: &ServerLimits,
        added_source_bytes: usize,
    ) -> Result<(), CapacityKind> {
        if added_source_bytes > limits.max_stored_blob_bytes {
            return Err(CapacityKind::Blobs);
        }
        loop {
            let jobs_full = self.jobs.len() >= limits.max_retained_jobs;
            let bytes_full = self.retained_bytes().saturating_add(added_source_bytes)
                > limits.max_stored_blob_bytes;
            if !jobs_full && !bytes_full {
                return Ok(());
            }
            let Some(job_id) = self.oldest_evictable_job_id() else {
                return Err(if jobs_full {
                    CapacityKind::Jobs
                } else {
                    CapacityKind::Blobs
                });
            };
            self.evict_job(&job_id);
        }
    }

    fn make_room_for_blob(
        &mut self,
        limits: &ServerLimits,
        added_blob_bytes: usize,
    ) -> Result<(), CapacityKind> {
        if added_blob_bytes > limits.max_stored_blob_bytes {
            return Err(CapacityKind::Blobs);
        }
        while self.retained_bytes().saturating_add(added_blob_bytes) > limits.max_stored_blob_bytes
        {
            let Some(job_id) = self.oldest_evictable_job_id() else {
                return Err(CapacityKind::Blobs);
            };
            self.evict_job(&job_id);
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Lifecycle {
    last_activity: Instant,
    admitted_requests: usize,
    shutdown_latched: bool,
}

struct RequestAdmission {
    state: Arc<BridgeState>,
    _capacity_permit: OwnedSemaphorePermit,
}

impl Drop for RequestAdmission {
    fn drop(&mut self) {
        let mut lifecycle = self
            .state
            .lifecycle
            .lock()
            .expect("lifecycle lock poisoned");
        lifecycle.admitted_requests = lifecycle
            .admitted_requests
            .checked_sub(1)
            .expect("request admission count underflow");
        lifecycle.last_activity = Instant::now();
    }
}

struct AdmittedBody {
    inner: Body,
    _admission: Arc<RequestAdmission>,
}

impl HttpBody for AdmittedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(context)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

pub struct BridgeState {
    config: BridgeConfig,
    control_secret: [u8; SECRET_BYTES],
    pipeline: Arc<dyn CleaningPipeline>,
    setup: Option<Arc<ModelSetup>>,
    decoded_images: Mutex<DecodedImageCache>,
    result_cache: Arc<ResultCache>,
    sessions: Mutex<Vec<Session>>,
    storage: RwLock<Storage>,
    lifecycle: Mutex<Lifecycle>,
    request_capacity: Arc<Semaphore>,
    active_jobs: AtomicUsize,
    warmup_state: AtomicU8,
}

impl BridgeState {
    pub fn new(
        config: BridgeConfig,
        control_secret: [u8; SECRET_BYTES],
        cache_root: PathBuf,
    ) -> Arc<Self> {
        let pipeline = Arc::new(KoharuPipeline::new(cache_root.clone()));
        let setup = ModelSetup::new(
            ManagedResourcePaths::discover()
                .expect("browser companion requires a managed resource directory"),
            cache_root.clone(),
        )
        .expect("embedded model setup manifest must be valid");
        Self::with_pipeline_and_setup(
            config,
            control_secret,
            pipeline,
            Some(Arc::new(setup)),
            cache_root,
        )
    }

    fn with_pipeline_and_setup(
        config: BridgeConfig,
        control_secret: [u8; SECRET_BYTES],
        pipeline: Arc<dyn CleaningPipeline>,
        setup: Option<Arc<ModelSetup>>,
        cache_root: PathBuf,
    ) -> Arc<Self> {
        assert_ne!(config.port, 0, "browser daemon requires a bound port");
        assert!(
            config.limits.max_http_body_bytes >= config.limits.max_upload_bytes,
            "HTTP body limit must cover an upload"
        );
        assert!(
            config.limits.max_retained_jobs > 0,
            "at least one browser job must be retainable"
        );
        assert!(
            config.limits.max_concurrent_requests > 0,
            "at least one authenticated request must be admissible"
        );
        let request_capacity = Arc::new(Semaphore::new(config.limits.max_concurrent_requests));
        Arc::new(Self {
            config,
            control_secret,
            pipeline,
            setup,
            decoded_images: Mutex::new(DecodedImageCache::default()),
            result_cache: Arc::new(ResultCache::new(cache_root.join("results"))),
            sessions: Mutex::new(Vec::new()),
            storage: RwLock::new(Storage::default()),
            lifecycle: Mutex::new(Lifecycle {
                last_activity: Instant::now(),
                admitted_requests: 0,
                shutdown_latched: false,
            }),
            request_capacity,
            active_jobs: AtomicUsize::new(0),
            warmup_state: AtomicU8::new(0),
        })
    }

    fn resources_ready(&self) -> bool {
        self.setup.as_ref().map_or_else(
            || self.pipeline.resources_ready(),
            |setup| setup.resources_ready(),
        )
    }

    pub(crate) fn start_pipeline_warmup(self: &Arc<Self>) -> bool {
        if !self.resources_ready() {
            return false;
        }
        if self
            .warmup_state
            .compare_exchange(
                WARMUP_NOT_STARTED,
                WARMUP_RUNNING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        let state = Arc::clone(self);
        let pipeline = Arc::clone(&self.pipeline);
        tokio::spawn(async move {
            let next = match pipeline.warm_up().await {
                Ok(()) => WARMUP_READY,
                Err(error) => {
                    eprintln!("Hskify model warm-up failed: {error}");
                    WARMUP_FAILED
                }
            };
            state.warmup_state.store(next, Ordering::Release);
        });
        true
    }

    fn retry_pipeline_warmup(self: &Arc<Self>) -> bool {
        let _ = self.warmup_state.compare_exchange(
            WARMUP_FAILED,
            WARMUP_NOT_STARTED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.start_pipeline_warmup()
    }

    fn pipeline_ready(&self) -> bool {
        self.resources_ready() && self.warmup_state.load(Ordering::Acquire) == WARMUP_READY
    }

    fn effective_setup_status(&self, mut status: BrowserSetupStatus) -> BrowserSetupStatus {
        if status.state != BrowserSetupState::Ready {
            return status;
        }
        match self.warmup_state.load(Ordering::Acquire) {
            WARMUP_READY => status,
            WARMUP_FAILED => {
                status.state = BrowserSetupState::Failed;
                status.message = "Hskify could not get ready. Please try again.".to_owned();
                status.error_code = Some("MODEL_WARMUP_FAILED".to_owned());
                status
            }
            _ => {
                status.state = BrowserSetupState::Warming;
                status.message = "Hskify is getting ready.".to_owned();
                status
            }
        }
    }

    pub fn port(&self) -> u16 {
        self.config.port
    }

    pub fn active_job_count(&self) -> usize {
        self.active_jobs.load(Ordering::Acquire)
    }

    pub fn issue_session(&self, origin: &str) -> Result<NativeReadyResponse, ApiError> {
        validate_extension_origin(origin).map_err(|_| {
            ApiError::bad_request(
                "INVALID_EXTENSION_ORIGIN",
                "The extension origin is not canonical.",
            )
        })?;
        let (raw, encoded) = generate_secret().map_err(|_| ApiError::internal())?;
        let now = unix_ms();
        let ttl_ms = u64::try_from(self.config.session_ttl.as_millis()).unwrap_or(u64::MAX);
        let expires_at = now.saturating_add(ttl_ms.max(1));
        let mut sessions = self.sessions.lock().expect("session lock poisoned");
        sessions.retain(|session| session.expires_at_unix_ms > now);
        if sessions.len() >= MAX_SESSIONS {
            sessions.remove(0);
        }
        sessions.push(Session {
            token: raw,
            origin: origin.to_owned(),
            expires_at_unix_ms: expires_at,
        });
        drop(sessions);
        self.touch();

        Ok(NativeReadyResponse {
            message_type: NativeReadyType::Ready,
            build_fingerprint: BUILD_FINGERPRINT.to_owned(),
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
            port: self.config.port,
            token: encoded,
            session_expires_at_unix_ms: expires_at,
            capabilities: BrowserCapabilities {
                source_languages: vec![crate::contracts::SOURCE_LANGUAGE.to_owned()],
                target_languages: vec![crate::contracts::TARGET_LANGUAGE.to_owned()],
                hsk_levels: vec![
                    HskLevel::One,
                    HskLevel::Two,
                    HskLevel::Three,
                    HskLevel::Four,
                    HskLevel::Five,
                    HskLevel::Six,
                ],
                models_ready: self.pipeline_ready(),
            },
        })
    }

    /// Acquire a publisher for pipeline integration without exposing storage
    /// internals or sequence assignment.
    pub fn job_update_sink(self: &Arc<Self>, job_id: &str) -> Option<JobUpdateSink> {
        self.storage
            .read()
            .expect("storage lock poisoned")
            .jobs
            .get(job_id)
            .cloned()
            .map(|record| JobUpdateSink {
                state: self.clone(),
                record,
            })
    }

    fn touch(&self) {
        self.lifecycle
            .lock()
            .expect("lifecycle lock poisoned")
            .last_activity = Instant::now();
    }

    fn try_admit_authenticated(self: &Arc<Self>) -> Result<Arc<RequestAdmission>, ApiError> {
        let capacity_permit = self
            .request_capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| {
                ApiError::too_many_requests(
                    "REQUEST_CAPACITY_EXHAUSTED",
                    "The local browser companion is at its bounded request capacity.",
                )
            })?;
        let mut lifecycle = self.lifecycle.lock().expect("lifecycle lock poisoned");
        if lifecycle.shutdown_latched {
            return Err(ApiError::service_unavailable(
                "DAEMON_SHUTTING_DOWN",
                "The local browser companion is shutting down; start a fresh session.",
            ));
        }
        lifecycle.admitted_requests = lifecycle
            .admitted_requests
            .checked_add(1)
            .expect("request admission count overflow");
        lifecycle.last_activity = Instant::now();
        drop(lifecycle);
        Ok(Arc::new(RequestAdmission {
            state: self.clone(),
            _capacity_permit: capacity_permit,
        }))
    }

    fn try_latch_idle_shutdown(&self) -> bool {
        let mut lifecycle = self.lifecycle.lock().expect("lifecycle lock poisoned");
        if lifecycle.shutdown_latched {
            return true;
        }
        if self.active_job_count() == 0
            && lifecycle.admitted_requests == 0
            && lifecycle.last_activity.elapsed() >= self.config.idle_timeout
        {
            lifecycle.shutdown_latched = true;
            return true;
        }
        false
    }

    fn origin_has_session(&self, origin: &str) -> bool {
        let now = unix_ms();
        let mut sessions = self.sessions.lock().expect("session lock poisoned");
        sessions.retain(|session| session.expires_at_unix_ms > now);
        sessions.iter().any(|session| session.origin == origin)
    }

    fn authenticate(&self, origin: &str, token: &str) -> bool {
        let Ok(candidate) = decode_secret(token) else {
            return false;
        };
        let now = unix_ms();
        let mut sessions = self.sessions.lock().expect("session lock poisoned");
        sessions.retain(|session| session.expires_at_unix_ms > now);
        let mut accepted = false;
        for session in sessions.iter() {
            accepted |= secrets_equal(&session.token, &candidate) && session.origin == origin;
        }
        accepted
    }

    fn control_authenticates(&self, candidate: &str) -> bool {
        decode_secret(candidate)
            .map(|candidate| secrets_equal(&self.control_secret, &candidate))
            .unwrap_or(false)
    }

    fn reserve_uploaded_job(
        self: &Arc<Self>,
        source: Option<Arc<DynamicImage>>,
        visible_rects: Vec<NormalizedRect>,
    ) -> Result<(String, Arc<JobRecord>, JobUpdateSink), ApiError> {
        let mut storage = self.storage.write().expect("storage lock poisoned");
        let source_bytes = source.as_ref().map_or(0, |image| {
            usize::try_from(
                u64::from(image.width())
                    .saturating_mul(u64::from(image.height()))
                    .saturating_mul(3),
            )
            .unwrap_or(usize::MAX)
        });
        storage
            .make_room_for_job(&self.config.limits, source_bytes)
            .map_err(capacity_api_error)?;
        let job_id = loop {
            let candidate = format!("job-{}", Uuid::new_v4());
            if !storage.jobs.contains_key(&candidate) {
                break candidate;
            }
        };
        let record = Arc::new(JobRecord::new(
            storage.next_order(),
            job_id.clone(),
            source,
            visible_rects,
        ));
        let sink = JobUpdateSink {
            state: self.clone(),
            record: record.clone(),
        };
        sink.publish(JobUpdateDraft::Progress {
            stage: BrowserJobStage::Queued,
            stage_progress: None,
            overall_progress: Some(0.0),
            current: None,
            total: None,
            message: "Queued for local cleaning and translation".to_owned(),
        })
        .map_err(|_| ApiError::internal())?;
        storage.jobs.insert(job_id.clone(), record.clone());
        self.active_jobs.fetch_add(1, Ordering::AcqRel);
        drop(storage);
        self.touch();
        Ok((job_id, record, sink))
    }

    fn store_generated_patch_png(
        &self,
        record: &JobRecord,
        rect: NormalizedRect,
        bytes: Vec<u8>,
    ) -> Result<RegionPatch, PublishError> {
        validate_generated_patch_png(&bytes, &self.config.limits)?;
        self.store_patch_bytes(record, rect, bytes.into())
    }

    fn store_cached_patch_png(
        &self,
        record: &JobRecord,
        rect: NormalizedRect,
        bytes: Arc<[u8]>,
    ) -> Result<RegionPatch, PublishError> {
        if bytes.len() > self.config.limits.max_patch_blob_bytes {
            return Err(PublishError::PatchTooLarge);
        }
        self.store_patch_bytes(record, rect, bytes)
    }

    fn store_patch_bytes(
        &self,
        record: &JobRecord,
        rect: NormalizedRect,
        bytes: Arc<[u8]>,
    ) -> Result<RegionPatch, PublishError> {
        let mut storage = self.storage.write().expect("storage lock poisoned");
        if !storage.jobs.contains_key(&record.job_id) {
            return Err(PublishError::Terminal);
        }
        storage
            .make_room_for_blob(&self.config.limits, bytes.len())
            .map_err(|_| PublishError::StorageFull)?;
        let blob_id = loop {
            let candidate = format!("patch-{}", Uuid::new_v4());
            if !storage.blobs.contains_key(&candidate) {
                break candidate;
            }
        };
        storage.blobs.insert(
            blob_id.clone(),
            StoredBlob {
                bytes,
                content_type: "image/png",
                owner_job_id: record.job_id.clone(),
            },
        );
        drop(storage);
        self.touch();
        Ok(RegionPatch {
            blob_id,
            mime_type: PatchMimeType::Png,
            rect,
        })
    }

    fn patch_belongs_to_job(&self, job_id: &str, blob_id: &str) -> bool {
        self.storage
            .read()
            .expect("storage lock poisoned")
            .blobs
            .get(blob_id)
            .is_some_and(|blob| blob.owner_job_id == job_id)
    }

    fn completed_cache_job(&self, record: &JobRecord) -> Result<CachedJob, CleaningError> {
        let completed_regions = record.progressive_regions();
        let preserved_artwork = record.preserved_artwork_regions();
        let storage = self.storage.read().expect("storage lock poisoned");
        let regions = completed_regions
            .into_iter()
            .map(|region| {
                let patch_png = storage
                    .blobs
                    .get(&region.patch.blob_id)
                    .filter(|blob| blob.owner_job_id == record.job_id)
                    .map(|blob| blob.bytes.clone())
                    .ok_or_else(|| {
                        CleaningError::new(
                            "CACHE_FAILED",
                            "A completed region patch was unavailable for persistence.",
                        )
                    })?;
                let lookup_context = record.lookup_context(&region.id).ok_or_else(|| {
                    CleaningError::new(
                        "CACHE_FAILED",
                        "A completed region lookup context was unavailable for persistence.",
                    )
                })?;
                Ok(CachedRegion {
                    region,
                    lookup_context,
                    patch_png,
                })
            })
            .collect::<Result<Vec<_>, CleaningError>>()?;
        Ok(CachedJob {
            regions,
            preserved_artwork,
        })
    }
}

fn validate_generated_patch_png(bytes: &[u8], limits: &ServerLimits) -> Result<(), PublishError> {
    if bytes.len() > limits.max_patch_blob_bytes {
        return Err(PublishError::PatchTooLarge);
    }
    if bytes.len() < 8 || bytes[..8] != [137, 80, 78, 71, 13, 10, 26, 10] {
        return Err(PublishError::InvalidPatch);
    }
    if bytes.len() < 24 || &bytes[12..16] != b"IHDR" {
        return Err(PublishError::InvalidPatch);
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().expect("PNG width header"));
    let height = u32::from_be_bytes(bytes[20..24].try_into().expect("PNG height header"));
    if width == 0 || height == 0 || width > limits.max_dimension || height > limits.max_dimension {
        return Err(PublishError::InvalidPatch);
    }
    Ok(())
}

fn capacity_api_error(kind: CapacityKind) -> ApiError {
    match kind {
        CapacityKind::Jobs => ApiError::too_many_requests(
            "JOB_LIMIT_REACHED",
            "All retained browser jobs are still active.",
        ),
        CapacityKind::Blobs => ApiError::too_many_requests(
            "BLOB_LIMIT_REACHED",
            "Active browser jobs occupy the bounded blob cache.",
        ),
    }
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: &'static str,
    retryable: bool,
}

impl ApiError {
    fn bad_request(code: &'static str, message: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message,
            retryable: false,
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "UNAUTHORIZED",
            message: "The browser session is missing, expired, or invalid.",
            retryable: true,
        }
    }

    fn not_found(code: &'static str, message: &'static str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code,
            message,
            retryable: false,
        }
    }

    fn payload_too_large(code: &'static str, message: &'static str) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            code,
            message,
            retryable: false,
        }
    }

    fn too_many_requests(code: &'static str, message: &'static str) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            code,
            message,
            retryable: true,
        }
    }

    fn service_unavailable(code: &'static str, message: &'static str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code,
            message,
            retryable: true,
        }
    }

    fn unsupported_media(message: &'static str) -> Self {
        Self {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            code: "UNSUPPORTED_IMAGE",
            message,
            retryable: false,
        }
    }

    fn internal() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "INTERNAL_ERROR",
            message: "The local browser companion failed.",
            retryable: true,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorResponse {
            code: self.code.to_owned(),
            message: self.message.to_owned(),
            retryable: self.retryable,
        };
        let mut response = (self.status, Json(body)).into_response();
        harden_response(&mut response);
        response
    }
}

pub fn router(state: Arc<BridgeState>) -> Router {
    let max_body = state.config.limits.max_http_body_bytes;
    Router::new()
        .route(INTERNAL_SESSION_PATH, post(issue_internal_session))
        .route("/health", get(health))
        .route("/setup", get(setup))
        .route("/setup/models", post(setup_models))
        .route("/jobs", post(create_job))
        .route("/jobs/{job_id}", axum::routing::delete(cancel_job))
        .route("/jobs/{job_id}/viewport", put(update_viewport))
        .route("/jobs/{job_id}/updates", get(job_updates))
        .route("/lookup", post(lookup))
        .route("/blobs/{patch_id}", get(blob))
        .route("/fonts/{font_id}", get(font))
        .fallback(not_found)
        .layer(axum::extract::DefaultBodyLimit::max(max_body))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            security_boundary,
        ))
        .with_state(state)
}

pub async fn wait_until_idle(state: Arc<BridgeState>) {
    let interval = state
        .config
        .idle_timeout
        .checked_div(4)
        .unwrap_or(Duration::from_millis(25))
        .clamp(Duration::from_millis(25), Duration::from_secs(1));
    loop {
        sleep(interval).await;
        if state.try_latch_idle_shutdown() {
            return;
        }
    }
}

async fn security_boundary(
    State(state): State<Arc<BridgeState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    let is_internal = path == INTERNAL_SESSION_PATH;
    let is_browser = browser_path(path);
    if !is_internal && !is_browser {
        return StatusCode::NOT_FOUND.into_response();
    }

    if !valid_host(request.headers(), state.port()) {
        return ApiError::bad_request(
            "INVALID_HOST",
            "The Host header must name this IPv4 loopback endpoint.",
        )
        .into_response();
    }

    if is_internal {
        let authenticated = single_header(request.headers(), CONTROL_HEADER)
            .map(|value| state.control_authenticates(value))
            .unwrap_or(false);
        if !authenticated {
            return ApiError::unauthorized().into_response();
        }
        let admission = match state.try_admit_authenticated() {
            Ok(admission) => admission,
            Err(error) => return error.into_response(),
        };
        request.extensions_mut().insert(admission.clone());
        return admitted_response(next.run(request).await, admission);
    }

    let standard_origin = single_header(request.headers(), ORIGIN.as_str());
    let extension_origin = single_header(request.headers(), EXTENSION_ORIGIN_HEADER);
    let origin = match (standard_origin, extension_origin) {
        (Some(standard), Some(extension)) if standard == extension => standard,
        (Some(standard), None) => standard,
        (None, Some(extension)) => extension,
        _ => return ApiError::unauthorized().into_response(),
    };
    if validate_extension_origin(origin).is_err() || !state.origin_has_session(origin) {
        return ApiError::unauthorized().into_response();
    }
    let origin = origin.to_owned();

    if request.method() == Method::OPTIONS {
        return preflight(request.headers(), &origin);
    }

    let bearer = single_header(request.headers(), AUTHORIZATION.as_str())
        .and_then(|value| value.strip_prefix("Bearer "));
    if !bearer
        .map(|token| state.authenticate(&origin, token))
        .unwrap_or(false)
    {
        return with_cors(ApiError::unauthorized().into_response(), &origin);
    }

    let admission = match state.try_admit_authenticated() {
        Ok(admission) => admission,
        Err(error) => return with_cors(error.into_response(), &origin),
    };
    request.extensions_mut().insert(admission.clone());
    let response = admitted_response(next.run(request).await, admission);
    with_cors(response, &origin)
}

fn browser_path(path: &str) -> bool {
    if matches!(
        path,
        "/health" | "/setup" | "/setup/models" | "/jobs" | "/lookup"
    ) {
        return true;
    }
    if let Some(rest) = path.strip_prefix("/jobs/") {
        let mut segments = rest.split('/');
        let Some(job_id) = segments.next() else {
            return false;
        };
        if job_id.is_empty() {
            return false;
        }
        return matches!(
            (segments.next(), segments.next()),
            (None, None) | (Some("viewport" | "updates"), None)
        );
    }
    ["/blobs/", "/fonts/"].into_iter().any(|prefix| {
        path.strip_prefix(prefix)
            .is_some_and(|identifier| !identifier.is_empty() && !identifier.contains('/'))
    })
}

fn admitted_response(response: Response, admission: Arc<RequestAdmission>) -> Response {
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(AdmittedBody {
            inner: body,
            _admission: admission,
        }),
    )
}

fn valid_host(headers: &HeaderMap, port: u16) -> bool {
    let expected = format!("127.0.0.1:{port}");
    single_header(headers, HOST.as_str()) == Some(expected.as_str())
}

fn single_header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let name = HeaderName::from_bytes(name.as_bytes()).ok()?;
    let mut values = headers.get_all(name).iter();
    let value = values.next()?.to_str().ok()?;
    if values.next().is_some() {
        return None;
    }
    Some(value)
}

fn preflight(headers: &HeaderMap, origin: &str) -> Response {
    let Some(method) = single_header(headers, "access-control-request-method") else {
        return ApiError::bad_request("INVALID_PREFLIGHT", "Missing preflight method.")
            .into_response();
    };
    if !matches!(method, "GET" | "POST" | "PUT" | "DELETE") {
        return ApiError::bad_request("INVALID_PREFLIGHT", "The requested method is not allowed.")
            .into_response();
    }
    let Some(requested_headers) = single_header(headers, "access-control-request-headers") else {
        return ApiError::bad_request("INVALID_PREFLIGHT", "Missing preflight headers.")
            .into_response();
    };
    let mut saw_authorization = false;
    for value in requested_headers.split(',').map(str::trim) {
        if value.eq_ignore_ascii_case("authorization") {
            saw_authorization = true;
        } else if value.eq_ignore_ascii_case(EXTENSION_ORIGIN_HEADER)
            || value.eq_ignore_ascii_case("content-type")
        {
        } else {
            return ApiError::bad_request(
                "INVALID_PREFLIGHT",
                "The requested header is not allowed.",
            )
            .into_response();
        }
    }
    if !saw_authorization {
        return ApiError::bad_request("INVALID_PREFLIGHT", "The Authorization header is required.")
            .into_response();
    }

    let mut response = StatusCode::NO_CONTENT.into_response();
    let response_headers = response.headers_mut();
    response_headers.insert(
        ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_str(method).expect("known method"),
    );
    response_headers.insert(
        ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("authorization, content-type, x-hsk-manga-extension-origin"),
    );
    response_headers.insert(ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("300"));
    with_cors(response, origin)
}

fn with_cors(mut response: Response, origin: &str) -> Response {
    if let Ok(origin) = HeaderValue::from_str(origin) {
        response
            .headers_mut()
            .insert(ACCESS_CONTROL_ALLOW_ORIGIN, origin);
        response
            .headers_mut()
            .insert(VARY, HeaderValue::from_static("Origin"));
    }
    harden_response(&mut response);
    response
}

fn harden_response(response: &mut Response) {
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response.headers_mut().insert(
        HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InternalSessionRequest {
    extension_origin: String,
}

async fn issue_internal_session(
    State(state): State<Arc<BridgeState>>,
    request: Request,
) -> Result<Json<NativeReadyResponse>, ApiError> {
    let request: InternalSessionRequest = parse_json_body(request, MAX_INTERNAL_BODY_BYTES).await?;
    Ok(Json(state.issue_session(&request.extension_origin)?))
}

async fn health(State(state): State<Arc<BridgeState>>) -> Json<HealthResponse> {
    state.start_pipeline_warmup();
    let setup_status = state.effective_setup_status(
        state
            .setup
            .as_ref()
            .map_or_else(|| resource_setup_status(&state), |setup| setup.status()),
    );
    Json(HealthResponse {
        build_fingerprint: BUILD_FINGERPRINT.to_owned(),
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
        status: HealthStatus::Ready,
        setup_state: setup_status.state,
        resource_identities: state.setup.as_ref().map_or_else(
            || fixtures::health().resource_identities,
            |setup| setup.resource_identities(),
        ),
    })
}

async fn setup(State(state): State<Arc<BridgeState>>) -> Json<BrowserSetupStatus> {
    state.start_pipeline_warmup();
    Json(
        state.effective_setup_status(
            state
                .setup
                .as_ref()
                .map_or_else(|| resource_setup_status(&state), |setup| setup.status()),
        ),
    )
}

async fn setup_models(State(state): State<Arc<BridgeState>>) -> Json<BrowserSetupStatus> {
    if state.resources_ready() {
        state.retry_pipeline_warmup();
        return Json(
            state.effective_setup_status(
                state
                    .setup
                    .as_ref()
                    .map_or_else(|| resource_setup_status(&state), |setup| setup.status()),
            ),
        );
    }
    Json(
        state
            .setup
            .as_ref()
            .map_or_else(|| resource_setup_status(&state), |setup| setup.start()),
    )
}

fn resource_setup_status(state: &BridgeState) -> BrowserSetupStatus {
    if state.resources_ready() {
        BrowserSetupStatus {
            state: BrowserSetupState::Ready,
            model_id: "qwen3.5-4b".to_owned(),
            current_file: None,
            completed_bytes: None,
            total_bytes: None,
            required_disk_bytes: None,
            message: "Local translation and language resources are ready.".to_owned(),
            error_code: None,
        }
    } else {
        BrowserSetupStatus {
            state: BrowserSetupState::MissingModels,
            model_id: "qwen3.5-4b".to_owned(),
            current_file: None,
            completed_bytes: None,
            total_bytes: None,
            required_disk_bytes: None,
            message: "Local translation and language resources are missing.".to_owned(),
            error_code: None,
        }
    }
}

async fn create_job(
    State(state): State<Arc<BridgeState>>,
    Extension(admission): Extension<Arc<RequestAdmission>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<impl IntoResponse, ApiError> {
    let mut multipart = multipart.map_err(|_| {
        ApiError::bad_request(
            "INVALID_MULTIPART",
            "The job request must be multipart form data.",
        )
    })?;
    let mut image: Option<(Vec<u8>, Option<String>)> = None;
    let mut metadata: Option<Vec<u8>> = None;
    let mut field_count = 0_u8;

    while let Some(field) = multipart.next_field().await.map_err(|_| {
        ApiError::payload_too_large(
            "UPLOAD_TOO_LARGE",
            "The multipart upload exceeds the browser-mode limit.",
        )
    })? {
        field_count = field_count.saturating_add(1);
        if field_count > 2 {
            return Err(ApiError::bad_request(
                "INVALID_MULTIPART",
                "Only image and request fields are accepted.",
            ));
        }
        let name = field.name().unwrap_or_default().to_owned();
        let content_type = field.content_type().map(ToOwned::to_owned);
        match name.as_str() {
            "image" if image.is_none() => {
                let bytes = read_multipart_field(
                    field,
                    state.config.limits.max_upload_bytes,
                    ApiError::payload_too_large(
                        "UPLOAD_TOO_LARGE",
                        "The source image exceeds the byte limit.",
                    ),
                )
                .await?;
                image = Some((bytes, content_type));
            }
            "request" if metadata.is_none() => {
                if content_type.as_deref() != Some("application/json") {
                    return Err(ApiError::bad_request(
                        "INVALID_REQUEST_CONTENT_TYPE",
                        "The multipart request field must be application/json.",
                    ));
                }
                let bytes = read_multipart_field(
                    field,
                    state.config.limits.max_metadata_bytes,
                    ApiError::payload_too_large(
                        "REQUEST_TOO_LARGE",
                        "The job metadata exceeds the byte limit.",
                    ),
                )
                .await?;
                metadata = Some(bytes);
            }
            "image" | "request" => {
                return Err(ApiError::bad_request(
                    "DUPLICATE_MULTIPART_FIELD",
                    "Multipart fields must appear exactly once.",
                ));
            }
            _ => {
                return Err(ApiError::bad_request(
                    "UNKNOWN_MULTIPART_FIELD",
                    "Only image and request fields are accepted.",
                ));
            }
        }
    }

    let (image, field_mime) = image.ok_or_else(|| {
        ApiError::bad_request("MISSING_IMAGE", "The multipart image field is required.")
    })?;
    let metadata = metadata.ok_or_else(|| {
        ApiError::bad_request(
            "MISSING_REQUEST",
            "The multipart request field is required.",
        )
    })?;
    let create_request: CreateJobRequest = serde_json::from_slice(&metadata).map_err(|_| {
        ApiError::bad_request("INVALID_JOB_REQUEST", "The job metadata is invalid.")
    })?;
    create_request.validate().map_err(|_| {
        ApiError::bad_request(
            "INVALID_JOB_REQUEST",
            "The job metadata failed build and semantic validation.",
        )
    })?;
    if field_mime.as_deref() != Some(create_request.source_mime_type.as_str()) {
        return Err(ApiError::unsupported_media(
            "The multipart image MIME type does not match the request.",
        ));
    }

    let pipeline_request = create_request.pipeline_request();
    let limits = state.config.limits.clone();
    let validation_request = pipeline_request.clone();
    let validation_state = state.clone();
    let prepared = tokio::task::spawn_blocking(move || {
        let _admission = admission;
        let format = validate_image_upload_identity(&image, &validation_request, &limits)?;
        let cached = match validation_state.result_cache.load(&validation_request) {
            Ok(cached) => cached,
            Err(_) => {
                validation_state
                    .result_cache
                    .invalidate(&validation_request)
                    .map_err(|_| ApiError::internal())?;
                None
            }
        };
        if let Some(cached) = cached {
            return Ok(PreparedUpload::Cached(cached));
        }
        let source = decode_image_upload(
            image,
            format,
            &validation_request,
            &limits,
            &validation_state.decoded_images,
        )?;
        Ok(PreparedUpload::Source(source))
    })
    .await
    .map_err(|_| ApiError::internal())??;

    let (source, cached) = match prepared {
        PreparedUpload::Cached(cached) => (None, Some(cached)),
        PreparedUpload::Source(source) => (Some(source), None),
    };
    let (job_id, record, sink) =
        state.reserve_uploaded_job(source, create_request.visible_rects)?;
    tokio::spawn(run_cleaning_job(
        state,
        record,
        pipeline_request,
        sink,
        cached,
    ));

    Ok((
        StatusCode::ACCEPTED,
        Json(BrowserJobCreated {
            build_fingerprint: BUILD_FINGERPRINT.to_owned(),
            job_id,
        }),
    ))
}

async fn read_multipart_field(
    mut field: Field<'_>,
    limit: usize,
    too_large: ApiError,
) -> Result<Vec<u8>, ApiError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(|_| {
        ApiError::bad_request(
            "INVALID_MULTIPART",
            "The multipart field could not be read.",
        )
    })? {
        let new_length = bytes.len().checked_add(chunk.len()).ok_or_else(|| {
            ApiError::payload_too_large(
                "UPLOAD_TOO_LARGE",
                "The multipart field length overflowed.",
            )
        })?;
        if new_length > limit {
            return Err(too_large);
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

#[derive(Debug)]
enum PreparedUpload {
    Cached(CachedJob),
    Source(Arc<DynamicImage>),
}

fn validate_image_upload_identity(
    image: &[u8],
    request: &BrowserJobRequest,
    limits: &ServerLimits,
) -> Result<ImageFormat, ApiError> {
    if image.is_empty() {
        return Err(ApiError::unsupported_media("The source image is empty."));
    }
    if image.len() > limits.max_upload_bytes {
        return Err(ApiError::payload_too_large(
            "UPLOAD_TOO_LARGE",
            "The source image exceeds the byte limit.",
        ));
    }
    if !sha256_hex(&image).eq_ignore_ascii_case(&request.source_sha256) {
        return Err(ApiError::bad_request(
            "SOURCE_HASH_MISMATCH",
            "The uploaded image does not match sourceSha256.",
        ));
    }
    let format = image::guess_format(&image)
        .map_err(|_| ApiError::unsupported_media("The source image format is not recognized."))?;
    let expected_mime = match format {
        ImageFormat::Png => "image/png",
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::WebP => "image/webp",
        ImageFormat::Gif => "image/gif",
        _ => {
            return Err(ApiError::unsupported_media(
                "The source image format is not supported.",
            ));
        }
    };
    if request.source_mime_type != expected_mime {
        return Err(ApiError::unsupported_media(
            "The source MIME type does not match the image bytes.",
        ));
    }
    let declared_pixels = u64::from(request.natural_width) * u64::from(request.natural_height);
    if declared_pixels > limits.max_pixels
        || request.natural_width > limits.max_dimension
        || request.natural_height > limits.max_dimension
    {
        return Err(ApiError::payload_too_large(
            "IMAGE_TOO_LARGE",
            "The decoded image exceeds the configured pixel limit.",
        ));
    }

    let mut decoder_limits = Limits::default();
    decoder_limits.max_image_width = Some(limits.max_dimension);
    decoder_limits.max_image_height = Some(limits.max_dimension);
    decoder_limits.max_alloc = Some(limits.max_decoded_bytes);
    let mut reader = ImageReader::new(Cursor::new(image));
    reader.set_format(format);
    reader.limits(decoder_limits);
    let (width, height) = reader.into_dimensions().map_err(|_| {
        ApiError::unsupported_media(
            "The source image header could not be decoded within safe limits.",
        )
    })?;
    let pixels = u64::from(width) * u64::from(height);
    if width != request.natural_width
        || height != request.natural_height
        || pixels > limits.max_pixels
    {
        return Err(ApiError::bad_request(
            "IMAGE_DIMENSION_MISMATCH",
            "Image dimensions do not match the job metadata.",
        ));
    }
    Ok(format)
}

fn decode_image_upload(
    image: Vec<u8>,
    format: ImageFormat,
    request: &BrowserJobRequest,
    limits: &ServerLimits,
    decoded_images: &Mutex<DecodedImageCache>,
) -> Result<Arc<DynamicImage>, ApiError> {
    if let Some(source) = decoded_images
        .lock()
        .expect("decoded image cache lock poisoned")
        .get(&request.source_sha256)
    {
        let (width, height) = source.dimensions();
        if width != request.natural_width || height != request.natural_height {
            return Err(ApiError::bad_request(
                "IMAGE_DIMENSION_MISMATCH",
                "Decoded dimensions do not match the job metadata.",
            ));
        }
        return Ok(source);
    }

    let mut decoder_limits = Limits::default();
    decoder_limits.max_image_width = Some(limits.max_dimension);
    decoder_limits.max_image_height = Some(limits.max_dimension);
    decoder_limits.max_alloc = Some(limits.max_decoded_bytes);
    let mut reader = ImageReader::new(Cursor::new(image.as_slice()));
    reader.set_format(format);
    reader.limits(decoder_limits);
    let decoded = reader.decode().map_err(|_| {
        ApiError::unsupported_media("The source image could not be decoded within safe limits.")
    })?;
    let (width, height) = decoded.dimensions();
    let pixels = u64::from(width) * u64::from(height);
    if width != request.natural_width
        || height != request.natural_height
        || pixels > limits.max_pixels
    {
        return Err(ApiError::bad_request(
            "IMAGE_DIMENSION_MISMATCH",
            "Decoded dimensions do not match the job metadata.",
        ));
    }
    let source = Arc::new(DynamicImage::ImageRgb8(decoded.into_rgb8()));
    let source = decoded_images
        .lock()
        .expect("decoded image cache lock poisoned")
        .insert(request.source_sha256.clone(), source);
    Ok(source)
}

async fn run_cleaning_job(
    state: Arc<BridgeState>,
    record: Arc<JobRecord>,
    request: BrowserJobRequest,
    sink: JobUpdateSink,
    cached: Option<CachedJob>,
) {
    if record.cancel.load(Ordering::Acquire) {
        finish_active(&state, &record);
        return;
    }

    if let Some(cached) = cached {
        record.release_source();
        let result = replay_cached_job(&sink, cached);
        match result {
            Ok(()) => {
                if !record.cancel.load(Ordering::Acquire) && !record.is_terminal() {
                    let _ = sink.publish(JobUpdateDraft::Complete {
                        message: Some("Exact cached translation replayed".to_owned()),
                    });
                }
            }
            Err(error) => {
                if !record.cancel.load(Ordering::Acquire) {
                    fail_job(&state, &record, &sink, error);
                }
            }
        }
        finish_active(&state, &record);
        return;
    }

    if !state.resources_ready() {
        fail_job(
            &state,
            &record,
            &sink,
            CleaningError::new(
                "RESOURCES_NOT_READY",
                "Verified local model and language resources are not ready.",
            ),
        );
        finish_active(&state, &record);
        return;
    }

    let Some(source) = record.source() else {
        fail_job(
            &state,
            &record,
            &sink,
            CleaningError {
                code: "SOURCE_UNAVAILABLE",
                message: "The bounded decoded source image is no longer available.".to_owned(),
            },
        );
        finish_active(&state, &record);
        return;
    };
    let result = state
        .pipeline
        .run(
            CleaningInput {
                source,
                request: request.clone(),
            },
            record.cancel.clone(),
            sink.clone(),
        )
        .await;
    match result {
        Ok(()) => {
            if !record.cancel.load(Ordering::Acquire) && !record.is_terminal() {
                let cached = match state.completed_cache_job(&record) {
                    Ok(cached) => cached,
                    Err(error) => {
                        fail_job(&state, &record, &sink, error);
                        finish_active(&state, &record);
                        return;
                    }
                };
                let cache = state.result_cache.clone();
                let cache_request = request.clone();
                match tokio::task::spawn_blocking(move || cache.store(&cache_request, &cached))
                    .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) => {
                        fail_job(
                            &state,
                            &record,
                            &sink,
                            CleaningError::new(
                                "CACHE_FAILED",
                                "The completed result could not be persisted.",
                            ),
                        );
                        finish_active(&state, &record);
                        return;
                    }
                    Err(_) => {
                        fail_job(
                            &state,
                            &record,
                            &sink,
                            CleaningError::new(
                                "CACHE_FAILED",
                                "The completed result persistence task did not complete.",
                            ),
                        );
                        finish_active(&state, &record);
                        return;
                    }
                }
                let _ = sink.publish(JobUpdateDraft::Complete {
                    message: Some(
                        "Local cleaning and HSK translation complete and persisted".to_owned(),
                    ),
                });
            }
        }
        Err(error) => {
            if !record.cancel.load(Ordering::Acquire) {
                fail_job(&state, &record, &sink, error);
            }
        }
    }
    finish_active(&state, &record);
}

fn replay_cached_job(sink: &JobUpdateSink, cached: CachedJob) -> Result<(), CleaningError> {
    for region in cached.preserved_artwork {
        sink.publish(JobUpdateDraft::ArtworkPreserved { region })
            .map_err(|error| CleaningError::new("CACHE_REPLAY_FAILED", error.to_string()))?;
    }
    for cached_region in cached.regions {
        if sink.is_cancelled() {
            return Err(CleaningError::cancelled());
        }
        let mut region = cached_region.region;
        region.patch = sink
            .store_cached_patch_png(region.patch.rect.clone(), cached_region.patch_png)
            .map_err(|error| CleaningError::new("CACHE_REPLAY_FAILED", error.to_string()))?;
        sink.remember_region_for_lookup(region.id.clone(), cached_region.lookup_context);
        sink.publish(JobUpdateDraft::RegionReady {
            region: Box::new(region),
        })
        .map_err(|error| CleaningError::new("CACHE_REPLAY_FAILED", error.to_string()))?;
    }
    Ok(())
}

fn fail_job(state: &BridgeState, record: &JobRecord, sink: &JobUpdateSink, error: CleaningError) {
    record.release_source();
    if record.cancel.load(Ordering::Acquire) || record.is_terminal() {
        return;
    }
    let _ = sink.publish(JobUpdateDraft::Failed {
        code: error.code.to_owned(),
        message: error.message,
        retryable: true,
    });
    state.touch();
}

fn finish_active(state: &BridgeState, record: &JobRecord) {
    record.release_source();
    if record.active.swap(false, Ordering::AcqRel) {
        state.active_jobs.fetch_sub(1, Ordering::AcqRel);
        state.touch();
    }
}

fn find_job(state: &BridgeState, job_id: &str) -> Result<Arc<JobRecord>, ApiError> {
    state
        .storage
        .read()
        .expect("storage lock poisoned")
        .jobs
        .get(job_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("JOB_NOT_FOUND", "The browser job does not exist."))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct UpdatesQuery {
    #[serde(default)]
    after: u64,
    #[serde(default = "default_update_wait_ms")]
    wait_ms: u64,
}

const fn default_update_wait_ms() -> u64 {
    DEFAULT_UPDATE_WAIT_MS
}

async fn job_updates(
    State(state): State<Arc<BridgeState>>,
    Path(job_id): Path<String>,
    Query(query): Query<UpdatesQuery>,
) -> Result<Json<JobUpdatesResponse>, ApiError> {
    let job = find_job(&state, &job_id)?;
    let wait = Duration::from_millis(query.wait_ms.min(MAX_UPDATE_WAIT_MS));
    let deadline = Instant::now() + wait;
    loop {
        let notified = job.updates_notify.notified();
        let replay = job.replay_after(query.after);
        if query.after > replay.latest {
            return Err(ApiError::bad_request(
                "INVALID_UPDATE_SEQUENCE",
                "after must not exceed the latest published sequence.",
            ));
        }
        if !replay.updates.is_empty() || replay.terminal || wait.is_zero() {
            let next_sequence = replay
                .updates
                .last()
                .map(JobUpdate::sequence)
                .unwrap_or(query.after);
            let response = JobUpdatesResponse {
                job_id,
                next_sequence,
                updates: replay.updates,
            };
            response.validate().map_err(|_| ApiError::internal())?;
            return Ok(Json(response));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() || timeout(remaining, notified).await.is_err() {
            return Ok(Json(JobUpdatesResponse {
                job_id,
                next_sequence: query.after,
                updates: Vec::new(),
            }));
        }
    }
}

async fn update_viewport(
    State(state): State<Arc<BridgeState>>,
    Path(job_id): Path<String>,
    request: Request,
) -> Result<StatusCode, ApiError> {
    let request: ViewportUpdateRequest = parse_json_body(request, MAX_VIEWPORT_BODY_BYTES).await?;
    request.validate().map_err(|_| {
        ApiError::bad_request(
            "INVALID_VIEWPORT",
            "The viewport update failed semantic validation.",
        )
    })?;
    let job = find_job(&state, &job_id)?;
    job.update_viewport(request);
    state.touch();
    Ok(StatusCode::NO_CONTENT)
}

async fn cancel_job(
    State(state): State<Arc<BridgeState>>,
    Path(job_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let job = find_job(&state, &job_id)?;
    job.cancel.store(true, Ordering::Release);
    if !job.is_terminal() {
        let sink = JobUpdateSink {
            state: state.clone(),
            record: job.clone(),
        };
        let _ = sink.publish(JobUpdateDraft::Cancelled {
            message: Some("Cancelled".to_owned()),
        });
        finish_active(&state, &job);
    }
    job.release_source();
    Ok(StatusCode::NO_CONTENT)
}

async fn lookup(
    State(state): State<Arc<BridgeState>>,
    request: Request,
) -> Result<Json<crate::contracts::LookupResult>, ApiError> {
    let request: LookupRequest = parse_json_body(request, MAX_LOOKUP_BODY_BYTES).await?;
    request.validate().map_err(|_| {
        ApiError::bad_request(
            "INVALID_LOOKUP_REQUEST",
            "The lookup request failed semantic validation.",
        )
    })?;
    let region = if let (Some(job_id), Some(region_id)) = (&request.job_id, &request.region_id) {
        let job = find_job(&state, job_id)?;
        Some(job.lookup_context(region_id).ok_or_else(|| {
            ApiError::not_found("REGION_NOT_FOUND", "The browser region does not exist.")
        })?)
    } else {
        None
    };
    let input = match request.interaction {
        LookupInteraction::Selection => LookupInput::Selection(
            request
                .selected_text
                .expect("validated selection lookup contains selected text"),
        ),
        LookupInteraction::Hover => {
            let character_offset = request
                .character_offset
                .expect("validated hover lookup contains a character offset");
            let region = region
                .as_ref()
                .expect("validated hover request contains a translated region");
            let character_offset = usize::try_from(character_offset).map_err(|_| {
                ApiError::bad_request("INVALID_LOOKUP_OFFSET", "The hovered character is invalid.")
            })?;
            if character_offset >= region.displayed_chinese.chars().count() {
                return Err(ApiError::bad_request(
                    "INVALID_LOOKUP_OFFSET",
                    "The hovered character is outside the translated region.",
                ));
            }
            LookupInput::Hover {
                displayed_text: region.displayed_chinese.clone(),
                character_offset,
            }
        }
    };
    let result = state.pipeline.lookup(input, region).await.map_err(|_| {
        ApiError::service_unavailable(
            "LANGUAGE_RESOURCES_NOT_READY",
            "The local HSK and dictionary resources are unavailable.",
        )
    })?;
    result.validate().map_err(|_| ApiError::internal())?;
    Ok(Json(result))
}

async fn blob(
    State(state): State<Arc<BridgeState>>,
    Path(patch_id): Path<String>,
) -> Result<Response, ApiError> {
    let blob = state
        .storage
        .read()
        .expect("storage lock poisoned")
        .blobs
        .get(&patch_id)
        .cloned()
        .ok_or_else(|| {
            ApiError::not_found("BLOB_NOT_FOUND", "The browser patch does not exist.")
        })?;
    Ok(arc_bytes_response(blob.bytes, blob.content_type))
}

async fn font(
    State(state): State<Arc<BridgeState>>,
    Path(font_id): Path<String>,
) -> Result<Response, ApiError> {
    if let Some(setup) = &state.setup {
        let path = setup.font_path(&font_id).ok_or_else(|| {
            ApiError::not_found("FONT_NOT_FOUND", "The browser font does not exist.")
        })?;
        let metadata = tokio::fs::metadata(&path).await.map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ApiError::not_found("FONT_NOT_FOUND", "The browser font does not exist.")
            } else {
                ApiError::internal()
            }
        })?;
        if !metadata.is_file() {
            return Err(ApiError::not_found(
                "FONT_NOT_FOUND",
                "The browser font does not exist.",
            ));
        }
        if metadata.len() > MAX_FONT_BYTES {
            return Err(ApiError::payload_too_large(
                "FONT_TOO_LARGE",
                "The browser font is too large.",
            ));
        }
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|_| ApiError::internal())?;
        return Ok(bytes_response(bytes, "font/ttf"));
    }
    let bytes = fixtures::font_bytes(&font_id)
        .ok_or_else(|| ApiError::not_found("FONT_NOT_FOUND", "The browser font does not exist."))?;
    Ok(bytes_response(bytes.to_vec(), "font/ttf"))
}

fn bytes_response(bytes: Vec<u8>, content_type: &'static str) -> Response {
    let length = bytes.len();
    let mut response = Response::new(Body::from(bytes));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&length.to_string()).expect("byte length header"),
    );
    response
}

fn arc_bytes_response(bytes: Arc<[u8]>, content_type: &'static str) -> Response {
    let length = bytes.len();
    let mut response = Response::new(Body::from(Bytes::from_owner(bytes)));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&length.to_string()).expect("byte length header"),
    );
    response
}

async fn parse_json_body<T: DeserializeOwned>(
    request: Request,
    limit: usize,
) -> Result<T, ApiError> {
    if single_header(request.headers(), CONTENT_TYPE.as_str()) != Some("application/json") {
        return Err(ApiError::bad_request(
            "INVALID_CONTENT_TYPE",
            "The request Content-Type must be application/json.",
        ));
    }
    let bytes = to_bytes(request.into_body(), limit).await.map_err(|_| {
        ApiError::payload_too_large("REQUEST_TOO_LARGE", "The JSON body is too large.")
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|_| ApiError::bad_request("INVALID_JSON", "The JSON body is invalid."))
}

async fn not_found() -> StatusCode {
    StatusCode::NOT_FOUND
}

fn unix_ms() -> u64 {
    u64::try_from(Utc::now().timestamp_millis()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct CountingPipeline {
        runs: AtomicUsize,
        warmups: AtomicUsize,
        ready: bool,
        sabotage_cache_root: Option<PathBuf>,
    }

    #[async_trait::async_trait]
    impl CleaningPipeline for CountingPipeline {
        async fn warm_up(&self) -> Result<(), CleaningError> {
            self.warmups.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn run(
            &self,
            _input: CleaningInput,
            _cancel: Arc<AtomicBool>,
            _sink: JobUpdateSink,
        ) -> Result<(), CleaningError> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            if let Some(path) = &self.sabotage_cache_root {
                std::fs::write(path, b"not a cache directory")
                    .map_err(|error| CleaningError::new("TEST_FAILED", error.to_string()))?;
            }
            Ok(())
        }

        async fn lookup(
            &self,
            _input: LookupInput,
            _region: Option<RegionLookupContext>,
        ) -> Result<crate::contracts::LookupResult, CleaningError> {
            Err(CleaningError::new("UNUSED", "lookup is not used"))
        }

        fn resources_ready(&self) -> bool {
            self.ready
        }
    }

    fn valid_png() -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::new_rgba8(2, 2)
            .write_to(&mut cursor, ImageFormat::Png)
            .unwrap();
        cursor.into_inner()
    }

    fn cached_fixture_region() -> ProgressiveRegion {
        fixtures::updates("cached-job")
            .updates
            .into_iter()
            .find_map(|update| match update {
                JobUpdate::RegionReady { region, .. } => Some(*region),
                _ => None,
            })
            .expect("cached fixture region")
    }

    fn lookup_context_for(region: &ProgressiveRegion) -> RegionLookupContext {
        RegionLookupContext {
            source_english: region.source_english.clone(),
            base_chinese: region.base_chinese.clone(),
            displayed_chinese: region.displayed_chinese.clone(),
            proper_names: Vec::new(),
        }
    }

    #[test]
    fn default_idle_timeout_is_thirty_minutes() {
        assert_eq!(
            BridgeConfig::for_port(1234).idle_timeout,
            Duration::from_secs(30 * 60)
        );
    }

    #[test]
    fn default_image_limits_match_the_browser_contract() {
        let limits = ServerLimits::default();
        assert_eq!(limits.max_upload_bytes, 20 * 1024 * 1024);
        assert_eq!(limits.max_pixels, 25_000_000);
        assert_eq!(limits.max_dimension, 16_384);
    }

    #[test]
    fn generated_patch_validation_checks_png_header_without_decoding() {
        let temp = tempfile::tempdir().unwrap();
        let pipeline = Arc::new(CountingPipeline {
            runs: AtomicUsize::new(0),
            warmups: AtomicUsize::new(0),
            ready: true,
            sabotage_cache_root: None,
        });
        let state = BridgeState::with_pipeline_and_setup(
            BridgeConfig::for_port(1234),
            [7; SECRET_BYTES],
            pipeline,
            None,
            temp.path().to_path_buf(),
        );
        let request: CreateJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        let (_, _record, sink) = state
            .reserve_uploaded_job(None, request.visible_rects)
            .unwrap();
        let rect = NormalizedRect {
            x: 0.1,
            y: 0.1,
            width: 0.2,
            height: 0.2,
        };

        let mut invalid_signature = valid_png();
        invalid_signature[0] = 0;
        assert!(matches!(
            sink.store_generated_patch_png(rect, invalid_signature),
            Err(PublishError::InvalidPatch)
        ));

        let mut invalid_ihdr = valid_png();
        invalid_ihdr[12] = b'X';
        assert!(matches!(
            sink.store_generated_patch_png(rect, invalid_ihdr),
            Err(PublishError::InvalidPatch)
        ));

        let mut zero_width = valid_png();
        zero_width[16..20].copy_from_slice(&0_u32.to_be_bytes());
        assert!(matches!(
            sink.store_generated_patch_png(rect, zero_width),
            Err(PublishError::InvalidPatch)
        ));

        let mut oversized = valid_png();
        oversized[16..20].copy_from_slice(&16_385_u32.to_be_bytes());
        assert!(matches!(
            sink.store_generated_patch_png(rect, oversized),
            Err(PublishError::InvalidPatch)
        ));

        assert!(sink.store_generated_patch_png(rect, valid_png()).is_ok());
    }

    #[test]
    fn cached_and_completed_patch_paths_share_arc_storage() {
        let temp = tempfile::tempdir().unwrap();
        let pipeline = Arc::new(CountingPipeline {
            runs: AtomicUsize::new(0),
            warmups: AtomicUsize::new(0),
            ready: true,
            sabotage_cache_root: None,
        });
        let state = BridgeState::with_pipeline_and_setup(
            BridgeConfig::for_port(1234),
            [7; SECRET_BYTES],
            pipeline,
            None,
            temp.path().to_path_buf(),
        );
        let request: CreateJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        let (_, _record, sink) = state
            .reserve_uploaded_job(None, request.visible_rects.clone())
            .unwrap();
        let cached_bytes: Arc<[u8]> = Arc::from(valid_png());
        let region = cached_fixture_region();
        replay_cached_job(
            &sink,
            CachedJob {
                regions: vec![CachedRegion {
                    lookup_context: lookup_context_for(&region),
                    region: region.clone(),
                    patch_png: cached_bytes.clone(),
                }],
                preserved_artwork: Vec::new(),
            },
        )
        .unwrap();
        let replayed_blob = state
            .storage
            .read()
            .unwrap()
            .blobs
            .values()
            .find(|blob| blob.owner_job_id == sink.job_id())
            .expect("replayed cached patch")
            .bytes
            .clone();
        assert!(Arc::ptr_eq(&replayed_blob, &cached_bytes));

        let record = Arc::new(JobRecord::new(
            1,
            "job-completed-cache".to_owned(),
            None,
            request.visible_rects,
        ));
        {
            let mut log = record.log.lock().unwrap();
            log.progressive_regions
                .insert(region.id.clone(), region.clone());
            log.lookup_contexts
                .insert(region.id.clone(), lookup_context_for(&region));
        }
        state.storage.write().unwrap().blobs.insert(
            region.patch.blob_id.clone(),
            StoredBlob {
                bytes: cached_bytes.clone(),
                content_type: "image/png",
                owner_job_id: record.job_id.clone(),
            },
        );
        let completed = state.completed_cache_job(&record).unwrap();
        assert!(Arc::ptr_eq(&completed.regions[0].patch_png, &cached_bytes));
    }

    #[test]
    fn downloaded_resources_are_not_reported_ready_until_models_are_resident() {
        let temp = tempfile::tempdir().unwrap();
        let pipeline = Arc::new(CountingPipeline {
            runs: AtomicUsize::new(0),
            warmups: AtomicUsize::new(0),
            ready: true,
            sabotage_cache_root: None,
        });
        let state = BridgeState::with_pipeline_and_setup(
            BridgeConfig::for_port(1234),
            [7; SECRET_BYTES],
            pipeline,
            None,
            temp.path().to_path_buf(),
        );

        let warming = state.effective_setup_status(resource_setup_status(&state));
        assert_eq!(warming.state, BrowserSetupState::Warming);
        assert!(
            !state
                .issue_session("moz-extension://00000000-0000-4000-8000-000000000001")
                .unwrap()
                .capabilities
                .models_ready
        );

        state.warmup_state.store(WARMUP_READY, Ordering::Release);
        let ready = state.effective_setup_status(resource_setup_status(&state));
        assert_eq!(ready.state, BrowserSetupState::Ready);
        assert!(
            state
                .issue_session("moz-extension://00000000-0000-4000-8000-000000000001")
                .unwrap()
                .capabilities
                .models_ready
        );

        state.warmup_state.store(WARMUP_FAILED, Ordering::Release);
        let failed = state.effective_setup_status(resource_setup_status(&state));
        assert_eq!(failed.state, BrowserSetupState::Failed);
        assert_eq!(failed.error_code.as_deref(), Some("MODEL_WARMUP_FAILED"));
    }

    #[tokio::test]
    async fn ready_pipeline_warms_once_in_background() {
        let temp = tempfile::tempdir().unwrap();
        let pipeline = Arc::new(CountingPipeline {
            runs: AtomicUsize::new(0),
            warmups: AtomicUsize::new(0),
            ready: true,
            sabotage_cache_root: None,
        });
        let state = BridgeState::with_pipeline_and_setup(
            BridgeConfig::for_port(1234),
            [7; SECRET_BYTES],
            pipeline.clone(),
            None,
            temp.path().to_path_buf(),
        );

        assert!(state.start_pipeline_warmup());
        assert!(!state.start_pipeline_warmup());
        while pipeline.warmups.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        while state.warmup_state.load(Ordering::Acquire) != WARMUP_READY {
            tokio::task::yield_now().await;
        }

        assert_eq!(pipeline.warmups.load(Ordering::Acquire), 1);
        assert!(!state.start_pipeline_warmup());
    }

    #[tokio::test]
    async fn exact_cache_replay_never_invokes_the_inference_pipeline() {
        let temp = tempfile::tempdir().unwrap();
        let pipeline = Arc::new(CountingPipeline {
            runs: AtomicUsize::new(0),
            warmups: AtomicUsize::new(0),
            ready: false,
            sabotage_cache_root: None,
        });
        let state = BridgeState::with_pipeline_and_setup(
            BridgeConfig::for_port(1234),
            [7; SECRET_BYTES],
            pipeline.clone(),
            None,
            temp.path().to_path_buf(),
        );
        let create_request: CreateJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        let request = create_request.pipeline_request();
        let region = fixtures::updates("cached-job")
            .updates
            .into_iter()
            .find_map(|update| match update {
                JobUpdate::RegionReady { region, .. } => Some(*region),
                _ => None,
            })
            .unwrap();
        state
            .result_cache
            .store(
                &request,
                &CachedJob {
                    regions: vec![CachedRegion {
                        lookup_context: RegionLookupContext {
                            source_english: region.source_english.clone(),
                            base_chinese: region.base_chinese.clone(),
                            displayed_chinese: region.displayed_chinese.clone(),
                            proper_names: vec![hsk_control::ProperName {
                                text: "\u{5c0f}\u{660e}".to_owned(),
                                reason: hsk_control::ProperNameReason::PersonName,
                            }],
                        },
                        region,
                        patch_png: Arc::from(valid_png()),
                    }],
                    preserved_artwork: Vec::new(),
                },
            )
            .unwrap();
        let (_, record, sink) = state
            .reserve_uploaded_job(None, create_request.visible_rects)
            .unwrap();
        let cached = state.result_cache.load(&request).unwrap();

        run_cleaning_job(state, record.clone(), request, sink, cached).await;

        assert_eq!(pipeline.runs.load(Ordering::Relaxed), 0);
        assert!(record.is_terminal());
        assert!(
            record
                .replay_after(0)
                .updates
                .iter()
                .any(|update| matches!(update, JobUpdate::RegionReady { .. }))
        );
        assert_eq!(
            record
                .lookup_context("aaaaaaaa-region-0001")
                .expect("cached lookup context")
                .proper_names,
            vec![hsk_control::ProperName {
                text: "\u{5c0f}\u{660e}".to_owned(),
                reason: hsk_control::ProperNameReason::PersonName,
            }]
        );
    }

    #[tokio::test]
    async fn cache_miss_does_not_invoke_an_unready_pipeline() {
        let temp = tempfile::tempdir().unwrap();
        let pipeline = Arc::new(CountingPipeline {
            runs: AtomicUsize::new(0),
            warmups: AtomicUsize::new(0),
            ready: false,
            sabotage_cache_root: None,
        });
        let state = BridgeState::with_pipeline_and_setup(
            BridgeConfig::for_port(1234),
            [7; SECRET_BYTES],
            pipeline.clone(),
            None,
            temp.path().to_path_buf(),
        );
        let create_request: CreateJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        let request = create_request.pipeline_request();
        let (_, record, sink) = state
            .reserve_uploaded_job(None, create_request.visible_rects)
            .unwrap();

        run_cleaning_job(state, record.clone(), request, sink, None).await;

        assert_eq!(pipeline.runs.load(Ordering::Relaxed), 0);
        assert!(record.replay_after(0).updates.iter().any(|update| {
            matches!(update, JobUpdate::Failed { code, .. } if code == "RESOURCES_NOT_READY")
        }));
    }

    #[tokio::test]
    async fn persistence_failure_publishes_failed_instead_of_complete() {
        let temp = tempfile::tempdir().unwrap();
        let cache_root = temp.path().join("results");
        let pipeline = Arc::new(CountingPipeline {
            runs: AtomicUsize::new(0),
            warmups: AtomicUsize::new(0),
            ready: true,
            sabotage_cache_root: Some(cache_root),
        });
        let state = BridgeState::with_pipeline_and_setup(
            BridgeConfig::for_port(1234),
            [7; SECRET_BYTES],
            pipeline.clone(),
            None,
            temp.path().to_path_buf(),
        );
        let create_request: CreateJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        let request = create_request.pipeline_request();
        let source = Arc::new(DynamicImage::new_rgb8(1, 1));
        let (_, record, sink) = state
            .reserve_uploaded_job(Some(source), create_request.visible_rects)
            .unwrap();

        run_cleaning_job(state, record.clone(), request, sink, None).await;

        assert_eq!(pipeline.runs.load(Ordering::Relaxed), 1);
        let updates = record.replay_after(0).updates;
        assert!(updates.iter().any(
            |update| matches!(update, JobUpdate::Failed { code, .. } if code == "CACHE_FAILED")
        ));
        assert!(
            !updates
                .iter()
                .any(|update| matches!(update, JobUpdate::Complete { .. }))
        );
    }

    #[test]
    fn only_unversioned_browser_paths_cross_the_security_boundary() {
        for path in [
            "/health",
            "/setup",
            "/setup/models",
            "/jobs",
            "/jobs/job-1",
            "/jobs/job-1/viewport",
            "/jobs/job-1/updates",
            "/blobs/patch-1",
            "/lookup",
            "/fonts/hmt-sans",
        ] {
            assert!(browser_path(path), "{path}");
        }
        for removed in [
            "/browser/v1/health",
            "/browser/v1/jobs",
            "/browser/v1/jobs/job-1/result",
            "/browser/v1/jobs/job-1/retranslate",
            "/jobs/job-1/result",
            "/jobs/job-1/retranslate",
        ] {
            assert!(!browser_path(removed), "{removed}");
        }
    }

    #[test]
    fn update_log_is_append_only_and_terminal() {
        let request: CreateJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        let record = JobRecord::new(0, "job-test".to_owned(), None, request.visible_rects);
        let first = record
            .append(JobUpdateDraft::Progress {
                stage: BrowserJobStage::Queued,
                stage_progress: None,
                overall_progress: Some(0.0),
                current: None,
                total: None,
                message: "Queued".to_owned(),
            })
            .unwrap();
        let terminal = record
            .append(JobUpdateDraft::Complete {
                message: Some("Done".to_owned()),
            })
            .unwrap();
        assert_eq!(first.sequence(), 1);
        assert_eq!(terminal.sequence(), 2);
        assert!(matches!(
            record.append(JobUpdateDraft::Complete { message: None }),
            Err(PublishError::Terminal)
        ));
        assert_eq!(
            record
                .replay_after(0)
                .updates
                .iter()
                .map(JobUpdate::sequence)
                .collect::<Vec<_>>(),
            [1, 2]
        );
    }

    #[test]
    fn cancellation_wins_the_terminal_race() {
        let request: CreateJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        let record = JobRecord::new(0, "job-test".to_owned(), None, request.visible_rects);
        record.cancel.store(true, Ordering::Release);
        assert!(matches!(
            record.append(JobUpdateDraft::Progress {
                stage: BrowserJobStage::Decoding,
                stage_progress: None,
                overall_progress: Some(0.1),
                current: None,
                total: None,
                message: "Decoding".to_owned(),
            }),
            Err(PublishError::Cancelled)
        ));
        assert_eq!(
            record
                .append(JobUpdateDraft::Cancelled {
                    message: Some("Cancelled".to_owned()),
                })
                .unwrap()
                .sequence(),
            1
        );
        assert!(record.replay_after(1).terminal);
    }

    #[test]
    fn host_check_accepts_only_the_exact_ipv4_listener() {
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("127.0.0.1:43127"));
        assert!(valid_host(&headers, 43127));

        headers.insert(HOST, HeaderValue::from_static("localhost:43127"));
        assert!(!valid_host(&headers, 43127));
        headers.insert(HOST, HeaderValue::from_static("127.0.0.1:43127"));
        headers.append(HOST, HeaderValue::from_static("127.0.0.1:43127"));
        assert!(!valid_host(&headers, 43127));
    }
}
