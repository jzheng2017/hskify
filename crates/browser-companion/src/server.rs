//! Secure `/browser/v1` loopback service backed by Koharu's cleaning pipeline.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes, to_bytes};
use axum::extract::multipart::{Field, MultipartRejection};
use axum::extract::{DefaultBodyLimit, Extension, Multipart, Path, Request, State};
use axum::http::header::{
    ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
    ACCESS_CONTROL_MAX_AGE, AUTHORIZATION, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_TYPE, HOST,
    ORIGIN, VARY,
};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::Utc;
use http_body::{Body as HttpBody, Frame, SizeHint};
use image::{GenericImageView, ImageFormat, ImageReader, Limits};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::sync::{Mutex as AsyncMutex, OwnedSemaphorePermit, Semaphore};
use tokio::time::sleep;
use uuid::Uuid;

use crate::contracts::{
    BrowserCapabilities, BrowserJobCreated, BrowserJobRequest, BrowserJobResult, BrowserJobState,
    BrowserJobStatus, BrowserSetupState, BrowserSetupStatus, ErrorResponse, HealthResponse,
    HealthStatus, HskLevel, LookupRequest, NativeReadyResponse, NativeReadyType,
    RetranslateRequest, Validate,
};
use crate::crypto::{SECRET_BYTES, decode_secret, generate_secret, secrets_equal, sha256_hex};
use crate::fixtures;
use crate::origin::validate_extension_origin;
use crate::pipeline_adapter::{
    CleaningError, CleaningInput, CleaningOutput, CleaningPipeline, CleaningProgress,
    KoharuPipeline, RetranslationInput, RetranslationOutput,
};
use crate::{CONTROL_HEADER, PROTOCOL_HEADER};

const INTERNAL_SESSION_PATH: &str = "/browser-internal/v1/session";
const MAX_INTERNAL_BODY_BYTES: usize = 4 * 1024;
const MAX_LOOKUP_BODY_BYTES: usize = 16 * 1024;
const MAX_RETRANSLATE_BODY_BYTES: usize = 64 * 1024;
const MAX_SESSIONS: usize = 64;
const DEFAULT_MAX_RETAINED_JOBS: usize = 128;

#[derive(Debug, Clone)]
pub struct ServerLimits {
    pub max_upload_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_http_body_bytes: usize,
    pub max_pixels: u64,
    pub max_dimension: u32,
    pub max_decoded_bytes: u64,
    pub max_clean_blob_bytes: usize,
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
            max_clean_blob_bytes: 64 * MIB,
            max_retained_jobs: DEFAULT_MAX_RETAINED_JOBS,
            max_stored_blob_bytes: 256 * MIB,
            max_concurrent_requests: 4,
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
            idle_timeout: Duration::from_secs(10 * 60),
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
    sha256: String,
}

#[derive(Debug)]
struct JobRecord {
    sequence: u64,
    status: RwLock<BrowserJobStatus>,
    result: RwLock<Option<BrowserJobResult>>,
    request: RwLock<BrowserJobRequest>,
    source_bytes: Mutex<Option<Arc<[u8]>>>,
    cancel: Arc<AtomicBool>,
    active: AtomicBool,
}

impl JobRecord {
    fn new(
        sequence: u64,
        status: BrowserJobStatus,
        source_bytes: Arc<[u8]>,
        request: BrowserJobRequest,
    ) -> Self {
        Self {
            sequence,
            status: RwLock::new(status),
            result: RwLock::new(None),
            request: RwLock::new(request),
            source_bytes: Mutex::new(Some(source_bytes)),
            cancel: Arc::new(AtomicBool::new(false)),
            active: AtomicBool::new(true),
        }
    }

    fn status(&self) -> BrowserJobStatus {
        self.status
            .read()
            .expect("job status lock poisoned")
            .clone()
    }

    fn result(&self) -> Option<BrowserJobResult> {
        self.result
            .read()
            .expect("job result lock poisoned")
            .clone()
    }

    fn request(&self) -> BrowserJobRequest {
        self.request
            .read()
            .expect("job request lock poisoned")
            .clone()
    }

    fn replace_request(&self, request: BrowserJobRequest) {
        *self.request.write().expect("job request lock poisoned") = request;
    }

    fn source_bytes(&self) -> Option<Arc<[u8]>> {
        self.source_bytes
            .lock()
            .expect("job source lock poisoned")
            .clone()
    }

    fn release_source(&self) {
        self.source_bytes
            .lock()
            .expect("job source lock poisoned")
            .take();
    }

    fn retained_source_bytes(&self) -> usize {
        self.source_bytes
            .lock()
            .expect("job source lock poisoned")
            .as_ref()
            .map_or(0, |bytes| bytes.len())
    }

    fn update_progress(&self, progress: CleaningProgress) -> bool {
        let mut current = self.status.write().expect("job status lock poisoned");
        if self.cancel.load(Ordering::Acquire) || current.state != BrowserJobState::Running {
            return false;
        }
        let overall_progress = match (current.overall_progress, progress.overall_progress) {
            (Some(previous), Some(next)) => Some(previous.max(next)),
            (previous, next) => next.or(previous),
        };
        let stage_progress = if current.stage == progress.stage {
            current.stage_progress
        } else {
            None
        };
        *current = BrowserJobStatus {
            revision: current.revision.saturating_add(1),
            job_id: current.job_id.clone(),
            state: BrowserJobState::Running,
            stage: progress.stage,
            stage_progress,
            overall_progress,
            current: progress.current,
            total: progress.total,
            message: progress.message,
            error_code: None,
        };
        true
    }

    fn is_evictable(&self) -> bool {
        !self.active.load(Ordering::Acquire)
            && self.status.read().expect("job status lock poisoned").state
                != BrowserJobState::Running
    }
}

#[derive(Debug, Default)]
struct Storage {
    jobs: HashMap<String, Arc<JobRecord>>,
    blobs: HashMap<String, StoredBlob>,
    next_job_sequence: u64,
}

impl Storage {
    fn next_sequence(&mut self) -> u64 {
        let sequence = self.next_job_sequence;
        self.next_job_sequence = self.next_job_sequence.saturating_add(1);
        sequence
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

    fn identical_blob_id(&self, sha256: &str, bytes: &[u8], content_type: &str) -> Option<String> {
        self.blobs
            .iter()
            .filter(|(_, blob)| {
                blob.sha256 == sha256
                    && blob.content_type == content_type
                    && blob.bytes.as_ref() == bytes
            })
            .map(|(blob_id, _)| blob_id)
            .min()
            .cloned()
    }

    fn oldest_evictable_job_id(&self) -> Option<String> {
        self.jobs
            .iter()
            .filter(|(_, job)| job.is_evictable())
            .min_by(|(left_id, left), (right_id, right)| {
                left.sequence
                    .cmp(&right.sequence)
                    .then_with(|| left_id.cmp(right_id))
            })
            .map(|(job_id, _)| job_id.clone())
    }

    fn evict_job(&mut self, job_id: &str) {
        let Some(job) = self.jobs.remove(job_id) else {
            return;
        };
        let Some(result) = job.result() else {
            return;
        };
        let blob_id = result.clean_image_blob_id;
        let still_referenced = self
            .jobs
            .values()
            .filter_map(|retained| retained.result())
            .any(|retained| retained.clean_image_blob_id == blob_id);
        if !still_referenced {
            self.blobs.remove(&blob_id);
        }
    }

    fn make_room_for_job(
        &mut self,
        limits: &ServerLimits,
        added_source_bytes: usize,
    ) -> Result<(), ApiError> {
        if added_source_bytes > limits.max_stored_blob_bytes {
            return Err(ApiError::too_many_requests(
                "BLOB_LIMIT_REACHED",
                "The browser cache has reached its bounded capacity.",
            ));
        }

        loop {
            let jobs_full = self.jobs.len() >= limits.max_retained_jobs;
            let blobs_full = self.retained_bytes().saturating_add(added_source_bytes)
                > limits.max_stored_blob_bytes;
            if !jobs_full && !blobs_full {
                return Ok(());
            }

            let Some(job_id) = self.oldest_evictable_job_id() else {
                return if jobs_full {
                    Err(ApiError::too_many_requests(
                        "JOB_LIMIT_REACHED",
                        "All retained browser jobs are still active.",
                    ))
                } else {
                    Err(ApiError::too_many_requests(
                        "BLOB_LIMIT_REACHED",
                        "Active browser jobs still occupy the bounded blob cache.",
                    ))
                };
            };
            self.evict_job(&job_id);
        }
    }

    fn make_room_for_blob(
        &mut self,
        limits: &ServerLimits,
        added_blob_bytes: usize,
    ) -> Result<(), ApiError> {
        if added_blob_bytes > limits.max_stored_blob_bytes {
            return Err(ApiError::too_many_requests(
                "BLOB_LIMIT_REACHED",
                "The cleaned image exceeds the bounded browser cache.",
            ));
        }
        while self.retained_bytes().saturating_add(added_blob_bytes) > limits.max_stored_blob_bytes
        {
            let Some(job_id) = self.oldest_evictable_job_id() else {
                return Err(ApiError::too_many_requests(
                    "BLOB_LIMIT_REACHED",
                    "Active browser jobs still occupy the bounded blob cache.",
                ));
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
    pipeline_gate: AsyncMutex<()>,
    sessions: Mutex<Vec<Session>>,
    storage: RwLock<Storage>,
    lifecycle: Mutex<Lifecycle>,
    request_capacity: Arc<Semaphore>,
    active_jobs: AtomicUsize,
}

impl BridgeState {
    pub fn new(
        config: BridgeConfig,
        control_secret: [u8; SECRET_BYTES],
        cache_root: PathBuf,
    ) -> Arc<Self> {
        Self::with_pipeline(
            config,
            control_secret,
            Arc::new(KoharuPipeline::new(cache_root)),
        )
    }

    fn with_pipeline(
        config: BridgeConfig,
        control_secret: [u8; SECRET_BYTES],
        pipeline: Arc<dyn CleaningPipeline>,
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
            pipeline_gate: AsyncMutex::new(()),
            sessions: Mutex::new(Vec::new()),
            storage: RwLock::new(Storage::default()),
            lifecycle: Mutex::new(Lifecycle {
                last_activity: Instant::now(),
                admitted_requests: 0,
                shutdown_latched: false,
            }),
            request_capacity,
            active_jobs: AtomicUsize::new(0),
        })
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
            protocol_version: crate::contracts::PROTOCOL_VERSION,
            engine_version: fixtures::health().engine_version,
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
                models_ready: self.pipeline.resources_ready(),
            },
        })
    }

    fn touch(&self) {
        self.lifecycle
            .lock()
            .expect("lifecycle lock poisoned")
            .last_activity = Instant::now();
    }

    #[cfg(test)]
    fn admitted_request_count(&self) -> usize {
        self.lifecycle
            .lock()
            .expect("lifecycle lock poisoned")
            .admitted_requests
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
            let same_secret = secrets_equal(&session.token, &candidate);
            accepted |= same_secret && session.origin == origin;
        }
        accepted
    }

    fn control_authenticates(&self, candidate: &str) -> bool {
        decode_secret(candidate)
            .map(|candidate| secrets_equal(&self.control_secret, &candidate))
            .unwrap_or(false)
    }

    fn reserve_uploaded_job(
        &self,
        source_bytes: Arc<[u8]>,
        request: &BrowserJobRequest,
    ) -> Result<(String, Arc<JobRecord>), ApiError> {
        let mut storage = self.storage.write().expect("storage lock poisoned");
        storage.make_room_for_job(&self.config.limits, source_bytes.len())?;
        let job_id = loop {
            let candidate = format!("job-{}", Uuid::new_v4());
            if !storage.jobs.contains_key(&candidate) {
                break candidate;
            }
        };
        let status = BrowserJobStatus {
            revision: 1,
            job_id: job_id.clone(),
            state: BrowserJobState::Running,
            stage: crate::contracts::BrowserJobStage::Queued,
            stage_progress: None,
            overall_progress: Some(0.0),
            current: None,
            total: None,
            message: "Queued for local cleaning and translation".to_owned(),
            error_code: None,
        };
        status.validate().map_err(|_| ApiError::internal())?;
        let sequence = storage.next_sequence();
        let record = Arc::new(JobRecord::new(
            sequence,
            status,
            source_bytes,
            request.clone(),
        ));
        let previous = storage.jobs.insert(job_id.clone(), record.clone());
        debug_assert!(previous.is_none());
        self.active_jobs.fetch_add(1, Ordering::AcqRel);
        drop(storage);
        self.touch();

        Ok((job_id, record))
    }

    fn complete_job(
        &self,
        record: &JobRecord,
        request: &BrowserJobRequest,
        output: CleaningOutput,
    ) -> Result<bool, CleaningError> {
        if output.clean_image.len() > self.config.limits.max_clean_blob_bytes {
            return Err(CleaningError {
                code: "CLEAN_IMAGE_TOO_LARGE",
                message: "The Koharu cleaned image exceeds the browser blob limit.".to_owned(),
            });
        }
        let mut status = record.status.write().expect("job status lock poisoned");
        if record.cancel.load(Ordering::Acquire) || status.state != BrowserJobState::Running {
            return Ok(false);
        }

        let content_type = match output.clean_image_mime_type {
            crate::contracts::CleanImageMimeType::Png => "image/png",
            crate::contracts::CleanImageMimeType::Webp => "image/webp",
        };
        let clean_sha256 = sha256_hex(&output.clean_image);
        record.release_source();
        let mut storage = self.storage.write().expect("storage lock poisoned");
        let existing_blob_id =
            storage.identical_blob_id(&clean_sha256, &output.clean_image, content_type);
        let (blob_id, new_blob) = match existing_blob_id {
            Some(blob_id) => (blob_id, None),
            None => {
                let blob_id = loop {
                    let candidate = format!("blob-{}", Uuid::new_v4());
                    if !storage.blobs.contains_key(&candidate) {
                        break candidate;
                    }
                };
                let blob = StoredBlob {
                    bytes: output.clean_image.into(),
                    content_type,
                    sha256: clean_sha256,
                };
                (blob_id, Some(blob))
            }
        };
        let added_blob_bytes = new_blob.as_ref().map_or(0, |blob| blob.bytes.len());
        storage
            .make_room_for_blob(&self.config.limits, added_blob_bytes)
            .map_err(|error| CleaningError {
                code: error.code,
                message: error.message.to_owned(),
            })?;
        let result = BrowserJobResult {
            protocol_version: crate::contracts::PROTOCOL_VERSION,
            job_id: status.job_id.clone(),
            source_sha256: request.source_sha256.clone(),
            source_width: request.natural_width,
            source_height: request.natural_height,
            clean_image_blob_id: blob_id.clone(),
            clean_image_mime_type: output.clean_image_mime_type,
            regions: output.regions,
            warnings: output.warnings,
            cache: output.cache,
        };
        result.validate().map_err(|error| CleaningError {
            code: "INVALID_PIPELINE_RESULT",
            message: format!("Koharu result failed browser protocol validation: {error}"),
        })?;
        if let Some(blob) = new_blob {
            let previous = storage.blobs.insert(blob_id, blob);
            debug_assert!(previous.is_none());
        }
        *record.result.write().expect("job result lock poisoned") = Some(result);
        *status = BrowserJobStatus {
            revision: status.revision.saturating_add(1),
            job_id: status.job_id.clone(),
            state: BrowserJobState::Complete,
            stage: crate::contracts::BrowserJobStage::Complete,
            stage_progress: Some(1.0),
            overall_progress: Some(1.0),
            current: None,
            total: None,
            message: "Local cleaning and HSK translation complete".to_owned(),
            error_code: None,
        };
        drop(storage);
        self.touch();
        Ok(true)
    }

    fn begin_retranslation(
        &self,
        record: &Arc<JobRecord>,
        request: RetranslateRequest,
    ) -> Result<(BrowserJobRequest, BrowserJobResult, bool), ApiError> {
        let mut status = record.status.write().expect("job status lock poisoned");
        if status.state != BrowserJobState::Complete {
            return Err(ApiError::conflict(
                "JOB_NOT_COMPLETE",
                "Only a complete browser job can be retranslated.",
            ));
        }
        let base_result = record.result().ok_or_else(ApiError::internal)?;
        let previous_request = record.request();
        let mut next_request = previous_request.clone();
        next_request.settings.hsk_standard = request.settings.hsk_standard;
        next_request.settings.hsk_level = request.settings.hsk_level;
        next_request.preceding_context = request.preceding_context;
        next_request.validate().map_err(|_| {
            ApiError::bad_request(
                "INVALID_RETRANSLATE_REQUEST",
                "The retranslation request failed protocol validation.",
            )
        })?;
        let translation_cache_hit = previous_request.settings.hsk_level
            == next_request.settings.hsk_level
            && dialogue_contexts_equal(
                previous_request.preceding_context.as_deref(),
                next_request.preceding_context.as_deref(),
            );

        record.cancel.store(false, Ordering::Release);
        if record
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(ApiError::conflict(
                "JOB_ALREADY_RUNNING",
                "The browser job is already running.",
            ));
        }
        *status = BrowserJobStatus {
            revision: status.revision.saturating_add(1),
            job_id: status.job_id.clone(),
            state: BrowserJobState::Running,
            stage: crate::contracts::BrowserJobStage::Queued,
            stage_progress: None,
            overall_progress: Some(0.55),
            current: None,
            total: None,
            message: if translation_cache_hit {
                "Reusing cached HSK translation".to_owned()
            } else {
                "Queued for HSK retranslation; cleaning caches retained".to_owned()
            },
            error_code: None,
        };
        self.active_jobs.fetch_add(1, Ordering::AcqRel);
        self.touch();
        Ok((next_request, base_result, translation_cache_hit))
    }

    fn complete_retranslation(
        &self,
        record: &JobRecord,
        request: BrowserJobRequest,
        base: BrowserJobResult,
        output: RetranslationOutput,
    ) -> Result<bool, CleaningError> {
        let mut status = record.status.write().expect("job status lock poisoned");
        if record.cancel.load(Ordering::Acquire) || status.state != BrowserJobState::Running {
            return Ok(false);
        }
        let result = BrowserJobResult {
            protocol_version: crate::contracts::PROTOCOL_VERSION,
            job_id: status.job_id.clone(),
            source_sha256: base.source_sha256,
            source_width: base.source_width,
            source_height: base.source_height,
            clean_image_blob_id: base.clean_image_blob_id,
            clean_image_mime_type: base.clean_image_mime_type,
            regions: output.regions,
            warnings: output.warnings,
            cache: output.cache,
        };
        result.validate().map_err(|error| CleaningError {
            code: "INVALID_PIPELINE_RESULT",
            message: format!("Retranslation failed browser protocol validation: {error}"),
        })?;
        *record.result.write().expect("job result lock poisoned") = Some(result);
        record.replace_request(request);
        *status = BrowserJobStatus {
            revision: status.revision.saturating_add(1),
            job_id: status.job_id.clone(),
            state: BrowserJobState::Complete,
            stage: crate::contracts::BrowserJobStage::Complete,
            stage_progress: Some(1.0),
            overall_progress: Some(1.0),
            current: None,
            total: None,
            message: "HSK retranslation complete; cleaning caches retained".to_owned(),
            error_code: None,
        };
        self.touch();
        Ok(true)
    }
}

fn dialogue_contexts_equal(
    left: Option<&[crate::contracts::DialogueContext]>,
    right: Option<&[crate::contracts::DialogueContext]>,
) -> bool {
    let left = left.unwrap_or_default();
    let right = right.unwrap_or_default();
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.source_english == right.source_english && left.chinese == right.chinese
        })
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

    fn conflict(code: &'static str, message: &'static str) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code,
            message,
            retryable: true,
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
            protocol_version: crate::contracts::PROTOCOL_VERSION,
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
        .route("/browser/v1/health", get(health))
        .route("/browser/v1/setup", get(setup))
        .route("/browser/v1/setup/models", post(setup_models))
        .route("/browser/v1/jobs", post(create_job))
        .route(
            "/browser/v1/jobs/{job_id}",
            get(job_status).delete(cancel_job),
        )
        .route("/browser/v1/jobs/{job_id}/result", get(job_result))
        .route(
            "/browser/v1/jobs/{job_id}/retranslate",
            post(retranslate_job),
        )
        .route("/browser/v1/lookup", post(lookup))
        .route("/browser/v1/blobs/{blob_id}", get(blob))
        .route("/browser/v1/fonts/{font_id}", get(font))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(max_body))
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
    let is_browser = path == "/browser/v1" || path.starts_with("/browser/v1/");
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

    let origin = match single_header(request.headers(), ORIGIN.as_str()) {
        Some(value)
            if validate_extension_origin(value).is_ok() && state.origin_has_session(value) =>
        {
            value.to_owned()
        }
        _ => return ApiError::unauthorized().into_response(),
    };

    if request.method() == Method::OPTIONS {
        return preflight(request.headers(), &origin);
    }

    if single_header(request.headers(), PROTOCOL_HEADER)
        != Some(crate::contracts::PROTOCOL_VERSION.to_string().as_str())
    {
        return with_cors(
            ApiError::bad_request(
                "PROTOCOL_REQUIRED",
                "X-HSK-Manga-Protocol must be exactly 1.",
            )
            .into_response(),
            &origin,
        );
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
    if !matches!(method, "GET" | "POST" | "DELETE") {
        return ApiError::bad_request("INVALID_PREFLIGHT", "The requested method is not allowed.")
            .into_response();
    }
    let Some(requested_headers) = single_header(headers, "access-control-request-headers") else {
        return ApiError::bad_request("INVALID_PREFLIGHT", "Missing preflight headers.")
            .into_response();
    };
    let mut saw_authorization = false;
    let mut saw_protocol = false;
    for value in requested_headers.split(',').map(str::trim) {
        if value.eq_ignore_ascii_case("authorization") {
            saw_authorization = true;
        } else if value.eq_ignore_ascii_case(PROTOCOL_HEADER) {
            saw_protocol = true;
        } else if !value.eq_ignore_ascii_case("content-type") {
            return ApiError::bad_request(
                "INVALID_PREFLIGHT",
                "The requested header is not allowed.",
            )
            .into_response();
        }
    }
    if !saw_authorization || !saw_protocol {
        return ApiError::bad_request(
            "INVALID_PREFLIGHT",
            "Authorization and protocol headers are required.",
        )
        .into_response();
    }

    let mut response = StatusCode::NO_CONTENT.into_response();
    let headers = response.headers_mut();
    headers.insert(
        ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_str(method).expect("known method"),
    );
    headers.insert(
        ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("authorization, content-type, x-hsk-manga-protocol"),
    );
    headers.insert(ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("300"));
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
    let response = state.issue_session(&request.extension_origin)?;
    response.validate().map_err(|_| ApiError::internal())?;
    Ok(Json(response))
}

async fn health(State(state): State<Arc<BridgeState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        protocol_version: crate::contracts::PROTOCOL_VERSION,
        engine_version: fixtures::health().engine_version,
        status: HealthStatus::Ready,
        setup_state: if state.pipeline.resources_ready() {
            BrowserSetupState::Ready
        } else {
            BrowserSetupState::MissingModels
        },
    })
}

async fn setup(State(state): State<Arc<BridgeState>>) -> Json<BrowserSetupStatus> {
    Json(resource_setup_status(&state))
}

async fn setup_models(State(state): State<Arc<BridgeState>>) -> Json<BrowserSetupStatus> {
    Json(resource_setup_status(&state))
}

fn resource_setup_status(state: &BridgeState) -> BrowserSetupStatus {
    if state.pipeline.resources_ready() {
        BrowserSetupStatus {
            state: BrowserSetupState::Ready,
            selected_pack_id: Some("qwen3.5-4b-hsk2-v1".to_owned()),
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
            selected_pack_id: None,
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
    let job_request: BrowserJobRequest = serde_json::from_slice(&metadata).map_err(|_| {
        ApiError::bad_request("INVALID_JOB_REQUEST", "The job metadata is invalid.")
    })?;
    job_request.validate().map_err(|_| {
        ApiError::bad_request(
            "INVALID_JOB_REQUEST",
            "The job metadata failed protocol validation.",
        )
    })?;
    if field_mime.as_deref() != Some(job_request.source_mime_type.as_str()) {
        return Err(ApiError::unsupported_media(
            "The multipart image MIME type does not match the request.",
        ));
    }

    let limits = state.config.limits.clone();
    let validation_request = job_request.clone();
    let validated = tokio::task::spawn_blocking(move || {
        let _admission = admission;
        validate_image_upload(image, &validation_request, &limits)
    })
    .await
    .map_err(|_| ApiError::internal())??;

    let source_bytes: Arc<[u8]> = validated.source.into();
    let (job_id, record) = state.reserve_uploaded_job(source_bytes, &job_request)?;
    tokio::spawn(run_cleaning_job(state.clone(), record, job_request));

    Ok((
        StatusCode::ACCEPTED,
        Json(BrowserJobCreated {
            protocol_version: crate::contracts::PROTOCOL_VERSION,
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
struct ValidatedUpload {
    source: Vec<u8>,
}

fn validate_image_upload(
    image: Vec<u8>,
    request: &BrowserJobRequest,
    limits: &ServerLimits,
) -> Result<ValidatedUpload, ApiError> {
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

    Ok(ValidatedUpload { source: image })
}

async fn run_cleaning_job(
    state: Arc<BridgeState>,
    record: Arc<JobRecord>,
    request: BrowserJobRequest,
) {
    let _pipeline_guard = state.pipeline_gate.lock().await;
    if record.cancel.load(Ordering::Acquire) {
        finish_active(&state, &record);
        return;
    }
    let Some(source_bytes) = record.source_bytes() else {
        fail_job(
            &state,
            &record,
            CleaningError {
                code: "SOURCE_UNAVAILABLE",
                message: "The bounded source upload is no longer available.".to_owned(),
            },
        );
        finish_active(&state, &record);
        return;
    };
    let progress = {
        let state = state.clone();
        let record = record.clone();
        Arc::new(move |progress: CleaningProgress| {
            if record.update_progress(progress) {
                state.touch();
            }
        })
    };
    let output = state
        .pipeline
        .run(
            CleaningInput {
                source_bytes,
                request: request.clone(),
            },
            record.cancel.clone(),
            progress,
        )
        .await;
    match output {
        Ok(output) => match state.complete_job(&record, &request, output) {
            Ok(_) => {}
            Err(error) => fail_job(&state, &record, error),
        },
        Err(error) => {
            if !record.cancel.load(Ordering::Acquire) {
                fail_job(&state, &record, error);
            }
        }
    }
    finish_active(&state, &record);
}

async fn run_retranslation_job(
    state: Arc<BridgeState>,
    record: Arc<JobRecord>,
    request: BrowserJobRequest,
    base_result: BrowserJobResult,
    translation_cache_hit: bool,
) {
    let _pipeline_guard = state.pipeline_gate.lock().await;
    if record.cancel.load(Ordering::Acquire) {
        finish_active(&state, &record);
        return;
    }
    let progress = {
        let state = state.clone();
        let record = record.clone();
        Arc::new(move |progress: CleaningProgress| {
            if record.update_progress(progress) {
                state.touch();
            }
        })
    };
    let output = if translation_cache_hit {
        progress(CleaningProgress {
            stage: crate::contracts::BrowserJobStage::Packaging,
            overall_progress: Some(0.98),
            current: None,
            total: None,
            message: "Packaging cached HSK translation".to_owned(),
        });
        Ok(RetranslationOutput {
            regions: base_result.regions.clone(),
            warnings: base_result.warnings.clone(),
            cache: crate::contracts::BrowserCacheStatus {
                detection_hit: true,
                ocr_hit: true,
                inpaint_hit: true,
                translation_hit: true,
            },
        })
    } else {
        state
            .pipeline
            .retranslate(
                RetranslationInput {
                    request: request.clone(),
                    base_result: base_result.clone(),
                },
                record.cancel.clone(),
                progress,
            )
            .await
    };
    match output {
        Ok(output) => {
            if let Err(error) = state.complete_retranslation(&record, request, base_result, output)
            {
                fail_job(&state, &record, error);
            }
        }
        Err(error) => {
            if !record.cancel.load(Ordering::Acquire) {
                fail_job(&state, &record, error);
            }
        }
    }
    finish_active(&state, &record);
}

fn fail_job(state: &BridgeState, record: &JobRecord, error: CleaningError) {
    record.release_source();
    let mut current = record.status.write().expect("job status lock poisoned");
    if record.cancel.load(Ordering::Acquire) || current.state != BrowserJobState::Running {
        return;
    }
    *current = BrowserJobStatus {
        revision: current.revision.saturating_add(1),
        job_id: current.job_id.clone(),
        state: BrowserJobState::Failed,
        stage: crate::contracts::BrowserJobStage::Failed,
        stage_progress: None,
        overall_progress: current.overall_progress,
        current: None,
        total: None,
        message: error.message,
        error_code: Some(error.code.to_owned()),
    };
    state.touch();
}

fn finish_active(state: &BridgeState, record: &JobRecord) {
    record.release_source();
    if record.active.swap(false, Ordering::AcqRel) {
        let mut lifecycle = state.lifecycle.lock().expect("lifecycle lock poisoned");
        state.active_jobs.fetch_sub(1, Ordering::AcqRel);
        lifecycle.last_activity = Instant::now();
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

async fn job_status(
    State(state): State<Arc<BridgeState>>,
    Path(job_id): Path<String>,
) -> Result<Json<BrowserJobStatus>, ApiError> {
    Ok(Json(find_job(&state, &job_id)?.status()))
}

async fn job_result(
    State(state): State<Arc<BridgeState>>,
    Path(job_id): Path<String>,
) -> Result<Json<BrowserJobResult>, ApiError> {
    let job = find_job(&state, &job_id)?;
    if job.status().state != BrowserJobState::Complete {
        return Err(ApiError::conflict(
            "JOB_NOT_COMPLETE",
            "The browser job has no complete result yet.",
        ));
    }
    Ok(Json(job.result().ok_or_else(ApiError::internal)?))
}

async fn cancel_job(
    State(state): State<Arc<BridgeState>>,
    Path(job_id): Path<String>,
) -> Result<Json<BrowserJobStatus>, ApiError> {
    let job = find_job(&state, &job_id)?;
    job.cancel.store(true, Ordering::Release);
    let (status, did_cancel) = {
        let mut current = job.status.write().expect("job status lock poisoned");
        if current.state == BrowserJobState::Running {
            let status = BrowserJobStatus {
                revision: current.revision.saturating_add(1),
                job_id,
                state: BrowserJobState::Cancelled,
                stage: crate::contracts::BrowserJobStage::Cancelled,
                stage_progress: None,
                overall_progress: current.overall_progress,
                current: None,
                total: None,
                message: "Cancelled".to_owned(),
                error_code: None,
            };
            status.validate().map_err(|_| ApiError::internal())?;
            *current = status;
            (current.clone(), true)
        } else {
            (current.clone(), false)
        }
    };
    if did_cancel {
        finish_active(&state, &job);
    }
    Ok(Json(status))
}

async fn retranslate_job(
    State(state): State<Arc<BridgeState>>,
    Path(job_id): Path<String>,
    request: Request,
) -> Result<Response, ApiError> {
    let request: RetranslateRequest = parse_json_body(request, MAX_RETRANSLATE_BODY_BYTES).await?;
    request.validate().map_err(|_| {
        ApiError::bad_request(
            "INVALID_RETRANSLATE_REQUEST",
            "The retranslation request failed protocol validation.",
        )
    })?;
    let job = find_job(&state, &job_id)?;
    for _ in 0..100 {
        if !job.active.load(Ordering::Acquire) {
            break;
        }
        tokio::task::yield_now().await;
    }
    let (next_request, base_result, translation_cache_hit) =
        state.begin_retranslation(&job, request)?;
    tokio::spawn(run_retranslation_job(
        state,
        job,
        next_request,
        base_result,
        translation_cache_hit,
    ));
    Ok((
        StatusCode::ACCEPTED,
        Json(BrowserJobCreated {
            protocol_version: crate::contracts::PROTOCOL_VERSION,
            job_id,
        }),
    )
        .into_response())
}

async fn lookup(
    State(state): State<Arc<BridgeState>>,
    request: Request,
) -> Result<Json<crate::contracts::LookupResult>, ApiError> {
    let request: LookupRequest = parse_json_body(request, MAX_LOOKUP_BODY_BYTES).await?;
    request.validate().map_err(|_| {
        ApiError::bad_request(
            "INVALID_LOOKUP_REQUEST",
            "The lookup request failed protocol validation.",
        )
    })?;
    let region = if let (Some(job_id), Some(region_id)) = (&request.job_id, &request.region_id) {
        let job = find_job(&state, job_id)?;
        let job_result = job.result().ok_or_else(|| {
            ApiError::conflict(
                "JOB_NOT_COMPLETE",
                "The browser job has no complete result yet.",
            )
        })?;
        Some(
            job_result
                .regions
                .iter()
                .find(|region| &region.id == region_id)
                .ok_or_else(|| {
                    ApiError::not_found("REGION_NOT_FOUND", "The browser region does not exist.")
                })?
                .clone(),
        )
    } else {
        None
    };
    let result = state
        .pipeline
        .lookup(request.selected_text, region)
        .await
        .map_err(|_| {
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
    Path(blob_id): Path<String>,
) -> Result<Response, ApiError> {
    let blob = state
        .storage
        .read()
        .expect("storage lock poisoned")
        .blobs
        .get(&blob_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("BLOB_NOT_FOUND", "The browser blob does not exist."))?;
    Ok(arc_bytes_response(blob.bytes, blob.content_type))
}

async fn font(Path(font_id): Path<String>) -> Result<Response, ApiError> {
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
    use std::convert::Infallible;
    use std::io::Write as _;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    use axum::body::{Bytes, to_bytes};
    use http_body::{Body as HttpBody, Frame};
    use image::{DynamicImage, RgbImage, RgbaImage};
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;

    const ORIGIN_VALUE: &str = "moz-extension://00000000-0000-4000-8000-000000000001";

    struct PanicIfReadBody;

    impl HttpBody for PanicIfReadBody {
        type Data = Bytes;
        type Error = Infallible;

        fn poll_frame(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            panic!("security middleware must reject this request before reading its body")
        }
    }

    #[derive(Debug)]
    struct GatedBodyState {
        polled: AtomicBool,
        released: AtomicBool,
        waker: Mutex<Option<Waker>>,
    }

    impl GatedBodyState {
        fn release(&self) {
            self.released.store(true, Ordering::Release);
            if let Some(waker) = self.waker.lock().expect("gate waker lock").take() {
                waker.wake();
            }
        }
    }

    #[derive(Debug)]
    struct GatedBody {
        state: Arc<GatedBodyState>,
        bytes: Option<Bytes>,
    }

    impl GatedBody {
        fn new(bytes: Vec<u8>) -> (Self, Arc<GatedBodyState>) {
            let state = Arc::new(GatedBodyState {
                polled: AtomicBool::new(false),
                released: AtomicBool::new(false),
                waker: Mutex::new(None),
            });
            (
                Self {
                    state: state.clone(),
                    bytes: Some(Bytes::from(bytes)),
                },
                state,
            )
        }
    }

    impl HttpBody for GatedBody {
        type Data = Bytes;
        type Error = Infallible;

        fn poll_frame(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            self.state.polled.store(true, Ordering::Release);
            if !self.state.released.load(Ordering::Acquire) {
                *self.state.waker.lock().expect("gate waker lock") = Some(context.waker().clone());
                return Poll::Pending;
            }
            Poll::Ready(self.bytes.take().map(|bytes| Ok(Frame::data(bytes))))
        }
    }

    struct FixtureCleaningPipeline {
        stage_delay: Duration,
        hsk_control: Arc<hsk_control::HskControl>,
        resources_ready: bool,
    }

    #[async_trait::async_trait]
    impl CleaningPipeline for FixtureCleaningPipeline {
        async fn run(
            &self,
            input: CleaningInput,
            cancel: Arc<AtomicBool>,
            progress: crate::pipeline_adapter::CleaningProgressSink,
        ) -> std::result::Result<CleaningOutput, CleaningError> {
            for (index, (stage, message)) in [
                (
                    crate::contracts::BrowserJobStage::Detecting,
                    "Detecting fixture text",
                ),
                (
                    crate::contracts::BrowserJobStage::Ocr,
                    "Reading fixture text",
                ),
                (
                    crate::contracts::BrowserJobStage::Inpainting,
                    "Cleaning fixture text",
                ),
                (
                    crate::contracts::BrowserJobStage::Packaging,
                    "Packaging fixture result",
                ),
            ]
            .into_iter()
            .enumerate()
            {
                sleep(self.stage_delay).await;
                if cancel.load(Ordering::Acquire) {
                    return Err(CleaningError {
                        code: "CANCELLED",
                        message: "Fixture cleaning was cancelled.".to_owned(),
                    });
                }
                progress(CleaningProgress {
                    stage,
                    overall_progress: Some((index + 1) as f32 / 5.0),
                    current: Some(u32::try_from(index + 1).unwrap()),
                    total: Some(4),
                    message: message.to_owned(),
                });
            }
            let fixture = fixtures::result("fixture-job", "fixture-blob", &input.request);
            let clean_image_mime_type = match image::guess_format(input.source_bytes.as_ref()) {
                Ok(ImageFormat::WebP) => crate::contracts::CleanImageMimeType::Webp,
                _ => crate::contracts::CleanImageMimeType::Png,
            };
            Ok(CleaningOutput {
                clean_image: input.source_bytes.to_vec(),
                clean_image_mime_type,
                regions: fixture.regions,
                warnings: fixture.warnings,
                cache: fixture.cache,
            })
        }

        async fn retranslate(
            &self,
            input: RetranslationInput,
            cancel: Arc<AtomicBool>,
            progress: crate::pipeline_adapter::CleaningProgressSink,
        ) -> std::result::Result<RetranslationOutput, CleaningError> {
            if cancel.load(Ordering::Acquire) {
                return Err(CleaningError {
                    code: "CANCELLED",
                    message: "Fixture retranslation was cancelled.".to_owned(),
                });
            }
            progress(CleaningProgress {
                stage: crate::contracts::BrowserJobStage::HskRewriting,
                overall_progress: Some(0.75),
                current: Some(1),
                total: Some(1),
                message: "Rewriting fixture translation".to_owned(),
            });
            sleep(self.stage_delay).await;
            let mut regions = input.base_result.regions;
            for region in &mut regions {
                region.displayed_chinese = "你好".to_owned();
                region.pinyin = "nǐ hǎo".to_owned();
                region.vocabulary.requested_hsk_level = input.request.settings.hsk_level;
                region.vocabulary.strictly_valid = true;
                region.vocabulary.exceptions.clear();
            }
            Ok(RetranslationOutput {
                regions,
                warnings: input
                    .base_result
                    .warnings
                    .into_iter()
                    .filter(|warning| {
                        !matches!(
                            warning.code,
                            crate::contracts::BrowserWarningCode::HskException
                                | crate::contracts::BrowserWarningCode::HskRewriteFailed
                        )
                    })
                    .collect(),
                cache: crate::contracts::BrowserCacheStatus {
                    detection_hit: true,
                    ocr_hit: true,
                    inpaint_hit: true,
                    translation_hit: false,
                },
            })
        }

        async fn lookup(
            &self,
            selected_text: String,
            region: Option<crate::contracts::BrowserRegion>,
        ) -> std::result::Result<crate::contracts::LookupResult, CleaningError> {
            let proper_names = region
                .as_ref()
                .map(crate::pipeline_adapter::proper_names_from_region)
                .unwrap_or_default();
            let context = region
                .as_ref()
                .map(|region| hsk_control::LookupRegionContext {
                    displayed_chinese: region.displayed_chinese.clone(),
                    faithful_chinese: region.faithful_chinese.clone(),
                    source_english: region.source_english.clone(),
                });
            Ok(crate::pipeline_adapter::browser_lookup_result(
                self.hsk_control
                    .lookup_with_region_context(&selected_text, &proper_names, context),
            ))
        }

        fn resources_ready(&self) -> bool {
            self.resources_ready
        }
    }

    fn test_hsk_control() -> Arc<hsk_control::HskControl> {
        Arc::new(
            hsk_control::HskControl::from_json_with_policy(
                include_str!("../../../data/hsk/test-seed.normalized.json"),
                include_str!("../../../data/dictionary/test-seed.normalized.json"),
                hsk_control::LoadPolicy::AllowIncompleteTestSeed,
            )
            .expect("valid project-authored HSK test seed"),
        )
    }

    fn test_state(
        config: BridgeConfig,
        secret: [u8; SECRET_BYTES],
        stage_delay: Duration,
    ) -> Arc<BridgeState> {
        BridgeState::with_pipeline(
            config,
            secret,
            Arc::new(FixtureCleaningPipeline {
                stage_delay,
                hsk_control: test_hsk_control(),
                resources_ready: true,
            }),
        )
    }

    fn state_with_delay(delay: Duration) -> (Arc<BridgeState>, String) {
        let mut config = BridgeConfig::for_port(43127);
        config.idle_timeout = Duration::from_secs(30);
        let state = test_state(config, [9_u8; SECRET_BYTES], delay);
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        (state, token)
    }

    fn authorized(method: Method, path: &str, token: &str) -> axum::http::request::Builder {
        Request::builder()
            .method(method)
            .uri(path)
            .header(HOST, "127.0.0.1:43127")
            .header(ORIGIN, ORIGIN_VALUE)
            .header(PROTOCOL_HEADER, "1")
            .header(AUTHORIZATION, format!("Bearer {token}"))
    }

    async fn body_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn png() -> Vec<u8> {
        png_with_color([255, 255, 255, 255])
    }

    fn png_with_color(color: [u8; 4]) -> Vec<u8> {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 3, image::Rgba(color)));
        let mut output = Cursor::new(Vec::new());
        image.write_to(&mut output, ImageFormat::Png).unwrap();
        output.into_inner()
    }

    fn jpeg_with_dimensions(width: u32, height: u32) -> Vec<u8> {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(
            width,
            height,
            image::Rgb([127, 96, 64]),
        ));
        let mut output = Cursor::new(Vec::new());
        image.write_to(&mut output, ImageFormat::Jpeg).unwrap();
        output.into_inner()
    }

    fn multipart_body(image: &[u8]) -> (String, Vec<u8>) {
        let boundary = "hsk-manga-test-boundary";
        let mut request: Value = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        request["sourceSha256"] = json!(sha256_hex(image));
        request["naturalWidth"] = json!(2);
        request["naturalHeight"] = json!(3);
        let metadata = serde_json::to_vec(&request).unwrap();
        let mut body = Vec::new();
        write!(
            body,
            "--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n"
        )
        .unwrap();
        body.extend_from_slice(&metadata);
        write!(
            body,
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"fixture.png\"\r\nContent-Type: image/png\r\n\r\n"
        )
        .unwrap();
        body.extend_from_slice(image);
        write!(body, "\r\n--{boundary}--\r\n").unwrap();
        (format!("multipart/form-data; boundary={boundary}"), body)
    }

    async fn submit_job(app: &Router, token: &str, image: &[u8]) -> Response {
        let (content_type, body) = multipart_body(image);
        app.clone()
            .oneshot(
                authorized(Method::POST, "/browser/v1/jobs", token)
                    .header(CONTENT_TYPE, content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn submit_accepted_job(app: &Router, token: &str, image: &[u8]) -> String {
        let response = submit_job(app, token, image).await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        body_json(response).await["jobId"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    async fn cancel(app: &Router, token: &str, job_id: &str) {
        let response = app
            .clone()
            .oneshot(
                authorized(Method::DELETE, &format!("/browser/v1/jobs/{job_id}"), token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_json(response).await["state"], "cancelled");
    }

    async fn wait_for_completion(app: &Router, token: &str, job_id: &str) {
        for _ in 0..100 {
            sleep(Duration::from_millis(10)).await;
            let response = app
                .clone()
                .oneshot(
                    authorized(Method::GET, &format!("/browser/v1/jobs/{job_id}"), token)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            if body_json(response).await["state"] == "complete" {
                return;
            }
        }
        panic!("fixture job should reach a terminal status");
    }

    #[tokio::test]
    async fn missing_language_resources_are_truthful_in_handshake_health_and_setup() {
        let state = BridgeState::with_pipeline(
            BridgeConfig::for_port(43127),
            [13_u8; SECRET_BYTES],
            Arc::new(FixtureCleaningPipeline {
                stage_delay: Duration::from_millis(1),
                hsk_control: test_hsk_control(),
                resources_ready: false,
            }),
        );
        let ready = state.issue_session(ORIGIN_VALUE).unwrap();
        assert!(!ready.capabilities.models_ready);
        let token = ready.token;
        let app = router(state);

        let health = app
            .clone()
            .oneshot(
                authorized(Method::GET, "/browser/v1/health", &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body_json(health).await["setupState"], "missing-models");
        let setup = app
            .clone()
            .oneshot(
                authorized(Method::GET, "/browser/v1/setup", &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body_json(setup).await["state"], "missing-models");
    }

    #[tokio::test]
    async fn auth_host_protocol_and_origin_are_enforced_before_routes() {
        let (state, token) = state_with_delay(Duration::from_millis(5));
        let app = router(state);

        let good = app
            .clone()
            .oneshot(
                authorized(Method::GET, "/browser/v1/health", &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(good.status(), StatusCode::OK);
        assert_eq!(
            good.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(),
            ORIGIN_VALUE
        );

        for request in [
            authorized(
                Method::GET,
                "/browser/v1/health",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            )
            .body(Body::empty())
            .unwrap(),
            Request::builder()
                .method(Method::GET)
                .uri("/browser/v1/health")
                .header(HOST, "localhost:43127")
                .header(ORIGIN, ORIGIN_VALUE)
                .header(PROTOCOL_HEADER, "1")
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
            Request::builder()
                .method(Method::GET)
                .uri("/browser/v1/health")
                .header(HOST, "127.0.0.1:43127")
                .header(ORIGIN, "https://attacker.invalid")
                .header(PROTOCOL_HEADER, "1")
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
            Request::builder()
                .method(Method::GET)
                .uri("/browser/v1/health")
                .header(HOST, "127.0.0.1:43127")
                .header(ORIGIN, ORIGIN_VALUE)
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        ] {
            let response = app.clone().oneshot(request).await.unwrap();
            assert!(!response.status().is_success());
        }

        for path in ["/api/v1/projects", "/mcp"] {
            let hidden = app
                .clone()
                .oneshot(
                    authorized(Method::GET, path, &token)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(hidden.status(), StatusCode::NOT_FOUND);
        }
    }

    #[tokio::test]
    async fn security_rejections_do_not_poll_request_bodies() {
        let (state, token) = state_with_delay(Duration::from_millis(5));
        let app = router(state);
        let requests = [
            authorized(
                Method::POST,
                "/browser/v1/jobs",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            )
            .header(CONTENT_TYPE, "multipart/form-data; boundary=x")
            .body(Body::new(PanicIfReadBody))
            .unwrap(),
            Request::builder()
                .method(Method::POST)
                .uri("/browser/v1/jobs")
                .header(HOST, "localhost:43127")
                .header(ORIGIN, ORIGIN_VALUE)
                .header(PROTOCOL_HEADER, "1")
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::new(PanicIfReadBody))
                .unwrap(),
            Request::builder()
                .method(Method::POST)
                .uri("/browser/v1/jobs")
                .header(HOST, "127.0.0.1:43127")
                .header(ORIGIN, "https://attacker.invalid")
                .header(PROTOCOL_HEADER, "1")
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::new(PanicIfReadBody))
                .unwrap(),
            Request::builder()
                .method(Method::POST)
                .uri("/browser/v1/jobs")
                .header(HOST, "127.0.0.1:43127")
                .header(ORIGIN, ORIGIN_VALUE)
                .header(AUTHORIZATION, format!("Bearer {token}"))
                .body(Body::new(PanicIfReadBody))
                .unwrap(),
        ];
        for request in requests {
            assert!(
                !app.clone()
                    .oneshot(request)
                    .await
                    .unwrap()
                    .status()
                    .is_success()
            );
        }
    }

    #[tokio::test]
    async fn authenticated_request_capacity_rejects_concurrency_before_polling_bodies() {
        let mut config = BridgeConfig::for_port(43127);
        config.limits.max_concurrent_requests = 1;
        let state = test_state(config, [8_u8; SECRET_BYTES], Duration::from_secs(30));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state.clone());
        let image = png();
        let (content_type, body) = multipart_body(&image);
        let (gated_body, gate) = GatedBody::new(body);
        let first_app = app.clone();
        let first_token = token.clone();
        let first = tokio::spawn(async move {
            first_app
                .oneshot(
                    authorized(Method::POST, "/browser/v1/jobs", &first_token)
                        .header(CONTENT_TYPE, content_type)
                        .body(Body::new(gated_body))
                        .unwrap(),
                )
                .await
                .unwrap()
        });

        for _ in 0..100 {
            if gate.polled.load(Ordering::Acquire) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(gate.polled.load(Ordering::Acquire));
        assert_eq!(state.admitted_request_count(), 1);

        let mut attempts = tokio::task::JoinSet::new();
        for _ in 0..32 {
            let attempt_app = app.clone();
            let attempt_token = token.clone();
            attempts.spawn(async move {
                attempt_app
                    .oneshot(
                        authorized(Method::POST, "/browser/v1/jobs", &attempt_token)
                            .header(CONTENT_TYPE, "multipart/form-data; boundary=x")
                            .body(Body::new(PanicIfReadBody))
                            .unwrap(),
                    )
                    .await
                    .unwrap()
            });
        }
        while let Some(response) = attempts.join_next().await {
            let response = response.unwrap();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(
                body_json(response).await["code"],
                "REQUEST_CAPACITY_EXHAUSTED"
            );
        }

        gate.release();
        let response = first.await.unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let job_id = body_json(response).await["jobId"]
            .as_str()
            .unwrap()
            .to_owned();
        cancel(&app, &token, &job_id).await;
    }

    #[tokio::test]
    async fn stalled_authenticated_upload_prevents_idle_latch_until_its_job_finishes() {
        let mut config = BridgeConfig::for_port(43127);
        config.idle_timeout = Duration::from_millis(40);
        config.limits.max_concurrent_requests = 2;
        let state = test_state(config, [10_u8; SECRET_BYTES], Duration::from_secs(30));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state.clone());
        let shutdown = tokio::spawn(wait_until_idle(state.clone()));
        let image = png();
        let (content_type, body) = multipart_body(&image);
        let (gated_body, gate) = GatedBody::new(body);
        let upload_app = app.clone();
        let upload_token = token.clone();
        let upload = tokio::spawn(async move {
            upload_app
                .oneshot(
                    authorized(Method::POST, "/browser/v1/jobs", &upload_token)
                        .header(CONTENT_TYPE, content_type)
                        .body(Body::new(gated_body))
                        .unwrap(),
                )
                .await
                .unwrap()
        });

        for _ in 0..100 {
            if gate.polled.load(Ordering::Acquire) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(gate.polled.load(Ordering::Acquire));
        sleep(Duration::from_millis(120)).await;
        assert!(
            !shutdown.is_finished(),
            "idle shutdown must not latch while an authenticated upload is admitted"
        );

        gate.release();
        let response = upload.await.unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let job_id = body_json(response).await["jobId"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(state.active_job_count(), 1);
        sleep(Duration::from_millis(120)).await;
        assert!(
            !shutdown.is_finished(),
            "the job created by the stalled request must keep the daemon alive"
        );

        cancel(&app, &token, &job_id).await;
        tokio::time::timeout(Duration::from_secs(1), shutdown)
            .await
            .expect("idle shutdown after request and job finish")
            .expect("idle shutdown task");
    }

    #[tokio::test]
    async fn cors_preflight_is_exact_and_rejects_extra_headers() {
        let (state, _) = state_with_delay(Duration::from_millis(5));
        let app = router(state);
        let valid = Request::builder()
            .method(Method::OPTIONS)
            .uri("/browser/v1/jobs")
            .header(HOST, "127.0.0.1:43127")
            .header(ORIGIN, ORIGIN_VALUE)
            .header("access-control-request-method", "POST")
            .header(
                "access-control-request-headers",
                "authorization, content-type, x-hsk-manga-protocol",
            )
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(valid).await.unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert_eq!(
            response.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(),
            ORIGIN_VALUE
        );

        let invalid = Request::builder()
            .method(Method::OPTIONS)
            .uri("/browser/v1/jobs")
            .header(HOST, "127.0.0.1:43127")
            .header(ORIGIN, ORIGIN_VALUE)
            .header("access-control-request-method", "POST")
            .header(
                "access-control-request-headers",
                "authorization, x-hsk-manga-protocol, x-evil",
            )
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(invalid).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[tokio::test]
    async fn upload_progress_result_blob_font_and_cancellation_work() {
        let (state, token) = state_with_delay(Duration::from_millis(15));
        let app = router(state);
        let image = png();
        let (content_type, body) = multipart_body(&image);
        let response = app
            .clone()
            .oneshot(
                authorized(Method::POST, "/browser/v1/jobs", &token)
                    .header(CONTENT_TYPE, content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let created = body_json(response).await;
        let job_id = created["jobId"].as_str().unwrap();

        let running = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/jobs/{job_id}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(running.status(), StatusCode::OK);

        let mut completed = false;
        for _ in 0..100 {
            sleep(Duration::from_millis(10)).await;
            let status = app
                .clone()
                .oneshot(
                    authorized(Method::GET, &format!("/browser/v1/jobs/{job_id}"), &token)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            if body_json(status).await["state"] == "complete" {
                completed = true;
                break;
            }
        }
        assert!(completed, "fixture job should reach a terminal status");
        let result_response = app
            .clone()
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/jobs/{job_id}/result"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result_response.status(), StatusCode::OK);
        let result = body_json(result_response).await;
        let blob_id = result["cleanImageBlobId"].as_str().unwrap();
        let region_id = result["regions"][0]["id"].as_str().unwrap();

        let lookup_response = app
            .clone()
            .oneshot(
                authorized(Method::POST, "/browser/v1/lookup", &token)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "selectedText": "离开",
                            "jobId": job_id,
                            "regionId": region_id
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(lookup_response.status(), StatusCode::OK);
        assert_eq!(
            body_json(lookup_response).await["region"]["sourceEnglish"],
            "We have to leave now!"
        );

        let retranslate_response = app
            .clone()
            .oneshot(
                authorized(
                    Method::POST,
                    &format!("/browser/v1/jobs/{job_id}/retranslate"),
                    &token,
                )
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(include_str!(
                    "../../../fixtures/contracts/retranslate.valid.json"
                )))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(retranslate_response.status(), StatusCode::ACCEPTED);
        assert_eq!(body_json(retranslate_response).await["jobId"], job_id);
        wait_for_completion(&app, &token, job_id).await;
        let translated = app
            .clone()
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/jobs/{job_id}/result"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        let translated = body_json(translated).await;
        assert_eq!(translated["cleanImageBlobId"], blob_id);
        assert_eq!(translated["regions"][0]["displayedChinese"], "你好");
        assert_eq!(
            translated["regions"][0]["vocabulary"]["requestedHskLevel"],
            1
        );
        assert_eq!(translated["cache"]["detectionHit"], true);
        assert_eq!(translated["cache"]["ocrHit"], true);
        assert_eq!(translated["cache"]["inpaintHit"], true);
        assert_eq!(translated["cache"]["translationHit"], false);

        let blob_response = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/blobs/{blob_id}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(blob_response.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(blob_response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .as_ref(),
            image
        );

        let font_response = app
            .clone()
            .oneshot(
                authorized(Method::GET, "/browser/v1/fonts/fixture-sans", &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(font_response.status(), StatusCode::OK);
        assert!(
            !to_bytes(font_response.into_body(), 1024)
                .await
                .unwrap()
                .is_empty()
        );

        let (content_type, body) = multipart_body(&png());
        let second = app
            .clone()
            .oneshot(
                authorized(Method::POST, "/browser/v1/jobs", &token)
                    .header(CONTENT_TYPE, content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let second = body_json(second).await;
        let second_id = second["jobId"].as_str().unwrap();
        let cancelled = app
            .clone()
            .oneshot(
                authorized(
                    Method::DELETE,
                    &format!("/browser/v1/jobs/{second_id}"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body_json(cancelled).await["state"], "cancelled");
        sleep(Duration::from_millis(100)).await;
        let still_cancelled = app
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/jobs/{second_id}"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body_json(still_cancelled).await["state"], "cancelled");
    }

    #[tokio::test]
    async fn blob_transfer_is_zero_copy_and_holds_the_global_response_budget() {
        let mut config = BridgeConfig::for_port(43127);
        config.limits.max_concurrent_requests = 1;
        let state = test_state(config, [11_u8; SECRET_BYTES], Duration::from_millis(1));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state.clone());
        let job_id = submit_accepted_job(&app, &token, &png()).await;
        wait_for_completion(&app, &token, &job_id).await;
        let (blob_id, retained_bytes) = {
            let storage = state.storage.read().expect("storage lock");
            let blob_id = storage.jobs[&job_id]
                .result()
                .expect("completed fixture result")
                .clean_image_blob_id;
            let bytes = storage.blobs[&blob_id].bytes.clone();
            (blob_id, bytes)
        };
        let baseline_references = Arc::strong_count(&retained_bytes);

        let blob_response = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/blobs/{blob_id}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(blob_response.status(), StatusCode::OK);
        assert!(
            Arc::strong_count(&retained_bytes) > baseline_references,
            "the response body must retain the stored Arc instead of copying the blob"
        );

        let saturated = app
            .clone()
            .oneshot(
                authorized(Method::GET, "/browser/v1/health", &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(saturated.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            body_json(saturated).await["code"],
            "REQUEST_CAPACITY_EXHAUSTED"
        );

        drop(blob_response);
        let recovered = app
            .clone()
            .oneshot(
                authorized(Method::GET, "/browser/v1/health", &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recovered.status(), StatusCode::OK);
        let _ = body_json(recovered).await;
        assert_eq!(Arc::strong_count(&retained_bytes), baseline_references);
    }

    #[tokio::test]
    async fn active_saturation_recovers_by_evicting_oldest_terminal_jobs() {
        let mut config = BridgeConfig::for_port(43127);
        config.limits.max_retained_jobs = 2;
        let state = test_state(config, [4_u8; SECRET_BYTES], Duration::from_secs(30));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state);
        let image = png();

        let first = submit_accepted_job(&app, &token, &image).await;
        let second = submit_accepted_job(&app, &token, &image).await;
        let saturated = submit_job(&app, &token, &image).await;
        assert_eq!(saturated.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body_json(saturated).await["code"], "JOB_LIMIT_REACHED");

        cancel(&app, &token, &first).await;
        let third = submit_accepted_job(&app, &token, &image).await;
        let first_status = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/jobs/{first}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first_status.status(), StatusCode::NOT_FOUND);
        let second_status = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/jobs/{second}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body_json(second_status).await["state"], "running");

        cancel(&app, &token, &second).await;
        cancel(&app, &token, &third).await;
        let fourth = submit_accepted_job(&app, &token, &image).await;
        let evicted_second = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/jobs/{second}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(evicted_second.status(), StatusCode::NOT_FOUND);
        let retained_third = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/jobs/{third}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body_json(retained_third).await["state"], "cancelled");
        cancel(&app, &token, &fourth).await;
    }

    #[tokio::test]
    async fn identical_clean_content_is_deduplicated_after_pipeline_completion() {
        let image = png();
        let mut config = BridgeConfig::for_port(43127);
        config.limits.max_retained_jobs = 3;
        config.limits.max_stored_blob_bytes = image.len() * 3;
        let state = test_state(config, [3_u8; SECRET_BYTES], Duration::from_millis(1));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state.clone());

        let first = submit_accepted_job(&app, &token, &image).await;
        wait_for_completion(&app, &token, &first).await;
        let second = submit_accepted_job(&app, &token, &image).await;
        wait_for_completion(&app, &token, &second).await;
        {
            let storage = state.storage.read().expect("storage lock");
            assert_eq!(storage.blobs.len(), 1);
            assert_eq!(
                storage.jobs[&first]
                    .result()
                    .expect("first fixture result")
                    .clean_image_blob_id,
                storage.jobs[&second]
                    .result()
                    .expect("second fixture result")
                    .clean_image_blob_id
            );
            let blob = storage.blobs.values().next().unwrap();
            assert_eq!(blob.sha256, sha256_hex(&image));
            assert_eq!(blob.bytes.len(), image.len());
        }
    }

    #[tokio::test]
    async fn terminal_job_eviction_keeps_shared_blobs_until_last_reference_leaves() {
        let mut config = BridgeConfig::for_port(43127);
        config.limits.max_retained_jobs = 2;
        let state = test_state(config, [2_u8; SECRET_BYTES], Duration::from_millis(1));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state.clone());
        let shared_image = png();

        let first = submit_accepted_job(&app, &token, &shared_image).await;
        wait_for_completion(&app, &token, &first).await;
        let second = submit_accepted_job(&app, &token, &shared_image).await;
        wait_for_completion(&app, &token, &second).await;
        let shared_blob_id = {
            let storage = state.storage.read().expect("storage lock");
            storage.jobs[&first]
                .result()
                .expect("first fixture result")
                .clean_image_blob_id
        };

        let third = submit_accepted_job(&app, &token, &png_with_color([255, 0, 0, 255])).await;
        wait_for_completion(&app, &token, &third).await;
        let shared_blob = app
            .clone()
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/blobs/{shared_blob_id}"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(shared_blob.status(), StatusCode::OK);

        let fourth = submit_accepted_job(&app, &token, &png_with_color([0, 0, 255, 255])).await;
        wait_for_completion(&app, &token, &fourth).await;
        let evicted_blob = app
            .clone()
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/blobs/{shared_blob_id}"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(evicted_blob.status(), StatusCode::NOT_FOUND);
        assert_eq!(state.storage.read().expect("storage lock").blobs.len(), 2);
    }

    #[tokio::test]
    async fn completed_job_and_unreferenced_blob_are_reclaimed_under_pressure() {
        let mut config = BridgeConfig::for_port(43127);
        config.limits.max_retained_jobs = 1;
        let state = test_state(config, [1_u8; SECRET_BYTES], Duration::from_millis(1));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state.clone());

        let first = submit_accepted_job(&app, &token, &png()).await;
        wait_for_completion(&app, &token, &first).await;
        for _ in 0..100 {
            if state.active_job_count() == 0 {
                break;
            }
            sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(state.active_job_count(), 0);
        let result = app
            .clone()
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/jobs/{first}/result"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        let blob_id = body_json(result).await["cleanImageBlobId"]
            .as_str()
            .unwrap()
            .to_owned();

        let second = submit_accepted_job(&app, &token, &png_with_color([0, 255, 0, 255])).await;
        let old_job = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/jobs/{first}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(old_job.status(), StatusCode::NOT_FOUND);
        let old_blob = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/blobs/{blob_id}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(old_blob.status(), StatusCode::NOT_FOUND);
        cancel(&app, &token, &second).await;
    }

    #[tokio::test]
    async fn retranslation_reuses_clean_blob_without_eviction() {
        let mut config = BridgeConfig::for_port(43127);
        config.limits.max_retained_jobs = 1;
        let state = test_state(config, [6_u8; SECRET_BYTES], Duration::from_millis(1));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state.clone());

        let original = submit_accepted_job(&app, &token, &png()).await;
        wait_for_completion(&app, &token, &original).await;
        for _ in 0..100 {
            if state.active_job_count() == 0 {
                break;
            }
            sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(state.active_job_count(), 0);
        let blob_id = state.storage.read().expect("storage lock").jobs[&original]
            .result()
            .expect("completed fixture result")
            .clean_image_blob_id;

        let response = app
            .clone()
            .oneshot(
                authorized(
                    Method::POST,
                    &format!("/browser/v1/jobs/{original}/retranslate"),
                    &token,
                )
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(include_str!(
                    "../../../fixtures/contracts/retranslate.valid.json"
                )))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(body_json(response).await["jobId"], original.as_str());
        wait_for_completion(&app, &token, &original).await;

        let original_job = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/jobs/{original}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(original_job.status(), StatusCode::OK);
        let result = app
            .clone()
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/jobs/{original}/result"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body_json(result).await["cleanImageBlobId"], blob_id);
        let referenced_blob = app
            .clone()
            .oneshot(
                authorized(Method::GET, &format!("/browser/v1/blobs/{blob_id}"), &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(referenced_blob.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn identical_retranslation_uses_translation_cache_and_never_reruns_cleaning() {
        let state = test_state(
            BridgeConfig::for_port(43127),
            [12_u8; SECRET_BYTES],
            Duration::from_millis(1),
        );
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state);
        let job_id = submit_accepted_job(&app, &token, &png()).await;
        wait_for_completion(&app, &token, &job_id).await;

        let before = app
            .clone()
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/jobs/{job_id}/result"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        let before = body_json(before).await;
        let response = app
            .clone()
            .oneshot(
                authorized(
                    Method::POST,
                    &format!("/browser/v1/jobs/{job_id}/retranslate"),
                    &token,
                )
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "protocolVersion": 1,
                        "settings": {
                            "hskStandard": "2.0",
                            "hskLevel": 2
                        },
                        "precedingContext": [{
                            "sourceEnglish": "Are you ready?",
                            "chinese": "你准备好了吗？"
                        }]
                    }))
                    .unwrap(),
                ))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        wait_for_completion(&app, &token, &job_id).await;

        let after = app
            .clone()
            .oneshot(
                authorized(
                    Method::GET,
                    &format!("/browser/v1/jobs/{job_id}/result"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        let after = body_json(after).await;
        assert_eq!(after["cleanImageBlobId"], before["cleanImageBlobId"]);
        assert_eq!(after["regions"], before["regions"]);
        assert_eq!(after["cache"]["detectionHit"], true);
        assert_eq!(after["cache"]["ocrHit"], true);
        assert_eq!(after["cache"]["inpaintHit"], true);
        assert_eq!(after["cache"]["translationHit"], true);
    }

    #[tokio::test]
    async fn byte_pixel_mime_and_sha_limits_are_enforced() {
        let image = png();
        let mut request: BrowserJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        request.source_sha256 = sha256_hex(&image);
        request.natural_width = 2;
        request.natural_height = 3;
        request.source_mime_type = "image/png".into();
        let mut limits = ServerLimits::default();
        assert!(validate_image_upload(image.clone(), &request, &limits).is_ok());

        request.source_sha256 = "a".repeat(64);
        assert!(matches!(
            validate_image_upload(image.clone(), &request, &limits),
            Err(ApiError {
                code: "SOURCE_HASH_MISMATCH",
                ..
            })
        ));
        request.source_sha256 = sha256_hex(&image);
        request.source_mime_type = "image/jpeg".into();
        assert!(matches!(
            validate_image_upload(image.clone(), &request, &limits),
            Err(ApiError {
                code: "UNSUPPORTED_IMAGE",
                ..
            })
        ));
        request.source_mime_type = "image/png".into();
        limits.max_pixels = 5;
        assert!(matches!(
            validate_image_upload(image, &request, &limits),
            Err(ApiError {
                code: "IMAGE_TOO_LARGE",
                ..
            })
        ));
    }

    #[test]
    fn tall_images_keep_pixel_and_decode_limits() {
        let image = jpeg_with_dimensions(32, 4_096);
        let mut request: BrowserJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))
        .unwrap();
        request.source_sha256 = sha256_hex(&image);
        request.natural_width = 32;
        request.natural_height = 4_096;
        request.source_mime_type = "image/jpeg".into();

        let limits = ServerLimits::default();
        assert!(
            validate_image_upload(image.clone(), &request, &limits).is_ok(),
            "a tall narrow reader image within every configured bound should decode"
        );

        let mut pixel_limited = limits.clone();
        pixel_limited.max_pixels =
            u64::from(request.natural_width) * u64::from(request.natural_height) - 1;
        assert!(matches!(
            validate_image_upload(image.clone(), &request, &pixel_limited),
            Err(ApiError {
                code: "IMAGE_TOO_LARGE",
                ..
            })
        ));

        let mut decode_limited = limits.clone();
        decode_limited.max_decoded_bytes = 128;
        assert!(matches!(
            validate_image_upload(image.clone(), &request, &decode_limited),
            Err(ApiError {
                code: "UNSUPPORTED_IMAGE",
                ..
            })
        ));
    }

    #[test]
    fn every_handshake_issues_a_fresh_expiring_token() {
        let (state, first) = state_with_delay(Duration::from_millis(5));
        let second = state.issue_session(ORIGIN_VALUE).unwrap();
        assert_ne!(first, second.token);
        assert_eq!(decode_secret(&second.token).unwrap().len(), SECRET_BYTES);
        assert!(second.session_expires_at_unix_ms > unix_ms());
    }

    #[tokio::test]
    async fn expired_token_is_rejected_and_fresh_handshake_recovers() {
        let mut config = BridgeConfig::for_port(43127);
        config.session_ttl = Duration::from_millis(50);
        let state = test_state(config, [7_u8; SECRET_BYTES], Duration::from_millis(5));
        let expired = state.issue_session(ORIGIN_VALUE).unwrap().token;
        sleep(Duration::from_millis(75)).await;
        let fresh = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state);

        let old_response = app
            .clone()
            .oneshot(
                authorized(Method::GET, "/browser/v1/health", &expired)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(old_response.status(), StatusCode::UNAUTHORIZED);
        let fresh_response = app
            .oneshot(
                authorized(Method::GET, "/browser/v1/health", &fresh)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(fresh_response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn multipart_route_enforces_streaming_upload_limit() {
        let image = png();
        let mut config = BridgeConfig::for_port(43127);
        config.limits.max_upload_bytes = image.len() - 1;
        config.limits.max_http_body_bytes = 1024 * 1024;
        let state = test_state(config, [5_u8; SECRET_BYTES], Duration::from_millis(5));
        let token = state.issue_session(ORIGIN_VALUE).unwrap().token;
        let app = router(state);
        let (content_type, body) = multipart_body(&image);
        let response = app
            .oneshot(
                authorized(Method::POST, "/browser/v1/jobs", &token)
                    .header(CONTENT_TYPE, content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(body_json(response).await["code"], "UPLOAD_TOO_LARGE");
    }

    #[tokio::test]
    async fn setup_endpoints_return_frozen_ready_state() {
        let (state, token) = state_with_delay(Duration::from_millis(5));
        let app = router(state);
        for (method, path) in [
            (Method::GET, "/browser/v1/setup"),
            (Method::POST, "/browser/v1/setup/models"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    authorized(method, path, &token)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(body_json(response).await["state"], "ready");
        }
    }
}
