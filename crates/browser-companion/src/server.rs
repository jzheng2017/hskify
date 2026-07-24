//! Secure, fixture-backed `/browser/v1` loopback service.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use axum::body::{Body, to_bytes};
use axum::extract::multipart::{Field, MultipartRejection};
use axum::extract::{DefaultBodyLimit, Multipart, Path, Request, State};
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
use image::{GenericImageView, ImageFormat, ImageReader, Limits};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::time::sleep;
use uuid::Uuid;

use crate::contracts::{
    BrowserCapabilities, BrowserJobCreated, BrowserJobRequest, BrowserJobResult, BrowserJobState,
    BrowserJobStatus, ErrorResponse, HealthResponse, HskLevel, LookupRegion, LookupRequest,
    NativeReadyResponse, NativeReadyType, RetranslateRequest, Validate,
};
use crate::crypto::{SECRET_BYTES, decode_secret, generate_secret, secrets_equal, sha256_hex};
use crate::fixtures;
use crate::origin::validate_extension_origin;
use crate::{CONTROL_HEADER, PROTOCOL_HEADER};

const INTERNAL_SESSION_PATH: &str = "/browser-internal/v1/session";
const MAX_INTERNAL_BODY_BYTES: usize = 4 * 1024;
const MAX_LOOKUP_BODY_BYTES: usize = 16 * 1024;
const MAX_RETRANSLATE_BODY_BYTES: usize = 64 * 1024;
const MAX_SESSIONS: usize = 64;
const MAX_RETAINED_JOBS: usize = 128;

#[derive(Debug, Clone)]
pub struct ServerLimits {
    pub max_upload_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_http_body_bytes: usize,
    pub max_pixels: u64,
    pub max_dimension: u32,
    pub max_decoded_bytes: u64,
    pub max_clean_blob_bytes: usize,
    pub max_stored_blob_bytes: usize,
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
            max_stored_blob_bytes: 256 * MIB,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub port: u16,
    pub session_ttl: Duration,
    pub idle_timeout: Duration,
    pub fixture_stage_delay: Duration,
    pub limits: ServerLimits,
}

impl BridgeConfig {
    pub fn for_port(port: u16) -> Self {
        Self {
            port,
            session_ttl: Duration::from_secs(15 * 60),
            idle_timeout: Duration::from_secs(10 * 60),
            fixture_stage_delay: Duration::from_millis(120),
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
}

#[derive(Debug)]
struct JobRecord {
    status: RwLock<BrowserJobStatus>,
    result: BrowserJobResult,
    cancel: AtomicBool,
    active: AtomicBool,
}

impl JobRecord {
    fn new(status: BrowserJobStatus, result: BrowserJobResult) -> Self {
        Self {
            status: RwLock::new(status),
            result,
            cancel: AtomicBool::new(false),
            active: AtomicBool::new(true),
        }
    }

    fn status(&self) -> BrowserJobStatus {
        self.status
            .read()
            .expect("job status lock poisoned")
            .clone()
    }

    fn update_progress(&self, status: BrowserJobStatus) -> bool {
        let mut current = self.status.write().expect("job status lock poisoned");
        if self.cancel.load(Ordering::Acquire) || current.state != BrowserJobState::Running {
            return false;
        }
        *current = status;
        true
    }
}

#[derive(Debug)]
pub struct BridgeState {
    config: BridgeConfig,
    control_secret: [u8; SECRET_BYTES],
    sessions: Mutex<Vec<Session>>,
    jobs: RwLock<HashMap<String, Arc<JobRecord>>>,
    blobs: RwLock<HashMap<String, StoredBlob>>,
    last_activity: Mutex<Instant>,
    active_jobs: AtomicUsize,
}

impl BridgeState {
    pub fn new(config: BridgeConfig, control_secret: [u8; SECRET_BYTES]) -> Arc<Self> {
        assert_ne!(config.port, 0, "browser daemon requires a bound port");
        assert!(
            config.limits.max_http_body_bytes >= config.limits.max_upload_bytes,
            "HTTP body limit must cover an upload"
        );
        Arc::new(Self {
            config,
            control_secret,
            sessions: Mutex::new(Vec::new()),
            jobs: RwLock::new(HashMap::new()),
            blobs: RwLock::new(HashMap::new()),
            last_activity: Mutex::new(Instant::now()),
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
                models_ready: true,
            },
        })
    }

    fn touch(&self) {
        *self.last_activity.lock().expect("activity lock poisoned") = Instant::now();
    }

    fn idle_for(&self) -> Duration {
        self.last_activity
            .lock()
            .expect("activity lock poisoned")
            .elapsed()
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
        if state.active_job_count() == 0 && state.idle_for() >= state.config.idle_timeout {
            return;
        }
    }
}

async fn security_boundary(
    State(state): State<Arc<BridgeState>>,
    request: Request,
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
        state.touch();
        return next.run(request).await;
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

    state.touch();
    let response = next.run(request).await;
    with_cors(response, &origin)
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

async fn health() -> Json<HealthResponse> {
    Json(fixtures::health())
}

async fn setup() -> Json<crate::contracts::BrowserSetupStatus> {
    Json(fixtures::setup())
}

async fn setup_models() -> Json<crate::contracts::BrowserSetupStatus> {
    Json(fixtures::setup())
}

async fn create_job(
    State(state): State<Arc<BridgeState>>,
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
        validate_image_upload(image, &validation_request, &limits)
    })
    .await
    .map_err(|_| ApiError::internal())??;

    let blob_id = format!("blob-{}", Uuid::new_v4());
    let stored_blob = StoredBlob {
        bytes: validated.clean_png.into(),
        content_type: "image/png",
    };
    let job_id = format!("job-{}", Uuid::new_v4());
    let statuses = fixtures::progress(&job_id);
    let result = fixtures::result(&job_id, &blob_id, &job_request);
    let record = Arc::new(JobRecord::new(statuses[0].clone(), result));
    let mut jobs = state.jobs.write().expect("job lock poisoned");
    if jobs.len() >= MAX_RETAINED_JOBS {
        return Err(ApiError::too_many_requests(
            "JOB_LIMIT_REACHED",
            "Too many browser jobs are retained by this daemon.",
        ));
    }
    let mut blobs = state.blobs.write().expect("blob lock poisoned");
    let stored_bytes = blobs
        .values()
        .try_fold(0_usize, |total, blob| total.checked_add(blob.bytes.len()))
        .unwrap_or(usize::MAX);
    if stored_bytes.saturating_add(stored_blob.bytes.len())
        > state.config.limits.max_stored_blob_bytes
    {
        return Err(ApiError::too_many_requests(
            "BLOB_LIMIT_REACHED",
            "The fixture blob cache has reached its bounded capacity.",
        ));
    }
    blobs.insert(blob_id, stored_blob);
    jobs.insert(job_id.clone(), record.clone());
    drop(blobs);
    drop(jobs);
    state.active_jobs.fetch_add(1, Ordering::AcqRel);
    tokio::spawn(run_fixture_job(state.clone(), record, statuses));

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
    clean_png: Vec<u8>,
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

    let clean_png = if format == ImageFormat::Png {
        image
    } else {
        let mut output = Cursor::new(Vec::new());
        decoded
            .write_to(&mut output, ImageFormat::Png)
            .map_err(|_| ApiError::internal())?;
        output.into_inner()
    };
    if clean_png.len() > limits.max_clean_blob_bytes {
        return Err(ApiError::payload_too_large(
            "CLEAN_IMAGE_TOO_LARGE",
            "The clean fixture image exceeds the blob limit.",
        ));
    }
    Ok(ValidatedUpload { clean_png })
}

async fn run_fixture_job(
    state: Arc<BridgeState>,
    record: Arc<JobRecord>,
    statuses: Vec<BrowserJobStatus>,
) {
    for status in statuses.into_iter().skip(1) {
        sleep(state.config.fixture_stage_delay).await;
        if record.cancel.load(Ordering::Acquire) {
            finish_active(&state, &record);
            return;
        }
        if !record.update_progress(status) {
            finish_active(&state, &record);
            return;
        }
        state.touch();
    }
    finish_active(&state, &record);
}

fn finish_active(state: &BridgeState, record: &JobRecord) {
    if record.active.swap(false, Ordering::AcqRel) {
        state.active_jobs.fetch_sub(1, Ordering::AcqRel);
        state.touch();
    }
}

fn find_job(state: &BridgeState, job_id: &str) -> Result<Arc<JobRecord>, ApiError> {
    state
        .jobs
        .read()
        .expect("job lock poisoned")
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
    Ok(Json(job.result.clone()))
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
) -> Result<impl IntoResponse, ApiError> {
    let request: RetranslateRequest = parse_json_body(request, MAX_RETRANSLATE_BODY_BYTES).await?;
    request.validate().map_err(|_| {
        ApiError::bad_request(
            "INVALID_RETRANSLATE_REQUEST",
            "The retranslation request failed protocol validation.",
        )
    })?;
    let original = find_job(&state, &job_id)?;
    if original.status().state != BrowserJobState::Complete {
        return Err(ApiError::conflict(
            "JOB_NOT_COMPLETE",
            "Only a completed browser job can be retranslated.",
        ));
    }
    let new_job_id = format!("job-{}", Uuid::new_v4());
    let mut result = original.result.clone();
    result.job_id.clone_from(&new_job_id);
    for region in &mut result.regions {
        region.vocabulary.requested_hsk_level = request.settings.hsk_level;
    }
    result.validate().map_err(|_| ApiError::internal())?;
    let statuses = fixtures::progress(&new_job_id);
    let record = Arc::new(JobRecord::new(statuses[0].clone(), result));
    let mut jobs = state.jobs.write().expect("job lock poisoned");
    if jobs.len() >= MAX_RETAINED_JOBS {
        return Err(ApiError::too_many_requests(
            "JOB_LIMIT_REACHED",
            "Too many browser jobs are retained by this daemon.",
        ));
    }
    jobs.insert(new_job_id.clone(), record.clone());
    drop(jobs);
    state.active_jobs.fetch_add(1, Ordering::AcqRel);
    tokio::spawn(run_fixture_job(state.clone(), record, statuses));
    Ok((
        StatusCode::ACCEPTED,
        Json(BrowserJobCreated {
            protocol_version: crate::contracts::PROTOCOL_VERSION,
            job_id: new_job_id,
        }),
    ))
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
    let mut result = fixtures::lookup(&request.selected_text);
    if let (Some(job_id), Some(region_id)) = (&request.job_id, &request.region_id) {
        let job = find_job(&state, job_id)?;
        let region = job
            .result
            .regions
            .iter()
            .find(|region| &region.id == region_id)
            .ok_or_else(|| {
                ApiError::not_found("REGION_NOT_FOUND", "The browser region does not exist.")
            })?;
        result.region = Some(LookupRegion {
            displayed_chinese: region.displayed_chinese.clone(),
            faithful_chinese: region.faithful_chinese.clone(),
            source_english: region.source_english.clone(),
        });
    }
    result.validate().map_err(|_| ApiError::internal())?;
    Ok(Json(result))
}

async fn blob(
    State(state): State<Arc<BridgeState>>,
    Path(blob_id): Path<String>,
) -> Result<Response, ApiError> {
    let blob = state
        .blobs
        .read()
        .expect("blob lock poisoned")
        .get(&blob_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("BLOB_NOT_FOUND", "The browser blob does not exist."))?;
    Ok(bytes_response(
        blob.bytes.as_ref().to_vec(),
        blob.content_type,
    ))
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
    use std::task::{Context, Poll};

    use axum::body::{Bytes, to_bytes};
    use http_body::{Body as HttpBody, Frame};
    use image::{DynamicImage, RgbaImage};
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

    fn state_with_delay(delay: Duration) -> (Arc<BridgeState>, String) {
        let mut config = BridgeConfig::for_port(43127);
        config.fixture_stage_delay = delay;
        config.idle_timeout = Duration::from_secs(30);
        let state = BridgeState::new(config, [9_u8; SECRET_BYTES]);
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
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            2,
            3,
            image::Rgba([255, 255, 255, 255]),
        ));
        let mut output = Cursor::new(Vec::new());
        image.write_to(&mut output, ImageFormat::Png).unwrap();
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
        let retranslated_id = body_json(retranslate_response).await["jobId"]
            .as_str()
            .unwrap()
            .to_owned();
        let stopped_retranslation = app
            .clone()
            .oneshot(
                authorized(
                    Method::DELETE,
                    &format!("/browser/v1/jobs/{retranslated_id}"),
                    &token,
                )
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(body_json(stopped_retranslation).await["state"], "cancelled");

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
        let state = BridgeState::new(config, [7_u8; SECRET_BYTES]);
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
        let state = BridgeState::new(config, [5_u8; SECRET_BYTES]);
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
