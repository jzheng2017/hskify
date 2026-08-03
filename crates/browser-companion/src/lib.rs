//! Hskify's strict Firefox loopback companion.
//!
//! The extension and daemon must carry the exact same build fingerprint.
//! There is no protocol negotiation, migration adapter, or legacy result API.

pub mod chapter_session;
pub mod contracts;
pub mod crypto;
mod cuda_scheduler;
pub mod daemon;
mod decoded_cache;
pub mod discovery;
// Contract fixtures are compiled only for the crate's unit tests.  The
// shipped daemon must never serve synthetic health, font, or translation
// payloads when managed resources are unavailable; a missing setup is a real
// setup failure, not a test backend.
#[cfg(test)]
pub mod fixtures;
pub mod launcher;
pub mod native_framing;
pub mod origin;
mod pipeline_adapter;
mod result_cache;
pub mod server;
mod setup;

pub use contracts::{
    BUILD_FINGERPRINT, BrowserJobCreated, BrowserSetupStatus, ContractError, CreateJobRequest,
    ErrorResponse, HealthResponse, JobUpdate, JobUpdatesResponse, LookupRequest, LookupResult,
    NativeHandshakeRequest, NativeReadyResponse, NormalizedRect, TranslatedRegion, Validate,
    ViewportUpdateRequest,
};

/// Permanent Firefox add-on ID frozen by ADR 0001.
pub const FIREFOX_EXTENSION_ID: &str = "hsk-manga-translator@local.hskify";

/// Native host name frozen by ADR 0001.
pub const NATIVE_HOST_NAME: &str = "local.hskify.hsk_manga";

/// Explicit Firefox extension origin used when privileged extension fetches
/// omit the standard `Origin` header.
pub const EXTENSION_ORIGIN_HEADER: &str = "x-hsk-manga-extension-origin";

/// Internal launcher-to-daemon control header. This is never exposed to Firefox.
pub const CONTROL_HEADER: &str = "x-hsk-manga-control";
