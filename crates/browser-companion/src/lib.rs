//! Firefox browser-companion protocol and secure loopback adapter.
//!
//! Protocol version 1 is frozen by the cross-language fixtures in
//! `fixtures/contracts`. The HTTP service and native launcher build on these
//! types; they must not accept a structurally valid value until semantic
//! validation also succeeds.

pub mod contracts;
pub mod crypto;
pub mod daemon;
pub mod discovery;
pub mod fixtures;
pub mod launcher;
pub mod native_framing;
pub mod origin;
mod pipeline_adapter;
pub mod server;

pub use contracts::{
    BrowserJobCreated, BrowserJobRequest, BrowserJobResult, BrowserJobStatus, BrowserSetupStatus,
    ContractError, ErrorResponse, HealthResponse, LookupRequest, LookupResult,
    NativeHandshakeRequest, NativeReadyResponse, RetranslateRequest, Validate,
};

/// Permanent Firefox add-on ID frozen by ADR 0001.
pub const FIREFOX_EXTENSION_ID: &str = "hsk-manga-translator@local.mangalations";

/// Native host name frozen by ADR 0001.
pub const NATIVE_HOST_NAME: &str = "local.mangalations.hsk_manga";

/// Browser protocol header name.
pub const PROTOCOL_HEADER: &str = "x-hsk-manga-protocol";

/// Internal launcher-to-daemon control header. This is never exposed to Firefox.
pub const CONTROL_HEADER: &str = "x-hsk-manga-control";
