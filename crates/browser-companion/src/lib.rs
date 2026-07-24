//! Firefox browser-companion protocol and secure loopback adapter.
//!
//! Protocol version 1 is frozen by the cross-language fixtures in
//! `fixtures/contracts`. The HTTP service and native launcher build on these
//! types; they must not accept a structurally valid value until semantic
//! validation also succeeds.

pub mod contracts;

pub use contracts::{
    BrowserJobCreated, BrowserJobRequest, BrowserJobResult, BrowserJobStatus, BrowserSetupStatus,
    ContractError, ErrorResponse, HealthResponse, LookupRequest, LookupResult,
    NativeHandshakeRequest, NativeReadyResponse, RetranslateRequest, Validate,
};
