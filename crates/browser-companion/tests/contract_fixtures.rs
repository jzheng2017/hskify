use std::fs;
use std::path::{Path, PathBuf};

use browser_companion::contracts::{
    BrowserJobCreated, BrowserJobRequest, BrowserJobResult, BrowserJobStatus, BrowserSetupStatus,
    ErrorResponse, HealthResponse, LookupResult, NativeHandshakeRequest, NativeReadyResponse,
    RetranslateRequest, Validate,
};
use serde::de::DeserializeOwned;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/contracts")
        .canonicalize()
        .expect("contract fixture directory")
}

fn read<T: DeserializeOwned>(name: &str) -> T {
    let path = fixture_root().join(name);
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn validate_sequence(name: &str) {
    let values: Vec<BrowserJobStatus> = read(name);
    let mut previous_revision = 0;
    let mut previous_overall = 0.0;
    for status in values {
        status.validate().expect("valid progress status");
        assert!(status.revision > previous_revision);
        if let Some(overall) = status.overall_progress {
            assert!(overall >= previous_overall);
            previous_overall = overall;
        }
        previous_revision = status.revision;
    }
}

#[test]
fn valid_job_request_is_shared_contract() {
    let value: BrowserJobRequest = read("job-request.valid.json");
    value.validate().expect("valid job request");
}

#[test]
fn complete_result_is_shared_contract() {
    let value: BrowserJobResult = read("job-result.complete.json");
    value.validate().expect("valid result");
}

#[test]
fn progress_sequences_are_monotonic_and_valid() {
    validate_sequence("progress.success.json");
    validate_sequence("progress.failure.json");
    validate_sequence("progress.cancellation.json");
    validate_sequence("progress.reconnect.json");
}

#[test]
fn lookup_and_setup_fixtures_are_valid() {
    let lookup: LookupResult = read("lookup.valid.json");
    lookup.validate().expect("valid lookup");
    let setup: BrowserSetupStatus = read("setup.ready.json");
    setup.validate().expect("valid setup");
}

#[test]
fn supporting_http_fixtures_are_valid() {
    let health: HealthResponse = read("health.ready.json");
    health.validate().expect("valid health response");
    let created: BrowserJobCreated = read("job-created.valid.json");
    created.validate().expect("valid job-created response");
    let retranslate: RetranslateRequest = read("retranslate.valid.json");
    retranslate.validate().expect("valid retranslate request");
    let error: ErrorResponse = read("error.valid.json");
    error.validate().expect("valid error response");
}

#[test]
fn native_fixtures_are_valid() {
    let request: NativeHandshakeRequest = read("native-request.valid.json");
    request.validate().expect("valid native request");
    let ready: NativeReadyResponse = read("native-ready.valid.json");
    ready.validate().expect("valid native response");
}

#[test]
fn invalid_semantic_fixtures_are_rejected() {
    let request: BrowserJobRequest = read("invalid/job-request.protocol-version.json");
    assert!(request.validate().is_err());

    let result: BrowserJobResult = read("invalid/job-result.out-of-range-point.json");
    assert!(result.validate().is_err());

    let status: BrowserJobStatus = read("invalid/progress.terminal-mismatch.json");
    assert!(status.validate().is_err());
}
