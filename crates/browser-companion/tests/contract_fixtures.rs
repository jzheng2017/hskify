use std::fs;
use std::path::{Path, PathBuf};

use browser_companion::contracts::{
    BrowserJobCreated, BrowserSetupStatus, CreateJobRequest, ErrorResponse, HealthResponse,
    JobUpdatesResponse, LookupResult, NativeHandshakeRequest, NativeReadyResponse, Validate,
    ViewportUpdateRequest,
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

#[test]
fn job_request_and_viewport_are_unversioned_and_valid() {
    let request: CreateJobRequest = read("job-request.valid.json");
    request.validate().expect("valid job request");
    let viewport: ViewportUpdateRequest = read("viewport.valid.json");
    viewport.validate().expect("valid viewport");

    let serialized = serde_json::to_value(request).unwrap();
    assert_eq!(
        serialized["buildFingerprint"],
        "hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-27-r6"
    );
    assert!(serialized.get("protocolVersion").is_none());
}

#[test]
fn progressive_sequences_are_monotonic_and_replayable() {
    for name in [
        "job-updates.success.json",
        "job-updates.failure.json",
        "job-updates.cancelled.json",
        "job-updates.replay.json",
    ] {
        let response: JobUpdatesResponse = read(name);
        response
            .validate()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

#[test]
fn progressive_region_payloads_use_the_compact_wire_shapes() {
    let response: JobUpdatesResponse = read("job-updates.success.json");
    let serialized = serde_json::to_value(response).unwrap();
    let ready = &serialized["updates"][1];
    assert_eq!(ready["type"], "regionReady");
    assert!(ready["region"].get("textPolygon").is_some());
    assert!(ready["region"].get("kind").is_none());
    assert!(ready["region"].get("geometry").is_none());
    assert!(ready["region"].get("rotationDegrees").is_none());

    let refined = &serialized["updates"][2];
    assert_eq!(refined["type"], "regionRefined");
    assert_eq!(refined["regionId"], "aaaaaaaa-region-0001");
    assert!(refined.get("region").is_none());
    assert!(refined.get("patch").is_none());
}

#[test]
fn lookup_setup_health_created_and_errors_are_valid() {
    let lookup: LookupResult = read("lookup.valid.json");
    lookup.validate().expect("valid lookup");
    let setup: BrowserSetupStatus = read("setup.ready.json");
    setup.validate().expect("valid setup");
    assert_eq!(setup.model_id, "qwen3.5-4b");
    let health: HealthResponse = read("health.ready.json");
    health.validate().expect("valid health response");
    let created: BrowserJobCreated = read("job-created.valid.json");
    created.validate().expect("valid job-created response");
    let error: ErrorResponse = read("error.valid.json");
    error.validate().expect("valid error response");
}

#[test]
fn native_handshake_uses_exact_build_affinity() {
    let request: NativeHandshakeRequest = read("native-request.valid.json");
    request.validate().expect("valid native request");
    let ready: NativeReadyResponse = read("native-ready.valid.json");
    ready.validate().expect("valid native response");

    let serialized = serde_json::to_value(ready).unwrap();
    assert!(serialized.get("protocolVersion").is_none());
    assert_eq!(serialized["engineVersion"], "0.61.2");

    let serialized = serde_json::to_value(request).unwrap();
    assert_eq!(serialized["extensionVersion"], "0.1.0");
    assert!(serialized.get("protocolVersion").is_none());
}

#[test]
fn invalid_semantic_fixtures_are_rejected() {
    let request: CreateJobRequest = read("invalid/job-request.build-fingerprint.json");
    assert!(request.validate().is_err());

    let viewport: ViewportUpdateRequest = read("invalid/viewport.out-of-bounds.json");
    assert!(viewport.validate().is_err());

    let updates: JobUpdatesResponse = read("invalid/job-updates.nonmonotonic.json");
    assert!(updates.validate().is_err());
}

#[test]
fn removed_protocol_fields_are_not_accepted() {
    let mut value: serde_json::Value = read("job-request.valid.json");
    value["protocolVersion"] = 1.into();
    assert!(serde_json::from_value::<CreateJobRequest>(value).is_err());
}
