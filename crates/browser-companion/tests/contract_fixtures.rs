use std::fs;
use std::path::{Path, PathBuf};

use browser_companion::contracts::{
    BUILD_FINGERPRINT, BrowserJobCreated, BrowserJobSettings, BrowserSetupStatus,
    DocumentJobRequest, DocumentSourceBlock, ErrorResponse, FocusUpdateRequest, HealthResponse,
    HskLevel, ImageJobRequest, JobUpdatesResponse, LearningMode, LookupResult,
    MAX_DOCUMENT_BLOCK_BYTES, MAX_DOCUMENT_BLOCKS, MAX_DOCUMENT_BYTES, MAX_VISIBLE_BLOCK_IDS,
    NativeHandshakeRequest, NativeReadyResponse, SourceProvenance, SourceSpanKind, Validate,
    canonical_document_sha256,
};
use serde::Deserialize;
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RepeatedDocumentRequestFixture {
    fixture_type: String,
    page_session_id: String,
    block_count: usize,
    text_seed: String,
    text_repeat: usize,
}

fn repeated_document_request_fixture(name: &str) -> DocumentJobRequest {
    let descriptor: RepeatedDocumentRequestFixture = read(name);
    assert_eq!(descriptor.fixture_type, "repeatedDocumentRequest");
    let text = descriptor.text_seed.repeat(descriptor.text_repeat);
    let blocks = (0..descriptor.block_count)
        .map(|item_order| DocumentSourceBlock {
            item_id: format!("block-{item_order}"),
            source_index: 0,
            item_order: u32::try_from(item_order).expect("fixture item order fits u32"),
            kind: SourceSpanKind::Prose,
            provenance: SourceProvenance::Dom,
            text: text.clone(),
        })
        .collect::<Vec<_>>();
    DocumentJobRequest {
        build_fingerprint: BUILD_FINGERPRINT.to_owned(),
        page_session_id: descriptor.page_session_id,
        source_sha256: canonical_document_sha256(&blocks),
        settings: BrowserJobSettings {
            source_language: "en".to_owned(),
            target_language: "zh-CN".to_owned(),
            hsk_standard: "2.0".to_owned(),
            hsk_level: HskLevel::Three,
            learning_mode: LearningMode::Natural,
        },
        blocks,
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RepeatedDocumentFocusFixture {
    fixture_type: String,
    visible_block_count: usize,
    item_id_prefix: String,
    active: bool,
}

fn repeated_document_focus_fixture(name: &str) -> FocusUpdateRequest {
    let descriptor: RepeatedDocumentFocusFixture = read(name);
    assert_eq!(descriptor.fixture_type, "repeatedDocumentFocus");
    serde_json::from_value(serde_json::json!({
        "kind": "document",
        "visibleBlockIds": (0..descriptor.visible_block_count)
            .map(|index| format!("{}{index}", descriptor.item_id_prefix))
            .collect::<Vec<_>>(),
        "active": descriptor.active,
    }))
    .expect("expanded document focus fixture has the tagged wire shape")
}

#[test]
fn image_and_document_requests_and_focus_are_unversioned_and_valid() {
    let request: ImageJobRequest = read("job-request.valid.json");
    request.validate().expect("valid job request");
    let image_focus: FocusUpdateRequest = read("focus-image.valid.json");
    image_focus.validate().expect("valid image focus");
    let document_focus: FocusUpdateRequest = read("focus-document.valid.json");
    document_focus.validate().expect("valid document focus");
    let document: DocumentJobRequest = read("document-job-request.valid.json");
    document.validate().expect("valid document request");

    let serialized = serde_json::to_value(request).unwrap();
    assert_eq!(
        serialized["buildFingerprint"],
        "hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-08-09-r8"
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
        "document-updates.success.json",
    ] {
        let response: JobUpdatesResponse = read(name);
        response
            .validate()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        if name == "job-updates.replay.json" {
            response
                .validate_after(3)
                .unwrap_or_else(|error| panic!("{name} cursor: {error}"));
        }
    }
}

#[test]
fn progressive_region_payloads_use_the_compact_wire_shapes() {
    let response: JobUpdatesResponse = read("job-updates.success.json");
    let serialized = serde_json::to_value(response).unwrap();
    let ready = &serialized["updates"][1];
    assert_eq!(ready["type"], "imageRegionReady");
    assert!(ready["region"].get("textPolygon").is_some());
    assert_eq!(ready["region"]["kind"], "dialogue");
    assert!(ready["region"].get("text").is_some());
    assert!(ready["region"].get("geometry").is_none());
    assert!(ready["region"].get("rotationDegrees").is_none());

    assert_eq!(serialized["updates"][2]["type"], "complete");
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
    let request: ImageJobRequest = read("invalid/job-request.build-fingerprint.json");
    assert!(request.validate().is_err());

    let focus: FocusUpdateRequest = read("invalid/focus-image.out-of-bounds.json");
    assert!(focus.validate().is_err());

    let updates: JobUpdatesResponse = read("invalid/job-updates.nonmonotonic.json");
    assert!(updates.validate().is_err());

    let gap: JobUpdatesResponse = read("invalid/job-updates.gap.json");
    assert!(gap.validate_after(0).is_err());

    let terminal_middle: JobUpdatesResponse = read("invalid/job-updates.terminal-middle.json");
    assert!(terminal_middle.validate_after(0).is_err());

    let document: DocumentJobRequest = read("invalid/document-job-request.hash-mismatch.json");
    assert!(document.validate().is_err());

    let unknown: serde_json::Value = read("invalid/document-job-request.unknown-layout.json");
    assert!(serde_json::from_value::<DocumentJobRequest>(unknown).is_err());
}

#[test]
fn shared_document_size_and_visible_focus_bounds_are_rejected() {
    let total = repeated_document_request_fixture(
        "invalid/document-job-request.total-bytes.descriptor.json",
    );
    assert!(serde_json::to_vec(&total).unwrap().len() > MAX_DOCUMENT_BYTES);
    let error = total.validate().unwrap_err();
    assert_eq!(error.path, "$");

    let block_count = repeated_document_request_fixture(
        "invalid/document-job-request.block-count.descriptor.json",
    );
    assert_eq!(block_count.blocks.len(), MAX_DOCUMENT_BLOCKS + 1);
    let error = block_count.validate().unwrap_err();
    assert_eq!(error.path, "blocks");

    let block_bytes = repeated_document_request_fixture(
        "invalid/document-job-request.block-bytes.descriptor.json",
    );
    assert!(block_bytes.blocks[0].text.len() > MAX_DOCUMENT_BLOCK_BYTES);
    let error = block_bytes.validate().unwrap_err();
    assert_eq!(error.path, "blocks[0].text");

    let focus = repeated_document_focus_fixture(
        "invalid/focus-document.visible-block-count.descriptor.json",
    );
    let FocusUpdateRequest::Document {
        visible_block_ids, ..
    } = &focus
    else {
        panic!("expected expanded document focus fixture");
    };
    assert_eq!(visible_block_ids.len(), MAX_VISIBLE_BLOCK_IDS + 1);
    let error = focus.validate().unwrap_err();
    assert_eq!(error.path, "visibleBlockIds");
}

#[test]
fn shared_tagged_focus_and_update_modality_mismatches_are_rejected() {
    for name in [
        "invalid/focus-document.image-fields.json",
        "invalid/focus-image.document-fields.json",
    ] {
        let value: serde_json::Value = read(name);
        assert!(
            serde_json::from_value::<FocusUpdateRequest>(value).is_err(),
            "{name}"
        );
    }
    for name in [
        "invalid/job-updates.document-tag-image-field.json",
        "invalid/job-updates.image-tag-document-field.json",
    ] {
        let value: serde_json::Value = read(name);
        assert!(
            serde_json::from_value::<JobUpdatesResponse>(value).is_err(),
            "{name}"
        );
    }
}

#[test]
fn removed_protocol_fields_are_not_accepted() {
    let mut value: serde_json::Value = read("job-request.valid.json");
    value["protocolVersion"] = 1.into();
    assert!(serde_json::from_value::<ImageJobRequest>(value).is_err());
}
