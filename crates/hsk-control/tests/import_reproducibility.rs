use std::{fs, path::Path};

use hsk_control::{
    DatasetCompleteness, Delimiter, HskControl, HskControlError, generate_dictionary_artifact,
    generate_hsk_artifact, parse_import_metadata,
};

#[test]
fn committed_hsk_seed_is_exactly_reproducible() {
    let root = repository_root();
    let source = fs::read(root.join("data/hsk/test-seed-source.tsv")).unwrap();
    let metadata =
        parse_import_metadata(&fs::read(root.join("data/hsk/test-seed-import.json")).unwrap())
            .unwrap();
    let first = generate_hsk_artifact(&source, &metadata, Delimiter::Tab).unwrap();
    let second = generate_hsk_artifact(&source, &metadata, Delimiter::Tab).unwrap();
    let committed = fs::read(root.join("data/hsk/test-seed.normalized.json")).unwrap();

    assert_eq!(first, second);
    assert_eq!(first, committed);
    assert_eq!(metadata.completeness, DatasetCompleteness::TestSeed);
    assert_eq!(metadata.licence.spdx_expression, "GPL-3.0-only");
    assert!(metadata.licence.redistribution_allowed);
    assert!(metadata.licence.attribution.contains("not an official HSK"));
}

#[test]
fn committed_dictionary_seed_is_exactly_reproducible() {
    let root = repository_root();
    let source = fs::read(root.join("data/dictionary/test-seed-cedict.u8")).unwrap();
    let metadata = parse_import_metadata(
        &fs::read(root.join("data/dictionary/test-seed-import.json")).unwrap(),
    )
    .unwrap();
    let first = generate_dictionary_artifact(&source, &metadata).unwrap();
    let second = generate_dictionary_artifact(&source, &metadata).unwrap();
    let committed = fs::read(root.join("data/dictionary/test-seed.normalized.json")).unwrap();

    assert_eq!(first, second);
    assert_eq!(first, committed);
    let generated = String::from_utf8(first).unwrap();
    assert!(generated.contains("\"pinyin\": \"lí kāi\""));
    assert!(
        metadata
            .licence
            .attribution
            .contains("no CC-CEDICT entries")
    );
}

#[test]
fn import_rejects_hash_mismatch_and_unaudited_redistribution() {
    let root = repository_root();
    let source = fs::read(root.join("data/hsk/test-seed-source.tsv")).unwrap();
    let mut metadata =
        parse_import_metadata(&fs::read(root.join("data/hsk/test-seed-import.json")).unwrap())
            .unwrap();
    metadata.source.sha256 = "f".repeat(64);
    assert!(matches!(
        generate_hsk_artifact(&source, &metadata, Delimiter::Tab),
        Err(HskControlError::SourceHashMismatch { .. })
    ));

    metadata.source.sha256 = hsk_control::sha256_hex(&source);
    metadata.licence.redistribution_allowed = false;
    assert!(matches!(
        generate_hsk_artifact(&source, &metadata, Delimiter::Tab),
        Err(HskControlError::LicenceAudit(_))
    ));
}

#[test]
fn generated_count_audits_are_checked_at_runtime() {
    let mut hsk: serde_json::Value =
        serde_json::from_str(hsk_control::EMBEDDED_HSK_TEST_SEED).unwrap();
    hsk["auditedEntryCount"] = serde_json::json!(999);
    let error = HskControl::from_json_with_policy(
        &serde_json::to_string(&hsk).unwrap(),
        hsk_control::EMBEDDED_DICTIONARY_TEST_SEED,
        hsk_control::LoadPolicy::AllowIncompleteTestSeed,
    )
    .err()
    .expect("tampered count must fail");
    assert!(matches!(error, HskControlError::InvalidData(_)));
}

fn repository_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap()
}
