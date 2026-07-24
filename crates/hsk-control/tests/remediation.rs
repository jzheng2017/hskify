use hsk_control::{
    DatasetCompleteness, DatasetKind, Delimiter, DictionaryArtifact, DictionaryEntry, HSK_STANDARD,
    HskArtifact, HskControl, HskControlError, HskEntry, HskLevel, ImportMetadata, LicenceAudit,
    LookupRegionContext, PreservationViolation, SourceAudit, TextNormalizer,
    UNICODE_NORMALIZATION_CRATE_VERSION, UNICODE_NORMALIZATION_TABLES_SHA256,
    UNICODE_NORMALIZATION_UNICODE_VERSION, ViolationReason, generate_hsk_artifact, sha256_hex,
};
use jieba_rs::Jieba;

#[test]
fn disallowed_span_may_begin_and_end_inside_primary_tokens() {
    let control = control(
        "cross-primary-token-boundary",
        vec![
            hsk_entry("甲乙", HskLevel::ONE),
            hsk_entry("丙", HskLevel::ONE),
        ],
        vec![dictionary_entry("乙丙")],
    );

    let report = control.validate("甲乙丙", HskLevel::ONE, &[]);
    assert_eq!(
        report
            .violations
            .iter()
            .map(|violation| (
                violation.text.as_str(),
                violation.start_char,
                violation.end_char,
                &violation.reason,
            ))
            .collect::<Vec<_>>(),
        vec![("乙丙", 1, 3, &ViolationReason::KnownDictionaryWord)]
    );
}

#[test]
fn overlapping_disallowed_spans_are_all_reported_deterministically() {
    let control = control(
        "overlapping-spans",
        ["甲", "乙", "丙", "丁"]
            .into_iter()
            .map(|word| hsk_entry(word, HskLevel::ONE))
            .collect(),
        ["甲乙丙", "乙丙", "丙丁"]
            .into_iter()
            .map(dictionary_entry)
            .collect(),
    );

    let report = control.validate("甲乙丙丁", HskLevel::ONE, &[]);
    assert_eq!(
        report
            .violations
            .iter()
            .map(|violation| (
                violation.text.as_str(),
                violation.start_char,
                violation.end_char
            ))
            .collect::<Vec<_>>(),
        vec![("甲乙丙", 0, 3), ("乙丙", 1, 3), ("丙丁", 2, 4)]
    );
}

#[test]
fn allowed_whole_headword_suppresses_incidental_internal_dictionary_spelling() {
    let control = control(
        "allowed-container",
        vec![hsk_entry("甲乙丙", HskLevel::ONE)],
        vec![dictionary_entry("乙丙")],
    );

    let report = control.validate("甲乙丙", HskLevel::ONE, &[]);
    assert!(report.strictly_valid, "{:?}", report.violations);
}

#[test]
fn higher_level_cross_boundary_span_keeps_its_required_level() {
    let control = control(
        "higher-level-cross-boundary",
        vec![
            hsk_entry("甲乙", HskLevel::ONE),
            hsk_entry("丙", HskLevel::ONE),
            hsk_entry("乙丙", HskLevel::TWO),
        ],
        vec![dictionary_entry("乙丙")],
    );

    let report = control.validate("甲乙丙", HskLevel::ONE, &[]);
    assert_eq!(report.violations.len(), 1);
    assert_eq!(report.violations[0].text, "乙丙");
    assert_eq!(
        report.violations[0].reason,
        ViolationReason::AboveSelectedHskLevel {
            required_level: HskLevel::TWO
        }
    );
}

#[test]
fn omitted_independently_usable_defaults_false_for_test_seed_entries() {
    let entry: HskEntry = serde_json::from_value(serde_json::json!({
        "simplified": "甲",
        "pinyin": "jiǎ",
        "glosses": ["first"],
        "level": 1
    }))
    .unwrap();
    assert!(!entry.independently_usable);

    let mut artifact = hsk_artifact(
        "test-seed-default",
        DatasetCompleteness::TestSeed,
        vec![hsk_entry("甲", HskLevel::ONE)],
    );
    artifact.entries[0].independently_usable = false;
    let mut json = serde_json::to_value(artifact).unwrap();
    json["entries"][0]
        .as_object_mut()
        .unwrap()
        .remove("independentlyUsable");
    let parsed: HskArtifact = serde_json::from_value(json).unwrap();
    assert!(!parsed.entries[0].independently_usable);
}

#[test]
fn complete_artifact_deserialization_requires_explicit_independent_use_audit() {
    let artifact = hsk_artifact(
        "complete-explicit-audit",
        DatasetCompleteness::Complete,
        vec![hsk_entry("甲", HskLevel::ONE)],
    );
    let mut json = serde_json::to_value(artifact).unwrap();
    json["entries"][0]
        .as_object_mut()
        .unwrap()
        .remove("independentlyUsable");

    let error = serde_json::from_value::<HskArtifact>(json).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("must explicitly audit independentlyUsable")
    );
}

#[test]
fn complete_import_requires_header_and_per_row_independent_use_audit() {
    let missing_header = b"level,simplified,pinyin,gloss\n1,\xe7\x94\xb2,jia,first\n";
    let metadata = import_metadata(missing_header, DatasetCompleteness::Complete);
    assert!(matches!(
        generate_hsk_artifact(missing_header, &metadata, Delimiter::Comma),
        Err(HskControlError::InvalidData(message))
            if message.contains("independently_usable header")
    ));

    let blank_value =
        b"level,simplified,pinyin,gloss,independently_usable\n1,\xe7\x94\xb2,jia,first,\n";
    let metadata = import_metadata(blank_value, DatasetCompleteness::Complete);
    assert!(matches!(
        generate_hsk_artifact(blank_value, &metadata, Delimiter::Comma),
        Err(HskControlError::InvalidData(message))
            if message.contains("must explicitly audit independentlyUsable")
    ));
}

#[test]
fn test_seed_import_without_independent_use_column_defaults_false() {
    let source = b"level,simplified,pinyin,gloss\n1,\xe7\x94\xb2,jia,first\n";
    let metadata = import_metadata(source, DatasetCompleteness::TestSeed);
    let generated = generate_hsk_artifact(source, &metadata, Delimiter::Comma).unwrap();
    let artifact: HskArtifact = serde_json::from_slice(&generated).unwrap();
    assert!(!artifact.entries[0].independently_usable);
}

#[test]
fn lexicalized_feichang_does_not_create_negation_addition_or_removal() {
    let control = negation_control();
    for lexicalized in ["非常好", "不错"] {
        assert!(
            preservation_violations(&control, lexicalized, "很好").is_empty(),
            "removing lexicalized {lexicalized} must not count as removed negation"
        );
        assert!(
            preservation_violations(&control, "很好", lexicalized).is_empty(),
            "adding lexicalized {lexicalized} must not count as added negation"
        );
    }
}

#[test]
fn multiple_clause_negation_preserves_every_marker_occurrence() {
    let control = negation_control();
    assert!(preservation_violations(&control, "我不吃也不喝", "我不吃也不喝").is_empty());
    assert_eq!(
        preservation_violations(&control, "我不吃也不喝", "我不吃也喝"),
        vec![PreservationViolation::NegationMarkersChanged {
            expected: vec!["不".into(), "不".into()],
            actual: vec!["不".into()],
        }]
    );
}

#[test]
fn real_negation_markers_remain_preserved_end_to_end() {
    let control = negation_control();
    for (negative, positive, marker) in [
        ("不好", "好", "不"),
        ("没来", "来", "没"),
        ("没有来", "有来", "没"),
        ("别走", "走", "别"),
        ("未完成", "完成", "未"),
        ("非会员", "会员", "非"),
        ("莫来", "来", "莫"),
    ] {
        assert_eq!(
            preservation_violations(&control, negative, positive),
            vec![PreservationViolation::NegationMarkersChanged {
                expected: vec![marker.into()],
                actual: vec![],
            }],
            "{negative:?} -> {positive:?}"
        );
    }
}

#[test]
fn negation_diagnostics_cover_added_replaced_and_reordered_markers() {
    let control = negation_control();
    for (reference, candidate, expected, actual) in [
        ("我来", "我不来", vec![], vec!["不"]),
        ("我不来", "我没来", vec!["不"], vec!["没"]),
        (
            "我不来，你别走",
            "我别来，你不走",
            vec!["不", "别"],
            vec!["别", "不"],
        ),
    ] {
        assert_eq!(
            preservation_violations(&control, reference, candidate),
            vec![PreservationViolation::NegationMarkersChanged {
                expected: expected.into_iter().map(str::to_owned).collect(),
                actual: actual.into_iter().map(str::to_owned).collect(),
            }],
            "{reference:?} -> {candidate:?}"
        );
    }
}

#[test]
fn one_negative_and_justified_marker_aliases_are_preserved() {
    let control = negation_control();
    assert!(preservation_violations(&control, "我不吃", "我不喝").is_empty());
    assert!(preservation_violations(&control, "我没有来", "我没来").is_empty());
    assert!(preservation_violations(&control, "他并非会员", "他非会员").is_empty());
}

#[test]
fn lexicalized_words_do_not_hide_real_negation_units() {
    let control = negation_control();
    assert!(
        preservation_violations(&control, "他非常不好", "他非常好").contains(
            &PreservationViolation::NegationMarkersChanged {
                expected: vec!["不".into()],
                actual: vec![],
            }
        )
    );
    assert!(preservation_violations(&control, "不错，我不来", "很好，我不来").is_empty());
}

#[test]
fn repeated_proper_name_occurrences_are_preserved() {
    let control = negation_control();
    let names = [hsk_control::ProperName {
        text: "小明".into(),
        reason: hsk_control::ProperNameReason::PersonName,
    }];
    assert_eq!(
        preservation_violations_with_names(&control, "小明问小明", "小明问", &names),
        vec![PreservationViolation::ProperNameOccurrencesChanged {
            text: "小明".into(),
            expected: 2,
            actual: 1,
        }]
    );
}

#[test]
fn lookup_can_carry_optional_frozen_region_context_without_protocol_dependency() {
    let control = control(
        "lookup-region",
        vec![hsk_entry("离开", HskLevel::TWO)],
        vec![dictionary_entry("离开")],
    );
    let region = LookupRegionContext {
        displayed_chinese: "我们现在要走！".into(),
        faithful_chinese: "我们得马上离开！".into(),
        source_english: "We have to leave now!".into(),
    };
    let result = control.lookup_with_region_context("离开", &[], Some(region.clone()));
    assert_eq!(result.region, Some(region));
    let json = serde_json::to_value(result).unwrap();
    assert_eq!(json["region"]["displayedChinese"], "我们现在要走！");
    assert_eq!(json["region"]["faithfulChinese"], "我们得马上离开！");
    assert_eq!(json["region"]["sourceEnglish"], "We have to leave now!");

    let plain = serde_json::to_value(control.lookup("离开", &[])).unwrap();
    assert!(plain.get("region").is_none());
}

#[test]
fn cache_dependencies_are_exact_and_normalization_tables_match_unicode_17() {
    assert_eq!(UNICODE_NORMALIZATION_CRATE_VERSION, "0.1.25");
    assert_eq!(
        unicode_normalization::UNICODE_VERSION,
        UNICODE_NORMALIZATION_UNICODE_VERSION
    );
    assert_eq!(UNICODE_NORMALIZATION_TABLES_SHA256.len(), 64);
    assert!(
        UNICODE_NORMALIZATION_TABLES_SHA256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
    assert_eq!(TextNormalizer::new().normalize("Ａ\u{030a}"), "Å");

    let jieba = Jieba::new();
    assert!(jieba.has_word("非常"));
    assert_eq!(
        jieba
            .cut("他非常好", false)
            .into_iter()
            .map(|token| token.word)
            .collect::<Vec<_>>(),
        vec!["他", "非常", "好"]
    );
    assert_eq!(
        hsk_control::JIEBA_EMBEDDED_DICTIONARY_SHA256,
        "139519822fe8ab9e10d9d07e68ea0451045380aedaf54ecc51e2a28c6b42a13f"
    );
    assert_eq!(hsk_control::JIEBA_CRATE_VERSION, "0.10.1");
}

fn preservation_violations(
    control: &HskControl,
    reference: &str,
    candidate: &str,
) -> Vec<PreservationViolation> {
    preservation_violations_with_names(control, reference, candidate, &[])
}

fn preservation_violations_with_names(
    control: &HskControl,
    reference: &str,
    candidate: &str,
    proper_names: &[hsk_control::ProperName],
) -> Vec<PreservationViolation> {
    match control
        .correction_loop(HskLevel::ONE, reference, proper_names)
        .evaluate(candidate)
    {
        hsk_control::CorrectionOutcome::Accepted { .. } => Vec::new(),
        hsk_control::CorrectionOutcome::Retry {
            preservation_violations,
            ..
        }
        | hsk_control::CorrectionOutcome::Failed {
            preservation_violations,
            ..
        } => preservation_violations,
        hsk_control::CorrectionOutcome::Terminated => unreachable!(),
    }
}

fn negation_control() -> HskControl {
    control(
        "negation",
        [
            "常", "很", "好", "错", "我", "你", "他", "吃", "也", "喝", "不", "没", "没有", "有",
            "来", "别", "走", "未", "完成", "非", "会员", "莫", "问",
        ]
        .into_iter()
        .map(|word| hsk_entry(word, HskLevel::ONE))
        .collect(),
        vec![dictionary_entry("词典")],
    )
}

fn control(
    revision: &str,
    hsk_entries: Vec<HskEntry>,
    dictionary_entries: Vec<DictionaryEntry>,
) -> HskControl {
    let hsk = hsk_artifact(revision, DatasetCompleteness::Complete, hsk_entries);
    let dictionary = DictionaryArtifact {
        schema_version: 1,
        format: "CC-CEDICT".into(),
        dataset_revision: revision.into(),
        completeness: DatasetCompleteness::Complete,
        source: source(revision),
        licence: licence(),
        audited_entry_count: dictionary_entries.len(),
        entries: dictionary_entries,
    };
    HskControl::from_json(
        &serde_json::to_string(&hsk).unwrap(),
        &serde_json::to_string(&dictionary).unwrap(),
    )
    .unwrap()
}

fn hsk_artifact(
    revision: &str,
    completeness: DatasetCompleteness,
    entries: Vec<HskEntry>,
) -> HskArtifact {
    let mut level_counts = [0usize; 6];
    for entry in &entries {
        level_counts[entry.level.index()] += 1;
    }
    HskArtifact {
        schema_version: 1,
        standard: HSK_STANDARD.into(),
        dataset_revision: revision.into(),
        completeness,
        source: source(revision),
        licence: licence(),
        audited_entry_count: entries.len(),
        audited_level_counts: level_counts,
        entries,
    }
}

fn hsk_entry(word: &str, level: HskLevel) -> HskEntry {
    HskEntry {
        simplified: word.into(),
        pinyin: "test".into(),
        glosses: vec![format!("definition for {word}")],
        level,
        simpler_words: Vec::new(),
        independently_usable: true,
        frequency_rank: None,
    }
}

fn dictionary_entry(word: &str) -> DictionaryEntry {
    DictionaryEntry {
        traditional: word.into(),
        simplified: word.into(),
        pinyin: "test".into(),
        definitions: vec![format!("definition for {word}")],
        frequency_rank: None,
    }
}

fn import_metadata(source_bytes: &[u8], completeness: DatasetCompleteness) -> ImportMetadata {
    ImportMetadata {
        schema_version: 1,
        kind: DatasetKind::Hsk20,
        standard: Some(HSK_STANDARD.into()),
        dataset_revision: "import-audit".into(),
        completeness,
        source: SourceAudit {
            name: "test source".into(),
            url: "project://test-source".into(),
            revision: "1".into(),
            sha256: sha256_hex(source_bytes),
        },
        licence: licence(),
        expected_entry_count: (completeness == DatasetCompleteness::Complete).then_some(1),
        expected_level_counts: (completeness == DatasetCompleteness::Complete)
            .then_some([1, 0, 0, 0, 0, 0]),
    }
}

fn source(revision: &str) -> SourceAudit {
    SourceAudit {
        name: "test".into(),
        url: "project://test".into(),
        revision: revision.into(),
        sha256: "0".repeat(64),
    }
}

fn licence() -> LicenceAudit {
    LicenceAudit {
        spdx_expression: "GPL-3.0-only".into(),
        url: "project://LICENSE".into(),
        attribution: "project-authored test data".into(),
        redistribution_allowed: true,
    }
}
