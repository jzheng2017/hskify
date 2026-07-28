#![cfg(feature = "test-seeds")]

use hsk_control::{
    AllowedWordTrie, DatasetCompleteness, DictionaryArtifact, DictionaryEntry,
    EMBEDDED_DICTIONARY_TEST_SEED, EMBEDDED_HSK_TEST_SEED, HSK_STANDARD, HskArtifact, HskControl,
    HskControlError, HskEntry, HskLevel, LicenceAudit, ProperName, ProperNameReason, SourceAudit,
    TextNormalizer, ViolationReason,
};

#[test]
fn production_policy_rejects_the_incomplete_embedded_seed() {
    let error = HskControl::from_json(EMBEDDED_HSK_TEST_SEED, EMBEDDED_DICTIONARY_TEST_SEED)
        .err()
        .expect("production load must reject the seed");
    assert!(matches!(
        error,
        HskControlError::DatasetIncomplete {
            resource: "HSK",
            completeness: DatasetCompleteness::TestSeed,
            ..
        }
    ));
}

#[test]
fn cumulative_union_is_monotonic_for_all_six_levels() {
    let control = seed();
    let mut previous: Option<&std::collections::BTreeSet<String>> = None;
    for level in HskLevel::ALL {
        let allowed = control.allowed_words(level);
        if let Some(previous) = previous {
            assert!(
                previous.is_subset(allowed),
                "level {level} lost an earlier word"
            );
        }
        for entry in control.hsk_dataset().entries() {
            assert_eq!(
                allowed.contains(&entry.simplified),
                entry.level <= level,
                "word={} selected={level}",
                entry.simplified
            );
        }
        previous = Some(allowed);
    }
}

#[test]
fn unicode_zero_width_punctuation_whitespace_and_traditional_are_normalized() {
    let normalizer = TextNormalizer::new();
    assert_eq!(normalizer.normalize("  學\u{200b}習！\nＡ  "), "学习！ A");
    assert_eq!(normalizer.normalize("臺灣軟體"), "台湾软件");
    assert_eq!(normalizer.normalize("三．五，１２３"), "三.五，123");
}

#[test]
fn known_higher_level_compound_cannot_pass_by_character_splitting() {
    let report = seed().validate("马上", HskLevel::ONE, &[]);
    assert!(!report.strictly_valid);
    assert_eq!(report.violations.len(), 1);
    assert_eq!(report.violations[0].text, "马上");
    assert_eq!(
        report.violations[0].reason,
        ViolationReason::AboveSelectedHskLevel {
            required_level: HskLevel::TWO
        }
    );
}

#[test]
fn numeric_prefix_cannot_hide_a_known_higher_level_compound() {
    let report = seed().validate("三马上", HskLevel::ONE, &[]);
    assert!(!report.strictly_valid);
    assert!(
        report
            .violations
            .iter()
            .any(|violation| violation.text.contains("马上"))
    );
}

#[test]
fn dictionary_phrase_with_allowed_surface_words_is_not_a_shadow_hsk_violation() {
    let hsk = vec![
        entry("研究", HskLevel::ONE, &["research"]),
        entry("生", HskLevel::ONE, &["student"]),
    ];
    let dictionary = vec![dictionary_entry(
        "研究生",
        "yán jiū shēng",
        &["graduate student"],
    )];
    let control = custom_control("ambiguous-compound", hsk, dictionary);

    let report = control.validate("研究生", HskLevel::ONE, &[]);
    assert!(
        report.strictly_valid,
        "dictionary phrase boundaries must not override the selected HSK surface vocabulary: {:?}",
        report.violations
    );
    assert_eq!(report.lexical_token_count, 2);
}

#[test]
fn trie_dynamic_programming_covers_the_complete_span() {
    let mut trie = AllowedWordTrie::new();
    for word in ["我们", "现在", "学习"] {
        trie.insert(word);
    }
    assert_eq!(
        trie.best_decomposition("我们现在学习"),
        Some(vec!["我们".into(), "现在".into(), "学习".into()])
    );
    assert!(!trie.can_decompose("我们未知"));
}

#[test]
fn explicit_names_and_chinese_numbers_are_allowed_and_reported() {
    let proper_names = [ProperName {
        text: "小明".into(),
        reason: ProperNameReason::PersonName,
    }];
    let report = seed().validate("小明有一百二十三个", HskLevel::ONE, &proper_names);
    assert!(report.strictly_valid, "{:?}", report.violations);
    assert_eq!(report.exceptions.len(), 1);
    assert_eq!(report.exceptions[0].text, "小明");
    assert_eq!(report.exceptions[0].start_char, 0);
    assert_eq!(report.exceptions[0].end_char, 2);
}

#[test]
fn violations_use_exact_unicode_character_offsets() {
    let report = seed().validate("🙂你好，斡旋!", HskLevel::FIVE, &[]);
    assert_eq!(report.normalized_text, "🙂你好，斡旋！");
    assert_eq!(report.violations.len(), 1);
    let violation = &report.violations[0];
    assert_eq!(violation.text, "斡旋");
    assert_eq!((violation.start_char, violation.end_char), (4, 6));
    assert_eq!(
        report
            .normalized_text
            .chars()
            .skip(violation.start_char)
            .take(violation.end_char - violation.start_char)
            .collect::<String>(),
        violation.text
    );
}

#[test]
fn hsk_one_words_are_allowed_at_five_and_hsk_six_words_are_not() {
    let control = seed();
    assert!(control.validate("你好", HskLevel::FIVE, &[]).strictly_valid);
    let report = control.validate("斡旋", HskLevel::FIVE, &[]);
    assert_eq!(report.violations.len(), 1);
    assert!(
        report.violations[0]
            .suggested_words
            .contains(&"说话".into())
    );
    assert_eq!(
        report.violations[0].reason,
        ViolationReason::AboveSelectedHskLevel {
            required_level: HskLevel::SIX
        }
    );
}

#[test]
fn strict_fixture_has_no_unlabelled_vocabulary_violation() {
    let report = seed().validate("我们现在有三个", HskLevel::ONE, &[]);
    assert!(report.strictly_valid, "{:?}", report.violations);
    assert!(report.exceptions.is_empty());
}

#[test]
fn dictionary_lookup_uses_longest_match_pinyin_gloss_and_hsk_overlay() {
    let control = seed();
    let lookup = control.lookup("研究生，离开", &[]);
    assert_eq!(lookup.tokens.len(), 2);
    assert_eq!(lookup.tokens[0].simplified, "研究生");
    assert_eq!(lookup.tokens[0].pinyin, "yán jiū shēng");
    assert!(
        lookup.tokens[0]
            .definitions
            .contains(&"graduate student".into())
    );
    assert_eq!(lookup.tokens[0].hsk_level, None);
    assert_eq!(lookup.tokens[1].simplified, "离开");
    assert_eq!(lookup.tokens[1].pinyin, "lí kāi");
    assert_eq!(lookup.tokens[1].hsk_level, Some(HskLevel::TWO));
}

#[test]
fn hover_lookup_is_longest_match_anchored_at_the_hovered_character() {
    let control = seed();
    let whole = control
        .lookup_at_with_region_context("研究生", 0, &[], None)
        .expect("the first character starts a dictionary expression");
    let later = control
        .lookup_at_with_region_context("研究生", 2, &[], None)
        .expect("a later component starts its own anchored lookup");

    assert_eq!(whole.tokens.len(), 1);
    assert_eq!(whole.selected_text, "研究生");
    assert_eq!(whole.tokens[0].simplified, "研究生");
    assert_eq!(later.tokens.len(), 1);
    assert_eq!(later.selected_text, "生");
    assert_eq!(later.tokens[0].simplified, "生");
    assert!(
        control
            .lookup_at_with_region_context("研究生。", 3, &[], None)
            .is_none(),
        "punctuation must not jump forward to an unrelated word"
    );
}

#[test]
fn lookup_marks_only_explicit_proper_names() {
    let names = [ProperName {
        text: "小明".into(),
        reason: ProperNameReason::PersonName,
    }];
    let lookup = seed().lookup("小明", &names);
    assert_eq!(lookup.tokens.len(), 1);
    assert!(lookup.tokens[0].proper_name);
}

#[test]
fn lookup_composes_pinyin_for_a_proper_name_missing_as_a_whole_dictionary_entry() {
    let control = custom_control(
        "proper-name-pinyin",
        vec![entry("我", HskLevel::ONE, &["I"])],
        vec![
            dictionary_entry("阿", "ā", &["prefix used in names"]),
            dictionary_entry("忠", "zhōng", &["loyal"]),
        ],
    );
    let names = [ProperName {
        text: "阿忠".into(),
        reason: ProperNameReason::PersonName,
    }];

    let lookup = control.lookup("阿忠", &names);

    assert_eq!(lookup.tokens.len(), 1);
    assert_eq!(lookup.tokens[0].pinyin, "ā zhōng");
    assert!(lookup.tokens[0].proper_name);
}

#[test]
fn cache_revision_changes_when_a_dataset_revision_changes() {
    let hsk = vec![entry("我", HskLevel::ONE, &["I"])];
    let dictionary = vec![dictionary_entry("我", "wǒ", &["I", "me"])];
    let first = custom_control("revision-a", hsk.clone(), dictionary.clone());
    let second = custom_control("revision-b", hsk, dictionary);
    assert_ne!(first.cache_revision(), second.cache_revision());
}

fn seed() -> HskControl {
    HskControl::from_embedded_test_seed().expect("embedded seed is internally valid")
}

fn custom_control(
    revision: &str,
    hsk_entries: Vec<HskEntry>,
    dictionary_entries: Vec<DictionaryEntry>,
) -> HskControl {
    let source = SourceAudit {
        name: "test".into(),
        url: "project://test".into(),
        revision: revision.into(),
        sha256: "0".repeat(64),
    };
    let licence = LicenceAudit {
        spdx_expression: "GPL-3.0-only".into(),
        url: "project://LICENSE".into(),
        attribution: "test".into(),
        redistribution_allowed: true,
    };
    let mut level_counts = [0usize; 6];
    for entry in &hsk_entries {
        level_counts[entry.level.index()] += 1;
    }
    let hsk = HskArtifact {
        schema_version: 1,
        standard: HSK_STANDARD.into(),
        dataset_revision: revision.into(),
        completeness: DatasetCompleteness::Complete,
        source: source.clone(),
        licence: licence.clone(),
        audited_entry_count: hsk_entries.len(),
        audited_level_counts: level_counts,
        entries: hsk_entries,
    };
    let dictionary = DictionaryArtifact {
        schema_version: 1,
        format: "CC-CEDICT".into(),
        dataset_revision: revision.into(),
        completeness: DatasetCompleteness::Complete,
        source,
        licence,
        audited_entry_count: dictionary_entries.len(),
        entries: dictionary_entries,
    };
    HskControl::from_json(
        &serde_json::to_string(&hsk).unwrap(),
        &serde_json::to_string(&dictionary).unwrap(),
    )
    .unwrap()
}

fn entry(word: &str, level: HskLevel, glosses: &[&str]) -> HskEntry {
    HskEntry {
        simplified: word.into(),
        pinyin: "test".into(),
        glosses: glosses.iter().map(|value| (*value).into()).collect(),
        level,
        simpler_words: Vec::new(),
        independently_usable: true,
        frequency_rank: None,
    }
}

fn dictionary_entry(word: &str, pinyin: &str, definitions: &[&str]) -> DictionaryEntry {
    DictionaryEntry {
        traditional: word.into(),
        simplified: word.into(),
        pinyin: pinyin.into(),
        definitions: definitions.iter().map(|value| (*value).into()).collect(),
        frequency_rank: None,
    }
}
