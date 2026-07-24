use std::time::Instant;

use hsk_control::{
    DatasetCompleteness, DictionaryArtifact, DictionaryEntry, HSK_STANDARD, HskArtifact,
    HskControl, HskEntry, HskLevel, LicenceAudit, SourceAudit,
};

/// Full-scale synthetic smoke: roughly the current CC-CEDICT entry count plus
/// 5,000 HSK-shaped records. No third-party lexical data is embedded.
#[test]
#[ignore = "explicit full-scale performance smoke"]
fn full_lexicon_scale_load_validate_and_lookup() {
    const HSK_ENTRIES: usize = 5_000;
    const DICTIONARY_ENTRIES: usize = 125_000;

    let source = SourceAudit {
        name: "synthetic performance data".into(),
        url: "project://synthetic-performance".into(),
        revision: "1".into(),
        sha256: "0".repeat(64),
    };
    let licence = LicenceAudit {
        spdx_expression: "GPL-3.0-only".into(),
        url: "project://LICENSE".into(),
        attribution: "generated during test".into(),
        redistribution_allowed: true,
    };

    let mut level_counts = [0usize; 6];
    let hsk_entries = (0..HSK_ENTRIES)
        .map(|index| {
            let level = HskLevel::new((index % 6 + 1) as u8).unwrap();
            level_counts[level.index()] += 1;
            HskEntry {
                simplified: format!("学词{index:05}"),
                pinyin: "xué cí".into(),
                glosses: vec![format!("synthetic term {index}")],
                level,
                simpler_words: Vec::new(),
                independently_usable: true,
                frequency_rank: Some(index as u32 + 1),
            }
        })
        .collect::<Vec<_>>();
    let dictionary_entries = (0..DICTIONARY_ENTRIES)
        .map(|index| DictionaryEntry {
            traditional: format!("辭條{index:06}"),
            simplified: format!("辞条{index:06}"),
            pinyin: "cí tiáo".into(),
            definitions: vec![format!("synthetic dictionary entry {index}")],
            frequency_rank: Some(index as u32 + 1),
        })
        .collect::<Vec<_>>();

    let hsk = HskArtifact {
        schema_version: 1,
        standard: HSK_STANDARD.into(),
        dataset_revision: "synthetic-full-scale-1".into(),
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
        dataset_revision: "synthetic-full-scale-1".into(),
        completeness: DatasetCompleteness::Complete,
        source,
        licence,
        audited_entry_count: dictionary_entries.len(),
        entries: dictionary_entries,
    };

    let started = Instant::now();
    let control = HskControl::from_json(
        &serde_json::to_string(&hsk).unwrap(),
        &serde_json::to_string(&dictionary).unwrap(),
    )
    .unwrap();
    let load_elapsed = started.elapsed();
    assert!(
        load_elapsed.as_secs() < 120,
        "synthetic full-scale resources took {load_elapsed:?} to load"
    );

    let operations_started = Instant::now();
    for _ in 0..1_000 {
        let report = control.validate("学词00001", HskLevel::SIX, &[]);
        assert!(report.strictly_valid);
        let lookup = control.lookup("辞条124999", &[]);
        assert_eq!(lookup.tokens[0].simplified, "辞条124999");
    }
    let operations_elapsed = operations_started.elapsed();
    assert!(
        operations_elapsed.as_secs() < 30,
        "1,000 validate+lookup operations took {operations_elapsed:?}"
    );
}
