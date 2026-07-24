use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    DATA_SCHEMA_VERSION, DatasetCompleteness, DictionaryArtifact, DictionaryEntry, HskControlError,
    HskLevel, LicenceAudit, LoadPolicy, Result, SourceAudit, TextNormalizer, trie::AllowedWordTrie,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LookupToken {
    pub simplified: String,
    pub pinyin: String,
    pub definitions: Vec<String>,
    pub hsk_level: Option<HskLevel>,
    pub proper_name: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LookupResult {
    pub selected_text: String,
    pub tokens: Vec<LookupToken>,
}

/// Parsed CC-CEDICT-compatible entries and a deterministic longest-match trie.
#[derive(Debug, Clone)]
pub struct LocalDictionary {
    entries: Vec<DictionaryEntry>,
    by_word: BTreeMap<String, Vec<usize>>,
    trie: AllowedWordTrie,
    dataset_revision: String,
    cache_revision: String,
    completeness: DatasetCompleteness,
    source: SourceAudit,
    licence: LicenceAudit,
}

impl LocalDictionary {
    pub fn from_json(json: &str, policy: LoadPolicy) -> Result<Self> {
        let artifact = serde_json::from_str::<DictionaryArtifact>(json)?;
        Self::from_artifact(artifact, policy, &TextNormalizer::new())
    }

    pub(crate) fn from_artifact(
        mut artifact: DictionaryArtifact,
        policy: LoadPolicy,
        normalizer: &TextNormalizer,
    ) -> Result<Self> {
        if artifact.schema_version != DATA_SCHEMA_VERSION {
            return Err(HskControlError::InvalidData(format!(
                "unsupported dictionary schema {}; expected {DATA_SCHEMA_VERSION}",
                artifact.schema_version
            )));
        }
        if artifact.format != "CC-CEDICT" {
            return Err(HskControlError::InvalidData(format!(
                "unsupported dictionary format {:?}",
                artifact.format
            )));
        }
        if artifact.dataset_revision.trim().is_empty() {
            return Err(HskControlError::InvalidData(
                "dictionary dataset revision is empty".into(),
            ));
        }
        super::dataset::validate_audit(&artifact.source, &artifact.licence)?;
        if artifact.completeness != DatasetCompleteness::Complete
            && policy == LoadPolicy::RequireComplete
        {
            return Err(HskControlError::DatasetIncomplete {
                resource: "dictionary",
                revision: artifact.dataset_revision,
                completeness: artifact.completeness,
            });
        }

        artifact.entries.sort_by(|left, right| {
            left.simplified
                .cmp(&right.simplified)
                .then_with(|| left.pinyin.cmp(&right.pinyin))
                .then_with(|| left.definitions.cmp(&right.definitions))
        });
        artifact.entries.dedup();
        if artifact.audited_entry_count != artifact.entries.len() {
            return Err(HskControlError::InvalidData(format!(
                "dictionary artifact contains {} unique entries but its audit records {}",
                artifact.entries.len(),
                artifact.audited_entry_count
            )));
        }
        if artifact.entries.is_empty() {
            return Err(HskControlError::InvalidData(
                "dictionary artifact contains no entries".into(),
            ));
        }

        let mut by_word: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut trie = AllowedWordTrie::new();
        for (index, entry) in artifact.entries.iter().enumerate() {
            validate_entry(entry, normalizer)?;
            by_word
                .entry(entry.simplified.clone())
                .or_default()
                .push(index);
            trie.insert(&entry.simplified);
        }

        let hash_bytes = serde_json::to_vec(&(
            DATA_SCHEMA_VERSION,
            &artifact.dataset_revision,
            artifact.completeness,
            &artifact.source,
            artifact.audited_entry_count,
            &artifact.entries,
        ))?;
        let cache_revision = format!("cedict-sha256:{}", crate::sha256_hex(&hash_bytes));

        Ok(Self {
            entries: artifact.entries,
            by_word,
            trie,
            dataset_revision: artifact.dataset_revision,
            cache_revision,
            completeness: artifact.completeness,
            source: artifact.source,
            licence: artifact.licence,
        })
    }

    pub fn entries(&self) -> &[DictionaryEntry] {
        &self.entries
    }

    pub fn entries_for(&self, word: &str) -> impl Iterator<Item = &DictionaryEntry> {
        self.by_word
            .get(word)
            .into_iter()
            .flatten()
            .map(|index| &self.entries[*index])
    }

    pub fn contains_word(&self, word: &str) -> bool {
        self.by_word.contains_key(word)
    }

    pub(crate) fn longest_match(&self, characters: &[char], start: usize) -> Option<usize> {
        self.trie.longest_match(characters, start)
    }

    pub fn words(&self) -> impl Iterator<Item = &str> {
        self.by_word.keys().map(String::as_str)
    }

    pub fn dataset_revision(&self) -> &str {
        &self.dataset_revision
    }

    pub fn cache_revision(&self) -> &str {
        &self.cache_revision
    }

    pub fn completeness(&self) -> DatasetCompleteness {
        self.completeness
    }

    pub fn source(&self) -> &SourceAudit {
        &self.source
    }

    pub fn licence(&self) -> &LicenceAudit {
        &self.licence
    }

    pub(crate) fn merged_fields(&self, word: &str) -> (String, Vec<String>) {
        let mut pinyin = BTreeSet::new();
        let mut definitions = BTreeSet::new();
        for entry in self.entries_for(word) {
            pinyin.insert(entry.pinyin.clone());
            definitions.extend(entry.definitions.iter().cloned());
        }
        (
            pinyin.into_iter().collect::<Vec<_>>().join(" / "),
            definitions.into_iter().collect(),
        )
    }
}

fn validate_entry(entry: &DictionaryEntry, normalizer: &TextNormalizer) -> Result<()> {
    if entry.traditional.trim().is_empty()
        || entry.simplified.trim().is_empty()
        || entry.pinyin.trim().is_empty()
        || entry.definitions.is_empty()
        || entry
            .definitions
            .iter()
            .any(|definition| definition.trim().is_empty())
    {
        return Err(HskControlError::InvalidData(format!(
            "incomplete dictionary entry {:?}",
            entry.simplified
        )));
    }
    let normalized = normalizer.normalize(&entry.simplified);
    if normalized != entry.simplified {
        return Err(HskControlError::InvalidData(format!(
            "dictionary entry {:?} is not normalized Simplified Chinese (normalizes to {normalized:?})",
            entry.simplified
        )));
    }
    Ok(())
}
