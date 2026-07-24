use std::{
    array,
    collections::{BTreeMap, BTreeSet},
};

use serde::Serialize;

use crate::{
    DATA_SCHEMA_VERSION, DatasetCompleteness, HSK_STANDARD, HskArtifact, HskControlError, HskEntry,
    HskLevel, LicenceAudit, LoadPolicy, Result, SourceAudit, TextNormalizer, sha256_hex,
};

/// Validated, normalized HSK data with cumulative level sets.
#[derive(Debug, Clone)]
pub struct HskDataset {
    entries: Vec<HskEntry>,
    by_word: BTreeMap<String, usize>,
    cumulative: [BTreeSet<String>; 6],
    dataset_revision: String,
    cache_revision: String,
    completeness: DatasetCompleteness,
    source: SourceAudit,
    licence: LicenceAudit,
}

impl HskDataset {
    pub fn from_json(json: &str, policy: LoadPolicy) -> Result<Self> {
        let artifact = serde_json::from_str::<HskArtifact>(json)?;
        Self::from_artifact(artifact, policy, &TextNormalizer::new())
    }

    pub(crate) fn from_artifact(
        mut artifact: HskArtifact,
        policy: LoadPolicy,
        normalizer: &TextNormalizer,
    ) -> Result<Self> {
        validate_header(
            artifact.schema_version,
            &artifact.standard,
            &artifact.dataset_revision,
            artifact.completeness,
            &artifact.source,
            &artifact.licence,
            policy,
        )?;

        artifact.entries.sort_by(|left, right| {
            left.level
                .cmp(&right.level)
                .then_with(|| left.simplified.cmp(&right.simplified))
                .then_with(|| left.pinyin.cmp(&right.pinyin))
        });

        if artifact.entries.is_empty() {
            return Err(HskControlError::InvalidData(
                "HSK artifact contains no entries".into(),
            ));
        }
        if artifact.audited_entry_count != artifact.entries.len() {
            return Err(HskControlError::InvalidData(format!(
                "HSK artifact contains {} entries but its audit records {}",
                artifact.entries.len(),
                artifact.audited_entry_count
            )));
        }

        let mut by_word = BTreeMap::new();
        let mut cumulative: [BTreeSet<String>; 6] = array::from_fn(|_| BTreeSet::new());
        let mut actual_level_counts = [0usize; 6];
        for (index, entry) in artifact.entries.iter().enumerate() {
            validate_entry(entry, normalizer)?;
            if by_word.insert(entry.simplified.clone(), index).is_some() {
                return Err(HskControlError::InvalidData(format!(
                    "duplicate HSK word {:?}",
                    entry.simplified
                )));
            }
            actual_level_counts[entry.level.index()] += 1;
            for allowed in cumulative.iter_mut().skip(entry.level.index()) {
                allowed.insert(entry.simplified.clone());
            }
        }
        if actual_level_counts != artifact.audited_level_counts {
            return Err(HskControlError::InvalidData(format!(
                "HSK artifact level counts {actual_level_counts:?} do not match its audit {:?}",
                artifact.audited_level_counts
            )));
        }

        let cache_revision = dataset_hash(
            &artifact.dataset_revision,
            artifact.completeness,
            &artifact.source,
            &(artifact.audited_entry_count, artifact.audited_level_counts),
            &artifact.entries,
        )?;

        Ok(Self {
            entries: artifact.entries,
            by_word,
            cumulative,
            dataset_revision: artifact.dataset_revision,
            cache_revision,
            completeness: artifact.completeness,
            source: artifact.source,
            licence: artifact.licence,
        })
    }

    pub fn entries(&self) -> &[HskEntry] {
        &self.entries
    }

    pub fn entry(&self, word: &str) -> Option<&HskEntry> {
        self.by_word.get(word).map(|index| &self.entries[*index])
    }

    pub fn level_of(&self, word: &str) -> Option<HskLevel> {
        self.entry(word).map(|entry| entry.level)
    }

    pub fn is_allowed(&self, word: &str, selected_level: HskLevel) -> bool {
        self.cumulative[selected_level.index()].contains(word)
    }

    pub fn allowed_words(&self, selected_level: HskLevel) -> &BTreeSet<String> {
        &self.cumulative[selected_level.index()]
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
}

fn validate_header(
    schema_version: u32,
    standard: &str,
    dataset_revision: &str,
    completeness: DatasetCompleteness,
    source: &SourceAudit,
    licence: &LicenceAudit,
    policy: LoadPolicy,
) -> Result<()> {
    if schema_version != DATA_SCHEMA_VERSION {
        return Err(HskControlError::InvalidData(format!(
            "unsupported HSK schema {schema_version}; expected {DATA_SCHEMA_VERSION}"
        )));
    }
    if standard != HSK_STANDARD {
        return Err(HskControlError::InvalidData(format!(
            "unsupported HSK standard {standard:?}; expected {HSK_STANDARD:?}"
        )));
    }
    if dataset_revision.trim().is_empty() {
        return Err(HskControlError::InvalidData(
            "HSK dataset revision is empty".into(),
        ));
    }
    validate_audit(source, licence)?;
    if completeness != DatasetCompleteness::Complete && policy == LoadPolicy::RequireComplete {
        return Err(HskControlError::DatasetIncomplete {
            resource: "HSK",
            revision: dataset_revision.to_owned(),
            completeness,
        });
    }
    Ok(())
}

pub(crate) fn validate_audit(source: &SourceAudit, licence: &LicenceAudit) -> Result<()> {
    if source.name.trim().is_empty()
        || source.url.trim().is_empty()
        || source.revision.trim().is_empty()
        || source.sha256.len() != 64
        || !source
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(HskControlError::InvalidData(
            "artifact has an invalid source audit".into(),
        ));
    }
    if licence.spdx_expression.trim().is_empty()
        || licence.url.trim().is_empty()
        || licence.attribution.trim().is_empty()
        || !licence.redistribution_allowed
    {
        return Err(HskControlError::LicenceAudit(
            "artifact licence audit is incomplete or disallows redistribution".into(),
        ));
    }
    Ok(())
}

fn validate_entry(entry: &HskEntry, normalizer: &TextNormalizer) -> Result<()> {
    if entry.simplified.trim().is_empty()
        || entry.pinyin.trim().is_empty()
        || entry.glosses.is_empty()
        || entry.glosses.iter().any(|gloss| gloss.trim().is_empty())
    {
        return Err(HskControlError::InvalidData(format!(
            "incomplete HSK entry {:?}",
            entry.simplified
        )));
    }
    let normalized = normalizer.normalize(&entry.simplified);
    if normalized != entry.simplified {
        return Err(HskControlError::InvalidData(format!(
            "HSK entry {:?} is not normalized Simplified Chinese (normalizes to {normalized:?})",
            entry.simplified
        )));
    }
    Ok(())
}

fn dataset_hash<T: Serialize>(
    revision: &str,
    completeness: DatasetCompleteness,
    source: &SourceAudit,
    audited_counts: &(usize, [usize; 6]),
    entries: &T,
) -> Result<String> {
    let bytes = serde_json::to_vec(&(
        DATA_SCHEMA_VERSION,
        revision,
        completeness,
        source,
        audited_counts,
        entries,
    ))?;
    Ok(format!("hsk2-sha256:{}", sha256_hex(&bytes)))
}
