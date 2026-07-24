use std::{fmt, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use crate::{DATA_SCHEMA_VERSION, HSK_STANDARD, HskControlError, Result};

/// One of the six levels in the established HSK 2.0 system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HskLevel(u8);

impl HskLevel {
    pub const ONE: Self = Self(1);
    pub const TWO: Self = Self(2);
    pub const THREE: Self = Self(3);
    pub const FOUR: Self = Self(4);
    pub const FIVE: Self = Self(5);
    pub const SIX: Self = Self(6);
    pub const ALL: [Self; 6] = [
        Self::ONE,
        Self::TWO,
        Self::THREE,
        Self::FOUR,
        Self::FIVE,
        Self::SIX,
    ];

    pub fn new(value: u8) -> Result<Self> {
        if (1..=6).contains(&value) {
            Ok(Self(value))
        } else {
            Err(HskControlError::InvalidHskLevel(value))
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    pub const fn index(self) -> usize {
        (self.0 - 1) as usize
    }
}

impl TryFrom<u8> for HskLevel {
    type Error = HskControlError;

    fn try_from(value: u8) -> Result<Self> {
        Self::new(value)
    }
}

impl FromStr for HskLevel {
    type Err = HskControlError;

    fn from_str(value: &str) -> Result<Self> {
        value
            .parse::<u8>()
            .map_err(|_| HskControlError::InvalidData(format!("invalid HSK level {value:?}")))
            .and_then(Self::new)
    }
}

impl fmt::Display for HskLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Serialize for HskLevel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(self.0)
    }
}

impl<'de> Deserialize<'de> for HskLevel {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetCompleteness {
    Complete,
    TestSeed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetKind {
    #[serde(rename = "hsk-2.0")]
    Hsk20,
    #[serde(rename = "cc-cedict")]
    CcCedict,
}

/// Loading incomplete data always requires an explicit test-only opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadPolicy {
    RequireComplete,
    AllowIncompleteTestSeed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceAudit {
    pub name: String,
    pub url: String,
    pub revision: String,
    /// Lowercase SHA-256 of the exact source bytes supplied to the importer.
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LicenceAudit {
    pub spdx_expression: String,
    pub url: String,
    pub attribution: String,
    /// Generated artifacts intended for distribution are rejected unless this
    /// field was affirmatively audited as true.
    pub redistribution_allowed: bool,
}

/// Sidecar supplied to a deterministic importer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportMetadata {
    pub schema_version: u32,
    pub kind: DatasetKind,
    pub standard: Option<String>,
    pub dataset_revision: String,
    pub completeness: DatasetCompleteness,
    pub source: SourceAudit,
    pub licence: LicenceAudit,
    pub expected_entry_count: Option<usize>,
    pub expected_level_counts: Option<[usize; 6]>,
}

impl ImportMetadata {
    pub fn validate_common(&self, expected_kind: DatasetKind) -> Result<()> {
        if self.schema_version != DATA_SCHEMA_VERSION {
            return Err(HskControlError::InvalidData(format!(
                "unsupported metadata schema {}; expected {DATA_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if self.kind != expected_kind {
            return Err(HskControlError::InvalidData(format!(
                "metadata kind {:?} does not match {:?}",
                self.kind, expected_kind
            )));
        }
        if expected_kind == DatasetKind::Hsk20 && self.standard.as_deref() != Some(HSK_STANDARD) {
            return Err(HskControlError::InvalidData(format!(
                "HSK artifact standard must be {HSK_STANDARD:?}"
            )));
        }
        if self.dataset_revision.trim().is_empty()
            || self.source.name.trim().is_empty()
            || self.source.url.trim().is_empty()
            || self.source.revision.trim().is_empty()
        {
            return Err(HskControlError::InvalidData(
                "dataset/source identity fields must be non-empty".into(),
            ));
        }
        if self.source.sha256.len() != 64
            || !self
                .source
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(HskControlError::InvalidData(
                "source.sha256 must be 64 lowercase hexadecimal characters".into(),
            ));
        }
        if self.licence.spdx_expression.trim().is_empty()
            || self.licence.url.trim().is_empty()
            || self.licence.attribution.trim().is_empty()
        {
            return Err(HskControlError::LicenceAudit(
                "SPDX expression, licence URL, and attribution are required".into(),
            ));
        }
        if !self.licence.redistribution_allowed {
            return Err(HskControlError::LicenceAudit(
                "redistributionAllowed was not affirmatively audited".into(),
            ));
        }
        if self.completeness == DatasetCompleteness::Complete && self.expected_entry_count.is_none()
        {
            return Err(HskControlError::InvalidData(
                "complete datasets require expectedEntryCount".into(),
            ));
        }
        if expected_kind == DatasetKind::Hsk20
            && self.completeness == DatasetCompleteness::Complete
            && self.expected_level_counts.is_none()
        {
            return Err(HskControlError::InvalidData(
                "complete HSK datasets require expectedLevelCounts".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HskEntry {
    pub simplified: String,
    pub pinyin: String,
    pub glosses: Vec<String>,
    pub level: HskLevel,
    #[serde(default)]
    pub simpler_words: Vec<String>,
    /// Whether an entry may participate as one component of an otherwise
    /// unknown compound. Omission is deliberately conservative.
    #[serde(default)]
    pub independently_usable: bool,
    #[serde(default)]
    pub frequency_rank: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DictionaryEntry {
    pub traditional: String,
    pub simplified: String,
    /// Human-readable pinyin. The importer converts numbered CC-CEDICT pinyin
    /// to tone marks deterministically.
    pub pinyin: String,
    pub definitions: Vec<String>,
    #[serde(default)]
    pub frequency_rank: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HskArtifact {
    pub schema_version: u32,
    pub standard: String,
    pub dataset_revision: String,
    pub completeness: DatasetCompleteness,
    pub source: SourceAudit,
    pub licence: LicenceAudit,
    pub audited_entry_count: usize,
    pub audited_level_counts: [usize; 6],
    pub entries: Vec<HskEntry>,
}

impl<'de> Deserialize<'de> for HskArtifact {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct WireArtifact {
            schema_version: u32,
            standard: String,
            dataset_revision: String,
            completeness: DatasetCompleteness,
            source: SourceAudit,
            licence: LicenceAudit,
            audited_entry_count: usize,
            audited_level_counts: [usize; 6],
            entries: Vec<serde_json::Value>,
        }

        let wire = WireArtifact::deserialize(deserializer)?;
        let mut entries = Vec::with_capacity(wire.entries.len());
        for (index, value) in wire.entries.into_iter().enumerate() {
            let independently_usable_is_explicit_bool = value
                .get("independentlyUsable")
                .is_some_and(serde_json::Value::is_boolean);
            if wire.completeness == DatasetCompleteness::Complete
                && !independently_usable_is_explicit_bool
            {
                let word = value
                    .get("simplified")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("<unknown>");
                return Err(D::Error::custom(format!(
                    "complete HSK entry {index} ({word:?}) must explicitly audit independentlyUsable as true or false"
                )));
            }
            entries.push(serde_json::from_value(value).map_err(D::Error::custom)?);
        }

        Ok(Self {
            schema_version: wire.schema_version,
            standard: wire.standard,
            dataset_revision: wire.dataset_revision,
            completeness: wire.completeness,
            source: wire.source,
            licence: wire.licence,
            audited_entry_count: wire.audited_entry_count,
            audited_level_counts: wire.audited_level_counts,
            entries,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DictionaryArtifact {
    pub schema_version: u32,
    pub format: String,
    pub dataset_revision: String,
    pub completeness: DatasetCompleteness,
    pub source: SourceAudit,
    pub licence: LicenceAudit,
    pub audited_entry_count: usize,
    pub entries: Vec<DictionaryEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProperNameReason {
    PersonName,
    PlaceName,
    Title,
    UnavoidableProperNoun,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProperName {
    pub text: String,
    pub reason: ProperNameReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HskException {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub reason: ProperNameReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ViolationReason {
    AboveSelectedHskLevel { required_level: HskLevel },
    KnownDictionaryWord,
    UnknownChineseWord,
    NonChineseLexicalToken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HskViolation {
    pub text: String,
    /// Inclusive character offset into `ValidationReport.normalized_text`.
    pub start_char: usize,
    /// Exclusive character offset into `ValidationReport.normalized_text`.
    pub end_char: usize,
    pub reason: ViolationReason,
    pub suggested_words: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationReport {
    pub normalized_text: String,
    pub requested_level: HskLevel,
    pub strictly_valid: bool,
    pub violations: Vec<HskViolation>,
    pub exceptions: Vec<HskException>,
    pub cache_revision: String,
}
