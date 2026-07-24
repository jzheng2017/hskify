//! Deterministic HSK 2.0 vocabulary validation and local dictionary lookup.
//!
//! The crate intentionally has no network, model, browser, or async dependency.
//! Full runtime resources are loaded from licence-audited generated artifacts.
//! The repository's embedded, explicitly incomplete fixtures are compiled only
//! when the non-default `test-seeds` feature is enabled.

mod correction;
mod dataset;
mod dictionary;
mod error;
mod import;
mod model;
mod normalization;
mod trie;
mod validator;

pub use correction::{
    CorrectionContext, CorrectionLoop, CorrectionOutcome, MAX_CORRECTION_ATTEMPTS,
    PreservationViolation,
};
pub use dataset::HskDataset;
pub use dictionary::{LocalDictionary, LookupRegionContext, LookupResult, LookupToken};
pub use error::{HskControlError, Result};
pub use import::{
    Delimiter, generate_dictionary_artifact, generate_hsk_artifact, parse_import_metadata,
    sha256_hex,
};
pub use model::{
    DatasetCompleteness, DatasetKind, DictionaryArtifact, DictionaryEntry, HskArtifact, HskEntry,
    HskException, HskLevel, HskViolation, ImportMetadata, LicenceAudit, LoadPolicy, ProperName,
    ProperNameReason, SourceAudit, ValidationReport, ViolationReason,
};
pub use normalization::{
    NORMALIZATION_REVISION, TextNormalizer, UNICODE_NORMALIZATION_CRATE_VERSION,
    UNICODE_NORMALIZATION_TABLES_SHA256, UNICODE_NORMALIZATION_UNICODE_VERSION, is_han,
    is_numeric_token,
};
pub use trie::AllowedWordTrie;
pub use validator::HskControl;

/// HSK vocabulary standard supported by this crate.
pub const HSK_STANDARD: &str = "2.0";

/// Version of the generated artifact schema.
pub const DATA_SCHEMA_VERSION: u32 = 1;

/// Segmentation policy revision included in cache identities.
pub const SEGMENTATION_REVISION: &str =
    "jieba-full-lexicon-boundary-independent-conservative-span-guard-v2";

/// Dictionary lookup policy revision included in cache identities.
pub const LOOKUP_REVISION: &str = "longest-match-simplified-optional-region-context-v2";

/// Correction-preservation policy revision included in cache identities.
pub const PRESERVATION_REVISION: &str = "numbers-names-token-context-negation-v2";

/// Exact `jieba-rs` release whose segmentation behavior is cache-relevant.
pub const JIEBA_CRATE_VERSION: &str = "0.10.1";

/// SHA-256 of `jieba-rs` 0.10.1's embedded `src/data/dict.txt`.
pub const JIEBA_EMBEDDED_DICTIONARY_SHA256: &str =
    "139519822fe8ab9e10d9d07e68ea0451045380aedaf54ecc51e2a28c6b42a13f";

/// Embedded HSK test seed. Available only through the non-default
/// `test-seeds` feature and never accepted under the default load policy.
#[cfg(feature = "test-seeds")]
pub const EMBEDDED_HSK_TEST_SEED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/hsk/test-seed.normalized.json"
));

/// Embedded dictionary test seed. Available only through the non-default
/// `test-seeds` feature and never accepted under the default load policy.
#[cfg(feature = "test-seeds")]
pub const EMBEDDED_DICTIONARY_TEST_SEED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/dictionary/test-seed.normalized.json"
));
