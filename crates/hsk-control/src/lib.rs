//! Deterministic HSK 2.0 vocabulary validation and local dictionary lookup.
//!
//! The crate intentionally has no network, model, browser, or async dependency.
//! Full runtime resources are loaded from licence-audited generated artifacts.
//! The repository's embedded data is an explicitly incomplete test seed; callers
//! cannot construct a production engine from it without opting into
//! [`LoadPolicy::AllowIncompleteTestSeed`].

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
pub use dictionary::{LocalDictionary, LookupResult, LookupToken};
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
pub use normalization::{NORMALIZATION_REVISION, TextNormalizer, is_han, is_numeric_token};
pub use trie::AllowedWordTrie;
pub use validator::HskControl;

/// HSK vocabulary standard supported by this crate.
pub const HSK_STANDARD: &str = "2.0";

/// Version of the generated artifact schema.
pub const DATA_SCHEMA_VERSION: u32 = 1;

/// Segmentation policy revision included in cache identities.
pub const SEGMENTATION_REVISION: &str = "jieba-0.10-full-lexicon-conservative-compound-guard-v1";

/// Dictionary lookup policy revision included in cache identities.
pub const LOOKUP_REVISION: &str = "longest-match-simplified-v1";

/// Embedded HSK test seed. It is never accepted under the default load policy.
pub const EMBEDDED_HSK_TEST_SEED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/hsk/test-seed.normalized.json"
));

/// Embedded dictionary test seed. It is never accepted under the default load
/// policy.
pub const EMBEDDED_DICTIONARY_TEST_SEED: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/dictionary/test-seed.normalized.json"
));
