use crate::DatasetCompleteness;
use thiserror::Error;

/// Errors produced by dataset import, construction, validation, and lookup.
#[derive(Debug, Error)]
pub enum HskControlError {
    #[error("invalid HSK level {0}; expected 1 through 6")]
    InvalidHskLevel(u8),

    #[error("invalid data: {0}")]
    InvalidData(String),

    #[error(
        "{resource} dataset revision {revision:?} is {completeness:?}; a complete audited dataset is required"
    )]
    DatasetIncomplete {
        resource: &'static str,
        revision: String,
        completeness: DatasetCompleteness,
    },

    #[error("source hash mismatch: metadata says {expected}, calculated {actual}")]
    SourceHashMismatch { expected: String, actual: String },

    #[error("licence audit rejected the source: {0}")]
    LicenceAudit(String),

    #[error("CC-CEDICT parse error on line {line}: {message}")]
    CedictParse { line: usize, message: String },

    #[error("CSV/TSV parse error: {0}")]
    Csv(#[from] csv::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, HskControlError>;
