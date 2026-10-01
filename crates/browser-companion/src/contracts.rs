use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Exact build affinity shared by the extension, native host, and daemon.
///
/// This is deliberately not a negotiable protocol version. A mismatched build
/// must restart the native/daemon pair that shipped with the extension.
pub const BUILD_FINGERPRINT: &str = "hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10";
pub const HSK_STANDARD: &str = "2.0";
pub const SOURCE_LANGUAGE: &str = "en";
pub const TARGET_LANGUAGE: &str = "zh-CN";
pub const MAX_VISIBLE_RECTS: usize = 64;
pub const MAX_VISIBLE_BLOCK_IDS: usize = 64;
pub const MAX_CHAPTER_PAGE_ORDER: usize = 100_000;
pub const MAX_DOCUMENT_BYTES: usize = 1024 * 1024;
pub const MAX_DOCUMENT_BLOCKS: usize = 2_000;
pub const MAX_DOCUMENT_BLOCK_BYTES: usize = 16 * 1024;
const MAX_ITEM_ID_CHARS: usize = 512;
const MAX_OUTPUT_TEXT_CHARS: usize = 16 * 1024;
const MAX_PINYIN_CHARS: usize = 32 * 1024;
const MAX_MESSAGE_CHARS: usize = 2_048;
const MAX_POLYGON_POINTS: usize = 2_048;
const MAX_HSK_ITEMS: usize = 512;
const MAX_DEFINITIONS: usize = 32;
const MAX_LOOKUP_STRING_CHARS: usize = 8_192;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{path}: {message}")]
pub struct ContractError {
    pub path: String,
    pub message: String,
}

impl ContractError {
    fn at(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

pub trait Validate {
    fn validate(&self) -> Result<(), ContractError>;
}

fn require_build_fingerprint(path: &str, value: &str) -> Result<(), ContractError> {
    if value == BUILD_FINGERPRINT {
        Ok(())
    } else {
        Err(ContractError::at(
            path,
            format!("expected exact build fingerprint {BUILD_FINGERPRINT}"),
        ))
    }
}

fn require_nonempty(path: &str, value: &str) -> Result<(), ContractError> {
    if value.trim().is_empty() {
        Err(ContractError::at(path, "must not be empty"))
    } else {
        Ok(())
    }
}

fn require_nonempty_at_most(path: &str, value: &str, maximum: usize) -> Result<(), ContractError> {
    require_nonempty(path, value)?;
    if value.chars().count() > maximum {
        Err(ContractError::at(
            path,
            format!("must contain at most {maximum} characters"),
        ))
    } else {
        Ok(())
    }
}

fn require_at_most(path: &str, value: &str, maximum: usize) -> Result<(), ContractError> {
    if value.chars().count() > maximum {
        Err(ContractError::at(
            path,
            format!("must contain at most {maximum} characters"),
        ))
    } else {
        Ok(())
    }
}

fn require_nonempty_utf8_at_most(
    path: &str,
    value: &str,
    maximum: usize,
) -> Result<(), ContractError> {
    require_nonempty(path, value)?;
    if value.len() > maximum {
        Err(ContractError::at(
            path,
            format!("must contain at most {maximum} UTF-8 bytes"),
        ))
    } else {
        Ok(())
    }
}

fn require_unit(path: &str, value: f32) -> Result<(), ContractError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ContractError::at(
            path,
            "must be a finite number from 0 to 1",
        ))
    }
}

fn require_sha256(path: &str, value: &str) -> Result<(), ContractError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ContractError::at(
            path,
            "must be a 64-character hexadecimal SHA-256",
        ))
    }
}

fn require_polygon(path: &str, points: &[Point]) -> Result<(), ContractError> {
    if points.len() < 3 || points.len() > MAX_POLYGON_POINTS {
        return Err(ContractError::at(
            path,
            format!("must contain between 3 and {MAX_POLYGON_POINTS} points"),
        ));
    }
    for (index, point) in points.iter().enumerate() {
        point.validate_at(&format!("{path}[{index}]"))?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum HskLevel {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
}

impl TryFrom<u8> for HskLevel {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            5 => Ok(Self::Five),
            6 => Ok(Self::Six),
            _ => Err(format!("HSK level must be from 1 through 6, got {value}")),
        }
    }
}

impl From<HskLevel> for u8 {
    fn from(value: HskLevel) -> Self {
        value as u8
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NativeHandshakeRequest {
    #[serde(rename = "type")]
    pub message_type: NativeRequestType,
    pub build_fingerprint: String,
    pub extension_version: String,
    pub extension_origin: String,
}

impl Validate for NativeHandshakeRequest {
    fn validate(&self) -> Result<(), ContractError> {
        require_build_fingerprint("buildFingerprint", &self.build_fingerprint)?;
        require_nonempty_at_most("extensionVersion", &self.extension_version, 128)?;
        if !self.extension_origin.starts_with("moz-extension://")
            || self.extension_origin.len() <= "moz-extension://".len()
        {
            return Err(ContractError::at(
                "extensionOrigin",
                "must be a non-empty moz-extension origin",
            ));
        }
        if self.extension_origin.ends_with('/') {
            return Err(ContractError::at(
                "extensionOrigin",
                "must not contain a trailing slash",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeRequestType {
    StartOrDiscoverDaemon,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NativeReadyResponse {
    #[serde(rename = "type")]
    pub message_type: NativeReadyType,
    pub build_fingerprint: String,
    pub engine_version: String,
    pub port: u16,
    pub token: String,
    pub session_expires_at_unix_ms: u64,
    pub capabilities: BrowserCapabilities,
}

impl Validate for NativeReadyResponse {
    fn validate(&self) -> Result<(), ContractError> {
        require_build_fingerprint("buildFingerprint", &self.build_fingerprint)?;
        require_nonempty_at_most("engineVersion", &self.engine_version, 128)?;
        if self.port == 0 {
            return Err(ContractError::at("port", "must be non-zero"));
        }
        if self.token.len() < 43
            || !self
                .token
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        {
            return Err(ContractError::at(
                "token",
                "must be a base64url-encoded 256-bit or stronger token",
            ));
        }
        if self.session_expires_at_unix_ms == 0 {
            return Err(ContractError::at(
                "sessionExpiresAtUnixMs",
                "must be a non-zero Unix timestamp",
            ));
        }
        self.capabilities.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeReadyType {
    Ready,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserCapabilities {
    pub source_languages: Vec<String>,
    pub target_languages: Vec<String>,
    pub hsk_levels: Vec<HskLevel>,
    pub models_ready: bool,
}

impl Validate for BrowserCapabilities {
    fn validate(&self) -> Result<(), ContractError> {
        if self.source_languages != [SOURCE_LANGUAGE] {
            return Err(ContractError::at(
                "capabilities.sourceLanguages",
                "this build supports English only",
            ));
        }
        if self.target_languages != [TARGET_LANGUAGE] {
            return Err(ContractError::at(
                "capabilities.targetLanguages",
                "this build supports Simplified Chinese only",
            ));
        }
        let expected = [
            HskLevel::One,
            HskLevel::Two,
            HskLevel::Three,
            HskLevel::Four,
            HskLevel::Five,
            HskLevel::Six,
        ];
        if self.hsk_levels.as_slice() != expected {
            return Err(ContractError::at(
                "capabilities.hskLevels",
                "must contain levels 1 through 6 in order",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResourceIdentity {
    pub id: String,
    pub repository: String,
    pub repository_revision: String,
    pub filename: String,
    pub bytes: u64,
    pub sha256: String,
}

impl ResourceIdentity {
    fn validate_at(&self, index: usize) -> Result<(), ContractError> {
        let path = format!("resourceIdentities[{index}]");
        if self.id.is_empty()
            || self.id.len() > 128
            || self.id.split('-').any(|part| {
                part.is_empty()
                    || !part
                        .bytes()
                        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            })
        {
            return Err(ContractError::at(
                format!("{path}.id"),
                "must be a lowercase kebab-case identifier",
            ));
        }
        let mut repository_parts = self.repository.split('/');
        let valid_repository_part = |part: &str| {
            part.bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        };
        if self.repository.len() > 256
            || !repository_parts.next().is_some_and(valid_repository_part)
            || !repository_parts.next().is_some_and(valid_repository_part)
            || repository_parts.next().is_some()
        {
            return Err(ContractError::at(
                format!("{path}.repository"),
                "must contain exactly one owner/name repository",
            ));
        }
        if self.repository_revision.len() != 40
            || !self
                .repository_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(ContractError::at(
                format!("{path}.repositoryRevision"),
                "must be a lowercase 40-character hexadecimal revision",
            ));
        }
        if self.filename.is_empty()
            || self.filename.len() > 255
            || matches!(self.filename.as_str(), "." | "..")
            || !self
                .filename
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            || !self
                .filename
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(ContractError::at(
                format!("{path}.filename"),
                "must be a safe ASCII filename",
            ));
        }
        if self.bytes == 0 || self.bytes > 9_007_199_254_740_991 {
            return Err(ContractError::at(
                format!("{path}.bytes"),
                "must be a positive JavaScript-safe integer",
            ));
        }
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        {
            return Err(ContractError::at(
                format!("{path}.sha256"),
                "must be a lowercase 64-character hexadecimal SHA-256",
            ));
        }
        Ok(())
    }
}

pub(crate) fn validate_resource_identities(
    resource_identities: &[ResourceIdentity],
) -> Result<(), ContractError> {
    if resource_identities.is_empty() {
        return Err(ContractError::at("resourceIdentities", "must not be empty"));
    }
    if resource_identities.len() > 256 {
        return Err(ContractError::at(
            "resourceIdentities",
            "must contain at most 256 entries",
        ));
    }
    let mut previous_id: Option<&str> = None;
    for (index, identity) in resource_identities.iter().enumerate() {
        identity.validate_at(index)?;
        if previous_id.is_some_and(|previous| previous >= identity.id.as_str()) {
            return Err(ContractError::at(
                format!("resourceIdentities[{index}].id"),
                "must be unique and sorted in ascending ordinal order",
            ));
        }
        previous_id = Some(identity.id.as_str());
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HealthResponse {
    pub build_fingerprint: String,
    pub engine_version: String,
    pub status: HealthStatus,
    pub setup_state: BrowserSetupState,
    pub resource_identities: Vec<ResourceIdentity>,
}

impl Validate for HealthResponse {
    fn validate(&self) -> Result<(), ContractError> {
        require_build_fingerprint("buildFingerprint", &self.build_fingerprint)?;
        require_nonempty_at_most("engineVersion", &self.engine_version, 128)?;
        validate_resource_identities(&self.resource_identities)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Ready,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NormalizedRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// The browser's immutable acquisition contract records which rendered
/// surface produced the submitted pixels.  The pipeline uses this only for
/// surface provenance and safe policy decisions; OCR and translation always
/// consume the captured raster itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BrowserSurfaceKind {
    Image,
    Background,
    Canvas,
    Webgl,
    Frame,
}

impl NormalizedRect {
    pub fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        for (field, value) in [
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ] {
            if !value.is_finite() {
                return Err(ContractError::at(
                    format!("{path}.{field}"),
                    "must be finite",
                ));
            }
        }
        if self.x < 0.0
            || self.y < 0.0
            || self.width <= 0.0
            || self.height <= 0.0
            || self.x + self.width > 1.0 + f32::EPSILON
            || self.y + self.height > 1.0 + f32::EPSILON
        {
            return Err(ContractError::at(
                path,
                "must be a positive rectangle contained in normalized image space",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageJobRequest {
    pub build_fingerprint: String,
    pub client_request_id: String,
    pub retry_item_ids: Vec<String>,
    pub client_image_id: String,
    pub source_sha256: String,
    pub source_mime_type: String,
    pub natural_width: u32,
    pub natural_height: u32,
    pub page_session_id: String,
    pub source_index: u32,
    pub chapter_source_order: Vec<u32>,
    pub surface_kind: BrowserSurfaceKind,
    pub reading_direction: ReadingDirection,
    pub settings: BrowserJobSettings,
    pub visible_rects: Vec<NormalizedRect>,
}

impl Validate for ImageJobRequest {
    fn validate(&self) -> Result<(), ContractError> {
        validate_creation_identity(&self.client_request_id, &self.retry_item_ids)?;
        require_build_fingerprint("buildFingerprint", &self.build_fingerprint)?;
        validate_job_fields(self)?;
        if self.chapter_source_order.is_empty() {
            return Err(ContractError::at(
                "chapterSourceOrder",
                "must contain at least the submitted source index",
            ));
        }
        if self.chapter_source_order.len() > MAX_CHAPTER_PAGE_ORDER {
            return Err(ContractError::at(
                "chapterSourceOrder",
                format!("must contain at most {MAX_CHAPTER_PAGE_ORDER} source indexes"),
            ));
        }
        if !self.chapter_source_order.contains(&self.source_index)
            || self
                .chapter_source_order
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != self.chapter_source_order.len()
        {
            return Err(ContractError::at(
                "chapterSourceOrder",
                "must contain unique source identities in DOM order and include sourceIndex",
            ));
        }
        if self.visible_rects.len() > MAX_VISIBLE_RECTS {
            return Err(ContractError::at(
                "visibleRects",
                format!("must contain at most {MAX_VISIBLE_RECTS} rectangles"),
            ));
        }
        for (index, rect) in self.visible_rects.iter().enumerate() {
            rect.validate_at(&format!("visibleRects[{index}]"))?;
        }
        Ok(())
    }
}

fn validate_job_fields(request: &ImageJobRequest) -> Result<(), ContractError> {
    require_nonempty("clientImageId", &request.client_image_id)?;
    require_sha256("sourceSha256", &request.source_sha256)?;
    if !matches!(
        request.source_mime_type.as_str(),
        "image/png" | "image/jpeg" | "image/webp" | "image/gif"
    ) {
        return Err(ContractError::at(
            "sourceMimeType",
            "must be a supported raster image MIME type",
        ));
    }
    if request.natural_width == 0 || request.natural_height == 0 {
        return Err(ContractError::at(
            "naturalWidth",
            "decoded image dimensions must be non-zero",
        ));
    }
    require_nonempty("pageSessionId", &request.page_session_id)?;
    request.settings.validate()?;
    Ok(())
}

/// Validated input passed from the HTTP boundary into the image pipeline.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ImagePipelineInput {
    pub retry_item_ids: Vec<String>,
    pub surrounding_context: Vec<crate::chapter_session::ChapterContextUnit>,
    pub source_sha256: String,
    pub source_mime_type: String,
    pub natural_width: u32,
    pub natural_height: u32,
    pub page_session_id: String,
    pub source_index: u32,
    pub chapter_source_order: Vec<u32>,
    pub surface_kind: BrowserSurfaceKind,
    pub reading_direction: ReadingDirection,
    pub settings: BrowserJobSettings,
}

impl ImageJobRequest {
    pub(crate) fn pipeline_input(&self) -> ImagePipelineInput {
        ImagePipelineInput {
            retry_item_ids: self.retry_item_ids.clone(),
            surrounding_context: Vec::new(),
            source_sha256: self.source_sha256.clone(),
            source_mime_type: self.source_mime_type.clone(),
            natural_width: self.natural_width,
            natural_height: self.natural_height,
            page_session_id: self.page_session_id.clone(),
            source_index: self.source_index,
            chapter_source_order: self.chapter_source_order.clone(),
            surface_kind: self.surface_kind,
            reading_direction: self.reading_direction,
            settings: self.settings.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserJobCreated {
    pub build_fingerprint: String,
    pub job_id: String,
}

impl Validate for BrowserJobCreated {
    fn validate(&self) -> Result<(), ContractError> {
        require_build_fingerprint("buildFingerprint", &self.build_fingerprint)?;
        require_nonempty("jobId", &self.job_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserJobSettings {
    pub source_language: String,
    pub target_language: String,
    pub hsk_standard: String,
    pub hsk_level: HskLevel,
    pub learning_mode: LearningMode,
}

impl Validate for BrowserJobSettings {
    fn validate(&self) -> Result<(), ContractError> {
        if self.source_language != SOURCE_LANGUAGE {
            return Err(ContractError::at(
                "settings.sourceLanguage",
                "this build supports English only",
            ));
        }
        if self.target_language != TARGET_LANGUAGE {
            return Err(ContractError::at(
                "settings.targetLanguage",
                "this build supports Simplified Chinese only",
            ));
        }
        if self.hsk_standard != HSK_STANDARD {
            return Err(ContractError::at(
                "settings.hskStandard",
                "this build supports HSK 2.0 only",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReadingDirection {
    Ltr,
    Rtl,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LearningMode {
    #[default]
    Natural,
    Strict,
}

/// Generic ordered input understood by the shared language service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceSpanKind {
    Prose,
    Heading,
    Dialogue,
    Caption,
    Thought,
    Sfx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceProvenance {
    Dom,
    Ocr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChapterKind {
    Image,
    Document,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WarmupRequest {
    pub kind: ChapterKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DocumentSourceBlock {
    pub parent_block_id: String,
    pub sub_item_order: u32,
    pub item_id: String,
    pub source_index: u32,
    pub item_order: u32,
    pub kind: SourceSpanKind,
    pub provenance: SourceProvenance,
    pub text: String,
}

impl DocumentSourceBlock {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_nonempty_at_most(&format!("{path}.parentBlockId"), &self.parent_block_id, 256)?;
        require_nonempty_at_most(&format!("{path}.itemId"), &self.item_id, 256)?;
        if self.provenance != SourceProvenance::Dom {
            return Err(ContractError::at(
                format!("{path}.provenance"),
                "document blocks must have DOM provenance",
            ));
        }
        if self.text.trim().is_empty() {
            return Err(ContractError::at(
                format!("{path}.text"),
                "must not be empty",
            ));
        }
        if self.text.as_bytes().len() > MAX_DOCUMENT_BLOCK_BYTES {
            return Err(ContractError::at(
                format!("{path}.text"),
                format!("must contain at most {MAX_DOCUMENT_BLOCK_BYTES} UTF-8 bytes"),
            ));
        }
        if self.text.contains('\r') || self.text.contains('\0') || self.text.trim() != self.text {
            return Err(ContractError::at(
                format!("{path}.text"),
                "must be deterministically normalized, trimmed text using LF line breaks",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DocumentJobRequest {
    pub build_fingerprint: String,
    pub client_request_id: String,
    pub retry_item_ids: Vec<String>,
    pub focus: FocusUpdateRequest,
    pub page_session_id: String,
    pub source_sha256: String,
    pub settings: BrowserJobSettings,
    pub blocks: Vec<DocumentSourceBlock>,
}

impl Validate for DocumentJobRequest {
    fn validate(&self) -> Result<(), ContractError> {
        validate_creation_identity(&self.client_request_id, &self.retry_item_ids)?;
        require_build_fingerprint("buildFingerprint", &self.build_fingerprint)?;
        require_nonempty_at_most("pageSessionId", &self.page_session_id, 256)?;
        require_sha256("sourceSha256", &self.source_sha256)?;
        self.settings.validate()?;
        if self.blocks.is_empty() || self.blocks.len() > MAX_DOCUMENT_BLOCKS {
            return Err(ContractError::at(
                "blocks",
                format!("must contain between 1 and {MAX_DOCUMENT_BLOCKS} blocks"),
            ));
        }
        let mut ids = HashSet::with_capacity(self.blocks.len());
        let mut parent_orders = std::collections::HashMap::new();
        let mut previous = None;
        for (index, block) in self.blocks.iter().enumerate() {
            block.validate_at(&format!("blocks[{index}]"))?;
            let expected_order = parent_orders
                .entry(block.parent_block_id.as_str())
                .or_insert(0u32);
            if block.sub_item_order != *expected_order {
                return Err(ContractError::at(
                    format!("blocks[{index}].subItemOrder"),
                    "must start at zero and increase consecutively within its parent",
                ));
            }
            *expected_order += 1;
            if !ids.insert(block.item_id.as_str()) {
                return Err(ContractError::at(
                    format!("blocks[{index}].itemId"),
                    "must be unique",
                ));
            }
            let position = (block.source_index, block.item_order);
            if previous.is_some_and(|value| value >= position) {
                return Err(ContractError::at(
                    format!("blocks[{index}]"),
                    "blocks must be strictly ordered by sourceIndex and itemOrder",
                ));
            }
            previous = Some(position);
        }
        if self
            .retry_item_ids
            .iter()
            .any(|id| !ids.contains(id.as_str()))
        {
            return Err(ContractError::at(
                "retryItemIds",
                "must identify submitted source blocks",
            ));
        }
        self.focus.validate()?;
        match &self.focus {
            FocusUpdateRequest::Document {
                visible_block_ids, ..
            } if visible_block_ids.iter().all(|id| ids.contains(id.as_str())) => {}
            _ => {
                return Err(ContractError::at(
                    "focus",
                    "must identify submitted document blocks",
                ));
            }
        }
        let serialized_bytes = serde_json::to_vec(self)
            .map_err(|_| ContractError::at("$", "document request could not be serialized"))?
            .len();
        if serialized_bytes > MAX_DOCUMENT_BYTES {
            return Err(ContractError::at(
                "$",
                format!("must contain at most {MAX_DOCUMENT_BYTES} UTF-8 bytes"),
            ));
        }
        let canonical = canonical_document_sha256(&self.blocks);
        if canonical != self.source_sha256 {
            return Err(ContractError::at(
                "sourceSha256",
                "does not match the native canonical document hash",
            ));
        }
        Ok(())
    }
}

fn validate_creation_identity(id: &str, retries: &[String]) -> Result<(), ContractError> {
    require_nonempty_at_most("clientRequestId", id, 128)?;
    if retries.len() > MAX_DOCUMENT_BLOCKS {
        return Err(ContractError::at("retryItemIds", "too many retry items"));
    }
    let mut unique = HashSet::with_capacity(retries.len());
    for id in retries {
        require_nonempty_at_most("retryItemIds", id, 256)?;
        if !unique.insert(id) {
            return Err(ContractError::at("retryItemIds", "must be unique"));
        }
    }
    Ok(())
}

/// Hashes the complete ordered text snapshot. The record and unit separators
/// are literal bytes in the unversioned contract.
#[must_use]
pub fn canonical_document_sha256(blocks: &[DocumentSourceBlock]) -> String {
    let mut hasher = Sha256::new();
    for block in blocks {
        let record = format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1e}",
            block.item_id,
            block.parent_block_id,
            block.sub_item_order,
            block.source_index,
            block.item_order,
            match block.kind {
                SourceSpanKind::Prose => "prose",
                SourceSpanKind::Heading => "heading",
                SourceSpanKind::Dialogue => "dialogue",
                SourceSpanKind::Caption => "caption",
                SourceSpanKind::Thought => "thought",
                SourceSpanKind::Sfx => "sfx",
            },
            match block.provenance {
                SourceProvenance::Dom => "dom",
                SourceProvenance::Ocr => "ocr",
            },
            block.text,
        );
        hasher.update(record.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_unit(&format!("{path}.x"), self.x)?;
        require_unit(&format!("{path}.y"), self.y)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserTextColorBand {
    pub position: f32,
    pub foreground: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outline_color: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserTextStyle {
    pub font_id: String,
    pub category: FontCategory,
    pub foreground: String,
    pub weight: u16,
    pub italic_degrees: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outline_color: Option<String>,
    pub outline_width_ratio: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_color: Option<String>,
    pub shadow_x_ratio: f32,
    pub shadow_y_ratio: f32,
    pub alignment: TextAlignment,
    pub writing_mode: WritingMode,
    pub line_height: f32,
    pub letter_spacing_em: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub color_bands: Vec<BrowserTextColorBand>,
}

impl BrowserTextColorBand {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_unit(&format!("{path}.position"), self.position)?;
        if !is_css_color(&self.foreground) {
            return Err(ContractError::at(
                format!("{path}.foreground"),
                "must be a hexadecimal CSS color",
            ));
        }
        if let Some(value) = &self.outline_color
            && !is_css_color(value)
        {
            return Err(ContractError::at(
                format!("{path}.outlineColor"),
                "must be a hexadecimal CSS color",
            ));
        }
        Ok(())
    }
}

impl BrowserTextStyle {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_nonempty(&format!("{path}.fontId"), &self.font_id)?;
        if !is_css_color(&self.foreground) {
            return Err(ContractError::at(
                format!("{path}.foreground"),
                "must be a hexadecimal CSS color",
            ));
        }
        if let Some(value) = &self.outline_color
            && !is_css_color(value)
        {
            return Err(ContractError::at(
                format!("{path}.outlineColor"),
                "must be a hexadecimal CSS color",
            ));
        }
        if let Some(value) = &self.shadow_color
            && !is_css_color(value)
        {
            return Err(ContractError::at(
                format!("{path}.shadowColor"),
                "must be a hexadecimal CSS color",
            ));
        }
        if !(1..=1000).contains(&self.weight) {
            return Err(ContractError::at(
                format!("{path}.weight"),
                "must be from 1 through 1000",
            ));
        }
        for (field, value) in [
            ("italicDegrees", self.italic_degrees),
            ("outlineWidthRatio", self.outline_width_ratio),
            ("shadowXRatio", self.shadow_x_ratio),
            ("shadowYRatio", self.shadow_y_ratio),
            ("lineHeight", self.line_height),
            ("letterSpacingEm", self.letter_spacing_em),
        ] {
            if !value.is_finite() {
                return Err(ContractError::at(
                    format!("{path}.{field}"),
                    "must be finite",
                ));
            }
        }
        if self.outline_width_ratio < 0.0 || self.line_height <= 0.0 {
            return Err(ContractError::at(
                format!("{path}.lineHeight"),
                "ratios must be non-negative and line height must be positive",
            ));
        }
        let mut previous_position = None;
        for (index, band) in self.color_bands.iter().enumerate() {
            band.validate_at(&format!("{path}.colorBands[{index}]"))?;
            if previous_position.is_some_and(|previous| band.position <= previous) {
                return Err(ContractError::at(
                    format!("{path}.colorBands[{index}].position"),
                    "must be strictly greater than the preceding color band",
                ));
            }
            previous_position = Some(band.position);
        }
        Ok(())
    }
}

fn is_css_color(value: &str) -> bool {
    matches!(value.len(), 4 | 5 | 7 | 9)
        && value.starts_with('#')
        && value[1..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FontCategory {
    Sans,
    Serif,
    Handwritten,
    Display,
    Brush,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextAlignment {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WritingMode {
    #[serde(rename = "horizontal-tb")]
    HorizontalTb,
    #[serde(rename = "vertical-rl")]
    VerticalRl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserTextLayout {
    pub suggested_lines: Vec<String>,
    pub font_size_to_image_width: f32,
    pub safe_polygon: Vec<Point>,
}

impl BrowserTextLayout {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        if !self.font_size_to_image_width.is_finite() || self.font_size_to_image_width <= 0.0 {
            return Err(ContractError::at(
                format!("{path}.fontSizeToImageWidth"),
                "must be a positive finite ratio",
            ));
        }
        require_polygon(&format!("{path}.safePolygon"), &self.safe_polygon)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "lowercase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum FocusUpdateRequest {
    Image {
        visible_rects: Vec<NormalizedRect>,
        active: bool,
    },
    Document {
        visible_block_ids: Vec<String>,
        active: bool,
    },
}

impl FocusUpdateRequest {
    #[must_use]
    pub fn active(&self) -> bool {
        match self {
            Self::Image { active, .. } | Self::Document { active, .. } => *active,
        }
    }
}

impl Validate for FocusUpdateRequest {
    fn validate(&self) -> Result<(), ContractError> {
        match self {
            Self::Image { visible_rects, .. } => {
                if visible_rects.len() > MAX_VISIBLE_RECTS {
                    return Err(ContractError::at(
                        "visibleRects",
                        format!("must contain at most {MAX_VISIBLE_RECTS} rectangles"),
                    ));
                }
                for (index, rect) in visible_rects.iter().enumerate() {
                    rect.validate_at(&format!("visibleRects[{index}]"))?;
                }
            }
            Self::Document {
                visible_block_ids, ..
            } => {
                if visible_block_ids.len() > MAX_VISIBLE_BLOCK_IDS {
                    return Err(ContractError::at(
                        "visibleBlockIds",
                        format!("must contain at most {MAX_VISIBLE_BLOCK_IDS} IDs"),
                    ));
                }
                let mut ids = HashSet::with_capacity(visible_block_ids.len());
                for (index, id) in visible_block_ids.iter().enumerate() {
                    require_nonempty_at_most(&format!("visibleBlockIds[{index}]"), id, 256)?;
                    if !ids.insert(id.as_str()) {
                        return Err(ContractError::at(
                            format!("visibleBlockIds[{index}]"),
                            "must be unique",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatchMimeType {
    #[serde(rename = "image/png")]
    Png,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegionPatch {
    pub blob_id: String,
    pub mime_type: PatchMimeType,
    pub rect: NormalizedRect,
}

impl RegionPatch {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_nonempty(&format!("{path}.blobId"), &self.blob_id)?;
        self.rect.validate_at(&format!("{path}.rect"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HskRepairState {
    NotNeeded,
    Pending,
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TranslatedHskStatus {
    pub requested_level: HskLevel,
    pub learning_mode: LearningMode,
    pub strictly_valid: bool,
    pub level_coverage: f32,
    pub above_level_tokens: Vec<String>,
    pub teaching_terms: Vec<TeachingTerm>,
    pub repair_state: HskRepairState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeachingTerm {
    pub text: String,
    pub start_char: usize,
    pub end_char: usize,
    pub pinyin: String,
    pub definitions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_level: Option<HskLevel>,
    pub reason: TeachingTermReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TeachingTermReason {
    AboveLevel,
    OutsideList,
}

impl TranslatedHskStatus {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_unit(&format!("{path}.levelCoverage"), self.level_coverage)?;
        if self.above_level_tokens.len() > MAX_HSK_ITEMS {
            return Err(ContractError::at(
                format!("{path}.aboveLevelTokens"),
                format!("must contain at most {MAX_HSK_ITEMS} items"),
            ));
        }
        let mut tokens = HashSet::with_capacity(self.above_level_tokens.len());
        for (index, token) in self.above_level_tokens.iter().enumerate() {
            require_nonempty_at_most(&format!("{path}.aboveLevelTokens[{index}]"), token, 256)?;
            if !tokens.insert(token.as_str()) {
                return Err(ContractError::at(
                    format!("{path}.aboveLevelTokens[{index}]"),
                    "duplicate above-level token",
                ));
            }
        }
        if self.teaching_terms.len() > MAX_HSK_ITEMS {
            return Err(ContractError::at(
                format!("{path}.teachingTerms"),
                format!("must contain at most {MAX_HSK_ITEMS} items"),
            ));
        }
        if self.strictly_valid
            && (!self.above_level_tokens.is_empty() || !self.teaching_terms.is_empty())
        {
            return Err(ContractError::at(
                format!("{path}.strictlyValid"),
                "strictly valid text cannot retain teaching exceptions",
            ));
        }
        let mut previous_end = 0;
        for (index, term) in self.teaching_terms.iter().enumerate() {
            let term_path = format!("{path}.teachingTerms[{index}]");
            require_nonempty_at_most(&format!("{term_path}.text"), &term.text, 256)?;
            require_nonempty_at_most(&format!("{term_path}.pinyin"), &term.pinyin, 512)?;
            if term.definitions.is_empty()
                || term.definitions.len() > MAX_DEFINITIONS
                || term.definitions.iter().any(|definition| {
                    definition.trim().is_empty() || definition.chars().count() > 2_048
                })
            {
                return Err(ContractError::at(
                    format!("{term_path}.definitions"),
                    "must contain non-empty definitions",
                ));
            }
            if term.start_char >= term.end_char {
                return Err(ContractError::at(
                    format!("{term_path}.endChar"),
                    "must be greater than startChar",
                ));
            }
            if index > 0 && term.start_char < previous_end {
                return Err(ContractError::at(
                    format!("{term_path}.startChar"),
                    "teaching terms must be ordered and non-overlapping",
                ));
            }
            previous_end = term.end_char;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TranslatedText {
    pub termination: koharu_llm::GenerationTermination,
    pub protected_names: Vec<koharu_app::llm::ProtectedName>,
    pub source_text: String,
    pub base_chinese: String,
    pub displayed_chinese: String,
    pub pinyin: String,
    pub hsk: TranslatedHskStatus,
}

impl TranslatedText {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        if self.termination != koharu_llm::GenerationTermination::Stop
            || self.protected_names.len() > 32
            || self
                .protected_names
                .iter()
                .any(|name| !name.is_anchored(&self.source_text, &self.displayed_chinese))
        {
            return Err(ContractError::at(
                path,
                "translation must be complete with source-anchored names",
            ));
        }
        require_nonempty_at_most(
            &format!("{path}.sourceText"),
            &self.source_text,
            MAX_OUTPUT_TEXT_CHARS,
        )?;
        require_nonempty_at_most(
            &format!("{path}.baseChinese"),
            &self.base_chinese,
            MAX_OUTPUT_TEXT_CHARS,
        )?;
        require_nonempty_at_most(
            &format!("{path}.displayedChinese"),
            &self.displayed_chinese,
            MAX_OUTPUT_TEXT_CHARS,
        )?;
        require_nonempty_at_most(&format!("{path}.pinyin"), &self.pinyin, MAX_PINYIN_CHARS)?;
        self.hsk.validate_at(&format!("{path}.hsk"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DocumentBlockReady {
    pub parent_block_id: String,
    pub sub_item_order: u32,
    pub item_id: String,
    pub source_index: u32,
    pub item_order: u32,
    pub kind: SourceSpanKind,
    pub text: TranslatedText,
}

impl DocumentBlockReady {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_nonempty_at_most(&format!("{path}.itemId"), &self.item_id, MAX_ITEM_ID_CHARS)?;
        self.text.validate_at(&format!("{path}.text"))?;
        if matches!(
            self.text.hsk.repair_state,
            HskRepairState::Pending | HskRepairState::Rejected
        ) {
            return Err(ContractError::at(
                format!("{path}.text.hsk.repairState"),
                "documentBlockReady may publish only a terminal translation",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DocumentBlockPreserved {
    pub parent_block_id: String,
    pub sub_item_order: u32,
    pub item_id: String,
    pub source_index: u32,
    pub item_order: u32,
    pub kind: SourceSpanKind,
    pub source_text: String,
    pub reason: String,
}

impl DocumentBlockPreserved {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_nonempty_at_most(&format!("{path}.itemId"), &self.item_id, MAX_ITEM_ID_CHARS)?;
        require_nonempty_utf8_at_most(
            &format!("{path}.sourceText"),
            &self.source_text,
            MAX_DOCUMENT_BLOCK_BYTES,
        )?;
        require_nonempty_at_most(&format!("{path}.reason"), &self.reason, MAX_MESSAGE_CHARS)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageRegionRole {
    Dialogue,
    Narration,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegionConfidenceEvidence {
    pub ocr_consensus: f32,
    pub geometry_coverage: f32,
    pub context_consistency: f32,
    pub cleanup_score: f32,
}

impl RegionConfidenceEvidence {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        for (field, value) in [
            ("ocrConsensus", self.ocr_consensus),
            ("geometryCoverage", self.geometry_coverage),
            ("contextConsistency", self.context_consistency),
            ("cleanupScore", self.cleanup_score),
        ] {
            require_unit(&format!("{path}.{field}"), value)?;
        }
        Ok(())
    }
}

fn polygon_bounds(points: &[Point]) -> Option<(f32, f32, f32, f32)> {
    let first = points.first()?;
    Some(points.iter().skip(1).fold(
        (first.x, first.y, first.x, first.y),
        |(x0, y0, x1, y1), point| {
            (
                x0.min(point.x),
                y0.min(point.y),
                x1.max(point.x),
                y1.max(point.y),
            )
        },
    ))
}

/// Browser-facing image result. The image pipeline keeps its geometric work
/// separate, while all language output is carried by the same payload used by
/// document blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageRegionReady {
    pub item_id: String,
    pub text_polygon: Vec<Point>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bubble_polygon: Option<Vec<Point>>,
    pub patch: RegionPatch,
    pub text: TranslatedText,
    pub provenance: SourceProvenance,
    pub kind: SourceSpanKind,
    pub confidence: f32,
    pub item_order: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_group: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_evidence: Option<RegionConfidenceEvidence>,
    pub style: BrowserTextStyle,
    pub layout: BrowserTextLayout,
}

impl ImageRegionReady {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_nonempty_at_most(&format!("{path}.itemId"), &self.item_id, MAX_ITEM_ID_CHARS)?;
        require_polygon(&format!("{path}.textPolygon"), &self.text_polygon)?;
        if let Some(points) = &self.bubble_polygon {
            require_polygon(&format!("{path}.bubblePolygon"), points)?;
        }
        self.patch.validate_at(&format!("{path}.patch"))?;
        let (text_x0, text_y0, text_x1, text_y1) =
            polygon_bounds(&self.text_polygon).expect("validated polygon is non-empty");
        let patch_x0 = self.patch.rect.x;
        let patch_y0 = self.patch.rect.y;
        let patch_x1 = patch_x0 + self.patch.rect.width;
        let patch_y1 = patch_y0 + self.patch.rect.height;
        if text_x1.min(patch_x1) - text_x0.max(patch_x0) <= f32::EPSILON
            || text_y1.min(patch_y1) - text_y0.max(patch_y0) <= f32::EPSILON
        {
            return Err(ContractError::at(
                format!("{path}.patch.rect"),
                "must overlap the source text polygon",
            ));
        }
        require_unit(&format!("{path}.confidence"), self.confidence)?;
        if self.provenance != SourceProvenance::Ocr {
            return Err(ContractError::at(
                format!("{path}.provenance"),
                "image results must have OCR provenance",
            ));
        }
        self.text.validate_at(&format!("{path}.text"))?;
        if matches!(
            self.text.hsk.repair_state,
            HskRepairState::Pending | HskRepairState::Rejected
        ) {
            return Err(ContractError::at(
                format!("{path}.text.hsk.repairState"),
                "imageRegionReady may publish only a terminal translation",
            ));
        }
        if let Some(evidence) = &self.confidence_evidence {
            evidence.validate_at(&format!("{path}.confidenceEvidence"))?;
        }
        if let Some(context_group) = &self.context_group {
            require_nonempty_at_most(
                &format!("{path}.contextGroup"),
                context_group,
                MAX_LOOKUP_STRING_CHARS,
            )?;
        }
        self.style.validate_at(&format!("{path}.style"))?;
        self.layout.validate_at(&format!("{path}.layout"))
    }
}

impl Validate for ImageRegionReady {
    fn validate(&self) -> Result<(), ContractError> {
        self.validate_at("region")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ImageRegionPreserved {
    pub disposition: PreservationDisposition,
    pub item_id: String,
    pub text_polygon: Vec<Point>,
    pub source_text: String,
    pub confidence: f32,
    pub item_order: u32,
    pub reason: String,
}

impl ImageRegionPreserved {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_nonempty_at_most(&format!("{path}.itemId"), &self.item_id, MAX_ITEM_ID_CHARS)?;
        require_polygon(&format!("{path}.textPolygon"), &self.text_polygon)?;
        require_at_most(
            &format!("{path}.sourceText"),
            &self.source_text,
            MAX_OUTPUT_TEXT_CHARS,
        )?;
        if self.source_text.trim().is_empty()
            && self.disposition != PreservationDisposition::Excluded
        {
            return Err(ContractError::at(
                format!("{path}.sourceText"),
                "may be empty only for artwork-preserved image items",
            ));
        }
        require_unit(&format!("{path}.confidence"), self.confidence)?;
        require_nonempty_at_most(&format!("{path}.reason"), &self.reason, MAX_MESSAGE_CHARS)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PreservationDisposition {
    Excluded,
    Failed,
}

impl Validate for ImageRegionPreserved {
    fn validate(&self) -> Result<(), ContractError> {
        self.validate_at("region")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum JobUpdate {
    Progress {
        sequence: u64,
        stage: BrowserJobStage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stage_progress: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        overall_progress: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        current: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total: Option<u32>,
        message: String,
    },
    ImageRegionReady {
        sequence: u64,
        region: Box<ImageRegionReady>,
    },
    ImageRegionPreserved {
        sequence: u64,
        region: ImageRegionPreserved,
    },
    DocumentBlockReady {
        sequence: u64,
        block: DocumentBlockReady,
    },
    DocumentBlockPreserved {
        sequence: u64,
        block: DocumentBlockPreserved,
    },
    Complete {
        sequence: u64,
        translated_count: u32,
        preserved_count: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    Failed {
        sequence: u64,
        code: String,
        message: String,
        retryable: bool,
    },
    Cancelled {
        sequence: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
}

impl JobUpdate {
    pub fn sequence(&self) -> u64 {
        match self {
            Self::Progress { sequence, .. }
            | Self::ImageRegionReady { sequence, .. }
            | Self::ImageRegionPreserved { sequence, .. }
            | Self::DocumentBlockReady { sequence, .. }
            | Self::DocumentBlockPreserved { sequence, .. }
            | Self::Complete { sequence, .. }
            | Self::Failed { sequence, .. }
            | Self::Cancelled { sequence, .. } => *sequence,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Complete { .. } | Self::Failed { .. } | Self::Cancelled { .. }
        )
    }
}

impl Validate for JobUpdate {
    fn validate(&self) -> Result<(), ContractError> {
        if self.sequence() == 0 {
            return Err(ContractError::at("sequence", "must start at 1"));
        }
        match self {
            Self::Progress {
                stage: _,
                stage_progress,
                overall_progress,
                current,
                total,
                message,
                ..
            } => {
                if let Some(value) = stage_progress {
                    require_unit("stageProgress", *value)?;
                }
                if let Some(value) = overall_progress {
                    require_unit("overallProgress", *value)?;
                }
                if current.is_some() != total.is_some() {
                    return Err(ContractError::at(
                        "current",
                        "current and total must be present together",
                    ));
                }
                if let (Some(current), Some(total)) = (current, total)
                    && (*total == 0 || current > total)
                {
                    return Err(ContractError::at(
                        "current",
                        "must be less than or equal to a non-zero total",
                    ));
                }
                require_nonempty_at_most("message", message, MAX_MESSAGE_CHARS)
            }
            Self::ImageRegionReady { region, .. } => region.validate_at("region"),
            Self::ImageRegionPreserved { region, .. } => region.validate_at("region"),
            Self::DocumentBlockReady { block, .. } => block.validate_at("block"),
            Self::DocumentBlockPreserved { block, .. } => block.validate_at("block"),
            Self::Complete { message, .. } | Self::Cancelled { message, .. } => {
                if let Some(message) = message {
                    require_nonempty_at_most("message", message, MAX_MESSAGE_CHARS)?;
                }
                Ok(())
            }
            Self::Failed { code, message, .. } => {
                require_nonempty_at_most("code", code, 256)?;
                require_nonempty_at_most("message", message, MAX_MESSAGE_CHARS)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JobUpdatesResponse {
    pub job_id: String,
    pub next_sequence: u64,
    pub updates: Vec<JobUpdate>,
}

impl Validate for JobUpdatesResponse {
    fn validate(&self) -> Result<(), ContractError> {
        require_nonempty_at_most("jobId", &self.job_id, MAX_ITEM_ID_CHARS)?;
        if self.updates.len() > 1_024 {
            return Err(ContractError::at(
                "updates",
                "must contain at most 1024 updates",
            ));
        }
        let mut previous = None;
        for (index, update) in self.updates.iter().enumerate() {
            update.validate()?;
            if previous.is_some_and(|value: u64| value.checked_add(1) != Some(update.sequence())) {
                return Err(ContractError::at(
                    "updates",
                    "update sequences must be contiguous",
                ));
            }
            if update.is_terminal() && index + 1 != self.updates.len() {
                return Err(ContractError::at(
                    format!("updates[{index}]"),
                    "terminal updates must be the final update in a batch",
                ));
            }
            previous = Some(update.sequence());
        }
        if let Some(last) = self.updates.last()
            && self.next_sequence != last.sequence()
        {
            return Err(ContractError::at(
                "nextSequence",
                "must equal the last returned update sequence",
            ));
        }
        Ok(())
    }
}

impl JobUpdatesResponse {
    pub fn validate_after(&self, after: u64) -> Result<(), ContractError> {
        self.validate()?;
        if let Some(first) = self.updates.first()
            && first.sequence() != after.saturating_add(1)
        {
            return Err(ContractError::at(
                "updates[0].sequence",
                "must immediately follow the requested acknowledgement sequence",
            ));
        }
        if self.updates.is_empty() && self.next_sequence != after {
            return Err(ContractError::at(
                "nextSequence",
                "must equal the requested acknowledgement when no updates are returned",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserJobStage {
    Warming,
    Queued,
    Registering,
    Decoding,
    Detecting,
    Ocr,
    Inpainting,
    Translating,
    HskValidating,
    Styling,
    Packaging,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserSetupStatus {
    pub state: BrowserSetupState,
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_disk_bytes: Option<u64>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

impl Validate for BrowserSetupStatus {
    fn validate(&self) -> Result<(), ContractError> {
        require_nonempty_at_most("modelId", &self.model_id, 128)?;
        if self.model_id != "qwen3.5-4b" {
            return Err(ContractError::at(
                "modelId",
                "must identify the mandatory qwen3.5-4b model",
            ));
        }
        require_nonempty("message", &self.message)?;
        if self.completed_bytes.is_some() != self.total_bytes.is_some() {
            return Err(ContractError::at(
                "completedBytes",
                "completed and total bytes must be present together",
            ));
        }
        if let (Some(completed), Some(total)) = (self.completed_bytes, self.total_bytes)
            && completed > total
        {
            return Err(ContractError::at(
                "completedBytes",
                "must not exceed total bytes",
            ));
        }
        if self.state == BrowserSetupState::Failed
            && self.error_code.as_deref().is_none_or(str::is_empty)
        {
            return Err(ContractError::at(
                "errorCode",
                "failed setup requires an error code",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserSetupState {
    MissingModels,
    Downloading,
    Verifying,
    Warming,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LookupInteraction {
    Selection,
    Hover,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LookupContext {
    pub source_text: String,
    pub base_chinese: String,
    pub displayed_chinese: String,
    pub proper_names: Vec<hsk_control::ProperName>,
}
impl Validate for LookupContext {
    fn validate(&self) -> Result<(), ContractError> {
        for (path, text) in [
            ("context.sourceText", &self.source_text),
            ("context.baseChinese", &self.base_chinese),
            ("context.displayedChinese", &self.displayed_chinese),
        ] {
            require_at_most(path, text, MAX_DOCUMENT_BLOCK_BYTES)?;
        }
        if self.proper_names.len() > 32 {
            return Err(ContractError::at("context.properNames", "too many names"));
        }
        for name in &self.proper_names {
            require_nonempty_at_most("context.properNames.text", &name.text, 128)?;
            if !self.displayed_chinese.contains(&name.text) {
                return Err(ContractError::at(
                    "context.properNames",
                    "name is absent from displayed text",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LookupRequest {
    pub interaction: LookupInteraction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_offset: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<LookupContext>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}

impl Validate for LookupRequest {
    fn validate(&self) -> Result<(), ContractError> {
        match self.interaction {
            LookupInteraction::Selection => {
                let selected_text = self.selected_text.as_deref().ok_or_else(|| {
                    ContractError::at("selectedText", "selection lookup requires selected text")
                })?;
                if self.character_offset.is_some() {
                    return Err(ContractError::at(
                        "characterOffset",
                        "selection lookup cannot contain a character offset",
                    ));
                }
                require_nonempty("selectedText", selected_text)?;
                if selected_text.chars().count() > 256 {
                    return Err(ContractError::at(
                        "selectedText",
                        "must contain at most 256 characters",
                    ));
                }
            }
            LookupInteraction::Hover => {
                if self.selected_text.is_some() {
                    return Err(ContractError::at(
                        "selectedText",
                        "hover lookup cannot contain selected text",
                    ));
                }
                if self.character_offset.is_none() {
                    return Err(ContractError::at(
                        "characterOffset",
                        "hover lookup requires a character offset",
                    ));
                }
            }
        }
        if let Some(item_id) = &self.item_id {
            require_nonempty_at_most("itemId", item_id, 256)?;
        }
        if let Some(context) = &self.context {
            context.validate()?;
        }
        if self.item_id.is_some() != self.context.is_some() {
            return Err(ContractError::at(
                "context",
                "itemId and context must be paired",
            ));
        }
        if let (Some(offset), Some(context)) = (self.character_offset, &self.context) {
            if offset as usize >= context.displayed_chinese.chars().count() {
                return Err(ContractError::at(
                    "characterOffset",
                    "must identify a character in the item context",
                ));
            }
        }
        if self.interaction == LookupInteraction::Hover
            && (self.context.is_none() || self.item_id.is_none())
        {
            return Err(ContractError::at(
                "context",
                "hover lookup requires bounded item context",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LookupResult {
    pub selected_text: String,
    pub tokens: Vec<LookupToken>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<LookupItem>,
}

impl Validate for LookupResult {
    fn validate(&self) -> Result<(), ContractError> {
        require_nonempty_at_most("selectedText", &self.selected_text, MAX_LOOKUP_STRING_CHARS)?;
        if self.tokens.len() > 512 {
            return Err(ContractError::at(
                "tokens",
                "must contain at most 512 tokens",
            ));
        }
        for (index, token) in self.tokens.iter().enumerate() {
            require_nonempty_at_most(
                &format!("tokens[{index}].simplified"),
                &token.simplified,
                MAX_LOOKUP_STRING_CHARS,
            )?;
            if !token.proper_name {
                require_nonempty_at_most(
                    &format!("tokens[{index}].pinyin"),
                    &token.pinyin,
                    MAX_LOOKUP_STRING_CHARS,
                )?;
            } else {
                require_at_most(
                    &format!("tokens[{index}].pinyin"),
                    &token.pinyin,
                    MAX_LOOKUP_STRING_CHARS,
                )?;
            }
            if token.definitions.len() > MAX_DEFINITIONS
                || token
                    .definitions
                    .iter()
                    .any(|item| item.trim().is_empty() || item.chars().count() > MAX_MESSAGE_CHARS)
            {
                return Err(ContractError::at(
                    format!("tokens[{index}].definitions"),
                    "definitions must not contain empty values",
                ));
            }
        }
        if let Some(item) = &self.item {
            require_at_most(
                "item.displayedChinese",
                &item.displayed_chinese,
                MAX_OUTPUT_TEXT_CHARS,
            )?;
            require_at_most(
                "item.baseChinese",
                &item.base_chinese,
                MAX_OUTPUT_TEXT_CHARS,
            )?;
            require_at_most("item.sourceText", &item.source_text, MAX_OUTPUT_TEXT_CHARS)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LookupToken {
    pub simplified: String,
    pub pinyin: String,
    pub definitions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hsk_level: Option<HskLevel>,
    pub proper_name: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LookupItem {
    pub displayed_chinese: String,
    pub base_chinese: String,
    pub source_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

impl Validate for ErrorResponse {
    fn validate(&self) -> Result<(), ContractError> {
        require_nonempty_at_most("code", &self.code, 256)?;
        require_nonempty_at_most("message", &self.message, MAX_MESSAGE_CHARS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translated_region(patch_rect: NormalizedRect) -> ImageRegionReady {
        let text_polygon = vec![
            Point { x: 0.2, y: 0.3 },
            Point { x: 0.4, y: 0.3 },
            Point { x: 0.4, y: 0.4 },
            Point { x: 0.2, y: 0.4 },
        ];
        ImageRegionReady {
            item_id: "region-1".to_owned(),
            text_polygon: text_polygon.clone(),
            bubble_polygon: Some(text_polygon.clone()),
            patch: RegionPatch {
                blob_id: "patch-1".to_owned(),
                mime_type: PatchMimeType::Png,
                rect: patch_rect,
            },
            text: TranslatedText {
                termination: koharu_llm::GenerationTermination::Stop,
                protected_names: Vec::new(),
                source_text: "HELLO".to_owned(),
                base_chinese: "你好".to_owned(),
                displayed_chinese: "你好".to_owned(),
                pinyin: "nǐ hǎo".to_owned(),
                hsk: TranslatedHskStatus {
                    requested_level: HskLevel::try_from(3).unwrap(),
                    learning_mode: LearningMode::Natural,
                    strictly_valid: true,
                    level_coverage: 1.0,
                    above_level_tokens: Vec::new(),
                    teaching_terms: Vec::new(),
                    repair_state: HskRepairState::NotNeeded,
                },
            },
            provenance: SourceProvenance::Ocr,
            kind: SourceSpanKind::Dialogue,
            confidence: 0.99,
            item_order: 1,
            context_group: None,
            confidence_evidence: Some(RegionConfidenceEvidence {
                ocr_consensus: 0.99,
                geometry_coverage: 1.0,
                context_consistency: 1.0,
                cleanup_score: 1.0,
            }),
            style: BrowserTextStyle {
                font_id: "hskify-sans".to_owned(),
                category: FontCategory::Sans,
                foreground: "#000".to_owned(),
                weight: 400,
                italic_degrees: 0.0,
                outline_color: None,
                outline_width_ratio: 0.0,
                shadow_color: None,
                shadow_x_ratio: 0.0,
                shadow_y_ratio: 0.0,
                alignment: TextAlignment::Center,
                writing_mode: WritingMode::HorizontalTb,
                line_height: 1.1,
                letter_spacing_em: 0.0,
                color_bands: Vec::new(),
            },
            layout: BrowserTextLayout {
                suggested_lines: vec!["你好".to_owned()],
                font_size_to_image_width: 0.05,
                safe_polygon: text_polygon,
            },
        }
    }

    fn resource_identity(id: &str) -> ResourceIdentity {
        ResourceIdentity {
            id: id.to_owned(),
            repository: "owner/repository".to_owned(),
            repository_revision: "a".repeat(40),
            filename: "model.bin".to_owned(),
            bytes: 1,
            sha256: "b".repeat(64),
        }
    }

    fn document_request() -> DocumentJobRequest {
        serde_json::from_str(include_str!(
            "../../../fixtures/contracts/document-job-request.valid.json"
        ))
        .unwrap()
    }

    #[test]
    fn hsk_level_rejects_values_outside_supported_range() {
        assert!(HskLevel::try_from(0).is_err());
        assert!(HskLevel::try_from(7).is_err());
    }

    #[test]
    fn css_color_accepts_rgb_and_rgba_hex() {
        assert!(is_css_color("#123"));
        assert!(is_css_color("#1234"));
        assert!(is_css_color("#112233"));
        assert!(is_css_color("#11223344"));
        assert!(!is_css_color("red"));
    }

    #[test]
    fn text_color_bands_are_validated_as_an_ordered_source_structure() {
        let valid = BrowserTextStyle {
            font_id: "hskify-sans".to_owned(),
            category: FontCategory::Sans,
            foreground: "#000".to_owned(),
            weight: 600,
            italic_degrees: 0.0,
            outline_color: None,
            outline_width_ratio: 0.0,
            shadow_color: None,
            shadow_x_ratio: 0.0,
            shadow_y_ratio: 0.0,
            alignment: TextAlignment::Center,
            writing_mode: WritingMode::HorizontalTb,
            line_height: 1.1,
            letter_spacing_em: 0.0,
            color_bands: vec![
                BrowserTextColorBand {
                    position: 0.25,
                    foreground: "#111".to_owned(),
                    outline_color: None,
                },
                BrowserTextColorBand {
                    position: 0.75,
                    foreground: "#2580df".to_owned(),
                    outline_color: Some("#fff".to_owned()),
                },
            ],
        };
        assert!(valid.validate_at("style").is_ok());

        let mut unordered = valid;
        unordered.color_bands.reverse();
        assert!(unordered.validate_at("style").is_err());
    }

    #[test]
    fn translated_region_rejects_cleanup_disconnected_from_source_text() {
        translated_region(NormalizedRect {
            x: 0.21,
            y: 0.31,
            width: 0.18,
            height: 0.08,
        })
        .validate()
        .unwrap();

        let error = translated_region(NormalizedRect {
            x: 0.6,
            y: 0.6,
            width: 0.1,
            height: 0.1,
        })
        .validate()
        .unwrap_err();
        assert!(error.to_string().contains("must overlap"));
    }

    #[test]
    fn region_ready_rejects_a_pending_translation() {
        let mut region = translated_region(NormalizedRect {
            x: 0.21,
            y: 0.31,
            width: 0.18,
            height: 0.08,
        });
        region.text.hsk.repair_state = HskRepairState::Pending;
        let update = JobUpdate::ImageRegionReady {
            sequence: 1,
            region: Box::new(region),
        };

        assert!(
            update
                .validate()
                .unwrap_err()
                .to_string()
                .contains("terminal translation")
        );
    }

    #[test]
    fn lookup_contract_distinguishes_selection_from_position_anchored_hover() {
        let hover: LookupRequest = serde_json::from_value(serde_json::json!({
            "interaction": "hover",
            "characterOffset": 2,
            "context": {"sourceText": "The graduate student left.", "displayedChinese": "研究生离开。", "baseChinese": "研究生离开。", "properNames": []},
            "itemId": "region-1"
        }))
        .unwrap();
        hover.validate().unwrap();
        assert_eq!(hover.interaction, LookupInteraction::Hover);
        assert_eq!(hover.character_offset, Some(2));

        let selection: LookupRequest = serde_json::from_value(serde_json::json!({
            "interaction": "selection",
            "selectedText": "研究生"
        }))
        .unwrap();
        selection.validate().unwrap();

        let hover_without_region: LookupRequest = serde_json::from_value(serde_json::json!({
            "interaction": "hover",
            "characterOffset": 0
        }))
        .unwrap();
        assert!(hover_without_region.validate().is_err());
    }

    #[test]
    fn document_hash_is_raw_and_bounds_are_exact() {
        let mut request = document_request();
        request.blocks[1].text = "A literal \\ remains authoritative.".to_owned();
        request.source_sha256 = canonical_document_sha256(&request.blocks);
        request.validate().unwrap();

        let template = request.blocks[0].clone();
        request.blocks = (0..MAX_DOCUMENT_BLOCKS)
            .map(|index| DocumentSourceBlock {
                parent_block_id: format!("parent-{index}"),
                sub_item_order: 0,
                item_id: format!("block-{index}"),
                source_index: index as u32,
                item_order: 0,
                kind: SourceSpanKind::Prose,
                provenance: SourceProvenance::Dom,
                text: "x".to_owned(),
            })
            .collect();
        request.source_sha256 = canonical_document_sha256(&request.blocks);
        request.validate().unwrap();
        request.blocks.push(DocumentSourceBlock {
            item_id: "over-limit".to_owned(),
            source_index: MAX_DOCUMENT_BLOCKS as u32,
            item_order: 0,
            ..template.clone()
        });
        request.source_sha256 = canonical_document_sha256(&request.blocks);
        assert!(request.validate().is_err());

        request.blocks = vec![DocumentSourceBlock {
            item_id: "exact-block".to_owned(),
            source_index: 0,
            item_order: 0,
            text: "a".repeat(MAX_DOCUMENT_BLOCK_BYTES),
            ..template.clone()
        }];
        request.source_sha256 = canonical_document_sha256(&request.blocks);
        request.validate().unwrap();
        request.blocks[0].text.push('a');
        request.source_sha256 = canonical_document_sha256(&request.blocks);
        assert!(request.validate().is_err());

        request.blocks = (0..65)
            .map(|index| DocumentSourceBlock {
                parent_block_id: format!("parent-{index}"),
                sub_item_order: 0,
                item_id: format!("large-{index}"),
                source_index: index,
                item_order: 0,
                kind: SourceSpanKind::Prose,
                provenance: SourceProvenance::Dom,
                text: "z".repeat(MAX_DOCUMENT_BLOCK_BYTES),
            })
            .collect();
        request.source_sha256 = canonical_document_sha256(&request.blocks);
        assert!(request.validate().is_err());
    }

    #[test]
    fn document_focus_accepts_sixty_four_ids_and_rejects_sixty_five() {
        let focus = |count| FocusUpdateRequest::Document {
            visible_block_ids: (0..count).map(|index| format!("block-{index}")).collect(),
            active: true,
        };
        focus(MAX_VISIBLE_BLOCK_IDS).validate().unwrap();
        assert!(focus(MAX_VISIBLE_BLOCK_IDS + 1).validate().is_err());
    }

    #[test]
    fn update_batches_reject_gaps_terminal_middle_and_overlarge_windows() {
        let progress = |sequence| JobUpdate::Progress {
            sequence,
            stage: BrowserJobStage::Queued,
            stage_progress: None,
            overall_progress: None,
            current: None,
            total: None,
            message: "Queued".to_owned(),
        };
        let gap = JobUpdatesResponse {
            job_id: "job-gap".to_owned(),
            next_sequence: 3,
            updates: vec![progress(1), progress(3)],
        };
        assert!(gap.validate_after(0).is_err());
        let terminal_middle = JobUpdatesResponse {
            job_id: "job-terminal".to_owned(),
            next_sequence: 2,
            updates: vec![
                JobUpdate::Complete {
                    sequence: 1,
                    translated_count: 0,
                    preserved_count: 0,
                    message: None,
                },
                progress(2),
            ],
        };
        assert!(terminal_middle.validate_after(0).is_err());
        let oversized = JobUpdatesResponse {
            job_id: "job-large".to_owned(),
            next_sequence: 1_025,
            updates: (1..=1_025).map(progress).collect(),
        };
        assert!(oversized.validate_after(0).is_err());
    }

    #[test]
    fn health_requires_lowercase_sorted_exact_resource_identities() {
        let valid = HealthResponse {
            build_fingerprint: BUILD_FINGERPRINT.to_owned(),
            engine_version: "test".to_owned(),
            status: HealthStatus::Ready,
            setup_state: BrowserSetupState::Ready,
            resource_identities: vec![
                resource_identity("detector-config"),
                resource_identity("translation-model"),
            ],
        };
        valid.validate().unwrap();

        let mut unsorted = valid.clone();
        unsorted.resource_identities.reverse();
        assert!(unsorted.validate().is_err());

        let mut uppercase_digest = valid;
        uppercase_digest.resource_identities[0].sha256 = "B".repeat(64);
        assert!(uppercase_digest.validate().is_err());
    }
}
