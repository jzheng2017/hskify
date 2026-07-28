use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Exact build affinity shared by the extension, native host, and daemon.
///
/// This is deliberately not a negotiable protocol version. A mismatched build
/// must restart the native/daemon pair that shipped with the extension.
pub const BUILD_FINGERPRINT: &str = "hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-27-r6";
pub const HSK_STANDARD: &str = "2.0";
pub const SOURCE_LANGUAGE: &str = "en";
pub const TARGET_LANGUAGE: &str = "zh-CN";
pub const MAX_PRECEDING_CONTEXT: usize = 6;
pub const MAX_PROPER_NAME_GLOSSARY: usize = 64;
pub const MAX_VISIBLE_RECTS: usize = 64;

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
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(ContractError::at(
            path,
            "must be a 64-character hexadecimal SHA-256",
        ))
    }
}

fn require_polygon(path: &str, points: &[Point]) -> Result<(), ContractError> {
    if points.len() < 3 {
        return Err(ContractError::at(
            path,
            "must contain at least three points",
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
pub struct CreateJobRequest {
    pub build_fingerprint: String,
    pub client_image_id: String,
    pub source_sha256: String,
    pub source_mime_type: String,
    pub natural_width: u32,
    pub natural_height: u32,
    pub page_session_id: String,
    pub page_index: u32,
    pub settings: BrowserJobSettings,
    pub visible_rects: Vec<NormalizedRect>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preceding_context: Option<Vec<DialogueContext>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proper_name_glossary: Option<Vec<ProperNameGlossaryEntry>>,
}

impl Validate for CreateJobRequest {
    fn validate(&self) -> Result<(), ContractError> {
        require_build_fingerprint("buildFingerprint", &self.build_fingerprint)?;
        validate_job_fields(self)?;
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

fn validate_job_fields(request: &CreateJobRequest) -> Result<(), ContractError> {
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
    if request
        .preceding_context
        .as_ref()
        .is_some_and(|items| items.len() > MAX_PRECEDING_CONTEXT)
    {
        return Err(ContractError::at(
            "precedingContext",
            format!("must contain at most {MAX_PRECEDING_CONTEXT} entries"),
        ));
    }
    if let Some(items) = &request.preceding_context {
        for (index, item) in items.iter().enumerate() {
            require_nonempty(
                &format!("precedingContext[{index}].sourceEnglish"),
                &item.source_english,
            )?;
            require_nonempty(&format!("precedingContext[{index}].chinese"), &item.chinese)?;
        }
    }
    if request
        .proper_name_glossary
        .as_ref()
        .is_some_and(|items| items.len() > MAX_PROPER_NAME_GLOSSARY)
    {
        return Err(ContractError::at(
            "properNameGlossary",
            format!("must contain at most {MAX_PROPER_NAME_GLOSSARY} entries"),
        ));
    }
    if let Some(items) = &request.proper_name_glossary {
        let mut source_names = HashSet::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            require_nonempty_at_most(
                &format!("properNameGlossary[{index}].sourceEnglish"),
                &item.source_english,
                256,
            )?;
            require_nonempty_at_most(
                &format!("properNameGlossary[{index}].chinese"),
                &item.chinese,
                128,
            )?;
            let normalized = item.source_english.trim().to_ascii_lowercase();
            if !source_names.insert(normalized) {
                return Err(ContractError::at(
                    format!("properNameGlossary[{index}].sourceEnglish"),
                    "must be unique ignoring ASCII case",
                ));
            }
        }
    }
    Ok(())
}

/// Validated input passed from the HTTP boundary into the cleaning pipeline.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct BrowserJobRequest {
    pub source_sha256: String,
    pub source_mime_type: String,
    pub natural_width: u32,
    pub natural_height: u32,
    pub page_session_id: String,
    pub settings: BrowserJobSettings,
    pub preceding_context: Option<Vec<DialogueContext>>,
    pub proper_name_glossary: Option<Vec<ProperNameGlossaryEntry>>,
}

impl CreateJobRequest {
    pub(crate) fn pipeline_request(&self) -> BrowserJobRequest {
        BrowserJobRequest {
            source_sha256: self.source_sha256.clone(),
            source_mime_type: self.source_mime_type.clone(),
            natural_width: self.natural_width,
            natural_height: self.natural_height,
            page_session_id: self.page_session_id.clone(),
            settings: self.settings.clone(),
            preceding_context: self.preceding_context.clone(),
            proper_name_glossary: self.proper_name_glossary.clone(),
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
    pub reading_direction: ReadingDirection,
    pub translate_sound_effects: bool,
    pub name_translation: NameTranslation,
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
        if self.translate_sound_effects {
            return Err(ContractError::at(
                "settings.translateSoundEffects",
                "sound-effect translation is disabled in this build",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReadingDirection {
    Auto,
    Ltr,
    Rtl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NameTranslation {
    KeepOriginal,
    Chinese,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DialogueContext {
    pub source_english: String,
    pub chinese: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProperNameGlossaryEntry {
    pub source_english: String,
    pub chinese: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        if self.outline_width_ratio < 0.0 {
            return Err(ContractError::at(
                format!("{path}.outlineWidthRatio"),
                "must not be negative",
            ));
        }
        if self.line_height <= 0.0 {
            return Err(ContractError::at(
                format!("{path}.lineHeight"),
                "must be positive",
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safe_polygon: Option<Vec<Point>>,
}

impl BrowserTextLayout {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        if !self.font_size_to_image_width.is_finite() || self.font_size_to_image_width <= 0.0 {
            return Err(ContractError::at(
                format!("{path}.fontSizeToImageWidth"),
                "must be a positive finite ratio",
            ));
        }
        if let Some(points) = &self.safe_polygon {
            require_polygon(&format!("{path}.safePolygon"), points)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ViewportUpdateRequest {
    pub visible_rects: Vec<NormalizedRect>,
    pub active: bool,
}

impl Validate for ViewportUpdateRequest {
    fn validate(&self) -> Result<(), ContractError> {
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
pub struct ProgressiveHskStatus {
    pub requested_level: HskLevel,
    pub strictly_valid: bool,
    pub above_level_tokens: Vec<String>,
    pub repair_state: HskRepairState,
}

impl ProgressiveHskStatus {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        let mut tokens = HashSet::with_capacity(self.above_level_tokens.len());
        for (index, token) in self.above_level_tokens.iter().enumerate() {
            require_nonempty(&format!("{path}.aboveLevelTokens[{index}]"), token)?;
            if !tokens.insert(token.as_str()) {
                return Err(ContractError::at(
                    format!("{path}.aboveLevelTokens[{index}]"),
                    "duplicate above-level token",
                ));
            }
        }
        if self.strictly_valid && !self.above_level_tokens.is_empty() {
            return Err(ContractError::at(
                format!("{path}.strictlyValid"),
                "strictly valid text cannot retain above-level tokens",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProgressiveRegion {
    pub id: String,
    pub text_polygon: Vec<Point>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bubble_polygon: Option<Vec<Point>>,
    pub patch: RegionPatch,
    pub source_english: String,
    pub base_chinese: String,
    pub displayed_chinese: String,
    pub pinyin: String,
    pub ocr_confidence: f32,
    pub reading_order: u32,
    pub style: BrowserTextStyle,
    pub layout: BrowserTextLayout,
    pub hsk: ProgressiveHskStatus,
}

impl ProgressiveRegion {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        require_nonempty(&format!("{path}.id"), &self.id)?;
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
        let overlap_width = text_x1.min(patch_x1) - text_x0.max(patch_x0);
        let overlap_height = text_y1.min(patch_y1) - text_y0.max(patch_y0);
        if overlap_width <= f32::EPSILON || overlap_height <= f32::EPSILON {
            return Err(ContractError::at(
                format!("{path}.patch.rect"),
                "must overlap the source text polygon",
            ));
        }
        require_nonempty(&format!("{path}.sourceEnglish"), &self.source_english)?;
        require_nonempty(&format!("{path}.baseChinese"), &self.base_chinese)?;
        require_nonempty(&format!("{path}.displayedChinese"), &self.displayed_chinese)?;
        require_nonempty(&format!("{path}.pinyin"), &self.pinyin)?;
        require_unit(&format!("{path}.ocrConfidence"), self.ocr_confidence)?;
        self.style.validate_at(&format!("{path}.style"))?;
        self.layout.validate_at(&format!("{path}.layout"))?;
        self.hsk.validate_at(&format!("{path}.hsk"))
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

impl Validate for ProgressiveRegion {
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
    RegionReady {
        sequence: u64,
        region: Box<ProgressiveRegion>,
    },
    RegionRefined {
        sequence: u64,
        region_id: String,
        displayed_chinese: String,
        pinyin: String,
        hsk: ProgressiveHskStatus,
    },
    Complete {
        sequence: u64,
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
            | Self::RegionReady { sequence, .. }
            | Self::RegionRefined { sequence, .. }
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
                require_nonempty("message", message)
            }
            Self::RegionReady { region, .. } => region.validate(),
            Self::RegionRefined {
                region_id,
                displayed_chinese,
                pinyin,
                hsk,
                ..
            } => {
                require_nonempty("regionId", region_id)?;
                require_nonempty("displayedChinese", displayed_chinese)?;
                require_nonempty("pinyin", pinyin)?;
                hsk.validate_at("hsk")
            }
            Self::Complete { message, .. } | Self::Cancelled { message, .. } => {
                if let Some(message) = message {
                    require_nonempty("message", message)?;
                }
                Ok(())
            }
            Self::Failed { code, message, .. } => {
                require_nonempty("code", code)?;
                require_nonempty("message", message)
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
        require_nonempty("jobId", &self.job_id)?;
        let mut previous = 0;
        for update in &self.updates {
            update.validate()?;
            if update.sequence() <= previous {
                return Err(ContractError::at(
                    "updates",
                    "update sequences must be strictly increasing",
                ));
            }
            previous = update.sequence();
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserJobStage {
    Queued,
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
    Ready,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LookupInteraction {
    Selection,
    Hover,
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
    pub job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_id: Option<String>,
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
        if self.job_id.is_some() != self.region_id.is_some() {
            return Err(ContractError::at(
                "regionId",
                "jobId and regionId must be present together",
            ));
        }
        if self.interaction == LookupInteraction::Hover && self.job_id.is_none() {
            return Err(ContractError::at(
                "jobId",
                "hover lookup requires a translated job and region",
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
    pub region: Option<LookupRegion>,
}

impl Validate for LookupResult {
    fn validate(&self) -> Result<(), ContractError> {
        require_nonempty("selectedText", &self.selected_text)?;
        for (index, token) in self.tokens.iter().enumerate() {
            require_nonempty(&format!("tokens[{index}].simplified"), &token.simplified)?;
            if !token.proper_name {
                require_nonempty(&format!("tokens[{index}].pinyin"), &token.pinyin)?;
            }
            if token.definitions.iter().any(|item| item.trim().is_empty()) {
                return Err(ContractError::at(
                    format!("tokens[{index}].definitions"),
                    "definitions must not contain empty values",
                ));
            }
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
pub struct LookupRegion {
    pub displayed_chinese: String,
    pub base_chinese: String,
    pub source_english: String,
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
        require_nonempty("code", &self.code)?;
        require_nonempty("message", &self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progressive_region(patch_rect: NormalizedRect) -> ProgressiveRegion {
        let text_polygon = vec![
            Point { x: 0.2, y: 0.3 },
            Point { x: 0.4, y: 0.3 },
            Point { x: 0.4, y: 0.4 },
            Point { x: 0.2, y: 0.4 },
        ];
        ProgressiveRegion {
            id: "region-1".to_owned(),
            text_polygon: text_polygon.clone(),
            bubble_polygon: Some(text_polygon.clone()),
            patch: RegionPatch {
                blob_id: "patch-1".to_owned(),
                mime_type: PatchMimeType::Png,
                rect: patch_rect,
            },
            source_english: "HELLO".to_owned(),
            base_chinese: "ä½ å¥½".to_owned(),
            displayed_chinese: "ä½ å¥½".to_owned(),
            pinyin: "nÇ hÇŽo".to_owned(),
            ocr_confidence: 0.99,
            reading_order: 1,
            style: BrowserTextStyle {
                font_id: "hmt-sans".to_owned(),
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
                suggested_lines: vec!["ä½ å¥½".to_owned()],
                font_size_to_image_width: 0.05,
                safe_polygon: Some(text_polygon),
            },
            hsk: ProgressiveHskStatus {
                requested_level: HskLevel::try_from(3).unwrap(),
                strictly_valid: true,
                above_level_tokens: Vec::new(),
                repair_state: HskRepairState::NotNeeded,
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
            font_id: "hmt-sans".to_owned(),
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
    fn progressive_region_rejects_cleanup_disconnected_from_source_text() {
        progressive_region(NormalizedRect {
            x: 0.21,
            y: 0.31,
            width: 0.18,
            height: 0.08,
        })
        .validate()
        .unwrap();

        let error = progressive_region(NormalizedRect {
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
    fn lookup_contract_distinguishes_selection_from_position_anchored_hover() {
        let hover: LookupRequest = serde_json::from_value(serde_json::json!({
            "interaction": "hover",
            "characterOffset": 2,
            "jobId": "job-1",
            "regionId": "region-1"
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
