use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const PROTOCOL_VERSION: u8 = 1;
pub const HSK_STANDARD: &str = "2.0";
pub const SOURCE_LANGUAGE: &str = "en";
pub const TARGET_LANGUAGE: &str = "zh-CN";
pub const MAX_PRECEDING_CONTEXT: usize = 12;

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

fn require_protocol(value: u8) -> Result<(), ContractError> {
    if value == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(ContractError::at(
            "protocolVersion",
            format!("expected {PROTOCOL_VERSION}, got {value}"),
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
    pub protocol_version: u8,
    pub extension_version: String,
    pub extension_origin: String,
}

impl Validate for NativeHandshakeRequest {
    fn validate(&self) -> Result<(), ContractError> {
        require_protocol(self.protocol_version)?;
        require_nonempty("extensionVersion", &self.extension_version)?;
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
    pub protocol_version: u8,
    pub engine_version: String,
    pub port: u16,
    pub token: String,
    pub session_expires_at_unix_ms: u64,
    pub capabilities: BrowserCapabilities,
}

impl Validate for NativeReadyResponse {
    fn validate(&self) -> Result<(), ContractError> {
        require_protocol(self.protocol_version)?;
        require_nonempty("engineVersion", &self.engine_version)?;
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
                "protocol v1 supports English only",
            ));
        }
        if self.target_languages != [TARGET_LANGUAGE] {
            return Err(ContractError::at(
                "capabilities.targetLanguages",
                "protocol v1 supports Simplified Chinese only",
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
pub struct HealthResponse {
    pub protocol_version: u8,
    pub engine_version: String,
    pub status: HealthStatus,
    pub setup_state: BrowserSetupState,
}

impl Validate for HealthResponse {
    fn validate(&self) -> Result<(), ContractError> {
        require_protocol(self.protocol_version)?;
        require_nonempty("engineVersion", &self.engine_version)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Ready,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserJobRequest {
    pub protocol_version: u8,
    pub client_image_id: String,
    pub source_sha256: String,
    pub source_mime_type: String,
    pub natural_width: u32,
    pub natural_height: u32,
    pub page_session_id: String,
    pub page_index: u32,
    pub settings: BrowserJobSettings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preceding_context: Option<Vec<DialogueContext>>,
}

impl Validate for BrowserJobRequest {
    fn validate(&self) -> Result<(), ContractError> {
        require_protocol(self.protocol_version)?;
        require_nonempty("clientImageId", &self.client_image_id)?;
        require_sha256("sourceSha256", &self.source_sha256)?;
        if !matches!(
            self.source_mime_type.as_str(),
            "image/png" | "image/jpeg" | "image/webp" | "image/gif"
        ) {
            return Err(ContractError::at(
                "sourceMimeType",
                "must be a supported raster image MIME type",
            ));
        }
        if self.natural_width == 0 || self.natural_height == 0 {
            return Err(ContractError::at(
                "naturalWidth",
                "decoded image dimensions must be non-zero",
            ));
        }
        require_nonempty("pageSessionId", &self.page_session_id)?;
        self.settings.validate()?;
        if self
            .preceding_context
            .as_ref()
            .is_some_and(|items| items.len() > MAX_PRECEDING_CONTEXT)
        {
            return Err(ContractError::at(
                "precedingContext",
                format!("must contain at most {MAX_PRECEDING_CONTEXT} entries"),
            ));
        }
        if let Some(items) = &self.preceding_context {
            for (index, item) in items.iter().enumerate() {
                require_nonempty(
                    &format!("precedingContext[{index}].sourceEnglish"),
                    &item.source_english,
                )?;
                require_nonempty(&format!("precedingContext[{index}].chinese"), &item.chinese)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserJobCreated {
    pub protocol_version: u8,
    pub job_id: String,
}

impl Validate for BrowserJobCreated {
    fn validate(&self) -> Result<(), ContractError> {
        require_protocol(self.protocol_version)?;
        require_nonempty("jobId", &self.job_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RetranslateRequest {
    pub protocol_version: u8,
    pub settings: RetranslateSettings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preceding_context: Option<Vec<DialogueContext>>,
}

impl Validate for RetranslateRequest {
    fn validate(&self) -> Result<(), ContractError> {
        require_protocol(self.protocol_version)?;
        self.settings.validate()?;
        if self
            .preceding_context
            .as_ref()
            .is_some_and(|items| items.len() > MAX_PRECEDING_CONTEXT)
        {
            return Err(ContractError::at(
                "precedingContext",
                format!("must contain at most {MAX_PRECEDING_CONTEXT} entries"),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RetranslateSettings {
    pub hsk_standard: String,
    pub hsk_level: HskLevel,
}

impl Validate for RetranslateSettings {
    fn validate(&self) -> Result<(), ContractError> {
        if self.hsk_standard != HSK_STANDARD {
            return Err(ContractError::at(
                "settings.hskStandard",
                "protocol v1 supports HSK 2.0 only",
            ));
        }
        Ok(())
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
}

impl Validate for BrowserJobSettings {
    fn validate(&self) -> Result<(), ContractError> {
        if self.source_language != SOURCE_LANGUAGE {
            return Err(ContractError::at(
                "settings.sourceLanguage",
                "protocol v1 supports English only",
            ));
        }
        if self.target_language != TARGET_LANGUAGE {
            return Err(ContractError::at(
                "settings.targetLanguage",
                "protocol v1 supports Simplified Chinese only",
            ));
        }
        if self.hsk_standard != HSK_STANDARD {
            return Err(ContractError::at(
                "settings.hskStandard",
                "protocol v1 supports HSK 2.0 only",
            ));
        }
        if self.translate_sound_effects {
            return Err(ContractError::at(
                "settings.translateSoundEffects",
                "sound-effect translation is disabled in protocol v1",
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DialogueContext {
    pub source_english: String,
    pub chinese: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserJobResult {
    pub protocol_version: u8,
    pub job_id: String,
    pub source_sha256: String,
    pub source_width: u32,
    pub source_height: u32,
    pub clean_image_blob_id: String,
    pub clean_image_mime_type: CleanImageMimeType,
    pub regions: Vec<BrowserRegion>,
    pub warnings: Vec<BrowserWarning>,
    pub cache: BrowserCacheStatus,
}

impl Validate for BrowserJobResult {
    fn validate(&self) -> Result<(), ContractError> {
        require_protocol(self.protocol_version)?;
        require_nonempty("jobId", &self.job_id)?;
        require_sha256("sourceSha256", &self.source_sha256)?;
        if self.source_width == 0 || self.source_height == 0 {
            return Err(ContractError::at(
                "sourceWidth",
                "source dimensions must be non-zero",
            ));
        }
        require_nonempty("cleanImageBlobId", &self.clean_image_blob_id)?;
        let mut ids = HashSet::with_capacity(self.regions.len());
        for (index, region) in self.regions.iter().enumerate() {
            region.validate_at(index)?;
            if !ids.insert(region.id.as_str()) {
                return Err(ContractError::at(
                    format!("regions[{index}].id"),
                    "region IDs must be unique",
                ));
            }
        }
        for (index, warning) in self.warnings.iter().enumerate() {
            require_nonempty(&format!("warnings[{index}].message"), &warning.message)?;
            if let Some(region_id) = &warning.region_id
                && !ids.contains(region_id.as_str())
            {
                return Err(ContractError::at(
                    format!("warnings[{index}].regionId"),
                    "must reference a region in this result",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanImageMimeType {
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/webp")]
    Webp,
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
pub struct BrowserRegion {
    pub id: String,
    pub kind: RegionKind,
    pub text_polygon: Vec<Point>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bubble_polygon: Option<Vec<Point>>,
    pub rotation_degrees: f32,
    pub source_english: String,
    pub faithful_chinese: String,
    pub displayed_chinese: String,
    pub pinyin: String,
    pub ocr_confidence: f32,
    pub reading_order: u32,
    pub vocabulary: VocabularyStatus,
    pub style: BrowserTextStyle,
    pub layout: BrowserTextLayout,
}

impl BrowserRegion {
    fn validate_at(&self, index: usize) -> Result<(), ContractError> {
        let path = format!("regions[{index}]");
        require_nonempty(&format!("{path}.id"), &self.id)?;
        require_polygon(&format!("{path}.textPolygon"), &self.text_polygon)?;
        if let Some(points) = &self.bubble_polygon {
            require_polygon(&format!("{path}.bubblePolygon"), points)?;
        }
        if !self.rotation_degrees.is_finite() {
            return Err(ContractError::at(
                format!("{path}.rotationDegrees"),
                "must be finite",
            ));
        }
        require_unit(&format!("{path}.ocrConfidence"), self.ocr_confidence)?;
        if self.kind != RegionKind::Sfx {
            require_nonempty(&format!("{path}.sourceEnglish"), &self.source_english)?;
            require_nonempty(&format!("{path}.faithfulChinese"), &self.faithful_chinese)?;
            require_nonempty(&format!("{path}.displayedChinese"), &self.displayed_chinese)?;
        }
        self.vocabulary.validate_at(&format!("{path}.vocabulary"))?;
        self.style.validate_at(&format!("{path}.style"))?;
        self.layout.validate_at(&format!("{path}.layout"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegionKind {
    Dialogue,
    Caption,
    Thought,
    Sfx,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VocabularyStatus {
    pub requested_hsk_level: HskLevel,
    pub strictly_valid: bool,
    pub exceptions: Vec<VocabularyException>,
}

impl VocabularyStatus {
    fn validate_at(&self, path: &str) -> Result<(), ContractError> {
        let mut texts = HashSet::new();
        for (index, exception) in self.exceptions.iter().enumerate() {
            require_nonempty(&format!("{path}.exceptions[{index}].text"), &exception.text)?;
            if !texts.insert(exception.text.as_str()) {
                return Err(ContractError::at(
                    format!("{path}.exceptions[{index}].text"),
                    "duplicate exception",
                ));
            }
        }
        if self.strictly_valid && !self.exceptions.is_empty() {
            return Err(ContractError::at(
                format!("{path}.strictlyValid"),
                "cannot be strict when vocabulary exceptions are present",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VocabularyException {
    pub text: String,
    pub reason: VocabularyExceptionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VocabularyExceptionReason {
    PersonName,
    PlaceName,
    Title,
    UnavoidableProperNoun,
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
pub struct BrowserWarning {
    pub code: BrowserWarningCode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_id: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BrowserWarningCode {
    LowOcrConfidence,
    HskException,
    HskRewriteFailed,
    TextFitDegraded,
    StyleLowConfidence,
    SfxSkipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserCacheStatus {
    pub detection_hit: bool,
    pub ocr_hit: bool,
    pub inpaint_hit: bool,
    pub translation_hit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserJobStatus {
    pub revision: u64,
    pub job_id: String,
    pub state: BrowserJobState,
    pub stage: BrowserJobStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_progress: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overall_progress: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u32>,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

impl Validate for BrowserJobStatus {
    fn validate(&self) -> Result<(), ContractError> {
        if self.revision == 0 {
            return Err(ContractError::at("revision", "must start at 1"));
        }
        require_nonempty("jobId", &self.job_id)?;
        require_nonempty("message", &self.message)?;
        if let Some(value) = self.stage_progress {
            require_unit("stageProgress", value)?;
        }
        if let Some(value) = self.overall_progress {
            require_unit("overallProgress", value)?;
        }
        if self.current.is_some() != self.total.is_some() {
            return Err(ContractError::at(
                "current",
                "current and total must be present together",
            ));
        }
        if let (Some(current), Some(total)) = (self.current, self.total)
            && (total == 0 || current > total)
        {
            return Err(ContractError::at(
                "current",
                "must be less than or equal to a non-zero total",
            ));
        }
        let stage_matches = match self.state {
            BrowserJobState::Running => !self.stage.is_terminal(),
            BrowserJobState::Complete => self.stage == BrowserJobStage::Complete,
            BrowserJobState::Failed => self.stage == BrowserJobStage::Failed,
            BrowserJobState::Cancelled => self.stage == BrowserJobStage::Cancelled,
        };
        if !stage_matches {
            return Err(ContractError::at("stage", "must agree with the job state"));
        }
        if self.state == BrowserJobState::Failed
            && self.error_code.as_deref().is_none_or(str::is_empty)
        {
            return Err(ContractError::at(
                "errorCode",
                "failed jobs require an error code",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BrowserJobState {
    Running,
    Complete,
    Failed,
    Cancelled,
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
    HskRewriting,
    HskValidating,
    Styling,
    Packaging,
    Complete,
    Failed,
    Cancelled,
}

impl BrowserJobStage {
    fn is_terminal(self) -> bool {
        matches!(self, Self::Complete | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserSetupStatus {
    pub state: BrowserSetupState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_pack_id: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LookupRequest {
    pub selected_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_id: Option<String>,
}

impl Validate for LookupRequest {
    fn validate(&self) -> Result<(), ContractError> {
        require_nonempty("selectedText", &self.selected_text)?;
        if self.selected_text.chars().count() > 256 {
            return Err(ContractError::at(
                "selectedText",
                "must contain at most 256 characters",
            ));
        }
        if self.job_id.is_some() != self.region_id.is_some() {
            return Err(ContractError::at(
                "regionId",
                "jobId and regionId must be present together",
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
    pub faithful_chinese: String,
    pub source_english: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ErrorResponse {
    pub protocol_version: u8,
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

impl Validate for ErrorResponse {
    fn validate(&self) -> Result<(), ContractError> {
        require_protocol(self.protocol_version)?;
        require_nonempty("code", &self.code)?;
        require_nonempty("message", &self.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsk_level_rejects_values_outside_v2_range() {
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
}
