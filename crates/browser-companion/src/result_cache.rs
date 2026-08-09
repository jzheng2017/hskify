use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use hsk_control::{
    JIEBA_CRATE_VERSION, JIEBA_EMBEDDED_DICTIONARY_SHA256, LOOKUP_REVISION, NORMALIZATION_REVISION,
    SEGMENTATION_REVISION, UNICODE_NORMALIZATION_CRATE_VERSION,
    UNICODE_NORMALIZATION_TABLES_SHA256,
};
use image::{ImageFormat, ImageReader, Limits};
use koharu_app::llm::{
    HSK_TRANSLATION_MODEL_REVISION, HSK_TRANSLATION_PROMPT_HASH, HSK_TRANSLATION_VALIDATOR_HASH,
};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::chapter_session::ChapterContextUnit;
use crate::contracts::{
    BUILD_FINGERPRINT, BrowserJobSettings, BrowserSurfaceKind, DocumentBlockPreserved,
    DocumentBlockReady, DocumentJobRequest, DocumentSourceBlock, ImagePipelineInput,
    ImageRegionPreserved, ImageRegionReady, ReadingDirection, Validate,
};
use crate::crypto::sha256_hex;
use crate::pipeline_adapter::ItemLookupContext;
use crate::setup::{
    DICTIONARY_RESOURCE_BYTES, DICTIONARY_RESOURCE_SHA256, HSK_RESOURCE_BYTES, HSK_RESOURCE_SHA256,
};

pub(crate) const RESULT_CACHE_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const RESULT_CACHE_MAX_ENTRY_BYTES: u64 = 512 * 1024 * 1024;
const RESULT_CACHE_MAX_DECODED_PATCH_BYTES: u64 = 256 * 1024 * 1024;
const RESULT_CACHE_SCHEMA: &str = "hskify-tagged-chapter-result-2026-08-09-v1";
const RESULT_CACHE_PIPELINE_REVISION: &str =
    "shared-language-image-document-pipeline-v1-2026-08-09";
const MODEL_RESOURCE_MANIFEST: &[u8] = include_bytes!("../../../data/model-packs/manifest.v1.json");

#[derive(Debug, Clone)]
pub(crate) struct CachedImageRegion {
    pub region: ImageRegionReady,
    pub lookup_context: ItemLookupContext,
    pub patch_png: Arc<[u8]>,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedImageJob {
    pub regions: Vec<CachedImageRegion>,
    pub preserved: Vec<ImageRegionPreserved>,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedDocumentJob {
    pub blocks: Vec<DocumentBlockReady>,
    pub preserved: Vec<DocumentBlockPreserved>,
    pub lookup_contexts: Vec<(String, ItemLookupContext)>,
}

impl CachedDocumentJob {
    fn validate(&self) -> Result<()> {
        let mut item_ids = HashSet::new();
        for block in &self.blocks {
            JobUpdateValidation::document_ready(block)?;
            if !item_ids.insert(block.item_id.as_str()) {
                bail!("cached document contains duplicate terminal item identity");
            }
        }
        for block in &self.preserved {
            JobUpdateValidation::document_preserved(block)?;
            if !item_ids.insert(block.item_id.as_str()) {
                bail!("cached document contains duplicate terminal item identity");
            }
        }
        let lookup_ids = self
            .lookup_contexts
            .iter()
            .map(|(item_id, _)| item_id.as_str())
            .collect::<HashSet<_>>();
        if lookup_ids.len() != self.lookup_contexts.len()
            || lookup_ids.len() != item_ids.len()
            || !item_ids.iter().all(|item_id| lookup_ids.contains(item_id))
        {
            bail!("cached document lookup contexts must match every terminal item exactly");
        }
        Ok(())
    }

    fn validate_against(&self, request: &DocumentJobRequest) -> Result<()> {
        self.validate()?;
        enum Terminal<'a> {
            Ready(&'a DocumentBlockReady),
            Preserved(&'a DocumentBlockPreserved),
        }
        let mut terminals = HashMap::with_capacity(self.blocks.len() + self.preserved.len());
        for block in &self.blocks {
            terminals.insert(block.item_id.as_str(), Terminal::Ready(block));
        }
        for block in &self.preserved {
            terminals.insert(block.item_id.as_str(), Terminal::Preserved(block));
        }
        if terminals.len() != request.blocks.len() {
            bail!("cached document must contain exactly one terminal item per source block");
        }
        let lookup = self
            .lookup_contexts
            .iter()
            .map(|(item_id, context)| (item_id.as_str(), context))
            .collect::<HashMap<_, _>>();
        for source in &request.blocks {
            let terminal = terminals.get(source.item_id.as_str()).ok_or_else(|| {
                anyhow::anyhow!("cached document is missing source item {}", source.item_id)
            })?;
            let (source_index, item_order, kind, source_text, base_text, displayed_text) =
                match terminal {
                    Terminal::Ready(block) => (
                        block.source_index,
                        block.item_order,
                        block.kind,
                        block.text.source_text.as_str(),
                        block.text.base_chinese.as_str(),
                        block.text.displayed_chinese.as_str(),
                    ),
                    Terminal::Preserved(block) => (
                        block.source_index,
                        block.item_order,
                        block.kind,
                        block.source_text.as_str(),
                        block.source_text.as_str(),
                        block.source_text.as_str(),
                    ),
                };
            if source_index != source.source_index
                || item_order != source.item_order
                || kind != source.kind
                || source_text != source.text
            {
                bail!("cached document terminal item does not match its source block");
            }
            let context = lookup
                .get(source.item_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("cached document lookup context is missing"))?;
            if context.source_text != source.text
                || context.base_chinese != base_text
                || context.displayed_chinese != displayed_text
            {
                bail!("cached document lookup context does not match its terminal item");
            }
        }
        Ok(())
    }
}

struct JobUpdateValidation;

impl JobUpdateValidation {
    fn document_ready(block: &DocumentBlockReady) -> Result<()> {
        crate::contracts::JobUpdate::DocumentBlockReady {
            sequence: 1,
            block: block.clone(),
        }
        .validate()
        .map_err(anyhow::Error::new)
    }

    fn document_preserved(block: &DocumentBlockPreserved) -> Result<()> {
        crate::contracts::JobUpdate::DocumentBlockPreserved {
            sequence: 1,
            block: block.clone(),
        }
        .validate()
        .map_err(anyhow::Error::new)
    }
}

impl CachedImageJob {
    fn validate(&self) -> Result<()> {
        let mut region_ids = HashSet::new();
        let mut patch_ids = HashSet::new();
        for cached in &self.regions {
            cached
                .region
                .validate()
                .context("validate cached image translation")?;
            cached
                .lookup_context
                .validate_against(&cached.region)
                .context("validate cached lookup context")?;
            if !region_ids.insert(cached.region.item_id.as_str()) {
                bail!("cached image contains duplicate terminal item identity");
            }
            if !patch_ids.insert(cached.region.patch.blob_id.as_str()) {
                bail!("cached job contains duplicate patch identity");
            }
        }
        for region in &self.preserved {
            crate::contracts::JobUpdate::ImageRegionPreserved {
                sequence: 1,
                region: region.clone(),
            }
            .validate()
            .context("validate cached image preservation")?;
            if !region_ids.insert(region.item_id.as_str()) {
                bail!("cached image contains duplicate terminal item identity");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredJob {
    schema: String,
    build_fingerprint: String,
    pipeline_fingerprint: String,
    key: String,
    result: StoredChapterResult,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
enum StoredChapterResult {
    Image {
        regions: Vec<StoredRegion>,
        preserved: Vec<ImageRegionPreserved>,
    },
    Document {
        blocks: Vec<DocumentBlockReady>,
        preserved: Vec<DocumentBlockPreserved>,
        lookup_contexts: Vec<StoredLookupContext>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredLookupContext {
    item_id: String,
    context: ItemLookupContext,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredRegion {
    region: ImageRegionReady,
    lookup_context: ItemLookupContext,
    patch_png_base64: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageCacheIdentity<'a> {
    source_sha256: &'a str,
    natural_width: u32,
    natural_height: u32,
    surface_kind: BrowserSurfaceKind,
    reading_direction: ReadingDirection,
    settings: &'a BrowserJobSettings,
    surrounding_context: &'a [ChapterContextUnit],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DocumentCacheIdentity<'a> {
    source_sha256: &'a str,
    settings: &'a BrowserJobSettings,
    blocks: &'a [DocumentSourceBlock],
}

#[derive(Serialize)]
#[serde(tag = "kind", content = "source", rename_all = "lowercase")]
enum CacheSourceIdentity<'a> {
    Image(ImageCacheIdentity<'a>),
    Document(DocumentCacheIdentity<'a>),
}

fn image_cache_identity<'a>(
    request: &'a ImagePipelineInput,
    surrounding_context: &'a [ChapterContextUnit],
) -> CacheSourceIdentity<'a> {
    CacheSourceIdentity::Image(ImageCacheIdentity {
        source_sha256: &request.source_sha256,
        natural_width: request.natural_width,
        natural_height: request.natural_height,
        surface_kind: request.surface_kind,
        reading_direction: request.reading_direction,
        settings: &request.settings,
        surrounding_context,
    })
}

fn document_cache_identity(request: &DocumentJobRequest) -> CacheSourceIdentity<'_> {
    CacheSourceIdentity::Document(DocumentCacheIdentity {
        source_sha256: &request.source_sha256,
        settings: &request.settings,
        blocks: &request.blocks,
    })
}

#[derive(Debug)]
pub(crate) struct ResultCache {
    root: PathBuf,
    max_bytes: u64,
    max_entry_bytes: u64,
    max_decoded_patch_bytes: u64,
}

impl ResultCache {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self {
            root,
            max_bytes: RESULT_CACHE_MAX_BYTES,
            max_entry_bytes: RESULT_CACHE_MAX_ENTRY_BYTES,
            max_decoded_patch_bytes: RESULT_CACHE_MAX_DECODED_PATCH_BYTES,
        }
    }

    #[cfg(test)]
    fn with_limit(root: PathBuf, max_bytes: u64) -> Self {
        Self {
            root,
            max_bytes,
            max_entry_bytes: RESULT_CACHE_MAX_ENTRY_BYTES,
            max_decoded_patch_bytes: RESULT_CACHE_MAX_DECODED_PATCH_BYTES,
        }
    }

    #[cfg(test)]
    fn with_load_limits(root: PathBuf, max_entry_bytes: u64, max_decoded_patch_bytes: u64) -> Self {
        Self {
            root,
            max_bytes: RESULT_CACHE_MAX_BYTES,
            max_entry_bytes,
            max_decoded_patch_bytes,
        }
    }

    pub(crate) fn key_image(
        request: &ImagePipelineInput,
        surrounding_context: &[ChapterContextUnit],
    ) -> Result<String> {
        Self::key_with_pipeline_fingerprint(
            image_cache_identity(request, surrounding_context),
            &pipeline_fingerprint()?,
        )
    }

    pub(crate) fn key_document(request: &DocumentJobRequest) -> Result<String> {
        Self::key_with_pipeline_fingerprint(
            document_cache_identity(request),
            &pipeline_fingerprint()?,
        )
    }

    fn key_with_pipeline_fingerprint(
        request: CacheSourceIdentity<'_>,
        pipeline_fingerprint: &str,
    ) -> Result<String> {
        let material = serde_json::to_vec(&(
            RESULT_CACHE_SCHEMA,
            BUILD_FINGERPRINT,
            pipeline_fingerprint,
            request,
        ))
        .context("serialize result-cache identity")?;
        Ok(sha256_hex(&material))
    }

    pub(crate) fn load_image(
        &self,
        request: &ImagePipelineInput,
        surrounding_context: &[ChapterContextUnit],
    ) -> Result<Option<CachedImageJob>> {
        let pipeline_fingerprint = pipeline_fingerprint()?;
        let key = Self::key_with_pipeline_fingerprint(
            image_cache_identity(request, surrounding_context),
            &pipeline_fingerprint,
        )?;
        let Some(stored) = self.load_stored(&key, &pipeline_fingerprint)? else {
            return Ok(None);
        };

        let StoredChapterResult::Image {
            regions: stored_regions,
            preserved,
        } = stored.result
        else {
            bail!("result cache modality does not match the image request");
        };

        let mut regions = Vec::with_capacity(stored_regions.len());
        let mut decoded_patch_bytes = 0_u64;
        for stored_region in stored_regions {
            stored_region
                .region
                .validate()
                .context("validate cached translated region")?;
            stored_region
                .lookup_context
                .validate_against(&stored_region.region)
                .context("validate cached lookup context")?;
            let decoded_upper_bound =
                base64_decoded_upper_bound(stored_region.patch_png_base64.len())?;
            if decoded_upper_bound
                > self
                    .max_decoded_patch_bytes
                    .saturating_sub(decoded_patch_bytes)
            {
                bail!(
                    "cached PNG patches exceed the {} byte decoded limit",
                    self.max_decoded_patch_bytes
                );
            }
            let patch_png = BASE64
                .decode(stored_region.patch_png_base64)
                .context("decode cached PNG patch")?;
            decoded_patch_bytes = decoded_patch_bytes
                .checked_add(
                    u64::try_from(patch_png.len()).context("decoded PNG length overflowed")?,
                )
                .context("aggregate decoded PNG length overflowed")?;
            if decoded_patch_bytes > self.max_decoded_patch_bytes {
                bail!(
                    "cached PNG patches exceed the {} byte decoded limit",
                    self.max_decoded_patch_bytes
                );
            }
            validate_cached_png(&patch_png, self.max_decoded_patch_bytes)?;
            regions.push(CachedImageRegion {
                region: stored_region.region,
                lookup_context: stored_region.lookup_context,
                patch_png: Arc::from(patch_png),
            });
        }
        let cached = CachedImageJob { regions, preserved };
        cached.validate()?;
        Ok(Some(cached))
    }

    pub(crate) fn load_document(
        &self,
        request: &DocumentJobRequest,
    ) -> Result<Option<CachedDocumentJob>> {
        let pipeline_fingerprint = pipeline_fingerprint()?;
        let key = Self::key_with_pipeline_fingerprint(
            document_cache_identity(request),
            &pipeline_fingerprint,
        )?;
        let Some(stored) = self.load_stored(&key, &pipeline_fingerprint)? else {
            return Ok(None);
        };
        let StoredChapterResult::Document {
            blocks,
            preserved,
            lookup_contexts,
        } = stored.result
        else {
            bail!("result cache modality does not match the document request");
        };
        let cached = CachedDocumentJob {
            blocks,
            preserved,
            lookup_contexts: lookup_contexts
                .into_iter()
                .map(|stored| (stored.item_id, stored.context))
                .collect(),
        };
        cached.validate_against(request)?;
        Ok(Some(cached))
    }

    fn load_stored(&self, key: &str, pipeline_fingerprint: &str) -> Result<Option<StoredJob>> {
        let path = self.entry_path(key);
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| format!("open result cache {}", path.display()));
            }
        };
        let metadata = file
            .metadata()
            .with_context(|| format!("inspect result cache {}", path.display()))?;
        if !metadata.is_file() || metadata.len() > self.max_entry_bytes {
            bail!("result cache entry is not a bounded regular file");
        }
        let mut bytes = Vec::with_capacity(
            usize::try_from(metadata.len()).context("result cache entry does not fit in memory")?,
        );
        BufReader::new(file)
            .take(self.max_entry_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .with_context(|| format!("read result cache {}", path.display()))?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > self.max_entry_bytes {
            bail!("result cache entry exceeds its byte limit");
        }
        let stored: StoredJob = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse result cache {}", path.display()))?;
        if stored.schema != RESULT_CACHE_SCHEMA
            || stored.build_fingerprint != BUILD_FINGERPRINT
            || stored.pipeline_fingerprint != pipeline_fingerprint
            || stored.key != key
        {
            bail!("result cache identity does not match the current build");
        }
        Ok(Some(stored))
    }

    pub(crate) fn invalidate_image(
        &self,
        request: &ImagePipelineInput,
        surrounding_context: &[ChapterContextUnit],
    ) -> Result<()> {
        let path = self.entry_path(&Self::key_image(request, surrounding_context)?);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => {
                Err(error).with_context(|| format!("invalidate result cache {}", path.display()))
            }
        }
    }

    pub(crate) fn invalidate_document(&self, request: &DocumentJobRequest) -> Result<()> {
        let path = self.entry_path(&Self::key_document(request)?);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => {
                Err(error).with_context(|| format!("invalidate result cache {}", path.display()))
            }
        }
    }

    /// Persist one complete job in a single atomic rename. Callers invoke this
    /// only after visible processing has finished; no tile/OCR/translation
    /// phase performs synchronous intermediate writes.
    pub(crate) fn store_image(
        &self,
        request: &ImagePipelineInput,
        surrounding_context: &[ChapterContextUnit],
        job: &CachedImageJob,
    ) -> Result<()> {
        job.validate().context("validate completed result cache")?;
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create result cache {}", self.root.display()))?;
        let pipeline_fingerprint = pipeline_fingerprint()?;
        let key = Self::key_with_pipeline_fingerprint(
            image_cache_identity(request, surrounding_context),
            &pipeline_fingerprint,
        )?;
        let mut decoded_patch_bytes = 0_u64;
        let regions = job
            .regions
            .iter()
            .map(|cached| {
                cached
                    .region
                    .validate()
                    .context("validate translated region before caching")?;
                cached
                    .lookup_context
                    .validate_against(&cached.region)
                    .context("validate lookup context before caching")?;
                validate_png(&cached.patch_png)?;
                decoded_patch_bytes = decoded_patch_bytes
                    .checked_add(
                        u64::try_from(cached.patch_png.len())
                            .context("PNG patch length overflowed")?,
                    )
                    .context("aggregate PNG patch length overflowed")?;
                if decoded_patch_bytes > self.max_decoded_patch_bytes {
                    bail!(
                        "PNG patches exceed the {} byte decoded cache limit",
                        self.max_decoded_patch_bytes
                    );
                }
                Ok(StoredRegion {
                    region: cached.region.clone(),
                    lookup_context: cached.lookup_context.clone(),
                    patch_png_base64: BASE64.encode(cached.patch_png.as_ref()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stored = StoredJob {
            schema: RESULT_CACHE_SCHEMA.to_owned(),
            build_fingerprint: BUILD_FINGERPRINT.to_owned(),
            pipeline_fingerprint,
            key: key.clone(),
            result: StoredChapterResult::Image {
                regions,
                preserved: job.preserved.clone(),
            },
        };

        self.store_stored(stored)
    }

    pub(crate) fn store_document(
        &self,
        request: &DocumentJobRequest,
        job: &CachedDocumentJob,
    ) -> Result<()> {
        job.validate_against(request)
            .context("validate completed document cache")?;
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create result cache {}", self.root.display()))?;
        let pipeline_fingerprint = pipeline_fingerprint()?;
        let key = Self::key_with_pipeline_fingerprint(
            document_cache_identity(request),
            &pipeline_fingerprint,
        )?;
        self.store_stored(StoredJob {
            schema: RESULT_CACHE_SCHEMA.to_owned(),
            build_fingerprint: BUILD_FINGERPRINT.to_owned(),
            pipeline_fingerprint,
            key,
            result: StoredChapterResult::Document {
                blocks: job.blocks.clone(),
                preserved: job.preserved.clone(),
                lookup_contexts: job
                    .lookup_contexts
                    .iter()
                    .map(|(item_id, context)| StoredLookupContext {
                        item_id: item_id.clone(),
                        context: context.clone(),
                    })
                    .collect(),
            },
        })
    }

    fn store_stored(&self, stored: StoredJob) -> Result<()> {
        let target = self.entry_path(&stored.key);

        let mut temporary = NamedTempFile::new_in(&self.root)
            .with_context(|| format!("create atomic cache file in {}", self.root.display()))?;
        {
            let mut writer = BufWriter::new(temporary.as_file_mut());
            serde_json::to_writer(&mut writer, &stored)
                .context("serialize completed result cache")?;
            writer.flush().context("flush completed result cache")?;
        }
        let serialized_bytes = temporary
            .as_file()
            .metadata()
            .context("inspect completed result cache")?
            .len();
        if serialized_bytes > self.max_entry_bytes || serialized_bytes > self.max_bytes {
            bail!("completed result exceeds the persistent cache entry limit");
        }
        temporary
            .as_file()
            .sync_all()
            .context("sync completed result cache")?;
        temporary
            .persist(&target)
            .map_err(|error| error.error)
            .with_context(|| format!("install result cache {}", target.display()))?;
        self.prune_to_limit(Some(&target))
    }

    fn entry_path(&self, key: &str) -> PathBuf {
        self.root.join(format!("{key}.json"))
    }

    fn prune_to_limit(&self, protected: Option<&Path>) -> Result<()> {
        let directory = match fs::read_dir(&self.root) {
            Ok(directory) => directory,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("scan result cache {}", self.root.display()));
            }
        };
        let mut entries = Vec::new();
        for entry in directory {
            let entry =
                entry.with_context(|| format!("read result cache {}", self.root.display()))?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let metadata = entry
                .metadata()
                .with_context(|| format!("inspect result cache {}", path.display()))?;
            if !metadata.is_file() {
                continue;
            }
            let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            entries.push((path, metadata.len(), modified));
        }
        let mut total = entries
            .iter()
            .fold(0_u64, |sum, (_, bytes, _)| sum.saturating_add(*bytes));
        entries.sort_by_key(|(_, _, modified)| *modified);

        for (path, bytes, _) in entries {
            if total <= self.max_bytes {
                break;
            }
            if protected.is_some_and(|protected| protected == path) {
                continue;
            }
            match fs::remove_file(&path) {
                Ok(()) => total = total.saturating_sub(bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("evict result cache {}", path.display()));
                }
            }
        }
        if total > self.max_bytes {
            bail!("completed result exceeds the 2 GiB persistent cache limit");
        }
        Ok(())
    }
}

fn pipeline_fingerprint() -> Result<String> {
    let model_resources = sha256_hex(MODEL_RESOURCE_MANIFEST);
    let material = serde_json::to_vec(&(
        RESULT_CACHE_PIPELINE_REVISION,
        model_resources,
        HSK_TRANSLATION_MODEL_REVISION,
        HSK_TRANSLATION_PROMPT_HASH,
        HSK_TRANSLATION_VALIDATOR_HASH,
        (HSK_RESOURCE_BYTES, HSK_RESOURCE_SHA256),
        (DICTIONARY_RESOURCE_BYTES, DICTIONARY_RESOURCE_SHA256),
        (
            NORMALIZATION_REVISION,
            SEGMENTATION_REVISION,
            LOOKUP_REVISION,
            JIEBA_CRATE_VERSION,
            JIEBA_EMBEDDED_DICTIONARY_SHA256,
            UNICODE_NORMALIZATION_CRATE_VERSION,
            UNICODE_NORMALIZATION_TABLES_SHA256,
        ),
    ))
    .context("serialize result-cache pipeline fingerprint")?;
    Ok(sha256_hex(&material))
}

fn base64_decoded_upper_bound(encoded_len: usize) -> Result<u64> {
    let encoded_len = u64::try_from(encoded_len).context("base64 length does not fit in u64")?;
    encoded_len
        .checked_add(3)
        .and_then(|length| length.checked_div(4))
        .and_then(|quartets| quartets.checked_mul(3))
        .context("base64 decoded length overflowed")
}

fn validate_png(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 8 || bytes[..8] != [137, 80, 78, 71, 13, 10, 26, 10] {
        bail!("cached patch is not a PNG");
    }
    Ok(())
}

fn validate_cached_png(bytes: &[u8], max_decoded_bytes: u64) -> Result<()> {
    let mut limits = Limits::default();
    limits.max_image_width = Some(16_384);
    limits.max_image_height = Some(16_384);
    limits.max_alloc = Some(max_decoded_bytes);
    let mut reader = ImageReader::new(Cursor::new(bytes));
    reader.set_format(ImageFormat::Png);
    reader.limits(limits);
    reader
        .decode()
        .context("decode cached PNG patch within safe limits")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{
        DocumentJobRequest, HskLevel, JobUpdate, JobUpdatesResponse, LearningMode, NormalizedRect,
    };
    use hsk_control::{ProperName, ProperNameReason};
    use image::{DynamicImage, ImageFormat};

    fn image_request() -> Result<ImagePipelineInput> {
        Ok(
            serde_json::from_str::<crate::contracts::ImageJobRequest>(include_str!(
                "../../../fixtures/contracts/job-request.valid.json"
            ))?
            .pipeline_input(),
        )
    }

    fn document_request() -> Result<DocumentJobRequest> {
        Ok(serde_json::from_str(include_str!(
            "../../../fixtures/contracts/document-job-request.valid.json"
        ))?)
    }

    fn image_region() -> ImageRegionReady {
        let response: JobUpdatesResponse = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-updates.success.json"
        ))
        .expect("valid image update fixture");
        response
            .updates
            .into_iter()
            .find_map(|update| match update {
                JobUpdate::ImageRegionReady { region, .. } => Some(*region),
                _ => None,
            })
            .expect("image fixture has a translated region")
    }

    fn document_job() -> CachedDocumentJob {
        let response: JobUpdatesResponse = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/document-updates.success.json"
        ))
        .expect("valid document update fixture");
        let mut blocks = Vec::new();
        let mut preserved = Vec::new();
        for update in response.updates {
            match update {
                JobUpdate::DocumentBlockReady { block, .. } => blocks.push(block),
                JobUpdate::DocumentBlockPreserved { block, .. } => preserved.push(block),
                _ => {}
            }
        }
        let lookup_contexts = blocks
            .iter()
            .map(|block| {
                (
                    block.item_id.clone(),
                    ItemLookupContext {
                        source_text: block.text.source_text.clone(),
                        base_chinese: block.text.base_chinese.clone(),
                        displayed_chinese: block.text.displayed_chinese.clone(),
                        proper_names: Vec::new(),
                    },
                )
            })
            .chain(preserved.iter().map(|block| {
                (
                    block.item_id.clone(),
                    ItemLookupContext {
                        source_text: block.source_text.clone(),
                        base_chinese: block.source_text.clone(),
                        displayed_chinese: block.source_text.clone(),
                        proper_names: Vec::new(),
                    },
                )
            }))
            .collect();
        CachedDocumentJob {
            blocks,
            preserved,
            lookup_contexts,
        }
    }

    fn lookup_context(region: &ImageRegionReady) -> ItemLookupContext {
        ItemLookupContext {
            source_text: region.text.source_text.clone(),
            base_chinese: region.text.base_chinese.clone(),
            displayed_chinese: region.text.displayed_chinese.clone(),
            proper_names: vec![ProperName {
                text: "小明".to_owned(),
                reason: ProperNameReason::PersonName,
            }],
        }
    }

    fn png() -> Arc<[u8]> {
        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::new_rgba8(2, 2)
            .write_to(&mut cursor, ImageFormat::Png)
            .expect("encode test PNG");
        Arc::from(cursor.into_inner())
    }

    fn image_job() -> CachedImageJob {
        let region = image_region();
        CachedImageJob {
            regions: vec![CachedImageRegion {
                lookup_context: lookup_context(&region),
                region,
                patch_png: png(),
            }],
            preserved: Vec::new(),
        }
    }

    fn context(text: &str) -> ChapterContextUnit {
        ChapterContextUnit {
            source_index: 0,
            item_order: 0,
            item_id: "context-0".to_owned(),
            source_text: text.to_owned(),
            displayed_text: Some("上下文".to_owned()),
        }
    }

    #[test]
    fn image_key_excludes_transport_identity_and_includes_real_context() -> Result<()> {
        let first_wire: crate::contracts::ImageJobRequest = serde_json::from_str(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))?;
        let mut transport_only = first_wire.clone();
        transport_only.client_image_id = "different-dom-image".to_owned();
        transport_only.page_session_id = "different-session".to_owned();
        transport_only.visible_rects = vec![NormalizedRect {
            x: 0.25,
            y: 0.25,
            width: 0.5,
            height: 0.5,
        }];
        let first = first_wire.pipeline_input();
        let same = transport_only.pipeline_input();

        assert_eq!(
            ResultCache::key_image(&first, &[context("before")])?,
            ResultCache::key_image(&same, &[context("before")])?
        );
        assert_ne!(
            ResultCache::key_image(&first, &[context("before")])?,
            ResultCache::key_image(&same, &[context("changed")])?
        );

        let mut strict = same;
        strict.settings.learning_mode = LearningMode::Strict;
        strict.settings.hsk_level = HskLevel::Three;
        assert_ne!(
            ResultCache::key_image(&first, &[context("before")])?,
            ResultCache::key_image(&strict, &[context("before")])?
        );
        Ok(())
    }

    #[test]
    fn document_key_excludes_session_and_includes_ordered_source_and_settings() -> Result<()> {
        let first = document_request()?;
        let mut same = first.clone();
        same.page_session_id = "another-session".to_owned();
        assert_eq!(
            ResultCache::key_document(&first)?,
            ResultCache::key_document(&same)?
        );

        same.blocks[1].text.push_str(" Changed.");
        same.source_sha256 = crate::contracts::canonical_document_sha256(&same.blocks);
        assert_ne!(
            ResultCache::key_document(&first)?,
            ResultCache::key_document(&same)?
        );

        let mut strict = first.clone();
        strict.settings.learning_mode = LearningMode::Strict;
        assert_ne!(
            ResultCache::key_document(&first)?,
            ResultCache::key_document(&strict)?
        );
        Ok(())
    }

    #[test]
    fn tagged_image_and_document_entries_round_trip() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let cache = ResultCache::new(directory.path().to_path_buf());
        let image_request = image_request()?;
        let image_context = [context("before")];
        let image = image_job();
        cache.store_image(&image_request, &image_context, &image)?;
        let loaded_image = cache
            .load_image(&image_request, &image_context)?
            .expect("image cache hit");
        assert_eq!(loaded_image.regions.len(), 1);
        assert_eq!(
            loaded_image.regions[0].region.item_id,
            image.regions[0].region.item_id
        );
        assert_eq!(loaded_image.regions[0].patch_png.as_ref(), png().as_ref());

        let document_request = document_request()?;
        let document = document_job();
        cache.store_document(&document_request, &document)?;
        let loaded_document = cache
            .load_document(&document_request)?
            .expect("document cache hit");
        assert_eq!(loaded_document.blocks.len(), 1);
        assert_eq!(loaded_document.preserved.len(), 1);
        assert_eq!(loaded_document.lookup_contexts.len(), 2);
        Ok(())
    }

    #[test]
    fn document_cache_rejects_missing_reordered_or_source_mismatched_terminals() -> Result<()> {
        let request = document_request()?;
        let mut missing = document_job();
        missing.preserved.clear();
        missing
            .lookup_contexts
            .retain(|(item_id, _)| item_id != "block-1");
        assert!(missing.validate_against(&request).is_err());

        let mut wrong_position = document_job();
        wrong_position.blocks[0].source_index = 99;
        assert!(wrong_position.validate_against(&request).is_err());

        let mut wrong_source = document_job();
        wrong_source.blocks[0].text.source_text = "Different source".to_owned();
        assert!(wrong_source.validate_against(&request).is_err());

        let directory = tempfile::tempdir()?;
        let cache = ResultCache::new(directory.path().to_path_buf());
        assert!(cache.store_document(&request, &wrong_source).is_err());

        let valid = document_job();
        cache.store_document(&request, &valid)?;
        let entry = cache.entry_path(&ResultCache::key_document(&request)?);
        let mut stored: StoredJob = serde_json::from_slice(&fs::read(&entry)?)?;
        let StoredChapterResult::Document { blocks, .. } = &mut stored.result else {
            panic!("expected document cache entry");
        };
        blocks[0].text.source_text = "Tampered source".to_owned();
        fs::write(&entry, serde_json::to_vec(&stored)?)?;
        assert!(cache.load_document(&request).is_err());
        Ok(())
    }

    #[test]
    fn tagged_modality_mismatch_is_rejected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let cache = ResultCache::new(directory.path().to_path_buf());
        let image_request = image_request()?;
        let document_request = document_request()?;
        let pipeline_fingerprint = pipeline_fingerprint()?;
        let image_key = ResultCache::key_document(&document_request)?;
        fs::create_dir_all(directory.path())?;
        let stored = StoredJob {
            schema: RESULT_CACHE_SCHEMA.to_owned(),
            build_fingerprint: BUILD_FINGERPRINT.to_owned(),
            pipeline_fingerprint,
            key: image_key.clone(),
            result: StoredChapterResult::Image {
                regions: Vec::new(),
                preserved: Vec::new(),
            },
        };
        fs::write(cache.entry_path(&image_key), serde_json::to_vec(&stored)?)?;

        assert!(cache.load_document(&document_request).is_err());
        assert!(cache.load_image(&image_request, &[])?.is_none());
        Ok(())
    }

    #[test]
    fn bounded_cache_rejects_oversize_entries_and_decoded_patches() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let image_request = image_request()?;
        let image = image_job();
        let tiny = ResultCache::with_limit(directory.path().to_path_buf(), 1);
        assert!(tiny.store_image(&image_request, &[], &image).is_err());

        let directory = tempfile::tempdir()?;
        let cache = ResultCache::new(directory.path().to_path_buf());
        cache.store_image(&image_request, &[], &image)?;
        let entry_bytes =
            fs::metadata(cache.entry_path(&ResultCache::key_image(&image_request, &[])?))?.len();
        let file_bounded =
            ResultCache::with_load_limits(directory.path().to_path_buf(), entry_bytes - 1, 1024);
        assert!(file_bounded.load_image(&image_request, &[]).is_err());
        let patch_bounded =
            ResultCache::with_load_limits(directory.path().to_path_buf(), 1024 * 1024, 4);
        assert!(patch_bounded.load_image(&image_request, &[]).is_err());
        Ok(())
    }

    #[test]
    fn miss_never_scans_or_mutates_unrelated_files() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let orphan = directory.path().join("orphan.json");
        fs::write(&orphan, b"unrelated")?;
        let cache = ResultCache::with_limit(directory.path().to_path_buf(), 4);

        assert!(cache.load_image(&image_request()?, &[])?.is_none());
        assert!(orphan.exists());
        Ok(())
    }

    #[test]
    fn key_changes_with_pipeline_fingerprint() -> Result<()> {
        let request = document_request()?;
        assert_ne!(
            ResultCache::key_with_pipeline_fingerprint(
                document_cache_identity(&request),
                "pipeline-a"
            )?,
            ResultCache::key_with_pipeline_fingerprint(
                document_cache_identity(&request),
                "pipeline-b"
            )?
        );
        Ok(())
    }
}
