use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use koharu_app::llm::{
    HSK_TRANSLATION_MODEL_REVISION, HSK_TRANSLATION_PROMPT_HASH, HSK_TRANSLATION_VALIDATOR_HASH,
};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::contracts::{BUILD_FINGERPRINT, BrowserJobRequest, ProgressiveRegion, Validate};
use crate::crypto::sha256_hex;
use crate::pipeline_adapter::RegionLookupContext;
use crate::setup::{
    DICTIONARY_RESOURCE_BYTES, DICTIONARY_RESOURCE_SHA256, HSK_RESOURCE_BYTES, HSK_RESOURCE_SHA256,
};

pub(crate) const RESULT_CACHE_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const RESULT_CACHE_MAX_ENTRY_BYTES: u64 = 512 * 1024 * 1024;
const RESULT_CACHE_MAX_DECODED_PATCH_BYTES: u64 = 256 * 1024 * 1024;
const RESULT_CACHE_SCHEMA: &str = "hskify-progressive-result-2026-07-27-v5";
const RESULT_CACHE_PIPELINE_REVISION: &str =
    "direct-browser-pipeline-segmentation-recall-furniture-verifier-v18-2026-07-27";
const MODEL_RESOURCE_MANIFEST: &[u8] = include_bytes!("../../../data/model-packs/manifest.v1.json");

#[derive(Debug, Clone)]
pub(crate) struct CachedRegion {
    pub region: ProgressiveRegion,
    pub lookup_context: RegionLookupContext,
    pub patch_png: Vec<u8>,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedJob {
    pub regions: Vec<CachedRegion>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredJob {
    schema: String,
    build_fingerprint: String,
    pipeline_fingerprint: String,
    key: String,
    regions: Vec<StoredRegion>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredRegion {
    region: ProgressiveRegion,
    lookup_context: RegionLookupContext,
    patch_png_base64: String,
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

    pub(crate) fn key(request: &BrowserJobRequest) -> Result<String> {
        Self::key_with_pipeline_fingerprint(request, &pipeline_fingerprint()?)
    }

    fn key_with_pipeline_fingerprint(
        request: &BrowserJobRequest,
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

    pub(crate) fn load(&self, request: &BrowserJobRequest) -> Result<Option<CachedJob>> {
        let pipeline_fingerprint = pipeline_fingerprint()?;
        let key = Self::key_with_pipeline_fingerprint(request, &pipeline_fingerprint)?;
        let path = self.entry_path(&key);
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
        if !metadata.is_file() {
            bail!("result cache entry is not a regular file");
        }
        if metadata.len() > self.max_entry_bytes {
            bail!(
                "result cache entry exceeds the {} byte file limit",
                self.max_entry_bytes
            );
        }
        let mut bytes = Vec::with_capacity(
            usize::try_from(metadata.len()).context("result cache entry does not fit in memory")?,
        );
        BufReader::new(file)
            .take(self.max_entry_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .with_context(|| format!("read result cache {}", path.display()))?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > self.max_entry_bytes {
            bail!(
                "result cache entry exceeds the {} byte file limit",
                self.max_entry_bytes
            );
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

        let mut regions = Vec::with_capacity(stored.regions.len());
        let mut decoded_patch_bytes = 0_u64;
        for stored_region in stored.regions {
            stored_region
                .region
                .validate()
                .context("validate cached progressive region")?;
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
            validate_png(&patch_png)?;
            regions.push(CachedRegion {
                region: stored_region.region,
                lookup_context: stored_region.lookup_context,
                patch_png,
            });
        }
        Ok(Some(CachedJob { regions }))
    }

    pub(crate) fn invalidate(&self, request: &BrowserJobRequest) -> Result<()> {
        let path = self.entry_path(&Self::key(request)?);
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
    pub(crate) fn store(&self, request: &BrowserJobRequest, job: &CachedJob) -> Result<()> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create result cache {}", self.root.display()))?;
        let pipeline_fingerprint = pipeline_fingerprint()?;
        let key = Self::key_with_pipeline_fingerprint(request, &pipeline_fingerprint)?;
        let target = self.entry_path(&key);
        let mut decoded_patch_bytes = 0_u64;
        let regions = job
            .regions
            .iter()
            .map(|cached| {
                cached
                    .region
                    .validate()
                    .context("validate progressive region before caching")?;
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
                    patch_png_base64: BASE64.encode(&cached.patch_png),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stored = StoredJob {
            schema: RESULT_CACHE_SCHEMA.to_owned(),
            build_fingerprint: BUILD_FINGERPRINT.to_owned(),
            pipeline_fingerprint,
            key: key.clone(),
            regions,
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{
        BrowserTextLayout, BrowserTextStyle, FontCategory, HskLevel, HskRepairState,
        NormalizedRect, PatchMimeType, Point, ProgressiveHskStatus, RegionPatch, TextAlignment,
        WritingMode,
    };
    use hsk_control::{ProperName, ProperNameReason};

    fn region(id: &str) -> ProgressiveRegion {
        let polygon = vec![
            Point { x: 0.1, y: 0.1 },
            Point { x: 0.2, y: 0.1 },
            Point { x: 0.2, y: 0.2 },
        ];
        ProgressiveRegion {
            id: id.to_owned(),
            text_polygon: polygon.clone(),
            bubble_polygon: Some(polygon),
            patch: RegionPatch {
                blob_id: format!("patch-{id}"),
                mime_type: PatchMimeType::Png,
                rect: NormalizedRect {
                    x: 0.1,
                    y: 0.1,
                    width: 0.1,
                    height: 0.1,
                },
            },
            source_english: "Hello".to_owned(),
            base_chinese: "\u{4f60}\u{597d}".to_owned(),
            displayed_chinese: "\u{4f60}\u{597d}".to_owned(),
            pinyin: "n\u{01d0} h\u{01ce}o".to_owned(),
            ocr_confidence: 0.99,
            reading_order: 0,
            style: BrowserTextStyle {
                font_id: "hmt-sans".to_owned(),
                category: FontCategory::Sans,
                weight: 600,
                italic_degrees: 0.0,
                foreground: "#000".to_owned(),
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
                suggested_lines: vec!["\u{4f60}\u{597d}".to_owned()],
                font_size_to_image_width: 0.02,
                safe_polygon: None,
            },
            hsk: ProgressiveHskStatus {
                requested_level: HskLevel::Two,
                strictly_valid: true,
                above_level_tokens: Vec::new(),
                repair_state: HskRepairState::NotNeeded,
            },
        }
    }

    fn png() -> Vec<u8> {
        vec![137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3, 4]
    }

    fn lookup_context() -> RegionLookupContext {
        RegionLookupContext {
            source_english: "Hello".to_owned(),
            base_chinese: "\u{4f60}\u{597d}".to_owned(),
            displayed_chinese: "\u{4f60}\u{597d}".to_owned(),
            proper_names: vec![ProperName {
                text: "\u{5c0f}\u{660e}".to_owned(),
                reason: ProperNameReason::PersonName,
            }],
        }
    }

    fn request() -> Result<BrowserJobRequest> {
        Ok(
            serde_json::from_str::<crate::contracts::CreateJobRequest>(include_str!(
                "../../../fixtures/contracts/job-request.valid.json"
            ))?
            .pipeline_request(),
        )
    }

    #[test]
    fn key_scopes_chapter_entity_memory_but_ignores_dom_and_viewport_identity() -> Result<()> {
        let first = serde_json::from_str::<crate::contracts::CreateJobRequest>(include_str!(
            "../../../fixtures/contracts/job-request.valid.json"
        ))?;
        let mut same_chapter = first.clone();
        same_chapter.client_image_id = "different-dom-image".to_owned();
        same_chapter.page_index = same_chapter.page_index.saturating_add(7);
        same_chapter.visible_rects = vec![NormalizedRect {
            x: 0.25,
            y: 0.25,
            width: 0.5,
            height: 0.5,
        }];
        let mut other_chapter = same_chapter.clone();
        other_chapter.page_session_id = "different-page-session".to_owned();

        assert_eq!(
            ResultCache::key(&first.pipeline_request())?,
            ResultCache::key(&same_chapter.pipeline_request())?
        );
        assert_ne!(
            ResultCache::key(&first.pipeline_request())?,
            ResultCache::key(&other_chapter.pipeline_request())?
        );
        Ok(())
    }

    #[test]
    fn atomic_round_trip_uses_exact_request_identity() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let cache = ResultCache::new(directory.path().to_path_buf());
        let request = request()?;
        let job = CachedJob {
            regions: vec![CachedRegion {
                region: region("a"),
                lookup_context: lookup_context(),
                patch_png: png(),
            }],
        };

        cache.store(&request, &job)?;
        cache.store(&request, &job)?;
        let loaded = cache.load(&request)?.expect("cache hit");

        assert_eq!(loaded.regions.len(), 1);
        assert_eq!(loaded.regions[0].region.id, "a");
        assert_eq!(loaded.regions[0].lookup_context, lookup_context());
        assert_eq!(loaded.regions[0].patch_png, png());
        Ok(())
    }

    #[test]
    fn byte_limit_evicts_old_entries() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let cache = ResultCache::with_limit(directory.path().to_path_buf(), 1);
        let request = request()?;
        let job = CachedJob {
            regions: vec![CachedRegion {
                region: region("a"),
                lookup_context: lookup_context(),
                patch_png: png(),
            }],
        };

        assert!(cache.store(&request, &job).is_err());
        Ok(())
    }

    #[test]
    fn load_rejects_an_entry_larger_than_its_file_bound() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let request = request()?;
        let cache = ResultCache::new(directory.path().to_path_buf());
        cache.store(
            &request,
            &CachedJob {
                regions: vec![CachedRegion {
                    region: region("a"),
                    lookup_context: lookup_context(),
                    patch_png: png(),
                }],
            },
        )?;
        let entry_bytes = fs::metadata(cache.entry_path(&ResultCache::key(&request)?))?.len();
        let bounded =
            ResultCache::with_load_limits(directory.path().to_path_buf(), entry_bytes - 1, 1024);

        assert!(bounded.load(&request).is_err());
        Ok(())
    }

    #[test]
    fn load_bounds_aggregate_decoded_patch_bytes_before_decode() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let request = request()?;
        let cache = ResultCache::new(directory.path().to_path_buf());
        cache.store(
            &request,
            &CachedJob {
                regions: vec![
                    CachedRegion {
                        region: region("a"),
                        lookup_context: lookup_context(),
                        patch_png: png(),
                    },
                    CachedRegion {
                        region: region("b"),
                        lookup_context: lookup_context(),
                        patch_png: png(),
                    },
                ],
            },
        )?;
        let bounded =
            ResultCache::with_load_limits(directory.path().to_path_buf(), 1024 * 1024, 12);

        assert!(bounded.load(&request).is_err());
        Ok(())
    }

    #[test]
    fn load_miss_does_not_scan_or_mutate_unrelated_entries() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let orphan = directory.path().join("orphan.json");
        fs::write(&orphan, b"oversized")?;
        let cache = ResultCache::with_limit(directory.path().to_path_buf(), 4);

        assert!(cache.load(&request()?)?.is_none());
        assert!(orphan.exists());
        Ok(())
    }

    #[test]
    fn key_changes_with_the_pipeline_fingerprint() -> Result<()> {
        let request = request()?;

        assert_ne!(
            ResultCache::key_with_pipeline_fingerprint(&request, "pipeline-a")?,
            ResultCache::key_with_pipeline_fingerprint(&request, "pipeline-b")?
        );
        Ok(())
    }
}
