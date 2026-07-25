use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result, anyhow, bail};
use koharu_core::{DownloadProgress, DownloadStatus};
use koharu_runtime::{ComputePolicy, RuntimeManager};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::contracts::{BrowserSetupState, BrowserSetupStatus, Validate};

const MODEL_MANIFEST: &str = include_str!("../../../data/model-packs/manifest.v1.json");
const RESOURCES_DIRECTORY_ENV: &str = "HSK_MANGA_RESOURCES_DIR";
const HSK_RESOURCE_ENV: &str = "HSK_MANGA_HSK_PATH";
const DICTIONARY_RESOURCE_ENV: &str = "HSK_MANGA_DICTIONARY_PATH";
const QWEN_RESOURCE_ENV: &str = "HSK_MANGA_QWEN_MODEL_PATH";
const HSK_RESOURCE_FILE: &str = "hsk-2.0.normalized.json";
const DICTIONARY_RESOURCE_FILE: &str = "cc-cedict.normalized.json";
const EXPECTED_MODEL_FILE: &str = "Qwen3.5-4B-Q4_K_M.gguf";
const SANS_FONT_FILE: &str = "NotoSansSC-VF.ttf";
const SERIF_FONT_FILE: &str = "NotoSerifSC-VF.ttf";

#[derive(Debug, Clone)]
pub(crate) struct ManagedResourcePaths {
    pub(crate) hsk: PathBuf,
    pub(crate) dictionary: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) fonts: PathBuf,
}

impl ManagedResourcePaths {
    pub(crate) fn discover() -> Result<Self> {
        let root = nonempty_env_path(RESOURCES_DIRECTORY_ENV)
            .or_else(default_resource_root)
            .context(
                "cannot determine the per-user resource directory; set HSK_MANGA_RESOURCES_DIR",
            )?;
        Ok(Self {
            hsk: nonempty_env_path(HSK_RESOURCE_ENV)
                .unwrap_or_else(|| root.join(HSK_RESOURCE_FILE)),
            dictionary: nonempty_env_path(DICTIONARY_RESOURCE_ENV)
                .unwrap_or_else(|| root.join(DICTIONARY_RESOURCE_FILE)),
            model: nonempty_env_path(QWEN_RESOURCE_ENV)
                .unwrap_or_else(|| root.join("models").join(EXPECTED_MODEL_FILE)),
            fonts: root.join("fonts"),
        })
    }

    pub(crate) fn language_data_present(&self) -> bool {
        self.hsk.is_file() && self.dictionary.is_file()
    }
}

fn nonempty_env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn default_resource_root() -> Option<PathBuf> {
    env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(dirs::data_local_dir)
        .map(|root| {
            root.join("Hskify")
                .join("HSKMangaTranslator")
                .join("resources")
        })
}

#[derive(Debug, Clone)]
struct SelectedModel {
    pack_id: String,
    url: String,
    filename: String,
    sha256: String,
    bytes: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelManifest {
    manifest_version: u8,
    selection: ModelSelection,
    packs: Vec<ModelPack>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelSelection {
    status: String,
    standard_pack_id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelPack {
    id: String,
    runtime_model_id: String,
    files: Vec<ModelFile>,
}

#[derive(Deserialize)]
struct ModelFile {
    id: String,
    url: String,
    filename: String,
    sha256: String,
    bytes: u64,
}

fn selected_model() -> Result<SelectedModel> {
    let manifest: ModelManifest =
        serde_json::from_str(MODEL_MANIFEST).context("parse embedded model pack manifest")?;
    if manifest.manifest_version != 1 || manifest.selection.status != "selected" {
        bail!("embedded model manifest does not select a version-1 pack");
    }
    let mut packs = manifest
        .packs
        .into_iter()
        .filter(|pack| pack.id == manifest.selection.standard_pack_id);
    let pack = packs
        .next()
        .context("selected model pack is missing from the manifest")?;
    if packs.next().is_some() {
        bail!("selected model pack appears more than once");
    }
    if pack.runtime_model_id != "qwen3.5-4b" {
        bail!("selected pack does not use the approved Qwen 4B runtime model");
    }
    let mut files = pack
        .files
        .into_iter()
        .filter(|file| file.id == "translation-model");
    let file = files
        .next()
        .context("selected pack has no translation-model file")?;
    if files.next().is_some() {
        bail!("selected pack has more than one translation-model file");
    }
    if file.filename != EXPECTED_MODEL_FILE
        || file.sha256.len() != 64
        || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        || file.bytes == 0
        || !file.url.starts_with("https://")
    {
        bail!("selected translation model metadata is invalid");
    }
    Ok(SelectedModel {
        pack_id: pack.id,
        url: file.url,
        filename: file.filename,
        sha256: file.sha256.to_ascii_lowercase(),
        bytes: file.bytes,
    })
}

pub(crate) struct ModelSetup {
    resources: ManagedResourcePaths,
    runtime_root: PathBuf,
    selected: SelectedModel,
    active: AtomicBool,
    state: RwLock<Option<BrowserSetupStatus>>,
}

impl ModelSetup {
    pub(crate) fn new(resources: ManagedResourcePaths, cache_root: PathBuf) -> Result<Self> {
        Ok(Self {
            resources,
            runtime_root: cache_root.join("setup-runtime-v1"),
            selected: selected_model()?,
            active: AtomicBool::new(false),
            state: RwLock::new(None),
        })
    }

    pub(crate) fn resources_ready(&self) -> bool {
        self.resources.language_data_present()
            && model_has_expected_size(&self.resources.model, self.selected.bytes)
    }

    pub(crate) fn font_path(&self, font_id: &str) -> Option<PathBuf> {
        let filename = match font_id {
            "hmt-sans" | "hmt-display" => SANS_FONT_FILE,
            "hmt-serif" | "hmt-handwritten" | "hmt-brush" => SERIF_FONT_FILE,
            _ => return None,
        };
        Some(self.resources.fonts.join(filename))
    }

    pub(crate) fn status(&self) -> BrowserSetupStatus {
        if self.resources_ready() {
            return self.ready_status();
        }
        if let Some(status) = self
            .state
            .read()
            .expect("setup state lock poisoned")
            .clone()
            && (self.active.load(Ordering::Acquire) || status.state == BrowserSetupState::Failed)
        {
            return status;
        }
        self.missing_status()
    }

    pub(crate) fn start(self: &Arc<Self>) -> BrowserSetupStatus {
        if self.resources_ready() {
            return self.ready_status();
        }
        if !self.resources.language_data_present() {
            let status = self.failed_status(
                "LANGUAGE_RESOURCES_MISSING",
                "HSK and dictionary data are missing. Reinstall the local engine bundle.",
            );
            self.replace_status(status.clone());
            return status;
        }
        if self
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return self.status();
        }

        let status = BrowserSetupStatus {
            state: BrowserSetupState::Downloading,
            selected_pack_id: Some(self.selected.pack_id.clone()),
            current_file: Some(self.selected.filename.clone()),
            completed_bytes: Some(0),
            total_bytes: Some(self.selected.bytes),
            required_disk_bytes: Some(self.selected.bytes),
            message: format!("Preparing to download {}.", self.selected.filename),
            error_code: None,
        };
        self.replace_status(status.clone());
        let setup = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = setup.download_and_install().await {
                setup.replace_status(setup.failed_status(
                    "MODEL_SETUP_FAILED",
                    &format!("Local model setup failed: {error:#}"),
                ));
            }
            setup.active.store(false, Ordering::Release);
        });
        status
    }

    async fn download_and_install(self: &Arc<Self>) -> Result<()> {
        let model_parent = self
            .resources
            .model
            .parent()
            .context("managed model path has no parent directory")?;
        tokio::fs::create_dir_all(model_parent)
            .await
            .with_context(|| format!("create model directory {}", model_parent.display()))?;
        let available = fs4::available_space(model_parent)
            .with_context(|| format!("inspect free disk space at {}", model_parent.display()))?;
        if available < self.selected.bytes {
            bail!(
                "not enough free disk space: {} bytes available, {} required",
                available,
                self.selected.bytes
            );
        }

        let runtime = RuntimeManager::new(&self.runtime_root, ComputePolicy::PreferGpu)
            .context("initialize Koharu's managed downloader")?;
        let mut progress_rx = runtime.subscribe_downloads();
        let progress_setup = Arc::clone(self);
        let progress_file = self.selected.filename.clone();
        let progress_task = tokio::spawn(async move {
            while let Ok(progress) = progress_rx.recv().await {
                if progress.filename == progress_file {
                    progress_setup.record_download_progress(progress);
                }
            }
        });
        let downloaded = runtime
            .downloads()
            .pinned_url(&self.selected.url, &self.selected.filename)
            .await;
        progress_task.abort();
        let downloaded = downloaded?;

        self.replace_status(BrowserSetupStatus {
            state: BrowserSetupState::Verifying,
            selected_pack_id: Some(self.selected.pack_id.clone()),
            current_file: Some(self.selected.filename.clone()),
            completed_bytes: Some(self.selected.bytes),
            total_bytes: Some(self.selected.bytes),
            required_disk_bytes: Some(self.selected.bytes),
            message: format!("Verifying {}.", self.selected.filename),
            error_code: None,
        });
        let selected = self.selected.clone();
        let verify_path = downloaded.clone();
        if let Err(error) =
            tokio::task::spawn_blocking(move || verify_model_file(&verify_path, &selected))
                .await
                .context("join model verification task")?
        {
            tokio::fs::remove_file(&downloaded).await.ok();
            return Err(error);
        }

        if self.resources.model.is_dir() {
            bail!(
                "managed model destination is a directory: {}",
                self.resources.model.display()
            );
        }
        if self.resources.model.exists() {
            tokio::fs::remove_file(&self.resources.model)
                .await
                .with_context(|| {
                    format!(
                        "remove invalid managed model {}",
                        self.resources.model.display()
                    )
                })?;
        }
        if let Err(rename_error) = tokio::fs::rename(&downloaded, &self.resources.model).await {
            tokio::fs::copy(&downloaded, &self.resources.model)
                .await
                .with_context(|| {
                    format!(
                        "install verified model {} after rename failed: {rename_error}",
                        self.resources.model.display()
                    )
                })?;
            tokio::fs::remove_file(&downloaded).await.ok();
        }
        self.replace_status(self.ready_status());
        Ok(())
    }

    fn record_download_progress(&self, progress: DownloadProgress) {
        let (state, message, error_code) = match progress.status {
            DownloadStatus::Started | DownloadStatus::Downloading => (
                BrowserSetupState::Downloading,
                format!("Downloading {}.", progress.filename),
                None,
            ),
            DownloadStatus::Completed => (
                BrowserSetupState::Verifying,
                format!("Verifying {}.", progress.filename),
                None,
            ),
            DownloadStatus::Failed { reason } => (
                BrowserSetupState::Failed,
                format!("Model download failed: {reason}"),
                Some("MODEL_DOWNLOAD_FAILED".to_owned()),
            ),
        };
        self.replace_status(BrowserSetupStatus {
            state,
            selected_pack_id: Some(self.selected.pack_id.clone()),
            current_file: Some(progress.filename),
            completed_bytes: Some(progress.downloaded.min(self.selected.bytes)),
            total_bytes: Some(self.selected.bytes),
            required_disk_bytes: Some(self.selected.bytes),
            message,
            error_code,
        });
    }

    fn missing_status(&self) -> BrowserSetupStatus {
        BrowserSetupStatus {
            state: BrowserSetupState::MissingModels,
            selected_pack_id: Some(self.selected.pack_id.clone()),
            current_file: Some(self.selected.filename.clone()),
            completed_bytes: Some(0),
            total_bytes: Some(self.selected.bytes),
            required_disk_bytes: Some(self.selected.bytes),
            message: format!(
                "Download {} ({:.1} GiB) for local translation.",
                self.selected.filename,
                self.selected.bytes as f64 / 1024_f64.powi(3)
            ),
            error_code: None,
        }
    }

    fn ready_status(&self) -> BrowserSetupStatus {
        BrowserSetupStatus {
            state: BrowserSetupState::Ready,
            selected_pack_id: Some(self.selected.pack_id.clone()),
            current_file: None,
            completed_bytes: None,
            total_bytes: None,
            required_disk_bytes: None,
            message: "Local translation and language resources are ready.".to_owned(),
            error_code: None,
        }
    }

    fn failed_status(&self, code: &str, message: &str) -> BrowserSetupStatus {
        BrowserSetupStatus {
            state: BrowserSetupState::Failed,
            selected_pack_id: Some(self.selected.pack_id.clone()),
            current_file: Some(self.selected.filename.clone()),
            completed_bytes: None,
            total_bytes: None,
            required_disk_bytes: Some(self.selected.bytes),
            message: message.to_owned(),
            error_code: Some(code.to_owned()),
        }
    }

    fn replace_status(&self, status: BrowserSetupStatus) {
        debug_assert!(status.validate().is_ok());
        *self.state.write().expect("setup state lock poisoned") = Some(status);
    }
}

fn model_has_expected_size(path: &Path, expected: u64) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() == expected)
}

fn verify_model_file(path: &Path, selected: &SelectedModel) -> Result<()> {
    let metadata = path
        .metadata()
        .with_context(|| format!("inspect downloaded model {}", path.display()))?;
    if !metadata.is_file() || metadata.len() != selected.bytes {
        bail!(
            "model byte count mismatch: expected {}, got {}",
            selected.bytes,
            metadata.len()
        );
    }
    let file =
        File::open(path).with_context(|| format!("open downloaded model {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("hash downloaded model {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != selected.sha256 {
        return Err(anyhow!(
            "model SHA-256 mismatch: expected {}, got {actual}",
            selected.sha256
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_selects_the_frozen_qwen_file() {
        let selected = selected_model().unwrap();
        assert_eq!(selected.pack_id, "standard-v1");
        assert_eq!(selected.filename, EXPECTED_MODEL_FILE);
        assert_eq!(selected.bytes, 2_740_937_888);
        assert_eq!(
            selected.sha256,
            "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
        );
    }

    #[test]
    fn verification_rejects_wrong_bytes_and_accepts_the_exact_digest() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tiny.gguf");
        std::fs::write(&path, b"tiny model").unwrap();
        let selected = SelectedModel {
            pack_id: "test".to_owned(),
            url: "https://example.test/tiny.gguf".to_owned(),
            filename: "tiny.gguf".to_owned(),
            sha256: crate::crypto::sha256_hex(b"tiny model"),
            bytes: 10,
        };
        verify_model_file(&path, &selected).unwrap();
        std::fs::write(&path, b"bad").unwrap();
        assert!(verify_model_file(&path, &selected).is_err());
    }

    #[test]
    fn status_is_missing_until_all_managed_resources_have_expected_sizes() {
        let temp = tempfile::tempdir().unwrap();
        let resources = ManagedResourcePaths {
            hsk: temp.path().join("hsk.json"),
            dictionary: temp.path().join("dictionary.json"),
            model: temp.path().join("model.gguf"),
            fonts: temp.path().join("fonts"),
        };
        let mut setup = ModelSetup::new(resources.clone(), temp.path().join("cache")).unwrap();
        setup.selected.bytes = 4;
        assert_eq!(setup.status().state, BrowserSetupState::MissingModels);
        std::fs::write(&resources.hsk, b"{}").unwrap();
        std::fs::write(&resources.dictionary, b"{}").unwrap();
        std::fs::write(&resources.model, b"1234").unwrap();
        assert_eq!(setup.status().state, BrowserSetupState::Ready);
    }

    #[test]
    fn font_ids_map_only_to_packaged_cjk_fonts() {
        let temp = tempfile::tempdir().unwrap();
        let resources = ManagedResourcePaths {
            hsk: temp.path().join("hsk.json"),
            dictionary: temp.path().join("dictionary.json"),
            model: temp.path().join("model.gguf"),
            fonts: temp.path().join("fonts"),
        };
        let setup = ModelSetup::new(resources.clone(), temp.path().join("cache")).unwrap();

        assert_eq!(
            setup.font_path("hmt-sans"),
            Some(resources.fonts.join(SANS_FONT_FILE))
        );
        assert_eq!(
            setup.font_path("hmt-serif"),
            Some(resources.fonts.join(SERIF_FONT_FILE))
        );
        assert!(setup.font_path("fixture-sans").is_none());
    }
}
