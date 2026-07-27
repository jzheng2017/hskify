use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result, anyhow, bail};
use chrono::NaiveDate;
use koharu_runtime::{ComputePolicy, RuntimeManager};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::contracts::{
    BUILD_FINGERPRINT, BrowserSetupState, BrowserSetupStatus, ResourceIdentity, Validate,
    validate_resource_identities,
};
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
pub(crate) const HSK_RESOURCE_BYTES: u64 = 1_219_917;
pub(crate) const HSK_RESOURCE_SHA256: &str =
    "e603244c49d6a231426e9696574e98bd1e76fbea68f56e76ea98695d26ce478f";
pub(crate) const DICTIONARY_RESOURCE_BYTES: u64 = 28_604_488;
pub(crate) const DICTIONARY_RESOURCE_SHA256: &str =
    "4011f023d27e576559ae0f2afe6fd0cc4458f96d225baa80f0ddbc9bb0344f33";
const SANS_FONT_BYTES: u64 = 17_773_244;
const SANS_FONT_SHA256: &str = "763146584cf0710223441356b4395e279021b0806c196614377a7a0174ae074a";
const SERIF_FONT_BYTES: u64 = 25_129_160;
const SERIF_FONT_SHA256: &str = "a4aed9985a5916fbf6690456f8732a9fccd517938e353165d4142b4f11a39280";

pub(crate) const TRANSLATION_MODEL_ID: &str = "translation-model";
pub(crate) const OCR_CONFIG_ID: &str = "pp-ocr-v5-english-recognizer-config";
pub(crate) const OCR_MODEL_ID: &str = "pp-ocr-v5-english-recognizer-model";
pub(crate) const DETECTOR_MODEL_ID: &str = "pp-ocr-v5-mobile-detector-model";

#[derive(Debug, Clone)]
pub(crate) struct ManagedResourcePaths {
    root: PathBuf,
    pub(crate) hsk: PathBuf,
    pub(crate) dictionary: PathBuf,
    pub(crate) model: PathBuf,
    pub(crate) fonts: PathBuf,
    resident_models: PathBuf,
}

impl ManagedResourcePaths {
    pub(crate) fn discover() -> Result<Self> {
        let root = nonempty_env_path(RESOURCES_DIRECTORY_ENV)
            .or_else(default_resource_root)
            .context(
                "cannot determine the per-user resource directory; set HSK_MANGA_RESOURCES_DIR",
            )?;
        Ok(Self {
            root: root.clone(),
            hsk: nonempty_env_path(HSK_RESOURCE_ENV)
                .unwrap_or_else(|| root.join(HSK_RESOURCE_FILE)),
            dictionary: nonempty_env_path(DICTIONARY_RESOURCE_ENV)
                .unwrap_or_else(|| root.join(DICTIONARY_RESOURCE_FILE)),
            model: nonempty_env_path(QWEN_RESOURCE_ENV)
                .unwrap_or_else(|| root.join("models").join(EXPECTED_MODEL_FILE)),
            fonts: root.join("fonts"),
            resident_models: root.join("models").join("resident"),
        })
    }

    pub(crate) fn bundled_resources_have_expected_sizes(&self) -> bool {
        file_has_expected_size(&self.hsk, HSK_RESOURCE_BYTES)
            && file_has_expected_size(&self.dictionary, DICTIONARY_RESOURCE_BYTES)
            && file_has_expected_size(&self.fonts.join(SANS_FONT_FILE), SANS_FONT_BYTES)
            && file_has_expected_size(&self.fonts.join(SERIF_FONT_FILE), SERIF_FONT_BYTES)
    }

    fn resource_path(&self, identity: &ResourceIdentity) -> PathBuf {
        if identity.id == TRANSLATION_MODEL_ID {
            self.model.clone()
        } else {
            self.resident_models
                .join(&identity.id)
                .join(&identity.filename)
        }
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
        .map(|root| root.join("Hskify").join("resources"))
}

#[derive(Debug, Clone)]
struct ModelResources {
    model_id: String,
    identities: Vec<ResourceIdentity>,
    urls: BTreeMap<String, String>,
}

impl ModelResources {
    fn total_bytes(&self) -> Result<u64> {
        self.identities.iter().try_fold(0_u64, |total, identity| {
            total
                .checked_add(identity.bytes)
                .context("resident resource byte total overflowed")
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ModelManifest {
    manifest_version: u8,
    generated_at: String,
    translation_model_id: String,
    resource_identities: Vec<ManifestResourceIdentity>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ManifestResourceIdentity {
    id: String,
    repository: String,
    repository_revision: String,
    filename: String,
    url: String,
    bytes: u64,
    sha256: String,
}

impl ManifestResourceIdentity {
    fn identity(&self) -> ResourceIdentity {
        ResourceIdentity {
            id: self.id.clone(),
            repository: self.repository.clone(),
            repository_revision: self.repository_revision.clone(),
            filename: self.filename.clone(),
            bytes: self.bytes,
            sha256: self.sha256.clone(),
        }
    }
}

fn model_resources() -> Result<ModelResources> {
    let manifest: ModelManifest =
        serde_json::from_str(MODEL_MANIFEST).context("parse embedded model manifest")?;
    if manifest.manifest_version != 1 {
        bail!("embedded model manifest is not version 1");
    }
    NaiveDate::parse_from_str(&manifest.generated_at, "%Y-%m-%d")
        .context("embedded model manifest has an invalid generation date")?;
    if manifest.translation_model_id != "qwen3.5-4b" {
        bail!("embedded model manifest does not require qwen3.5-4b");
    }

    let identities = manifest
        .resource_identities
        .iter()
        .map(ManifestResourceIdentity::identity)
        .collect::<Vec<_>>();
    validate_resource_identities(&identities)
        .map_err(|error| anyhow!("invalid resident resource identities: {error}"))?;
    let required_ids = [
        OCR_CONFIG_ID,
        OCR_MODEL_ID,
        DETECTOR_MODEL_ID,
        TRANSLATION_MODEL_ID,
    ];
    if identities.len() != required_ids.len()
        || identities
            .iter()
            .map(|identity| identity.id.as_str())
            .ne(required_ids)
    {
        bail!("embedded model manifest must contain exactly the four required resource identities");
    }

    for resource in &manifest.resource_identities {
        if resource.url != pinned_url(&resource.identity()) {
            bail!(
                "resource URL does not match its pinned identity: {}",
                resource.id
            );
        }
    }
    if identities
        .last()
        .is_none_or(|identity| identity.filename != EXPECTED_MODEL_FILE)
    {
        bail!("translation-model is not the approved Qwen 4B artifact");
    }

    Ok(ModelResources {
        model_id: manifest.translation_model_id,
        identities,
        urls: manifest
            .resource_identities
            .into_iter()
            .map(|resource| (resource.id, resource.url))
            .collect(),
    })
}

fn pinned_url(identity: &ResourceIdentity) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        identity.repository, identity.repository_revision, identity.filename
    )
}

fn model_resource_paths(
    managed: &ManagedResourcePaths,
    model: &ModelResources,
) -> BTreeMap<String, PathBuf> {
    model
        .identities
        .iter()
        .map(|identity| (identity.id.clone(), managed.resource_path(identity)))
        .collect()
}

#[derive(Debug, Clone)]
pub(crate) struct ResidentResourcePaths {
    runtime_root: PathBuf,
    pub(crate) hsk: PathBuf,
    pub(crate) dictionary: PathBuf,
    resident: BTreeMap<String, PathBuf>,
}

impl ResidentResourcePaths {
    pub(crate) fn discover() -> Result<Self> {
        let managed = ManagedResourcePaths::discover()?;
        let model = model_resources()?;
        let paths = model_resource_paths(&managed, &model);
        Self::from_resident(&managed, &paths)
    }

    fn from_resident(
        managed: &ManagedResourcePaths,
        paths: &BTreeMap<String, PathBuf>,
    ) -> Result<Self> {
        for required_id in [
            OCR_CONFIG_ID,
            OCR_MODEL_ID,
            DETECTOR_MODEL_ID,
            TRANSLATION_MODEL_ID,
        ] {
            if !paths.contains_key(required_id) {
                bail!("resident resource path is missing: {required_id}");
            }
        }
        Ok(Self {
            runtime_root: managed.root.clone(),
            hsk: managed.hsk.clone(),
            dictionary: managed.dictionary.clone(),
            resident: paths.clone(),
        })
    }

    pub(crate) fn path(&self, id: &str) -> Result<&Path> {
        self.resident
            .get(id)
            .map(PathBuf::as_path)
            .with_context(|| format!("resident resource path is missing: {id}"))
    }

    pub(crate) fn runtime_root(&self) -> &Path {
        &self.runtime_root
    }
}

pub(crate) struct ModelSetup {
    resources: ManagedResourcePaths,
    ready_marker: PathBuf,
    model: ModelResources,
    resource_paths: BTreeMap<String, PathBuf>,
    total_bytes: u64,
    ready: AtomicBool,
    active: AtomicBool,
    state: RwLock<Option<BrowserSetupStatus>>,
}

impl ModelSetup {
    pub(crate) fn new(resources: ManagedResourcePaths, cache_root: PathBuf) -> Result<Self> {
        let runtime_root = cache_root.join("browser-runtime");
        let model = model_resources()?;
        let total_bytes = model.total_bytes()?;
        let resource_paths = model_resource_paths(&resources, &model);
        let setup = Self {
            resources,
            ready_marker: runtime_root.join("models.ready"),
            model,
            resource_paths,
            total_bytes,
            ready: AtomicBool::new(false),
            active: AtomicBool::new(false),
            state: RwLock::new(None),
        };
        setup.restore_readiness_marker();
        Ok(setup)
    }

    pub(crate) fn resource_identities(&self) -> Vec<ResourceIdentity> {
        self.model.identities.clone()
    }

    pub(crate) fn resources_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
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
        if !self.resources.bundled_resources_have_expected_sizes() {
            let status = self.failed_status(
                "BUNDLED_RESOURCES_INVALID",
                "The exact HSK, dictionary, and CJK font resources are missing. Reinstall the matching Hskify build.",
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

        let current = self.first_unready_identity();
        let status = self.progress_status(
            BrowserSetupState::Downloading,
            current.map(|identity| identity.filename.clone()),
            0,
            current.map_or_else(
                || "Preparing resident resources.".to_owned(),
                |identity| format!("Preparing to download {}.", identity.filename),
            ),
        );
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
        koharu_runtime::require_hskify_cuda_target()
            .context("model setup requires the exact Hskify CUDA target")?;
        if self.ready_marker.is_dir() {
            bail!(
                "model readiness marker is a directory: {}",
                self.ready_marker.display()
            );
        }
        if self.ready_marker.exists() {
            tokio::fs::remove_file(&self.ready_marker)
                .await
                .with_context(|| {
                    format!(
                        "invalidate model readiness marker {}",
                        self.ready_marker.display()
                    )
                })?;
        }

        verify_bundled_resources(self.resources.clone())
            .await
            .context("verify exact bundled HSK, dictionary, and font resources")?;

        let runtime_root = self
            .ready_marker
            .parent()
            .context("model readiness marker has no parent directory")?;
        let runtime = RuntimeManager::new(runtime_root, ComputePolicy::CudaRequired)
            .context("initialize Koharu's managed downloader")?;
        let mut completed = 0_u64;
        for identity in self.model.identities.clone() {
            let destination = self
                .resource_paths
                .get(&identity.id)
                .cloned()
                .with_context(|| format!("managed path is missing for {}", identity.id))?;
            self.replace_status(self.progress_status(
                BrowserSetupState::Verifying,
                Some(identity.filename.clone()),
                completed,
                format!("Verifying {}.", identity.filename),
            ));
            if file_has_expected_size(&destination, identity.bytes)
                && verify_resource_async(destination.clone(), identity.clone())
                    .await
                    .is_ok()
            {
                completed += identity.bytes;
                continue;
            }

            let parent = destination
                .parent()
                .context("managed resource path has no parent directory")?;
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("create model directory {}", parent.display()))?;
            let available = fs4::available_space(parent)
                .with_context(|| format!("inspect free disk space at {}", parent.display()))?;
            if available < identity.bytes {
                bail!(
                    "not enough free disk space for {}: {} bytes available, {} required",
                    identity.id,
                    available,
                    identity.bytes
                );
            }

            self.replace_status(self.progress_status(
                BrowserSetupState::Downloading,
                Some(identity.filename.clone()),
                completed,
                format!("Downloading {}.", identity.filename),
            ));
            let cache_filename = format!(
                "{}-{}-{}",
                identity.id,
                &identity.sha256[..16],
                identity.filename
            );
            let downloaded = runtime
                .downloads()
                .pinned_url(
                    self.model
                        .urls
                        .get(&identity.id)
                        .context("managed resource URL is missing")?,
                    &cache_filename,
                )
                .await?;

            self.replace_status(self.progress_status(
                BrowserSetupState::Verifying,
                Some(identity.filename.clone()),
                completed,
                format!("Verifying {}.", identity.filename),
            ));
            if let Err(error) = verify_resource_async(downloaded.clone(), identity.clone()).await {
                tokio::fs::remove_file(&downloaded).await.ok();
                return Err(error);
            }
            atomic_install_verified(&downloaded, &destination, &identity).await?;
            completed += identity.bytes;
        }

        self.mark_ready().await?;
        self.replace_status(self.ready_status());
        Ok(())
    }

    async fn mark_ready(&self) -> Result<()> {
        self.replace_status(self.progress_status(
            BrowserSetupState::Verifying,
            None,
            self.total_bytes,
            "Finalizing verified resident resources.".to_owned(),
        ));
        ResidentResourcePaths::from_resident(&self.resources, &self.resource_paths)?;
        let runtime_root = self
            .ready_marker
            .parent()
            .context("model readiness marker has no parent directory")?;
        tokio::fs::create_dir_all(runtime_root)
            .await
            .with_context(|| {
                format!(
                    "create browser runtime directory {}",
                    runtime_root.display()
                )
            })?;
        write_atomic_text(&self.ready_marker, &self.expected_marker()?).await?;
        self.ready.store(true, Ordering::Release);
        Ok(())
    }

    fn restore_readiness_marker(&self) {
        let ready = self
            .expected_marker()
            .ok()
            .and_then(|expected| {
                std::fs::read_to_string(&self.ready_marker)
                    .ok()
                    .map(|actual| actual == expected)
            })
            .unwrap_or(false);
        self.ready.store(ready, Ordering::Release);
    }

    fn progress_status(
        &self,
        state: BrowserSetupState,
        current_file: Option<String>,
        completed_bytes: u64,
        message: String,
    ) -> BrowserSetupStatus {
        BrowserSetupStatus {
            state,
            model_id: self.model.model_id.clone(),
            current_file,
            completed_bytes: Some(completed_bytes.min(self.total_bytes)),
            total_bytes: Some(self.total_bytes),
            required_disk_bytes: Some(self.total_bytes),
            message,
            error_code: None,
        }
    }

    fn first_unready_identity(&self) -> Option<&ResourceIdentity> {
        self.model.identities.iter().find(|identity| {
            self.resource_paths
                .get(&identity.id)
                .is_none_or(|path| !file_has_expected_size(path, identity.bytes))
        })
    }

    fn missing_status(&self) -> BrowserSetupStatus {
        let current = self
            .first_unready_identity()
            .or_else(|| self.model.identities.first());
        BrowserSetupStatus {
            state: BrowserSetupState::MissingModels,
            model_id: self.model.model_id.clone(),
            current_file: current.map(|identity| identity.filename.clone()),
            completed_bytes: Some(0),
            total_bytes: Some(self.total_bytes),
            required_disk_bytes: Some(self.total_bytes),
            message: format!(
                "Download and verify {:.1} GiB of pinned resident resources.",
                self.total_bytes as f64 / 1024_f64.powi(3)
            ),
            error_code: None,
        }
    }

    fn ready_status(&self) -> BrowserSetupStatus {
        BrowserSetupStatus {
            state: BrowserSetupState::Ready,
            model_id: self.model.model_id.clone(),
            current_file: None,
            completed_bytes: None,
            total_bytes: None,
            required_disk_bytes: None,
            message: "Local translation and resident model resources are ready.".to_owned(),
            error_code: None,
        }
    }

    fn failed_status(&self, code: &str, message: &str) -> BrowserSetupStatus {
        BrowserSetupStatus {
            state: BrowserSetupState::Failed,
            model_id: self.model.model_id.clone(),
            current_file: self
                .first_unready_identity()
                .map(|identity| identity.filename.clone()),
            completed_bytes: None,
            total_bytes: None,
            required_disk_bytes: Some(self.total_bytes),
            message: message.to_owned(),
            error_code: Some(code.to_owned()),
        }
    }

    fn expected_marker(&self) -> Result<String> {
        let installations = self
            .model
            .identities
            .iter()
            .map(|identity| {
                let path = self
                    .resource_paths
                    .get(&identity.id)
                    .with_context(|| format!("managed path is missing for {}", identity.id))?;
                Ok(ReadinessInstallation {
                    id: identity.id.clone(),
                    path: path.to_string_lossy().into_owned(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        serde_json::to_string(&ReadinessMarker {
            build_fingerprint: BUILD_FINGERPRINT,
            resource_identities: &self.model.identities,
            installations,
        })
        .context("serialize exact resident resource readiness marker")
    }

    fn replace_status(&self, status: BrowserSetupStatus) {
        debug_assert!(status.validate().is_ok());
        *self.state.write().expect("setup state lock poisoned") = Some(status);
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReadinessMarker<'a> {
    build_fingerprint: &'static str,
    resource_identities: &'a [ResourceIdentity],
    installations: Vec<ReadinessInstallation>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReadinessInstallation {
    id: String,
    path: String,
}

fn file_has_expected_size(path: &Path, expected: u64) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() == expected)
}

async fn verify_resource_async(path: PathBuf, identity: ResourceIdentity) -> Result<()> {
    tokio::task::spawn_blocking(move || verify_resource_file(&path, &identity))
        .await
        .context("join resident resource verification task")?
}

async fn verify_bundled_resources(resources: ManagedResourcePaths) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        verify_exact_file(
            &resources.hsk,
            HSK_RESOURCE_FILE,
            HSK_RESOURCE_BYTES,
            HSK_RESOURCE_SHA256,
        )?;
        verify_exact_file(
            &resources.dictionary,
            DICTIONARY_RESOURCE_FILE,
            DICTIONARY_RESOURCE_BYTES,
            DICTIONARY_RESOURCE_SHA256,
        )?;
        verify_exact_file(
            &resources.fonts.join(SANS_FONT_FILE),
            SANS_FONT_FILE,
            SANS_FONT_BYTES,
            SANS_FONT_SHA256,
        )?;
        verify_exact_file(
            &resources.fonts.join(SERIF_FONT_FILE),
            SERIF_FONT_FILE,
            SERIF_FONT_BYTES,
            SERIF_FONT_SHA256,
        )
    })
    .await
    .context("join bundled resource verification task")?
}

fn verify_resource_file(path: &Path, identity: &ResourceIdentity) -> Result<()> {
    verify_exact_file(path, &identity.id, identity.bytes, &identity.sha256)
}

fn verify_exact_file(
    path: &Path,
    id: &str,
    expected_bytes: u64,
    expected_sha256: &str,
) -> Result<()> {
    let metadata = path
        .metadata()
        .with_context(|| format!("inspect resident resource {}", path.display()))?;
    if !metadata.is_file() || metadata.len() != expected_bytes {
        bail!(
            "{} byte count mismatch: expected {}, got {}",
            id,
            expected_bytes,
            metadata.len()
        );
    }
    let file =
        File::open(path).with_context(|| format!("open resident resource {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("hash resident resource {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != expected_sha256 {
        return Err(anyhow!(
            "{} SHA-256 mismatch: expected {}, got {actual}",
            id,
            expected_sha256
        ));
    }
    Ok(())
}

async fn atomic_install_verified(
    source: &Path,
    destination: &Path,
    identity: &ResourceIdentity,
) -> Result<()> {
    let parent = destination
        .parent()
        .context("managed resource destination has no parent")?;
    let temporary = parent.join(format!(
        ".{}.{}.{}.tmp",
        identity.filename,
        std::process::id(),
        &identity.sha256[..16]
    ));
    if temporary.exists() {
        tokio::fs::remove_file(&temporary)
            .await
            .with_context(|| format!("remove stale install file {}", temporary.display()))?;
    }
    tokio::fs::copy(source, &temporary).await.with_context(|| {
        format!(
            "copy verified {} to atomic install file {}",
            identity.id,
            temporary.display()
        )
    })?;
    if let Err(error) = verify_resource_async(temporary.clone(), identity.clone()).await {
        tokio::fs::remove_file(&temporary).await.ok();
        return Err(error);
    }

    if destination.is_dir() {
        tokio::fs::remove_file(&temporary).await.ok();
        bail!(
            "managed resource destination is a directory: {}",
            destination.display()
        );
    }
    if destination.exists() {
        if verify_resource_async(destination.to_path_buf(), identity.clone())
            .await
            .is_ok()
        {
            tokio::fs::remove_file(&temporary).await.ok();
            return Ok(());
        }
        tokio::fs::remove_file(destination)
            .await
            .with_context(|| format!("remove invalid resource {}", destination.display()))?;
    }
    tokio::fs::rename(&temporary, destination)
        .await
        .with_context(|| {
            format!(
                "atomically install {} at {}",
                identity.id,
                destination.display()
            )
        })
}

async fn write_atomic_text(destination: &Path, contents: &str) -> Result<()> {
    let parent = destination
        .parent()
        .context("readiness marker has no parent directory")?;
    let temporary = parent.join(format!(".models.ready.{}.tmp", std::process::id()));
    if temporary.exists() {
        tokio::fs::remove_file(&temporary)
            .await
            .with_context(|| format!("remove stale readiness marker {}", temporary.display()))?;
    }
    tokio::fs::write(&temporary, contents)
        .await
        .with_context(|| format!("write readiness marker {}", temporary.display()))?;
    if destination.exists() {
        tokio::fs::remove_file(destination)
            .await
            .with_context(|| format!("replace readiness marker {}", destination.display()))?;
    }
    tokio::fs::rename(&temporary, destination)
        .await
        .with_context(|| {
            format!(
                "atomically install readiness marker {}",
                destination.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_managed_resources(root: &Path) -> ManagedResourcePaths {
        ManagedResourcePaths {
            root: root.to_path_buf(),
            hsk: root.join("hsk.json"),
            dictionary: root.join("dictionary.json"),
            model: root.join("model.gguf"),
            fonts: root.join("fonts"),
            resident_models: root.join("resident"),
        }
    }

    #[test]
    fn embedded_manifest_requires_qwen_4b_and_all_sorted_frozen_resources() {
        let model = model_resources().unwrap();
        assert_eq!(model.model_id, "qwen3.5-4b");
        assert_eq!(
            model
                .identities
                .iter()
                .map(|identity| identity.id.as_str())
                .collect::<Vec<_>>(),
            [
                OCR_CONFIG_ID,
                OCR_MODEL_ID,
                DETECTOR_MODEL_ID,
                TRANSLATION_MODEL_ID,
            ]
        );
        let ocr_config = model
            .identities
            .iter()
            .find(|identity| identity.id == OCR_CONFIG_ID)
            .unwrap();
        assert_eq!(
            ocr_config.repository,
            "PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx"
        );
        assert_eq!(
            ocr_config.repository_revision,
            "3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b"
        );
        assert_eq!(ocr_config.filename, "inference.yml");
        assert_eq!(ocr_config.bytes, 3_964);
        assert_eq!(
            ocr_config.sha256,
            "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067"
        );
        let ocr_model = model
            .identities
            .iter()
            .find(|identity| identity.id == OCR_MODEL_ID)
            .unwrap();
        assert_eq!(
            ocr_model.repository,
            "PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx"
        );
        assert_eq!(
            ocr_model.repository_revision,
            "3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b"
        );
        assert_eq!(ocr_model.filename, "inference.onnx");
        assert_eq!(ocr_model.bytes, 7_848_423);
        assert_eq!(
            ocr_model.sha256,
            "b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f"
        );
        let detector_model = model
            .identities
            .iter()
            .find(|identity| identity.id == DETECTOR_MODEL_ID)
            .unwrap();
        assert_eq!(
            detector_model.repository,
            "PaddlePaddle/PP-OCRv5_mobile_det_onnx"
        );
        assert_eq!(
            detector_model.repository_revision,
            "e6f4fa85f00e168c862bc462aebca69eef9b3d3d"
        );
        assert_eq!(detector_model.filename, "inference.onnx");
        assert_eq!(detector_model.bytes, 4_826_518);
        assert_eq!(
            detector_model.sha256,
            "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d"
        );
        let translation = model
            .identities
            .iter()
            .find(|identity| identity.id == TRANSLATION_MODEL_ID)
            .unwrap();
        assert_eq!(translation.filename, EXPECTED_MODEL_FILE);
        assert_eq!(translation.bytes, 2_740_937_888);
        assert_eq!(
            translation.sha256,
            "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
        );
    }

    #[test]
    fn verification_rejects_wrong_bytes_and_accepts_the_exact_digest() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("tiny.bin");
        std::fs::write(&path, b"tiny model").unwrap();
        let identity = ResourceIdentity {
            id: "tiny-resource".to_owned(),
            repository: "example/test".to_owned(),
            repository_revision: "a".repeat(40),
            filename: "tiny.bin".to_owned(),
            sha256: crate::crypto::sha256_hex(b"tiny model"),
            bytes: 10,
        };
        verify_resource_file(&path, &identity).unwrap();
        std::fs::write(&path, b"bad").unwrap();
        assert!(verify_resource_file(&path, &identity).is_err());
    }

    #[test]
    fn readiness_requires_every_declared_file_and_the_exact_marker() {
        let temp = tempfile::tempdir().unwrap();
        let cache_root = temp.path().join("cache");
        let resources = test_managed_resources(temp.path());
        let mut setup = ModelSetup::new(resources.clone(), cache_root.clone()).unwrap();
        std::fs::write(&resources.hsk, b"{}").unwrap();
        std::fs::write(&resources.dictionary, b"{}").unwrap();

        for identity in &mut setup.model.identities {
            let contents = identity.id.as_bytes();
            identity.bytes = contents.len() as u64;
            identity.sha256 = crate::crypto::sha256_hex(contents);
            let path = setup.resource_paths.get(&identity.id).unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }
        setup.total_bytes = setup.model.total_bytes().unwrap();
        assert_eq!(setup.status().state, BrowserSetupState::MissingModels);
        std::fs::create_dir_all(cache_root.join("browser-runtime")).unwrap();
        std::fs::write(&setup.ready_marker, setup.expected_marker().unwrap()).unwrap();
        setup.restore_readiness_marker();
        assert_eq!(
            setup.status().state,
            BrowserSetupState::Ready,
            "the exact installer marker should make verified files ready without warming CUDA"
        );
    }

    #[test]
    fn font_ids_map_only_to_packaged_cjk_fonts() {
        let temp = tempfile::tempdir().unwrap();
        let resources = test_managed_resources(temp.path());
        let cache_root = temp.path().join("cache");
        let setup = ModelSetup::new(resources.clone(), cache_root).unwrap();

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
