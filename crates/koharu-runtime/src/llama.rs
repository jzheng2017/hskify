use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::Runtime;
use crate::archive::{self, ExtractPolicy};
use crate::install::InstallState;
use crate::loader::{add_runtime_search_path, preload_library};

const LLAMA_CPP_TAG: &str = env!("LLAMA_CPP_TAG");
const RELEASE_BASE_URL: &str = "https://github.com/ggml-org/llama.cpp/releases/download";
const LLAMA_EXTRACT_REVISION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaDistribution {
    WindowsCuda13X64,
}

impl LlamaDistribution {
    #[allow(clippy::needless_return)]
    fn detect(runtime: &Runtime) -> Result<Self> {
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            if !runtime.wants_gpu() {
                bail!("Hskify's llama runtime requires CUDA; CPU mode is disabled");
            }
            crate::cuda::require_hskify_cuda_target()
                .context("Hskify's llama runtime requires the exact CUDA target")?;
            return Ok(Self::WindowsCuda13X64);
        }

        #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
        bail!(
            "Hskify's performance build requires 64-bit Windows with NVIDIA CUDA 13.1; \
             detected os={}, arch={}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    }

    fn id(self) -> &'static str {
        match self {
            Self::WindowsCuda13X64 => "windows-cuda13-x64",
        }
    }

    fn assets(self) -> Vec<String> {
        let tag = LLAMA_CPP_TAG;
        match self {
            Self::WindowsCuda13X64 => vec![format!("llama-{tag}-bin-win-cuda-13.1-x64.zip")],
        }
    }

    fn libraries(self) -> &'static [&'static str] {
        match self {
            Self::WindowsCuda13X64 => &[
                "libomp140.x86_64.dll",
                "ggml-base.dll",
                "ggml.dll",
                "ggml-cpu-alderlake.dll",
                "ggml-cpu-cannonlake.dll",
                "ggml-cpu-cascadelake.dll",
                "ggml-cpu-cooperlake.dll",
                "ggml-cpu-haswell.dll",
                "ggml-cpu-icelake.dll",
                "ggml-cpu-ivybridge.dll",
                "ggml-cpu-piledriver.dll",
                "ggml-cpu-sandybridge.dll",
                "ggml-cpu-sapphirerapids.dll",
                "ggml-cpu-skylakex.dll",
                "ggml-cpu-sse42.dll",
                "ggml-cpu-x64.dll",
                "ggml-cpu-zen4.dll",
                "ggml-cuda.dll",
                "ggml-rpc.dll",
                "llama.dll",
                "mtmd.dll",
            ],
        }
    }

    fn install_dir(self, runtime: &Runtime) -> PathBuf {
        runtime
            .root()
            .join("runtime")
            .join("llama.cpp")
            .join(LLAMA_CPP_TAG)
            .join(self.id())
    }

    fn source_id(self) -> String {
        format!(
            "llama-{LLAMA_CPP_TAG}-{}-extract-{LLAMA_EXTRACT_REVISION}",
            self.id()
        )
    }
}

pub(crate) fn package_enabled(runtime: &Runtime) -> bool {
    runtime.cuda_required() || LlamaDistribution::detect(runtime).is_ok()
}

pub(crate) fn package_present(runtime: &Runtime) -> Result<bool> {
    let distribution = LlamaDistribution::detect(runtime)?;
    let install_dir = distribution.install_dir(runtime);
    let source_id = distribution.source_id();
    let install = InstallState::new(&install_dir, &source_id);
    if !install.is_current() {
        return Ok(false);
    }

    Ok(distribution
        .libraries()
        .iter()
        .all(|library| install_dir.join(library).exists()))
}

pub(crate) async fn package_prepare(runtime: &Runtime) -> Result<()> {
    ensure_ready(runtime).await
}

pub(crate) async fn ensure_ready(runtime: &Runtime) -> Result<()> {
    let distribution = LlamaDistribution::detect(runtime)?;
    let install_dir = distribution.install_dir(runtime);
    let source_id = distribution.source_id();
    let install = InstallState::new(&install_dir, &source_id);

    if !install.is_current() {
        install.reset()?;

        for asset in &distribution.assets() {
            let url = format!("{RELEASE_BASE_URL}/{LLAMA_CPP_TAG}/{asset}");
            let archive = runtime
                .downloads()
                .cached_download(&url, asset)
                .await
                .with_context(|| format!("failed to download `{url}`"))?;
            let kind = archive::detect_kind(asset)?;
            archive::extract(
                &archive,
                &install_dir,
                kind,
                ExtractPolicy::RuntimeLibraries,
            )?;
        }

        for library in distribution.libraries() {
            if !install_dir.join(library).exists() {
                bail!(
                    "required library `{library}` missing from `{}`",
                    install_dir.display()
                );
            }
        }

        install.commit()?;
    }

    add_runtime_search_path(&install_dir)?;
    for library in distribution.libraries() {
        preload_library(&install_dir.join(library))?;
    }

    Ok(())
}

pub(crate) fn runtime_dir(runtime: &Runtime) -> Result<PathBuf> {
    Ok(LlamaDistribution::detect(runtime)?.install_dir(runtime))
}

crate::declare_native_package!(
    id: "runtime:llama",
    bootstrap: true,
    order: 20,
    enabled: crate::llama::package_enabled,
    present: crate::llama::package_present,
    prepare: crate::llama::package_prepare,
);

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;

    fn touch(path: &Path) {
        fs::write(path, b"ok").unwrap();
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    #[test]
    #[ignore = "requires the exact Hskify CUDA target"]
    fn detect_returns_a_variant_for_current_platform() {
        let runtime = Runtime::new("/tmp/koharu-runtime", crate::ComputePolicy::PreferGpu).unwrap();
        let distribution = LlamaDistribution::detect(&runtime).unwrap();
        assert_eq!(distribution.id(), "windows-cuda13-x64");
        assert!(!distribution.assets().is_empty());
        assert!(!distribution.libraries().is_empty());
    }

    #[test]
    fn install_dir_includes_tag_and_id() {
        let runtime = Runtime::new("/tmp/koharu-runtime", crate::ComputePolicy::PreferGpu).unwrap();
        let distribution = LlamaDistribution::WindowsCuda13X64;
        let dir = distribution.install_dir(&runtime);
        assert!(
            dir.ends_with(
                std::path::Path::new("llama.cpp")
                    .join(LLAMA_CPP_TAG)
                    .join("windows-cuda13-x64")
            )
        );
        assert!(distribution.source_id().ends_with("-extract-2"));
    }

    #[test]
    fn required_policy_keeps_llama_package_enabled() {
        let runtime =
            Runtime::new("unused", crate::ComputePolicy::CudaRequired).expect("create runtime");
        assert!(package_enabled(&runtime));
    }

    #[test]
    fn preload_order_matches_libraries() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();
        let runtime = LlamaDistribution::WindowsCuda13X64;

        for library in runtime.libraries() {
            touch(&root.join(library));
        }

        let paths: Vec<PathBuf> = runtime
            .libraries()
            .iter()
            .map(|library| root.join(library))
            .collect();
        assert!(paths.iter().all(|path| path.exists()));
        assert_eq!(paths.len(), runtime.libraries().len());
    }

    #[test]
    fn llama_runtime_does_not_own_cuda_runtime_or_cublas() {
        let distribution = LlamaDistribution::WindowsCuda13X64;
        assert_eq!(distribution.assets().len(), 1);
        for duplicate in ["cudart64_13.dll", "cublasLt64_13.dll", "cublas64_13.dll"] {
            assert!(!distribution.libraries().contains(&duplicate));
        }
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    #[test]
    fn windows_runtime_rejects_cpu_policy() {
        let runtime = Runtime::new("/tmp/koharu-runtime", crate::ComputePolicy::CpuOnly).unwrap();
        let error = LlamaDistribution::detect(&runtime).unwrap_err();
        assert!(error.to_string().contains("CPU mode is disabled"));
    }
}
