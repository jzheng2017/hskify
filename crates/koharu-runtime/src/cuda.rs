use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use serde::Deserialize;
use std::ffi::{CStr, c_char};
use std::fmt;

use crate::Runtime;
use crate::archive::{self, ArchiveKind, ExtractPolicy};
use crate::install::InstallState;
use crate::loader::{add_runtime_search_path, preload_library};

const CUDA_SUCCESS: i32 = 0;
const CUDA_13_1_DRIVER_VERSION: i32 = 13010;
const CUDA_EXTRACT_REVISION: u32 = 5;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const HSKIFY_TARGET_DEVICE_NAME: &str = "NVIDIA GeForce RTX 4080 SUPER";
const HSKIFY_TARGET_COMPUTE_CAPABILITY: (i32, i32) = (8, 9);
const HSKIFY_TARGET_MIN_MEMORY_MIB: usize = 16_000;
const MIB: usize = 1024 * 1024;

type CuInit = unsafe extern "C" fn(flags: u32) -> i32;
type CuDriverGetVersion = unsafe extern "C" fn(driver_version: *mut i32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(device: *mut i32, ordinal: i32) -> i32;
type CuDeviceGetAttribute = unsafe extern "C" fn(pi: *mut i32, attrib: i32, dev: i32) -> i32;
type CuDeviceGetName = unsafe extern "C" fn(name: *mut c_char, len: i32, dev: i32) -> i32;
type CuDeviceTotalMem = unsafe extern "C" fn(bytes: *mut usize, dev: i32) -> i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CudaDriverVersion {
    raw: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub name: String,
    pub total_memory_bytes: usize,
    pub compute_capability: (i32, i32),
}

#[derive(Debug, Deserialize)]
struct PypiRelease {
    urls: Vec<PypiFile>,
}

#[derive(Debug, Deserialize)]
struct PypiFile {
    filename: String,
    url: String,
}

#[allow(dead_code)]
struct WheelSpec {
    package: &'static str,
    windows_dylibs: &'static [&'static str],
    linux_dylibs: &'static [&'static str],
}

const WHEELS: &[WheelSpec] = &[
    WheelSpec {
        package: "nvidia-cuda-runtime/13.1.80",
        windows_dylibs: &["cudart64_13.dll"],
        linux_dylibs: &["libcudart.so.13"],
    },
    WheelSpec {
        package: "nvidia-cuda-nvrtc/13.1.80",
        windows_dylibs: &["nvrtc64_130_0.dll", "nvrtc-builtins64_131.dll"],
        linux_dylibs: &["libnvrtc.so.13", "libnvrtc-builtins.so.13.1"],
    },
    WheelSpec {
        package: "nvidia-cublas/13.2.0.9",
        windows_dylibs: &["cublasLt64_13.dll", "cublas64_13.dll"],
        linux_dylibs: &["libcublasLt.so.13", "libcublas.so.13"],
    },
    WheelSpec {
        package: "nvidia-cufft/12.1.0.31",
        windows_dylibs: &["cufft64_12.dll"],
        linux_dylibs: &["libcufft.so.12"],
    },
    WheelSpec {
        package: "nvidia-curand/10.4.1.81",
        windows_dylibs: &["curand64_10.dll"],
        linux_dylibs: &["libcurand.so.10"],
    },
    WheelSpec {
        package: "nvidia-cudnn-cu13/9.17.0.29",
        windows_dylibs: &[
            "cudnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll",
        ],
        linux_dylibs: &[
            "libcudnn.so.9",
            "libcudnn_adv.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
            "libcudnn_engines_tensor_ir.so.9",
            "libcudnn_graph.so.9",
            "libcudnn_heuristic.so.9",
            "libcudnn_ops.so.9",
        ],
    },
];

impl CudaDriverVersion {
    pub const fn from_raw(raw: i32) -> Self {
        Self { raw }
    }

    pub const fn raw(self) -> i32 {
        self.raw
    }

    pub const fn major(self) -> i32 {
        self.raw / 1000
    }

    pub const fn minor(self) -> i32 {
        (self.raw % 1000) / 10
    }

    pub const fn supports_cuda_13_1(self) -> bool {
        self.raw >= CUDA_13_1_DRIVER_VERSION
    }
}

impl fmt::Display for CudaDriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major(), self.minor())
    }
}

pub fn driver_version() -> Result<CudaDriverVersion> {
    let library_name = if cfg!(target_os = "windows") {
        "nvcuda.dll"
    } else {
        "libcuda.so"
    };

    unsafe {
        let library = Library::new(library_name)
            .with_context(|| format!("failed to load NVIDIA driver library `{library_name}`"))?;
        let cu_init = *library
            .get::<CuInit>(b"cuInit\0")
            .context("failed to load cuInit from NVIDIA driver")?;
        let cu_driver_get_version = *library
            .get::<CuDriverGetVersion>(b"cuDriverGetVersion\0")
            .context("failed to load cuDriverGetVersion from NVIDIA driver")?;

        let status = cu_init(0);
        if status != CUDA_SUCCESS {
            bail!("cuInit failed with CUDA driver error code {status}");
        }

        let mut raw = 0;
        let status = cu_driver_get_version(&mut raw);
        if status != CUDA_SUCCESS {
            bail!("cuDriverGetVersion failed with CUDA driver error code {status}");
        }

        Ok(CudaDriverVersion::from_raw(raw))
    }
}

/// Query the compute capability of CUDA device 0.
///
/// Returns `(major, minor)` e.g. `(8, 0)` for Ampere, `(8, 9)` for Ada.
pub fn compute_capability() -> Result<(i32, i32)> {
    Ok(cuda_device_info()?.compute_capability)
}

/// Query the exact identity and capacity of CUDA device 0 through the NVIDIA
/// driver API. Hskify's performance build uses this at runtime as well as at
/// build time so a copied binary cannot silently run on another GPU.
pub fn cuda_device_info() -> Result<CudaDeviceInfo> {
    let library_name = if cfg!(target_os = "windows") {
        "nvcuda.dll"
    } else {
        "libcuda.so"
    };

    unsafe {
        let library = Library::new(library_name)
            .with_context(|| format!("failed to load `{library_name}`"))?;
        let cu_init = *library
            .get::<CuInit>(b"cuInit\0")
            .context("cuInit not found")?;
        let cu_device_get = *library
            .get::<CuDeviceGet>(b"cuDeviceGet\0")
            .context("cuDeviceGet not found")?;
        let cu_device_get_attribute = *library
            .get::<CuDeviceGetAttribute>(b"cuDeviceGetAttribute\0")
            .context("cuDeviceGetAttribute not found")?;
        let cu_device_get_name = *library
            .get::<CuDeviceGetName>(b"cuDeviceGetName\0")
            .context("cuDeviceGetName not found")?;
        let cu_device_total_mem = *library
            .get::<CuDeviceTotalMem>(b"cuDeviceTotalMem_v2\0")
            .or_else(|_| library.get::<CuDeviceTotalMem>(b"cuDeviceTotalMem\0"))
            .context("cuDeviceTotalMem not found")?;

        let status = cu_init(0);
        if status != CUDA_SUCCESS {
            bail!("cuInit failed with error code {status}");
        }

        let mut dev = 0;
        let status = cu_device_get(&mut dev, 0);
        if status != CUDA_SUCCESS {
            bail!("cuDeviceGet failed with error code {status}");
        }

        let mut major = 0;
        let status = cu_device_get_attribute(
            &mut major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        );
        if status != CUDA_SUCCESS {
            bail!("cuDeviceGetAttribute(MAJOR) failed with error code {status}");
        }

        let mut minor = 0;
        let status = cu_device_get_attribute(
            &mut minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            dev,
        );
        if status != CUDA_SUCCESS {
            bail!("cuDeviceGetAttribute(MINOR) failed with error code {status}");
        }

        let mut name = [0 as c_char; 256];
        let status = cu_device_get_name(name.as_mut_ptr(), name.len() as i32, dev);
        if status != CUDA_SUCCESS {
            bail!("cuDeviceGetName failed with error code {status}");
        }
        let name = CStr::from_ptr(name.as_ptr())
            .to_str()
            .context("CUDA device name is not valid UTF-8")?
            .to_owned();

        let mut total_memory_bytes = 0usize;
        let status = cu_device_total_mem(&mut total_memory_bytes, dev);
        if status != CUDA_SUCCESS {
            bail!("cuDeviceTotalMem failed with error code {status}");
        }

        Ok(CudaDeviceInfo {
            name,
            total_memory_bytes,
            compute_capability: (major, minor),
        })
    }
}

/// Require the one runtime target supported by the Hskify product.
///
/// This deliberately has no compatibility or CPU path. Callers must propagate
/// the error and stop before creating product state or downloading assets.
pub fn require_hskify_cuda_target() -> Result<CudaDeviceInfo> {
    #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
    bail!(
        "Hskify requires 64-bit Windows; detected os={}, arch={}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let driver =
        driver_version().context("Hskify requires the NVIDIA CUDA driver API 13.1 or newer")?;
    let info = cuda_device_info().context("Hskify could not query CUDA device 0")?;
    validate_hskify_cuda_target(driver, &info)?;
    Ok(info)
}

fn validate_hskify_cuda_target(driver: CudaDriverVersion, info: &CudaDeviceInfo) -> Result<()> {
    if !driver.supports_cuda_13_1() {
        bail!(
            "Hskify requires the NVIDIA CUDA driver API 13.1 or newer; driver reports CUDA {driver}"
        );
    }
    let total_memory_mib = info.total_memory_bytes / MIB;
    if info.name != HSKIFY_TARGET_DEVICE_NAME
        || info.compute_capability != HSKIFY_TARGET_COMPUTE_CAPABILITY
        || total_memory_mib < HSKIFY_TARGET_MIN_MEMORY_MIB
    {
        bail!(
            "Hskify requires {HSKIFY_TARGET_DEVICE_NAME} with at least \
             {HSKIFY_TARGET_MIN_MEMORY_MIB} MiB and compute capability {}.{}; \
             CUDA device 0 is `{}` with {total_memory_mib} MiB and compute {}.{}",
            HSKIFY_TARGET_COMPUTE_CAPABILITY.0,
            HSKIFY_TARGET_COMPUTE_CAPABILITY.1,
            info.name,
            info.compute_capability.0,
            info.compute_capability.1
        );
    }
    Ok(())
}

pub(crate) fn package_enabled(runtime: &Runtime) -> bool {
    runtime.cuda_required()
        || (runtime.wants_gpu()
            && driver_library_available()
            && driver_version()
                .map(|version| version.supports_cuda_13_1())
                .unwrap_or(false))
}

pub(crate) fn package_present(runtime: &Runtime) -> Result<bool> {
    let install_dir = install_dir(runtime);
    let source_id = source_id()?;
    let install = InstallState::new(&install_dir, &source_id);
    if !install.is_current() {
        return Ok(false);
    }

    Ok(WHEELS
        .iter()
        .flat_map(|wheel| wheel.dylibs().iter())
        .all(|dylib| install_dir.join(dylib).exists()))
}

pub(crate) async fn package_prepare(runtime: &Runtime) -> Result<()> {
    if runtime.cuda_required() {
        require_hskify_cuda_target()?;
    }
    ensure_ready(runtime).await
}

pub(crate) async fn ensure_ready(runtime: &Runtime) -> Result<()> {
    let install_dir = install_dir(runtime);
    let source_id = source_id()?;
    let install = InstallState::new(&install_dir, &source_id);

    if !install.is_current() {
        install.reset()?;

        for wheel in WHEELS {
            let asset = select_wheel(runtime, wheel).await?;
            let archive = runtime
                .downloads()
                .cached_download(&asset.url, &asset.filename)
                .await
                .with_context(|| format!("failed to download `{}`", asset.url))?;
            archive::extract(
                &archive,
                &install_dir,
                ArchiveKind::Zip,
                ExtractPolicy::Selected(wheel.dylibs()),
            )?;
        }

        install.commit()?;
    }

    add_runtime_search_path(&install_dir)?;
    for wheel in WHEELS {
        for dylib in wheel.dylibs() {
            let path = install_dir.join(dylib);
            if path.exists() {
                preload_library(&path)?;
            }
        }
    }

    Ok(())
}

crate::declare_native_package!(
    id: "runtime:cuda",
    bootstrap: true,
    order: 10,
    enabled: package_enabled,
    present: package_present,
    prepare: package_prepare,
);

struct WheelAsset {
    url: String,
    filename: String,
}

fn driver_library_available() -> bool {
    #[cfg(target_os = "windows")]
    return unsafe { Library::new("nvcuda.dll") }.is_ok();

    #[cfg(target_os = "linux")]
    return unsafe { Library::new("libcuda.so.1") }.is_ok();

    #[allow(unreachable_code)]
    false
}

fn install_dir(runtime: &Runtime) -> std::path::PathBuf {
    runtime.root().join("runtime").join("cuda")
}

fn platform_tags() -> Result<&'static [&'static str]> {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return Ok(&["win_amd64"]);

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return Ok(&["manylinux_2_27_x86_64", "manylinux_2_17_x86_64"]);

    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64")
    )))]
    bail!(
        "CUDA wheels unsupported on {}/{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

impl WheelSpec {
    fn dylibs(&self) -> &'static [&'static str] {
        #[cfg(target_os = "windows")]
        return self.windows_dylibs;

        #[cfg(target_os = "linux")]
        return self.linux_dylibs;

        #[allow(unreachable_code)]
        &[]
    }
}

fn source_id() -> Result<String> {
    let packages = WHEELS.iter().map(|wheel| wheel.package).collect::<Vec<_>>();
    Ok(format!(
        "cuda;platform={};wheels={};extract={}",
        platform_tags()?.join(","),
        packages.join(","),
        CUDA_EXTRACT_REVISION
    ))
}

async fn select_wheel(runtime: &Runtime, wheel: &WheelSpec) -> Result<WheelAsset> {
    let (distribution, version) = wheel
        .package
        .split_once('/')
        .ok_or_else(|| anyhow!("invalid wheel package `{}`", wheel.package))?;

    let metadata_url = format!("https://pypi.org/pypi/{distribution}/{version}/json");
    let release: PypiRelease = runtime
        .http_client()
        .get(&metadata_url)
        .send()
        .await
        .with_context(|| format!("failed to fetch `{metadata_url}`"))?
        .json()
        .await
        .with_context(|| format!("failed to parse metadata for `{distribution}`"))?;

    let tags = platform_tags()?;
    for file in release.urls {
        if file.filename.ends_with(".whl") && tags.iter().any(|tag| file.filename.contains(tag)) {
            return Ok(WheelAsset {
                url: file.url,
                filename: file.filename,
            });
        }
    }

    bail!("no wheel found for `{distribution}` {version} on {tags:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_id_includes_platform() {
        let id = source_id().unwrap();
        assert!(id.contains("cuda"));
        assert!(id.contains("platform="));
        assert!(id.contains("extract=5"));
    }

    #[test]
    fn required_policy_keeps_cuda_package_enabled() {
        let runtime =
            Runtime::new("unused", crate::ComputePolicy::CudaRequired).expect("create runtime");
        assert!(package_enabled(&runtime));
    }

    #[test]
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    fn wheels_have_dylibs_for_current_platform() {
        for wheel in WHEELS {
            assert!(
                !wheel.dylibs().is_empty(),
                "{} has no dylibs",
                wheel.package
            );
        }
    }

    #[test]
    fn preload_order_follows_wheel_declaration() {
        let tempdir = tempfile::tempdir().unwrap();
        let root = tempdir.path();

        for wheel in WHEELS {
            for dylib in wheel.dylibs() {
                std::fs::write(root.join(dylib), b"ok").unwrap();
            }
        }

        let all_dylibs: Vec<&str> = WHEELS
            .iter()
            .flat_map(|wheel| wheel.dylibs().iter().copied())
            .collect();
        for dylib in &all_dylibs {
            assert!(root.join(dylib).exists());
        }
    }

    #[test]
    fn pinned_wheel_set_is_release_aligned() {
        let packages = WHEELS.iter().map(|wheel| wheel.package).collect::<Vec<_>>();
        assert_eq!(
            packages,
            [
                "nvidia-cuda-runtime/13.1.80",
                "nvidia-cuda-nvrtc/13.1.80",
                "nvidia-cublas/13.2.0.9",
                "nvidia-cufft/12.1.0.31",
                "nvidia-curand/10.4.1.81",
                "nvidia-cudnn-cu13/9.17.0.29",
            ]
        );
    }

    #[test]
    fn cuda_runtime_matches_the_pinned_windows_wheel() {
        let wheel = WHEELS
            .iter()
            .find(|wheel| wheel.package == "nvidia-cudnn-cu13/9.17.0.29")
            .expect("missing pinned cuDNN runtime wheel");

        #[cfg(target_os = "windows")]
        assert_eq!(
            wheel.dylibs(),
            [
                "cudnn64_9.dll",
                "cudnn_adv64_9.dll",
                "cudnn_cnn64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll",
                "cudnn_graph64_9.dll",
                "cudnn_heuristic64_9.dll",
                "cudnn_ops64_9.dll",
            ]
        );
    }

    #[test]
    fn parses_major_minor_from_driver_version() {
        let version = CudaDriverVersion::from_raw(13010);
        assert_eq!(version.major(), 13);
        assert_eq!(version.minor(), 1);
        assert_eq!(version.to_string(), "13.1");
    }

    #[test]
    fn checks_cuda_13_1_threshold() {
        assert!(CudaDriverVersion::from_raw(13010).supports_cuda_13_1());
        assert!(CudaDriverVersion::from_raw(13020).supports_cuda_13_1());
        assert!(!CudaDriverVersion::from_raw(13000).supports_cuda_13_1());
        assert!(!CudaDriverVersion::from_raw(12080).supports_cuda_13_1());
    }

    #[test]
    fn exact_hskify_target_is_accepted() {
        let info = CudaDeviceInfo {
            name: HSKIFY_TARGET_DEVICE_NAME.to_owned(),
            total_memory_bytes: HSKIFY_TARGET_MIN_MEMORY_MIB * MIB,
            compute_capability: HSKIFY_TARGET_COMPUTE_CAPABILITY,
        };
        validate_hskify_cuda_target(CudaDriverVersion::from_raw(13010), &info).unwrap();
    }

    #[test]
    fn hskify_target_rejects_old_driver_and_wrong_device() {
        let info = CudaDeviceInfo {
            name: "NVIDIA GeForce RTX 4090".to_owned(),
            total_memory_bytes: 24_000 * MIB,
            compute_capability: HSKIFY_TARGET_COMPUTE_CAPABILITY,
        };
        let old_driver =
            validate_hskify_cuda_target(CudaDriverVersion::from_raw(13000), &info).unwrap_err();
        assert!(old_driver.to_string().contains("driver API 13.1"));

        let wrong_device =
            validate_hskify_cuda_target(CudaDriverVersion::from_raw(13010), &info).unwrap_err();
        assert!(wrong_device.to_string().contains(HSKIFY_TARGET_DEVICE_NAME));
    }
}
