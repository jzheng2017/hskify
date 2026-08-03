mod hf_hub;

pub mod anime_text;
pub mod aot_inpainting;
pub mod comic_text_bubble_detector;
pub mod comic_text_detector;
pub mod flux2_klein;
pub mod font_detector;
pub mod inpainting;
pub mod lama;
pub mod loading;
pub mod manga_ocr;
pub mod manga_text_segmentation_2025;
pub mod mit48px_ocr;
pub mod ocr;
mod ops;
pub mod paddleocr_vl;
pub mod pp_doclayout_v3;
pub mod probability_map;
pub mod speech_bubble_segmentation;
pub mod types;

pub use types::{FontPrediction, NamedFontPrediction, Quad, TextDirection, TextRegion, TopFont};

use anyhow::{Context, Result};

pub use candle_core::Device;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        koharu_runtime::require_hskify_cuda_target()
            .context("required Hskify CUDA target validation failed")?;
        Device::new_cuda(0).context("failed to create Hskify CUDA context on device 0")
    }
}
