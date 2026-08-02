mod model;

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops::sigmoid;
use image::{DynamicImage, imageops::FilterType};
use koharu_runtime::RuntimeManager;

use crate::{device, loading, probability_map::ProbabilityMap};

const REPO: &str = "mayocream/manga-text-segmentation-2025";
const SAFETENSORS_FILENAME: &str = "model.safetensors";
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
// The comic detector is the high-resolution primary text signal. This model
// supplies missed-text proposals, which are subsequently verified and rebuilt
// against source-resolution glyph evidence before cleanup. Keeping the
// complementary proposal pass below this budget avoids repeating a second
// full-resolution vision pipeline for every reader tile.
const GPU_MAX_PIXELS: u64 = 400 * 960;
const CPU_MAX_PIXELS: u64 = 1_280 * 1_280;
// The recovery pass is page-scoped rather than detector-batch-scoped. Its
// graph has a large fixed launch cost, so a 16 GiB performance deployment
// admits a complete ordinary ten-tile reader strip in one forward.
const GPU_RECOVERY_BATCH_SIZE: u64 = 10;
const GPU_BATCH_PIXEL_BUDGET: u64 = GPU_RECOVERY_BATCH_SIZE * GPU_MAX_PIXELS;
pub const DEFAULT_TEXT_MASK_THRESHOLD: f32 = 0.1;

#[derive(Debug)]
pub struct MangaTextSegmentation {
    model: model::MangaTextSegmentationModel,
    device: Device,
    dtype: DType,
    mean: Tensor,
    std: Tensor,
}

struct PreparedInput {
    pixel_values: Tensor,
    original_width: u32,
    original_height: u32,
    resized_width: u32,
    resized_height: u32,
}

impl MangaTextSegmentation {
    pub async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        let safetensors = resolve_safetensors_path(runtime).await?;
        Self::load_from_path(&safetensors, cpu)
    }

    pub fn load_from_path(path: impl AsRef<Path>, cpu: bool) -> Result<Self> {
        let device = device(cpu)?;
        let dtype = loading::model_dtype(&device);
        let model = loading::load_mmaped_safetensors_path_with_dtype(
            path.as_ref(),
            &device,
            dtype,
            model::MangaTextSegmentationModel::load,
        )?;
        let mean = Tensor::from_slice(&IMAGENET_MEAN, (1, 3, 1, 1), &device)?.to_dtype(dtype)?;
        let std = Tensor::from_slice(&IMAGENET_STD, (1, 3, 1, 1), &device)?.to_dtype(dtype)?;
        Ok(Self {
            model,
            device,
            dtype,
            mean,
            std,
        })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<ProbabilityMap> {
        self.inference_batch(std::slice::from_ref(image))?
            .pop()
            .context("text segmentation returned no result")
    }

    /// Segments multiple equally-shaped reader tiles in one model forward.
    ///
    /// Reader images are tiled to a common geometry, so batching here avoids
    /// launching the full segmentation network once per tile. Mixed edge-tile
    /// shapes remain supported and are split into compatible microbatches.
    pub fn inference_batch(&self, images: &[DynamicImage]) -> Result<Vec<ProbabilityMap>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(images.len());
        let mut offset = 0;
        while offset < images.len() {
            let count = if self.device.is_cuda() {
                compatible_batch_len(&images[offset..], GPU_MAX_PIXELS, GPU_BATCH_PIXEL_BUDGET)
            } else {
                1
            };
            let chunk = &images[offset..offset + count];
            let preprocess_started = Instant::now();
            let prepared = chunk
                .iter()
                .map(|image| self.preprocess(image))
                .collect::<Result<Vec<_>>>()?;
            let preprocess_elapsed = preprocess_started.elapsed();
            let shape = prepared[0].pixel_values.dims4()?;
            if prepared
                .iter()
                .any(|input| input.pixel_values.dims4().ok() != Some(shape))
            {
                for input in prepared {
                    outputs.push(self.forward_prepared(input)?);
                }
                offset += count;
                continue;
            }

            let forward_started = Instant::now();
            let tensors = prepared
                .iter()
                .map(|input| &input.pixel_values)
                .collect::<Vec<_>>();
            let logits = self.model.forward(&Tensor::cat(&tensors, 0)?)?;
            let forward_elapsed = forward_started.elapsed();
            let postprocess_started = Instant::now();
            let probabilities = sigmoid(&logits)?;
            for (index, input) in prepared.iter().enumerate() {
                outputs.push(self.postprocess(&probabilities, index, input)?);
            }
            tracing::info!(
                batch = prepared.len(),
                preprocess_ms = preprocess_elapsed.as_millis(),
                forward_ms = forward_elapsed.as_millis(),
                postprocess_ms = postprocess_started.elapsed().as_millis(),
                "batched manga text segmentation timings"
            );
            if std::env::var_os("HSKIFY_TRACE_PIPELINE_TIMING").is_some_and(|value| value == "1") {
                eprintln!(
                    "hskify-text-segmentation-timing batch={} shape={:?} preprocess_ms={} forward_ms={} postprocess_ms={}",
                    prepared.len(),
                    shape,
                    preprocess_elapsed.as_millis(),
                    forward_elapsed.as_millis(),
                    postprocess_started.elapsed().as_millis(),
                );
            }
            offset += count;
        }
        tracing::info!(
            images = images.len(),
            total_ms = started.elapsed().as_millis(),
            "manga text segmentation batch complete"
        );
        Ok(outputs)
    }

    fn forward_prepared(&self, prepared: PreparedInput) -> Result<ProbabilityMap> {
        let probabilities = sigmoid(&self.model.forward(&prepared.pixel_values)?)?;
        self.postprocess(&probabilities, 0, &prepared)
    }

    fn postprocess(
        &self,
        probabilities: &Tensor,
        batch_index: usize,
        prepared: &PreparedInput,
    ) -> Result<ProbabilityMap> {
        let probabilities = probabilities.i((
            batch_index,
            0,
            0..prepared.resized_height as usize,
            0..prepared.resized_width as usize,
        ))?;
        let probabilities = if prepared.resized_width != prepared.original_width
            || prepared.resized_height != prepared.original_height
        {
            probabilities
                .unsqueeze(0)?
                .unsqueeze(0)?
                .interpolate2d(
                    prepared.original_height as usize,
                    prepared.original_width as usize,
                )?
                .squeeze(0)?
                .squeeze(0)?
        } else {
            probabilities
        }
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
        Ok(ProbabilityMap {
            width: prepared.original_width,
            height: prepared.original_height,
            values: probabilities.flatten_all()?.to_vec1::<f32>()?,
        })
    }

    fn preprocess(&self, image: &DynamicImage) -> Result<PreparedInput> {
        let rgb = image.to_rgb8();
        let (original_width, original_height) = rgb.dimensions();
        let (resized_width, resized_height) = scaled_dimensions(
            original_width,
            original_height,
            if self.device.is_cuda() {
                GPU_MAX_PIXELS
            } else {
                CPU_MAX_PIXELS
            },
        );
        let rgb = if resized_width == original_width && resized_height == original_height {
            rgb
        } else {
            image::imageops::resize(&rgb, resized_width, resized_height, FilterType::Triangle)
        };
        let pad_h = (32 - resized_height % 32) % 32;
        let pad_w = (32 - resized_width % 32) % 32;

        let tensor = Tensor::from_vec(
            rgb.into_raw(),
            (1, resized_height as usize, resized_width as usize, 3),
            &self.device,
        )?
        .permute((0, 3, 1, 2))?
        .to_dtype(self.dtype)?;
        let tensor = (tensor * (1.0 / 255.0))?
            .broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)?;
        let tensor = tensor
            .pad_with_zeros(2, 0, pad_h as usize)?
            .pad_with_zeros(3, 0, pad_w as usize)?;

        Ok(PreparedInput {
            pixel_values: tensor,
            original_width,
            original_height,
            resized_width,
            resized_height,
        })
    }
}

fn compatible_batch_len(
    images: &[DynamicImage],
    per_image_max_pixels: u64,
    batch_pixel_budget: u64,
) -> usize {
    let Some(first) = images.first() else {
        return 0;
    };
    let first_size = scaled_dimensions(first.width(), first.height(), per_image_max_pixels);
    let first_pixels = u64::from(first_size.0) * u64::from(first_size.1);
    let maximum = (batch_pixel_budget / first_pixels.max(1)).max(1) as usize;
    images
        .iter()
        .take(maximum)
        .take_while(|image| {
            scaled_dimensions(image.width(), image.height(), per_image_max_pixels) == first_size
        })
        .count()
        .max(1)
}

pub async fn prefetch(runtime: &RuntimeManager) -> Result<()> {
    let _ = resolve_safetensors_path(runtime).await?;
    Ok(())
}

async fn resolve_safetensors_path(runtime: &RuntimeManager) -> Result<PathBuf> {
    runtime
        .downloads()
        .huggingface_model(REPO, SAFETENSORS_FILENAME)
        .await
        .with_context(|| format!("failed to download {SAFETENSORS_FILENAME} from {REPO}"))
}

fn scaled_dimensions(width: u32, height: u32, max_pixels: u64) -> (u32, u32) {
    let area = u64::from(width) * u64::from(height);
    if area <= max_pixels || max_pixels == 0 {
        return (width.max(1), height.max(1));
    }

    let scale = (max_pixels as f64 / area as f64).sqrt();
    let mut scaled_width = ((width as f64 * scale).floor() as u32).clamp(1, width.max(1));
    let mut scaled_height = ((height as f64 * scale).floor() as u32).clamp(1, height.max(1));
    while u64::from(scaled_width) * u64::from(scaled_height) > max_pixels {
        if scaled_width >= scaled_height && scaled_width > 1 {
            scaled_width -= 1;
        } else if scaled_height > 1 {
            scaled_height -= 1;
        } else {
            break;
        }
    }
    (scaled_width, scaled_height)
}

#[cfg(test)]
mod tests {
    use image::DynamicImage;

    use super::{compatible_batch_len, scaled_dimensions};

    #[test]
    fn scaled_dimensions_leave_small_inputs_unchanged() {
        assert_eq!(scaled_dimensions(800, 1200, 2_000_000), (800, 1200));
    }

    #[test]
    fn scaled_dimensions_reduce_large_inputs_to_budget() {
        let (width, height) = scaled_dimensions(3000, 4000, 2_000_000);
        assert!(u64::from(width) * u64::from(height) <= 2_000_000);
        assert!(width < 3000);
        assert!(height < 4000);
    }

    #[test]
    fn compatible_batches_are_limited_by_shape_and_total_pixels() {
        let same = DynamicImage::new_rgb8(900, 2_048);
        let edge = DynamicImage::new_rgb8(900, 1_024);
        assert_eq!(
            compatible_batch_len(
                &[same.clone(), same.clone(), same],
                640 * 960,
                2 * 640 * 960
            ),
            2
        );
        assert_eq!(
            compatible_batch_len(
                &[DynamicImage::new_rgb8(900, 2_048), edge],
                640 * 960,
                2 * 640 * 960
            ),
            1
        );
    }
}
