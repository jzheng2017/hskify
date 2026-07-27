//! CUDA PP-OCRv5 mobile text detection.
//!
//! The preprocessing constants and DB postprocessing follow the Apache-2.0
//! `paddle-ocr-rs` implementation at commit
//! `10b467f0082ad2c60ce73f913c7b1217833014c9`. Hskify keeps the code local so
//! the detector can share the workspace's exact ONNX Runtime build, execute a
//! true batch, and fail closed when CUDA is unavailable.

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::contours::{Contour, find_contours};
use imageproc::distance_transform::Norm;
use imageproc::drawing::draw_polygon_mut;
use imageproc::geometry::min_area_rect;
use imageproc::morphology::dilate;
use imageproc::point::Point;
use ort::ep::{CUDA, ExecutionProvider};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

pub(super) const DETECTOR_TILE_BATCH_SIZE: usize = 6;

const MODEL_MAX_SIDE: u32 = 1_280;
const PROBABILITY_THRESHOLD: f32 = 0.30;
const BOX_SCORE_THRESHOLD: f32 = 0.50;
const BOX_EXPANSION_RATIO: f32 = 1.60;
const MINIMUM_BOX_SIDE: f32 = 5.0;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STANDARD_DEVIATION: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PpOcrTextDetection {
    pub(super) bounds: [f32; 4],
    pub(super) score: f32,
}

#[derive(Debug, Clone, Copy)]
struct Scale {
    source_width: u32,
    source_height: u32,
    model_width: u32,
    model_height: u32,
}

impl Scale {
    fn for_image(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("PP-OCRv5 detector input dimensions must be non-zero");
        }
        let ratio = MODEL_MAX_SIDE as f32 / width.max(height) as f32;
        let model_width = round_down_to_32((width as f32 * ratio) as u32);
        let model_height = round_down_to_32((height as f32 * ratio) as u32);
        Ok(Self {
            source_width: width,
            source_height: height,
            model_width,
            model_height,
        })
    }

    fn source_x(self, model_x: f32) -> f32 {
        model_x * self.source_width as f32 / self.model_width as f32
    }

    fn source_y(self, model_y: f32) -> f32 {
        model_y * self.source_height as f32 / self.model_height as f32
    }
}

pub(super) struct PpOcrV5TextDetector {
    session: Session,
    input_buffer: Vec<f32>,
}

impl PpOcrV5TextDetector {
    pub(super) fn load(model_path: &Path) -> Result<Self> {
        let cuda = CUDA::default().with_device_id(0);
        if !cuda
            .is_available()
            .context("query ONNX Runtime CUDA availability for PP-OCRv5 detection")?
        {
            bail!("PP-OCRv5 detection requires the CUDA execution provider");
        }
        let session = Session::builder()
            .context("create PP-OCRv5 detector session builder")?
            .with_no_environment_execution_providers()
            .map_err(|error| {
                anyhow::anyhow!("ignore environment-provided detector execution providers: {error}")
            })?
            .with_execution_providers([cuda.build().error_on_failure()])
            .map_err(|error| {
                anyhow::anyhow!("register mandatory CUDA for PP-OCRv5 detection: {error}")
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| {
                anyhow::anyhow!("enable PP-OCRv5 detector graph optimizations: {error}")
            })?
            .with_memory_pattern(false)
            .map_err(|error| {
                anyhow::anyhow!(
                    "disable static memory patterns for dynamic detector batches: {error}"
                )
            })?
            .commit_from_file(model_path)
            .with_context(|| {
                format!(
                    "load pinned PP-OCRv5 mobile detector {}",
                    model_path.display()
                )
            })?;
        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            bail!("pinned PP-OCRv5 detector must expose exactly one input and one output");
        }
        Ok(Self {
            session,
            input_buffer: Vec::new(),
        })
    }

    pub(super) fn detect_tiles(
        &mut self,
        tiles: &[DynamicImage],
    ) -> Result<Vec<Vec<PpOcrTextDetection>>> {
        if tiles.is_empty() || tiles.len() > DETECTOR_TILE_BATCH_SIZE {
            bail!("PP-OCRv5 detector batch must contain 1..={DETECTOR_TILE_BATCH_SIZE} tiles");
        }
        let scales = tiles
            .iter()
            .map(|tile| Scale::for_image(tile.width(), tile.height()))
            .collect::<Result<Vec<_>>>()?;
        let model_width = scales
            .iter()
            .map(|scale| scale.model_width)
            .max()
            .expect("detector batch is non-empty");
        let model_height = scales
            .iter()
            .map(|scale| scale.model_height)
            .max()
            .expect("detector batch is non-empty");
        prepare_batch(
            tiles,
            &scales,
            model_width,
            model_height,
            &mut self.input_buffer,
        )?;
        let input = TensorRef::from_array_view((
            [tiles.len(), 3, model_height as usize, model_width as usize],
            self.input_buffer.as_slice(),
        ))
        .context("bind PP-OCRv5 detector input batch")?;
        let outputs = self
            .session
            .run(ort::inputs![input])
            .context("run true-batched CUDA PP-OCRv5 detection")?;
        let output = outputs
            .iter()
            .next()
            .map(|(_, output)| output)
            .context("PP-OCRv5 detector returned no output")?;
        let (shape, probabilities) = output
            .try_extract_tensor::<f32>()
            .context("extract PP-OCRv5 detector probability map")?;
        let expected_shape = [
            tiles.len() as i64,
            1,
            model_height as i64,
            model_width as i64,
        ];
        if shape.as_ref() != expected_shape {
            bail!("unexpected PP-OCRv5 detector output shape {shape}; expected {expected_shape:?}");
        }
        let plane = (model_width as usize)
            .checked_mul(model_height as usize)
            .context("PP-OCRv5 output plane overflowed")?;
        if probabilities.len() != plane.saturating_mul(tiles.len()) {
            bail!("PP-OCRv5 detector returned an incomplete probability batch");
        }
        scales
            .iter()
            .enumerate()
            .map(|(batch_index, scale)| {
                let start = batch_index * plane;
                postprocess_probability_map(
                    &probabilities[start..start + plane],
                    model_width,
                    model_height,
                    *scale,
                )
            })
            .collect()
    }
}

fn round_down_to_32(value: u32) -> u32 {
    (value / 32).max(1) * 32
}

fn normalize(channel: u8, index: usize) -> f32 {
    (channel as f32 / 255.0 - MEAN[index]) / STANDARD_DEVIATION[index]
}

fn prepare_batch(
    tiles: &[DynamicImage],
    scales: &[Scale],
    model_width: u32,
    model_height: u32,
    buffer: &mut Vec<f32>,
) -> Result<()> {
    let plane = (model_width as usize)
        .checked_mul(model_height as usize)
        .context("PP-OCRv5 detector input plane overflowed")?;
    let sample_elements = plane
        .checked_mul(3)
        .context("PP-OCRv5 detector sample size overflowed")?;
    let elements = sample_elements
        .checked_mul(tiles.len())
        .context("PP-OCRv5 detector batch size overflowed")?;
    buffer.resize(elements, 0.0);
    for (batch_index, (tile, scale)) in tiles.iter().zip(scales).enumerate() {
        if scale.model_width > model_width || scale.model_height > model_height {
            bail!("PP-OCRv5 detector scale exceeds the allocated batch tensor");
        }
        let batch_offset = batch_index * sample_elements;
        for channel in 0..3 {
            buffer[batch_offset + channel * plane..batch_offset + (channel + 1) * plane]
                .fill(normalize(255, channel));
        }
        let resized = image::imageops::resize(
            &tile.to_rgb8(),
            scale.model_width,
            scale.model_height,
            FilterType::Triangle,
        );
        for (x, y, pixel) in resized.enumerate_pixels() {
            let position = y as usize * model_width as usize + x as usize;
            for channel in 0..3 {
                buffer[batch_offset + channel * plane + position] =
                    normalize(pixel[channel], channel);
            }
        }
    }
    Ok(())
}

fn postprocess_probability_map(
    probabilities: &[f32],
    batch_width: u32,
    batch_height: u32,
    scale: Scale,
) -> Result<Vec<PpOcrTextDetection>> {
    let mut local_probabilities =
        Vec::with_capacity((scale.model_width * scale.model_height) as usize);
    for y in 0..scale.model_height {
        let start = (y * batch_width) as usize;
        local_probabilities
            .extend_from_slice(&probabilities[start..start + scale.model_width as usize]);
    }
    let threshold_values = local_probabilities
        .iter()
        .map(|probability| {
            if probability.is_finite() && *probability >= PROBABILITY_THRESHOLD {
                255
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let threshold = GrayImage::from_vec(scale.model_width, scale.model_height, threshold_values)
        .context("construct PP-OCRv5 detector threshold image")?;
    let threshold = dilate(&threshold, Norm::LInf, 1);
    let probability_image = ImageBuffer::<Luma<f32>, Vec<f32>>::from_vec(
        scale.model_width,
        scale.model_height,
        local_probabilities,
    )
    .context("construct PP-OCRv5 detector probability image")?;

    let mut detections = find_contours::<i32>(&threshold)
        .into_iter()
        .filter_map(|contour| {
            detection_from_contour(&contour, &probability_image, scale).transpose()
        })
        .collect::<Result<Vec<_>>>()?;
    detections.sort_by(|left, right| {
        left.bounds[1]
            .total_cmp(&right.bounds[1])
            .then_with(|| left.bounds[0].total_cmp(&right.bounds[0]))
            .then_with(|| right.score.total_cmp(&left.score))
    });
    let _ = batch_height;
    Ok(detections)
}

fn detection_from_contour(
    contour: &Contour<i32>,
    probabilities: &ImageBuffer<Luma<f32>, Vec<f32>>,
    scale: Scale,
) -> Result<Option<PpOcrTextDetection>> {
    if contour.points.len() <= 2 {
        return Ok(None);
    }
    let rectangle = min_area_rect(&contour.points);
    let minimum_side = rectangle
        .windows(2)
        .map(|pair| {
            let dx = pair[0].x as f32 - pair[1].x as f32;
            let dy = pair[0].y as f32 - pair[1].y as f32;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(f32::INFINITY, f32::min);
    if minimum_side < 3.0 {
        return Ok(None);
    }
    let score = contour_score(contour, probabilities)?;
    if !score.is_finite() || score < BOX_SCORE_THRESHOLD {
        return Ok(None);
    }
    let Some(minimum_x) = rectangle.iter().map(|point| point.x).min() else {
        return Ok(None);
    };
    let Some(maximum_x) = rectangle.iter().map(|point| point.x).max() else {
        return Ok(None);
    };
    let Some(minimum_y) = rectangle.iter().map(|point| point.y).min() else {
        return Ok(None);
    };
    let Some(maximum_y) = rectangle.iter().map(|point| point.y).max() else {
        return Ok(None);
    };
    let minimum_x = minimum_x as f32;
    let maximum_x = maximum_x as f32;
    let minimum_y = minimum_y as f32;
    let maximum_y = maximum_y as f32;
    let center_x = (minimum_x + maximum_x) * 0.5;
    let center_y = (minimum_y + maximum_y) * 0.5;
    let half_width = (maximum_x - minimum_x) * 0.5 * BOX_EXPANSION_RATIO;
    let half_height = (maximum_y - minimum_y) * 0.5 * BOX_EXPANSION_RATIO;
    let left = scale
        .source_x(center_x - half_width)
        .clamp(0.0, scale.source_width as f32);
    let top = scale
        .source_y(center_y - half_height)
        .clamp(0.0, scale.source_height as f32);
    let right = scale
        .source_x(center_x + half_width)
        .clamp(0.0, scale.source_width as f32);
    let bottom = scale
        .source_y(center_y + half_height)
        .clamp(0.0, scale.source_height as f32);
    if right - left < MINIMUM_BOX_SIDE || bottom - top < MINIMUM_BOX_SIDE {
        return Ok(None);
    }
    Ok(Some(PpOcrTextDetection {
        bounds: [left, top, right, bottom],
        score,
    }))
}

fn contour_score(
    contour: &Contour<i32>,
    probabilities: &ImageBuffer<Luma<f32>, Vec<f32>>,
) -> Result<f32> {
    let minimum_x = contour
        .points
        .iter()
        .map(|point| point.x)
        .min()
        .unwrap_or(0);
    let maximum_x = contour
        .points
        .iter()
        .map(|point| point.x)
        .max()
        .unwrap_or(-1);
    let minimum_y = contour
        .points
        .iter()
        .map(|point| point.y)
        .min()
        .unwrap_or(0);
    let maximum_y = contour
        .points
        .iter()
        .map(|point| point.y)
        .max()
        .unwrap_or(-1);
    if maximum_x < minimum_x || maximum_y < minimum_y {
        return Ok(0.0);
    }
    let left = minimum_x.max(0) as u32;
    let top = minimum_y.max(0) as u32;
    let right = (maximum_x + 1).clamp(0, probabilities.width() as i32) as u32;
    let bottom = (maximum_y + 1).clamp(0, probabilities.height() as i32) as u32;
    if right <= left || bottom <= top {
        return Ok(0.0);
    }
    let mut mask = GrayImage::new(right - left, bottom - top);
    let local = contour
        .points
        .iter()
        .map(|point| Point::new(point.x - left as i32, point.y - top as i32))
        .collect::<Vec<_>>();
    draw_polygon_mut(&mut mask, &local, Luma([255]));
    let mut total = 0.0_f32;
    let mut count = 0_u32;
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            if mask.get_pixel(x, y)[0] != 0 {
                total += probabilities.get_pixel(left + x, top + y)[0];
                count += 1;
            }
        }
    }
    Ok(if count == 0 {
        0.0
    } else {
        total / count as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_is_multiple_of_32_and_preserves_source_mapping() {
        let scale = Scale::for_image(900, 1_024).unwrap();
        assert_eq!((scale.model_width, scale.model_height), (1_120, 1_280));
        assert!((scale.source_x(1_120.0) - 900.0).abs() < f32::EPSILON);
        assert!((scale.source_y(1_280.0) - 1_024.0).abs() < f32::EPSILON);
    }

    #[test]
    fn preprocessing_pads_a_real_batch_with_white() {
        let first = DynamicImage::new_rgb8(64, 64);
        let second = DynamicImage::new_rgb8(32, 64);
        let scales = vec![
            Scale::for_image(64, 64).unwrap(),
            Scale::for_image(32, 64).unwrap(),
        ];
        let mut buffer = Vec::new();
        prepare_batch(&[first, second], &scales, 1_280, 1_280, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 2 * 3 * 1_280 * 1_280);
        assert!(buffer.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn probability_postprocessing_finds_a_text_component() {
        let scale = Scale {
            source_width: 64,
            source_height: 64,
            model_width: 64,
            model_height: 64,
        };
        let mut probabilities = vec![0.0; 64 * 64];
        for y in 20..30 {
            for x in 10..50 {
                probabilities[y * 64 + x] = 0.95;
            }
        }
        let detections = postprocess_probability_map(&probabilities, 64, 64, scale).unwrap();
        assert_eq!(detections.len(), 1);
        assert!(detections[0].score >= BOX_SCORE_THRESHOLD);
        assert!(detections[0].bounds[0] <= 10.0);
        assert!(detections[0].bounds[2] >= 49.0);
    }
}
