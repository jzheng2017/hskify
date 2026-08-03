//! PP-OCRv6-small text-line detection.
//!
//! The comic object detector is useful for bubble topology, but it is not a
//! complete text detector: narration, free text, rotated lettering, and
//! decorative text can all exist outside its object classes.  This adapter
//! owns the independent text-line proposals used by the recognizer.  It
//! deliberately returns geometry and calibrated visual confidence only; no
//! lexical or artwork-specific decisions are made here.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use koharu_ml::ocr::{TextDetection, TextDetector, TextRect};
use ort::ep::{CUDA, ExecutionProvider};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

const MODEL_NAME: &str = "PP-OCRv6_small_det";
const INPUT_NAME: &str = "x";
const OUTPUT_NAME: &str = "fetch_name_0";
const BATCH_LIMIT: usize = 6;
const MODEL_SIDE: usize = 736;
const MAP_THRESHOLD: f32 = 0.20;
const BOX_THRESHOLD: f32 = 0.45;
const UNCLIP_RATIO: f32 = 1.40;
const NMS_IOU: f32 = 0.50;

#[derive(Debug, Clone, Copy)]
struct DetectorTransform {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    /// Dimensions of the source image after aspect-preserving resize, in the
    /// detector's 736px model canvas.  DB maps can contain activations in the
    /// letterbox padding; those pixels are not source evidence and must never
    /// become text boxes.
    content_width: f32,
    content_height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PpOcrTextDetection {
    /// Bounds in source-image pixels, local to the supplied tile.
    pub(super) left: f32,
    pub(super) top: f32,
    pub(super) right: f32,
    pub(super) bottom: f32,
    pub(super) confidence: f32,
    /// Principal-axis angle of the connected text probability component.
    pub(super) rotation_radians: f32,
}

impl PpOcrTextDetection {
    fn area(self) -> f32 {
        (self.right - self.left).max(0.0) * (self.bottom - self.top).max(0.0)
    }

    fn iou(self, other: Self) -> f32 {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right.min(other.right);
        let bottom = self.bottom.min(other.bottom);
        let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
        if intersection <= 0.0 {
            return 0.0;
        }
        intersection / (self.area() + other.area() - intersection).max(f32::EPSILON)
    }
}

pub(super) struct PpOcrSmallDetector {
    session: Session,
    input_buffer: Vec<f32>,
}

impl TextDetector for PpOcrSmallDetector {
    fn detect_text(&mut self, image: &DynamicImage) -> Result<Vec<TextDetection>> {
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }
        let detections = self
            .detect_tiles(std::slice::from_ref(image))?
            .into_iter()
            .next()
            .unwrap_or_default();
        Ok(detections
            .into_iter()
            .map(|detection| {
                TextDetection::new(
                    TextRect::new(
                        detection.left / width as f32,
                        detection.top / height as f32,
                        detection.right / width as f32,
                        detection.bottom / height as f32,
                    ),
                    detection.confidence,
                    detection.rotation_radians,
                )
            })
            .collect())
    }
}

impl PpOcrSmallDetector {
    pub(super) fn load(model_path: &Path, config_path: &Path) -> Result<Self> {
        validate_config(config_path)?;
        let cuda = CUDA::default().with_device_id(0);
        if !cuda
            .is_available()
            .context("query ONNX Runtime CUDA execution-provider availability")?
        {
            bail!("ONNX Runtime was not compiled with its mandatory CUDA execution provider");
        }
        let session = Session::builder()
            .context("create PP-OCR detector ONNX Runtime session builder")?
            .with_no_environment_execution_providers()
            .map_err(|error| anyhow::anyhow!("clear environment ONNX providers: {error}"))?
            .with_execution_providers([cuda.build().error_on_failure()])
            .map_err(|error| anyhow::anyhow!("register PP-OCR detector CUDA provider: {error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| {
                anyhow::anyhow!("enable PP-OCR detector graph optimizations: {error}")
            })?
            .with_memory_pattern(false)
            .map_err(|error| {
                anyhow::anyhow!("disable PP-OCR detector static memory patterns: {error}")
            })?
            .commit_from_file(model_path)
            .with_context(|| format!("load PP-OCRv6-small detector {}", model_path.display()))?;
        if session.inputs().len() != 1 || session.inputs()[0].name() != INPUT_NAME {
            bail!("pinned PP-OCR detector input contract changed");
        }
        if session.outputs().len() != 1 || session.outputs()[0].name() != OUTPUT_NAME {
            bail!("pinned PP-OCR detector output contract changed");
        }
        Ok(Self {
            session,
            input_buffer: Vec::new(),
        })
    }

    /// Detect text components in one bounded tile batch.  The ONNX graph is
    /// run once for the whole batch; DB post-processing is CPU-side and
    /// independent per tile, so it does not serialize the CUDA language path.
    pub(super) fn detect_tiles(
        &mut self,
        tiles: &[DynamicImage],
    ) -> Result<Vec<Vec<PpOcrTextDetection>>> {
        if tiles.is_empty() {
            return Ok(Vec::new());
        }
        if tiles.len() > BATCH_LIMIT {
            bail!(
                "PP-OCR detector batch has {} tiles; maximum is {BATCH_LIMIT}",
                tiles.len()
            );
        }
        let transforms = preprocess_batch(tiles, &mut self.input_buffer);
        let batch = tiles.len();
        let input = TensorRef::from_array_view((
            [batch, 3, MODEL_SIDE, MODEL_SIDE],
            self.input_buffer.as_slice(),
        ))
        .context("bind PP-OCR detector input tensor")?;
        let outputs = self
            .session
            .run(ort::inputs![input])
            .context("run PP-OCRv6-small text detector")?;
        let output = &outputs[OUTPUT_NAME];
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .context("extract PP-OCR detector probability map")?;
        let (map_height, map_width) = probability_shape(shape, values.len(), batch)?;
        let per_sample = map_height
            .checked_mul(map_width)
            .context("PP-OCR detector probability map size overflowed")?;
        let mut result = Vec::with_capacity(batch);
        for (sample, tile) in tiles.iter().enumerate() {
            let start = sample
                .checked_mul(per_sample)
                .context("PP-OCR detector output offset overflowed")?;
            let end = start + per_sample;
            result.push(db_components(
                &values[start..end],
                map_width,
                map_height,
                tile.dimensions(),
                transforms[sample],
            ));
        }
        Ok(result)
    }
}

fn validate_config(path: &Path) -> Result<()> {
    let config = fs::read_to_string(path)
        .with_context(|| format!("read PP-OCR detector config {}", path.display()))?;
    if !config
        .lines()
        .any(|line| line.trim().strip_prefix("model_name: ") == Some(MODEL_NAME))
    {
        bail!("PP-OCR detector config is not {MODEL_NAME}");
    }
    for required in ["name: DBPostProcess", "box_thresh: 0.45", "thresh: 0.2"] {
        if !config.lines().any(|line| line.trim() == required) {
            bail!("PP-OCR detector config is missing {required}");
        }
    }
    Ok(())
}

fn preprocess_batch(tiles: &[DynamicImage], buffer: &mut Vec<f32>) -> Vec<DetectorTransform> {
    let channels = 3 * MODEL_SIDE * MODEL_SIDE;
    buffer.clear();
    buffer.resize(tiles.len() * channels, 0.0);
    let mut transforms = Vec::with_capacity(tiles.len());
    for (sample, tile) in tiles.iter().enumerate() {
        let (source_width, source_height) = tile.dimensions();
        let scale = (MODEL_SIDE as f32 / source_width.max(1) as f32)
            .min(MODEL_SIDE as f32 / source_height.max(1) as f32);
        let resized_width = ((source_width as f32 * scale).round() as usize).clamp(32, MODEL_SIDE);
        let resized_height =
            ((source_height as f32 * scale).round() as usize).clamp(32, MODEL_SIDE);
        let pad_x = (MODEL_SIDE - resized_width) / 2;
        let pad_y = (MODEL_SIDE - resized_height) / 2;
        let rgb = tile
            .resize_exact(
                resized_width as u32,
                resized_height as u32,
                FilterType::Triangle,
            )
            .to_rgb8();
        let plane = MODEL_SIDE * MODEL_SIDE;
        for y in 0..resized_height {
            for x in 0..resized_width {
                let [red, green, blue] = rgb.get_pixel(x as u32, y as u32).0;
                let model_y = y + pad_y;
                let model_x = x + pad_x;
                let offset = sample * channels + model_y * MODEL_SIDE + model_x;
                // The Paddle graph declares BGR input, HWC normalization, and
                // CHW conversion.  Keep that contract explicit here.
                buffer[offset] = (blue as f32 / 255.0 - 0.485) / 0.229;
                buffer[offset + plane] = (green as f32 / 255.0 - 0.456) / 0.224;
                buffer[offset + 2 * plane] = (red as f32 / 255.0 - 0.406) / 0.225;
            }
        }
        transforms.push(DetectorTransform {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            content_width: resized_width as f32,
            content_height: resized_height as f32,
        });
    }
    transforms
}

fn probability_shape(shape: &[i64], values: usize, batch: usize) -> Result<(usize, usize)> {
    let (map_height, map_width, shape_elements) = match shape {
        [batch_shape, _channels, height, width] => {
            if usize::try_from(*batch_shape).ok() != Some(batch) {
                bail!("PP-OCR detector returned a mismatched batch dimension");
            }
            let height = usize::try_from(*height).context("invalid PP-OCR detector height")?;
            let width = usize::try_from(*width).context("invalid PP-OCR detector width")?;
            (height, width, batch * height * width)
        }
        [batch_shape, height, width] => {
            if usize::try_from(*batch_shape).ok() != Some(batch) {
                bail!("PP-OCR detector returned a mismatched batch dimension");
            }
            let height = usize::try_from(*height).context("invalid PP-OCR detector height")?;
            let width = usize::try_from(*width).context("invalid PP-OCR detector width")?;
            (height, width, batch * height * width)
        }
        _ => bail!("PP-OCR detector output must be [N,1,H,W] or [N,H,W]"),
    };
    if map_height == 0 || map_width == 0 || shape_elements != values {
        bail!("PP-OCR detector output shape does not match its tensor data");
    }
    Ok((map_height, map_width))
}

fn db_components(
    probabilities: &[f32],
    width: usize,
    height: usize,
    source_dimensions: (u32, u32),
    transform: DetectorTransform,
) -> Vec<PpOcrTextDetection> {
    let mut visited = vec![false; probabilities.len()];
    let mut detections = Vec::new();
    for index in 0..probabilities.len() {
        let x = index % width;
        let y = index / width;
        if visited[index]
            || !map_point_is_content(x, y, width, height, transform)
            || probability(probabilities[index]) < MAP_THRESHOLD
        {
            continue;
        }
        let mut queue = vec![index];
        visited[index] = true;
        let mut points = Vec::new();
        while let Some(current) = queue.pop() {
            points.push(current);
            let x = current % width;
            let y = current / width;
            for (nx, ny) in neighbors(x, y, width, height) {
                let next = ny * width + nx;
                if !visited[next]
                    && map_point_is_content(nx, ny, width, height, transform)
                    && probability(probabilities[next]) >= MAP_THRESHOLD
                {
                    visited[next] = true;
                    queue.push(next);
                }
            }
        }
        let Some(detection) = component_detection(
            &points,
            probabilities,
            width,
            height,
            source_dimensions,
            transform,
        ) else {
            continue;
        };
        if detection.confidence >= BOX_THRESHOLD {
            detections.push(detection);
        }
    }
    detections.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.top.total_cmp(&right.top))
            .then_with(|| left.left.total_cmp(&right.left))
    });
    let mut kept = Vec::with_capacity(detections.len());
    for detection in detections {
        if kept.iter().all(|other| detection.iou(*other) < NMS_IOU) {
            kept.push(detection);
        }
    }
    kept.sort_by(|left, right| {
        left.top
            .total_cmp(&right.top)
            .then_with(|| left.left.total_cmp(&right.left))
    });
    kept
}

fn map_point_is_content(
    x: usize,
    y: usize,
    map_width: usize,
    map_height: usize,
    transform: DetectorTransform,
) -> bool {
    if map_width == 0 || map_height == 0 {
        return false;
    }
    let model_x = (x as f32 + 0.5) * MODEL_SIDE as f32 / map_width as f32;
    let model_y = (y as f32 + 0.5) * MODEL_SIDE as f32 / map_height as f32;
    model_x >= transform.pad_x
        && model_y >= transform.pad_y
        && model_x < transform.pad_x + transform.content_width
        && model_y < transform.pad_y + transform.content_height
}

fn component_detection(
    points: &[usize],
    probabilities: &[f32],
    width: usize,
    height: usize,
    source_dimensions: (u32, u32),
    transform: DetectorTransform,
) -> Option<PpOcrTextDetection> {
    if points.len() < 4 {
        return None;
    }
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut weight_sum = 0.0_f32;
    let mut weighted_x = 0.0_f32;
    let mut weighted_y = 0.0_f32;
    for &point in points {
        let x = point % width;
        let y = point / width;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        let weight = probability(probabilities[point]).max(MAP_THRESHOLD);
        weight_sum += weight;
        weighted_x += x as f32 * weight;
        weighted_y += y as f32 * weight;
    }
    let confidence = weight_sum / points.len() as f32;
    let width_pixels = max_x.saturating_sub(min_x).saturating_add(1);
    let height_pixels = max_y.saturating_sub(min_y).saturating_add(1);
    if width_pixels < 3 || height_pixels < 3 {
        return None;
    }
    let unclip_x = width_pixels as f32 * (UNCLIP_RATIO - 1.0) * 0.25;
    let unclip_y = height_pixels as f32 * (UNCLIP_RATIO - 1.0) * 0.25;
    let map_to_model_x = MODEL_SIDE as f32 / width.max(1) as f32;
    let map_to_model_y = MODEL_SIDE as f32 / height.max(1) as f32;
    let unclip_model_x = unclip_x * map_to_model_x;
    let unclip_model_y = unclip_y * map_to_model_y;
    let left = ((min_x as f32 * map_to_model_x - transform.pad_x - unclip_model_x)
        / transform.scale)
        .clamp(0.0, source_dimensions.0 as f32);
    let top = ((min_y as f32 * map_to_model_y - transform.pad_y - unclip_model_y)
        / transform.scale)
        .clamp(0.0, source_dimensions.1 as f32);
    let right = (((max_x + 1) as f32 * map_to_model_x - transform.pad_x + unclip_model_x)
        / transform.scale)
        .clamp(0.0, source_dimensions.0 as f32);
    let bottom = (((max_y + 1) as f32 * map_to_model_y - transform.pad_y + unclip_model_y)
        / transform.scale)
        .clamp(0.0, source_dimensions.1 as f32);
    let mean_x = weighted_x / weight_sum.max(f32::EPSILON);
    let mean_y = weighted_y / weight_sum.max(f32::EPSILON);
    let (mut covariance_xx, mut covariance_yy, mut covariance_xy) = (0.0, 0.0, 0.0);
    for &point in points {
        let x = point % width;
        let y = point / width;
        let weight = probability(probabilities[point]).max(MAP_THRESHOLD);
        let dx = x as f32 - mean_x;
        let dy = y as f32 - mean_y;
        covariance_xx += weight * dx * dx;
        covariance_yy += weight * dy * dy;
        covariance_xy += weight * dx * dy;
    }
    let rotation_radians = 0.5 * (2.0 * covariance_xy).atan2(covariance_xx - covariance_yy);
    Some(PpOcrTextDetection {
        left,
        top,
        right,
        bottom,
        confidence: confidence.clamp(0.0, 1.0),
        rotation_radians: if rotation_radians.is_finite() {
            rotation_radians
        } else {
            0.0
        },
    })
}

fn probability(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    if (0.0..=1.0).contains(&value) {
        value
    } else {
        1.0 / (1.0 + (-value).exp())
    }
}

fn neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    [
        x.checked_sub(1).map(|x| (x, y)),
        x.checked_add(1).filter(|x| *x < width).map(|x| (x, y)),
        y.checked_sub(1).map(|y| (x, y)),
        y.checked_add(1).filter(|y| *y < height).map(|y| (x, y)),
        x.checked_sub(1).zip(y.checked_sub(1)),
        x.checked_add(1)
            .filter(|x| *x < width)
            .zip(y.checked_sub(1)),
        x.checked_sub(1)
            .zip(y.checked_add(1).filter(|y| *y < height)),
        x.checked_add(1)
            .filter(|x| *x < width)
            .zip(y.checked_add(1).filter(|y| *y < height)),
    ]
    .into_iter()
    .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_postprocess_returns_ordered_components_and_suppresses_noise() {
        let mut map = vec![0.0; 32 * 32];
        for y in 4..8 {
            for x in 3..14 {
                map[y * 32 + x] = 0.9;
            }
        }
        map[30 * 32 + 30] = 0.99;
        let detections = db_components(
            &map,
            32,
            32,
            (320, 320),
            DetectorTransform {
                scale: 1.0,
                pad_x: 0.0,
                pad_y: 0.0,
                content_width: MODEL_SIDE as f32,
                content_height: MODEL_SIDE as f32,
            },
        );
        assert_eq!(detections.len(), 1);
        assert!(detections[0].left < detections[0].right);
        assert!(detections[0].top < detections[0].bottom);
        assert!(detections[0].confidence > 0.8);
    }

    #[test]
    fn probability_accepts_probability_maps_and_logits() {
        assert_eq!(probability(0.8), 0.8);
        assert_eq!(probability(0.0), 0.0);
        assert!(probability(2.0) > 0.8);
        assert!(probability(f32::NAN) == 0.0);
    }

    #[test]
    fn output_shape_requires_one_map_per_input() {
        assert!(probability_shape(&[2, 1, 8, 8], 128, 2).is_ok());
        assert!(probability_shape(&[2, 8, 8], 128, 2).is_ok());
        assert!(probability_shape(&[1, 8, 8], 64, 2).is_err());
    }

    #[test]
    fn preprocessing_preserves_aspect_ratio_inside_the_detector_canvas() {
        let tiles = vec![
            DynamicImage::new_rgb8(320, 736),
            DynamicImage::new_rgb8(736, 736),
        ];
        let mut buffer = Vec::new();
        let transforms = preprocess_batch(&tiles, &mut buffer);
        assert_eq!(transforms.len(), 2);
        assert_eq!(transforms[0].scale, 1.0);
        assert!(transforms[0].pad_x > 0.0);
        assert_eq!(transforms[0].pad_y, 0.0);
        assert_eq!(transforms[0].content_width, 320.0);
        assert_eq!(transforms[0].content_height, 736.0);
        assert_eq!(transforms[1].scale, 1.0);
        assert_eq!(transforms[1].pad_x, 0.0);
        assert_eq!(transforms[1].pad_y, 0.0);
        assert_eq!(buffer.len(), 2 * 3 * MODEL_SIDE * MODEL_SIDE);
    }

    #[test]
    fn detector_ignores_letterbox_padding_activations() {
        let mut map = vec![0.0; 32 * 32];
        // For a 320x736 source, the left 208 model pixels are padding. A
        // probability component there is not evidence from the source page.
        for y in 8..12 {
            for x in 1..6 {
                map[y * 32 + x] = 0.95;
            }
        }
        let detections = db_components(
            &map,
            32,
            32,
            (320, 736),
            DetectorTransform {
                scale: 1.0,
                pad_x: 208.0,
                pad_y: 0.0,
                content_width: 320.0,
                content_height: 736.0,
            },
        );
        assert!(detections.is_empty());
    }
}
