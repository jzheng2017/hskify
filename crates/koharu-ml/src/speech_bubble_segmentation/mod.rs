mod model;

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops::sigmoid;
use candle_transformers::object_detection::{Bbox, non_maximum_suppression};
use image::{
    DynamicImage, Rgb, RgbImage,
    imageops::{self, FilterType},
};
use koharu_runtime::RuntimeManager;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    comic_text_bubble_detector::DETECTOR_TILE_BATCH_SIZE, device, loading,
    probability_map::ProbabilityMap,
};

use self::model::{Multiples, YoloV8Seg, YoloV8SegOutputs};

const HF_REPO: &str = "mayocream/speech-bubble-segmentation";
const CONFIG_FILENAME: &str = "config.json";
const SAFETENSORS_FILENAME: &str = "model.safetensors";
// Cleanup crops originate from the same six-tile reader frontier as the
// detector. Preserve that batch through contour inference instead of
// re-splitting one page section into repeated two-item forwards.
const GPU_BATCH_PIXEL_BUDGET: u64 = DETECTOR_TILE_BATCH_SIZE as u64 * 640 * 640;

koharu_runtime::declare_hf_model_package!(
    id: "model:speech-bubble-segmentation:config",
    repo: HF_REPO,
    file: CONFIG_FILENAME,
    bootstrap: false,
    order: 116,
);
koharu_runtime::declare_hf_model_package!(
    id: "model:speech-bubble-segmentation:weights",
    repo: HF_REPO,
    file: SAFETENSORS_FILENAME,
    bootstrap: false,
    order: 117,
);

#[derive(Debug)]
pub struct SpeechBubbleSegmentation {
    model: YoloV8Seg,
    config: SpeechBubbleSegmentationConfig,
    device: Device,
    dtype: DType,
}

#[derive(Debug, Clone)]
struct PreparedInput {
    pixel_values: Tensor,
    original_width: u32,
    original_height: u32,
    pad_x: u32,
    pad_y: u32,
    scale: f32,
}

#[derive(Debug, Clone)]
pub struct SpeechBubbleSegmentationResult {
    pub image_width: u32,
    pub image_height: u32,
    pub regions: Vec<SpeechBubbleRegion>,
    pub probability_map: ProbabilityMap,
}

#[derive(Debug, Clone)]
pub struct SpeechBubbleRegionMask {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl SpeechBubbleRegionMask {
    pub fn empty(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            width: 0,
            height: 0,
            pixels: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.pixels.is_empty()
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechBubbleRegion {
    pub label_id: usize,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
    pub area: u32,
    #[serde(skip_serializing)]
    pub mask: SpeechBubbleRegionMask,
}

#[derive(Debug, Clone)]
struct RawSpeechBubbleRegion {
    label_id: usize,
    label: String,
    score: f32,
    bbox: [f32; 4],
    mask_coefficients: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct SpeechBubbleSegmentationConfig {
    model_type: String,
    variant: String,
    input_size: u32,
    num_classes: usize,
    num_masks: usize,
    num_prototypes: usize,
    reg_max: usize,
    class_names: Vec<String>,
    default_confidence_threshold: f32,
    default_nms_threshold: f32,
    mask_threshold: f32,
    letterbox_color: u8,
}

impl SpeechBubbleSegmentationConfig {
    fn validate(&self) -> Result<()> {
        if self.model_type != "yolov8-seg" {
            bail!("unsupported speech bubble model type {}", self.model_type);
        }
        if self.variant != "m" {
            bail!("unsupported YOLOv8 segmentation variant {}", self.variant);
        }
        if self.input_size == 0 || !self.input_size.is_multiple_of(32) {
            bail!("invalid input_size {}", self.input_size);
        }
        if self.num_classes == 0 {
            bail!("num_classes must be positive");
        }
        if self.class_names.len() != self.num_classes {
            bail!(
                "expected {} class names, found {}",
                self.num_classes,
                self.class_names.len()
            );
        }
        if self.num_masks == 0 {
            bail!("num_masks must be positive");
        }
        if self.num_prototypes == 0 {
            bail!("num_prototypes must be positive");
        }
        if self.reg_max == 0 {
            bail!("reg_max must be positive");
        }
        Ok(())
    }
}

impl SpeechBubbleSegmentation {
    pub async fn load(runtime: &RuntimeManager, cpu: bool) -> Result<Self> {
        let (config_path, weights_path) = resolve_model_paths(runtime).await?;
        Self::load_from_paths(&config_path, &weights_path, cpu)
    }

    pub fn load_from_paths(
        config_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
        cpu: bool,
    ) -> Result<Self> {
        let device = device(cpu)?;
        let dtype = loading::model_dtype(&device);
        let config = loading::read_json::<SpeechBubbleSegmentationConfig>(config_path.as_ref())
            .with_context(|| format!("failed to parse {}", config_path.as_ref().display()))?;
        config.validate()?;
        let multiples = variant_multiples(&config)?;
        let model = loading::load_mmaped_safetensors_path_with_dtype(
            weights_path.as_ref(),
            &device,
            dtype,
            |vb| {
                YoloV8Seg::load(
                    vb,
                    multiples,
                    config.num_classes,
                    config.num_masks,
                    config.num_prototypes,
                    config.reg_max,
                )
            },
        )?;

        Ok(Self {
            model,
            config,
            device,
            dtype,
        })
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference(&self, image: &DynamicImage) -> Result<SpeechBubbleSegmentationResult> {
        self.inference_with_thresholds(
            image,
            self.config.default_confidence_threshold,
            self.config.default_nms_threshold,
        )
    }

    #[instrument(level = "debug", skip_all)]
    pub fn inference_with_thresholds(
        &self,
        image: &DynamicImage,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<SpeechBubbleSegmentationResult> {
        self.inference_batch_with_thresholds(
            std::slice::from_ref(image),
            confidence_threshold,
            nms_threshold,
        )?
        .pop()
        .context("speech bubble segmentation returned no result")
    }

    pub fn inference_batch(
        &self,
        images: &[DynamicImage],
    ) -> Result<Vec<SpeechBubbleSegmentationResult>> {
        self.inference_batch_with_thresholds(
            images,
            self.config.default_confidence_threshold,
            self.config.default_nms_threshold,
        )
    }

    pub fn inference_batch_with_thresholds(
        &self,
        images: &[DynamicImage],
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<SpeechBubbleSegmentationResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let maximum_batch = if self.device.is_cuda() {
            usize::try_from(
                GPU_BATCH_PIXEL_BUDGET / u64::from(self.config.input_size).pow(2).max(1),
            )
            .unwrap_or(1)
            .max(1)
        } else {
            1
        };
        let mut results = Vec::with_capacity(images.len());
        for chunk in images.chunks(maximum_batch) {
            let preprocess_started = Instant::now();
            let prepared = chunk
                .iter()
                .map(|image| self.preprocess(image))
                .collect::<Result<Vec<_>>>()?;
            let preprocess_elapsed = preprocess_started.elapsed();
            let tensors = prepared
                .iter()
                .map(|input| &input.pixel_values)
                .collect::<Vec<_>>();
            let forward_started = Instant::now();
            let outputs = self.model.forward(&Tensor::cat(&tensors, 0)?)?;
            let forward_elapsed = forward_started.elapsed();
            let postprocess_started = Instant::now();
            results.extend(postprocess_batch(
                &outputs,
                &prepared,
                &self.config,
                confidence_threshold,
                nms_threshold,
            )?);
            tracing::info!(
                batch = prepared.len(),
                preprocess_ms = preprocess_elapsed.as_millis(),
                forward_ms = forward_elapsed.as_millis(),
                postprocess_ms = postprocess_started.elapsed().as_millis(),
                "batched speech bubble segmentation timings"
            );
        }
        tracing::info!(
            images = images.len(),
            total_ms = started.elapsed().as_millis(),
            "speech bubble segmentation batch complete"
        );
        Ok(results)
    }

    fn preprocess(&self, image: &DynamicImage) -> Result<PreparedInput> {
        let rgb = image.to_rgb8();
        let (original_width, original_height) = rgb.dimensions();
        let input_size = self.config.input_size;
        let scale = f32::min(
            input_size as f32 / original_width.max(1) as f32,
            input_size as f32 / original_height.max(1) as f32,
        );
        let resized_width = ((original_width as f32 * scale).round() as u32).clamp(1, input_size);
        let resized_height = ((original_height as f32 * scale).round() as u32).clamp(1, input_size);
        let pad_x = (input_size - resized_width) / 2;
        let pad_y = (input_size - resized_height) / 2;

        let resized = if resized_width == original_width && resized_height == original_height {
            rgb
        } else {
            imageops::resize(&rgb, resized_width, resized_height, FilterType::Triangle)
        };

        let mut letterboxed = RgbImage::from_pixel(
            input_size,
            input_size,
            Rgb([self.config.letterbox_color; 3]),
        );
        imageops::overlay(
            &mut letterboxed,
            &resized,
            i64::from(pad_x),
            i64::from(pad_y),
        );

        let pixel_values = Tensor::from_vec(
            letterboxed.into_raw(),
            (1, input_size as usize, input_size as usize, 3),
            &self.device,
        )?
        .permute((0, 3, 1, 2))?
        .to_dtype(self.dtype)?;
        let pixel_values = (pixel_values * (1.0 / 255.0))?;

        Ok(PreparedInput {
            pixel_values,
            original_width,
            original_height,
            pad_x,
            pad_y,
            scale,
        })
    }
}

pub async fn prefetch(runtime: &RuntimeManager) -> Result<()> {
    let _ = resolve_model_paths(runtime).await?;
    Ok(())
}

async fn resolve_model_paths(runtime: &RuntimeManager) -> Result<(PathBuf, PathBuf)> {
    let downloads = runtime.downloads();
    let config = downloads
        .huggingface_model(HF_REPO, CONFIG_FILENAME)
        .await
        .with_context(|| format!("failed to download {CONFIG_FILENAME} from {HF_REPO}"))?;
    let weights = downloads
        .huggingface_model(HF_REPO, SAFETENSORS_FILENAME)
        .await
        .with_context(|| format!("failed to download {SAFETENSORS_FILENAME} from {HF_REPO}"))?;
    Ok((config, weights))
}

fn variant_multiples(config: &SpeechBubbleSegmentationConfig) -> Result<Multiples> {
    match config.variant.as_str() {
        "m" => Ok(Multiples::m()),
        other => bail!("unsupported YOLOv8 segmentation variant {other}"),
    }
}

fn postprocess_batch(
    outputs: &YoloV8SegOutputs,
    prepared: &[PreparedInput],
    config: &SpeechBubbleSegmentationConfig,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<SpeechBubbleSegmentationResult>> {
    let pred = outputs.pred.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let proto = outputs
        .proto
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    if pred.dim(0)? != prepared.len() || proto.dim(0)? != prepared.len() {
        bail!("speech bubble segmentation returned an incomplete batch");
    }
    prepared
        .iter()
        .enumerate()
        .map(|(batch_index, input)| {
            postprocess_image(
                &pred.i(batch_index)?,
                &proto.i(batch_index)?,
                input,
                config,
                confidence_threshold,
                nms_threshold,
            )
        })
        .collect()
}

fn postprocess_image(
    pred: &Tensor,
    proto: &Tensor,
    prepared: &PreparedInput,
    config: &SpeechBubbleSegmentationConfig,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<SpeechBubbleSegmentationResult> {
    let raw_regions = extract_regions(pred, prepared, config, confidence_threshold, nms_threshold)?;
    let mut probability_map =
        ProbabilityMap::zeros(prepared.original_width, prepared.original_height);
    let mask_probabilities = build_mask_probabilities(proto, prepared, config, &raw_regions)?;

    let mut regions = Vec::with_capacity(raw_regions.len());
    for (region, mask) in raw_regions.iter().zip(mask_probabilities.iter()) {
        let (area, region_mask) = extract_region_contour_mask(
            &mut probability_map,
            mask,
            region.bbox,
            config.mask_threshold,
        )?;
        if area == 0 {
            continue;
        }
        regions.push(SpeechBubbleRegion {
            label_id: region.label_id,
            label: region.label.clone(),
            score: region.score,
            bbox: region.bbox,
            area,
            mask: region_mask,
        });
    }

    Ok(SpeechBubbleSegmentationResult {
        image_width: prepared.original_width,
        image_height: prepared.original_height,
        regions,
        probability_map,
    })
}

fn extract_regions(
    pred: &Tensor,
    prepared: &PreparedInput,
    config: &SpeechBubbleSegmentationConfig,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<RawSpeechBubbleRegion>> {
    let (channels, anchors) = pred.dims2()?;
    let expected_channels = 4 + config.num_classes + config.num_masks;
    if channels != expected_channels {
        bail!(
            "unexpected prediction shape ({channels}, {anchors}), expected channel count {expected_channels}"
        );
    }

    let mut grouped: Vec<Vec<Bbox<Vec<f32>>>> = vec![Vec::new(); config.num_classes];
    for anchor_idx in 0..anchors {
        let values = pred.i((.., anchor_idx))?.to_vec1::<f32>()?;
        let class_scores = &values[4..4 + config.num_classes];
        let Some((label_id, &score)) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
        else {
            continue;
        };
        if score < confidence_threshold {
            continue;
        }

        let bbox = map_bbox_to_original(
            [
                values[0] - values[2] * 0.5,
                values[1] - values[3] * 0.5,
                values[0] + values[2] * 0.5,
                values[1] + values[3] * 0.5,
            ],
            prepared,
        );
        if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            continue;
        }

        grouped[label_id].push(Bbox {
            xmin: bbox[0],
            ymin: bbox[1],
            xmax: bbox[2],
            ymax: bbox[3],
            confidence: score,
            data: values[4 + config.num_classes..].to_vec(),
        });
    }

    non_maximum_suppression(&mut grouped, nms_threshold);

    let mut regions = Vec::new();
    for (label_id, bboxes) in grouped.into_iter().enumerate() {
        let label = config
            .class_names
            .get(label_id)
            .cloned()
            .unwrap_or_else(|| format!("class-{label_id}"));
        for bbox in bboxes {
            regions.push(RawSpeechBubbleRegion {
                label_id,
                label: label.clone(),
                score: bbox.confidence,
                bbox: [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax],
                mask_coefficients: bbox.data,
            });
        }
    }
    regions.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(regions)
}

fn map_bbox_to_original(bbox: [f32; 4], prepared: &PreparedInput) -> [f32; 4] {
    let width = prepared.original_width as f32;
    let height = prepared.original_height as f32;
    let pad_x = prepared.pad_x as f32;
    let pad_y = prepared.pad_y as f32;
    [
        ((bbox[0] - pad_x) / prepared.scale).clamp(0.0, width),
        ((bbox[1] - pad_y) / prepared.scale).clamp(0.0, height),
        ((bbox[2] - pad_x) / prepared.scale).clamp(0.0, width),
        ((bbox[3] - pad_y) / prepared.scale).clamp(0.0, height),
    ]
}

fn build_mask_probabilities(
    proto: &Tensor,
    prepared: &PreparedInput,
    config: &SpeechBubbleSegmentationConfig,
    regions: &[RawSpeechBubbleRegion],
) -> Result<Vec<Vec<f32>>> {
    if regions.is_empty() {
        return Ok(Vec::new());
    }

    let (num_masks, proto_h, proto_w) = proto.dims3()?;
    if num_masks != config.num_masks {
        bail!(
            "unexpected proto channel count {num_masks}, expected {}",
            config.num_masks
        );
    }

    let coefficients = regions
        .iter()
        .flat_map(|region| region.mask_coefficients.iter().copied())
        .collect::<Vec<_>>();
    let coeffs = Tensor::from_vec(
        coefficients,
        (regions.len(), config.num_masks),
        &Device::Cpu,
    )?;
    let proto_flat = proto.reshape((config.num_masks, proto_h * proto_w))?;
    let mut masks = coeffs
        .matmul(&proto_flat)?
        .reshape((regions.len(), 1, proto_h, proto_w))?;

    let (top, left, bottom, right) = mask_crop_window(
        prepared.original_width,
        prepared.original_height,
        proto_w as u32,
        proto_h as u32,
    );
    masks = masks.i((.., .., top..bottom, left..right))?;
    masks = masks.interpolate2d(
        prepared.original_height as usize,
        prepared.original_width as usize,
    )?;
    let masks = sigmoid(&masks.squeeze(1)?)?;

    let mut outputs = Vec::with_capacity(regions.len());
    for index in 0..regions.len() {
        outputs.push(masks.i(index)?.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok(outputs)
}

fn mask_crop_window(
    original_width: u32,
    original_height: u32,
    proto_width: u32,
    proto_height: u32,
) -> (usize, usize, usize, usize) {
    let gain = f32::min(
        proto_height as f32 / original_height.max(1) as f32,
        proto_width as f32 / original_width.max(1) as f32,
    );
    let pad_w = (proto_width as f32 - original_width as f32 * gain) / 2.0;
    let pad_h = (proto_height as f32 - original_height as f32 * gain) / 2.0;
    let top = ((pad_h - 0.1).round()).clamp(0.0, proto_height as f32) as usize;
    let left = ((pad_w - 0.1).round()).clamp(0.0, proto_width as f32) as usize;
    let bottom =
        proto_height as usize - ((pad_h + 0.1).round()).clamp(0.0, proto_height as f32) as usize;
    let right =
        proto_width as usize - ((pad_w + 0.1).round()).clamp(0.0, proto_width as f32) as usize;
    let bottom = bottom.max(top + 1).min(proto_height as usize);
    let right = right.max(left + 1).min(proto_width as usize);
    (top, left, bottom, right)
}

fn extract_region_contour_mask(
    probability_map: &mut ProbabilityMap,
    mask: &[f32],
    bbox: [f32; 4],
    threshold: f32,
) -> Result<(u32, SpeechBubbleRegionMask)> {
    let width = probability_map.width as usize;
    let height = probability_map.height as usize;
    if mask.len() != width * height {
        bail!(
            "speech bubble mask length {} does not match image area {}",
            mask.len(),
            width * height
        );
    }

    let x1 = bbox[0].floor().clamp(0.0, probability_map.width as f32) as usize;
    let y1 = bbox[1].floor().clamp(0.0, probability_map.height as f32) as usize;
    let x2 = bbox[2].ceil().clamp(0.0, probability_map.width as f32) as usize;
    let y2 = bbox[3].ceil().clamp(0.0, probability_map.height as f32) as usize;
    if x2 <= x1 || y2 <= y1 {
        return Ok((0, SpeechBubbleRegionMask::empty(x1 as u32, y1 as u32)));
    }

    let mask_width = x2 - x1;
    let mask_height = y2 - y1;
    let mut pixels = vec![0u8; mask_width * mask_height];
    let mut area = 0u32;
    for y in y1..y2.min(height) {
        let row_offset = y * width;
        let local_row_offset = (y - y1) * mask_width;
        for x in x1..x2.min(width) {
            let idx = row_offset + x;
            let value = mask[idx];
            if value >= threshold {
                area += 1;
                pixels[local_row_offset + (x - x1)] = u8::MAX;
            }
            if value > probability_map.values[idx] {
                probability_map.values[idx] = value;
            }
        }
    }
    Ok((
        area,
        SpeechBubbleRegionMask {
            x: x1 as u32,
            y: y1 as u32,
            width: mask_width as u32,
            height: mask_height as u32,
            pixels,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        PreparedInput, extract_region_contour_mask, map_bbox_to_original, mask_crop_window,
    };
    use crate::probability_map::ProbabilityMap;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn map_bbox_to_original_removes_letterbox_padding() {
        let prepared = PreparedInput {
            pixel_values: Tensor::zeros((1, 3, 640, 640), DType::F32, &Device::Cpu)
                .expect("tensor"),
            original_width: 1000,
            original_height: 500,
            pad_x: 0,
            pad_y: 160,
            scale: 0.64,
        };

        let bbox = map_bbox_to_original([100.0, 200.0, 540.0, 440.0], &prepared);
        assert!((bbox[0] - 156.25).abs() < 1e-3);
        assert!((bbox[1] - 62.5).abs() < 1e-3);
        assert!((bbox[2] - 843.75).abs() < 1e-3);
        assert!((bbox[3] - 437.5).abs() < 1e-3);
    }

    #[test]
    fn mask_crop_window_matches_letterboxed_square_input() {
        let (top, left, bottom, right) = mask_crop_window(1000, 500, 160, 160);
        assert_eq!((top, left, bottom, right), (40, 0, 120, 160));
    }

    #[test]
    fn extract_region_contour_mask_keeps_thresholded_shape() -> anyhow::Result<()> {
        let mut probability_map = ProbabilityMap::zeros(6, 5);
        let mut mask = vec![0.0f32; 6 * 5];
        mask[1 + 1 * 6] = 0.9;
        mask[2 + 1 * 6] = 0.8;
        mask[2 + 2 * 6] = 0.7;
        mask[4 + 3 * 6] = 0.4;

        let (area, region_mask) =
            extract_region_contour_mask(&mut probability_map, &mask, [1.0, 1.0, 5.0, 4.0], 0.5)?;

        assert_eq!(area, 3);
        assert_eq!((region_mask.x, region_mask.y), (1, 1));
        assert_eq!((region_mask.width, region_mask.height), (4, 3));
        assert_eq!(region_mask.pixels[0], u8::MAX);
        assert_eq!(region_mask.pixels[1], u8::MAX);
        assert_eq!(region_mask.pixels[5], u8::MAX);
        assert_eq!(region_mask.pixels[11], 0);
        assert_eq!(probability_map.values[4 + 3 * 6], 0.4);
        Ok(())
    }
}
