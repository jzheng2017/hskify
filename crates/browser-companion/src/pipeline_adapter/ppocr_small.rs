use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use image::imageops::{FilterType, contrast};
use image::{DynamicImage, RgbImage};
use ort::ep::{CUDA, ExecutionProvider};
use ort::memory::Allocator;
use ort::session::{
    HasSelectedOutputs, OutputSelector, RunOptions, Session, builder::GraphOptimizationLevel,
};
use ort::value::{Tensor, TensorRef};

use koharu_ml::manga_text_segmentation_2025::DEFAULT_TEXT_MASK_THRESHOLD;
use koharu_ml::probability_map::ProbabilityMap;

pub(super) const MAX_LINE_BATCH_SIZE: usize = 8;

const MODEL_HEIGHT: usize = 48;
const MODEL_BASE_WIDTH: usize = 320;
const MODEL_MAX_WIDTH: usize = 3_200;
const MODEL_WIDTH_BUCKETS: &[usize] = &[
    320, 640, 960, 1_280, 1_600, 1_920, 2_240, 2_560, 2_880, 3_200,
];
const EXPECTED_INPUT_NAME: &str = "x";
const EXPECTED_OUTPUT_NAME: &str = "fetch_name_0";
const SUPPORTED_MODEL_NAMES: &[&str] = &["PP-OCRv6_small_rec"];
const OUTPUT_CACHE_MAX_ENTRIES: usize = 4;
const OUTPUT_CACHE_MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(super) struct PpOcrPrediction {
    pub(super) text: String,
    pub(super) confidence: f32,
    pub(super) text_color: [u8; 3],
    pub(super) stroke_color: [u8; 3],
    pub(super) has_stroke_color: bool,
    pub(super) appearance_bands: Vec<PpOcrAppearanceBand>,
    pub(super) ocr_lines: Vec<PpOcrLine>,
}

#[derive(Debug, Clone)]
pub(super) struct PpOcrAppearanceBand {
    pub(super) top_ratio: f32,
    pub(super) bottom_ratio: f32,
    pub(super) text_color: [u8; 3],
    pub(super) stroke_color: [u8; 3],
    pub(super) has_stroke_color: bool,
}

/// One decoded OCR line together with the crop coordinates used to obtain it.
/// Keeping this provenance lets the page pipeline discard a decorative line
/// that was accidentally included in a detector crop without guessing from
/// the final concatenated string.
#[derive(Debug, Clone)]
pub(super) struct PpOcrLine {
    pub(super) text: String,
    pub(super) confidence: f32,
    pub(super) bounds: CropBounds,
}

pub(super) struct PpOcrSmallRecognizer {
    session: Session,
    characters: Vec<String>,
    input_buffer: Vec<f32>,
    output_cache: BoundedShapeCache<RunOptions<HasSelectedOutputs>>,
}

struct ShapeCacheEntry<T> {
    batch: usize,
    width: usize,
    bytes: usize,
    value: T,
}

struct BoundedShapeCache<T> {
    entries: VecDeque<ShapeCacheEntry<T>>,
    retained_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl<T> BoundedShapeCache<T> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            retained_bytes: 0,
            max_entries,
            max_bytes,
        }
    }

    fn take(&mut self, batch: usize, width: usize) -> Option<ShapeCacheEntry<T>> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.batch == batch && entry.width == width)?;
        let entry = self
            .entries
            .remove(index)
            .expect("the output cache entry was just located");
        self.retained_bytes = self.retained_bytes.saturating_sub(entry.bytes);
        Some(entry)
    }

    fn insert(&mut self, entry: ShapeCacheEntry<T>) {
        if self.max_entries == 0 || entry.bytes > self.max_bytes {
            return;
        }
        if let Some(previous) = self.take(entry.batch, entry.width) {
            drop(previous);
        }
        while self.entries.len() >= self.max_entries
            || self.retained_bytes.saturating_add(entry.bytes) > self.max_bytes
        {
            let Some(evicted) = self.entries.pop_back() else {
                break;
            };
            self.retained_bytes = self.retained_bytes.saturating_sub(evicted.bytes);
        }
        self.retained_bytes += entry.bytes;
        self.entries.push_front(entry);
    }
}

#[derive(Debug, Clone)]
struct DecodeResult {
    text: String,
    confidence: f32,
}

struct LineSample {
    region_index: usize,
    image: DynamicImage,
    bounds: CropBounds,
}

#[derive(Debug, PartialEq, Eq)]
struct LineBatchPlan {
    width: usize,
    indices: Vec<usize>,
}

fn raw_line_model_width(image_width: u32, image_height: u32) -> usize {
    ((MODEL_HEIGHT as f64 * image_width as f64 / image_height.max(1) as f64) as usize)
        .clamp(MODEL_BASE_WIDTH, MODEL_MAX_WIDTH)
}

fn line_model_width_bucket(image_width: u32, image_height: u32) -> usize {
    let raw_width = raw_line_model_width(image_width, image_height);
    MODEL_WIDTH_BUCKETS
        .iter()
        .copied()
        .find(|bucket| raw_width <= *bucket)
        .unwrap_or(MODEL_MAX_WIDTH)
}

fn width_bucket_line_batches(lines: &[LineSample]) -> Vec<LineBatchPlan> {
    let mut buckets = Vec::<(usize, Vec<usize>)>::new();
    for (index, line) in lines.iter().enumerate() {
        let width = line_model_width_bucket(line.image.width(), line.image.height());
        if let Some((_, indices)) = buckets.iter_mut().find(|(bucket, _)| *bucket == width) {
            indices.push(index);
        } else {
            buckets.push((width, vec![index]));
        }
    }
    let mut plans = Vec::new();
    for (width, indices) in buckets {
        for chunk in indices.chunks(MAX_LINE_BATCH_SIZE) {
            plans.push(LineBatchPlan {
                width,
                indices: chunk.to_vec(),
            });
        }
    }
    plans
}

impl PpOcrSmallRecognizer {
    pub(super) fn load(model_path: &Path, config_path: &Path) -> Result<Self> {
        let characters = load_characters(config_path)?;
        let cuda = CUDA::default().with_device_id(0);
        if !cuda
            .is_available()
            .context("query ONNX Runtime CUDA execution-provider availability")?
        {
            bail!("ONNX Runtime was not compiled with its mandatory CUDA execution provider");
        }
        let session = Session::builder()
            .context("create PP-OCR ONNX Runtime session builder")?
            .with_no_environment_execution_providers()
            .map_err(|error| {
                anyhow::anyhow!(
                    "ignore environment-provided ONNX Runtime execution providers: {error}"
                )
            })?
            .with_execution_providers([cuda.build().error_on_failure()])
            .map_err(|error| {
                anyhow::anyhow!(
                    "register the required ONNX Runtime CUDA execution provider: {error}"
                )
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!("enable PP-OCR ONNX graph optimizations: {error}"))?
            .with_memory_pattern(false)
            .map_err(|error| {
                anyhow::anyhow!("disable static memory patterns for dynamic PP-OCR shapes: {error}")
            })?
            .commit_from_file(model_path)
            .with_context(|| {
                format!(
                    "load pinned PP-OCRv6-small ONNX recognizer {} with mandatory CUDA acceleration",
                    model_path.display()
                )
            })?;
        validate_model_contract(&session)?;
        Ok(Self {
            session,
            characters,
            input_buffer: Vec::new(),
            output_cache: BoundedShapeCache::new(OUTPUT_CACHE_MAX_ENTRIES, OUTPUT_CACHE_MAX_BYTES),
        })
    }

    pub(super) fn recognize_regions(
        &mut self,
        block_crops: &[DynamicImage],
        text_probabilities: &[ProbabilityMap],
    ) -> Result<Vec<PpOcrPrediction>> {
        if block_crops.is_empty() {
            return Ok(Vec::new());
        }
        if block_crops.len() > MAX_LINE_BATCH_SIZE {
            bail!(
                "PP-OCR region batch has {} crops; maximum is {MAX_LINE_BATCH_SIZE}",
                block_crops.len()
            );
        }
        if text_probabilities.len() != block_crops.len() {
            bail!("PP-OCR requires one learned text mask per region crop");
        }

        // The PP-OCR detector supplies the line polygons.  Recognition must
        // consume those polygons directly; learned glyph mattes are only
        // appearance/cleanup evidence and must never invent line boundaries.
        let lines = block_crops
            .iter()
            .enumerate()
            .map(|(region_index, crop)| LineSample {
                region_index,
                image: crop.clone(),
                bounds: CropBounds {
                    left: 0,
                    top: 0,
                    right: crop.width(),
                    bottom: crop.height(),
                },
            })
            .collect::<Vec<_>>();

        let mut grouped = (0..block_crops.len())
            .map(|_| Vec::<(DecodeResult, CropBounds)>::new())
            .collect::<Vec<_>>();
        let mut decoded_primary = lines
            .iter()
            .map(|_| None::<DecodeResult>)
            .collect::<Vec<_>>();
        for batch in width_bucket_line_batches(&lines) {
            let line_batch = batch
                .indices
                .iter()
                .map(|&index| &lines[index])
                .collect::<Vec<_>>();
            let decoded = self.run_line_batch(&line_batch, batch.width)?;
            for (index, prediction) in batch.indices.into_iter().zip(decoded) {
                decoded_primary[index] = Some(prediction);
            }
        }
        for (index, line) in lines.iter().enumerate() {
            if let Some(prediction) = decoded_primary[index].take() {
                grouped[line.region_index].push((prediction, line.bounds));
            }
        }
        // Detector polygons are the only line proposals. Any uncertain line
        // is handled by the calibrated OCR consensus gate; it is never
        // fabricated by string-prefix or special crop rules.
        for region in &mut grouped {
            region.sort_by(|(_, left_bounds), (_, right_bounds)| {
                left_bounds
                    .top
                    .cmp(&right_bounds.top)
                    .then_with(|| left_bounds.left.cmp(&right_bounds.left))
            });
        }

        Ok(grouped
            .into_iter()
            .zip(block_crops.iter().zip(text_probabilities))
            .map(|(predictions, (crop, probabilities))| {
                if std::env::var_os("HSKIFY_TRACE_REJECTED_OCR").is_some_and(|value| value == "1") {
                    eprintln!(
                        "hskify-ocr-sublines {:?}",
                        predictions
                            .iter()
                            .map(|(prediction, bounds)| (
                                prediction.text.as_str(),
                                prediction.confidence,
                                bounds.left,
                                bounds.top,
                                bounds.right,
                                bounds.bottom,
                            ))
                            .collect::<Vec<_>>()
                    );
                }
                let ocr_lines = predictions
                    .iter()
                    .map(|(prediction, bounds)| PpOcrLine {
                        text: prediction.text.clone(),
                        confidence: prediction.confidence,
                        bounds: *bounds,
                    })
                    .collect::<Vec<_>>();
                // A region is only as reliable as its weakest recognized
                // line.  Use a character-weighted geometric mean instead of
                // an arithmetic mean so a long, low-confidence line cannot
                // be hidden by a short high-confidence line (and retain the
                // per-line confidence as actual evidence rather than dead
                // metadata).
                let confidence = if ocr_lines.is_empty() {
                    0.0
                } else {
                    let (log_sum, weight_sum) =
                        ocr_lines
                            .iter()
                            .fold((0.0_f32, 0_usize), |(log_sum, weight_sum), line| {
                                let weight = line.text.chars().count().max(1);
                                (
                                    log_sum
                                        + line.confidence.clamp(f32::EPSILON, 1.0).ln()
                                            * weight as f32,
                                    weight_sum + weight,
                                )
                            });
                    (log_sum / weight_sum.max(1) as f32).exp().clamp(0.0, 1.0)
                };
                let appearance_bands = predictions
                    .iter()
                    .map(|(_, bounds)| {
                        let (text_color, stroke_color) =
                            inferred_text_appearance_within(crop, probabilities, *bounds);
                        PpOcrAppearanceBand {
                            top_ratio: bounds.top as f32 / crop.height().max(1) as f32,
                            bottom_ratio: bounds.bottom as f32 / crop.height().max(1) as f32,
                            text_color,
                            stroke_color: stroke_color.unwrap_or([255, 255, 255]),
                            has_stroke_color: stroke_color.is_some(),
                        }
                    })
                    .collect::<Vec<_>>();
                let primary_appearance = predictions
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, (prediction, _))| prediction.text.chars().count())
                    .and_then(|(index, _)| appearance_bands.get(index))
                    .cloned()
                    .unwrap_or_else(|| {
                        let (text_color, stroke_color) =
                            inferred_text_appearance(crop, probabilities);
                        PpOcrAppearanceBand {
                            top_ratio: 0.0,
                            bottom_ratio: 1.0,
                            text_color,
                            stroke_color: stroke_color.unwrap_or([255, 255, 255]),
                            has_stroke_color: stroke_color.is_some(),
                        }
                    });
                PpOcrPrediction {
                    text: predictions
                        .iter()
                        .map(|(prediction, _)| prediction.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    confidence,
                    text_color: primary_appearance.text_color,
                    stroke_color: primary_appearance.stroke_color,
                    has_stroke_color: primary_appearance.has_stroke_color,
                    appearance_bands,
                    ocr_lines,
                }
            })
            .collect())
    }

    /// Resolve only uncertain detector regions with a genuinely different
    /// visual view.  The primary pass remains the hot path; the alternate
    /// contrast view is batched by the same recognizer and is admitted only
    /// for low-confidence or non-alphabetic output.  Consensus is decided by
    /// the shared OCR evidence gate, so a plausible single-view transcript
    /// can never authorize cleanup on its own.
    pub(super) fn recognize_regions_with_consensus(
        &mut self,
        block_crops: &[DynamicImage],
        text_probabilities: &[ProbabilityMap],
    ) -> Result<Vec<PpOcrPrediction>> {
        let mut primary = self.recognize_regions(block_crops, text_probabilities)?;
        if primary.len() != block_crops.len() {
            bail!("PP-OCR primary pass returned an incomplete region batch");
        }
        let uncertain = primary
            .iter()
            .enumerate()
            .filter_map(|(index, prediction)| {
                let text = prediction.text.trim();
                (prediction.confidence < super::ocr::BROWSER_OCR_MIN_CONFIDENCE
                    || text.is_empty()
                    || !text
                        .chars()
                        .any(|character| character.is_ascii_alphabetic()))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        if uncertain.is_empty() {
            return Ok(primary);
        }

        let alternate_crops = uncertain
            .iter()
            .map(|&index| alternate_view(&block_crops[index]))
            .collect::<Vec<_>>();
        let alternate_probabilities = uncertain
            .iter()
            .map(|&index| text_probabilities[index].clone())
            .collect::<Vec<_>>();
        let alternate = self.recognize_regions(&alternate_crops, &alternate_probabilities)?;
        if alternate.len() != uncertain.len() {
            bail!("PP-OCR alternate pass returned an incomplete region batch");
        }
        for (alternate_index, &region_index) in uncertain.iter().enumerate() {
            let primary_prediction = &primary[region_index];
            let alternate_prediction = &alternate[alternate_index];
            let crop = &block_crops[region_index];
            let bounds = CropBounds {
                left: 0,
                top: 0,
                right: crop.width(),
                bottom: crop.height(),
            };
            let Some((text, confidence, _)) = super::ocr::resolve_line_hypotheses(
                (
                    primary_prediction.text.as_str(),
                    primary_prediction.confidence,
                    bounds,
                ),
                (
                    alternate_prediction.text.as_str(),
                    alternate_prediction.confidence,
                    bounds,
                ),
                crop.width(),
                crop.height(),
            ) else {
                // Retain the source region but make the candidate ineligible
                // for translation/cleanup publication.  No string repair or
                // crop-specific probe is allowed to manufacture evidence.
                primary[region_index].text.clear();
                primary[region_index].confidence = 0.0;
                primary[region_index].ocr_lines.clear();
                primary[region_index].appearance_bands.clear();
                continue;
            };
            primary[region_index].text = text;
            primary[region_index].confidence = confidence;
        }
        Ok(primary)
    }

    fn run_line_batch(
        &mut self,
        lines: &[&LineSample],
        target_width: usize,
    ) -> Result<Vec<DecodeResult>> {
        let target_width = preprocess_line_batch(lines, target_width, &mut self.input_buffer)?;
        let batch = lines.len();
        let input = TensorRef::from_array_view((
            [batch, 3, MODEL_HEIGHT, target_width],
            self.input_buffer.as_slice(),
        ))
        .context("bind reusable zero-copy PP-OCR input buffer")?;

        if let Some(cache) = self.output_cache.take(batch, target_width) {
            let decoded = {
                let outputs = self
                    .session
                    .run_with_options(ort::inputs![input], &cache.value)
                    .context("run CUDA PP-OCR with the caller-owned output buffer")?;
                decode_session_output(&outputs[EXPECTED_OUTPUT_NAME], batch, &self.characters)?
            };
            self.output_cache.insert(cache);
            return Ok(decoded);
        }

        let (decoded, output_shape) = {
            let outputs = self
                .session
                .run(ort::inputs![input])
                .context("run CUDA PP-OCR while discovering its dynamic output shape")?;
            let output = &outputs[EXPECTED_OUTPUT_NAME];
            let (shape, _) = output
                .try_extract_tensor::<f32>()
                .context("extract PP-OCR float output")?;
            let output_shape = shape
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension)
                        .context("PP-OCR returned a negative output dimension")
                })
                .collect::<Result<Vec<_>>>()?;
            (
                decode_session_output(output, batch, &self.characters)?,
                output_shape,
            )
        };
        if let Some(cache) = make_output_cache(batch, target_width, output_shape)? {
            self.output_cache.insert(cache);
        }
        Ok(decoded)
    }
}
fn alternate_view(image: &DynamicImage) -> DynamicImage {
    let rgb = image.to_rgb8();
    let mut grayscale = RgbImage::new(rgb.width(), rgb.height());
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let [red, green, blue] = pixel.0;
        let luminance = (0.299 * red as f32 + 0.587 * green as f32 + 0.114 * blue as f32)
            .round()
            .clamp(0.0, 255.0) as u8;
        grayscale.put_pixel(x, y, image::Rgb([luminance, luminance, luminance]));
    }
    DynamicImage::ImageRgb8(contrast(&grayscale, 1.35))
}

fn make_output_cache(
    batch: usize,
    width: usize,
    output_shape: Vec<usize>,
) -> Result<Option<ShapeCacheEntry<RunOptions<HasSelectedOutputs>>>> {
    let elements = output_shape.iter().try_fold(1_usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .context("PP-OCRv6-small output allocation shape overflowed")
    })?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("PP-OCRv6-small output allocation byte count overflowed")?;
    if bytes > OUTPUT_CACHE_MAX_BYTES {
        return Ok(None);
    }
    let output = Tensor::<f32>::new(&Allocator::default(), output_shape)
        .context("allocate caller-owned PP-OCRv6-small output tensor")?;
    let value = RunOptions::new()
        .context("create PP-OCRv6-small output run options")?
        .with_outputs(
            OutputSelector::no_default()
                .with(EXPECTED_OUTPUT_NAME)
                .preallocate(EXPECTED_OUTPUT_NAME, output),
        );
    Ok(Some(ShapeCacheEntry {
        batch,
        width,
        bytes,
        value,
    }))
}

fn validate_model_contract(session: &Session) -> Result<()> {
    if session.inputs().len() != 1 || session.inputs()[0].name() != EXPECTED_INPUT_NAME {
        bail!("pinned PP-OCRv6-small input contract changed");
    }
    if session.outputs().len() != 1 || session.outputs()[0].name() != EXPECTED_OUTPUT_NAME {
        bail!("pinned PP-OCRv6-small output contract changed");
    }
    Ok(())
}

fn load_characters(config_path: &Path) -> Result<Vec<String>> {
    let config = fs::read_to_string(config_path).with_context(|| {
        format!(
            "read pinned PP-OCRv6-small config {}",
            config_path.display()
        )
    })?;
    if !config.lines().any(|line| {
        let model_name = line.trim().strip_prefix("model_name: ");
        model_name.is_some_and(|name| SUPPORTED_MODEL_NAMES.contains(&name))
    }) {
        bail!("pinned PP-OCR config has an unexpected model name");
    }
    if !config
        .lines()
        .any(|line| line.trim() == "name: CTCLabelDecode")
    {
        bail!("pinned PP-OCRv6-small config does not select CTCLabelDecode");
    }

    let mut in_dictionary = false;
    let mut dictionary = Vec::new();
    for line in config.lines() {
        let trimmed = line.trim();
        if trimmed == "character_dict:" {
            in_dictionary = true;
            continue;
        }
        if !in_dictionary {
            continue;
        }
        let Some(raw_scalar) = line.strip_prefix("  - ") else {
            if !trimmed.is_empty() && !line.starts_with("    ") {
                break;
            }
            continue;
        };
        let character = parse_pinned_yaml_scalar(raw_scalar)?;
        if character.is_empty() {
            bail!("PP-OCR character dictionary contains an empty scalar");
        }
        dictionary.push(character);
    }
    if dictionary.is_empty() {
        bail!("PP-OCR character dictionary is empty");
    }
    let mut characters = Vec::with_capacity(dictionary.len() + 2);
    characters.push("blank".to_owned());
    characters.extend(dictionary);
    characters.push(" ".to_owned());
    Ok(characters)
}

fn parse_pinned_yaml_scalar(raw: &str) -> Result<String> {
    let raw = raw.trim_end();
    if raw.starts_with('\'') {
        if raw.len() < 2 || !raw.ends_with('\'') {
            bail!("unterminated single-quoted PP-OCR character scalar");
        }
        return Ok(raw[1..raw.len() - 1].replace("''", "'"));
    }
    if raw.starts_with('"') {
        bail!("unexpected double-quoted PP-OCR character scalar");
    }
    if raw.is_empty() {
        bail!("empty PP-OCR character scalar");
    }
    Ok(raw.to_owned())
}

fn preprocess_line_batch(
    lines: &[&LineSample],
    target_width: usize,
    buffer: &mut Vec<f32>,
) -> Result<usize> {
    if lines.is_empty() || lines.len() > MAX_LINE_BATCH_SIZE {
        bail!("PP-OCR line batch must contain 1..={MAX_LINE_BATCH_SIZE} images");
    }
    if !MODEL_WIDTH_BUCKETS.contains(&target_width) {
        bail!("PP-OCR line batch width is not a model bucket");
    }
    if lines
        .iter()
        .any(|line| line_model_width_bucket(line.image.width(), line.image.height()) > target_width)
    {
        bail!("PP-OCR line batch contains a crop wider than its model bucket");
    }
    let element_count = lines
        .len()
        .checked_mul(3)
        .and_then(|value| value.checked_mul(MODEL_HEIGHT))
        .and_then(|value| value.checked_mul(target_width))
        .context("PP-OCR input shape overflowed")?;
    buffer.resize(element_count, 0.0);
    buffer.fill(0.0);

    let plane = MODEL_HEIGHT * target_width;
    for (batch_index, line) in lines.iter().enumerate() {
        let rgb = line.image.to_rgb8();
        let ratio = rgb.width() as f64 / rgb.height().max(1) as f64;
        let resized_width = ((MODEL_HEIGHT as f64 * ratio).ceil() as usize).clamp(1, target_width);
        let resized = image::imageops::resize(
            &rgb,
            u32::try_from(resized_width).context("PP-OCR resized width overflowed")?,
            MODEL_HEIGHT as u32,
            FilterType::Triangle,
        );
        let batch_offset = batch_index * 3 * plane;
        for (x, y, pixel) in resized.enumerate_pixels() {
            let spatial = y as usize * target_width + x as usize;
            let [red, green, blue] = pixel.0;
            buffer[batch_offset + spatial] = normalize_channel(blue);
            buffer[batch_offset + plane + spatial] = normalize_channel(green);
            buffer[batch_offset + 2 * plane + spatial] = normalize_channel(red);
        }
    }
    Ok(target_width)
}

fn normalize_channel(channel: u8) -> f32 {
    (channel as f32 / 255.0 - 0.5) / 0.5
}

fn decode_session_output(
    output: &ort::value::DynValue,
    expected_batch: usize,
    characters: &[String],
) -> Result<Vec<DecodeResult>> {
    let (shape, probabilities) = output
        .try_extract_tensor::<f32>()
        .context("extract PP-OCRv6-small CTC probabilities")?;
    if shape.len() != 3 || shape[0] != expected_batch as i64 || shape[2] != characters.len() as i64
    {
        bail!(
            "unexpected PP-OCRv6-small output shape; expected [{expected_batch}, time, {}]",
            characters.len()
        );
    }
    let time_steps =
        usize::try_from(shape[1]).context("PP-OCRv6-small returned a negative time dimension")?;
    let expected_values = expected_batch
        .checked_mul(time_steps)
        .and_then(|value| value.checked_mul(characters.len()))
        .context("PP-OCRv6-small output shape overflowed")?;
    if probabilities.len() != expected_values {
        bail!("PP-OCRv6-small output value count did not match its shape");
    }

    let mut decoded = Vec::with_capacity(expected_batch);
    for batch_index in 0..expected_batch {
        let mut text = String::new();
        let mut selected_probability_log_sum = 0.0_f32;
        let mut selected_count = 0_usize;
        let mut previous = usize::MAX;
        for time_index in 0..time_steps {
            let offset = (batch_index * time_steps + time_index) * characters.len();
            let row = &probabilities[offset..offset + characters.len()];
            let mut token = 0_usize;
            let mut score = row[0];
            for (candidate, candidate_score) in row.iter().copied().enumerate().skip(1) {
                if candidate_score > score {
                    token = candidate;
                    score = candidate_score;
                }
            }
            if !score.is_finite() {
                bail!("PP-OCRv6-small returned a non-finite CTC probability");
            }
            if token != 0 && token != previous {
                text.push_str(&characters[token]);
                // A sequence is only as reliable as its weakest selected
                // tokens. The geometric mean prevents a long, plausible
                // letter soup from outranking a shorter line merely because
                // its arithmetic average stayed high.
                selected_probability_log_sum += score.clamp(f32::EPSILON, 1.0).ln();
                selected_count += 1;
            }
            previous = token;
        }
        decoded.push(DecodeResult {
            text,
            confidence: if selected_count == 0 {
                0.0
            } else {
                (selected_probability_log_sum / selected_count as f32)
                    .exp()
                    .clamp(0.0, 1.0)
            },
        });
    }
    Ok(decoded)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CropBounds {
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) right: u32,
    pub(super) bottom: u32,
}

fn inferred_text_appearance(
    image: &DynamicImage,
    probabilities: &ProbabilityMap,
) -> ([u8; 3], Option<[u8; 3]>) {
    let bounds = CropBounds {
        left: 0,
        top: 0,
        right: image.width(),
        bottom: image.height(),
    };
    inferred_text_appearance_within(image, probabilities, bounds)
}

fn inferred_text_appearance_within(
    image: &DynamicImage,
    probabilities: &ProbabilityMap,
    bounds: CropBounds,
) -> ([u8; 3], Option<[u8; 3]>) {
    let rgb = image.to_rgb8();
    if rgb.width() == 0
        || rgb.height() == 0
        || rgb.width() != probabilities.width
        || rgb.height() != probabilities.height
    {
        return ([0, 0, 0], None);
    }
    let bounds = CropBounds {
        left: bounds.left.min(rgb.width()),
        top: bounds.top.min(rgb.height()),
        right: bounds.right.min(rgb.width()),
        bottom: bounds.bottom.min(rgb.height()),
    };
    let maximum = probability_maximum_within(probabilities, bounds);
    let core_threshold = (maximum * 0.65).max(DEFAULT_TEXT_MASK_THRESHOLD);
    let foreground =
        dominant_masked_color(&rgb, probabilities, bounds, |value| value >= core_threshold)
            .or_else(|| {
                dominant_masked_color(&rgb, probabilities, bounds, |value| {
                    value >= DEFAULT_TEXT_MASK_THRESHOLD
                })
            })
            .unwrap_or([0, 0, 0]);
    let stroke = dominant_masked_color(&rgb, probabilities, bounds, |value| {
        value >= DEFAULT_TEXT_MASK_THRESHOLD && value < core_threshold
    });
    (foreground, stroke)
}

fn probability_maximum_within(probabilities: &ProbabilityMap, bounds: CropBounds) -> f32 {
    (bounds.top..bounds.bottom)
        .flat_map(|y| {
            (bounds.left..bounds.right).filter_map(move |x| {
                probabilities
                    .values
                    .get(y as usize * probabilities.width as usize + x as usize)
                    .copied()
            })
        })
        .fold(0.0, f32::max)
}

#[derive(Clone, Default)]
struct ColorBucket {
    count: usize,
    sums: [u64; 3],
}

impl ColorBucket {
    fn add(&mut self, color: [u8; 3]) {
        self.count += 1;
        for (sum, channel) in self.sums.iter_mut().zip(color) {
            *sum += u64::from(channel);
        }
    }

    fn mean(&self) -> [u8; 3] {
        self.sums
            .map(|sum| (sum / self.count.max(1) as u64).min(255) as u8)
    }
}

fn dominant_masked_color(
    image: &RgbImage,
    probabilities: &ProbabilityMap,
    bounds: CropBounds,
    include: impl Fn(f32) -> bool,
) -> Option<[u8; 3]> {
    let mut buckets = vec![ColorBucket::default(); 512];
    for y in bounds.top..bounds.bottom {
        for x in bounds.left..bounds.right {
            let index = y as usize * probabilities.width as usize + x as usize;
            let Some(probability) = probabilities.values.get(index) else {
                continue;
            };
            if !include(*probability) {
                continue;
            }
            let color = image.get_pixel(x, y).0;
            let bucket = ((usize::from(color[0]) >> 5) << 6)
                | ((usize::from(color[1]) >> 5) << 3)
                | (usize::from(color[2]) >> 5);
            buckets[bucket].add(color);
        }
    }
    buckets
        .into_iter()
        .max_by_key(|bucket| bucket.count)
        .filter(|bucket| bucket.count > 0)
        .map(|bucket| bucket.mean())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn line_sample_for_test(region_index: usize, width: u32, height: u32) -> LineSample {
        LineSample {
            region_index,
            image: DynamicImage::ImageRgb8(RgbImage::new(width, height)),
            bounds: CropBounds {
                left: 0,
                top: 0,
                right: width,
                bottom: height,
            },
        }
    }

    #[test]
    fn line_width_buckets_round_up_and_keep_first_seen_sample_order() {
        assert_eq!(raw_line_model_width(319, 48), 320);
        assert_eq!(line_model_width_bucket(319, 48), 320);
        assert_eq!(line_model_width_bucket(320, 48), 320);
        assert_eq!(line_model_width_bucket(640, 48), 640);
        assert_eq!(line_model_width_bucket(641, 48), 960);
        assert_eq!(line_model_width_bucket(3_201, 48), 3_200);

        let lines = vec![
            line_sample_for_test(7, 641, 48),
            line_sample_for_test(8, 319, 48),
            line_sample_for_test(9, 641, 48),
            line_sample_for_test(10, 1_280, 48),
        ];
        let plans = width_bucket_line_batches(&lines);

        assert_eq!(
            plans,
            vec![
                LineBatchPlan {
                    width: 960,
                    indices: vec![0, 2],
                },
                LineBatchPlan {
                    width: 320,
                    indices: vec![1],
                },
                LineBatchPlan {
                    width: 1_280,
                    indices: vec![3],
                },
            ]
        );
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan
                    .indices
                    .iter()
                    .map(|&index| lines[index].region_index)
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![7, 9], vec![8], vec![10]]
        );
    }

    #[test]
    fn width_bucket_batches_never_exceed_the_model_batch_limit() {
        let lines = (0..17)
            .map(|region_index| line_sample_for_test(region_index, 640, 48))
            .collect::<Vec<_>>();
        let plans = width_bucket_line_batches(&lines);

        assert!(
            plans
                .iter()
                .all(|plan| plan.indices.len() <= MAX_LINE_BATCH_SIZE)
        );
        assert_eq!(
            plans
                .iter()
                .flat_map(|plan| plan.indices.iter().copied())
                .collect::<Vec<_>>(),
            (0..17).collect::<Vec<_>>()
        );
    }

    #[test]
    fn appearance_keeps_white_text_and_its_dark_outline_on_a_colored_bubble() {
        let mut image = ImageBuffer::from_pixel(80, 40, Rgb([130, 25, 30]));
        for y in 8..32 {
            for x in 15..65 {
                image.put_pixel(x, y, Rgb([8, 8, 10]));
            }
        }
        for y in 12..28 {
            for x in 20..60 {
                image.put_pixel(x, y, Rgb([248, 248, 245]));
            }
        }

        let mut probabilities = ProbabilityMap::zeros(image.width(), image.height());
        for y in 8..32 {
            for x in 15..65 {
                probabilities.values[y as usize * image.width() as usize + x as usize] = 0.55;
            }
        }
        for y in 12..28 {
            for x in 20..60 {
                probabilities.values[y as usize * image.width() as usize + x as usize] = 0.98;
            }
        }
        let (text, stroke) =
            inferred_text_appearance(&DynamicImage::ImageRgb8(image), &probabilities);

        assert!(text[0] > 230 && text[1] > 230 && text[2] > 230);
        let stroke = stroke.expect("contrasting adjacent outline should be retained");
        assert!(stroke[0] < 30 && stroke[1] < 30 && stroke[2] < 30);
    }

    #[test]
    fn appearance_is_sampled_per_learned_text_line_instead_of_per_block() {
        let mut image = ImageBuffer::from_pixel(80, 48, Rgb([255, 255, 255]));
        let mut probabilities = ProbabilityMap::zeros(80, 48);
        for y in 4..18 {
            for x in 12..68 {
                image.put_pixel(x, y, Rgb([8, 8, 10]));
                probabilities.values[y as usize * 80 + x as usize] = 0.98;
            }
        }
        for y in 30..44 {
            for x in 12..68 {
                image.put_pixel(x, y, Rgb([25, 110, 220]));
                probabilities.values[y as usize * 80 + x as usize] = 0.98;
            }
        }
        let image = DynamicImage::ImageRgb8(image);

        let (top, _) = inferred_text_appearance_within(
            &image,
            &probabilities,
            CropBounds {
                left: 0,
                top: 0,
                right: 80,
                bottom: 24,
            },
        );
        let (bottom, _) = inferred_text_appearance_within(
            &image,
            &probabilities,
            CropBounds {
                left: 0,
                top: 24,
                right: 80,
                bottom: 48,
            },
        );

        assert!(top[0] < 20 && top[1] < 20 && top[2] < 20);
        assert!(bottom[2] > 200 && bottom[1] > 90);
    }

    #[test]
    fn pinned_yaml_single_quote_escape_matches_yaml_semantics() {
        assert_eq!(parse_pinned_yaml_scalar("''''").unwrap(), "'");
        assert_eq!(parse_pinned_yaml_scalar("'#'").unwrap(), "#");
        assert_eq!(parse_pinned_yaml_scalar("\\").unwrap(), "\\");
    }

    #[test]
    fn character_loader_accepts_the_v6_multilingual_dictionary_shape() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("inference.yml");
        let mut config = String::from(
            "Global:\n  model_name: PP-OCRv6_small_rec\nPostProcess:\n  name: CTCLabelDecode\n  character_dict:\n",
        );
        for character in [
            "'!'", "'a'", "'b'", "'c'", "'d'", "'e'", "'f'", "'g'", "'h'", "'i'", "'j'", "'k'",
            "'l'", "'m'", "'n'", "'o'", "'p'", "'q'", "'r'", "'s'", "'t'", "'u'", "'v'", "'w'",
            "'x'", "'y'", "'z'", "'你好'", "'🛁'", "' '",
        ] {
            config.push_str("  - ");
            config.push_str(character);
            config.push('\n');
        }
        std::fs::write(&path, config).unwrap();
        let characters = load_characters(&path).unwrap();
        assert_eq!(characters.first().map(String::as_str), Some("blank"));
        assert_eq!(characters.last().map(String::as_str), Some(" "));
        assert!(characters.iter().any(|character| character == "你好"));
        assert_eq!(characters.len(), 32);
    }

    #[test]
    fn output_shape_cache_is_lru_and_byte_bounded() {
        let mut cache = BoundedShapeCache::new(2, 10);
        cache.insert(ShapeCacheEntry {
            batch: 8,
            width: 320,
            bytes: 4,
            value: "320",
        });
        cache.insert(ShapeCacheEntry {
            batch: 8,
            width: 640,
            bytes: 4,
            value: "640",
        });
        let recently_used = cache.take(8, 320).unwrap();
        cache.insert(recently_used);
        cache.insert(ShapeCacheEntry {
            batch: 8,
            width: 960,
            bytes: 4,
            value: "960",
        });
        assert!(
            cache.take(8, 640).is_none(),
            "least-recent shape is evicted"
        );
        assert_eq!(cache.take(8, 320).unwrap().value, "320");
        assert_eq!(cache.take(8, 960).unwrap().value, "960");

        cache.insert(ShapeCacheEntry {
            batch: 4,
            width: 1_280,
            bytes: 6,
            value: "first-six",
        });
        cache.insert(ShapeCacheEntry {
            batch: 4,
            width: 1_600,
            bytes: 6,
            value: "second-six",
        });
        assert!(
            cache.take(4, 1_280).is_none(),
            "byte pressure evicts the least-recent shape"
        );
        assert_eq!(cache.take(4, 1_600).unwrap().value, "second-six");

        cache.insert(ShapeCacheEntry {
            batch: 8,
            width: 3_200,
            bytes: 11,
            value: "oversized",
        });
        assert!(
            cache.take(8, 3_200).is_none(),
            "one entry cannot exceed the total byte budget"
        );
    }
}
