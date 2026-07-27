use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use ort::ep::{CUDA, ExecutionProvider};
use ort::memory::Allocator;
use ort::session::{
    HasSelectedOutputs, OutputSelector, RunOptions, Session, builder::GraphOptimizationLevel,
};
use ort::value::{Tensor, TensorRef};

pub(super) const MAX_LINE_BATCH_SIZE: usize = 8;

const MODEL_HEIGHT: usize = 48;
const MODEL_BASE_WIDTH: usize = 320;
const MODEL_MAX_WIDTH: usize = 3_200;
const EXPECTED_CLASSES: usize = 438;
const EXPECTED_INPUT_NAME: &str = "x";
const EXPECTED_OUTPUT_NAME: &str = "fetch_name_0";
const EXPECTED_MODEL_NAME: &str = "en_PP-OCRv5_mobile_rec";
const EXPECTED_CHARACTER_DICTIONARY_LEN: usize = EXPECTED_CLASSES - 2;
const OUTPUT_CACHE_MAX_ENTRIES: usize = 4;
const OUTPUT_CACHE_MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(super) struct PpOcrPrediction {
    pub(super) text: String,
    pub(super) confidence: f32,
    /// Foreground pixels belonging to recognized text lines in crop coordinates.
    pub(super) ink_mask: Option<PpOcrInkMask>,
    pub(super) text_color: [u8; 3],
    pub(super) stroke_color: [u8; 3],
    pub(super) has_stroke_color: bool,
}

#[derive(Debug, Clone)]
pub(super) struct PpOcrInkMask {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) values: Vec<bool>,
}

pub(super) struct EnglishPpOcrV5 {
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
}

impl EnglishPpOcrV5 {
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
            .context("create PP-OCRv5 ONNX Runtime session builder")?
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
            .map_err(|error| anyhow::anyhow!("enable PP-OCRv5 ONNX graph optimizations: {error}"))?
            .with_memory_pattern(false)
            .map_err(|error| {
                anyhow::anyhow!(
                    "disable static memory patterns for dynamic PP-OCRv5 shapes: {error}"
                )
            })?
            .commit_from_file(model_path)
            .with_context(|| {
                format!(
                    "load pinned English PP-OCRv5 ONNX model {} with mandatory CUDA acceleration",
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
    ) -> Result<Vec<PpOcrPrediction>> {
        if block_crops.is_empty() {
            return Ok(Vec::new());
        }
        if block_crops.len() > MAX_LINE_BATCH_SIZE {
            bail!(
                "PP-OCRv5 region batch has {} crops; maximum is {MAX_LINE_BATCH_SIZE}",
                block_crops.len()
            );
        }

        let mut lines = Vec::new();
        let mut colors = Vec::with_capacity(block_crops.len());
        let mut ink_masks = Vec::with_capacity(block_crops.len());
        for (region_index, crop) in block_crops.iter().enumerate() {
            colors.push(inferred_text_color(crop));
            let (region_lines, ink_mask) = segment_text_line_bounds(crop);
            for bounds in region_lines {
                lines.push(LineSample {
                    region_index,
                    image: crop.crop_imm(
                        bounds.left,
                        bounds.top,
                        bounds.right - bounds.left,
                        bounds.bottom - bounds.top,
                    ),
                });
            }
            ink_masks.push(ink_mask);
        }

        let mut grouped = (0..block_crops.len())
            .map(|_| Vec::<DecodeResult>::new())
            .collect::<Vec<_>>();
        for line_batch in lines.chunks(MAX_LINE_BATCH_SIZE) {
            let decoded = self.run_line_batch(line_batch)?;
            for (line, prediction) in line_batch.iter().zip(decoded) {
                grouped[line.region_index].push(prediction);
            }
        }

        Ok(grouped
            .into_iter()
            .zip(colors.into_iter().zip(ink_masks))
            .map(|(predictions, (text_color, ink_mask))| {
                let confidence = if predictions.is_empty() {
                    0.0
                } else {
                    predictions
                        .iter()
                        .map(|prediction| prediction.confidence)
                        .sum::<f32>()
                        / predictions.len() as f32
                };
                PpOcrPrediction {
                    text: predictions
                        .iter()
                        .map(|prediction| prediction.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    confidence,
                    ink_mask,
                    text_color,
                    stroke_color: [255, 255, 255],
                    has_stroke_color: false,
                }
            })
            .collect())
    }

    fn run_line_batch(&mut self, lines: &[LineSample]) -> Result<Vec<DecodeResult>> {
        let target_width = preprocess_line_batch(lines, &mut self.input_buffer)?;
        let batch = lines.len();
        let input = TensorRef::from_array_view((
            [batch, 3, MODEL_HEIGHT, target_width],
            self.input_buffer.as_slice(),
        ))
        .context("bind reusable zero-copy PP-OCRv5 input buffer")?;

        if let Some(cache) = self.output_cache.take(batch, target_width) {
            let decoded = {
                let outputs = self
                    .session
                    .run_with_options(ort::inputs![input], &cache.value)
                    .context("run CUDA PP-OCRv5 with the caller-owned output buffer")?;
                decode_session_output(&outputs[EXPECTED_OUTPUT_NAME], batch, &self.characters)?
            };
            self.output_cache.insert(cache);
            return Ok(decoded);
        }

        let (decoded, output_shape) = {
            let outputs = self
                .session
                .run(ort::inputs![input])
                .context("run CUDA PP-OCRv5 while discovering its dynamic output shape")?;
            let output = &outputs[EXPECTED_OUTPUT_NAME];
            let (shape, _) = output
                .try_extract_tensor::<f32>()
                .context("extract PP-OCRv5 float output")?;
            let output_shape = shape
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension)
                        .context("PP-OCRv5 returned a negative output dimension")
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

fn make_output_cache(
    batch: usize,
    width: usize,
    output_shape: Vec<usize>,
) -> Result<Option<ShapeCacheEntry<RunOptions<HasSelectedOutputs>>>> {
    let elements = output_shape.iter().try_fold(1_usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .context("PP-OCRv5 output allocation shape overflowed")
    })?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("PP-OCRv5 output allocation byte count overflowed")?;
    if bytes > OUTPUT_CACHE_MAX_BYTES {
        return Ok(None);
    }
    let output = Tensor::<f32>::new(&Allocator::default(), output_shape)
        .context("allocate caller-owned PP-OCRv5 output tensor")?;
    let value = RunOptions::new()
        .context("create PP-OCRv5 output run options")?
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
        bail!("pinned PP-OCRv5 input contract changed: expected only `{EXPECTED_INPUT_NAME}`");
    }
    if session.outputs().len() != 1 || session.outputs()[0].name() != EXPECTED_OUTPUT_NAME {
        bail!("pinned PP-OCRv5 output contract changed: expected only `{EXPECTED_OUTPUT_NAME}`");
    }
    Ok(())
}

fn load_characters(config_path: &Path) -> Result<Vec<String>> {
    let config = fs::read_to_string(config_path)
        .with_context(|| format!("read pinned PP-OCRv5 config {}", config_path.display()))?;
    if !config
        .lines()
        .any(|line| line.trim().strip_prefix("model_name: ") == Some(EXPECTED_MODEL_NAME))
    {
        bail!("pinned PP-OCR config has an unexpected model name");
    }
    if !config
        .lines()
        .any(|line| line.trim() == "name: CTCLabelDecode")
    {
        bail!("pinned PP-OCRv5 config does not select CTCLabelDecode");
    }

    let mut in_dictionary = false;
    let mut dictionary = Vec::new();
    for line in config.lines() {
        if line == "  character_dict:" {
            in_dictionary = true;
            continue;
        }
        if !in_dictionary {
            continue;
        }
        let Some(raw_scalar) = line.strip_prefix("  - ") else {
            break;
        };
        let character = parse_pinned_yaml_scalar(raw_scalar)?;
        if character.chars().count() != 1 {
            bail!("PP-OCRv5 character dictionary contains a non-character scalar");
        }
        dictionary.push(character);
    }
    if dictionary.len() != EXPECTED_CHARACTER_DICTIONARY_LEN {
        bail!(
            "PP-OCRv5 character dictionary has {} entries; expected {EXPECTED_CHARACTER_DICTIONARY_LEN}",
            dictionary.len()
        );
    }

    let mut characters = Vec::with_capacity(EXPECTED_CLASSES);
    characters.push("blank".to_owned());
    characters.extend(dictionary);
    // The pinned Paddle config keeps use_space_char outside PostProcess. The
    // ONNX output contains exactly one additional class for ASCII space.
    characters.push(" ".to_owned());
    Ok(characters)
}

fn parse_pinned_yaml_scalar(raw: &str) -> Result<String> {
    let raw = raw.trim_end();
    if raw.starts_with('\'') {
        if raw.len() < 2 || !raw.ends_with('\'') {
            bail!("unterminated single-quoted PP-OCRv5 character scalar");
        }
        return Ok(raw[1..raw.len() - 1].replace("''", "'"));
    }
    if raw.starts_with('"') {
        bail!("unexpected double-quoted PP-OCRv5 character scalar");
    }
    if raw.is_empty() {
        bail!("empty PP-OCRv5 character scalar");
    }
    Ok(raw.to_owned())
}

fn preprocess_line_batch(lines: &[LineSample], buffer: &mut Vec<f32>) -> Result<usize> {
    if lines.is_empty() || lines.len() > MAX_LINE_BATCH_SIZE {
        bail!("PP-OCRv5 line batch must contain 1..={MAX_LINE_BATCH_SIZE} images");
    }
    let max_ratio = lines.iter().fold(
        MODEL_BASE_WIDTH as f64 / MODEL_HEIGHT as f64,
        |current, line| {
            let (width, height) = line.image.dimensions();
            current.max(width as f64 / height.max(1) as f64)
        },
    );
    let target_width =
        ((MODEL_HEIGHT as f64 * max_ratio) as usize).clamp(MODEL_BASE_WIDTH, MODEL_MAX_WIDTH);
    let element_count = lines
        .len()
        .checked_mul(3)
        .and_then(|value| value.checked_mul(MODEL_HEIGHT))
        .and_then(|value| value.checked_mul(target_width))
        .context("PP-OCRv5 input shape overflowed")?;
    buffer.resize(element_count, 0.0);
    buffer.fill(0.0);

    let plane = MODEL_HEIGHT * target_width;
    for (batch_index, line) in lines.iter().enumerate() {
        let rgb = line.image.to_rgb8();
        let ratio = rgb.width() as f64 / rgb.height().max(1) as f64;
        let resized_width = ((MODEL_HEIGHT as f64 * ratio).ceil() as usize).clamp(1, target_width);
        let resized = image::imageops::resize(
            &rgb,
            u32::try_from(resized_width).context("PP-OCRv5 resized width overflowed")?,
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
        .context("extract PP-OCRv5 CTC probabilities")?;
    if shape.len() != 3 || shape[0] != expected_batch as i64 || shape[2] != characters.len() as i64
    {
        bail!(
            "unexpected PP-OCRv5 output shape {shape}; expected [{expected_batch}, time, {}]",
            characters.len()
        );
    }
    let time_steps =
        usize::try_from(shape[1]).context("PP-OCRv5 returned a negative time dimension")?;
    let expected_values = expected_batch
        .checked_mul(time_steps)
        .and_then(|value| value.checked_mul(characters.len()))
        .context("PP-OCRv5 output shape overflowed")?;
    if probabilities.len() != expected_values {
        bail!(
            "PP-OCRv5 output contains {} values; expected {expected_values}",
            probabilities.len()
        );
    }

    let mut decoded = Vec::with_capacity(expected_batch);
    for batch_index in 0..expected_batch {
        let mut text = String::new();
        let mut selected_probability_sum = 0.0_f32;
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
                bail!("PP-OCRv5 returned a non-finite CTC probability");
            }
            if token != 0 && token != previous {
                text.push_str(&characters[token]);
                selected_probability_sum += score;
                selected_count += 1;
            }
            previous = token;
        }
        decoded.push(DecodeResult {
            text,
            confidence: if selected_count == 0 {
                0.0
            } else {
                selected_probability_sum / selected_count as f32
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

#[derive(Debug, Clone)]
struct ForegroundMask {
    width: usize,
    height: usize,
    values: Vec<bool>,
}

impl ForegroundMask {
    fn get(&self, x: usize, y: usize) -> bool {
        self.values[y * self.width + x]
    }
}

#[derive(Debug, Clone)]
struct InkComponent {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
    area: usize,
}

impl InkComponent {
    fn height(&self) -> usize {
        self.bottom - self.top
    }

    fn center_y(&self) -> f64 {
        (self.top + self.bottom) as f64 / 2.0
    }

    fn touches_border(&self, width: usize, height: usize) -> bool {
        self.left == 0 || self.top == 0 || self.right == width || self.bottom == height
    }
}

struct ComponentGroup {
    core: Vec<usize>,
    components: Vec<usize>,
    last_core_center: f64,
}

fn segment_text_line_bounds(block_crop: &DynamicImage) -> (Vec<CropBounds>, Option<PpOcrInkMask>) {
    let rgb = block_crop.to_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    if width <= 1 || height <= 1 {
        return (vec![full_bounds(&rgb)], None);
    }
    let mask = foreground_mask(&rgb);
    let (components, groups) = component_line_groups(&mask);
    if groups.is_empty() {
        return (vec![full_bounds(&rgb)], None);
    }

    let pad_y = 2_usize.max((height as f64 * 0.018).round_ties_even() as usize);
    let pad_x = 2_usize.max((width as f64 * 0.012).round_ties_even() as usize);
    let mut row_ink = vec![0_usize; height];
    for (y, count) in row_ink.iter_mut().enumerate() {
        *count = (0..width).filter(|x| mask.get(*x, y)).count();
    }

    let mut boundaries = Vec::with_capacity(groups.len().saturating_sub(1));
    for pair in groups.windows(2) {
        let search_top = (pair[0].0.floor() as usize).min(height - 1);
        let search_bottom = (pair[1].0.ceil() as usize).min(height - 1);
        let boundary = if search_bottom <= search_top {
            height.min(search_top + 1)
        } else {
            let minimum = row_ink[search_top..=search_bottom]
                .iter()
                .copied()
                .min()
                .unwrap_or(0);
            let minima = (search_top..=search_bottom)
                .filter(|index| row_ink[*index] == minimum)
                .collect::<Vec<_>>();
            minima[minima.len() / 2].clamp(1, height - 1)
        };
        boundaries.push(boundary);
    }

    let mut bounds = Vec::with_capacity(groups.len());
    for (index, (_, component_indices)) in groups.iter().enumerate() {
        let component_left = component_indices
            .iter()
            .map(|component| components[*component].left)
            .min()
            .unwrap_or(0);
        let component_right = component_indices
            .iter()
            .map(|component| components[*component].right)
            .max()
            .unwrap_or(width);
        let component_top = component_indices
            .iter()
            .map(|component| components[*component].top)
            .min()
            .unwrap_or(0);
        let component_bottom = component_indices
            .iter()
            .map(|component| components[*component].bottom)
            .max()
            .unwrap_or(height);
        let mut left = component_left.saturating_sub(pad_x);
        let mut right = component_right.saturating_add(pad_x).min(width);
        let mut top = component_top.saturating_sub(pad_y);
        let mut bottom = component_bottom.saturating_add(pad_y).min(height);
        if index > 0 {
            top = top.max(boundaries[index - 1]).min(component_top);
        }
        if index < boundaries.len() {
            bottom = bottom.min(boundaries[index]).max(component_bottom);
        }
        left = left.min(width);
        right = right.min(width);
        if right > left && bottom > top {
            bounds.push(CropBounds {
                left: left as u32,
                top: top as u32,
                right: right as u32,
                bottom: bottom as u32,
            });
        }
    }
    if bounds.is_empty() {
        (vec![full_bounds(&rgb)], None)
    } else {
        let values = (0..height)
            .flat_map(|y| {
                let bounds = &bounds;
                let mask = &mask;
                (0..width).map(move |x| {
                    mask.get(x, y)
                        && bounds.iter().any(|line| {
                            x >= line.left as usize
                                && x < line.right as usize
                                && y >= line.top as usize
                                && y < line.bottom as usize
                        })
                })
            })
            .collect();
        (
            bounds,
            Some(PpOcrInkMask {
                width: width as u32,
                height: height as u32,
                values,
            }),
        )
    }
}

fn full_bounds(image: &RgbImage) -> CropBounds {
    CropBounds {
        left: 0,
        top: 0,
        right: image.width(),
        bottom: image.height(),
    }
}

fn foreground_mask(image: &RgbImage) -> ForegroundMask {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let grayscale = image
        .pixels()
        .map(|pixel| {
            let [red, green, blue] = pixel.0;
            ((19_595_u32 * red as u32
                + 38_470_u32 * green as u32
                + 7_471_u32 * blue as u32
                + 32_768)
                >> 16) as u8
        })
        .collect::<Vec<_>>();
    let threshold = otsu_threshold(&grayscale);
    let mut border = Vec::with_capacity(width * 2 + height * 2);
    border.extend_from_slice(&grayscale[..width]);
    border.extend_from_slice(&grayscale[(height - 1) * width..height * width]);
    border.extend((0..height).map(|y| grayscale[y * width]));
    border.extend((0..height).map(|y| grayscale[y * width + width - 1]));
    border.sort_unstable();
    let background = if border.len() % 2 == 0 {
        (border[border.len() / 2 - 1] as f64 + border[border.len() / 2] as f64) / 2.0
    } else {
        border[border.len() / 2] as f64
    };
    let background_integer = background as i16;
    let threshold_integer = threshold as i16;
    let mut values = grayscale
        .iter()
        .map(|value| {
            let value = *value as i16;
            if background >= threshold as f64 {
                value <= threshold_integer.min(background_integer - 6)
            } else {
                value >= threshold_integer.max(background_integer + 6)
            }
        })
        .collect::<Vec<_>>();
    let foreground_count = values.iter().filter(|value| **value).count() as u64;
    let total = values.len() as u64;
    if foreground_count * 100 > total * 45 {
        let dark_count = grayscale
            .iter()
            .filter(|value| **value <= threshold)
            .count();
        let light_count = grayscale.len() - dark_count;
        let dark = dark_count <= light_count;
        for (mask_value, grayscale_value) in values.iter_mut().zip(&grayscale) {
            *mask_value = if dark {
                *grayscale_value <= threshold
            } else {
                *grayscale_value > threshold
            };
        }
    }
    ForegroundMask {
        width,
        height,
        values,
    }
}

fn otsu_threshold(grayscale: &[u8]) -> u8 {
    let mut histogram = [0_u64; 256];
    for value in grayscale {
        histogram[*value as usize] += 1;
    }
    let total = grayscale.len() as f64;
    let weighted_sum = histogram
        .iter()
        .enumerate()
        .map(|(value, count)| value as f64 * *count as f64)
        .sum::<f64>();
    let mut background_weight = 0.0_f64;
    let mut background_sum = 0.0_f64;
    let mut best_variance = -1.0_f64;
    let mut best_threshold = 127_u8;
    for (threshold, count) in histogram.iter().copied().enumerate() {
        background_weight += count as f64;
        if background_weight <= 0.0 {
            continue;
        }
        let foreground_weight = total - background_weight;
        if foreground_weight <= 0.0 {
            break;
        }
        background_sum += threshold as f64 * count as f64;
        let background_mean = background_sum / background_weight;
        let foreground_mean = (weighted_sum - background_sum) / foreground_weight;
        let variance =
            background_weight * foreground_weight * (background_mean - foreground_mean).powi(2);
        if variance > best_variance {
            best_variance = variance;
            best_threshold = threshold as u8;
        }
    }
    best_threshold
}

fn connected_ink_components(mask: &ForegroundMask) -> Vec<InkComponent> {
    let mut union_find = UnionFind::default();
    let mut records = Vec::<(usize, usize, usize, usize)>::new();
    let mut previous_runs = Vec::<(usize, usize, usize)>::new();
    for y in 0..mask.height {
        let mut current_runs = Vec::new();
        let mut x = 0_usize;
        while x < mask.width {
            while x < mask.width && !mask.get(x, y) {
                x += 1;
            }
            if x == mask.width {
                break;
            }
            let left = x;
            while x < mask.width && mask.get(x, y) {
                x += 1;
            }
            let right = x;
            let label = union_find.add();
            current_runs.push((left, right, label));
            records.push((label, y, left, right));
            for (previous_left, previous_right, previous_label) in &previous_runs {
                if left <= *previous_right && *previous_left <= right {
                    union_find.union(label, *previous_label);
                }
            }
        }
        previous_runs = current_runs;
    }

    let mut aggregates = BTreeMap::<usize, [usize; 5]>::new();
    for (label, y, left, right) in records {
        let root = union_find.find(label);
        aggregates
            .entry(root)
            .and_modify(|aggregate| {
                aggregate[0] = aggregate[0].min(left);
                aggregate[1] = aggregate[1].min(y);
                aggregate[2] = aggregate[2].max(right);
                aggregate[3] = aggregate[3].max(y + 1);
                aggregate[4] += right - left;
            })
            .or_insert([left, y, right, y + 1, right - left]);
    }
    let mut components = aggregates
        .into_values()
        .map(|aggregate| InkComponent {
            left: aggregate[0],
            top: aggregate[1],
            right: aggregate[2],
            bottom: aggregate[3],
            area: aggregate[4],
        })
        .collect::<Vec<_>>();
    components.sort_by_key(|component| {
        (
            component.top,
            component.left,
            component.bottom,
            component.right,
        )
    });
    components
}

fn components_for_group(mask: &ForegroundMask) -> Vec<InkComponent> {
    connected_ink_components(mask)
        .into_iter()
        .filter(|component| !component.touches_border(mask.width, mask.height))
        .collect()
}

fn component_line_groups(mask: &ForegroundMask) -> (Vec<InkComponent>, Vec<(f64, Vec<usize>)>) {
    let components = components_for_group(mask);
    if components.is_empty() {
        return (components, Vec::new());
    }
    let typical_height = percentile_75(
        components
            .iter()
            .map(|component| component.height() as f64)
            .collect(),
    );
    let minimum_core_height = 2.0_f64.max(typical_height * 0.55);
    let mut core_indices = components
        .iter()
        .enumerate()
        .filter_map(|(index, component)| {
            (component.height() as f64 >= minimum_core_height).then_some(index)
        })
        .collect::<Vec<_>>();
    if core_indices.is_empty() {
        let mut best = 0_usize;
        for index in 1..components.len() {
            if (components[index].area, components[index].height())
                > (components[best].area, components[best].height())
            {
                best = index;
            }
        }
        core_indices.push(best);
    }
    core_indices.sort_by(|left, right| {
        components[*left]
            .center_y()
            .total_cmp(&components[*right].center_y())
            .then_with(|| components[*left].left.cmp(&components[*right].left))
            .then_with(|| components[*left].top.cmp(&components[*right].top))
    });

    let maximum_center_gap = 3.0_f64.max(typical_height * 0.55);
    let mut groups = Vec::<ComponentGroup>::new();
    for index in core_indices.iter().copied() {
        let center = components[index].center_y();
        if groups
            .last()
            .is_none_or(|group| center - group.last_core_center > maximum_center_gap)
        {
            groups.push(ComponentGroup {
                core: vec![index],
                components: vec![index],
                last_core_center: center,
            });
        } else {
            let group = groups.last_mut().expect("group was just inspected");
            group.core.push(index);
            group.components.push(index);
            group.last_core_center = center;
        }
    }

    let core_set = core_indices.into_iter().collect::<HashSet<_>>();
    let maximum_vertical_gap = 2.0_f64.max(typical_height * 0.50);
    let maximum_horizontal_gap = 2.0_f64.max(typical_height * 0.75);
    let mut pending = components
        .iter()
        .enumerate()
        .filter_map(|(index, _)| (!core_set.contains(&index)).then_some(index))
        .collect::<Vec<_>>();
    loop {
        let mut attached = false;
        let mut remaining = Vec::with_capacity(pending.len());
        for component_index in pending {
            let component = &components[component_index];
            let mut best: Option<(f64, f64, f64, usize)> = None;
            for (group_index, group) in groups.iter().enumerate() {
                let group_left = group
                    .components
                    .iter()
                    .map(|index| components[*index].left)
                    .min()
                    .expect("groups contain core components");
                let group_top = group
                    .components
                    .iter()
                    .map(|index| components[*index].top)
                    .min()
                    .expect("groups contain core components");
                let group_right = group
                    .components
                    .iter()
                    .map(|index| components[*index].right)
                    .max()
                    .expect("groups contain core components");
                let group_bottom = group
                    .components
                    .iter()
                    .map(|index| components[*index].bottom)
                    .max()
                    .expect("groups contain core components");
                let group_center = median(
                    group
                        .core
                        .iter()
                        .map(|index| components[*index].center_y())
                        .collect(),
                );
                let candidate = (
                    group_top
                        .saturating_sub(component.bottom)
                        .max(component.top.saturating_sub(group_bottom)) as f64,
                    (component.center_y() - group_center).abs(),
                    group_left
                        .saturating_sub(component.right)
                        .max(component.left.saturating_sub(group_right)) as f64,
                    group_index,
                );
                if best
                    .as_ref()
                    .is_none_or(|current| compare_candidate(&candidate, current).is_lt())
                {
                    best = Some(candidate);
                }
            }
            let Some((vertical_gap, _, horizontal_gap, target_index)) = best else {
                remaining.push(component_index);
                continue;
            };
            if vertical_gap <= maximum_vertical_gap && horizontal_gap <= maximum_horizontal_gap {
                groups[target_index].components.push(component_index);
                attached = true;
            } else {
                remaining.push(component_index);
            }
        }
        if !attached || remaining.is_empty() {
            break;
        }
        pending = remaining;
    }

    let groups = groups
        .into_iter()
        .map(|mut group| {
            let center = median(
                group
                    .core
                    .iter()
                    .map(|index| components[*index].center_y())
                    .collect(),
            );
            group.components.sort_by_key(|index| {
                (
                    components[*index].top,
                    components[*index].left,
                    components[*index].bottom,
                    components[*index].right,
                )
            });
            (center, group.components)
        })
        .collect();
    (components, groups)
}

fn compare_candidate(left: &(f64, f64, f64, usize), right: &(f64, f64, f64, usize)) -> Ordering {
    left.0
        .total_cmp(&right.0)
        .then_with(|| left.1.total_cmp(&right.1))
        .then_with(|| left.2.total_cmp(&right.2))
        .then_with(|| left.3.cmp(&right.3))
}

fn percentile_75(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let rank = (values.len() - 1) as f64 * 0.75;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        values[lower] + (values[upper] - values[lower]) * (rank - lower as f64)
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    }
}

#[derive(Default)]
struct UnionFind {
    parents: Vec<usize>,
    ranks: Vec<u8>,
}

impl UnionFind {
    fn add(&mut self) -> usize {
        let label = self.parents.len();
        self.parents.push(label);
        self.ranks.push(0);
        label
    }

    fn find(&mut self, label: usize) -> usize {
        let mut root = label;
        while self.parents[root] != root {
            root = self.parents[root];
        }
        let mut current = label;
        while self.parents[current] != current {
            let parent = self.parents[current];
            self.parents[current] = root;
            current = parent;
        }
        root
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.ranks[left_root] < self.ranks[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parents[right_root] = left_root;
        if self.ranks[left_root] == self.ranks[right_root] {
            self.ranks[left_root] += 1;
        }
    }
}

fn inferred_text_color(image: &DynamicImage) -> [u8; 3] {
    let rgb = image.to_rgb8();
    if rgb.width() == 0 || rgb.height() == 0 {
        return [0, 0, 0];
    }
    let mask = foreground_mask(&rgb);
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for (index, pixel) in rgb.pixels().enumerate() {
        if mask.values[index] {
            for channel in 0..3 {
                channels[channel].push(pixel.0[channel]);
            }
        }
    }
    if channels[0].is_empty() {
        return [0, 0, 0];
    }
    channels.map(|mut values| {
        values.sort_unstable();
        values[values.len() / 2]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn splitter_finds_two_lines_without_annotation_or_expected_count() {
        let mut image = ImageBuffer::from_pixel(120, 60, Rgb([255, 255, 255]));
        for y in 10..21 {
            for x in 18..102 {
                if (x / 8) % 2 == 0 {
                    image.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        for y in 36..47 {
            for x in 12..108 {
                if (x / 8) % 2 == 0 {
                    image.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        let (bounds, ink_mask) = segment_text_line_bounds(&DynamicImage::ImageRgb8(image));
        let ink_mask = ink_mask.expect("synthetic text has foreground");
        assert_eq!(bounds.len(), 2);
        assert!(bounds[0].bottom <= bounds[1].top);
        assert_eq!(ink_mask.width, 120);
        assert_eq!(ink_mask.height, 60);
        assert!(ink_mask.values[10 * 120 + 18]);
        assert!(ink_mask.values[46 * 120 + 103]);
    }

    #[test]
    fn splitter_keeps_punctuation_that_chains_outward_from_a_text_line() {
        let mut image = ImageBuffer::from_pixel(120, 40, Rgb([255, 255, 255]));
        for x in [18..22, 29..33, 40..44] {
            for y in 24..28 {
                for point_x in x.clone() {
                    image.put_pixel(point_x, y, Rgb([0, 0, 0]));
                }
            }
        }
        for letter_x in (58..104).step_by(9) {
            for y in 10..30 {
                for x in letter_x..letter_x + 5 {
                    image.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }

        let (_, ink_mask) = segment_text_line_bounds(&DynamicImage::ImageRgb8(image));
        let ink_mask = ink_mask.expect("synthetic line has foreground");
        assert!(ink_mask.values[25 * 120 + 19]);
        assert!(ink_mask.values[25 * 120 + 30]);
        assert!(ink_mask.values[25 * 120 + 41]);
        assert!(ink_mask.values[20 * 120 + 60]);
    }

    #[test]
    fn splitter_never_cuts_an_assigned_descender_at_a_line_boundary() {
        let mut image = ImageBuffer::from_pixel(120, 42, Rgb([255, 255, 255]));
        for letter_x in [18, 38] {
            for y in 8..18 {
                for x in letter_x..letter_x + 6 {
                    image.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        for y in 8..26 {
            for x in 58..64 {
                image.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        for letter_x in [20, 40, 75] {
            for y in 24..34 {
                for x in letter_x..letter_x + 6 {
                    image.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }

        let (bounds, ink_mask) = segment_text_line_bounds(&DynamicImage::ImageRgb8(image));
        let ink_mask = ink_mask.expect("synthetic lines have foreground");
        assert_eq!(bounds.len(), 2);
        assert!(bounds[0].bottom >= 26);
        assert!(ink_mask.values[24 * 120 + 60]);
    }

    #[test]
    fn pinned_yaml_single_quote_escape_matches_yaml_semantics() {
        assert_eq!(parse_pinned_yaml_scalar("''''").unwrap(), "'");
        assert_eq!(parse_pinned_yaml_scalar("'#'").unwrap(), "#");
        assert_eq!(parse_pinned_yaml_scalar("\\").unwrap(), "\\");
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
