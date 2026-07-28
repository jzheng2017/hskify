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

use koharu_ml::manga_text_segmentation_2025::DEFAULT_TEXT_MASK_THRESHOLD;
use koharu_ml::probability_map::ProbabilityMap;

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
    pub(super) text_color: [u8; 3],
    pub(super) stroke_color: [u8; 3],
    pub(super) has_stroke_color: bool,
    pub(super) appearance_bands: Vec<PpOcrAppearanceBand>,
}

#[derive(Debug, Clone)]
pub(super) struct PpOcrAppearanceBand {
    pub(super) top_ratio: f32,
    pub(super) bottom_ratio: f32,
    pub(super) text_color: [u8; 3],
    pub(super) stroke_color: [u8; 3],
    pub(super) has_stroke_color: bool,
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
    bounds: CropBounds,
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
        text_probabilities: &[ProbabilityMap],
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
        if text_probabilities.len() != block_crops.len() {
            bail!("PP-OCRv5 requires one learned text mask per region crop");
        }

        let mut lines = Vec::new();
        for (region_index, (crop, probabilities)) in
            block_crops.iter().zip(text_probabilities).enumerate()
        {
            let region_lines = segment_text_line_bounds(crop, probabilities);
            for bounds in region_lines {
                lines.push(LineSample {
                    region_index,
                    image: crop.crop_imm(
                        bounds.left,
                        bounds.top,
                        bounds.right - bounds.left,
                        bounds.bottom - bounds.top,
                    ),
                    bounds,
                });
            }
        }

        let mut grouped = (0..block_crops.len())
            .map(|_| Vec::<(DecodeResult, CropBounds)>::new())
            .collect::<Vec<_>>();
        for line_batch in lines.chunks(MAX_LINE_BATCH_SIZE) {
            let decoded = self.run_line_batch(line_batch)?;
            for (line, prediction) in line_batch.iter().zip(decoded) {
                grouped[line.region_index].push((prediction, line.bounds));
            }
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
                let confidence = if predictions.is_empty() {
                    0.0
                } else {
                    predictions
                        .iter()
                        .map(|(prediction, _)| prediction.confidence)
                        .sum::<f32>()
                        / predictions.len() as f32
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

fn segment_text_line_bounds(
    block_crop: &DynamicImage,
    probabilities: &ProbabilityMap,
) -> Vec<CropBounds> {
    let rgb = block_crop.to_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    if width <= 1 || height <= 1 {
        return vec![full_bounds(&rgb)];
    }
    let mask = foreground_mask(probabilities);
    let (components, groups) = component_line_groups(&mask);
    if groups.is_empty() {
        return vec![full_bounds(&rgb)];
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
        vec![full_bounds(&rgb)]
    } else {
        bounds
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

fn foreground_mask(probabilities: &ProbabilityMap) -> ForegroundMask {
    ForegroundMask {
        width: probabilities.width as usize,
        height: probabilities.height as usize,
        values: probabilities
            .values
            .iter()
            .map(|value| *value >= DEFAULT_TEXT_MASK_THRESHOLD)
            .collect(),
    }
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

    fn text_probabilities_for(image: &RgbImage, text_probability: f32) -> ProbabilityMap {
        ProbabilityMap {
            width: image.width(),
            height: image.height(),
            values: image
                .pixels()
                .map(|pixel| {
                    if pixel.0 == [255, 255, 255] {
                        0.0
                    } else {
                        text_probability
                    }
                })
                .collect(),
        }
    }

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
        let probabilities = text_probabilities_for(&image, 0.95);
        let bounds = segment_text_line_bounds(&DynamicImage::ImageRgb8(image), &probabilities);
        assert_eq!(bounds.len(), 2);
        assert!(bounds[0].bottom <= bounds[1].top);
        assert!(bounds.iter().all(|line| line.right > line.left));
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

        let probabilities = text_probabilities_for(&image, 0.95);
        let bounds = segment_text_line_bounds(&DynamicImage::ImageRgb8(image), &probabilities);
        assert_eq!(bounds.len(), 1);
        assert!(bounds[0].left <= 18);
        assert!(bounds[0].right >= 104);
        assert!(bounds[0].top <= 10);
        assert!(bounds[0].bottom >= 28);
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

        let probabilities = text_probabilities_for(&image, 0.95);
        let bounds = segment_text_line_bounds(&DynamicImage::ImageRgb8(image), &probabilities);
        assert_eq!(bounds.len(), 2);
        assert!(bounds[0].bottom >= 26);
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
