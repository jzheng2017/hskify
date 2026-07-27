use std::io::Cursor;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageFormat, Rgba, RgbaImage};

use super::geometry::{PixelBounds, PixelRect};
use super::ppocr_v5::PpOcrInkMask;

const COARSE_DIFFUSION_STEPS: usize = 32;
const REFINEMENT_DIFFUSION_STEPS: usize = 8;
const LOCAL_TEXT_PADDING_RATIO: f32 = 0.08;
const MIN_TEXT_PADDING_PIXELS: f32 = 3.0;
const BUBBLE_CONTOUR_GUARD_PIXELS: f32 = 3.0;

#[derive(Debug)]
pub(super) struct PatchPng {
    pub bounds: PixelBounds,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy)]
pub(super) struct PlacedInkMask<'a> {
    pub(super) mask: &'a PpOcrInkMask,
    pub(super) crop_bounds: PixelBounds,
}

pub(super) fn make_cleanup_patch(
    source: &DynamicImage,
    text_rect: PixelRect,
    bubble_rect: PixelRect,
    inks: &[PlacedInkMask<'_>],
) -> Result<PatchPng> {
    let (bounds, patch, _) =
        make_cleanup_patch_image_with_ink(source, text_rect, bubble_rect, inks);
    let mut cursor = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(patch)
        .write_to(&mut cursor, ImageFormat::Png)
        .context("encode transparent cleanup patch")?;
    Ok(PatchPng {
        bounds,
        bytes: cursor.into_inner(),
    })
}

#[cfg(test)]
fn make_cleanup_patch_image(
    source: &DynamicImage,
    text_rect: PixelRect,
    bubble_rect: PixelRect,
) -> (PixelBounds, RgbaImage, FillStrategy) {
    make_cleanup_patch_image_with_ink(source, text_rect, bubble_rect, &[])
}

fn make_cleanup_patch_image_with_ink(
    source: &DynamicImage,
    text_rect: PixelRect,
    bubble_rect: PixelRect,
    inks: &[PlacedInkMask<'_>],
) -> (PixelBounds, RgbaImage, FillStrategy) {
    let (image_width, image_height) = source.dimensions();
    let local_text_scale = text_rect.width().min(text_rect.height()).max(1.0);
    let padding = (local_text_scale * LOCAL_TEXT_PADDING_RATIO).max(MIN_TEXT_PADDING_PIXELS);
    let dilation = (local_text_scale * 0.04).ceil().clamp(3.0, 8.0) as u32;
    let sample_margin = padding.clamp(4.0, 18.0);
    // The accepted layout bounds may include a balloon contour or surrounding
    // narration background. Keep a small interior guard when there is room,
    // but never clip the detected text bounds themselves.
    let guarded_bubble_x0 = (bubble_rect.x0 + BUBBLE_CONTOUR_GUARD_PIXELS).min(text_rect.x0);
    let guarded_bubble_y0 = (bubble_rect.y0 + BUBBLE_CONTOUR_GUARD_PIXELS).min(text_rect.y0);
    let guarded_bubble_x1 = (bubble_rect.x1 - BUBBLE_CONTOUR_GUARD_PIXELS).max(text_rect.x1);
    let guarded_bubble_y1 = (bubble_rect.y1 - BUBBLE_CONTOUR_GUARD_PIXELS).max(text_rect.y1);
    let fallback_erase_rect = PixelRect {
        x0: (text_rect.x0 - padding).max(guarded_bubble_x0).max(0.0),
        y0: (text_rect.y0 - padding).max(guarded_bubble_y0).max(0.0),
        x1: (text_rect.x1 + padding)
            .min(guarded_bubble_x1)
            .min(image_width as f32),
        y1: (text_rect.y1 + padding)
            .min(guarded_bubble_y1)
            .min(image_height as f32),
    };
    let ink_rect = inks
        .iter()
        .copied()
        .filter_map(placed_ink_bounds)
        .reduce(PixelRect::union);
    let erase_rect = ink_rect
        .map(|rect| rect.expand(dilation as f32, image_width, image_height))
        .and_then(|rect| rect.intersection(bubble_rect))
        .unwrap_or(fallback_erase_rect);
    let patch_rect = PixelRect {
        x0: (erase_rect.x0 - sample_margin).max(0.0),
        y0: (erase_rect.y0 - sample_margin).max(0.0),
        x1: (erase_rect.x1 + sample_margin).min(image_width as f32),
        y1: (erase_rect.y1 + sample_margin).min(image_height as f32),
    };
    let bounds = patch_rect.pixel_bounds(image_width, image_height);
    let pixels = source
        .crop_imm(bounds.x, bounds.y, bounds.width, bounds.height)
        .to_rgba8();
    let erase = InnerBounds {
        x0: (erase_rect.x0 - bounds.x as f32).floor().max(0.0) as u32,
        y0: (erase_rect.y0 - bounds.y as f32).floor().max(0.0) as u32,
        x1: (erase_rect.x1 - bounds.x as f32)
            .ceil()
            .min(bounds.width as f32) as u32,
        y1: (erase_rect.y1 - bounds.y as f32)
            .ceil()
            .min(bounds.height as f32) as u32,
    };
    let bubble = InnerBounds {
        x0: (bubble_rect.x0 - bounds.x as f32).floor().max(0.0) as u32,
        y0: (bubble_rect.y0 - bounds.y as f32).floor().max(0.0) as u32,
        x1: (bubble_rect.x1 - bounds.x as f32)
            .ceil()
            .min(bounds.width as f32) as u32,
        y1: (bubble_rect.y1 - bounds.y as f32)
            .ceil()
            .min(bounds.height as f32) as u32,
    };
    let mask = placed_ink_erase_mask(
        pixels.width(),
        pixels.height(),
        bounds,
        bubble,
        inks,
        dilation,
    )
    .unwrap_or_else(|| {
        rectangular_erase_mask(pixels.width(), pixels.height(), erase.intersection(bubble))
    });
    let (fill, strategy) = inpaint_colors(&pixels, &mask);
    let patch = transparent_patch_from_mask(&mask, &fill, pixels.width(), pixels.height());
    (bounds, patch, strategy)
}

fn placed_ink_bounds(placed: PlacedInkMask<'_>) -> Option<PixelRect> {
    let mask = placed.mask;
    if mask.width != placed.crop_bounds.width
        || mask.height != placed.crop_bounds.height
        || mask.values.len() != (mask.width as usize).saturating_mul(mask.height as usize)
    {
        return None;
    }
    let mut left = mask.width;
    let mut top = mask.height;
    let mut right = 0_u32;
    let mut bottom = 0_u32;
    let mut found = false;
    for y in 0..mask.height {
        for x in 0..mask.width {
            if mask.values[index(mask.width, x, y)] {
                found = true;
                left = left.min(x);
                top = top.min(y);
                right = right.max(x + 1);
                bottom = bottom.max(y + 1);
            }
        }
    }
    found.then(|| PixelRect {
        x0: (placed.crop_bounds.x + left) as f32,
        y0: (placed.crop_bounds.y + top) as f32,
        x1: (placed.crop_bounds.x + right) as f32,
        y1: (placed.crop_bounds.y + bottom) as f32,
    })
}

fn placed_ink_erase_mask(
    width: u32,
    height: u32,
    patch_bounds: PixelBounds,
    bubble: InnerBounds,
    placed_masks: &[PlacedInkMask<'_>],
    dilation: u32,
) -> Option<Vec<bool>> {
    let mut base = vec![false; (width * height) as usize];
    let mut found = false;
    for placed in placed_masks.iter().copied() {
        if placed_ink_bounds(placed).is_none() {
            continue;
        }
        for crop_y in 0..placed.mask.height {
            for crop_x in 0..placed.mask.width {
                if !placed.mask.values[index(placed.mask.width, crop_x, crop_y)] {
                    continue;
                }
                let global_x = placed.crop_bounds.x + crop_x;
                let global_y = placed.crop_bounds.y + crop_y;
                let Some(local_x) = global_x.checked_sub(patch_bounds.x) else {
                    continue;
                };
                let Some(local_y) = global_y.checked_sub(patch_bounds.y) else {
                    continue;
                };
                if local_x < width && local_y < height {
                    found = true;
                    base[index(width, local_x, local_y)] = true;
                }
            }
        }
    }
    if !found {
        return None;
    }
    let mut dilated = dilate_mask_square(&base, width, height, dilation);
    for y in 0..height {
        for x in 0..width {
            if x < bubble.x0 || x >= bubble.x1 || y < bubble.y0 || y >= bubble.y1 {
                dilated[index(width, x, y)] = false;
            }
        }
    }
    dilated.iter().any(|value| *value).then_some(dilated)
}

fn dilate_mask_square(mask: &[bool], width: u32, height: u32, radius: u32) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    let mut horizontal = vec![false; mask.len()];
    for y in 0..height {
        let mut prefix = vec![0_u32; width as usize + 1];
        for x in 0..width {
            prefix[x as usize + 1] = prefix[x as usize] + u32::from(mask[index(width, x, y)]);
        }
        for x in 0..width {
            let left = x.saturating_sub(radius) as usize;
            let right = x.saturating_add(radius).saturating_add(1).min(width) as usize;
            horizontal[index(width, x, y)] = prefix[right] > prefix[left];
        }
    }
    let mut output = vec![false; mask.len()];
    for x in 0..width {
        let mut prefix = vec![0_u32; height as usize + 1];
        for y in 0..height {
            prefix[y as usize + 1] = prefix[y as usize] + u32::from(horizontal[index(width, x, y)]);
        }
        for y in 0..height {
            let top = y.saturating_sub(radius) as usize;
            let bottom = y.saturating_add(radius).saturating_add(1).min(height) as usize;
            output[index(width, x, y)] = prefix[bottom] > prefix[top];
        }
    }
    output
}

#[derive(Clone, Copy)]
struct InnerBounds {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

impl InnerBounds {
    fn intersection(self, other: Self) -> Self {
        let x0 = self.x0.max(other.x0);
        let y0 = self.y0.max(other.y0);
        let x1 = self.x1.min(other.x1).max(x0);
        let y1 = self.y1.min(other.y1).max(y0);
        Self { x0, y0, x1, y1 }
    }
}

fn rectangular_erase_mask(width: u32, height: u32, erase: InnerBounds) -> Vec<bool> {
    let mut mask = vec![false; (width * height) as usize];
    for y in erase.y0.min(height)..erase.y1.min(height) {
        for x in erase.x0.min(width)..erase.x1.min(width) {
            mask[index(width, x, y)] = true;
        }
    }
    mask
}

fn border_colors(pixels: &RgbaImage) -> Vec<[u8; 3]> {
    let width = pixels.width();
    let height = pixels.height();
    let border = 2_u32.min(width).min(height);
    let mut colors = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if (x < border || y < border || x + border >= width || y + border >= height)
                && pixels.get_pixel(x, y)[3] != 0
            {
                let pixel = pixels.get_pixel(x, y);
                colors.push([pixel[0], pixel[1], pixel[2]]);
            }
        }
    }
    if colors.is_empty() {
        colors.push([255, 255, 255]);
    }
    colors
}

fn median_rgb(colors: &[[u8; 3]]) -> [u8; 3] {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for color in colors {
        for channel in 0..3 {
            channels[channel].push(color[channel]);
        }
    }
    for values in &mut channels {
        values.sort_unstable();
    }
    [
        channels[0][channels[0].len() / 2],
        channels[1][channels[1].len() / 2],
        channels[2][channels[2].len() / 2],
    ]
}

fn color_variation(colors: &[[u8; 3]], center: [u8; 3]) -> f32 {
    colors
        .iter()
        .map(|color| color_distance(*color, center))
        .sum::<f32>()
        / colors.len().max(1) as f32
}

fn color_distance(left: [u8; 3], right: [u8; 3]) -> f32 {
    let red = left[0] as f32 - right[0] as f32;
    let green = left[1] as f32 - right[1] as f32;
    let blue = left[2] as f32 - right[2] as f32;
    (red * red + green * green + blue * blue).sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FillStrategy {
    Flat,
    MultiscaleDiffusion,
}

fn inpaint_colors(pixels: &RgbaImage, mask: &[bool]) -> (Vec<[u8; 3]>, FillStrategy) {
    let width = pixels.width();
    let height = pixels.height();
    let mut boundary = Vec::<[u8; 3]>::new();
    for y in 0..height {
        for x in 0..width {
            if mask[index(width, x, y)] {
                continue;
            }
            if cardinal_neighbors(x, y, width, height)
                .any(|(next_x, next_y)| mask[index(width, next_x, next_y)])
            {
                let pixel = pixels.get_pixel(x, y);
                boundary.push([pixel[0], pixel[1], pixel[2]]);
            }
        }
    }
    if boundary.is_empty() {
        boundary = border_colors(pixels);
    }
    let flat_fill = median_rgb(&boundary);
    let variation = color_variation(&boundary, flat_fill);
    if variation <= 15.0 {
        return (vec![flat_fill; mask.len()], FillStrategy::Flat);
    }

    (
        multiscale_diffusion(pixels, mask, flat_fill),
        FillStrategy::MultiscaleDiffusion,
    )
}

#[derive(Clone)]
struct DiffusionLevel {
    width: u32,
    height: u32,
    colors: Vec<[f32; 3]>,
    fixed: Vec<bool>,
}

fn multiscale_diffusion(pixels: &RgbaImage, mask: &[bool], fallback: [u8; 3]) -> Vec<[u8; 3]> {
    let mut levels = vec![DiffusionLevel {
        width: pixels.width(),
        height: pixels.height(),
        colors: pixels
            .pixels()
            .map(|pixel| [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32])
            .collect(),
        fixed: mask.iter().map(|masked| !masked).collect(),
    }];
    while {
        let last = levels.last().expect("diffusion pyramid is non-empty");
        last.width > 8 || last.height > 8
    } {
        let coarse = downsample_level(levels.last().expect("diffusion pyramid is non-empty"));
        if coarse.width == levels.last().unwrap().width
            && coarse.height == levels.last().unwrap().height
        {
            break;
        }
        levels.push(coarse);
    }

    let fallback = [fallback[0] as f32, fallback[1] as f32, fallback[2] as f32];
    {
        let coarsest = levels.last_mut().expect("diffusion pyramid is non-empty");
        for (position, fixed) in coarsest.fixed.iter().enumerate() {
            if !fixed {
                coarsest.colors[position] = fallback;
            }
        }
        relax_level(coarsest, COARSE_DIFFUSION_STEPS);
    }

    for level_index in (0..levels.len().saturating_sub(1)).rev() {
        let coarse = levels[level_index + 1].clone();
        let fine = &mut levels[level_index];
        for y in 0..fine.height {
            for x in 0..fine.width {
                let position = index(fine.width, x, y);
                if !fine.fixed[position] {
                    fine.colors[position] = coarse.colors[index(coarse.width, x / 2, y / 2)];
                }
            }
        }
        relax_level(fine, REFINEMENT_DIFFUSION_STEPS);
    }

    levels
        .remove(0)
        .colors
        .into_iter()
        .map(|color| {
            [
                color[0].round().clamp(0.0, 255.0) as u8,
                color[1].round().clamp(0.0, 255.0) as u8,
                color[2].round().clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

fn downsample_level(fine: &DiffusionLevel) -> DiffusionLevel {
    let width = fine.width.div_ceil(2);
    let height = fine.height.div_ceil(2);
    let mut colors = vec![[0.0; 3]; (width * height) as usize];
    let mut fixed = vec![false; colors.len()];
    for coarse_y in 0..height {
        for coarse_x in 0..width {
            let mut known_sum = [0.0_f32; 3];
            let mut known_count = 0_u32;
            for fine_y in (coarse_y * 2)..(coarse_y * 2 + 2).min(fine.height) {
                for fine_x in (coarse_x * 2)..(coarse_x * 2 + 2).min(fine.width) {
                    let fine_position = index(fine.width, fine_x, fine_y);
                    if fine.fixed[fine_position] {
                        for channel in 0..3 {
                            known_sum[channel] += fine.colors[fine_position][channel];
                        }
                        known_count += 1;
                    }
                }
            }
            let coarse_position = index(width, coarse_x, coarse_y);
            if known_count > 0 {
                fixed[coarse_position] = true;
                for channel in 0..3 {
                    colors[coarse_position][channel] = known_sum[channel] / known_count as f32;
                }
            }
        }
    }
    DiffusionLevel {
        width,
        height,
        colors,
        fixed,
    }
}

fn relax_level(level: &mut DiffusionLevel, iterations: usize) {
    for _ in 0..iterations {
        let mut next = level.colors.clone();
        for y in 0..level.height {
            for x in 0..level.width {
                let position = index(level.width, x, y);
                if level.fixed[position] {
                    continue;
                }
                let mut sum = [0.0_f32; 3];
                let mut count = 0.0_f32;
                for (next_x, next_y) in cardinal_neighbors(x, y, level.width, level.height) {
                    let color = level.colors[index(level.width, next_x, next_y)];
                    for channel in 0..3 {
                        sum[channel] += color[channel];
                    }
                    count += 1.0;
                }
                if count > 0.0 {
                    for channel in 0..3 {
                        next[position][channel] = sum[channel] / count;
                    }
                }
            }
        }
        level.colors = next;
    }
}

fn cardinal_neighbors(x: u32, y: u32, width: u32, height: u32) -> impl Iterator<Item = (u32, u32)> {
    [
        x.checked_sub(1).map(|next_x| (next_x, y)),
        (x + 1 < width).then_some((x + 1, y)),
        y.checked_sub(1).map(|next_y| (x, next_y)),
        (y + 1 < height).then_some((x, y + 1)),
    ]
    .into_iter()
    .flatten()
}

fn transparent_patch_from_mask(
    mask: &[bool],
    fill: &[[u8; 3]],
    width: u32,
    height: u32,
) -> RgbaImage {
    let mut output = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let position = index(width, x, y);
            if mask[position] {
                output.put_pixel(
                    x,
                    y,
                    Rgba([fill[position][0], fill[position][1], fill[position][2], 255]),
                );
            }
        }
    }
    output
}

fn index(width: u32, x: u32, y: u32) -> usize {
    (y * width + x) as usize
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn cleanup_patch_alpha_is_bounded_exactly_by_the_mask() {
        let mask = vec![false, true, false, true, true, false];
        let fill = vec![[240, 241, 242]; mask.len()];
        let patch = transparent_patch_from_mask(&mask, &fill, 3, 2);
        for (position, pixel) in patch.pixels().enumerate() {
            assert_eq!(pixel[3], if mask[position] { 255 } else { 0 });
        }
    }

    #[test]
    fn ocr_ink_mask_is_dilated_locally_without_filling_the_text_rectangle() {
        let source =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(40, 30, Rgba([250, 250, 250, 255])));
        let text = PixelRect::new(10.0, 10.0, 30.0, 20.0).unwrap();
        let bubble = PixelRect::new(5.0, 5.0, 35.0, 25.0).unwrap();
        let mut values = vec![false; 24 * 14];
        values[index(24, 10, 6)] = true;
        let ink = PpOcrInkMask {
            width: 24,
            height: 14,
            values,
        };
        let placed = PlacedInkMask {
            mask: &ink,
            crop_bounds: PixelBounds {
                x: 8,
                y: 8,
                width: 24,
                height: 14,
            },
        };

        let (bounds, patch, _) =
            make_cleanup_patch_image_with_ink(&source, text, bubble, &[placed]);
        let alpha_pixels = patch.pixels().filter(|pixel| pixel[3] != 0).count();

        assert_eq!(alpha_pixels, 49);
        let center_x = 18 - bounds.x;
        let center_y = 14 - bounds.y;
        assert_eq!(patch.get_pixel(center_x, center_y)[3], 255);
        assert_eq!(patch.get_pixel(0, 0)[3], 0);
    }

    #[test]
    fn full_erase_mask_never_crosses_the_confirmed_bubble_interior() {
        let erase = InnerBounds {
            x0: 1,
            y0: 1,
            x1: 6,
            y1: 6,
        };
        let bubble = InnerBounds {
            x0: 2,
            y0: 2,
            x1: 5,
            y1: 5,
        };
        let mask = rectangular_erase_mask(7, 7, erase.intersection(bubble));

        for y in 0..7 {
            for x in 0..7 {
                assert_eq!(
                    mask[index(7, x, y)],
                    x >= bubble.x0 && x < bubble.x1 && y >= bubble.y0 && y < bubble.y1
                );
            }
        }
    }

    #[test]
    fn multiscale_diffusion_fills_every_masked_pixel_without_touching_source_pixels() {
        let mut pixels = RgbaImage::new(17, 17);
        for y in 0..17 {
            for x in 0..17 {
                pixels.put_pixel(x, y, Rgba([x as u8 * 8, y as u8 * 8, 64, 255]));
            }
        }
        let erase = InnerBounds {
            x0: 3,
            y0: 3,
            x1: 14,
            y1: 14,
        };
        let mask = rectangular_erase_mask(17, 17, erase);
        let fill = multiscale_diffusion(&pixels, &mask, [64, 64, 64]);

        for y in 0..17 {
            for x in 0..17 {
                let position = index(17, x, y);
                if !mask[position] {
                    let original = pixels.get_pixel(x, y);
                    assert_eq!(fill[position], [original[0], original[1], original[2]]);
                }
            }
        }
        assert!(fill[index(17, 8, 8)][0] > 0);
        assert!(fill[index(17, 8, 8)][1] > 0);
    }

    #[test]
    fn flat_bubble_uses_the_constant_local_fill_path() {
        let pixels = RgbaImage::from_pixel(21, 17, Rgba([242, 243, 244, 255]));
        let mask = rectangular_erase_mask(
            21,
            17,
            InnerBounds {
                x0: 4,
                y0: 3,
                x1: 17,
                y1: 14,
            },
        );
        let (fill, strategy) = inpaint_colors(&pixels, &mask);

        assert_eq!(strategy, FillStrategy::Flat);
        assert!(fill.iter().all(|color| *color == [242, 243, 244]));
    }

    #[test]
    fn gradient_bubble_uses_nonuniform_multiscale_diffusion() {
        let mut pixels = RgbaImage::new(41, 25);
        for y in 0..pixels.height() {
            for x in 0..pixels.width() {
                pixels.put_pixel(
                    x,
                    y,
                    Rgba([(40 + x * 4) as u8, (80 + y * 3) as u8, 160, 255]),
                );
            }
        }
        let mask = rectangular_erase_mask(
            pixels.width(),
            pixels.height(),
            InnerBounds {
                x0: 7,
                y0: 5,
                x1: 34,
                y1: 20,
            },
        );
        let (fill, strategy) = inpaint_colors(&pixels, &mask);

        assert_eq!(strategy, FillStrategy::MultiscaleDiffusion);
        assert!(fill[index(pixels.width(), 10, 12)][0] < fill[index(pixels.width(), 30, 12)][0]);
        assert!(fill[index(pixels.width(), 20, 7)][1] < fill[index(pixels.width(), 20, 17)][1]);
    }

    #[test]
    fn tight_border_patch_stays_clipped_and_transparent() {
        let source =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(24, 20, Rgba([250, 250, 250, 255])));
        let text = PixelRect::new(0.0, 1.0, 7.0, 7.0).unwrap();
        let bubble = PixelRect::new(0.0, 0.0, 11.0, 10.0).unwrap();
        let patch = make_cleanup_patch(&source, text, bubble, &[]).unwrap();

        assert_eq!(patch.bounds.x, 0);
        assert_eq!(&patch.bytes[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = image::load_from_memory_with_format(&patch.bytes, ImageFormat::Png)
            .unwrap()
            .to_rgba8();
        assert!(decoded.pixels().any(|pixel| pixel[3] == 0));
        assert!(decoded.pixels().any(|pixel| pixel[3] == 255));
    }

    #[test]
    fn vertical_padding_preserves_the_confirmed_bubble_contour_guard() {
        let source =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(100, 80, Rgba([250, 250, 250, 255])));
        let text = PixelRect::new(30.0, 14.0, 70.0, 39.0).unwrap();
        let bubble = PixelRect::new(20.0, 10.0, 80.0, 60.0).unwrap();
        let (bounds, patch, _) = make_cleanup_patch_image(&source, text, bubble);
        let mut first_opaque_y = u32::MAX;

        for y in 0..patch.height() {
            for x in 0..patch.width() {
                if patch.get_pixel(x, y)[3] != 0 {
                    first_opaque_y = first_opaque_y.min(bounds.y + y);
                }
            }
        }

        assert_eq!(first_opaque_y, 13);
    }

    #[test]
    #[ignore = "requires the complete chapter 5 gold fixture and its 36 hash-pinned source WebPs"]
    fn chapter5_cleanup_patch_correctness() {
        let repository = find_repository_root();
        let fixture = repository.join("fixtures/benchmarks/30-years-since-the-prologue-chapter-5");
        let source_root =
            repository.join(".cache/benchmarks/30-years-since-the-prologue-chapter-5/source");
        let manifest: Value = serde_json::from_slice(
            &fs::read(fixture.join("manifest.json")).expect("read benchmark manifest"),
        )
        .expect("parse benchmark manifest");
        assert_eq!(
            manifest["id"].as_str().expect("benchmark ID"),
            "30-years-since-the-prologue-chapter-5"
        );

        let mut region_count = 0_u64;
        let mut erase_pixels = 0_u64;
        let mut covered_erase_pixels = 0_u64;
        let mut erase_pixel_envelope_pixels = 0_u64;
        let mut covered_erase_pixel_envelope_pixels = 0_u64;
        let mut text_pixel_envelope_pixels = 0_u64;
        let mut covered_text_pixel_envelope_pixels = 0_u64;
        let mut erase_envelope_outside_bubble_polygon_pixels = 0_u64;
        let mut alpha_pixels = 0_u64;
        let mut alpha_outside_erase_pixels = 0_u64;
        let mut alpha_outside_erase_pixel_envelope_pixels = 0_u64;
        let mut alpha_outside_bubble_rect_pixels = 0_u64;
        let mut alpha_outside_bubble_polygon_pixels = 0_u64;
        let mut composite_changed_pixels = 0_u64;
        let mut changed_outside_alpha_pixels = 0_u64;
        let mut changed_outside_erase_pixels = 0_u64;
        let mut changed_outside_erase_pixel_envelope_pixels = 0_u64;
        let mut transparent_pixels = 0_u64;
        let mut flat_fill_regions = 0_u64;
        let mut diffusion_regions = 0_u64;
        let mut punctuation_only_regions = 0_u64;
        let mut punctuation_erase_pixels = 0_u64;
        let mut punctuation_covered_pixels = 0_u64;
        let mut minimum_region_coverage = 1.0_f64;
        let mut worst_region = String::new();
        let mut region_evidence = Vec::new();

        let images = manifest["images"]
            .as_array()
            .expect("manifest images are an array");
        let page_count = manifest["pageCount"].as_u64().expect("manifest pageCount");
        assert_eq!(images.len() as u64, page_count);
        let annotation_status = manifest["annotationStatus"]
            .as_object()
            .expect("manifest annotationStatus is an object");
        let completed_pages = annotation_status["completedPageCount"]
            .as_u64()
            .expect("completedPageCount");
        let required_pages = annotation_status["requiredPageCount"]
            .as_u64()
            .expect("requiredPageCount");
        assert_eq!(
            annotation_status["status"].as_str(),
            Some("complete"),
            "chapter 5 gold fixture is incomplete: completedPageCount={completed_pages}, requiredPageCount={required_pages}, reasonCode={:?}",
            annotation_status["reasonCode"].as_str()
        );
        assert_eq!(completed_pages, page_count);
        assert_eq!(required_pages, page_count);
        for image_entry in images {
            let page_order = image_entry["order"].as_u64().expect("image order");
            let file_name = image_entry["file"].as_str().expect("image file");
            let source_path = source_root.join(file_name);
            let source_bytes = fs::read(&source_path)
                .unwrap_or_else(|error| panic!("read {}: {error}", source_path.display()));
            let source_sha = format!("{:x}", Sha256::digest(&source_bytes));
            assert_eq!(
                source_sha,
                image_entry["sha256"].as_str().expect("source SHA-256"),
                "source hash for {file_name}"
            );
            let source = image::load_from_memory(&source_bytes)
                .unwrap_or_else(|error| panic!("decode {file_name}: {error}"));
            let annotation_path =
                fixture.join(image_entry["annotation"].as_str().expect("annotation path"));
            let annotation: Value =
                serde_json::from_slice(&fs::read(&annotation_path).expect("read annotation"))
                    .expect("parse annotation");
            let width = annotation["page"]["width"].as_u64().expect("page width") as u32;
            let height = annotation["page"]["height"].as_u64().expect("page height") as u32;
            assert_eq!(source.dimensions(), (width, height));

            for (region_index, region) in annotation["regions"]
                .as_array()
                .expect("annotation regions")
                .iter()
                .enumerate()
            {
                region_count += 1;
                let id = region["id"].as_str().expect("region ID");
                assert_eq!(id, format!("30ysp-ch5-p{page_order:03}-r{region_index:02}"));
                let source_english = region["sourceEnglish"].as_str().expect("source English");
                let text_polygon = normalized_polygon(&region["textPolygon"], width, height);
                let bubble_polygon = normalized_polygon(&region["bubblePolygon"], width, height);
                let erase_polygon =
                    normalized_polygon(&region["eraseMask"]["polygon"], width, height);
                let text_rect = polygon_bounds(&text_polygon);
                let bubble_rect = polygon_bounds(&bubble_polygon);
                let (bounds, patch, strategy) =
                    make_cleanup_patch_image(&source, text_rect, bubble_rect);
                match strategy {
                    FillStrategy::Flat => flat_fill_regions += 1,
                    FillStrategy::MultiscaleDiffusion => diffusion_regions += 1,
                }

                let region_erase_pixels = polygon_pixel_count(&erase_polygon);
                let erase_pixel_envelope = polygon_pixel_envelope(&erase_polygon);
                let region_erase_pixel_envelope_pixels = erase_pixel_envelope.area();
                let text_pixel_envelope = polygon_pixel_envelope(&text_polygon);
                let region_text_pixel_envelope_pixels = text_pixel_envelope.area();
                let mut region_covered_pixels = 0_u64;
                let mut region_covered_envelope_pixels = 0_u64;
                let mut region_covered_text_envelope_pixels = 0_u64;
                let mut region_erase_envelope_outside_bubble_polygon = 0_u64;
                let mut region_alpha_pixels = 0_u64;
                let mut region_alpha_outside_erase = 0_u64;
                let mut region_alpha_outside_erase_envelope = 0_u64;
                let mut region_alpha_outside_bubble_polygon = 0_u64;
                let mut region_changed_outside_erase = 0_u64;
                let mut region_changed_outside_erase_envelope = 0_u64;
                for local_y in 0..patch.height() {
                    for local_x in 0..patch.width() {
                        let global_x = bounds.x + local_x;
                        let global_y = bounds.y + local_y;
                        let point = (global_x as f32 + 0.5, global_y as f32 + 0.5);
                        let in_erase = point_in_polygon(point, &erase_polygon);
                        let in_erase_envelope = erase_pixel_envelope.contains(global_x, global_y);
                        let in_bubble_polygon = point_in_polygon(point, &bubble_polygon);
                        let in_bubble_rect = point.0 >= bubble_rect.x0
                            && point.0 <= bubble_rect.x1
                            && point.1 >= bubble_rect.y0
                            && point.1 <= bubble_rect.y1;
                        if in_erase_envelope && !in_bubble_polygon {
                            region_erase_envelope_outside_bubble_polygon += 1;
                        }
                        let patch_pixel = patch.get_pixel(local_x, local_y);
                        let alpha = patch_pixel[3] != 0;
                        if alpha {
                            region_alpha_pixels += 1;
                            if in_erase {
                                region_covered_pixels += 1;
                            } else {
                                region_alpha_outside_erase += 1;
                            }
                            if in_erase_envelope {
                                region_covered_envelope_pixels += 1;
                            } else {
                                region_alpha_outside_erase_envelope += 1;
                            }
                            if text_pixel_envelope.contains(global_x, global_y) {
                                region_covered_text_envelope_pixels += 1;
                            }
                            if !in_bubble_rect {
                                alpha_outside_bubble_rect_pixels += 1;
                            }
                            if !in_bubble_polygon {
                                region_alpha_outside_bubble_polygon += 1;
                            }
                        } else {
                            transparent_pixels += 1;
                        }

                        let source_pixel = source.get_pixel(global_x, global_y);
                        let composite = if alpha { *patch_pixel } else { source_pixel };
                        let changed = composite != source_pixel;
                        if changed {
                            composite_changed_pixels += 1;
                            if !alpha {
                                changed_outside_alpha_pixels += 1;
                            }
                            if !in_erase {
                                changed_outside_erase_pixels += 1;
                                region_changed_outside_erase += 1;
                            }
                            if !in_erase_envelope {
                                changed_outside_erase_pixel_envelope_pixels += 1;
                                region_changed_outside_erase_envelope += 1;
                            }
                        }
                    }
                }

                erase_pixels += region_erase_pixels;
                covered_erase_pixels += region_covered_pixels;
                erase_pixel_envelope_pixels += region_erase_pixel_envelope_pixels;
                covered_erase_pixel_envelope_pixels += region_covered_envelope_pixels;
                text_pixel_envelope_pixels += region_text_pixel_envelope_pixels;
                covered_text_pixel_envelope_pixels += region_covered_text_envelope_pixels;
                erase_envelope_outside_bubble_polygon_pixels +=
                    region_erase_envelope_outside_bubble_polygon;
                alpha_pixels += region_alpha_pixels;
                alpha_outside_erase_pixels += region_alpha_outside_erase;
                alpha_outside_erase_pixel_envelope_pixels += region_alpha_outside_erase_envelope;
                alpha_outside_bubble_polygon_pixels += region_alpha_outside_bubble_polygon;
                let coverage = region_covered_pixels as f64 / region_erase_pixels as f64;
                if coverage < minimum_region_coverage {
                    minimum_region_coverage = coverage;
                    worst_region = id.to_owned();
                }
                let punctuation_only = !source_english.chars().any(char::is_alphabetic);
                if punctuation_only {
                    punctuation_only_regions += 1;
                    punctuation_erase_pixels += region_erase_pixels;
                    punctuation_covered_pixels += region_covered_pixels;
                }
                region_evidence.push(json!({
                    "id": id,
                    "sourceEnglish": source_english,
                    "fillStrategy": match strategy {
                        FillStrategy::Flat => "flat-local-median",
                        FillStrategy::MultiscaleDiffusion => "local-multiscale-diffusion",
                    },
                    "erasePixels": region_erase_pixels,
                    "coveredErasePixels": region_covered_pixels,
                    "coverage": coverage,
                    "erasePixelEnvelopePixels": region_erase_pixel_envelope_pixels,
                    "coveredErasePixelEnvelopePixels": region_covered_envelope_pixels,
                    "textPixelEnvelopePixels": region_text_pixel_envelope_pixels,
                    "coveredTextPixelEnvelopePixels": region_covered_text_envelope_pixels,
                    "erasePixelEnvelopeOutsideBubblePolygonPixels": region_erase_envelope_outside_bubble_polygon,
                    "alphaPixels": region_alpha_pixels,
                    "alphaOutsideErasePolygonCenterPixels": region_alpha_outside_erase,
                    "alphaOutsideErasePixelEnvelopePixels": region_alpha_outside_erase_envelope,
                    "alphaOutsideBubblePolygonPixels": region_alpha_outside_bubble_polygon,
                    "changedOutsideErasePolygonCenterPixels": region_changed_outside_erase,
                    "changedOutsideErasePixelEnvelopePixels": region_changed_outside_erase_envelope,
                    "punctuationOnly": punctuation_only,
                }));
            }
        }

        assert_eq!(
            region_count,
            manifest["totalExpectedDialogueBubbleCount"]
                .as_u64()
                .expect("manifest totalExpectedDialogueBubbleCount")
        );
        assert_eq!(
            covered_text_pixel_envelope_pixels,
            text_pixel_envelope_pixels
        );
        assert_eq!(alpha_outside_bubble_rect_pixels, 0);
        assert_eq!(changed_outside_alpha_pixels, 0);
        assert!(transparent_pixels > 0);
        assert_eq!(
            punctuation_only_regions,
            manifest["totalExpectedUntouchedExclusionCount"]
                .as_u64()
                .expect("manifest totalExpectedUntouchedExclusionCount")
        );
        assert!(flat_fill_regions > 0);
        assert!(diffusion_regions > 0);

        let evidence = json!({
            "schemaVersion": 1,
            "benchmarkId": manifest["id"],
            "evaluationInput": "committed annotation geometry only; no detector, OCR, translation, or model inference",
            "sourceImages": images.len(),
            "regions": region_count,
            "erasePixels": erase_pixels,
            "coveredErasePixels": covered_erase_pixels,
            "eraseCoverage": covered_erase_pixels as f64 / erase_pixels as f64,
            "erasePixelEnvelopePixels": erase_pixel_envelope_pixels,
            "coveredErasePixelEnvelopePixels": covered_erase_pixel_envelope_pixels,
            "erasePixelEnvelopeCoverage": covered_erase_pixel_envelope_pixels as f64 / erase_pixel_envelope_pixels as f64,
            "textPixelEnvelopePixels": text_pixel_envelope_pixels,
            "coveredTextPixelEnvelopePixels": covered_text_pixel_envelope_pixels,
            "textPixelEnvelopeCoverage": covered_text_pixel_envelope_pixels as f64 / text_pixel_envelope_pixels as f64,
            "erasePixelEnvelopeOutsideEvaluationBubblePolygonPixels": erase_envelope_outside_bubble_polygon_pixels,
            "minimumRegionCoverage": minimum_region_coverage,
            "worstRegion": worst_region,
            "alphaPixels": alpha_pixels,
            "alphaOutsideErasePolygonCenterPixels": alpha_outside_erase_pixels,
            "alphaOutsideErasePixelEnvelopePixels": alpha_outside_erase_pixel_envelope_pixels,
            "alphaOutsideConfirmedBubbleRectPixels": alpha_outside_bubble_rect_pixels,
            "alphaOutsideEvaluationBubblePolygonPixels": alpha_outside_bubble_polygon_pixels,
            "compositeChangedPixels": composite_changed_pixels,
            "changedOutsideAlphaPixels": changed_outside_alpha_pixels,
            "changedOutsideErasePolygonCenterPixels": changed_outside_erase_pixels,
            "changedOutsideErasePixelEnvelopePixels": changed_outside_erase_pixel_envelope_pixels,
            "transparentPatchPixels": transparent_pixels,
            "fillStrategies": {
                "flatLocalMedianRegions": flat_fill_regions,
                "localMultiscaleDiffusionRegions": diffusion_regions,
            },
            "punctuationOnly": {
                "regions": punctuation_only_regions,
                "erasePixels": punctuation_erase_pixels,
                "coveredErasePixels": punctuation_covered_pixels,
                "coverage": punctuation_covered_pixels as f64 / punctuation_erase_pixels as f64,
            },
            "rasterization": "Polygon-center metrics count pixel centers inside annotation polygons. Pixel-envelope metrics count every pixel touched by an annotation polygon, matching the opaque whole-pixel production patch needed to cover antialiased edges.",
            "knownLimit": "The production text detector does not infer pixel-accurate balloon polygons. Polygon containment is reported diagnostically for annotated balloons; the runtime hard bound is the accepted story-region guard rectangle.",
            "regionEvidence": region_evidence,
        });
        let evidence_path = repository.join(
            ".cache/benchmark-evidence/30-years-since-the-prologue-chapter-5/cleanup-patch-correctness.json",
        );
        fs::create_dir_all(evidence_path.parent().expect("evidence parent"))
            .expect("create evidence directory");
        fs::write(
            &evidence_path,
            serde_json::to_vec_pretty(&evidence).expect("serialize evidence"),
        )
        .expect("write cleanup-patch evidence");
        eprintln!("cleanup patch evidence: {}", evidence_path.display());
    }

    fn normalized_polygon(value: &Value, width: u32, height: u32) -> Vec<(f32, f32)> {
        value
            .as_array()
            .expect("polygon array")
            .iter()
            .map(|point| {
                let point = point.as_array().expect("point array");
                (
                    point[0].as_f64().expect("normalized x") as f32 * width as f32,
                    point[1].as_f64().expect("normalized y") as f32 * height as f32,
                )
            })
            .collect()
    }

    fn polygon_bounds(polygon: &[(f32, f32)]) -> PixelRect {
        let x0 = polygon
            .iter()
            .map(|point| point.0)
            .fold(f32::INFINITY, f32::min);
        let y0 = polygon
            .iter()
            .map(|point| point.1)
            .fold(f32::INFINITY, f32::min);
        let x1 = polygon
            .iter()
            .map(|point| point.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let y1 = polygon
            .iter()
            .map(|point| point.1)
            .fold(f32::NEG_INFINITY, f32::max);
        PixelRect::new(x0, y0, x1, y1).expect("valid polygon bounds")
    }

    fn polygon_pixel_count(polygon: &[(f32, f32)]) -> u64 {
        let bounds = polygon_bounds(polygon);
        let mut count = 0_u64;
        for y in bounds.y0.floor() as u32..bounds.y1.ceil() as u32 {
            for x in bounds.x0.floor() as u32..bounds.x1.ceil() as u32 {
                if point_in_polygon((x as f32 + 0.5, y as f32 + 0.5), polygon) {
                    count += 1;
                }
            }
        }
        count
    }

    #[derive(Clone, Copy)]
    struct RasterBounds {
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    }

    impl RasterBounds {
        fn area(self) -> u64 {
            u64::from(self.x1 - self.x0) * u64::from(self.y1 - self.y0)
        }

        fn contains(self, x: u32, y: u32) -> bool {
            x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
        }
    }

    fn polygon_pixel_envelope(polygon: &[(f32, f32)]) -> RasterBounds {
        // Every committed chapter 5 erase polygon is an axis-aligned
        // rectangle. Fail loudly if the fixture evolves rather than silently
        // treating a non-rectangular polygon as its bounding box.
        let bounds = polygon_bounds(polygon);
        assert!(polygon.iter().all(|point| {
            (point.0 - bounds.x0).abs() < 0.01 || (point.0 - bounds.x1).abs() < 0.01
        }));
        assert!(polygon.iter().all(|point| {
            (point.1 - bounds.y0).abs() < 0.01 || (point.1 - bounds.y1).abs() < 0.01
        }));
        RasterBounds {
            x0: bounds.x0.floor() as u32,
            y0: bounds.y0.floor() as u32,
            x1: bounds.x1.ceil() as u32,
            y1: bounds.y1.ceil() as u32,
        }
    }

    fn point_in_polygon(point: (f32, f32), polygon: &[(f32, f32)]) -> bool {
        let mut inside = false;
        let mut previous = polygon.len() - 1;
        for current in 0..polygon.len() {
            let (x0, y0) = polygon[current];
            let (x1, y1) = polygon[previous];
            if (y0 > point.1) != (y1 > point.1)
                && point.0 < (x1 - x0) * (point.1 - y0) / (y1 - y0) + x0
            {
                inside = !inside;
            }
            previous = current;
        }
        inside
    }

    fn find_repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .find(|candidate| {
                candidate
                    .join("fixtures/benchmarks/30-years-since-the-prologue-chapter-5/manifest.json")
                    .is_file()
            })
            .expect("find repository root from harness manifest")
            .to_path_buf()
    }
}
