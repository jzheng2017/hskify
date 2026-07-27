use std::collections::{BTreeMap, VecDeque};
use std::io::Cursor;

use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, ImageFormat, Luma, Rgba, RgbaImage};
use imageproc::{
    contours::{BorderType, find_contours},
    distance_transform::Norm,
    morphology::erode,
};
use koharu_ml::{
    probability_map::ProbabilityMap, speech_bubble_segmentation::SpeechBubbleSegmentationResult,
    types::TextRegion,
};

use crate::contracts::Point;

use super::geometry::{PixelBounds, PixelRect};

const PATCH_EDGE_FEATHER_PIXELS: u32 = 2;
const MAX_POLYGON_POINTS: usize = 64;

#[derive(Debug)]
pub(super) struct PatchPng {
    pub bounds: PixelBounds,
    pub bytes: Vec<u8>,
}

pub(super) fn text_mask_for_regions(
    probabilities: &ProbabilityMap,
    bubbles: &GrayImage,
    regions: &[TextRegion],
    threshold: f32,
) -> GrayImage {
    let mut allowed = GrayImage::new(probabilities.width, probabilities.height);
    for region in regions {
        // Detector boxes commonly stop at the last full glyph and can omit
        // detached punctuation. Give the learned mask one measured glyph of
        // semantic support, then constrain it to the same segmented balloon.
        // This raises recall without turning the detector rectangle into an
        // erase mask or sampling source-image colors.
        let guard = region
            .detected_font_size_px
            .unwrap_or(region.height)
            .max(1.0);
        let text_rect = PixelRect {
            x0: region.x,
            y0: region.y,
            x1: region.x + region.width,
            y1: region.y + region.height,
        };
        let bubble_id = dominant_bubble_id(bubbles, text_rect);
        let x0 = (region.x - guard).floor().max(0.0) as u32;
        let y0 = (region.y - guard).floor().max(0.0) as u32;
        let x1 = (region.x + region.width + guard)
            .ceil()
            .clamp(x0 as f32, probabilities.width as f32) as u32;
        let y1 = (region.y + region.height + guard)
            .ceil()
            .clamp(y0 as f32, probabilities.height as f32) as u32;
        for y in y0..y1 {
            for x in x0..x1 {
                if bubble_id.is_none_or(|id| bubbles.get_pixel(x, y).0[0] == id) {
                    allowed.put_pixel(x, y, Luma([255]));
                }
            }
        }
    }

    GrayImage::from_fn(probabilities.width, probabilities.height, |x, y| {
        let index = y as usize * probabilities.width as usize + x as usize;
        if allowed.get_pixel(x, y).0[0] > 0
            && probabilities.values.get(index).copied().unwrap_or_default() >= threshold
        {
            Luma([255])
        } else {
            Luma([0])
        }
    })
}

pub(super) fn bubble_id_mask(result: &SpeechBubbleSegmentationResult) -> GrayImage {
    let mut mask = GrayImage::new(result.image_width, result.image_height);
    let mut regions = result.regions.iter().collect::<Vec<_>>();
    regions.sort_by_key(|region| std::cmp::Reverse(region.area));
    for (index, region) in regions.into_iter().take(255).enumerate() {
        if region.mask.is_empty() {
            continue;
        }
        let id = (index + 1) as u8;
        let source_width = region.mask.width as usize;
        let max_x = region
            .mask
            .width
            .min(result.image_width.saturating_sub(region.mask.x));
        let max_y = region
            .mask
            .height
            .min(result.image_height.saturating_sub(region.mask.y));
        for local_y in 0..max_y {
            let source_row = local_y as usize * source_width;
            for local_x in 0..max_x {
                if region.mask.pixels[source_row + local_x as usize] > 0 {
                    mask.put_pixel(region.mask.x + local_x, region.mask.y + local_y, Luma([id]));
                }
            }
        }
    }
    mask
}

pub(super) fn merge_binary_mask(
    destination: &mut GrayImage,
    source: &GrayImage,
    offset_x: u32,
    offset_y: u32,
) {
    let copy_width = source
        .width()
        .min(destination.width().saturating_sub(offset_x));
    let copy_height = source
        .height()
        .min(destination.height().saturating_sub(offset_y));
    for y in 0..copy_height {
        for x in 0..copy_width {
            if source.get_pixel(x, y).0[0] > 0 {
                destination.put_pixel(offset_x + x, offset_y + y, Luma([255]));
            }
        }
    }
}

pub(super) fn merge_probability_map(
    destination: &mut ProbabilityMap,
    source: &ProbabilityMap,
    offset_x: u32,
    offset_y: u32,
) {
    let copy_width = source.width.min(destination.width.saturating_sub(offset_x));
    let copy_height = source
        .height
        .min(destination.height.saturating_sub(offset_y));
    for y in 0..copy_height {
        for x in 0..copy_width {
            let source_index = y as usize * source.width as usize + x as usize;
            let destination_index =
                (offset_y + y) as usize * destination.width as usize + (offset_x + x) as usize;
            if let (Some(source_value), Some(destination_value)) = (
                source.values.get(source_index),
                destination.values.get_mut(destination_index),
            ) {
                *destination_value = destination_value.max(*source_value);
            }
        }
    }
}

pub(super) fn crop_probability_map(source: &ProbabilityMap, bounds: PixelBounds) -> ProbabilityMap {
    let width = bounds.width.min(source.width.saturating_sub(bounds.x));
    let height = bounds.height.min(source.height.saturating_sub(bounds.y));
    let mut crop = ProbabilityMap::zeros(width, height);
    for y in 0..height {
        let source_start = (bounds.y + y) as usize * source.width as usize + bounds.x as usize;
        let source_end = source_start + width as usize;
        let destination_start = y as usize * width as usize;
        let destination_end = destination_start + width as usize;
        if let (Some(source_row), Some(destination_row)) = (
            source.values.get(source_start..source_end),
            crop.values.get_mut(destination_start..destination_end),
        ) {
            destination_row.copy_from_slice(source_row);
        }
    }
    crop
}

pub(super) fn label_bubble_components(binary: &GrayImage) -> GrayImage {
    let (width, height) = binary.dimensions();
    let mut labels = GrayImage::new(width, height);
    let mut next_id = 1_u8;
    for seed_y in 0..height {
        for seed_x in 0..width {
            if binary.get_pixel(seed_x, seed_y).0[0] == 0
                || labels.get_pixel(seed_x, seed_y).0[0] != 0
            {
                continue;
            }
            let id = next_id;
            next_id = next_id.saturating_add(1).max(1);
            let mut queue = VecDeque::from([(seed_x, seed_y)]);
            labels.put_pixel(seed_x, seed_y, Luma([id]));
            while let Some((x, y)) = queue.pop_front() {
                for dy in -1_i32..=1 {
                    for dx in -1_i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            continue;
                        }
                        let nx = nx as u32;
                        let ny = ny as u32;
                        if binary.get_pixel(nx, ny).0[0] > 0 && labels.get_pixel(nx, ny).0[0] == 0 {
                            labels.put_pixel(nx, ny, Luma([id]));
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }
        }
    }
    labels
}

pub(super) fn make_inpainted_patch(
    inpainted: &DynamicImage,
    erase_mask: &GrayImage,
    support: PixelRect,
) -> Result<Option<PatchPng>> {
    let Some(bounds) = active_bounds(erase_mask, support) else {
        return Ok(None);
    };
    let image = inpainted.to_rgba8();
    let mut patch = RgbaImage::new(bounds.width, bounds.height);
    let alpha = feathered_alpha(erase_mask);
    for local_y in 0..bounds.height {
        for local_x in 0..bounds.width {
            let x = bounds.x + local_x;
            let y = bounds.y + local_y;
            let opacity = alpha.get_pixel(x, y).0[0];
            if opacity == 0 {
                continue;
            }
            let pixel = image.get_pixel(x, y);
            patch.put_pixel(
                local_x,
                local_y,
                Rgba([pixel.0[0], pixel.0[1], pixel.0[2], opacity]),
            );
        }
    }
    let mut cursor = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(patch)
        .write_to(&mut cursor, ImageFormat::Png)
        .context("encode model-inpainted cleanup patch")?;
    Ok(Some(PatchPng {
        bounds,
        bytes: cursor.into_inner(),
    }))
}

pub(super) fn region_polygons(
    bubbles: &GrayImage,
    text_rect: PixelRect,
    fallback_bubble_rect: PixelRect,
    measured_font_height: f32,
) -> (Vec<Point>, Vec<Point>) {
    let bubble_id = dominant_bubble_id(bubbles, text_rect)
        .or_else(|| dominant_bubble_id(bubbles, fallback_bubble_rect));
    let Some(bubble_id) = bubble_id else {
        let fallback = fallback_bubble_rect.polygon(bubbles.width(), bubbles.height());
        return (
            fallback.clone(),
            text_rect.polygon(bubbles.width(), bubbles.height()),
        );
    };
    let support = fallback_bubble_rect
        .union(text_rect)
        .expand(
            measured_font_height.max(1.0),
            bubbles.width(),
            bubbles.height(),
        )
        .pixel_bounds(bubbles.width(), bubbles.height());
    let binary = GrayImage::from_fn(support.width, support.height, |x, y| {
        if bubbles.get_pixel(support.x + x, support.y + y).0[0] == bubble_id {
            Luma([255])
        } else {
            Luma([0])
        }
    });
    let local_text = PixelRect {
        x0: text_rect.x0 - support.x as f32,
        y0: text_rect.y0 - support.y as f32,
        x1: text_rect.x1 - support.x as f32,
        y1: text_rect.y1 - support.y as f32,
    };
    let bubble_polygon = contour_polygon(
        &binary,
        local_text,
        support.x,
        support.y,
        bubbles.width(),
        bubbles.height(),
    )
    .unwrap_or_else(|| fallback_bubble_rect.polygon(bubbles.width(), bubbles.height()));

    // Clearance follows the measured source glyph height. This preserves a
    // comparable amount of breathing room for small and large balloons
    // without assuming a fixed percentage of the detector box.
    let clearance = (measured_font_height * 0.45).round().clamp(1.0, 255.0) as u8;
    let safe_mask = erode(&binary, Norm::LInf, clearance);
    let safe_polygon = contour_polygon(
        &safe_mask,
        local_text,
        support.x,
        support.y,
        bubbles.width(),
        bubbles.height(),
    )
    .filter(|polygon| !polygon.is_empty())
    .unwrap_or_else(|| text_rect.polygon(bubbles.width(), bubbles.height()));
    (bubble_polygon, safe_polygon)
}

pub(super) fn bubble_id_for_rect(mask: &GrayImage, rect: PixelRect) -> Option<u8> {
    dominant_bubble_id(mask, rect)
}

fn dominant_bubble_id(mask: &GrayImage, rect: PixelRect) -> Option<u8> {
    let bounds = rect.pixel_bounds(mask.width(), mask.height());
    let mut counts = BTreeMap::<u8, usize>::new();
    for y in bounds.y..bounds.y.saturating_add(bounds.height) {
        for x in bounds.x..bounds.x.saturating_add(bounds.width) {
            let id = mask.get_pixel(x, y).0[0];
            if id > 0 {
                *counts.entry(id).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .max_by_key(|(id, count)| (*count, std::cmp::Reverse(*id)))
        .map(|(id, _)| id)
}

fn contour_polygon(
    mask: &GrayImage,
    text_rect: PixelRect,
    offset_x: u32,
    offset_y: u32,
    image_width: u32,
    image_height: u32,
) -> Option<Vec<Point>> {
    let center = text_rect.center();
    let contours = find_contours::<i32>(mask);
    let contour = contours
        .into_iter()
        .filter(|contour| contour.border_type == BorderType::Outer && contour.points.len() >= 3)
        .filter_map(|contour| {
            let bounds = contour.points.iter().fold(
                (i32::MAX, i32::MAX, i32::MIN, i32::MIN),
                |(x0, y0, x1, y1), point| {
                    (
                        x0.min(point.x),
                        y0.min(point.y),
                        x1.max(point.x),
                        y1.max(point.y),
                    )
                },
            );
            let contains_center = center.0 >= bounds.0 as f32
                && center.0 <= bounds.2 as f32
                && center.1 >= bounds.1 as f32
                && center.1 <= bounds.3 as f32;
            let area = i64::from(bounds.2 - bounds.0) * i64::from(bounds.3 - bounds.1);
            contains_center.then_some((area, contour.points))
        })
        .max_by_key(|(area, _)| *area)?
        .1;

    let stride = contour.len().div_ceil(MAX_POLYGON_POINTS).max(1);
    let width = image_width.max(1) as f32;
    let height = image_height.max(1) as f32;
    let points = contour
        .into_iter()
        .step_by(stride)
        .map(|point| Point {
            x: ((offset_x as f32 + point.x as f32) / width).clamp(0.0, 1.0),
            y: ((offset_y as f32 + point.y as f32) / height).clamp(0.0, 1.0),
        })
        .collect::<Vec<_>>();
    (points.len() >= 3).then_some(points)
}

fn active_bounds(mask: &GrayImage, support: PixelRect) -> Option<PixelBounds> {
    let support = support.pixel_bounds(mask.width(), mask.height());
    let mut x0 = mask.width();
    let mut y0 = mask.height();
    let mut x1 = 0;
    let mut y1 = 0;
    for y in support.y..support.y.saturating_add(support.height) {
        for x in support.x..support.x.saturating_add(support.width) {
            if mask.get_pixel(x, y).0[0] == 0 {
                continue;
            }
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let padding = PATCH_EDGE_FEATHER_PIXELS + 1;
    x0 = x0.saturating_sub(padding);
    y0 = y0.saturating_sub(padding);
    x1 = x1.saturating_add(padding).min(mask.width());
    y1 = y1.saturating_add(padding).min(mask.height());
    Some(PixelBounds {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

fn feathered_alpha(mask: &GrayImage) -> GrayImage {
    let (width, height) = mask.dimensions();
    GrayImage::from_fn(width, height, |x, y| {
        if mask.get_pixel(x, y).0[0] == 0 {
            return Luma([0]);
        }
        let mut nearest_edge = PATCH_EDGE_FEATHER_PIXELS + 1;
        for dy in -(PATCH_EDGE_FEATHER_PIXELS as i32)..=PATCH_EDGE_FEATHER_PIXELS as i32 {
            for dx in -(PATCH_EDGE_FEATHER_PIXELS as i32)..=PATCH_EDGE_FEATHER_PIXELS as i32 {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0
                    || ny < 0
                    || nx >= width as i32
                    || ny >= height as i32
                    || mask.get_pixel(nx as u32, ny as u32).0[0] == 0
                {
                    nearest_edge = nearest_edge.min(dx.unsigned_abs().max(dy.unsigned_abs()));
                }
            }
        }
        let alpha = ((nearest_edge.min(PATCH_EDGE_FEATHER_PIXELS + 1) * 255)
            / (PATCH_EDGE_FEATHER_PIXELS + 1)) as u8;
        Luma([alpha])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_mask_uses_measured_punctuation_support_inside_the_same_bubble() {
        let probabilities = ProbabilityMap {
            width: 12,
            height: 8,
            values: vec![1.0; 96],
        };
        let bubbles = GrayImage::from_fn(12, 8, |x, _| if x < 8 { Luma([1]) } else { Luma([2]) });
        let mask = text_mask_for_regions(
            &probabilities,
            &bubbles,
            &[TextRegion {
                x: 3.0,
                y: 3.0,
                width: 3.0,
                height: 2.0,
                detected_font_size_px: Some(2.0),
                ..TextRegion::default()
            }],
            0.1,
        );
        assert_eq!(mask.get_pixel(0, 3).0[0], 0);
        assert_eq!(mask.get_pixel(1, 3).0[0], 255);
        assert_eq!(mask.get_pixel(7, 4).0[0], 255);
        assert_eq!(mask.get_pixel(8, 4).0[0], 0);
    }

    #[test]
    fn overlapping_tile_masks_are_stitched_and_relabelled_by_real_continuity() {
        let mut union = GrayImage::new(12, 8);
        let left = GrayImage::from_fn(8, 8, |x, y| {
            if x >= 5 && (2..6).contains(&y) {
                Luma([1])
            } else {
                Luma([0])
            }
        });
        let right = GrayImage::from_fn(8, 8, |x, y| {
            if x < 3 && (2..6).contains(&y) {
                Luma([7])
            } else {
                Luma([0])
            }
        });
        merge_binary_mask(&mut union, &left, 0, 0);
        merge_binary_mask(&mut union, &right, 5, 0);
        let labels = label_bubble_components(&union);
        assert_eq!(labels.get_pixel(5, 3).0[0], labels.get_pixel(7, 3).0[0]);
        assert_ne!(labels.get_pixel(5, 3).0[0], 0);
    }

    #[test]
    fn inpainted_patch_alpha_follows_the_semantic_mask() {
        let image =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(12, 12, Rgba([90, 100, 110, 255])));
        let mut mask = GrayImage::new(12, 12);
        for y in 4..8 {
            for x in 4..8 {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        let patch =
            make_inpainted_patch(&image, &mask, PixelRect::new(0.0, 0.0, 12.0, 12.0).unwrap())
                .unwrap()
                .unwrap();
        let decoded = image::load_from_memory(&patch.bytes).unwrap().to_rgba8();
        assert_eq!(decoded.get_pixel(0, 0).0[3], 0);
        assert!(decoded.pixels().any(|pixel| pixel.0[3] > 0));
    }
}
