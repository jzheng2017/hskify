//! Deterministic cleanup for confirmed speech-bubble dialogue.
//!
//! The browser adapter supplies a segment mask containing only OCR-confirmed
//! English dialogue boxes and a bubble-ID mask covering every erase pixel.
//! This engine replaces those pixels with the median background colour of the
//! matching bubble. It never touches text or artwork outside that mask and
//! does not load a neural inpainting model.

use anyhow::{Result, bail};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView, GrayImage, Rgba};
use koharu_core::{ImageRole, MaskRole, Op};

use crate::pipeline::artifacts::Artifact;
use crate::pipeline::engine::{Engine, EngineCtx, EngineInfo};
use crate::pipeline::engines::support::{
    find_mask_node, image_dimensions, load_source_image, upsert_image_blob,
};

pub struct Model;

#[async_trait]
impl Engine for Model {
    async fn run(&self, ctx: EngineCtx<'_>) -> Result<Vec<Op>> {
        let image = load_source_image(ctx.scene, ctx.page, ctx.blobs)?;
        let (_, segment_ref) = find_mask_node(ctx.scene, ctx.page, MaskRole::Segment)
            .ok_or_else(|| anyhow::anyhow!("no Segment mask on page"))?;
        let (_, bubble_ref) = find_mask_node(ctx.scene, ctx.page, MaskRole::Bubble)
            .ok_or_else(|| anyhow::anyhow!("no Bubble mask on page"))?;
        let segment = ctx.blobs.load_image(&segment_ref)?.to_luma8();
        let bubbles = ctx.blobs.load_image(&bubble_ref)?.to_luma8();
        let cleaned = fill_confirmed_dialogue(&image, &segment, &bubbles)?;
        let (width, height) = image_dimensions(&cleaned);
        let blob = ctx.blobs.put_webp(&cleaned)?;
        Ok(vec![upsert_image_blob(
            ctx.scene,
            ctx.page,
            ImageRole::Inpainted,
            blob,
            width,
            height,
        )])
    }
}

fn fill_confirmed_dialogue(
    image: &DynamicImage,
    segment: &GrayImage,
    bubbles: &GrayImage,
) -> Result<DynamicImage> {
    if image.dimensions() != segment.dimensions() || segment.dimensions() != bubbles.dimensions() {
        bail!(
            "image/mask/bubble dimensions differ: image is {:?}, mask is {:?}, bubble is {:?}",
            image.dimensions(),
            segment.dimensions(),
            bubbles.dimensions()
        );
    }

    // Histograms make the median deterministic and keep memory bounded
    // regardless of webtoon height.
    let mut histograms = vec![[[0_u32; 256]; 3]; 256];
    let mut background_counts = [0_u64; 256];
    let mut erase_counts = [0_u64; 256];
    let rgba = image.to_rgba8();
    for ((pixel, segment_pixel), bubble_pixel) in
        rgba.pixels().zip(segment.pixels()).zip(bubbles.pixels())
    {
        let bubble_id = usize::from(bubble_pixel.0[0]);
        if segment_pixel.0[0] > 0 {
            if bubble_id == 0 {
                bail!("dialogue erase mask contains a pixel outside every speech bubble");
            }
            erase_counts[bubble_id] += 1;
            continue;
        }
        if bubble_id == 0 {
            continue;
        }
        background_counts[bubble_id] += 1;
        for channel in 0..3 {
            histograms[bubble_id][channel][usize::from(pixel.0[channel])] += 1;
        }
    }

    let mut fill_colours = [[0_u8; 3]; 256];
    for bubble_id in 1..256 {
        if erase_counts[bubble_id] == 0 {
            continue;
        }
        if background_counts[bubble_id] == 0 {
            bail!("speech bubble {bubble_id} has no unmasked background sample");
        }
        for channel in 0..3 {
            fill_colours[bubble_id][channel] = histogram_median(
                &histograms[bubble_id][channel],
                background_counts[bubble_id],
            );
        }
    }

    let mut output = rgba;
    for ((pixel, segment_pixel), bubble_pixel) in output
        .pixels_mut()
        .zip(segment.pixels())
        .zip(bubbles.pixels())
    {
        if segment_pixel.0[0] == 0 {
            continue;
        }
        let bubble_id = usize::from(bubble_pixel.0[0]);
        let fill = fill_colours[bubble_id];
        *pixel = Rgba([fill[0], fill[1], fill[2], pixel.0[3]]);
    }
    Ok(DynamicImage::ImageRgba8(output))
}

fn histogram_median(histogram: &[u32; 256], count: u64) -> u8 {
    let target = count.div_ceil(2);
    let mut seen = 0_u64;
    for (value, occurrences) in histogram.iter().enumerate() {
        seen += u64::from(*occurrences);
        if seen >= target {
            return value as u8;
        }
    }
    255
}

inventory::submit! {
    EngineInfo {
        id: "dialogue-bubble-fill",
        name: "Confirmed Dialogue Bubble Fill",
        needs: &[Artifact::SegmentMask, Artifact::BubbleMask],
        produces: &[Artifact::Inpainted],
        load: |_runtime, _cpu| Box::pin(async move {
            Ok(Box::new(Model) as Box<dyn Engine>)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, RgbaImage};

    #[test]
    fn fills_only_confirmed_dialogue_with_its_bubble_background() {
        let mut image = RgbaImage::from_pixel(12, 10, Rgba([240, 241, 242, 255]));
        let mut segment = GrayImage::new(12, 10);
        let mut bubbles = GrayImage::new(12, 10);
        for y in 1..9 {
            for x in 1..9 {
                bubbles.put_pixel(x, y, Luma([3]));
            }
        }
        for y in 4..6 {
            for x in 3..7 {
                image.put_pixel(x, y, Rgba([5, 5, 5, 255]));
                segment.put_pixel(x, y, Luma([255]));
            }
        }
        image.put_pixel(10, 5, Rgba([1, 2, 3, 255]));

        let cleaned = fill_confirmed_dialogue(&DynamicImage::ImageRgba8(image), &segment, &bubbles)
            .unwrap()
            .to_rgba8();

        assert_eq!(cleaned.get_pixel(4, 4).0, [240, 241, 242, 255]);
        assert_eq!(cleaned.get_pixel(10, 5).0, [1, 2, 3, 255]);
    }

    #[test]
    fn rejects_any_erase_pixel_outside_a_speech_bubble() {
        let image = DynamicImage::ImageRgba8(RgbaImage::new(4, 4));
        let mut segment = GrayImage::new(4, 4);
        segment.put_pixel(2, 2, Luma([255]));
        let error = fill_confirmed_dialogue(&image, &segment, &GrayImage::new(4, 4))
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside every speech bubble"));
    }
}
