//! Comic Text & Bubble Detector (ogkalu RT-DETR). Emits text nodes and a
//! conservative bubble-ID mask from the same sliced detector pass.

use anyhow::Result;
use async_trait::async_trait;
use image::{DynamicImage, GrayImage, Luma};
use koharu_core::{MaskRole, Op, TextData};
use koharu_ml::comic_text_bubble_detector::{
    ComicTextBubbleDetection, ComicTextBubbleDetector, ComicTextBubbleRegion,
};

use crate::pipeline::artifacts::Artifact;
use crate::pipeline::engine::{Engine, EngineCtx, EngineInfo};
use crate::pipeline::engines::support::{
    clear_text_nodes_ops, load_source_image, new_text_node, page_node_count,
    sort_manga_reading_order, text_region_to_pair, upsert_mask_blob,
};

use std::thread;
use tokio::runtime::Builder;
use tokio::sync::{mpsc, oneshot};

const DETECTOR_NAME: &str = "comic-text-bubble-detector";

// 1. Define the communication protocol
struct DetectMessage {
    image: image::DynamicImage,
    respond_to: oneshot::Sender<Result<ComicTextBubbleDetection>>,
}

// 2. The Engine now acts as an Async Client to the dedicated thread
pub struct Model {
    sender: mpsc::Sender<DetectMessage>,
}

#[async_trait]
impl Engine for Model {
    async fn run(&self, ctx: EngineCtx<'_>) -> Result<Vec<Op>> {
        let image = load_source_image(ctx.scene, ctx.page, ctx.blobs)?;

        // Create a one-time return channel
        let (resp_tx, resp_rx) = oneshot::channel();

        // Send the image to the dedicated thread
        self.sender
            .send(DetectMessage {
                image,
                respond_to: resp_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("[SYS] Detector thread disconnected"))?;

        // Wait asynchronously without blocking Tokio workers
        let det = resp_rx
            .await
            .map_err(|_| anyhow::anyhow!("[SYS] Detector thread crashed"))??;

        let bubble_mask = paint_bubble_id_mask(det.image_width, det.image_height, &det.detections);
        let bubble_blob = ctx.blobs.put_webp(&DynamicImage::ImageLuma8(bubble_mask))?;

        let mut pairs: Vec<([f32; 4], TextData)> = det
            .text_blocks
            .into_iter()
            .map(|r| text_region_to_pair(r, DETECTOR_NAME))
            .collect();
        sort_manga_reading_order(&mut pairs, ctx.options.reading_order.unwrap_or_default());

        let mut ops = clear_text_nodes_ops(ctx.scene, ctx.page);
        let removed = ops.len();
        let mut running_len = page_node_count(ctx.scene, ctx.page).saturating_sub(removed);
        let mask_op = upsert_mask_blob(ctx.scene, ctx.page, MaskRole::Bubble, bubble_blob);
        if matches!(mask_op, Op::AddNode { .. }) {
            running_len += 1;
        }
        ops.push(mask_op);
        ops.reserve(pairs.len());
        for (bbox, text) in pairs {
            let node = new_text_node(bbox, text);
            ops.push(Op::AddNode {
                page: ctx.page,
                node,
                at: running_len,
            });
            running_len += 1;
        }
        Ok(ops)
    }
}

fn paint_bubble_id_mask(
    width: u32,
    height: u32,
    detections: &[ComicTextBubbleRegion],
) -> GrayImage {
    let mut mask = GrayImage::from_pixel(width, height, Luma([0]));
    let mut bubbles = detections
        .iter()
        .filter(|region| region.is_bubble())
        .collect::<Vec<_>>();
    bubbles.sort_by(|left, right| {
        bubble_area(right)
            .partial_cmp(&bubble_area(left))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // The detector returns bubble bounds rather than contours. An inset
    // ellipse is deliberately conservative: it includes central dialogue but
    // excludes panel art around the corners and the balloon outline.
    for (index, bubble) in bubbles.into_iter().take(255).enumerate() {
        let x0 = bubble.bbox[0].floor().max(0.0).min(width as f32) as u32;
        let y0 = bubble.bbox[1].floor().max(0.0).min(height as f32) as u32;
        let x1 = bubble.bbox[2].ceil().max(0.0).min(width as f32) as u32;
        let y1 = bubble.bbox[3].ceil().max(0.0).min(height as f32) as u32;
        if x1 <= x0 + 1 || y1 <= y0 + 1 {
            continue;
        }
        let center_x = (x0 as f32 + x1 as f32) * 0.5;
        let center_y = (y0 as f32 + y1 as f32) * 0.5;
        let radius_x = (x1 - x0) as f32 * 0.48;
        let radius_y = (y1 - y0) as f32 * 0.48;
        let id = (index + 1) as u8;
        for y in y0..y1 {
            let dy = (y as f32 + 0.5 - center_y) / radius_y;
            for x in x0..x1 {
                let dx = (x as f32 + 0.5 - center_x) / radius_x;
                if dx * dx + dy * dy <= 1.0 {
                    mask.put_pixel(x, y, Luma([id]));
                }
            }
        }
    }
    mask
}

fn bubble_area(region: &ComicTextBubbleRegion) -> f32 {
    (region.bbox[2] - region.bbox[0]).max(0.0) * (region.bbox[3] - region.bbox[1]).max(0.0)
}

// 3. Spawning the isolated OS Thread during Engine Load
inventory::submit! {
    EngineInfo {
        id: "comic-text-bubble-detector",
        name: "Comic Text & Bubble Detector",
        needs: &[],
        produces: &[Artifact::TextBoxes, Artifact::BubbleMask],
        load: |runtime, cpu| Box::pin(async move {
            // A detector instance is intentionally single-flight. Buffering
            // full webtoon images here only increases memory pressure.
            let (tx, mut rx) = mpsc::channel::<DetectMessage>(1);
            let runtime_clone = runtime.clone(); // Clone Arc for the thread

            thread::spawn(move || {
                // Initialize an isolated single-threaded runtime strictly for this OS thread
                let rt = Builder::new_current_thread().enable_all().build().unwrap();
                rt.block_on(async move {

                    // The CUDA context is now permanently tied to this specific thread
                    let detector = match ComicTextBubbleDetector::load(&runtime_clone, cpu).await {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::error!("Failed to load detector: {:?}", e);
                            return;
                        }
                    };

                    // Listen continuously for pipeline requests
                    while let Some(msg) = rx.recv().await {
                        let result = detector.inference(&msg.image);
                        let _ = msg.respond_to.send(result);
                    }
                });
            });

            Ok(Box::new(Model { sender: tx }) as Box<dyn Engine>)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(label_id: usize, bbox: [f32; 4]) -> ComicTextBubbleRegion {
        ComicTextBubbleRegion {
            label_id,
            label: if label_id == 0 {
                "bubble"
            } else {
                "text_bubble"
            }
            .to_owned(),
            score: 0.9,
            bbox,
        }
    }

    #[test]
    fn bubble_mask_uses_distinct_ellipses_and_ignores_text() {
        let detections = vec![
            region(0, [2.0, 2.0, 18.0, 14.0]),
            region(0, [21.0, 3.0, 29.0, 11.0]),
            region(1, [5.0, 5.0, 10.0, 9.0]),
        ];

        let mask = paint_bubble_id_mask(32, 16, &detections);

        assert_ne!(mask.get_pixel(10, 8).0[0], 0);
        assert_ne!(mask.get_pixel(25, 7).0[0], 0);
        assert_ne!(mask.get_pixel(10, 8).0[0], mask.get_pixel(25, 7).0[0]);
        assert_eq!(mask.get_pixel(2, 2).0[0], 0);
        assert_eq!(mask.get_pixel(6, 6).0[0], mask.get_pixel(10, 8).0[0]);
    }
}
