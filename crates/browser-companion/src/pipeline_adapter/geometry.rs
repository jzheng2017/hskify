use crate::contracts::{NormalizedRect, Point, ReadingDirection};

use koharu_ml::comic_text_bubble_detector::{
    ComicTextBubbleDetection, DETECTOR_TILE_CONFIDENCE_THRESHOLD,
};

const TILE_SIDE: u32 = 2_048;
const TILE_OVERLAP: u32 = 410;
const MIN_DETECTOR_SCORE: f32 = DETECTOR_TILE_CONFIDENCE_THRESHOLD;
const NORMALIZED_SERIALIZATION_EDGE_GUARD: f32 = 0.000_000_1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PixelRect {
    pub(super) x0: f32,
    pub(super) y0: f32,
    pub(super) x1: f32,
    pub(super) y1: f32,
}

impl PixelRect {
    pub(super) fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Self> {
        let rect = Self { x0, y0, x1, y1 };
        (x0.is_finite()
            && y0.is_finite()
            && x1.is_finite()
            && y1.is_finite()
            && rect.width() > 0.0
            && rect.height() > 0.0)
            .then_some(rect)
    }

    fn from_local_bounds(bounds: [f32; 4], tile: &Tile) -> Option<Self> {
        Self::new(
            tile.x as f32 + bounds[0],
            tile.y as f32 + bounds[1],
            tile.x as f32 + bounds[2],
            tile.y as f32 + bounds[3],
        )
    }

    pub(super) fn width(self) -> f32 {
        self.x1 - self.x0
    }

    pub(super) fn height(self) -> f32 {
        self.y1 - self.y0
    }

    fn area(self) -> f32 {
        self.width().max(0.0) * self.height().max(0.0)
    }

    pub(super) fn center(self) -> (f32, f32) {
        ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)
    }

    pub(super) fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    pub(super) fn intersection(self, other: Self) -> Option<Self> {
        Self::new(
            self.x0.max(other.x0),
            self.y0.max(other.y0),
            self.x1.min(other.x1),
            self.y1.min(other.y1),
        )
    }

    pub(super) fn expand(self, pixels: f32, image_width: u32, image_height: u32) -> Self {
        Self {
            x0: (self.x0 - pixels).max(0.0),
            y0: (self.y0 - pixels).max(0.0),
            x1: (self.x1 + pixels).min(image_width as f32),
            y1: (self.y1 + pixels).min(image_height as f32),
        }
    }

    pub(super) fn iou(self, other: Self) -> f32 {
        let intersection = self.intersection(other).map(Self::area).unwrap_or_default();
        let union = self.area() + other.area() - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    pub(super) fn overlap_over_smaller(self, other: Self) -> f32 {
        let smaller = self.area().min(other.area());
        if smaller <= 0.0 {
            return 0.0;
        }
        self.intersection(other).map(Self::area).unwrap_or_default() / smaller
    }

    pub(super) fn normalized(self, image_width: u32, image_height: u32) -> NormalizedRect {
        let width = image_width.max(1) as f32;
        let height = image_height.max(1) as f32;
        let x0 = (self.x0 / width).clamp(0.0, 1.0);
        let y0 = (self.y0 / height).clamp(0.0, 1.0);
        let x1 = (self.x1 / width).clamp(x0, 1.0);
        let y1 = (self.y1 / height).clamp(y0, 1.0);
        let normalized_width = if x1 == 1.0 {
            (x1 - x0 - NORMALIZED_SERIALIZATION_EDGE_GUARD).max(0.0)
        } else {
            x1 - x0
        };
        let normalized_height = if y1 == 1.0 {
            (y1 - y0 - NORMALIZED_SERIALIZATION_EDGE_GUARD).max(0.0)
        } else {
            y1 - y0
        };
        NormalizedRect {
            x: x0,
            y: y0,
            width: normalized_width,
            height: normalized_height,
        }
    }

    pub(super) fn polygon(self, image_width: u32, image_height: u32) -> Vec<Point> {
        let rect = self.normalized(image_width, image_height);
        vec![
            Point {
                x: rect.x,
                y: rect.y,
            },
            Point {
                x: rect.x + rect.width,
                y: rect.y,
            },
            Point {
                x: rect.x + rect.width,
                y: rect.y + rect.height,
            },
            Point {
                x: rect.x,
                y: rect.y + rect.height,
            },
        ]
    }

    pub(super) fn pixel_bounds(self, image_width: u32, image_height: u32) -> PixelBounds {
        let x0 = self.x0.floor().clamp(0.0, image_width as f32) as u32;
        let y0 = self.y0.floor().clamp(0.0, image_height as f32) as u32;
        let x1 = self.x1.ceil().clamp(x0 as f32, image_width as f32) as u32;
        let y1 = self.y1.ceil().clamp(y0 as f32, image_height as f32) as u32;
        PixelBounds {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        }
    }

    pub(super) fn intersects_viewport(
        self,
        visible_rects: &[NormalizedRect],
        image_width: u32,
        image_height: u32,
    ) -> bool {
        let own = self.normalized(image_width, image_height);
        visible_rects
            .iter()
            .any(|visible| normalized_rects_intersect(&own, visible))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PixelBounds {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl PixelBounds {
    pub(super) fn normalized(self, image_width: u32, image_height: u32) -> NormalizedRect {
        PixelRect {
            x0: self.x as f32,
            y0: self.y as f32,
            x1: self.x.saturating_add(self.width) as f32,
            y1: self.y.saturating_add(self.height) as f32,
        }
        .normalized(image_width, image_height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Tile {
    pub(super) id: usize,
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    ownership_x0: f32,
    ownership_y0: f32,
    ownership_x1: f32,
    ownership_y1: f32,
}

impl Tile {
    fn owns(self, x: f32, y: f32) -> bool {
        x >= self.ownership_x0
            && y >= self.ownership_y0
            && x < self.ownership_x1
            && y < self.ownership_y1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CandidateKind {
    StoryText,
    FreeText,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Candidate {
    pub(super) kind: CandidateKind,
    pub(super) text_rect: PixelRect,
    pub(super) bubble_rect: PixelRect,
    pub(super) confirmed_bubble_rect: PixelRect,
    pub(super) detector_confidence: f32,
}

pub(super) fn overlapping_tiles(image_width: u32, image_height: u32) -> Vec<Tile> {
    if image_width == 0 || image_height == 0 {
        return Vec::new();
    }
    let x_starts = tile_starts(image_width);
    let y_starts = tile_starts(image_height);
    let mut tiles = Vec::with_capacity(x_starts.len() * y_starts.len());
    for (row, &y) in y_starts.iter().enumerate() {
        for (column, &x) in x_starts.iter().enumerate() {
            let width = TILE_SIDE.min(image_width - x);
            let height = TILE_SIDE.min(image_height - y);
            let (ownership_x0, ownership_x1) = ownership_axis(&x_starts, column, image_width);
            let (ownership_y0, ownership_y1) = ownership_axis(&y_starts, row, image_height);
            tiles.push(Tile {
                id: row * x_starts.len() + column,
                x,
                y,
                width,
                height,
                ownership_x0,
                ownership_y0,
                ownership_x1,
                ownership_y1,
            });
        }
    }
    tiles
}

fn tile_starts(extent: u32) -> Vec<u32> {
    if extent <= TILE_SIDE {
        return vec![0];
    }
    let step = TILE_SIDE - TILE_OVERLAP;
    let last = extent - TILE_SIDE;
    let mut starts = (0..)
        .map(|index| index * step)
        .take_while(|start| *start < last)
        .collect::<Vec<_>>();
    starts.push(last);
    starts
}

fn ownership_axis(starts: &[u32], index: usize, extent: u32) -> (f32, f32) {
    let start = starts[index] as f32;
    let end = (starts[index] + TILE_SIDE).min(extent) as f32;
    let lower = if index == 0 {
        0.0
    } else {
        ((starts[index - 1] + TILE_SIDE).min(extent) as f32 + start) * 0.5
    };
    let upper = if index + 1 == starts.len() {
        extent as f32
    } else {
        (end + starts[index + 1] as f32) * 0.5
    };
    (lower, upper)
}

pub(super) fn prioritize_tiles(
    tiles: &mut [Tile],
    visible_rects: &[NormalizedRect],
    viewport_active: bool,
    image_width: u32,
    image_height: u32,
    reading_direction: ReadingDirection,
) {
    tiles.sort_by_key(|tile| {
        let rect = PixelRect {
            x0: tile.x as f32,
            y0: tile.y as f32,
            x1: (tile.x + tile.width) as f32,
            y1: (tile.y + tile.height) as f32,
        };
        let visible =
            viewport_active && rect.intersects_viewport(visible_rects, image_width, image_height);
        (
            !visible,
            reading_order_key(rect, image_width, image_height, reading_direction),
            tile.id,
        )
    });
}

pub(super) fn candidates_for_tile(
    detection: &ComicTextBubbleDetection,
    tile: &Tile,
    image_width: u32,
    image_height: u32,
) -> Vec<Candidate> {
    let bubbles = detection
        .detections
        .iter()
        .filter(|detection| detection.is_bubble())
        .filter_map(|detection| PixelRect::from_local_bounds(detection.bbox, tile))
        .collect::<Vec<_>>();
    let lines = detection
        .detections
        .iter()
        .filter(|detection| detection.is_text())
        .filter(|detection| detection.score.is_finite() && detection.score >= MIN_DETECTOR_SCORE)
        .filter_map(|detection| {
            let rect = PixelRect::from_local_bounds(detection.bbox, tile)?;
            let (center_x, center_y) = rect.center();
            let containing_bubble = bubbles
                .iter()
                .copied()
                .filter(|bubble| {
                    (center_x >= bubble.x0
                        && center_x <= bubble.x1
                        && center_y >= bubble.y0
                        && center_y <= bubble.y1)
                        || rect.overlap_over_smaller(*bubble) >= 0.10
                })
                .min_by(|left, right| {
                    (left.width() * left.height()).total_cmp(&(right.width() * right.height()))
                });
            let kind = if detection.label_id == 1 || containing_bubble.is_some() {
                CandidateKind::StoryText
            } else {
                CandidateKind::FreeText
            };
            tile.owns(center_x, center_y).then_some((
                rect,
                detection.score,
                kind,
                containing_bubble,
            ))
        })
        .collect::<Vec<_>>();
    lines
        .into_iter()
        .map(
            |(text_rect, detector_confidence, kind, containing_bubble)| {
                let layout_padding =
                    (text_rect.height().min(text_rect.width()) * 0.22).clamp(6.0, 28.0);
                let layout_rect = text_rect.expand(layout_padding, image_width, image_height);
                let confirmed_bubble_rect = containing_bubble.unwrap_or(layout_rect);
                Candidate {
                    kind,
                    text_rect,
                    bubble_rect: confirmed_bubble_rect.union(layout_rect),
                    confirmed_bubble_rect,
                    detector_confidence,
                }
            },
        )
        .collect()
}

pub(super) fn spatially_dedupe(
    mut candidates: Vec<Candidate>,
    seen_text_blocks: &[PixelRect],
) -> Vec<Candidate> {
    candidates.sort_by(|left, right| {
        candidate_kind_priority(right.kind)
            .cmp(&candidate_kind_priority(left.kind))
            .then_with(|| {
                right
                    .detector_confidence
                    .total_cmp(&left.detector_confidence)
            })
            .then_with(|| left.text_rect.y0.total_cmp(&right.text_rect.y0))
    });
    let mut accepted = Vec::<Candidate>::new();
    for candidate in candidates {
        let duplicate = seen_text_blocks
            .iter()
            .chain(accepted.iter().map(|item| &item.text_rect))
            .any(|existing| {
                candidate.text_rect.iou(*existing) >= 0.35
                    || candidate.text_rect.overlap_over_smaller(*existing) >= 0.72
            });
        if !duplicate {
            accepted.push(candidate);
        }
    }
    accepted
}

fn candidate_kind_priority(kind: CandidateKind) -> u8 {
    match kind {
        CandidateKind::StoryText => 1,
        CandidateKind::FreeText => 0,
    }
}

pub(super) fn text_candidate_is_confirmed(candidate: &Candidate) -> bool {
    candidate.detector_confidence.is_finite()
        && candidate.detector_confidence >= MIN_DETECTOR_SCORE
        && candidate.text_rect.width() >= 5.0
        && candidate.text_rect.height() >= 5.0
}

pub(super) fn ocr_crop_rect(
    candidate: &Candidate,
    image_width: u32,
    image_height: u32,
) -> PixelRect {
    candidate.text_rect.expand(3.0, image_width, image_height)
}

pub(super) fn reading_order_key(
    rect: PixelRect,
    image_width: u32,
    _image_height: u32,
    reading_direction: ReadingDirection,
) -> u32 {
    let (center_x, center_y) = rect.center();
    let y = center_y.max(0.0).round() as u64;
    let x = match reading_direction {
        ReadingDirection::Rtl => image_width as f32 - center_x,
        ReadingDirection::Auto | ReadingDirection::Ltr => center_x,
    }
    .max(0.0)
    .round() as u64;
    y.saturating_mul(image_width.max(1) as u64)
        .saturating_add(x)
        .min(u32::MAX as u64) as u32
}

fn normalized_rects_intersect(left: &NormalizedRect, right: &NormalizedRect) -> bool {
    left.x < right.x + right.width
        && left.x + left.width > right.x
        && left.y < right.y + right.height
        && left.y + left.height > right.y
}

#[cfg(test)]
mod tests {
    use super::*;
    use koharu_ml::comic_text_bubble_detector::ComicTextBubbleRegion;

    fn comic_region(label_id: usize, bbox: [f32; 4], score: f32) -> ComicTextBubbleRegion {
        ComicTextBubbleRegion {
            label_id,
            label: match label_id {
                0 => "bubble",
                1 => "text_bubble",
                2 => "text_free",
                _ => "unknown",
            }
            .to_owned(),
            score,
            bbox,
        }
    }

    fn comic_detection(detections: Vec<ComicTextBubbleRegion>) -> ComicTextBubbleDetection {
        ComicTextBubbleDetection {
            image_width: TILE_SIDE,
            image_height: TILE_SIDE,
            detections,
            text_blocks: Vec::new(),
        }
    }

    #[test]
    fn long_image_uses_fixed_overlapping_square_tiles() {
        let tiles = overlapping_tiles(900, 16_000);
        assert_eq!(tiles.len(), 10);
        assert!(tiles.iter().all(|tile| tile.width == 900));
        assert!(tiles.iter().all(|tile| tile.height == TILE_SIDE));
        assert_eq!(tiles.first().unwrap().y, 0);
        assert_eq!(tiles.last().unwrap().y + TILE_SIDE, 16_000);
    }

    #[test]
    fn viewport_tiles_overtake_offscreen_tiles() {
        let mut tiles = overlapping_tiles(900, 4_000);
        let viewport = [NormalizedRect {
            x: 0.0,
            y: 0.60,
            width: 1.0,
            height: 0.10,
        }];
        prioritize_tiles(
            &mut tiles,
            &viewport,
            true,
            900,
            4_000,
            ReadingDirection::Ltr,
        );
        assert!(tiles[0].y > 1_000);
    }

    #[test]
    fn overlap_ownership_emits_a_line_once() {
        let tiles = overlapping_tiles(900, 2_000);
        let global_y = 900.0;
        let emitted = tiles
            .iter()
            .filter(|tile| {
                let local_y = global_y - tile.y as f32;
                let detection = comic_detection(vec![comic_region(
                    1,
                    [100.0, local_y, 300.0, local_y + 40.0],
                    0.95,
                )]);
                !candidates_for_tile(&detection, tile, 900, 2_000).is_empty()
            })
            .count();
        assert_eq!(emitted, 1);
    }

    #[test]
    fn detector_lines_keep_their_shared_confirmed_bubble() {
        let tile = overlapping_tiles(900, 1_024)[0];
        let bubble = PixelRect::new(50.0, 50.0, 350.0, 220.0).unwrap();
        let detection = comic_detection(vec![
            comic_region(1, [100.0, 100.0, 300.0, 130.0], 0.9),
            comic_region(2, [120.0, 140.0, 280.0, 170.0], 0.8),
            comic_region(0, [50.0, 50.0, 350.0, 220.0], 0.99),
        ]);
        let candidates = candidates_for_tile(&detection, &tile, 900, 1_024);
        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.confirmed_bubble_rect == bubble)
        );
    }

    #[test]
    fn ocr_crop_uses_only_a_small_fixed_guard() {
        let candidate = Candidate {
            kind: CandidateKind::StoryText,
            text_rect: PixelRect::new(100.0, 200.0, 700.0, 500.0).unwrap(),
            bubble_rect: PixelRect::new(80.0, 180.0, 720.0, 520.0).unwrap(),
            confirmed_bubble_rect: PixelRect::new(80.0, 180.0, 720.0, 520.0).unwrap(),
            detector_confidence: 0.99,
        };

        assert_eq!(
            ocr_crop_rect(&candidate, 900, 1_000),
            PixelRect::new(97.0, 197.0, 703.0, 503.0).unwrap()
        );
    }

    #[test]
    fn normalized_pixel_bounds_never_round_past_the_image_edge() {
        let rect = PixelBounds {
            x: 700,
            y: 15_900,
            width: 200,
            height: 100,
        }
        .normalized(900, 16_000);

        assert!(rect.x + rect.width <= 1.0);
        assert!(rect.y + rect.height <= 1.0);
        let json = serde_json::to_value(rect).unwrap();
        assert!(json["x"].as_f64().unwrap() + json["width"].as_f64().unwrap() <= 1.0);
        assert!(json["y"].as_f64().unwrap() + json["height"].as_f64().unwrap() <= 1.0);
    }

    #[test]
    fn reading_order_is_top_to_bottom_then_directional() {
        let left = PixelRect::new(100.0, 100.0, 200.0, 140.0).unwrap();
        let right = PixelRect::new(600.0, 100.0, 700.0, 140.0).unwrap();
        assert!(
            reading_order_key(left, 900, 2_000, ReadingDirection::Ltr)
                < reading_order_key(right, 900, 2_000, ReadingDirection::Ltr)
        );
        assert!(
            reading_order_key(right, 900, 2_000, ReadingDirection::Rtl)
                < reading_order_key(left, 900, 2_000, ReadingDirection::Rtl)
        );
    }
}
