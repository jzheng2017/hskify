//! Shared evidence handling for comic OCR.
//!
//! OCR engines produce hypotheses, not truth.  A detector may split a line,
//! a recognizer may see a glyph differently after an image transform, and a
//! single high softmax score is not enough to publish text.  This module keeps
//! the decision about whether a region is safe to use independent of any one
//! model.  Browser and desktop pipelines can feed it hypotheses from their
//! respective detector/recognizer passes and use the same fail-closed rules.

use std::cmp::Ordering;

use anyhow::Result;
use image::DynamicImage;

/// A rectangle in normalized image coordinates.
///
/// Coordinates are clamped at construction time.  Keeping geometry in one
/// coordinate space lets evidence from differently scaled OCR views be
/// compared without converting through a model-specific crop first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextRect {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

/// Detector output shared by all OCR backends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextDetection {
    pub bounds: TextRect,
    pub confidence: f32,
    /// Rotation in radians in the source image.  Keeping this with the
    /// detector evidence lets recognizers rectify vertical/curved text
    /// without guessing from a crop's aspect ratio.
    pub rotation_radians: f32,
}

impl TextDetection {
    pub fn new(bounds: TextRect, confidence: f32, rotation_radians: f32) -> Self {
        Self {
            bounds,
            confidence: if confidence.is_finite() {
                confidence.clamp(0.0, 1.0)
            } else {
                0.0
            },
            rotation_radians: if rotation_radians.is_finite() {
                rotation_radians
            } else {
                0.0
            },
        }
    }
}

/// Model-independent text detector contract.
///
/// Comic bubble/object detectors remain useful for region grouping, but OCR
/// text boxes must come from a detector that can see arbitrary text (including
/// narration, SFX, rotated labels, and text outside bubbles). PP-OCRv6-small
/// adapters implement this contract; the browser pipeline should not need to
/// know which detector produced the boxes.
pub trait TextDetector {
    fn detect_text(&mut self, image: &DynamicImage) -> Result<Vec<TextDetection>>;
}

/// Model-independent text recognizer contract.
///
/// The caller supplies independently prepared line views and receives one
/// hypothesis per view.  A recognizer must report its calibrated confidence;
/// transcript selection is performed by [`select_consensus`].
pub trait TextRecognizer {
    fn recognize_text(&mut self, line_views: &[DynamicImage]) -> Result<Vec<OcrHypothesis>>;
}

impl TextRect {
    pub fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        let mut left = sanitize_coordinate(left);
        let mut top = sanitize_coordinate(top);
        let mut right = sanitize_coordinate(right);
        let mut bottom = sanitize_coordinate(bottom);
        if right < left {
            std::mem::swap(&mut left, &mut right);
        }
        if bottom < top {
            std::mem::swap(&mut top, &mut bottom);
        }
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    pub fn width(self) -> f32 {
        (self.right - self.left).max(0.0)
    }

    pub fn height(self) -> f32 {
        (self.bottom - self.top).max(0.0)
    }

    pub fn area(self) -> f32 {
        self.width() * self.height()
    }

    pub fn intersection(self, other: Self) -> Self {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right.min(other.right);
        let bottom = self.bottom.min(other.bottom);
        if right <= left || bottom <= top {
            // Keep an empty intersection empty.  Calling `new` with reversed
            // coordinates would reorder them and accidentally create area.
            Self {
                left,
                top,
                right: left,
                bottom: top,
            }
        } else {
            Self {
                left,
                top,
                right,
                bottom,
            }
        }
    }

    pub fn intersection_area(self, other: Self) -> f32 {
        self.intersection(other).area()
    }

    pub fn iou(self, other: Self) -> f32 {
        let intersection = self.intersection_area(other);
        if intersection <= 0.0 {
            return 0.0;
        }
        intersection / (self.area() + other.area() - intersection).max(f32::EPSILON)
    }
}

fn sanitize_coordinate(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// One independently produced OCR observation for a region.
///
/// `source` identifies the independent view (for example `native` and
/// `contrast`).  Repeating the same recognizer on the same pixels must use the
/// same source id; it cannot manufacture consensus by itself.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrHypothesis {
    pub text: String,
    pub confidence: f32,
    pub source: String,
    pub line_bounds: Vec<TextRect>,
}

impl OcrHypothesis {
    pub fn new(
        text: impl Into<String>,
        confidence: f32,
        source: impl Into<String>,
        line_bounds: Vec<TextRect>,
    ) -> Self {
        Self {
            text: text.into(),
            confidence: if confidence.is_finite() {
                confidence.clamp(0.0, 1.0)
            } else {
                0.0
            },
            source: source.into(),
            line_bounds,
        }
    }
}

/// Region-level evidence used to validate line coverage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcrRegion {
    /// Detector or bubble bounds that should contain the accepted lines.
    pub container: TextRect,
    /// Minimum fraction of the container covered by recognised line boxes.
    /// This deliberately defaults to a small occupancy value: speech bubbles
    /// often contain generous whitespace.  Callers may raise it for dense
    /// narration boxes or system panels.
    pub minimum_coverage: f32,
}

impl OcrRegion {
    pub fn new(container: TextRect, minimum_coverage: f32) -> Self {
        Self {
            container,
            minimum_coverage: unit_interval(minimum_coverage),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OcrConsensusConfig {
    /// A single view must meet this calibrated recognizer confidence.
    pub minimum_confidence: f32,
    /// Character similarity required for independent views to support one
    /// another.  This compares transcript content only; it does not perform
    /// spell correction or invent punctuation.
    pub minimum_similarity: f32,
    /// Minimum weighted fraction of all confidence supporting the selected
    /// transcript.  This is intentionally distinct from pairwise text
    /// similarity: two agreeing views can be accepted even when a third view
    /// is unrelated.
    pub minimum_agreement: f32,
    /// Minimum number of distinct source ids supporting the selected text.
    /// One source is allowed for fast paths, while callers requiring a second
    /// view can set this to two.
    pub minimum_independent_sources: usize,
}

impl Default for OcrConsensusConfig {
    fn default() -> Self {
        Self {
            minimum_confidence: 0.65,
            minimum_similarity: 0.72,
            minimum_agreement: 0.5,
            minimum_independent_sources: 1,
        }
    }
}

fn unit_interval(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrRejectReason {
    EmptyText,
    LowConfidence,
    InsufficientAgreement,
    InsufficientIndependentSources,
    InsufficientCoverage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OcrConsensus {
    pub text: String,
    pub confidence: f32,
    /// Weighted support from all hypotheses that agree with the selection.
    pub agreement: f32,
    pub coverage: f32,
    pub independent_sources: usize,
    pub selected_source: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OcrDecision {
    Accepted(OcrConsensus),
    Rejected {
        reason: OcrRejectReason,
        best_confidence: f32,
        best_agreement: f32,
        coverage: f32,
        independent_sources: usize,
    },
}

/// Resolve independent OCR views into one publication decision.
///
/// The score is based on weighted agreement and recognizer confidence.  The
/// exact transcript of the highest scoring hypothesis is returned; only
/// whitespace is normalised for comparison.  No lexical cleanup is done here,
/// because that would make a bad recognition look verified.
pub fn select_consensus(
    region: OcrRegion,
    hypotheses: &[OcrHypothesis],
    config: OcrConsensusConfig,
) -> OcrDecision {
    let minimum_confidence = unit_interval(config.minimum_confidence);
    let minimum_similarity = unit_interval(config.minimum_similarity);
    let minimum_agreement = unit_interval(config.minimum_agreement);
    let valid = hypotheses
        .iter()
        .filter(|hypothesis| !canonical_text(&hypothesis.text).is_empty())
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return OcrDecision::Rejected {
            reason: OcrRejectReason::EmptyText,
            best_confidence: 0.0,
            best_agreement: 0.0,
            coverage: 0.0,
            independent_sources: 0,
        };
    }

    let mut best: Option<(usize, f32, f32, usize, f32)> = None;
    for (index, hypothesis) in valid.iter().enumerate() {
        let canonical = canonical_text(&hypothesis.text);
        let mut source_support = Vec::<(&str, f32)>::new();
        for peer in &valid {
            let similarity = text_similarity(&canonical, &canonical_text(&peer.text));
            if similarity >= minimum_similarity {
                if let Some((_, confidence)) = source_support
                    .iter_mut()
                    .find(|(source, _)| *source == peer.source)
                {
                    *confidence = confidence.max(peer.confidence);
                } else {
                    source_support.push((&peer.source, peer.confidence));
                }
            }
        }
        let support = source_support
            .iter()
            .map(|(_, confidence)| *confidence)
            .sum::<f32>();
        // Confidence and agreement are kept separate for diagnostics.  The
        // product prefers a modestly lower-confidence consensus over a lone
        // high-confidence line of unrelated glyph soup.
        let agreement = (support
            / valid
                .iter()
                .map(|peer| peer.confidence)
                .sum::<f32>()
                .max(1e-6))
        .clamp(0.0, 1.0);
        let score = hypothesis.confidence * (0.5 + 0.5 * agreement);
        let coverage = line_coverage(region.container, &hypothesis.line_bounds);
        let candidate = (index, score, agreement, source_support.len(), coverage);
        let is_better = best.as_ref().is_none_or(|previous| {
            candidate.1.total_cmp(&previous.1) == Ordering::Greater
                || (candidate.1.total_cmp(&previous.1) == Ordering::Equal
                    && candidate.2.total_cmp(&previous.2) == Ordering::Greater)
        });
        if is_better {
            best = Some(candidate);
        }
    }

    let (best_index, _score, agreement, independent_sources, coverage) =
        best.expect("valid OCR hypotheses are non-empty");
    let selected = valid[best_index];
    let best_confidence = selected.confidence;
    let rejection = if best_confidence < minimum_confidence {
        Some(OcrRejectReason::LowConfidence)
    } else if agreement < minimum_agreement {
        Some(OcrRejectReason::InsufficientAgreement)
    } else if independent_sources < config.minimum_independent_sources {
        Some(OcrRejectReason::InsufficientIndependentSources)
    } else if coverage < region.minimum_coverage {
        Some(OcrRejectReason::InsufficientCoverage)
    } else {
        None
    };
    if let Some(reason) = rejection {
        OcrDecision::Rejected {
            reason,
            best_confidence,
            best_agreement: agreement,
            coverage,
            independent_sources,
        }
    } else {
        OcrDecision::Accepted(OcrConsensus {
            text: selected.text.clone(),
            confidence: best_confidence,
            agreement,
            coverage,
            independent_sources,
            selected_source: selected.source.clone(),
        })
    }
}

/// Fraction of a detector/container rectangle occupied by the union of line
/// boxes.  Overlapping boxes are counted once, so a duplicate crop cannot
/// inflate coverage.
pub fn line_coverage(container: TextRect, line_bounds: &[TextRect]) -> f32 {
    let container_area = container.area();
    if container_area <= 0.0 || line_bounds.is_empty() {
        return 0.0;
    }
    let clipped = line_bounds
        .iter()
        .map(|bounds| bounds.intersection(container))
        .filter(|bounds| bounds.area() > 0.0)
        .collect::<Vec<_>>();
    if clipped.is_empty() {
        return 0.0;
    }
    union_area(&clipped) / container_area
}

fn union_area(rects: &[TextRect]) -> f32 {
    let mut x_edges = rects
        .iter()
        .flat_map(|rect| [rect.left, rect.right])
        .collect::<Vec<_>>();
    x_edges.sort_by(f32::total_cmp);
    x_edges.dedup_by(|left, right| (*left - *right).abs() < f32::EPSILON);
    let mut area = 0.0;
    for x_pair in x_edges.windows(2) {
        let left = x_pair[0];
        let right = x_pair[1];
        if right <= left {
            continue;
        }
        let mut y_ranges = rects
            .iter()
            .filter(|rect| rect.left < right && rect.right > left)
            .map(|rect| (rect.top, rect.bottom))
            .collect::<Vec<_>>();
        y_ranges.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        });
        let mut covered_y = 0.0;
        let mut current: Option<(f32, f32)> = None;
        for (top, bottom) in y_ranges {
            if let Some((current_top, current_bottom)) = current {
                if top <= current_bottom {
                    current = Some((current_top, current_bottom.max(bottom)));
                } else {
                    covered_y += current_bottom - current_top;
                    current = Some((top, bottom));
                }
            } else {
                current = Some((top, bottom));
            }
        }
        if let Some((top, bottom)) = current {
            covered_y += bottom - top;
        }
        area += (right - left) * covered_y;
    }
    area
}

fn canonical_text(text: &str) -> String {
    let mut canonical = String::new();
    let mut pending_space = false;
    for character in text.chars() {
        if character.is_whitespace() {
            pending_space = !canonical.is_empty();
            continue;
        }
        if pending_space {
            canonical.push(' ');
            pending_space = false;
        }
        canonical.extend(character.to_lowercase());
    }
    canonical
}

fn text_similarity(left: &str, right: &str) -> f32 {
    if left == right {
        return 1.0;
    }
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_character) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_character) in right.iter().enumerate() {
            let replacement =
                previous[right_index] + usize::from(left_character != right_character);
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            current[right_index + 1] = replacement.min(insertion).min(deletion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let distance = previous[right.len()] as f32;
    1.0 - distance / left.len().max(right.len()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(minimum_coverage: f32) -> OcrRegion {
        OcrRegion::new(TextRect::new(0.0, 0.0, 1.0, 1.0), minimum_coverage)
    }

    fn hypothesis(text: &str, confidence: f32, source: &str) -> OcrHypothesis {
        OcrHypothesis::new(
            text,
            confidence,
            source,
            vec![TextRect::new(0.1, 0.2, 0.9, 0.35)],
        )
    }

    #[test]
    fn line_coverage_counts_overlaps_once_and_clips_to_container() {
        let container = TextRect::new(0.0, 0.0, 1.0, 1.0);
        let bounds = [
            TextRect::new(0.1, 0.1, 0.9, 0.3),
            TextRect::new(0.5, 0.2, 1.2, 0.4),
        ];
        // 0.8*0.2 + 0.5*0.1 (the second rectangle's non-overlap) = 0.22.
        assert!((line_coverage(container, &bounds) - 0.22).abs() < 1e-5);
    }

    #[test]
    fn disjoint_line_boxes_do_not_create_intersection_area() {
        let left = TextRect::new(0.0, 0.0, 0.2, 0.2);
        let right = TextRect::new(0.8, 0.8, 1.0, 1.0);
        assert_eq!(left.intersection_area(right), 0.0);
        assert_eq!(line_coverage(left, &[right]), 0.0);
    }

    #[test]
    fn consensus_prefers_two_agreeing_independent_views_over_lone_score() {
        let hypotheses = [
            hypothesis("THE PORTAL", 0.99, "native"),
            hypothesis("THE PORTAL", 0.80, "contrast"),
            hypothesis("THE PORTACONTOT", 0.995, "crop"),
        ];
        let decision = select_consensus(
            region(0.05),
            &hypotheses,
            OcrConsensusConfig {
                minimum_confidence: 0.6,
                minimum_similarity: 0.72,
                minimum_agreement: 0.5,
                minimum_independent_sources: 2,
            },
        );
        let OcrDecision::Accepted(consensus) = decision else {
            panic!("expected consensus acceptance");
        };
        assert_eq!(consensus.text, "THE PORTAL");
        assert_eq!(consensus.independent_sources, 2);
        assert!(consensus.agreement > 0.5);
    }

    #[test]
    fn divergent_views_fail_closed_even_when_each_is_confident() {
        let hypotheses = [
            hypothesis("THE PORTAL", 0.98, "native"),
            hypothesis("REINFORCEMENTS", 0.97, "contrast"),
        ];
        let decision = select_consensus(
            region(0.05),
            &hypotheses,
            OcrConsensusConfig {
                minimum_confidence: 0.6,
                minimum_similarity: 0.85,
                minimum_agreement: 0.85,
                minimum_independent_sources: 2,
            },
        );
        assert!(matches!(
            decision,
            OcrDecision::Rejected {
                reason: OcrRejectReason::InsufficientAgreement,
                ..
            }
        ));
    }

    #[test]
    fn repeated_hypotheses_from_one_view_cannot_fake_independent_support() {
        let hypotheses = [
            hypothesis("THE PORTAL", 0.99, "native"),
            hypothesis("THE PORTAL", 0.98, "native"),
        ];
        let decision = select_consensus(
            region(0.05),
            &hypotheses,
            OcrConsensusConfig {
                minimum_confidence: 0.6,
                minimum_similarity: 0.72,
                minimum_agreement: 0.5,
                minimum_independent_sources: 2,
            },
        );
        assert!(matches!(
            decision,
            OcrDecision::Rejected {
                reason: OcrRejectReason::InsufficientIndependentSources,
                ..
            }
        ));
    }

    #[test]
    fn low_line_coverage_is_rejected_without_erasing_the_source() {
        let mut candidate = hypothesis("ONE LINE", 0.95, "native");
        candidate.line_bounds = vec![TextRect::new(0.1, 0.1, 0.3, 0.12)];
        let decision = select_consensus(
            region(0.2),
            &[candidate],
            OcrConsensusConfig {
                minimum_confidence: 0.6,
                minimum_similarity: 0.5,
                minimum_agreement: 0.5,
                minimum_independent_sources: 1,
            },
        );
        assert!(matches!(
            decision,
            OcrDecision::Rejected {
                reason: OcrRejectReason::InsufficientCoverage,
                ..
            }
        ));
    }

    #[test]
    fn empty_or_whitespace_text_is_rejected() {
        let decision = select_consensus(
            region(0.0),
            &[hypothesis(" \n\t", 1.0, "native")],
            OcrConsensusConfig::default(),
        );
        assert!(matches!(
            decision,
            OcrDecision::Rejected {
                reason: OcrRejectReason::EmptyText,
                ..
            }
        ));
    }

    #[test]
    fn detector_evidence_sanitizes_non_finite_scores_and_rotation() {
        let detection =
            TextDetection::new(TextRect::new(-1.0, 0.1, 2.0, 0.9), f32::NAN, f32::INFINITY);
        assert_eq!(detection.confidence, 0.0);
        assert_eq!(detection.rotation_radians, 0.0);
        assert_eq!(detection.bounds.left, 0.0);
        assert_eq!(detection.bounds.right, 1.0);
    }
}
