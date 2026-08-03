//! Browser-facing adapter for the shared OCR evidence resolver.
//!
//! The resident recognizer and browser pipeline share this evidence gate.
//! Line geometry and model evidence stay in one place so every OCR backend
//! makes the same fail-closed consensus decision.

use koharu_ml::ocr::{
    OcrConsensusConfig, OcrDecision, OcrHypothesis, OcrRegion, TextRect, select_consensus,
};

use super::ppocr::CropBounds;

const NORMALIZED_CONTAINER: TextRect = TextRect {
    left: 0.0,
    top: 0.0,
    right: 1.0,
    bottom: 1.0,
};

/// The browser recognizer and the page-adjudication stage share this
/// confidence floor.  A line that clears the floor has calibrated evidence
/// and can enter the visible chapter stream immediately; a line below it must
/// receive a genuinely different OCR view before it can be used.  Keeping the
/// value here prevents the hot path and the consensus resolver from silently
/// drifting apart.
pub(super) const BROWSER_OCR_MIN_CONFIDENCE: f32 = 0.55;

/// Select the final transcript for two independently preprocessed views of a
/// detector line.  A disagreement is deliberately returned as `None`; the
/// caller can keep the original pixels and surface a retry/hover state rather
/// than publishing the more plausible-looking mistake.
pub(super) fn resolve_line_hypotheses(
    primary: (&str, f32, CropBounds),
    alternate: (&str, f32, CropBounds),
    crop_width: u32,
    crop_height: u32,
) -> Option<(String, f32, CropBounds)> {
    let hypotheses = [
        hypothesis(primary, "primary", crop_width, crop_height),
        hypothesis(alternate, "alternate", crop_width, crop_height),
    ];
    let decision = select_consensus(
        OcrRegion::new(NORMALIZED_CONTAINER, 0.0),
        &hypotheses,
        OcrConsensusConfig {
            minimum_confidence: BROWSER_OCR_MIN_CONFIDENCE,
            minimum_similarity: 0.72,
            minimum_agreement: 0.75,
            minimum_independent_sources: 2,
        },
    );
    let OcrDecision::Accepted(consensus) = decision else {
        return None;
    };
    let bounds = if consensus.selected_source == "primary" {
        primary.2
    } else {
        alternate.2
    };
    Some((consensus.text, consensus.confidence, bounds))
}

fn hypothesis(
    value: (&str, f32, CropBounds),
    source: &str,
    crop_width: u32,
    crop_height: u32,
) -> OcrHypothesis {
    let (_, confidence, bounds) = value;
    OcrHypothesis::new(
        value.0,
        confidence,
        source,
        vec![TextRect::new(
            bounds.left as f32 / crop_width.max(1) as f32,
            bounds.top as f32 / crop_height.max(1) as f32,
            bounds.right as f32 / crop_width.max(1) as f32,
            bounds.bottom as f32 / crop_height.max(1) as f32,
        )],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(top: u32, bottom: u32) -> CropBounds {
        CropBounds {
            left: 0,
            top,
            right: 100,
            bottom,
        }
    }

    #[test]
    fn matching_views_choose_the_high_confidence_transcript_and_geometry() {
        let selected = resolve_line_hypotheses(
            ("THE PORTAL", 0.81, bounds(10, 30)),
            ("THE PORTAL", 0.94, bounds(12, 32)),
            100,
            100,
        )
        .unwrap();
        assert_eq!(selected.0, "THE PORTAL");
        assert_eq!(selected.1, 0.94);
        assert_eq!(selected.2.top, 12);
    }

    #[test]
    fn divergent_views_are_not_published() {
        assert!(
            resolve_line_hypotheses(
                ("THE PORTAL", 0.98, bounds(10, 30)),
                ("THE PORTACONTOT", 0.99, bounds(10, 30)),
                100,
                100,
            )
            .is_none()
        );
    }
}
