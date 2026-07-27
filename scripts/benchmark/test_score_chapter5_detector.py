#!/usr/bin/env python3

from __future__ import annotations

import unittest

from PIL import Image

from score_chapter5_detector import (
    BenchmarkInputError,
    DETECTOR_GOLD_KINDS,
    SUPPORTED_REGION_KINDS,
    postprocess_detections,
    spatially_dedupe_candidates,
    validate_fixture,
)


def detection(
    index: int, label_id: int, score: float, bbox: list[float]
) -> dict[str, object]:
    return {
        "index": index,
        "labelId": label_id,
        "score": score,
        "bbox": bbox,
    }


class DetectorPostprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dialogue_page = Image.new("RGB", (100, 100), (220, 220, 220))

    def test_connected_bubble_splits_into_two_candidates(self) -> None:
        candidates, _ = postprocess_detections(
            [
                detection(0, 0, 0.95, [0.0, 0.0, 100.0, 100.0]),
                detection(1, 1, 0.90, [20.0, 10.0, 60.0, 30.0]),
                detection(2, 1, 0.92, [20.0, 70.0, 60.0, 90.0]),
            ],
            self.dialogue_page,
        )

        self.assertEqual(
            [candidate["candidateBounds"] for candidate in candidates],
            [[0.0, 0.0, 100.0, 50.0], [0.0, 50.0, 100.0, 100.0]],
        )

    def test_contained_text_detection_is_suppressed(self) -> None:
        candidates, stats = postprocess_detections(
            [
                detection(0, 0, 0.95, [0.0, 0.0, 100.0, 100.0]),
                detection(1, 1, 0.92, [20.0, 20.0, 80.0, 80.0]),
                detection(2, 1, 0.88, [30.0, 30.0, 70.0, 70.0]),
            ],
            self.dialogue_page,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(stats["withinBubbleDuplicateRejectedCount"], 1)

    def test_dark_caption_card_is_rejected(self) -> None:
        candidates, stats = postprocess_detections(
            [
                detection(0, 0, 0.95, [0.0, 0.0, 100.0, 100.0]),
                detection(1, 1, 0.92, [20.0, 20.0, 80.0, 80.0]),
            ],
            Image.new("RGB", (100, 100), (0, 0, 0)),
        )

        self.assertEqual(candidates, [])
        self.assertEqual(stats["darkCardRejectedCount"], 1)

    def test_low_joint_confidence_is_rejected(self) -> None:
        candidates, stats = postprocess_detections(
            [
                detection(0, 0, 0.95, [0.0, 0.0, 100.0, 100.0]),
                detection(1, 1, 0.39, [20.0, 20.0, 80.0, 80.0]),
            ],
            self.dialogue_page,
        )

        self.assertEqual(candidates, [])
        self.assertEqual(stats["lowConfidenceRejectedCount"], 1)

    def test_tile_dedupe_uses_text_geometry(self) -> None:
        first = {
            "jointConfidence": 0.91,
            "textBounds": [20.0, 20.0, 80.0, 60.0],
            "candidateBounds": [10.0, 10.0, 100.0, 80.0],
        }
        duplicate_from_another_tile = {
            "jointConfidence": 0.74,
            "textBounds": [22.0, 21.0, 82.0, 61.0],
            "candidateBounds": [0.0, 0.0, 110.0, 90.0],
        }
        connected_sibling = {
            "jointConfidence": 0.80,
            "textBounds": [20.0, 70.0, 80.0, 95.0],
            "candidateBounds": [10.0, 10.0, 100.0, 100.0],
        }

        accepted, rejected = spatially_dedupe_candidates(
            [duplicate_from_another_tile, connected_sibling, first]
        )

        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected, 1)
        self.assertIn(connected_sibling, accepted)


class FixtureValidationTests(unittest.TestCase):
    def test_narration_is_supported_but_not_detector_gold(self) -> None:
        self.assertIn("narration", SUPPORTED_REGION_KINDS)
        self.assertNotIn("narration", DETECTOR_GOLD_KINDS)

    def test_structural_fixture_loads_all_canonical_regions(self) -> None:
        manifest, pages, evidence = validate_fixture()

        self.assertEqual(len(pages), 36)
        self.assertEqual(evidence["regionCount"], 218)
        self.assertEqual(evidence["goldBubbleCount"], 165)
        self.assertEqual(evidence["narrationRegionCount"], 53)
        self.assertEqual(evidence["englishTranslationTargetCount"], 214)
        self.assertEqual(evidence["punctuationOnlyNonTranslationTargetCount"], 4)
        self.assertEqual(
            sum(len(page["regions"]) for page in pages),
            manifest["totalExpectedDialogueBubbleCount"],
        )
        fallback_regions = [
            region
            for page in pages
            for region in page["regions"]
            if region["goldGeometrySource"] == "textPolygon"
        ]
        self.assertEqual(len(fallback_regions), 32)

    def test_release_measurement_accepts_complete_translation_gold(self) -> None:
        manifest, pages, evidence = validate_fixture(require_complete_gold=True)

        self.assertEqual(manifest["annotationStatus"]["status"], "complete")
        self.assertEqual(len(pages), 36)
        self.assertEqual(evidence["englishTranslationTargetCount"], 214)


if __name__ == "__main__":
    unittest.main()
