#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from build_chapter5_annotations import (
    BENCHMARK_ID,
    HskTokenAnnotator,
    fill_missing_approved_translation_gold,
    generate,
    load_approved_translations,
    load_geometry_corrections,
    raw_bubble_polygons,
    review_candidates,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
FIXTURE_SCHEMA = (
    SCRIPT_ROOT.parents[1]
    / "fixtures"
    / "benchmarks"
    / "30-years-since-the-prologue-chapter-5"
    / "annotation.schema.json"
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_language_artifacts(root: Path) -> tuple[Path, Path]:
    hsk_path = root / "hsk.json"
    dictionary_path = root / "dictionary.json"
    write_json(
        hsk_path,
        {
            "schemaVersion": 1,
            "standard": "2.0",
            "datasetRevision": "test-hsk-v1",
            "completeness": "complete",
            "entries": [
                {"simplified": "你好", "pinyin": "nǐ hǎo", "level": 1},
                {"simplified": "稍后", "pinyin": "shāo hòu", "level": 6},
                {"simplified": "深奥", "pinyin": "shēn ào", "level": 6},
            ],
        },
    )
    write_json(
        dictionary_path,
        {
            "schemaVersion": 1,
            "datasetRevision": "test-dictionary-v1",
            "completeness": "complete",
            "entries": [
                {"simplified": "你好", "pinyin": "nǐ hǎo"},
                {"simplified": "稍后", "pinyin": "shāo hòu"},
                {"simplified": "深奥", "pinyin": "shēn ào"},
                {"simplified": "秘术", "pinyin": "mì shù"},
            ],
        },
    )
    return hsk_path, dictionary_path


class Chapter5AnnotationBuilderTests(unittest.TestCase):
    def test_geometry_correction_replaces_approximate_missed_region(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "geometry-corrections.json"
            write_json(
                path,
                {
                    "schemaVersion": 1,
                    "benchmarkId": BENCHMARK_ID,
                    "coordinateSpace": "source-image-pixels",
                    "policy": "Reviewed source-pixel bounds.",
                    "regions": {
                        "001.webp#1": {
                            "sourceEnglish": "Hello.",
                            "textBoundsPixels": [10, 20, 90, 40],
                            "containerStyle": "standard",
                            "evidence": {
                                "type": "ppocr-v5-mobile-full",
                                "predictionIndices": [3],
                            },
                        }
                    },
                },
            )
            corrections = load_geometry_corrections(path)
            used: set[str] = set()
            image = {
                "sourceFile": "001.webp",
                "width": 100,
                "height": 200,
                "proposals": [],
                "missedStoryTextRegions": [
                    {
                        "english": "Hello.",
                        "semanticKind": "dialogue",
                        "uncertainty": "low",
                        "geometry": {
                            "type": "rect",
                            "x": 0.5,
                            "y": 0.5,
                            "width": 0.4,
                            "height": 0.4,
                        },
                    }
                ],
            }

            candidates = review_candidates(image, corrections, used)

            self.assertEqual(used, {"001.webp#1"})
            self.assertEqual(
                candidates[0]["textPolygon"],
                [[0.1, 0.1], [0.9, 0.1], [0.9, 0.2], [0.1, 0.2]],
            )
            self.assertEqual(candidates[0]["containerStyle"], "standard")
            with self.assertRaisesRegex(
                ValueError, "geometry corrections are missing 001.webp#1"
            ):
                review_candidates(
                    image,
                    {
                        "002.webp#1": {
                            "sourceEnglish": "Unused.",
                            "textBoundsPixels": [1, 1, 2, 2],
                            "containerStyle": "standard",
                            "evidence": {
                                "type": "ppocr-v5-mobile-full",
                                "predictionIndices": [0],
                            },
                        }
                    },
                )

    def test_hsk_tokens_have_deterministic_spans_levels_and_above_markers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hsk_path, dictionary_path = write_language_artifacts(Path(temporary))
            annotator = HskTokenAnnotator(hsk_path, dictionary_path)

            tokens = annotator.annotate("你好，深奥秘术 X。")

            self.assertEqual(
                tokens,
                [
                    {
                        "text": "你好",
                        "startChar": 0,
                        "endChar": 2,
                        "pinyin": "nǐ hǎo",
                        "classification": "hsk",
                        "hskLevel": 1,
                        "aboveRequestedLevel": False,
                    },
                    {
                        "text": "深奥",
                        "startChar": 3,
                        "endChar": 5,
                        "pinyin": "shēn ào",
                        "classification": "hsk",
                        "hskLevel": 6,
                        "aboveRequestedLevel": True,
                    },
                    {
                        "text": "秘术",
                        "startChar": 5,
                        "endChar": 7,
                        "pinyin": "mì shù",
                        "classification": "non-hsk",
                        "aboveRequestedLevel": True,
                    },
                    {
                        "text": "X",
                        "startChar": 8,
                        "endChar": 9,
                        "pinyin": "X",
                        "classification": "non-lexical",
                        "aboveRequestedLevel": False,
                    },
                ],
            )
            for token in tokens:
                self.assertEqual(
                    "你好，深奥秘术 X。"[token["startChar"] : token["endChar"]],
                    token["text"],
                )

    def test_raw_detector_bubble_geometry_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(raw_bubble_polygons(root, 1, 100, 200), [])
            write_json(
                root / "001.json",
                {
                    "detections": [
                        {"labelId": 1, "bbox": [20, 40, 60, 80]},
                        {"labelId": 0, "bbox": [10, 20, 90, 100]},
                    ]
                },
            )
            self.assertEqual(
                raw_bubble_polygons(root, 1, 100, 200),
                [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.5], [0.1, 0.5]]],
            )

    def test_approved_translation_loader_rejects_duplicate_region_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "approved-translations.json"
            path.write_text(
                """
{
  "30ysp-ch5-p001-r00": {
    "sourceEnglish": "First.",
    "simplifiedChinese": "第一。",
    "pinyin": "Dì-yī."
  },
  "30ysp-ch5-p001-r00": {
    "sourceEnglish": "Second.",
    "simplifiedChinese": "第二。",
    "pinyin": "Dì-èr."
  }
}
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_approved_translations(path)

    def test_approved_translation_never_overwrites_existing_gold(self) -> None:
        region = {
            "id": "30ysp-ch5-p001-r00",
            "sourceEnglish": "Hello.",
            "simplifiedChinese": "已有译文。",
            "pinyin": "Yǐyǒu yìwén.",
        }
        used: set[str] = set()
        fill_missing_approved_translation_gold(
            region,
            {
                "30ysp-ch5-p001-r00": {
                    "sourceEnglish": "Hello.",
                    "simplifiedChinese": "不得覆盖。",
                    "pinyin": "Bùdé fùgài.",
                }
            },
            used,
        )
        self.assertEqual(region["simplifiedChinese"], "已有译文。")
        self.assertEqual(region["pinyin"], "Yǐyǒu yìwén.")
        self.assertEqual(used, set())

    def test_review_is_authoritative_and_missing_gold_stays_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture"
            product = root / "product"
            raw = root / "raw"
            fixture.mkdir()
            shutil.copyfile(FIXTURE_SCHEMA, fixture / "annotation.schema.json")
            source_hash = "a" * 64
            write_json(
                fixture / "manifest.json",
                {
                    "schemaVersion": 3,
                    "id": BENCHMARK_ID,
                    "pageCount": 1,
                    "annotationSchema": "annotation.schema.json",
                    "images": [
                        {
                            "order": 1,
                            "file": "001.webp",
                            "sha256": source_hash,
                            "bytes": 10,
                            "width": 100,
                            "height": 200,
                        }
                    ],
                },
            )
            accepted_polygon = [
                {"x": 0.1, "y": 0.1},
                {"x": 0.5, "y": 0.1},
                {"x": 0.5, "y": 0.2},
                {"x": 0.1, "y": 0.2},
            ]
            write_json(
                root / "review.json",
                {
                    "benchmarkId": BENCHMARK_ID,
                    "images": [
                        {
                            "imageIndex": 1,
                            "sourceFile": "001.webp",
                            "sourceSha256": source_hash,
                            "width": 100,
                            "height": 200,
                            "proposals": [
                                {
                                    "proposalIndex": 1,
                                    "regionId": "stable-accepted",
                                    "decision": "accepted",
                                    "correctedEnglish": "Hello.",
                                    "semanticKind": "dialogue",
                                    "uncertainty": "low",
                                    "proposalGeometry": {
                                        "textPolygon": accepted_polygon
                                    },
                                },
                                {
                                    "proposalIndex": 2,
                                    "regionId": "stable-rejected",
                                    "decision": "rejected",
                                    "correctedEnglish": None,
                                    "semanticKind": None,
                                    "uncertainty": "low",
                                    "proposalGeometry": {
                                        "textPolygon": accepted_polygon
                                    },
                                },
                            ],
                            "missedStoryTextRegions": [
                                {
                                    "english": "Later.",
                                    "semanticKind": "narration",
                                    "uncertainty": "low",
                                    "geometry": {
                                        "type": "rect",
                                        "x": 0.2,
                                        "y": 0.7,
                                        "width": 0.6,
                                        "height": 0.1,
                                    },
                                }
                            ],
                        }
                    ],
                },
            )
            write_json(
                product / "001.json",
                {
                    "updates": [
                        {
                            "type": "regionReady",
                            "region": {
                                "id": "stable-accepted",
                                "sourceEnglish": "HELLO.",
                                "textPolygon": accepted_polygon,
                                "bubblePolygon": [
                                    {"x": 0.05, "y": 0.05},
                                    {"x": 0.55, "y": 0.05},
                                    {"x": 0.55, "y": 0.25},
                                    {"x": 0.05, "y": 0.25},
                                ],
                                "patch": {
                                    "rect": {
                                        "x": 0.09,
                                        "y": 0.09,
                                        "width": 0.42,
                                        "height": 0.12,
                                    }
                                },
                                "displayedChinese": "你好。",
                                "pinyin": "nǐ hǎo",
                                "hsk": {
                                    "requestedLevel": 5,
                                    "strictlyValid": True,
                                    "aboveLevelTokens": [],
                                    "repairState": "not-needed",
                                },
                            },
                        },
                        {
                            "type": "regionReady",
                            "region": {
                                "id": "stable-rejected",
                                "sourceEnglish": "CREDITS",
                                "textPolygon": accepted_polygon,
                                "displayedChinese": "片尾",
                                "pinyin": "piàn wěi",
                            },
                        },
                    ]
                },
            )
            approved_path = fixture / "approved-translations.json"
            approved = {
                "30ysp-ch5-p001-r01": {
                    "sourceEnglish": "Later.",
                    "simplifiedChinese": "稍后。",
                    "pinyin": "Shāohòu.",
                }
            }
            write_json(approved_path, approved)
            hsk_path, dictionary_path = write_language_artifacts(fixture)

            first = generate(
                fixture,
                root / "review.json",
                product,
                raw,
                approved_path,
                hsk_path,
                dictionary_path,
            )
            annotation_path = fixture / "annotations" / "001.json"
            first_bytes = annotation_path.read_bytes()
            second = generate(
                fixture,
                root / "review.json",
                product,
                raw,
                approved_path,
                hsk_path,
                dictionary_path,
            )
            self.assertEqual(first_bytes, annotation_path.read_bytes())
            self.assertEqual(first, second)

            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            self.assertEqual(len(annotation["regions"]), 2)
            self.assertEqual(
                [region["sourceEnglish"] for region in annotation["regions"]],
                ["Hello.", "Later."],
            )
            self.assertEqual(
                [region["id"] for region in annotation["regions"]],
                ["30ysp-ch5-p001-r00", "30ysp-ch5-p001-r01"],
            )
            translated = annotation["regions"][0]
            self.assertEqual(translated["simplifiedChinese"], "你好。")
            self.assertEqual(translated["pinyin"], "nǐ hǎo")
            self.assertEqual(translated["hskTokens"][0]["text"], "你好")
            self.assertFalse(
                translated["hskTokens"][0]["aboveRequestedLevel"]
            )
            self.assertEqual(annotation["regions"][1]["simplifiedChinese"], "稍后。")
            self.assertEqual(annotation["regions"][1]["pinyin"], "Shāohòu.")
            self.assertEqual(annotation["regions"][1]["hskTokens"][0]["hskLevel"], 6)
            self.assertTrue(
                annotation["regions"][1]["hskTokens"][0]["aboveRequestedLevel"]
            )
            self.assertNotIn("bubblePolygon", annotation["regions"][1])
            self.assertEqual(
                first["annotationStatus"]["missingFieldCounts"],
                {"simplifiedChinese": 0, "pinyin": 0, "hskTokens": 0},
            )
            self.assertEqual(first["annotationStatus"]["status"], "complete")
            self.assertEqual(first["annotationStatus"]["totalMissingFieldCount"], 0)
            expected_hash = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
            self.assertEqual(first["images"][0]["annotationSha256"], expected_hash)
            approved_bytes = approved_path.read_bytes()
            self.assertEqual(
                first["approvedTranslations"],
                {
                    "path": "approved-translations.json",
                    "bytes": len(approved_bytes),
                    "sha256": hashlib.sha256(approved_bytes).hexdigest(),
                },
            )

            approved["30ysp-ch5-p001-r01"]["sourceEnglish"] = "Not later."
            write_json(approved_path, approved)
            with self.assertRaisesRegex(ValueError, "sourceEnglish mismatch"):
                generate(
                    fixture,
                    root / "review.json",
                    product,
                    raw,
                    approved_path,
                    hsk_path,
                    dictionary_path,
                )

            approved["30ysp-ch5-p001-r01"]["sourceEnglish"] = "Later."
            approved["30ysp-ch5-p001-r99"] = {
                "sourceEnglish": "Unused.",
                "simplifiedChinese": "未使用。",
                "pinyin": "Wèi shǐyòng.",
            }
            write_json(approved_path, approved)
            with self.assertRaisesRegex(ValueError, "did not fill missing"):
                generate(
                    fixture,
                    root / "review.json",
                    product,
                    raw,
                    approved_path,
                    hsk_path,
                    dictionary_path,
                )


if __name__ == "__main__":
    unittest.main()
