#!/usr/bin/env python3
"""Build the Chapter 5 gold annotations from reviewed local evidence.

The review file is authoritative for inclusion. Daemon output is used only as
optional translation evidence after a stable-id/geometry match. The generated
manifest remains incomplete until every translation target has Chinese, pinyin,
and token-level HSK gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "fixtures"
    / "benchmarks"
    / "30-years-since-the-prologue-chapter-5"
)
DEFAULT_REVIEW = REPO_ROOT / ".cache" / "ch5-gold-review-fresh" / "review.json"
DEFAULT_PRODUCT_AUDIT = (
    REPO_ROOT / ".cache" / "ch5-gold-proposal" / "run-2" / "audit"
)
DEFAULT_RAW_DETECTOR = REPO_ROOT / ".cache" / "ch5-raw-detector"
DEFAULT_APPROVED_TRANSLATIONS = DEFAULT_FIXTURE_ROOT / "approved-translations.json"
DEFAULT_GEOMETRY_CORRECTIONS = DEFAULT_FIXTURE_ROOT / "geometry-corrections.json"
DEFAULT_HSK_ARTIFACT = (
    REPO_ROOT / ".cache" / "language-data" / "hsk-2.0.normalized.json"
)
DEFAULT_DICTIONARY_ARTIFACT = (
    REPO_ROOT / ".cache" / "language-data" / "cc-cedict.normalized.json"
)

BENCHMARK_ID = "30-years-since-the-prologue-chapter-5"
REGION_ID_PREFIX = "30ysp-ch5"
REQUESTED_HSK_LEVEL = 5
INCLUDED_POLICY = (
    "English story text: dialogue, thoughts, narration, captions, and "
    "unbubbled spoken lines, including punctuation-only reactions and silent ellipses"
)
EXCLUDED_POLICY = (
    "sound effects, credits, scanlation promotion, translator notes, series "
    "branding, decorative labels, ambiguous OCR, and non-English text"
)
ERASE_OPERATION = (
    "erase every English glyph, outline, punctuation mark, and antialiased "
    "edge inside the polygon; preserve every pixel outside the erase mask"
)
LETTER_RE = re.compile(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")
REGION_ID_RE = re.compile(r"^30ysp-ch5-p\d{3}-r\d{2}$")
TRANSLATION_GOLD_FIELDS = ("simplifiedChinese", "pinyin", "hskTokens")
APPROVED_TRANSLATION_FIELDS = {
    "sourceEnglish",
    "simplifiedChinese",
    "pinyin",
}
GEOMETRY_CORRECTION_FIELDS = {
    "sourceEnglish",
    "textBoundsPixels",
    "containerStyle",
    "evidence",
}
GEOMETRY_CORRECTION_KEY_RE = re.compile(
    r"^(?P<source_file>[0-9]{3}\.webp)#(?P<missed_index>[1-9][0-9]*)$"
)
CONTAINER_STYLES = {
    "standard",
    "thought",
    "narration",
    "unbubbled",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_approved_translations(path: Path) -> dict[str, dict[str, str]]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{path} contains duplicate JSON key {key!r}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object keyed by canonical region ID")
    for region_id, entry in value.items():
        if not REGION_ID_RE.fullmatch(region_id):
            raise ValueError(f"{path} has invalid canonical region ID {region_id!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"{path} entry {region_id!r} must be an object")
        if set(entry) != APPROVED_TRANSLATION_FIELDS:
            raise ValueError(
                f"{path} entry {region_id!r} must contain exactly "
                "sourceEnglish, simplifiedChinese, and pinyin"
            )
        for field in APPROVED_TRANSLATION_FIELDS:
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(
                    f"{path} entry {region_id!r} field {field!r} "
                    "must be a non-empty string"
                )
    return value


def load_geometry_corrections(path: Path) -> dict[str, dict[str, Any]]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{path} contains duplicate JSON key {key!r}")
            output[key] = value
        return output

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    if set(value) != {
        "schemaVersion",
        "benchmarkId",
        "coordinateSpace",
        "policy",
        "regions",
    }:
        raise ValueError(f"{path} has unexpected top-level fields")
    if value["schemaVersion"] != 1 or value["benchmarkId"] != BENCHMARK_ID:
        raise ValueError(f"{path} does not target the Chapter 5 benchmark")
    if value["coordinateSpace"] != "source-image-pixels":
        raise ValueError(f"{path} has an unsupported coordinate space")
    if not isinstance(value["policy"], str) or not value["policy"].strip():
        raise ValueError(f"{path} has no geometry policy")
    regions = value["regions"]
    if not isinstance(regions, dict):
        raise ValueError(f"{path} regions must be an object")
    if not regions:
        raise ValueError(f"{path} regions must not be empty")

    for correction_key, correction in regions.items():
        if not GEOMETRY_CORRECTION_KEY_RE.fullmatch(correction_key):
            raise ValueError(
                f"{path} has invalid geometry correction key {correction_key!r}"
            )
        if not isinstance(correction, dict):
            raise ValueError(f"{path} entry {correction_key!r} must be an object")
        if set(correction) != GEOMETRY_CORRECTION_FIELDS:
            raise ValueError(
                f"{path} entry {correction_key!r} must contain exactly "
                "sourceEnglish, textBoundsPixels, containerStyle, and evidence"
            )
        if (
            not isinstance(correction["sourceEnglish"], str)
            or not correction["sourceEnglish"].strip()
        ):
            raise ValueError(
                f"{path} entry {correction_key!r} has invalid sourceEnglish"
            )
        bounds = correction["textBoundsPixels"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 4
            or any(type(value) is not int for value in bounds)
        ):
            raise ValueError(
                f"{path} entry {correction_key!r} textBoundsPixels must be "
                "four integers"
            )
        left, top, right, bottom = bounds
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError(
                f"{path} entry {correction_key!r} has invalid textBoundsPixels"
            )
        if correction["containerStyle"] not in CONTAINER_STYLES:
            raise ValueError(
                f"{path} entry {correction_key!r} has invalid containerStyle"
            )
        evidence = correction["evidence"]
        if not isinstance(evidence, dict) or evidence.get("type") not in {
            "ppocr-v5-mobile-full",
            "visual-punctuation-bounds",
        }:
            raise ValueError(
                f"{path} entry {correction_key!r} has invalid evidence"
            )
        if evidence["type"] == "ppocr-v5-mobile-full":
            if set(evidence) != {"type", "predictionIndices"}:
                raise ValueError(
                    f"{path} entry {correction_key!r} has unexpected OCR evidence"
                )
            indices = evidence["predictionIndices"]
            if (
                not isinstance(indices, list)
                or not indices
                or any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != len(indices)
            ):
                raise ValueError(
                    f"{path} entry {correction_key!r} has invalid predictionIndices"
                )
        elif set(evidence) != {"type", "thresholdedGlyphBoundsPixels"}:
            raise ValueError(
                f"{path} entry {correction_key!r} has unexpected visual evidence"
            )
        elif evidence["thresholdedGlyphBoundsPixels"] != bounds:
            raise ValueError(
                f"{path} entry {correction_key!r} visual bounds differ from "
                "textBoundsPixels"
            )
    return regions


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def file_identity(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


class HskTokenAnnotator:
    """Deterministic token metadata matching HskControl lookup semantics."""

    def __init__(
        self,
        hsk_path: Path,
        dictionary_path: Path,
        requested_level: int = REQUESTED_HSK_LEVEL,
    ) -> None:
        if requested_level not in range(1, 7):
            raise ValueError("requested HSK level must be between 1 and 6")
        self.requested_level = requested_level
        self.hsk_identity = file_identity(hsk_path)
        self.dictionary_identity = file_identity(dictionary_path)
        hsk = load_json(hsk_path)
        dictionary = load_json(dictionary_path)
        self._validate_artifact(hsk, hsk_path, kind="hsk")
        self._validate_artifact(dictionary, dictionary_path, kind="dictionary")
        self.hsk_revision = hsk["datasetRevision"]
        self.dictionary_revision = dictionary["datasetRevision"]

        self._hsk_by_word: dict[str, dict[str, Any]] = {}
        for entry in hsk["entries"]:
            word = entry.get("simplified")
            level = entry.get("level")
            if (
                not isinstance(word, str)
                or not word
                or not isinstance(level, int)
                or level not in range(1, 7)
                or word in self._hsk_by_word
            ):
                raise ValueError(f"{hsk_path} has an invalid or duplicate HSK entry")
            self._hsk_by_word[word] = entry

        dictionary_pinyin: dict[str, set[str]] = {}
        for entry in dictionary["entries"]:
            word = entry.get("simplified")
            pinyin = entry.get("pinyin")
            if not isinstance(word, str) or not word or not isinstance(pinyin, str):
                raise ValueError(f"{dictionary_path} has an invalid dictionary entry")
            if not pinyin.strip():
                raise ValueError(f"{dictionary_path} has an empty dictionary pinyin")
            dictionary_pinyin.setdefault(word, set()).add(pinyin)
        self._dictionary_pinyin = {
            word: " / ".join(sorted(values))
            for word, values in dictionary_pinyin.items()
        }

        words_by_first: dict[str, set[str]] = {}
        for word in self._hsk_by_word.keys() | self._dictionary_pinyin.keys():
            words_by_first.setdefault(word[0], set()).add(word)
        self._words_by_first = {
            first: sorted(words, key=lambda word: (-len(word), word))
            for first, words in words_by_first.items()
        }

    @staticmethod
    def _validate_artifact(value: Any, path: Path, *, kind: str) -> None:
        if not isinstance(value, dict) or value.get("schemaVersion") != 1:
            raise ValueError(f"{path} is not a schema-version-1 {kind} artifact")
        if value.get("completeness") != "complete":
            raise ValueError(f"{path} is not a complete {kind} artifact")
        if not isinstance(value.get("datasetRevision"), str) or not value[
            "datasetRevision"
        ].strip():
            raise ValueError(f"{path} has no {kind} dataset revision")
        entries = value.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path} has no {kind} entries")

    @staticmethod
    def _ignorable(character: str) -> bool:
        return character.isspace() or not character.isalnum()

    def _longest_match(self, text: str, start: int) -> int:
        for word in self._words_by_first.get(text[start], ()):
            if text.startswith(word, start):
                return start + len(word)
        return start + 1

    def annotate(self, text: str) -> list[dict[str, Any]]:
        tokens: list[dict[str, Any]] = []
        start = 0
        while start < len(text):
            if self._ignorable(text[start]):
                start += 1
                continue
            end = self._longest_match(text, start)
            word = text[start:end]
            hsk_entry = self._hsk_by_word.get(word)
            dictionary_pinyin = self._dictionary_pinyin.get(word)
            if hsk_entry is not None:
                level = hsk_entry["level"]
                token: dict[str, Any] = {
                    "text": word,
                    "startChar": start,
                    "endChar": end,
                    "pinyin": dictionary_pinyin or hsk_entry["pinyin"],
                    "classification": "hsk",
                    "hskLevel": level,
                    "aboveRequestedLevel": level > self.requested_level,
                }
            elif dictionary_pinyin is not None:
                token = {
                    "text": word,
                    "startChar": start,
                    "endChar": end,
                    "pinyin": dictionary_pinyin,
                    "classification": "non-hsk",
                    "aboveRequestedLevel": True,
                }
            else:
                # HskControl lookup also emits an individual unknown
                # alphanumeric scalar. Keep it verbatim for codes and Arabic
                # numerals rather than inventing a pronunciation.
                token = {
                    "text": word,
                    "startChar": start,
                    "endChar": end,
                    "pinyin": word,
                    "classification": "non-lexical",
                    "aboveRequestedLevel": False,
                }
            tokens.append(token)
            start = end
        return tokens

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "requestedLevel": self.requested_level,
            "segmentationPolicy": (
                "hsk-control-lookup-longest-match-union-character-offsets-v1"
            ),
            "hskArtifact": {
                "datasetRevision": self.hsk_revision,
                "bytes": self.hsk_identity[0],
                "sha256": self.hsk_identity[1],
            },
            "dictionaryArtifact": {
                "datasetRevision": self.dictionary_revision,
                "bytes": self.dictionary_identity[0],
                "sha256": self.dictionary_identity[1],
            },
        }


def clamp(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), 8)


def normalized_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()


def match_text(value: str) -> str:
    return NON_WORD_RE.sub(" ", value.lower()).strip()


def polygon_from_points(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    output: list[list[float]] = []
    for point in value:
        if isinstance(point, dict) and "x" in point and "y" in point:
            output.append([clamp(point["x"]), clamp(point["y"])])
        elif isinstance(point, list) and len(point) == 2:
            output.append([clamp(point[0]), clamp(point[1])])
        else:
            return None
    return output


def rect_polygon(x: float, y: float, width: float, height: float) -> list[list[float]]:
    left = clamp(x)
    top = clamp(y)
    right = clamp(x + width)
    bottom = clamp(y + height)
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def pixel_bounds_polygon(
    bounds: list[int], width: int, height: int
) -> list[list[float]]:
    left, top, right, bottom = bounds
    if right > width or bottom > height:
        raise ValueError(
            f"pixel bounds {bounds!r} exceed source dimensions {width}x{height}"
        )
    return [
        [clamp(left / width), clamp(top / height)],
        [clamp(right / width), clamp(top / height)],
        [clamp(right / width), clamp(bottom / height)],
        [clamp(left / width), clamp(bottom / height)],
    ]


def polygon_bounds(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_center(polygon: list[list[float]]) -> tuple[float, float]:
    left, top, right, bottom = polygon_bounds(polygon)
    return (left + right) / 2.0, (top + bottom) / 2.0


def bounds_iou(first: list[list[float]], second: list[list[float]]) -> float:
    a_left, a_top, a_right, a_bottom = polygon_bounds(first)
    b_left, b_top, b_right, b_bottom = polygon_bounds(second)
    overlap_width = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    overlap_height = max(0.0, min(a_bottom, b_bottom) - max(a_top, b_top))
    intersection = overlap_width * overlap_height
    first_area = max(0.0, a_right - a_left) * max(0.0, a_bottom - a_top)
    second_area = max(0.0, b_right - b_left) * max(0.0, b_bottom - b_top)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def center_inside(polygon: list[list[float]], container: list[list[float]]) -> bool:
    center_x, center_y = polygon_center(polygon)
    left, top, right, bottom = polygon_bounds(container)
    return left <= center_x <= right and top <= center_y <= bottom


def expanded_text_mask(
    text_polygon: list[list[float]], width: int, height: int
) -> list[list[float]]:
    left, top, right, bottom = polygon_bounds(text_polygon)
    x_pad = 3.0 / width
    y_pad = 3.0 / height
    return rect_polygon(
        left - x_pad,
        top - y_pad,
        (right - left) + (2 * x_pad),
        (bottom - top) + (2 * y_pad),
    )


def patch_mask(region: dict[str, Any]) -> list[list[float]] | None:
    rect = region.get("patch", {}).get("rect")
    if not isinstance(rect, dict):
        return None
    required = ("x", "y", "width", "height")
    if any(not isinstance(rect.get(key), (int, float)) for key in required):
        return None
    return rect_polygon(rect["x"], rect["y"], rect["width"], rect["height"])


def final_product_regions(path: Path, order: int) -> list[dict[str, Any]]:
    audit_path = path / f"{order:03d}.json"
    if not audit_path.is_file():
        return []
    audit = load_json(audit_path)
    latest: dict[str, dict[str, Any]] = {}
    for update in audit.get("updates", []):
        if update.get("type") not in {"regionReady", "regionRefined"}:
            continue
        region = update.get("region")
        if isinstance(region, dict) and isinstance(region.get("id"), str):
            latest[region["id"]] = region
    return list(latest.values())


def iter_detection_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(value, dict):
        return
    for key in ("detections", "candidates", "outputs", "results"):
        nested = value.get(key)
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    yield item


def detection_polygon(
    detection: dict[str, Any], width: int, height: int
) -> list[list[float]] | None:
    label_id = detection.get("labelId")
    label = str(detection.get("label", detection.get("class", ""))).lower()
    if isinstance(label_id, int) and label_id != 0:
        return None
    if label and "bubble" not in label and label not in {"0", "dialogue"}:
        return None
    direct = polygon_from_points(detection.get("bubblePolygon"))
    if direct is not None:
        return direct
    direct = polygon_from_points(detection.get("polygon"))
    if direct is not None:
        return direct
    bounds = detection.get("candidateBounds", detection.get("bbox"))
    if not isinstance(bounds, list) or len(bounds) != 4:
        return None
    left, top, right, bottom = (float(item) for item in bounds)
    if max(abs(left), abs(top), abs(right), abs(bottom)) > 1.0:
        left, right = left / width, right / width
        top, bottom = top / height, bottom / height
    if right <= left or bottom <= top:
        return None
    return rect_polygon(left, top, right - left, bottom - top)


def raw_bubble_polygons(
    path: Path, order: int, width: int, height: int
) -> list[list[list[float]]]:
    detector_path = path / f"{order:03d}.json"
    if not detector_path.is_file():
        return []
    output: list[list[list[float]]] = []
    for detection in iter_detection_objects(load_json(detector_path)):
        polygon = detection_polygon(detection, width, height)
        if polygon is not None:
            output.append(polygon)
    return output


def best_raw_bubble(
    text_polygon: list[list[float]], raw_polygons: list[list[list[float]]]
) -> list[list[float]] | None:
    containing = [
        polygon for polygon in raw_polygons if center_inside(text_polygon, polygon)
    ]
    if not containing:
        return None
    return min(
        containing,
        key=lambda polygon: (
            (polygon_bounds(polygon)[2] - polygon_bounds(polygon)[0])
            * (polygon_bounds(polygon)[3] - polygon_bounds(polygon)[1]),
            polygon_bounds(polygon),
        ),
    )


def review_candidates(
    image: dict[str, Any],
    geometry_corrections: dict[str, dict[str, Any]] | None = None,
    used_geometry_corrections: set[str] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    geometry_corrections = geometry_corrections or {}
    for proposal in image.get("proposals", []):
        if proposal.get("decision") != "accepted":
            continue
        text_polygon = polygon_from_points(
            proposal.get("proposalGeometry", {}).get("textPolygon")
        )
        if text_polygon is None:
            raise ValueError(
                f"{image['sourceFile']} proposal {proposal.get('proposalIndex')} "
                "has no valid text polygon"
            )
        output.append(
            {
                "reviewType": "acceptedProposal",
                "proposalIndex": proposal["proposalIndex"],
                "reviewRegionId": proposal["regionId"],
                "english": proposal["correctedEnglish"],
                "kind": proposal["semanticKind"],
                "uncertainty": proposal["uncertainty"],
                "textPolygon": text_polygon,
            }
        )
    for missed_index, missed in enumerate(
        image.get("missedStoryTextRegions", []), start=1
    ):
        correction_key = f"{image['sourceFile']}#{missed_index}"
        correction = geometry_corrections.get(correction_key)
        if geometry_corrections and correction is None:
            raise ValueError(
                f"geometry corrections are missing {correction_key}"
            )
        if correction is not None:
            if correction["sourceEnglish"] != missed["english"]:
                raise ValueError(
                    f"geometry correction sourceEnglish mismatch for "
                    f"{correction_key}: {correction['sourceEnglish']!r} != "
                    f"{missed['english']!r}"
                )
            text_polygon = pixel_bounds_polygon(
                correction["textBoundsPixels"],
                int(image["width"]),
                int(image["height"]),
            )
            if used_geometry_corrections is not None:
                used_geometry_corrections.add(correction_key)
        else:
            geometry = missed.get("geometry", {})
            if geometry.get("type") != "rect":
                raise ValueError(
                    f"{image['sourceFile']} missed region {missed_index} is not a rect"
                )
            text_polygon = rect_polygon(
                geometry["x"], geometry["y"], geometry["width"], geometry["height"]
            )
        output.append(
            {
                "reviewType": "missedStoryTextRegion",
                "missedIndex": missed_index,
                "english": missed["english"],
                "kind": missed["semanticKind"],
                "uncertainty": missed["uncertainty"],
                "textPolygon": text_polygon,
                **(
                    {
                        "geometryCorrectionKey": correction_key,
                        "containerStyle": correction["containerStyle"],
                    }
                    if correction is not None
                    else {}
                ),
            }
        )
    return output


def match_product_region(
    candidate: dict[str, Any],
    product_regions: list[dict[str, Any]],
    used_product_ids: set[str],
) -> tuple[dict[str, Any], str, float] | None:
    if candidate["reviewType"] == "acceptedProposal":
        for product in product_regions:
            if product.get("id") != candidate["reviewRegionId"]:
                continue
            product_polygon = polygon_from_points(product.get("textPolygon"))
            if product_polygon is None:
                return None
            iou = bounds_iou(candidate["textPolygon"], product_polygon)
            if iou >= 0.98:
                return product, "stable-region-id-and-text-geometry", iou
        return None

    expected_text = match_text(candidate["english"])
    matches: list[tuple[dict[str, Any], float]] = []
    for product in product_regions:
        stable_id = product.get("id")
        if not isinstance(stable_id, str) or stable_id in used_product_ids:
            continue
        product_polygon = polygon_from_points(product.get("textPolygon"))
        if product_polygon is None:
            continue
        if (
            match_text(str(product.get("sourceEnglish", ""))) == expected_text
            and center_inside(product_polygon, candidate["textPolygon"])
        ):
            matches.append((product, bounds_iou(candidate["textPolygon"], product_polygon)))
    if len(matches) != 1:
        return None
    return matches[0][0], "stable-region-id-source-text-and-geometry", matches[0][1]


def container_style(kind: str, bubble_polygon: list[list[float]] | None) -> str:
    if kind == "thought":
        return "thought"
    if kind == "narration":
        return "narration"
    return "standard" if bubble_polygon is not None else "unbubbled"


def copy_translation_gold(
    region: dict[str, Any], product: dict[str, Any]
) -> None:
    chinese = product.get("displayedChinese") or product.get("baseChinese")
    pinyin = product.get("pinyin")
    if isinstance(chinese, str) and chinese.strip():
        region["simplifiedChinese"] = chinese.strip()
    if isinstance(pinyin, str) and pinyin.strip():
        region["pinyin"] = pinyin.strip()
    tokens = product.get("hskTokens")
    if isinstance(tokens, list) and tokens:
        region["hskTokens"] = tokens
    hsk = product.get("hsk")
    if isinstance(hsk, dict):
        allowed = {
            key: hsk[key]
            for key in (
                "requestedLevel",
                "strictlyValid",
                "aboveLevelTokens",
                "repairState",
            )
            if key in hsk
        }
        if allowed:
            region["hskValidation"] = allowed


def fill_missing_approved_translation_gold(
    region: dict[str, Any],
    approved_translations: dict[str, dict[str, str]],
    used_approved_ids: set[str],
) -> None:
    region_id = region["id"]
    approved = approved_translations.get(region_id)
    if approved is None:
        return
    if approved["sourceEnglish"] != region["sourceEnglish"]:
        raise ValueError(
            f"approved translation sourceEnglish mismatch for {region_id!r}: "
            f"{approved['sourceEnglish']!r} != {region['sourceEnglish']!r}"
        )
    filled = False
    for field in ("simplifiedChinese", "pinyin"):
        if field not in region:
            region[field] = approved[field]
            filled = True
    if filled:
        used_approved_ids.add(region_id)


def build_annotation(
    image: dict[str, Any],
    product_audit: Path,
    raw_detector: Path,
    approved_translations: dict[str, dict[str, str]],
    used_approved_ids: set[str],
    token_annotator: HskTokenAnnotator,
    geometry_corrections: dict[str, dict[str, Any]] | None = None,
    used_geometry_corrections: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    order = int(image["imageIndex"])
    width = int(image["width"])
    height = int(image["height"])
    candidates = review_candidates(
        image,
        geometry_corrections,
        used_geometry_corrections,
    )
    candidates.sort(
        key=lambda candidate: (
            polygon_bounds(candidate["textPolygon"])[1],
            polygon_bounds(candidate["textPolygon"])[0],
            candidate.get("proposalIndex", 1_000_000),
            candidate.get("missedIndex", 1_000_000),
        )
    )
    products = final_product_regions(product_audit, order)
    raw_polygons = raw_bubble_polygons(raw_detector, order, width, height)
    used_product_ids: set[str] = set()
    regions: list[dict[str, Any]] = []
    missing = {field: 0 for field in TRANSLATION_GOLD_FIELDS}

    for reading_order, candidate in enumerate(candidates):
        match = match_product_region(candidate, products, used_product_ids)
        product = match[0] if match else None
        has_reviewed_geometry_correction = (
            "geometryCorrectionKey" in candidate
        )
        if has_reviewed_geometry_correction:
            # The committed correction is the geometry authority for manual
            # misses. Product patches and optional bubble-detector rectangles
            # remain translation/provenance evidence only.
            bubble_polygon = None
        else:
            product_polygon = (
                polygon_from_points(product.get("bubblePolygon"))
                if product is not None
                else None
            )
            bubble_polygon = product_polygon or best_raw_bubble(
                candidate["textPolygon"], raw_polygons
            )
        translation_target = bool(LETTER_RE.search(candidate["english"]))
        erase_polygon = (
            expanded_text_mask(candidate["textPolygon"], width, height)
            if has_reviewed_geometry_correction
            else (
                patch_mask(product)
                if product is not None
                else expanded_text_mask(candidate["textPolygon"], width, height)
            )
        )
        if erase_polygon is None:
            erase_polygon = expanded_text_mask(candidate["textPolygon"], width, height)

        region: dict[str, Any] = {
            "id": f"{REGION_ID_PREFIX}-p{order:03d}-r{reading_order:02d}",
            "kind": candidate["kind"],
            "containerStyle": candidate.get(
                "containerStyle",
                container_style(candidate["kind"], bubble_polygon),
            ),
            "readingOrder": reading_order,
            "sourceEnglish": candidate["english"],
            "normalizedEnglish": normalized_text(candidate["english"]),
            "textPolygon": candidate["textPolygon"],
            "eraseMask": {
                "encoding": "normalized-polygon-v1",
                "polygon": erase_polygon,
                "operation": ERASE_OPERATION,
            },
            "geometryAudit": (
                "detector-adjusted"
                if candidate["reviewType"] == "acceptedProposal"
                else "manual"
            ),
            "reviewProvenance": {
                "type": candidate["reviewType"],
                "uncertainty": candidate["uncertainty"],
            },
        }
        if candidate["reviewType"] == "acceptedProposal":
            region["reviewProvenance"]["proposalIndex"] = candidate["proposalIndex"]
            region["reviewProvenance"]["reviewRegionId"] = candidate["reviewRegionId"]
        else:
            region["reviewProvenance"]["missedIndex"] = candidate["missedIndex"]
        if bubble_polygon is not None:
            region["bubblePolygon"] = bubble_polygon
        if not translation_target:
            region["translationTarget"] = False

        if product is not None and match is not None:
            stable_id = product["id"]
            used_product_ids.add(stable_id)
            region["productionMatch"] = {
                "stableRegionId": stable_id,
                "method": match[1],
                "textGeometryIou": round(match[2], 8),
            }
            if translation_target:
                copy_translation_gold(region, product)

        if translation_target:
            fill_missing_approved_translation_gold(
                region, approved_translations, used_approved_ids
            )
            chinese = region.get("simplifiedChinese")
            if isinstance(chinese, str) and chinese:
                region["hskTokens"] = token_annotator.annotate(chinese)
            for field in TRANSLATION_GOLD_FIELDS:
                if field not in region:
                    missing[field] += 1
        regions.append(region)

    annotation = {
        "schemaVersion": 1,
        "page": {
            "order": order,
            "file": image["sourceFile"],
            "width": width,
            "height": height,
            "sourceSha256": image["sourceSha256"],
        },
        "policy": {
            "included": INCLUDED_POLICY,
            "excluded": EXCLUDED_POLICY,
            "detectorUse": (
                "proposal only; every region and omission was manually gated "
                "against the source image"
            ),
        },
        "regions": regions,
    }
    return annotation, missing


def generate(
    fixture_root: Path,
    review_path: Path,
    product_audit: Path,
    raw_detector: Path,
    approved_translations_path: Path,
    hsk_artifact_path: Path,
    dictionary_artifact_path: Path,
    geometry_corrections_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = fixture_root / "manifest.json"
    manifest = load_json(manifest_path)
    review = load_json(review_path)
    approved_translations = load_approved_translations(approved_translations_path)
    geometry_corrections = (
        load_geometry_corrections(geometry_corrections_path)
        if geometry_corrections_path is not None
        else {}
    )
    token_annotator = HskTokenAnnotator(
        hsk_artifact_path,
        dictionary_artifact_path,
    )
    if manifest.get("id") != BENCHMARK_ID or review.get("benchmarkId") != BENCHMARK_ID:
        raise ValueError("manifest and review must target the Chapter 5 benchmark")
    if len(manifest.get("images", [])) != len(review.get("images", [])):
        raise ValueError("manifest and review page counts differ")

    review_by_file = {image["sourceFile"]: image for image in review["images"]}
    annotations_root = fixture_root / "annotations"
    total_missing = {field: 0 for field in TRANSLATION_GOLD_FIELDS}
    missing_by_page: dict[str, dict[str, int]] = {}
    total_regions = 0
    total_dialogue_bubbles = 0
    total_targets = 0
    total_exclusions = 0
    total_narration = 0
    used_approved_ids: set[str] = set()
    used_geometry_corrections: set[str] = set()

    for image_entry in manifest["images"]:
        source_file = image_entry["file"]
        review_image = review_by_file.get(source_file)
        if review_image is None:
            raise ValueError(f"review is missing {source_file}")
        if review_image["sourceSha256"] != image_entry["sha256"]:
            raise ValueError(f"{source_file} source SHA-256 differs from review")
        annotation, page_missing = build_annotation(
            review_image,
            product_audit,
            raw_detector,
            approved_translations,
            used_approved_ids,
            token_annotator,
            geometry_corrections,
            used_geometry_corrections,
        )
        annotation_path = annotations_root / f"{image_entry['order']:03d}.json"
        write_json(annotation_path, annotation)
        annotation_bytes, annotation_sha256 = file_identity(annotation_path)
        regions = annotation["regions"]
        bubble_regions = [
            region for region in regions if region["kind"] in {"dialogue", "thought"}
        ]
        targets = [
            region for region in regions if region.get("translationTarget", True)
        ]
        exclusions = len(regions) - len(targets)
        narration_count = sum(region["kind"] == "narration" for region in regions)

        image_entry.update(
            {
                "annotation": f"annotations/{image_entry['order']:03d}.json",
                "annotationBytes": annotation_bytes,
                "annotationSha256": annotation_sha256,
                "expectedRegionCount": len(regions),
                "expectedDialogueBubbleCount": len(bubble_regions),
                "expectedEnglishTranslationTargetCount": len(targets),
                "expectedUntouchedExclusionCount": exclusions,
                "expectedNarrationCount": narration_count,
            }
        )
        page_missing = {
            field: count for field, count in page_missing.items() if count > 0
        }
        if page_missing:
            missing_by_page[source_file] = page_missing
        for field in TRANSLATION_GOLD_FIELDS:
            total_missing[field] += page_missing.get(field, 0)
        total_regions += len(regions)
        total_dialogue_bubbles += len(bubble_regions)
        total_targets += len(targets)
        total_exclusions += exclusions
        total_narration += narration_count

    unused_approved_ids = sorted(set(approved_translations) - used_approved_ids)
    if unused_approved_ids:
        raise ValueError(
            "approved translations contain entries that did not fill missing "
            f"Chinese or pinyin: {', '.join(unused_approved_ids)}"
        )
    unused_geometry_corrections = sorted(
        set(geometry_corrections) - used_geometry_corrections
    )
    if unused_geometry_corrections:
        raise ValueError(
            "geometry corrections contain unused entries: "
            f"{', '.join(unused_geometry_corrections)}"
        )

    schema_path = fixture_root / manifest["annotationSchema"]
    schema_bytes, schema_sha256 = file_identity(schema_path)
    try:
        approved_relative_path = approved_translations_path.relative_to(
            fixture_root
        ).as_posix()
    except ValueError as error:
        raise ValueError(
            "approved translations must be stored inside the fixture root"
        ) from error
    approved_bytes, approved_sha256 = file_identity(approved_translations_path)
    manifest["annotationSchemaBytes"] = schema_bytes
    manifest["annotationSchemaSha256"] = schema_sha256
    manifest["approvedTranslations"] = {
        "path": approved_relative_path,
        "bytes": approved_bytes,
        "sha256": approved_sha256,
    }
    if geometry_corrections_path is not None:
        try:
            geometry_relative_path = geometry_corrections_path.relative_to(
                fixture_root
            ).as_posix()
        except ValueError as error:
            raise ValueError(
                "geometry corrections must be stored inside the fixture root"
            ) from error
        geometry_bytes, geometry_sha256 = file_identity(
            geometry_corrections_path
        )
        manifest["geometryCorrections"] = {
            "path": geometry_relative_path,
            "bytes": geometry_bytes,
            "sha256": geometry_sha256,
            "correctedMissedStoryTextRegionCount": len(geometry_corrections),
        }
    manifest["hskTokenAnnotation"] = token_annotator.manifest_metadata()
    manifest["totalExpectedRegionCount"] = total_regions
    manifest["totalExpectedDialogueBubbleCount"] = total_dialogue_bubbles
    manifest["totalExpectedEnglishTranslationTargetCount"] = total_targets
    manifest["totalExpectedUntouchedExclusionCount"] = total_exclusions
    manifest["totalExpectedNarrationCount"] = total_narration
    missing_pages = list(missing_by_page)
    manifest["annotationStatus"] = {
        "status": "incomplete" if missing_pages else "complete",
        "reasonCode": (
            "gold-fields-missing" if missing_pages else "all-gold-fields-present"
        ),
        "reviewedPageCount": len(manifest["images"]),
        "generatedPageCount": len(manifest["images"]),
        "completedPageCount": len(manifest["images"]) - len(missing_pages),
        "requiredPageCount": len(manifest["images"]),
        "missingPages": missing_pages,
        "missingFieldCounts": total_missing,
        "totalMissingFieldCount": sum(total_missing.values()),
        "missingFieldsByPage": missing_by_page,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--product-audit", type=Path, default=DEFAULT_PRODUCT_AUDIT)
    parser.add_argument("--raw-detector", type=Path, default=DEFAULT_RAW_DETECTOR)
    parser.add_argument(
        "--approved-translations",
        type=Path,
        default=DEFAULT_APPROVED_TRANSLATIONS,
    )
    parser.add_argument(
        "--geometry-corrections",
        type=Path,
        default=DEFAULT_GEOMETRY_CORRECTIONS,
    )
    parser.add_argument(
        "--hsk-artifact",
        type=Path,
        default=DEFAULT_HSK_ARTIFACT,
    )
    parser.add_argument(
        "--dictionary-artifact",
        type=Path,
        default=DEFAULT_DICTIONARY_ARTIFACT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate(
        args.fixture_root.resolve(),
        args.review.resolve(),
        args.product_audit.resolve(),
        args.raw_detector.resolve(),
        args.approved_translations.resolve(),
        args.hsk_artifact.resolve(),
        args.dictionary_artifact.resolve(),
        args.geometry_corrections.resolve(),
    )
    status = manifest["annotationStatus"]
    print(
        f"Generated {manifest['totalExpectedRegionCount']} reviewed regions across "
        f"{manifest['pageCount']} pages; status={status['status']}; "
        f"missingFields={status['totalMissingFieldCount']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
