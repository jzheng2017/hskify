#!/usr/bin/env python3
"""Score chapter 5 production-postprocessed detector candidates against gold bubbles.

Inputs are the unmodified JSON documents written by the existing
`comic-text-bubble-detector --output` CLI and the exact source WebPs. The
scorer mirrors production geometry and cheap pixel confirmation, performs no
inference, and reads no browser job updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, UnidentifiedImageError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    REPOSITORY_ROOT
    / "fixtures"
    / "benchmarks"
    / "30-years-since-the-prologue-chapter-5"
)
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
EVIDENCE_SCHEMA_PATH = FIXTURE_ROOT / "detector-benchmark-evidence.schema.json"
DEFAULT_MINIMUM_IOU = 0.5
RAW_DETECTOR_MINIMUM_SCORE = 0.3
MINIMUM_DIALOGUE_JOINT_CONFIDENCE = 0.4
BUBBLE_LABEL_ID = 0
DIALOGUE_TEXT_LABEL_ID = 1
DUPLICATE_TEXT_IOU = 0.55
CONTAINED_TEXT_OVERLAP = 0.78
DARK_CARD_MAX_LUMA = 32
DARK_CARD_MINIMUM_DARK_PERCENT = 70
DARK_CARD_MINIMUM_DARK_RATIO = DARK_CARD_MINIMUM_DARK_PERCENT / 100
DETECTOR_GOLD_KINDS = frozenset(("dialogue", "thought"))
SUPPORTED_REGION_KINDS = DETECTOR_GOLD_KINDS | {"narration"}


class BenchmarkInputError(ValueError):
    """Raised when committed gold or detector evidence is malformed."""


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkInputError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkInputError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, name: str | None = None) -> dict[str, Any]:
    return {
        "file": name or path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkInputError(message)


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be a number",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def validate_polygon(value: Any, label: str) -> list[list[float]]:
    require(isinstance(value, list) and len(value) >= 4, f"{label} must have >= 4 points")
    polygon: list[list[float]] = []
    for index, point in enumerate(value):
        require(
            isinstance(point, list) and len(point) == 2,
            f"{label}[{index}] must be [x, y]",
        )
        x = finite_number(point[0], f"{label}[{index}][0]")
        y = finite_number(point[1], f"{label}[{index}][1]")
        require(0 <= x <= 1 and 0 <= y <= 1, f"{label}[{index}] is not normalized")
        polygon.append([x, y])
    return polygon


def validate_fixture(
    *, require_complete_gold: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    require(MANIFEST_PATH.is_file(), f"missing fixture manifest: {MANIFEST_PATH}")
    manifest = load_object(MANIFEST_PATH)
    require(manifest.get("schemaVersion") == 3, "fixture manifest schemaVersion must be 3")
    require(
        manifest.get("id") == "30-years-since-the-prologue-chapter-5",
        "unexpected benchmark id",
    )

    schema_path = FIXTURE_ROOT / str(manifest.get("annotationSchema", ""))
    require(schema_path.is_file(), f"missing annotation schema: {schema_path}")
    schema_identity = file_identity(
        schema_path, name=str(manifest["annotationSchema"])
    )
    require(
        schema_identity["bytes"] == manifest.get("annotationSchemaBytes"),
        "annotation schema byte count does not match the fixture manifest",
    )
    require(
        schema_identity["sha256"] == manifest.get("annotationSchemaSha256"),
        "annotation schema SHA-256 does not match the fixture manifest",
    )

    images = manifest.get("images")
    require(isinstance(images, list), "fixture manifest images must be an array")
    require(
        len(images) == manifest.get("pageCount"),
        "fixture manifest pageCount must equal the images array length",
    )
    annotation_status = manifest.get("annotationStatus")
    require(
        isinstance(annotation_status, dict),
        "fixture manifest annotationStatus must be an object",
    )
    completed_pages = annotation_status.get("completedPageCount")
    required_pages = annotation_status.get("requiredPageCount")
    require(
        annotation_status.get("status") in {"complete", "incomplete"}
        and annotation_status.get("reviewedPageCount") == manifest.get("pageCount")
        and annotation_status.get("generatedPageCount") == manifest.get("pageCount")
        and required_pages == manifest.get("pageCount")
        and type(completed_pages) is int
        and 0 <= completed_pages <= required_pages,
        "fixture manifest annotationStatus is inconsistent",
    )
    if require_complete_gold:
        require(
            annotation_status.get("status") == "complete"
            and completed_pages == required_pages
            and annotation_status.get("totalMissingFieldCount") == 0
            and annotation_status.get("missingPages") == [],
        "gold fixture is incomplete: "
        f"status={annotation_status.get('status')!r}, "
        f"completedPageCount={completed_pages!r}, "
        f"requiredPageCount={required_pages!r}, "
        f"reasonCode={annotation_status.get('reasonCode')!r}",
        )

    pages: list[dict[str, Any]] = []
    annotation_identities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_regions = 0
    total_gold = 0
    total_narration = 0
    total_targets = 0
    total_exclusions = 0
    for expected_order, image in enumerate(images, start=1):
        require(isinstance(image, dict), f"manifest image {expected_order} must be an object")
        require(image.get("order") == expected_order, "manifest page order is not contiguous")
        width = image.get("width")
        height = image.get("height")
        require(type(width) is int and width > 0, f"page {expected_order} width is invalid")
        require(type(height) is int and height > 0, f"page {expected_order} height is invalid")
        source_name = image.get("file")
        source_bytes = image.get("bytes")
        source_sha256 = image.get("sha256")
        require(
            isinstance(source_name, str) and source_name == f"{expected_order:03}.webp",
            f"page {expected_order} source file is invalid",
        )
        require(
            type(source_bytes) is int and source_bytes > 0,
            f"page {expected_order} source byte count is invalid",
        )
        require(
            isinstance(source_sha256, str)
            and len(source_sha256) == 64
            and all(character in "0123456789abcdef" for character in source_sha256),
            f"page {expected_order} source SHA-256 is invalid",
        )

        annotation_name = image.get("annotation")
        require(isinstance(annotation_name, str), f"page {expected_order} annotation is missing")
        annotation_path = FIXTURE_ROOT / annotation_name
        require(annotation_path.is_file(), f"missing annotation: {annotation_path}")
        identity = file_identity(annotation_path, name=annotation_name)
        require(
            identity["bytes"] == image.get("annotationBytes"),
            f"{annotation_name} byte count does not match the fixture manifest",
        )
        require(
            identity["sha256"] == image.get("annotationSha256"),
            f"{annotation_name} SHA-256 does not match the fixture manifest",
        )
        identity["page"] = expected_order
        annotation_identities.append(identity)

        annotation = load_object(annotation_path)
        page = annotation.get("page")
        require(isinstance(page, dict), f"{annotation_name}.page must be an object")
        require(page.get("order") == expected_order, f"{annotation_name} page order mismatch")
        require(page.get("file") == image.get("file"), f"{annotation_name} source file mismatch")
        require(
            page.get("sourceSha256") == image.get("sha256"),
            f"{annotation_name} source SHA-256 mismatch",
        )
        require(
            (page.get("width"), page.get("height")) == (width, height),
            f"{annotation_name} dimensions mismatch",
        )

        regions = annotation.get("regions")
        require(isinstance(regions, list), f"{annotation_name}.regions must be an array")
        require(
            len(regions) == image.get("expectedRegionCount"),
            f"{annotation_name} reviewed-region count mismatch",
        )
        targets = 0
        exclusions = 0
        narrations = 0
        normalized_regions: list[dict[str, Any]] = []
        for region_index, region in enumerate(regions):
            require(isinstance(region, dict), f"{annotation_name} region must be an object")
            region_id = region.get("id")
            expected_id = f"30ysp-ch5-p{expected_order:03d}-r{region_index:02d}"
            require(
                region_id == expected_id,
                f"{annotation_name} region id {region_id!r} != {expected_id!r}",
            )
            require(region_id not in seen_ids, f"duplicate gold region id: {region_id}")
            seen_ids.add(region_id)
            kind = region.get("kind")
            require(
                kind in SUPPORTED_REGION_KINDS,
                f"{region_id} kind is invalid",
            )
            text_polygon = validate_polygon(
                region.get("textPolygon"), f"{region_id}.textPolygon"
            )
            bubble_polygon = region.get("bubblePolygon")
            validated_bubble_polygon = (
                validate_polygon(bubble_polygon, f"{region_id}.bubblePolygon")
                if bubble_polygon is not None
                else None
            )
            translation_target = region.get("translationTarget") is not False
            require(
                kind != "narration" or translation_target,
                f"{region_id} narration must remain an OCR/translation target",
            )
            targets += int(translation_target)
            exclusions += int(not translation_target)
            narrations += int(kind == "narration")
            if kind in DETECTOR_GOLD_KINDS:
                normalized_regions.append(
                    {
                        "id": region_id,
                        "kind": kind,
                        "translationTarget": translation_target,
                        "goldPolygon": validated_bubble_polygon or text_polygon,
                        "goldGeometrySource": (
                            "bubblePolygon"
                            if validated_bubble_polygon is not None
                            else "textPolygon"
                        ),
                    }
                )

        require(
            len(normalized_regions) == image.get("expectedDialogueBubbleCount"),
            f"{annotation_name} dialogue/thought detector-gold count mismatch",
        )
        require(
            narrations == image.get("expectedNarrationCount"),
            f"{annotation_name} narration count mismatch",
        )
        require(
            targets == image.get("expectedEnglishTranslationTargetCount"),
            f"{annotation_name} translation-target count mismatch",
        )
        require(
            exclusions == image.get("expectedUntouchedExclusionCount"),
            f"{annotation_name} exclusion count mismatch",
        )
        total_regions += len(regions)
        total_gold += len(normalized_regions)
        total_narration += narrations
        total_targets += targets
        total_exclusions += exclusions
        pages.append(
            {
                "order": expected_order,
                "file": image["file"],
                "url": image.get("url")
                or f"urn:hskify:{manifest['id']}:{image['file']}",
                "bytes": image["bytes"],
                "sha256": image["sha256"],
                "width": width,
                "height": height,
                "regions": normalized_regions,
            }
        )

    manifest_totals = (
        ("totalExpectedRegionCount", total_regions),
        ("totalExpectedDialogueBubbleCount", total_gold),
        ("totalExpectedNarrationCount", total_narration),
        ("totalExpectedEnglishTranslationTargetCount", total_targets),
        ("totalExpectedUntouchedExclusionCount", total_exclusions),
    )
    for field, actual in manifest_totals:
        expected = manifest.get(field)
        if expected is not None:
            require(
                actual == expected,
                f"fixture {field} is {expected}, but annotations contain {actual}",
            )
    fixture_evidence = {
        "manifest": file_identity(MANIFEST_PATH, name="manifest.json"),
        "annotationSchema": schema_identity,
        "annotations": annotation_identities,
        "pageCount": len(pages),
        "regionCount": total_regions,
        "goldBubbleCount": total_gold,
        "narrationRegionCount": total_narration,
        "englishTranslationTargetCount": total_targets,
        "punctuationOnlyNonTranslationTargetCount": total_exclusions,
    }
    return manifest, pages, fixture_evidence


def rectangle_for_polygon(
    polygon: list[list[float]], width: int, height: int
) -> list[float]:
    xs = [point[0] * width for point in polygon]
    ys = [point[1] * height for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def rectangle_iou(left: list[float], right: list[float]) -> float:
    intersection = rectangle_intersection_area(left, right)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def rectangle_intersection_area(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return intersection_width * intersection_height


def rectangle_overlap_over_smaller(left: list[float], right: list[float]) -> float:
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    smaller = min(left_area, right_area)
    return rectangle_intersection_area(left, right) / smaller if smaller > 0 else 0.0


def rectangle_center(rectangle: list[float]) -> tuple[float, float]:
    return (
        (rectangle[0] + rectangle[2]) * 0.5,
        (rectangle[1] + rectangle[3]) * 0.5,
    )


def rectangle_contains_point(rectangle: list[float], x: float, y: float) -> bool:
    return rectangle[0] <= x <= rectangle[2] and rectangle[1] <= y <= rectangle[3]


def rectangle_contains_rectangle(
    outer: list[float], inner: list[float]
) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def duplicate_text_geometry(left: list[float], right: list[float]) -> bool:
    return (
        rectangle_iou(left, right) >= DUPLICATE_TEXT_IOU
        or rectangle_overlap_over_smaller(left, right) >= CONTAINED_TEXT_OVERLAP
    )


def dedupe_text_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks.sort(
        key=lambda block: (
            -block["jointConfidence"],
            block["textBounds"][1],
            block["textBounds"][0],
        )
    )
    accepted: list[dict[str, Any]] = []
    for block in blocks:
        if not any(
            duplicate_text_geometry(block["textBounds"], item["textBounds"])
            for item in accepted
        ):
            accepted.append(block)
    return accepted


def split_cut(
    left: list[float], right: list[float], *, vertical: bool
) -> float:
    if vertical:
        if left[3] <= right[1]:
            return (left[3] + right[1]) * 0.5
        return (rectangle_center(left)[1] + rectangle_center(right)[1]) * 0.5
    if left[2] <= right[0]:
        return (left[2] + right[0]) * 0.5
    return (rectangle_center(left)[0] + rectangle_center(right)[0]) * 0.5


def split_connected_bubble(
    bubble_bounds: list[float], blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(blocks) <= 1:
        return [
            {**block, "candidateBounds": list(bubble_bounds)}
            for block in blocks
        ]

    centers = [rectangle_center(block["textBounds"]) for block in blocks]
    average_width = sum(
        block["textBounds"][2] - block["textBounds"][0] for block in blocks
    ) / len(blocks)
    average_height = sum(
        block["textBounds"][3] - block["textBounds"][1] for block in blocks
    ) / len(blocks)
    horizontal_separation = (
        max(center[0] for center in centers) - min(center[0] for center in centers)
    ) / max(average_width, sys.float_info.epsilon)
    vertical_separation = (
        max(center[1] for center in centers) - min(center[1] for center in centers)
    ) / max(average_height, sys.float_info.epsilon)
    vertical = vertical_separation >= horizontal_separation
    blocks.sort(
        key=lambda block: (
            rectangle_center(block["textBounds"])[1 if vertical else 0],
            rectangle_center(block["textBounds"])[0 if vertical else 1],
        )
    )
    cuts = [
        split_cut(
            blocks[index]["textBounds"],
            blocks[index + 1]["textBounds"],
            vertical=vertical,
        )
        for index in range(len(blocks) - 1)
    ]
    candidates: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        candidate_bounds = list(bubble_bounds)
        if vertical:
            candidate_bounds[1] = cuts[index - 1] if index else bubble_bounds[1]
            candidate_bounds[3] = (
                cuts[index] if index < len(cuts) else bubble_bounds[3]
            )
        else:
            candidate_bounds[0] = cuts[index - 1] if index else bubble_bounds[0]
            candidate_bounds[2] = (
                cuts[index] if index < len(cuts) else bubble_bounds[2]
            )
        candidates.append({**block, "candidateBounds": candidate_bounds})
    return candidates


def spatially_dedupe_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    candidates.sort(
        key=lambda candidate: (
            -candidate["jointConfidence"],
            candidate["textBounds"][1],
            candidate["textBounds"][0],
        )
    )
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for candidate in candidates:
        if any(
            duplicate_text_geometry(
                candidate["textBounds"], existing["textBounds"]
            )
            for existing in accepted
        ):
            rejected += 1
            continue
        accepted.append(candidate)
    return accepted, rejected


def dark_pixel_profile(
    source: Image.Image, bounds: list[float]
) -> tuple[int, int]:
    left = max(0, math.floor(bounds[0]))
    top = max(0, math.floor(bounds[1]))
    right = min(source.width, math.ceil(bounds[2]))
    bottom = min(source.height, math.ceil(bounds[3]))
    require(right > left and bottom > top, "candidate crop is empty")
    rgb = source.crop((left, top, right, bottom)).convert("RGB").tobytes()
    dark_pixels = 0
    for offset in range(0, len(rgb), 3):
        red, green, blue = rgb[offset], rgb[offset + 1], rgb[offset + 2]
        luma = (2126 * red + 7152 * green + 722 * blue) // 10_000
        dark_pixels += int(luma <= DARK_CARD_MAX_LUMA)
    return dark_pixels, len(rgb) // 3


def postprocess_detections(
    detections: list[dict[str, Any]], source: Image.Image
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    bubbles = [
        detection
        for detection in detections
        if detection["labelId"] == BUBBLE_LABEL_ID
        and detection["score"] >= RAW_DETECTOR_MINIMUM_SCORE
    ]
    text_detections = [
        detection
        for detection in detections
        if detection["labelId"] == DIALOGUE_TEXT_LABEL_ID
        and detection["score"] >= RAW_DETECTOR_MINIMUM_SCORE
    ]
    grouped: dict[int, list[dict[str, Any]]] = {
        bubble["index"]: [] for bubble in bubbles
    }
    low_confidence_rejections = 0
    not_fully_contained_rejections = 0
    unassociated_text_rejections = 0
    for text in text_detections:
        center_x, center_y = rectangle_center(text["bbox"])
        containing = [
            bubble
            for bubble in bubbles
            if rectangle_contains_point(bubble["bbox"], center_x, center_y)
        ]
        if not containing:
            unassociated_text_rejections += 1
            continue
        bubble = min(
            containing,
            key=lambda item: (
                (item["bbox"][2] - item["bbox"][0])
                * (item["bbox"][3] - item["bbox"][1]),
                item["index"],
            ),
        )
        joint_confidence = min(text["score"], bubble["score"])
        if joint_confidence < MINIMUM_DIALOGUE_JOINT_CONFIDENCE:
            low_confidence_rejections += 1
            continue
        if not rectangle_contains_rectangle(bubble["bbox"], text["bbox"]):
            not_fully_contained_rejections += 1
            continue
        grouped[bubble["index"]].append(
            {
                "bubbleDetectionIndex": bubble["index"],
                "textDetectionIndex": text["index"],
                "bubbleScore": bubble["score"],
                "textScore": text["score"],
                "jointConfidence": joint_confidence,
                "textBounds": list(text["bbox"]),
            }
        )

    candidates: list[dict[str, Any]] = []
    within_bubble_duplicate_rejections = 0
    for bubble in bubbles:
        blocks = grouped[bubble["index"]]
        deduped_blocks = dedupe_text_blocks(blocks)
        within_bubble_duplicate_rejections += len(blocks) - len(deduped_blocks)
        candidates.extend(split_connected_bubble(bubble["bbox"], deduped_blocks))

    spatially_deduped, tile_duplicate_rejections = spatially_dedupe_candidates(
        candidates
    )

    confirmed: list[dict[str, Any]] = []
    dark_card_rejections = 0
    for candidate in spatially_deduped:
        dark_pixels, total_pixels = dark_pixel_profile(
            source, candidate["candidateBounds"]
        )
        if (
            dark_pixels * 100
            >= total_pixels * DARK_CARD_MINIMUM_DARK_PERCENT
        ):
            dark_card_rejections += 1
            continue
        candidate["darkPixelRatio"] = dark_pixels / total_pixels
        confirmed.append(candidate)
    confirmed.sort(
        key=lambda candidate: (
            candidate["textBounds"][1],
            candidate["textBounds"][0],
        )
    )
    for candidate_index, candidate in enumerate(confirmed):
        candidate["index"] = candidate_index
    stats = {
        "rawBubbleDetectionCount": len(bubbles),
        "rawDialogueTextDetectionCount": len(text_detections),
        "lowConfidenceRejectedCount": low_confidence_rejections,
        "notFullyContainedRejectedCount": not_fully_contained_rejections,
        "unassociatedTextRejectedCount": unassociated_text_rejections,
        "withinBubbleDuplicateRejectedCount": within_bubble_duplicate_rejections,
        "tileDuplicateRejectedCount": tile_duplicate_rejections,
        "darkCardRejectedCount": dark_card_rejections,
        "confirmedCandidateCount": len(confirmed),
    }
    return confirmed, stats


def read_detections(
    path: Path,
    *,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], int]:
    document = load_object(path)
    require(document.get("imageWidth") == width, f"{path.name} imageWidth mismatch")
    require(document.get("imageHeight") == height, f"{path.name} imageHeight mismatch")
    detections = document.get("detections")
    require(isinstance(detections, list), f"{path.name}.detections must be an array")
    normalized: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        require(isinstance(detection, dict), f"{path.name} detection {index} is invalid")
        label_id = detection.get("labelId")
        require(type(label_id) is int and label_id >= 0, f"{path.name} labelId is invalid")
        score = finite_number(detection.get("score"), f"{path.name} detection {index} score")
        require(0 <= score <= 1, f"{path.name} detection {index} score is out of range")
        bbox_value = detection.get("bbox")
        require(
            isinstance(bbox_value, list) and len(bbox_value) == 4,
            f"{path.name} detection {index} bbox must have four coordinates",
        )
        bbox = [
            finite_number(value, f"{path.name} detection {index} bbox")
            for value in bbox_value
        ]
        require(
            0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height,
            f"{path.name} detection {index} bbox is outside the source image",
        )
        normalized.append(
            {
                "index": index,
                "labelId": label_id,
                "score": score,
                "bbox": bbox,
            }
        )
    return normalized, len(detections)


def score_page(
    page: dict[str, Any],
    predictions_directory: Path,
    sources_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prediction_name = f"{Path(page['file']).stem}.json"
    prediction_path = predictions_directory / prediction_name
    require(prediction_path.is_file(), f"missing detector prediction: {prediction_path}")
    source_path = sources_directory / page["file"]
    require(source_path.is_file(), f"missing exact source image: {source_path}")
    source_identity = file_identity(source_path, name=page["file"])
    require(
        source_identity["bytes"] == page["bytes"],
        f"{page['file']} byte count does not match the fixture manifest",
    )
    require(
        source_identity["sha256"] == page["sha256"],
        f"{page['file']} SHA-256 does not match the fixture manifest",
    )
    try:
        with Image.open(source_path) as opened:
            opened.load()
            require(
                opened.size == (page["width"], page["height"]),
                f"{page['file']} decoded dimensions do not match the fixture manifest",
            )
            source = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise BenchmarkInputError(f"cannot decode exact source {source_path}: {error}") from error

    detections, raw_detection_count = read_detections(
        prediction_path,
        width=page["width"],
        height=page["height"],
    )
    predictions, postprocessing_stats = postprocess_detections(detections, source)
    gold = [
        {
            **region,
            "bbox": rectangle_for_polygon(
                region["goldPolygon"], page["width"], page["height"]
            ),
        }
        for region in page["regions"]
    ]
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for gold_index, expected in enumerate(gold):
            overlap = rectangle_iou(
                prediction["candidateBounds"], expected["bbox"]
            )
            if overlap >= DEFAULT_MINIMUM_IOU:
                candidates.append((overlap, prediction_index, gold_index))
    candidates.sort(
        key=lambda item: (
            -item[0],
            predictions[item[1]]["index"],
            gold[item[2]]["id"],
        )
    )

    used_predictions: set[int] = set()
    used_gold: set[int] = set()
    matches: list[dict[str, Any]] = []
    for overlap, prediction_index, gold_index in candidates:
        if prediction_index in used_predictions or gold_index in used_gold:
            continue
        used_predictions.add(prediction_index)
        used_gold.add(gold_index)
        prediction = predictions[prediction_index]
        expected = gold[gold_index]
        matches.append(
            {
                "goldRegionId": expected["id"],
                "goldKind": expected["kind"],
                "translationTarget": expected["translationTarget"],
                "goldGeometrySource": expected["goldGeometrySource"],
                "predictionIndex": prediction["index"],
                "bubbleDetectionIndex": prediction["bubbleDetectionIndex"],
                "textDetectionIndex": prediction["textDetectionIndex"],
                "predictionScore": prediction["jointConfidence"],
                "intersectionOverUnion": overlap,
                "goldRegionBounds": expected["bbox"],
                "predictionBounds": prediction["candidateBounds"],
                "textBounds": prediction["textBounds"],
                "darkPixelRatio": prediction["darkPixelRatio"],
            }
        )
    matches.sort(key=lambda match: match["goldRegionId"])

    matched_gold_ids = {match["goldRegionId"] for match in matches}
    matched_prediction_indexes = {match["predictionIndex"] for match in matches}
    prediction_count = len(predictions)
    gold_count = len(gold)
    matched_count = len(matches)
    page_evidence = {
        "page": page["order"],
        "sourceFile": page["file"],
        "predictionFile": prediction_name,
        "goldBubbleCount": gold_count,
        "bubblePredictionCount": prediction_count,
        "matchedBubbleCount": matched_count,
        "falsePositiveCount": prediction_count - matched_count,
        "falseNegativeCount": gold_count - matched_count,
        "precision": matched_count / prediction_count if prediction_count else 0.0,
        "recall": matched_count / gold_count if gold_count else 0.0,
        "matches": matches,
        "falsePositivePredictionIndexes": [
            prediction["index"]
            for prediction in predictions
            if prediction["index"] not in matched_prediction_indexes
        ],
        "missedGoldRegionIds": [
            region["id"] for region in gold if region["id"] not in matched_gold_ids
        ],
    }
    prediction_identity = file_identity(prediction_path, name=prediction_name)
    prediction_identity.update(
        {
            "page": page["order"],
            "rawDetectionCount": raw_detection_count,
            "bubblePredictionCount": prediction_count,
            **postprocessing_stats,
        }
    )
    source_identity.update(
        {
            "page": page["order"],
            "url": page["url"],
            "width": page["width"],
            "height": page["height"],
        }
    )
    return page_evidence, prediction_identity, source_identity


def gate(
    identifier: str,
    actual: float | int,
    *,
    operator: str,
    required: float | int,
) -> dict[str, Any]:
    if operator == "equal":
        passed = actual == required
    elif operator == "atLeast":
        passed = actual >= required
    else:
        raise AssertionError(f"unsupported gate operator: {operator}")
    return {
        "id": identifier,
        "status": "pass" if passed else "fail",
        "actual": actual,
        "operator": operator,
        "required": required,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="directory containing raw detector JSON files named by manifest page",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        required=True,
        help="directory containing exact source files named by the fixture manifest",
    )
    parser.add_argument("--output", type=Path, required=True, help="evidence JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        predictions_directory = args.predictions.resolve()
        sources_directory = args.sources.resolve()
        require(
            predictions_directory.is_dir(),
            f"predictions directory does not exist: {predictions_directory}",
        )
        require(
            sources_directory.is_dir(),
            f"sources directory does not exist: {sources_directory}",
        )
        manifest, gold_pages, fixture_evidence = validate_fixture(
            require_complete_gold=True
        )
        pages: list[dict[str, Any]] = []
        prediction_files: list[dict[str, Any]] = []
        source_files: list[dict[str, Any]] = []
        for page in gold_pages:
            page_evidence, prediction_identity, source_identity = score_page(
                page,
                predictions_directory,
                sources_directory,
            )
            pages.append(page_evidence)
            prediction_files.append(prediction_identity)
            source_files.append(source_identity)

        matches = [match for page in pages for match in page["matches"]]
        prediction_count = sum(page["bubblePredictionCount"] for page in pages)
        gold_count = sum(page["goldBubbleCount"] for page in pages)
        matched_count = len(matches)
        dialogue_gold = sum(
            region["kind"] == "dialogue" for page in gold_pages for region in page["regions"]
        )
        thought_gold = sum(
            region["kind"] == "thought" for page in gold_pages for region in page["regions"]
        )
        punctuation_gold = sum(
            not region["translationTarget"]
            for page in gold_pages
            for region in page["regions"]
        )
        matched_dialogue = sum(match["goldKind"] == "dialogue" for match in matches)
        matched_thought = sum(match["goldKind"] == "thought" for match in matches)
        matched_punctuation = sum(not match["translationTarget"] for match in matches)
        totals = {
            "goldBubbleCount": gold_count,
            "bubblePredictionCount": prediction_count,
            "matchedBubbleCount": matched_count,
            "falsePositiveCount": prediction_count - matched_count,
            "falseNegativeCount": gold_count - matched_count,
            "precision": matched_count / prediction_count if prediction_count else 0.0,
            "recall": matched_count / gold_count if gold_count else 0.0,
            "dialogueGoldCount": dialogue_gold,
            "matchedDialogueCount": matched_dialogue,
            "dialogueRecall": matched_dialogue / dialogue_gold if dialogue_gold else 0.0,
            "thoughtGoldCount": thought_gold,
            "matchedThoughtCount": matched_thought,
            "thoughtRecall": matched_thought / thought_gold if thought_gold else 0.0,
            "punctuationOnlyGoldCount": punctuation_gold,
            "matchedPunctuationOnlyCount": matched_punctuation,
            "punctuationOnlyRecall": (
                matched_punctuation / punctuation_gold if punctuation_gold else 0.0
            ),
        }
        expected_gold_count = manifest["totalExpectedDialogueBubbleCount"]
        gates = [
            gate(
                "all-manifest-gold-bubbles-loaded",
                gold_count,
                operator="equal",
                required=expected_gold_count,
            ),
            gate(
                "speech-thought-bubble-precision",
                totals["precision"],
                operator="atLeast",
                required=0.99,
            ),
            gate(
                "speech-thought-bubble-recall",
                totals["recall"],
                operator="atLeast",
                required=0.95,
            ),
        ]
        status = "pass" if all(gate["status"] == "pass" for gate in gates) else "fail"
        evidence = {
            "schemaVersion": 2,
            "benchmarkId": manifest["id"],
            "status": status,
            "recordedAtUtc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "evidenceSchema": file_identity(
                EVIDENCE_SCHEMA_PATH,
                name="detector-benchmark-evidence.schema.json",
            ),
            "fixture": fixture_evidence,
            "sources": {
                "directory": str(sources_directory),
                "format": "exact source WebP bytes named by fixture manifest",
                "decoder": f"Pillow {PIL.__version__}",
                "files": source_files,
            },
            "predictions": {
                "directory": str(predictions_directory),
                "format": "koharu-ml ComicTextBubbleDetection JSON",
                "bubbleLabelId": BUBBLE_LABEL_ID,
                "dialogueTextLabelId": DIALOGUE_TEXT_LABEL_ID,
                "minimumScore": RAW_DETECTOR_MINIMUM_SCORE,
                "files": prediction_files,
            },
            "postprocessing": {
                "minimumJointConfidence": MINIMUM_DIALOGUE_JOINT_CONFIDENCE,
                "textAssociation": "smallest fully-containing labelId=0 bbox",
                "duplicateTextMinimumIou": DUPLICATE_TEXT_IOU,
                "containedTextMinimumOverlap": CONTAINED_TEXT_OVERLAP,
                "connectedBubbleSplit": (
                    "dominant normalized text-center axis with midpoint cuts"
                ),
                "tileDedupeGeometry": "labelId=1 text bbox",
                "darkCardMaximumLuma": DARK_CARD_MAX_LUMA,
                "darkCardMinimumDarkPixelRatio": DARK_CARD_MINIMUM_DARK_RATIO,
                "luma": "floor((2126*R + 7152*G + 722*B) / 10000)",
            },
            "matcher": {
                "goldGeometry": (
                    "axis-aligned bounds of normalized bubblePolygon when present, "
                    "otherwise normalized textPolygon"
                ),
                "predictionGeometry": (
                    "postprocessed candidate bubble bounds in source pixels"
                ),
                "minimumIou": DEFAULT_MINIMUM_IOU,
                "assignment": "descending-IoU one-to-one greedy assignment per page",
            },
            "protocolBoundary": {
                "browserJobUpdatesRead": False,
                "regionReadyUsed": False,
                "nonAcceptedRegionsPublishedToBrowser": False,
                "evidenceSource": (
                    "standalone detector CLI JSON plus exact manifest source WebPs"
                ),
            },
            "pages": pages,
            "totals": totals,
            "gates": gates,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{status}: matched {matched_count}/{gold_count} gold bubbles from "
            f"{prediction_count} confirmed dialogue candidates; "
            f"precision={totals['precision']:.6f}, recall={totals['recall']:.6f}; "
            f"evidence={args.output.resolve()}"
        )
        return 0 if status == "pass" else 2
    except (BenchmarkInputError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
