# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "numpy==2.5.1",
#   "onnxruntime-gpu[cuda,cudnn]==1.28.0",
#   "Pillow==12.3.0",
#   "psutil==7.2.2",
#   "PyYAML==6.0.3",
#   "nvidia-ml-py==13.610.43",
# ]
# ///
"""Reproducible CUDA OCR benchmark for 30 Years Since the Prologue chapter 5.

The public entry points are intentionally split:

* ``prepare`` validates the frozen corpus and downloads/hash-checks the pinned
  model files. It never creates an ONNX Runtime session.
* ``run`` refuses to execute unless an explicit clearance token is supplied.
  Each model runs in a fresh child process so RAM and VRAM peaks are isolated.

The candidate models are text-line recognizers. The production boundary,
however, supplies one multiline text-block crop per bubble. This harness
therefore uses deterministic, annotation-independent connected-component
grouping to split each gold text-polygon crop into lines. Accuracy is
aggregated over the manifest-declared confident-English OCR targets.
Punctuation-only language-ambiguous regions remain available to detector,
geometry, erase-mask, and segmentation audits but never enter model input.
Gold text is used only for scoring and for reporting the segment-count
diagnostic; it never controls the splitter.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import psutil
import yaml
from PIL import Image, ImageDraw


CLEARANCE_TOKEN = "EXPLICITLY_CLEARED_SERIALIZED_GPU_RUN"
SCHEMA_VERSION = 1
OCR_LINE_BATCH_LIMIT = 8
MODEL_HEIGHT = 48
MODEL_BASE_WIDTH = 320
MODEL_MAX_WIDTH = 3200
SELECTION_CER_LIMIT = 0.03
DETECTOR_GOLD_KINDS = frozenset(("dialogue", "thought"))
SUPPORTED_REGION_KINDS = DETECTOR_GOLD_KINDS | {"narration"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = (
    REPOSITORY_ROOT
    / "fixtures"
    / "benchmarks"
    / "30-years-since-the-prologue-chapter-5"
)
SOURCE_ROOT = (
    REPOSITORY_ROOT
    / ".cache"
    / "benchmarks"
    / "30-years-since-the-prologue-chapter-5"
    / "source"
)
MODEL_CACHE_ROOT = REPOSITORY_ROOT / ".cache" / "ocr-benchmark" / "models"
DEFAULT_EVIDENCE_ROOT = REPOSITORY_ROOT / ".cache" / "ocr-benchmark" / "evidence"


@dataclasses.dataclass(frozen=True)
class ArtifactSpec:
    name: str
    bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    repository: str
    revision: str
    artifacts: tuple[ArtifactSpec, ...]
    expected_classes: int

    @property
    def cache_dir(self) -> Path:
        safe_repo = self.repository.rsplit("/", 1)[-1]
        return MODEL_CACHE_ROOT / f"{safe_repo}-{self.revision}"

    def artifact_path(self, name: str) -> Path:
        return self.cache_dir / name


MODELS: dict[str, ModelSpec] = {
    "ppocrv5-en-mobile": ModelSpec(
        key="ppocrv5-en-mobile",
        display_name="en_PP-OCRv5_mobile_rec ONNX",
        repository="PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx",
        revision="3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b",
        artifacts=(
            ArtifactSpec(
                "inference.onnx",
                7_848_423,
                "b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f",
            ),
            ArtifactSpec(
                "inference.yml",
                3_964,
                "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067",
            ),
        ),
        expected_classes=438,
    ),
    "ppocrv6-small": ModelSpec(
        key="ppocrv6-small",
        display_name="PP-OCRv6_small_rec ONNX",
        repository="PaddlePaddle/PP-OCRv6_small_rec_onnx",
        revision="b8f84f0b80c529de40b4fbb3544b84fa7233a513",
        artifacts=(
            ArtifactSpec(
                "inference.onnx",
                21_159_378,
                "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
            ),
            ArtifactSpec(
                "inference.yml",
                150_579,
                "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1",
            ),
        ),
        expected_classes=18_710,
    ),
}


@dataclasses.dataclass(frozen=True)
class GoldRegion:
    ordinal: int
    id: str
    page_file: str
    page_width: int
    page_height: int
    source_path: Path
    kind: str
    source_english: str
    translation_target: bool
    text_polygon: tuple[tuple[float, float], ...]
    bubble_polygon: tuple[tuple[float, float], ...] | None


@dataclasses.dataclass
class BubbleSample:
    region: GoldRegion
    block_crop: Image.Image
    line_bounds: list[tuple[int, int, int, int]]
    line_crops: list[Image.Image]


@dataclasses.dataclass
class DecodeResult:
    text: str
    confidence: float


@dataclasses.dataclass(frozen=True)
class InkComponent:
    left: int
    top: int
    right: int
    bottom: int
    area: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    def touches_border(self, width: int, height: int) -> bool:
        return (
            self.left == 0
            or self.top == 0
            or self.right == width
            or self.bottom == height
        )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as temp_handle:
        temp_path = Path(temp_handle.name)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "hskify-ocr-benchmark/1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare_model(spec: ModelSpec, allow_download: bool) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for artifact in spec.artifacts:
        path = spec.artifact_path(artifact.name)
        if not path.is_file() and allow_download:
            url = (
                f"https://huggingface.co/{spec.repository}/resolve/"
                f"{spec.revision}/{artifact.name}"
            )
            atomic_download(url, path)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run this script with the `prepare` command"
            )
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != artifact.bytes or actual_sha256 != artifact.sha256:
            raise RuntimeError(
                f"model artifact mismatch for {path}: "
                f"bytes={actual_bytes}, sha256={actual_sha256}"
            )
        records.append(
            {
                "name": artifact.name,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "path": str(path.resolve()),
            }
        )
    return {
        "key": spec.key,
        "displayName": spec.display_name,
        "repository": spec.repository,
        "revision": spec.revision,
        "artifacts": records,
    }


def _validate_polygon(value: Any, label: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) < 4:
        raise ValueError(f"{label} must have at least four points")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{label}[{index}] must be an [x,y] point")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"{label}[{index}] is outside normalized bounds")
        points.append((x, y))
    return tuple(points)


def _is_latin_letter(character: str) -> bool:
    code_point = ord(character)
    return (
        character.isascii()
        and character.isalpha()
        or 0x00C0 <= code_point <= 0x024F
        or 0x1E00 <= code_point <= 0x1EFF
    )


def _is_confident_english_text(text: str) -> bool:
    alphabetic = [character for character in text if character.isalpha()]
    return bool(alphabetic) and all(
        _is_latin_letter(character) for character in alphabetic
    )


def english_translation_targets(regions: Sequence[GoldRegion]) -> list[GoldRegion]:
    return [region for region in regions if region.translation_target]


def load_and_validate_corpus(
    *, require_complete_gold: bool = False
) -> tuple[list[GoldRegion], dict[str, Any]]:
    manifest_path = FIXTURE_ROOT / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schemaVersion") != 3:
        raise RuntimeError("fixture manifest schemaVersion must be 3")
    if manifest.get("id") != "30-years-since-the-prologue-chapter-5":
        raise RuntimeError("unexpected benchmark manifest id")
    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != manifest.get("pageCount"):
        raise RuntimeError(
            "fixture manifest pageCount must equal the images array length"
        )
    annotation_status = manifest.get("annotationStatus")
    if not isinstance(annotation_status, dict):
        raise RuntimeError("fixture manifest annotationStatus must be an object")
    completed_pages = annotation_status.get("completedPageCount")
    required_pages = annotation_status.get("requiredPageCount")
    if (
        annotation_status.get("status") not in {"complete", "incomplete"}
        or annotation_status.get("reviewedPageCount") != manifest.get("pageCount")
        or annotation_status.get("generatedPageCount") != manifest.get("pageCount")
        or required_pages != manifest.get("pageCount")
        or type(completed_pages) is not int
        or not 0 <= completed_pages <= required_pages
    ):
        raise RuntimeError("fixture manifest annotationStatus is inconsistent")
    if require_complete_gold and (
        annotation_status.get("status") != "complete"
        or completed_pages != required_pages
        or annotation_status.get("totalMissingFieldCount") != 0
        or annotation_status.get("missingPages") != []
    ):
        raise RuntimeError(
            "release measurement is blocked because gold fixture is incomplete: "
            f"status={annotation_status.get('status')!r}, "
            f"completedPageCount={completed_pages!r}, "
            f"requiredPageCount={required_pages!r}, "
            f"reasonCode={annotation_status.get('reasonCode')!r}"
        )

    regions: list[GoldRegion] = []
    source_records: list[dict[str, Any]] = []
    annotation_records: list[dict[str, Any]] = []
    ordinal = 0
    for expected_order, image_entry in enumerate(images, start=1):
        if image_entry.get("order") != expected_order:
            raise RuntimeError("manifest page order is not contiguous")
        source_path = SOURCE_ROOT / image_entry["file"]
        if not source_path.is_file():
            raise FileNotFoundError(f"missing ignored source WebP: {source_path}")
        source_bytes = source_path.stat().st_size
        source_hash = sha256_file(source_path)
        if (
            source_bytes != image_entry["bytes"]
            or source_hash != image_entry["sha256"]
        ):
            raise RuntimeError(f"source identity mismatch: {source_path}")

        annotation_path = FIXTURE_ROOT / image_entry["annotation"]
        annotation_bytes = annotation_path.read_bytes()
        annotation_hash = hashlib.sha256(annotation_bytes).hexdigest()
        if (
            len(annotation_bytes) != image_entry["annotationBytes"]
            or annotation_hash != image_entry["annotationSha256"]
        ):
            raise RuntimeError(f"annotation identity mismatch: {annotation_path}")
        annotation = json.loads(annotation_bytes)
        if annotation["page"]["file"] != image_entry["file"]:
            raise RuntimeError(f"annotation page mismatch: {annotation_path}")
        if annotation["page"]["sourceSha256"] != source_hash:
            raise RuntimeError(f"annotation source mismatch: {annotation_path}")
        source_records.append(
            {
                "file": image_entry["file"],
                "bytes": source_bytes,
                "sha256": source_hash,
                "width": image_entry["width"],
                "height": image_entry["height"],
            }
        )
        page_target_count = 0
        page_untouched_count = 0
        page_bubble_count = 0
        page_narration_count = 0
        annotation_regions = annotation["regions"]
        if len(annotation_regions) != image_entry["expectedRegionCount"]:
            raise RuntimeError(
                f"reviewed-region count mismatch: {annotation_path}"
            )
        for region_index, region in enumerate(annotation_regions):
            expected_id = f"30ysp-ch5-p{expected_order:03d}-r{region_index:02d}"
            if region.get("id") != expected_id:
                raise RuntimeError(
                    f"{annotation_path} region id {region.get('id')!r} "
                    f"!= {expected_id!r}"
                )
            kind = region.get("kind")
            if kind not in SUPPORTED_REGION_KINDS:
                raise RuntimeError(f"{region['id']} kind is invalid")
            page_bubble_count += int(kind in DETECTOR_GOLD_KINDS)
            page_narration_count += int(kind == "narration")
            confident_english = _is_confident_english_text(
                region["normalizedEnglish"]
            )
            has_marker = "translationTarget" in region
            if has_marker:
                if region["translationTarget"] is not False or confident_english:
                    raise RuntimeError(
                        f"invalid translationTarget marker: {region['id']}"
                    )
                translation_target = False
                page_untouched_count += 1
            else:
                if not confident_english:
                    raise RuntimeError(
                        f"{region['id']} cannot enter OCR/translation input "
                        "without confident Latin English"
                    )
                translation_target = True
                page_target_count += 1
            regions.append(
                GoldRegion(
                    ordinal=ordinal,
                    id=region["id"],
                    page_file=image_entry["file"],
                    page_width=int(image_entry["width"]),
                    page_height=int(image_entry["height"]),
                    source_path=source_path,
                    kind=kind,
                    source_english=region["sourceEnglish"],
                    translation_target=translation_target,
                    text_polygon=_validate_polygon(
                        region["textPolygon"], f"{region['id']}.textPolygon"
                    ),
                    bubble_polygon=(
                        _validate_polygon(
                            region["bubblePolygon"],
                            f"{region['id']}.bubblePolygon",
                        )
                        if "bubblePolygon" in region
                        else None
                    ),
                )
            )
            ordinal += 1
        if page_bubble_count != image_entry["expectedDialogueBubbleCount"]:
            raise RuntimeError(
                f"dialogue/thought detector-gold count mismatch: {annotation_path}"
            )
        if page_narration_count != image_entry["expectedNarrationCount"]:
            raise RuntimeError(f"narration count mismatch: {annotation_path}")
        if (
            page_target_count != image_entry["expectedEnglishTranslationTargetCount"]
            or page_untouched_count != image_entry["expectedUntouchedExclusionCount"]
        ):
            raise RuntimeError(
                f"translation eligibility count mismatch: {annotation_path}"
            )
        annotation_records.append(
            {
                "file": image_entry["annotation"],
                "bytes": len(annotation_bytes),
                "sha256": annotation_hash,
                "regions": len(annotation_regions),
                "detectedBubbleGoldCount": page_bubble_count,
                "narrationRegionCount": page_narration_count,
                "englishTranslationTargetCount": page_target_count,
                "untouchedExclusionCount": page_untouched_count,
            }
        )

    targets = english_translation_targets(regions)
    untouched_count = len(regions) - len(targets)
    detected_bubble_count = sum(
        region.kind in DETECTOR_GOLD_KINDS for region in regions
    )
    narration_count = sum(region.kind == "narration" for region in regions)
    source_bytes = sum(row["bytes"] for row in source_records)
    source_pixels = sum(
        row["width"] * row["height"] for row in source_records
    )
    if source_bytes != manifest.get("totalSourceBytes"):
        raise RuntimeError(
            f"source bytes {source_bytes} != {manifest.get('totalSourceBytes')}"
        )
    if source_pixels != manifest.get("totalSourcePixels"):
        raise RuntimeError(
            f"source pixels {source_pixels} != {manifest.get('totalSourcePixels')}"
        )
    manifest_totals = (
        ("totalExpectedRegionCount", len(regions)),
        ("totalExpectedDialogueBubbleCount", detected_bubble_count),
        ("totalExpectedNarrationCount", narration_count),
        ("totalExpectedEnglishTranslationTargetCount", len(targets)),
        ("totalExpectedUntouchedExclusionCount", untouched_count),
    )
    for field, actual in manifest_totals:
        expected = manifest.get(field)
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"fixture {field} is {expected}, but annotations contain {actual}"
            )
    return regions, {
        "benchmarkId": manifest["id"],
        "manifestPath": str(manifest_path.resolve()),
        "manifestBytes": len(manifest_bytes),
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pageCount": len(source_records),
        "regionCount": len(regions),
        "bubbleCount": detected_bubble_count,
        "detectedBubbleGoldCount": detected_bubble_count,
        "narrationRegionCount": narration_count,
        "englishTranslationTargetCount": len(targets),
        "untouchedExclusionCount": untouched_count,
        "sourceBytes": source_bytes,
        "sources": source_records,
        "annotations": annotation_records,
    }


def normalized_polygon_bounds(
    region: GoldRegion,
    polygon: Sequence[tuple[float, float]],
    padding: int = 0,
) -> tuple[int, int, int, int]:
    xs = [point[0] * region.page_width for point in polygon]
    ys = [point[1] * region.page_height for point in polygon]
    left = max(0, math.floor(min(xs)) - padding)
    top = max(0, math.floor(min(ys)) - padding)
    right = min(region.page_width, math.ceil(max(xs)) + padding)
    bottom = min(region.page_height, math.ceil(max(ys)) + padding)
    if right <= left or bottom <= top:
        raise RuntimeError(f"empty text crop for {region.id}")
    return left, top, right, bottom


def polygon_crop_bounds(region: GoldRegion, padding: int = 3) -> tuple[int, int, int, int]:
    return normalized_polygon_bounds(region, region.text_polygon, padding)


def otsu_threshold(gray: np.ndarray) -> int:
    values = np.clip(gray, 0, 255).astype(np.uint8, copy=False)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = values.size
    weighted_sum = float(np.dot(np.arange(256, dtype=np.float64), histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 127
    for threshold in range(256):
        count = histogram[threshold]
        background_weight += count
        if background_weight <= 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight <= 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_sum - background_sum) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def foreground_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=bool)
    border = np.concatenate(
        (gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1])
    )
    background = float(np.median(border))
    threshold = otsu_threshold(gray)
    if background >= threshold:
        mask = gray <= min(threshold, int(background) - 6)
    else:
        mask = gray >= max(threshold, int(background) + 6)

    # If the inferred foreground occupies most of the crop, the border was not
    # representative. Pick the lower-occupancy Otsu polarity deterministically.
    if float(mask.mean()) > 0.45:
        dark = gray <= threshold
        light = gray > threshold
        mask = dark if float(dark.mean()) <= float(light.mean()) else light
    return mask


def _active_runs(active: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(active_indices[0])
    previous = start
    for raw_index in active_indices[1:]:
        index = int(raw_index)
        if index - previous - 1 > max_gap:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    return runs


def _connected_ink_components(mask: np.ndarray) -> list[InkComponent]:
    """Return deterministic 8-connected components using row-run union-find."""
    height, _ = mask.shape
    parents: list[int] = []
    ranks: list[int] = []
    records: list[tuple[int, int, int, int]] = []
    previous_runs: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        root = label
        while parents[root] != root:
            root = parents[root]
        while parents[label] != label:
            parent = parents[label]
            parents[label] = root
            label = parent
        return root

    def union(left_label: int, right_label: int) -> None:
        left_root = find(left_label)
        right_root = find(right_label)
        if left_root == right_root:
            return
        if ranks[left_root] < ranks[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        if ranks[left_root] == ranks[right_root]:
            ranks[left_root] += 1

    for y in range(height):
        current_runs: list[tuple[int, int, int]] = []
        for left, right in _active_runs(mask[y, :], max_gap=0):
            label = len(parents)
            parents.append(label)
            ranks.append(0)
            current_runs.append((left, right, label))
            records.append((label, y, left, right))
            for previous_left, previous_right, previous_label in previous_runs:
                # Half-open runs one column apart are diagonally connected.
                if left <= previous_right and previous_left <= right:
                    union(label, previous_label)
        previous_runs = current_runs

    aggregates: dict[int, list[int]] = {}
    for label, y, left, right in records:
        root = find(label)
        aggregate = aggregates.get(root)
        if aggregate is None:
            aggregates[root] = [left, y, right, y + 1, right - left]
        else:
            aggregate[0] = min(aggregate[0], left)
            aggregate[1] = min(aggregate[1], y)
            aggregate[2] = max(aggregate[2], right)
            aggregate[3] = max(aggregate[3], y + 1)
            aggregate[4] += right - left

    return [
        InkComponent(
            left=aggregate[0],
            top=aggregate[1],
            right=aggregate[2],
            bottom=aggregate[3],
            area=aggregate[4],
        )
        for aggregate in sorted(
            aggregates.values(),
            key=lambda value: (value[1], value[0], value[3], value[2]),
        )
    ]


def _component_line_groups(
    mask: np.ndarray,
) -> list[tuple[float, list[InkComponent]]]:
    """Group glyph bodies and their punctuation without any annotations."""
    height, width = mask.shape
    components = [
        component
        for component in _connected_ink_components(mask)
        if not component.touches_border(width, height)
    ]
    if not components:
        return []

    # The upper quartile is stable when a line contains many punctuation dots
    # or apostrophes. Components below 55% of that scale are attached later.
    typical_height = float(
        np.percentile([component.height for component in components], 75)
    )
    minimum_core_height = max(2.0, typical_height * 0.55)
    core_components = [
        component
        for component in components
        if component.height >= minimum_core_height
    ]
    if not core_components:
        core_components = [
            max(components, key=lambda component: (component.area, component.height))
        ]

    core_components.sort(
        key=lambda component: (
            component.center_y,
            component.left,
            component.top,
        )
    )
    maximum_center_gap = max(3.0, typical_height * 0.55)
    groups: list[dict[str, Any]] = []
    for component in core_components:
        if (
            not groups
            or component.center_y - groups[-1]["lastCoreCenter"]
            > maximum_center_gap
        ):
            groups.append(
                {
                    "core": [component],
                    "components": [component],
                    "lastCoreCenter": component.center_y,
                }
            )
        else:
            groups[-1]["core"].append(component)
            groups[-1]["components"].append(component)
            groups[-1]["lastCoreCenter"] = component.center_y

    core_ids = {id(component) for component in core_components}
    maximum_vertical_gap = max(2.0, typical_height * 0.50)
    maximum_horizontal_gap = max(2.0, typical_height * 0.75)
    for component in components:
        if id(component) in core_ids:
            continue
        candidates: list[tuple[float, float, float, int]] = []
        for index, group in enumerate(groups):
            group_components = group["components"]
            group_left = min(item.left for item in group_components)
            group_top = min(item.top for item in group_components)
            group_right = max(item.right for item in group_components)
            group_bottom = max(item.bottom for item in group_components)
            group_center = float(
                np.median([item.center_y for item in group["core"]])
            )
            vertical_gap = max(
                0, group_top - component.bottom, component.top - group_bottom
            )
            horizontal_gap = max(
                0, group_left - component.right, component.left - group_right
            )
            candidates.append(
                (
                    float(vertical_gap),
                    abs(component.center_y - group_center),
                    float(horizontal_gap),
                    index,
                )
            )
        vertical_gap, _, horizontal_gap, target_index = min(candidates)
        if (
            vertical_gap <= maximum_vertical_gap
            and horizontal_gap <= maximum_horizontal_gap
        ):
            groups[target_index]["components"].append(component)

    return [
        (
            float(np.median([item.center_y for item in group["core"]])),
            sorted(
                group["components"],
                key=lambda component: (
                    component.top,
                    component.left,
                    component.bottom,
                    component.right,
                ),
            ),
        )
        for group in groups
    ]


def segment_text_line_bounds(
    block_crop: Image.Image,
) -> list[tuple[int, int, int, int]]:
    """Find line crop bounds without consulting gold text or line count."""
    rgb = block_crop.convert("RGB")
    width, height = rgb.size
    if width <= 1 or height <= 1:
        return [(0, 0, width, height)]
    mask = foreground_mask(rgb)
    groups = _component_line_groups(mask)
    if not groups:
        return [(0, 0, width, height)]

    pad_y = max(2, round(height * 0.018))
    pad_x = max(2, round(width * 0.012))
    row_ink = mask.sum(axis=1)
    boundaries: list[int] = []
    for (upper_center, _), (lower_center, _) in zip(groups, groups[1:]):
        search_top = max(0, math.floor(upper_center))
        search_bottom = min(height - 1, math.ceil(lower_center))
        if search_bottom <= search_top:
            boundary = min(height, search_top + 1)
        else:
            window = row_ink[search_top : search_bottom + 1]
            minimum = int(window.min())
            minima = np.flatnonzero(window == minimum)
            boundary = search_top + int(minima[len(minima) // 2])
            boundary = max(1, min(height - 1, boundary))
        boundaries.append(boundary)

    bounds: list[tuple[int, int, int, int]] = []
    for index, (_, components) in enumerate(groups):
        left = max(0, min(component.left for component in components) - pad_x)
        right = min(
            width, max(component.right for component in components) + pad_x
        )
        top = max(0, min(component.top for component in components) - pad_y)
        bottom = min(
            height, max(component.bottom for component in components) + pad_y
        )
        if index > 0:
            top = max(top, boundaries[index - 1])
        if index < len(boundaries):
            bottom = min(bottom, boundaries[index])
        if right > left and bottom > top:
            bounds.append((left, top, right, bottom))
    return bounds or [(0, 0, width, height)]


def segment_text_lines(block_crop: Image.Image) -> list[Image.Image]:
    """Split a text-block crop without consulting gold text or line count."""
    rgb = block_crop.convert("RGB")
    return [rgb.crop(bounds) for bounds in segment_text_line_bounds(rgb)]


def build_bubble_samples(regions: Sequence[GoldRegion]) -> list[BubbleSample]:
    page_cache: dict[Path, Image.Image] = {}
    samples: list[BubbleSample] = []
    try:
        for region in regions:
            page = page_cache.get(region.source_path)
            if page is None:
                with Image.open(region.source_path) as opened:
                    page = opened.convert("RGB")
                if page.size != (region.page_width, region.page_height):
                    raise RuntimeError(f"decoded size mismatch: {region.source_path}")
                page_cache[region.source_path] = page
            block = page.crop(polygon_crop_bounds(region))
            line_bounds = segment_text_line_bounds(block)
            samples.append(
                BubbleSample(
                    region=region,
                    block_crop=block,
                    line_bounds=line_bounds,
                    line_crops=[block.crop(bounds) for bounds in line_bounds],
                )
            )
    finally:
        for page in page_cache.values():
            page.close()
    return samples


def segmentation_summary(samples: Sequence[BubbleSample]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    matches = 0
    for sample in samples:
        gold_lines = len(sample.region.source_english.splitlines())
        detected_lines = len(sample.line_crops)
        if gold_lines == detected_lines:
            matches += 1
        rows.append(
            {
                "id": sample.region.id,
                "goldLineCountDiagnosticOnly": gold_lines,
                "detectedLineCount": detected_lines,
                "matches": gold_lines == detected_lines,
                "cropWidth": sample.block_crop.width,
                "cropHeight": sample.block_crop.height,
                "lineBounds": [list(bounds) for bounds in sample.line_bounds],
            }
        )
    return {
        "policy": {
            "crop": "axis-aligned textPolygon bounds expanded by 3 source pixels",
            "splitter": (
                "Otsu foreground mask plus 8-connected component line grouping"
            ),
            "borderInkPolicy": (
                "discard components connected to the multiline crop boundary"
            ),
            "punctuationPolicy": (
                "attach nearby small components to robust-height line groups"
            ),
            "usesGoldTextForSegmentation": False,
            "join": "top-to-bottom line predictions joined with one ASCII space",
        },
        "bubbleCount": len(samples),
        "detectedLineCount": sum(len(sample.line_crops) for sample in samples),
        "goldLineCountDiagnosticOnly": sum(
            len(sample.region.source_english.splitlines()) for sample in samples
        ),
        "bubbleLineCountMatches": matches,
        "bubbleLineCountMismatches": len(samples) - matches,
        "regions": rows,
    }


def write_inspection_montage(
    samples: Sequence[BubbleSample],
    destination: Path,
    limit: int | None = None,
) -> None:
    rows: list[Image.Image] = []
    selected = samples if limit is None else samples[:limit]
    for sample in selected:
        source_label = canonical_text(sample.region.source_english)
        label_height = 28
        scaled_block = sample.block_crop.copy()
        if scaled_block.width > 480:
            scale = 480 / scaled_block.width
            scaled_block = scaled_block.resize(
                (480, max(1, round(scaled_block.height * scale))),
                Image.Resampling.BILINEAR,
            )
        line_width = sum(line.width for line in sample.line_crops) + max(
            0, len(sample.line_crops) - 1
        ) * 4
        row_width = max(700, scaled_block.width, line_width)
        line_height = max((line.height for line in sample.line_crops), default=1)
        row_height = label_height + scaled_block.height + 6 + line_height + 10
        row = Image.new("RGB", (row_width, row_height), "white")
        draw = ImageDraw.Draw(row)
        draw.text(
            (4, 4),
            f"{sample.region.id} [{len(sample.line_crops)}] {source_label}",
            fill="black",
        )
        row.paste(scaled_block, (4, label_height))
        x = 4
        y = label_height + scaled_block.height + 6
        for line in sample.line_crops:
            row.paste(line, (x, y))
            x += line.width + 4
        rows.append(row)
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    montage = Image.new("RGB", (width, height), "#d0d0d0")
    y = 0
    for row in rows:
        montage.paste(row, (0, y))
        y += row.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    montage.save(destination, format="PNG", optimize=True)


def _save_audit_png(image: Image.Image, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=False, compress_level=9)
    return {
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "width": image.width,
        "height": image.height,
    }


def _segmentation_record_from_evidence(
    payload: dict[str, Any],
) -> dict[str, Any]:
    direct = payload.get("segmentation")
    if isinstance(direct, dict) and "detectedLineCount" in direct:
        return direct
    models = payload.get("models")
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            segmentation = model.get("segmentation")
            if (
                isinstance(segmentation, dict)
                and "detectedLineCount" in segmentation
            ):
                return segmentation
    raise ValueError("evidence does not contain a segmentation record")


def write_segmentation_audit(
    destination: Path,
    baseline_evidence: Path | None,
) -> Path:
    """Write image-only line crops plus explicitly diagnostic gold comparison."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"segmentation audit destination is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    regions, corpus = load_and_validate_corpus()
    samples = build_bubble_samples(regions)
    summary = segmentation_summary(samples)
    line_crop_records: list[dict[str, Any]] = []
    line_crop_root = destination / "line-crops"
    for sample in samples:
        artifacts = []
        for index, (bounds, crop) in enumerate(
            zip(sample.line_bounds, sample.line_crops, strict=True)
        ):
            artifact = _save_audit_png(
                crop,
                line_crop_root / sample.region.id / f"{index:02d}.png",
            )
            artifact["boundsWithinBlockCrop"] = list(bounds)
            artifacts.append(artifact)
        line_crop_records.append(
            {
                "id": sample.region.id,
                "blockCropWidth": sample.block_crop.width,
                "blockCropHeight": sample.block_crop.height,
                "detectedLineCount": len(sample.line_crops),
                "lines": artifacts,
            }
        )

    montage_path = destination / "inspection.png"
    write_inspection_montage(samples, montage_path)
    montage = {
        "path": str(montage_path.resolve()),
        "bytes": montage_path.stat().st_size,
        "sha256": sha256_file(montage_path),
    }

    baseline_comparison: dict[str, Any] | None = None
    if baseline_evidence is not None:
        baseline_bytes = baseline_evidence.read_bytes()
        baseline_payload = json.loads(baseline_bytes)
        baseline = _segmentation_record_from_evidence(baseline_payload)
        baseline_by_id = {
            row["id"]: row
            for row in baseline.get("regions", [])
            if isinstance(row, dict) and "id" in row
        }
        current_by_id = {row["id"]: row for row in summary["regions"]}
        changed_regions = []
        for region_id in sorted(set(baseline_by_id) & set(current_by_id)):
            before = int(baseline_by_id[region_id]["detectedLineCount"])
            after = int(current_by_id[region_id]["detectedLineCount"])
            if before != after:
                changed_regions.append(
                    {
                        "id": region_id,
                        "beforeDetectedLineCount": before,
                        "afterDetectedLineCount": after,
                        "delta": after - before,
                    }
                )
        baseline_comparison = {
            "evidence": {
                "path": str(baseline_evidence.resolve()),
                "bytes": len(baseline_bytes),
                "sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            },
            "goldAnnotationsUsedOnlyForOfflineDiagnostics": True,
            "before": {
                key: baseline.get(key)
                for key in (
                    "bubbleCount",
                    "detectedLineCount",
                    "goldLineCountDiagnosticOnly",
                    "bubbleLineCountMatches",
                    "bubbleLineCountMismatches",
                )
            },
            "after": {
                key: summary[key]
                for key in (
                    "bubbleCount",
                    "detectedLineCount",
                    "goldLineCountDiagnosticOnly",
                    "bubbleLineCountMatches",
                    "bubbleLineCountMismatches",
                )
            },
            "changedRegionCount": len(changed_regions),
            "changedRegions": changed_regions,
        }

    report = {
        "schemaVersion": 1,
        "kind": "ocr-line-segmentation-audit",
        "createdAtUtc": utc_now(),
        "repositoryRevision": repository_revision(),
        "corpus": corpus,
        "modelRunsExecuted": False,
        "onnxRuntimeSessionCreated": False,
        "inferenceInputs": ["multiline block crop pixels"],
        "annotationPolicy": {
            "usesSourceEnglishForSegmentation": False,
            "usesGoldLineCountForSegmentation": False,
            "goldAnnotationsUsedOnlyForOfflineDiagnostics": True,
            "inspectionMontageLabelsAreAuditOnly": True,
        },
        "segmentation": summary,
        "baselineComparison": baseline_comparison,
        "inspectionMontage": montage,
        "lineCrops": line_crop_records,
    }
    report_path = destination / "segmentation-audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    inventory = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    (destination / "sha256-inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report_path


def _polygon_as_json(
    polygon: Sequence[tuple[float, float]],
) -> list[list[float]]:
    return [[x, y] for x, y in polygon]


def audit_fixture_pair(
    destination: Path, left_id: str, right_id: str
) -> Path:
    """Preserve a manually identified geometry/text swap as exact evidence."""
    regions, corpus = load_and_validate_corpus()
    by_id = {region.id: region for region in regions}
    if left_id == right_id:
        raise RuntimeError("fixture audit region IDs must be distinct")
    try:
        left = by_id[left_id]
        right = by_id[right_id]
    except KeyError as error:
        raise RuntimeError(f"unknown fixture audit region ID: {error.args[0]}") from error
    if left.page_file != right.page_file:
        raise RuntimeError("known mismatch pair unexpectedly spans pages")
    if canonical_text(left.source_english) == canonical_text(right.source_english):
        raise RuntimeError("known mismatch references unexpectedly became equal")
    if left.bubble_polygon is None or right.bubble_polygon is None:
        raise RuntimeError(
            "fixture-pair bubble audit requires bubblePolygon on both selected regions"
        )

    with Image.open(left.source_path) as opened:
        page = opened.convert("RGB")
    try:
        records: list[dict[str, Any]] = []
        for region, observed_from in ((left, right), (right, left)):
            text_bounds = normalized_polygon_bounds(
                region, region.text_polygon, padding=3
            )
            bubble_bounds = normalized_polygon_bounds(
                region, region.bubble_polygon, padding=3
            )
            text_crop = page.crop(text_bounds)
            bubble_crop = page.crop(bubble_bounds)

            union_left = max(0, min(text_bounds[0], bubble_bounds[0]) - 24)
            union_top = max(0, min(text_bounds[1], bubble_bounds[1]) - 24)
            union_right = min(
                region.page_width, max(text_bounds[2], bubble_bounds[2]) + 24
            )
            union_bottom = min(
                region.page_height, max(text_bounds[3], bubble_bounds[3]) + 24
            )
            context_bounds = (
                union_left,
                union_top,
                union_right,
                union_bottom,
            )
            context = page.crop(context_bounds)
            draw = ImageDraw.Draw(context)

            def local_points(
                polygon: Sequence[tuple[float, float]],
            ) -> list[tuple[float, float]]:
                return [
                    (
                        x * region.page_width - union_left,
                        y * region.page_height - union_top,
                    )
                    for x, y in polygon
                ]

            text_points = local_points(region.text_polygon)
            bubble_points = local_points(region.bubble_polygon)
            draw.line(
                [*text_points, text_points[0]], fill="#ff0000", width=2
            )
            draw.line(
                [*bubble_points, bubble_points[0]], fill="#0080ff", width=2
            )

            prefix = destination / region.id
            artifacts = {
                "textPolygonCrop": _save_audit_png(
                    text_crop, prefix.with_suffix(".text.png")
                ),
                "bubblePolygonCrop": _save_audit_png(
                    bubble_crop, prefix.with_suffix(".bubble.png")
                ),
                "contextOverlay": _save_audit_png(
                    context, prefix.with_suffix(".context.png")
                ),
            }
            records.append(
                {
                    "regionId": region.id,
                    "pageFile": region.page_file,
                    "expectedSourceEnglish": region.source_english,
                    "observedTextManualTranscription": observed_from.source_english,
                    "observedTextMatchesExpectedRegionId": observed_from.id,
                    "normalizedExpected": canonical_text(region.source_english),
                    "normalizedObserved": canonical_text(
                        observed_from.source_english
                    ),
                    "textPolygon": _polygon_as_json(region.text_polygon),
                    "bubblePolygon": _polygon_as_json(region.bubble_polygon),
                    "textPolygonPixelBoundsWith3pxPadding": list(text_bounds),
                    "bubblePolygonPixelBoundsWith3pxPadding": list(
                        bubble_bounds
                    ),
                    "contextPixelBounds": list(context_bounds),
                    "artifacts": artifacts,
                }
            )
    finally:
        page.close()

    annotation_record = next(
        row
        for row in corpus["annotations"]
        if row["file"] == "annotations/001.json"
    )
    source_record = next(
        row for row in corpus["sources"] if row["file"] == "001.webp"
    )
    report = {
        "schemaVersion": 1,
        "kind": "ocr-gold-fixture-mismatch",
        "createdAtUtc": utc_now(),
        "repositoryRevision": repository_revision(),
        "benchmarkId": corpus["benchmarkId"],
        "goldInvalidForCer": True,
        "modelRunsExecuted": False,
        "reason": (
            "The expected sourceEnglish and both OCR geometry polygons are "
            "cross-associated between two different balloons. A model would "
            "therefore be scored against text that is not present in its crop."
        ),
        "page": source_record,
        "annotation": annotation_record,
        "swap": {
            "regionIds": [left_id, right_id],
            "relationship": (
                f"{left_id} crop contains {right_id} expected text; "
                f"{right_id} crop contains {left_id} expected text"
            ),
        },
        "overlayLegend": {
            "red": "textPolygon",
            "blue": "bubblePolygon",
        },
        "mismatches": records,
        "blockingImpact": {
            "detectedBubbleGoldCount": sum(
                region.kind in DETECTOR_GOLD_KINDS for region in regions
            ),
            "requiredEnglishOcrTargetCount": len(
                english_translation_targets(regions)
            ),
            "validEnglishOcrTargetCountForCer": None,
            "perModelCer": None,
            "selectionThreshold": SELECTION_CER_LIMIT,
            "selectionPermitted": False,
        },
    }
    report_path = destination / "mismatch-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    inventory = []
    for path in sorted(destination.glob("*")):
        if path.is_file():
            inventory.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    inventory_path = destination / "sha256-inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def load_characters(spec: ModelSpec) -> list[str]:
    config = yaml.safe_load(
        spec.artifact_path("inference.yml").read_text(encoding="utf-8")
    )
    characters = ["blank", *config["PostProcess"]["character_dict"]]
    # Paddle's inference configuration keeps ``use_space_char`` outside the
    # serialized PostProcess mapping. Both pinned exports contain one output
    # class beyond blank + character_dict, which is the appended ASCII space.
    if len(characters) + 1 == spec.expected_classes:
        characters.append(" ")
    if len(characters) != spec.expected_classes:
        raise RuntimeError(
            f"{spec.key} character count {len(characters)} "
            f"does not match ONNX output {spec.expected_classes}"
        )
    return characters


def preprocess_line_batch(images: Sequence[Image.Image]) -> np.ndarray:
    if not images or len(images) > OCR_LINE_BATCH_LIMIT:
        raise ValueError("line batch must contain 1..8 images")
    max_ratio = max(
        MODEL_BASE_WIDTH / MODEL_HEIGHT,
        *(image.width / max(1, image.height) for image in images),
    )
    target_width = min(MODEL_MAX_WIDTH, max(MODEL_BASE_WIDTH, int(MODEL_HEIGHT * max_ratio)))
    batch = np.zeros(
        (len(images), 3, MODEL_HEIGHT, target_width), dtype=np.float32
    )
    for index, image in enumerate(images):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = rgb[:, :, ::-1]
        ratio = bgr.shape[1] / float(max(1, bgr.shape[0]))
        resized_width = min(target_width, max(1, math.ceil(MODEL_HEIGHT * ratio)))
        resized = Image.fromarray(bgr, mode="RGB").resize(
            (resized_width, MODEL_HEIGHT), Image.Resampling.BILINEAR
        )
        normalized = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)
        normalized = normalized / 255.0
        normalized -= 0.5
        normalized /= 0.5
        batch[index, :, :, :resized_width] = normalized
    return np.ascontiguousarray(batch)


def decode_ctc(probabilities: np.ndarray, characters: Sequence[str]) -> list[DecodeResult]:
    if probabilities.ndim != 3 or probabilities.shape[2] != len(characters):
        raise ValueError(
            f"unexpected CTC output shape {probabilities.shape}; "
            f"character count is {len(characters)}"
        )
    token_ids = probabilities.argmax(axis=2)
    token_probabilities = probabilities.max(axis=2)
    results: list[DecodeResult] = []
    for ids, scores in zip(token_ids, token_probabilities, strict=True):
        selected_text: list[str] = []
        selected_scores: list[float] = []
        previous = -1
        for token_id, score in zip(ids.tolist(), scores.tolist(), strict=True):
            if token_id != 0 and token_id != previous:
                selected_text.append(characters[token_id])
                selected_scores.append(float(score))
            previous = token_id
        results.append(
            DecodeResult(
                text="".join(selected_text),
                confidence=(
                    float(sum(selected_scores) / len(selected_scores))
                    if selected_scores
                    else 0.0
                ),
            )
        )
    return results


def canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def edit_distance(reference: str, prediction: str) -> int:
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, prediction_character in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (reference_character != prediction_character),
                )
            )
        previous = current
    return previous[-1]


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "p50": nearest_rank(values, 0.50),
        "p95": nearest_rank(values, 0.95),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss = 0
        self.peak_private = 0
        self.peak_device_vram = 0
        self.peak_process_vram: int | None = None
        self.sample_count = 0
        self.nvml: Any | None = None
        self.handle: Any | None = None
        self.gpu_name: str | None = None
        self.driver_version: str | None = None
        self.total_vram: int | None = None
        self._init_nvml()
        baseline_memory = self.process.memory_info()
        self.baseline_rss = baseline_memory.rss
        self.baseline_private = getattr(baseline_memory, "private", None)
        self.baseline_device_vram = self._device_vram()
        self.baseline_process_vram = self._process_vram()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(self.handle)
            raw_driver = pynvml.nvmlSystemGetDriverVersion()
            self.gpu_name = (
                raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            )
            self.driver_version = (
                raw_driver.decode()
                if isinstance(raw_driver, bytes)
                else str(raw_driver)
            )
            self.total_vram = int(
                pynvml.nvmlDeviceGetMemoryInfo(self.handle).total
            )
        except Exception:
            self.nvml = None
            self.handle = None

    def _device_vram(self) -> int | None:
        if self.nvml is None or self.handle is None:
            return None
        try:
            return int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
        except Exception:
            return None

    def _process_vram(self) -> int | None:
        if self.nvml is None or self.handle is None:
            return None
        processes: list[Any] = []
        for name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            query = getattr(self.nvml, name, None)
            if query is None:
                continue
            try:
                processes.extend(query(self.handle))
            except Exception:
                continue
        used = 0
        found = False
        unavailable = getattr(self.nvml, "NVML_VALUE_NOT_AVAILABLE", None)
        for process in processes:
            if int(process.pid) != os.getpid():
                continue
            value = getattr(process, "usedGpuMemory", None)
            if value is None or value == unavailable:
                continue
            used += int(value)
            found = True
        return used if found else None

    def _sample(self) -> None:
        memory = self.process.memory_info()
        self.peak_rss = max(self.peak_rss, memory.rss)
        private = getattr(memory, "private", None)
        if private is not None:
            self.peak_private = max(self.peak_private, int(private))
        device_vram = self._device_vram()
        if device_vram is not None:
            self.peak_device_vram = max(self.peak_device_vram, device_vram)
        process_vram = self._process_vram()
        if process_vram is not None:
            self.peak_process_vram = max(self.peak_process_vram or 0, process_vram)
        self.sample_count += 1

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval_seconds)
        self._sample()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        final_memory = self.process.memory_info()
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        return {
            "sampleIntervalMs": self.interval_seconds * 1000,
            "sampleCount": self.sample_count,
            "processRssBaselineBytes": self.baseline_rss,
            "processRssPeakBytes": self.peak_rss,
            "processRssPeakDeltaBytes": max(
                0, self.peak_rss - self.baseline_rss
            ),
            "processPrivateBytesBaseline": self.baseline_private,
            "processPrivateBytesPeak": self.peak_private or None,
            "processPrivateBytesPeakDelta": (
                max(0, self.peak_private - self.baseline_private)
                if self.baseline_private is not None and self.peak_private
                else None
            ),
            "processPrivateBytesOsPeak": getattr(
                final_memory, "peak_pagefile", None
            ),
            "deviceVramBaselineBytes": self.baseline_device_vram,
            "deviceVramPeakBytes": self.peak_device_vram or None,
            "deviceVramPeakDeltaBytes": (
                max(0, self.peak_device_vram - self.baseline_device_vram)
                if self.baseline_device_vram is not None
                and self.peak_device_vram
                else None
            ),
            "processVramBaselineBytes": self.baseline_process_vram,
            "processVramPeakBytes": self.peak_process_vram,
            "gpu": {
                "name": self.gpu_name,
                "driverVersion": self.driver_version,
                "totalVramBytes": self.total_vram,
            },
        }


def run_line_batch(
    session: Any,
    images: Sequence[Image.Image],
    characters: Sequence[str],
) -> tuple[list[DecodeResult], dict[str, float]]:
    started = time.perf_counter_ns()
    tensor = preprocess_line_batch(images)
    preprocessed = time.perf_counter_ns()
    output = session.run(["fetch_name_0"], {"x": tensor})[0]
    inferred = time.perf_counter_ns()
    decoded = decode_ctc(output, characters)
    ended = time.perf_counter_ns()
    scale = 1e-6
    return decoded, {
        "preprocessMs": (preprocessed - started) * scale,
        "inferenceMs": (inferred - preprocessed) * scale,
        "postprocessMs": (ended - inferred) * scale,
        "endToEndMs": (ended - started) * scale,
    }


def recognize_bubbles(
    session: Any,
    samples: Sequence[BubbleSample],
    characters: Sequence[str],
    resegment: bool,
) -> tuple[list[DecodeResult], dict[str, float]]:
    started = time.perf_counter_ns()
    line_groups = [
        segment_text_lines(sample.block_crop)
        if resegment
        else sample.line_crops
        for sample in samples
    ]
    segmented = time.perf_counter_ns()
    flat_lines = [line for group in line_groups for line in group]
    flat_predictions: list[DecodeResult] = []
    preprocess_ms = 0.0
    inference_ms = 0.0
    postprocess_ms = 0.0
    for offset in range(0, len(flat_lines), OCR_LINE_BATCH_LIMIT):
        decoded, timings = run_line_batch(
            session,
            flat_lines[offset : offset + OCR_LINE_BATCH_LIMIT],
            characters,
        )
        flat_predictions.extend(decoded)
        preprocess_ms += timings["preprocessMs"]
        inference_ms += timings["inferenceMs"]
        postprocess_ms += timings["postprocessMs"]
    results: list[DecodeResult] = []
    prediction_offset = 0
    for group in line_groups:
        predictions = flat_predictions[
            prediction_offset : prediction_offset + len(group)
        ]
        prediction_offset += len(group)
        results.append(
            DecodeResult(
                text=" ".join(prediction.text for prediction in predictions),
                confidence=(
                    sum(prediction.confidence for prediction in predictions)
                    / len(predictions)
                    if predictions
                    else 0.0
                ),
            )
        )
    ended = time.perf_counter_ns()
    return results, {
        "segmentationMs": (segmented - started) * 1e-6,
        "preprocessMs": preprocess_ms,
        "inferenceMs": inference_ms,
        "postprocessMs": postprocess_ms,
        "endToEndMs": (ended - started) * 1e-6,
        "lineCount": len(flat_lines),
        "modelInvocationCount": math.ceil(
            len(flat_lines) / OCR_LINE_BATCH_LIMIT
        ),
    }


def accuracy_evidence(
    session: Any,
    samples: Sequence[BubbleSample],
    characters: Sequence[str],
) -> dict[str, Any]:
    predictions: list[DecodeResult] = []
    accuracy_timings: list[dict[str, float]] = []
    for offset in range(0, len(samples), OCR_LINE_BATCH_LIMIT):
        decoded, timings = recognize_bubbles(
            session,
            samples[offset : offset + OCR_LINE_BATCH_LIMIT],
            characters,
            resegment=False,
        )
        predictions.extend(decoded)
        accuracy_timings.append(timings)
    if len(predictions) != len(samples):
        raise RuntimeError(
            f"OCR returned {len(predictions)} of "
            f"{len(samples)} English targets"
        )

    rows: list[dict[str, Any]] = []
    total_distance = 0
    total_reference_characters = 0
    folded_distance = 0
    exact = 0
    for sample, prediction in zip(samples, predictions, strict=True):
        reference = canonical_text(sample.region.source_english)
        predicted = canonical_text(prediction.text)
        distance = edit_distance(reference, predicted)
        casefold_distance = edit_distance(reference.casefold(), predicted.casefold())
        reference_characters = len(reference)
        total_distance += distance
        folded_distance += casefold_distance
        total_reference_characters += reference_characters
        exact += int(reference == predicted)
        rows.append(
            {
                "id": sample.region.id,
                "reference": reference,
                "prediction": predicted,
                "confidence": prediction.confidence,
                "editDistance": distance,
                "referenceCharacters": reference_characters,
                "cer": distance / reference_characters,
                "caseInsensitiveEditDistance": casefold_distance,
                "exactMatch": reference == predicted,
                "detectedLineCount": len(sample.line_crops),
                "goldLineCountDiagnosticOnly": len(
                    sample.region.source_english.splitlines()
                ),
            }
        )
    return {
        "metric": {
            "name": "micro character error rate",
            "reference": "sourceEnglish",
            "normalization": "Unicode NFKC plus collapse all whitespace runs to one ASCII space; case and punctuation preserved",
            "formula": "sum Levenshtein edit distance / sum normalized reference characters",
            "selectionThreshold": SELECTION_CER_LIMIT,
        },
        "bubbleCount": len(rows),
        "referenceCharacters": total_reference_characters,
        "editDistance": total_distance,
        "cer": total_distance / total_reference_characters,
        "caseInsensitiveEditDistance": folded_distance,
        "caseInsensitiveCer": folded_distance / total_reference_characters,
        "exactMatchBubbles": exact,
        "accuracyPassTimings": accuracy_timings,
        "regions": rows,
    }


def exact_batches(
    items: Sequence[Any], batch_size: int, count: int, start: int = 0
) -> Iterable[list[Any]]:
    for batch_index in range(count):
        offset = start + batch_index * batch_size
        yield [
            items[(offset + item_index) % len(items)]
            for item_index in range(batch_size)
        ]


def line_latency_evidence(
    session: Any,
    samples: Sequence[BubbleSample],
    characters: Sequence[str],
    warmups: int,
    measured_samples: int,
) -> list[dict[str, Any]]:
    lines = [line for sample in samples for line in sample.line_crops]
    records: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4, 8):
        for batch in exact_batches(lines, batch_size, warmups):
            run_line_batch(session, batch, characters)
        timings: list[dict[str, float]] = []
        for batch in exact_batches(
            lines, batch_size, measured_samples, start=warmups * batch_size
        ):
            _, sample_timing = run_line_batch(session, batch, characters)
            timings.append(sample_timing)
        total_inference_seconds = (
            sum(sample["inferenceMs"] for sample in timings) / 1000
        )
        records.append(
            {
                "batchSize": batch_size,
                "warmups": warmups,
                "samples": measured_samples,
                "preprocessMs": latency_summary(
                    [sample["preprocessMs"] for sample in timings]
                ),
                "inferenceMs": latency_summary(
                    [sample["inferenceMs"] for sample in timings]
                ),
                "postprocessMs": latency_summary(
                    [sample["postprocessMs"] for sample in timings]
                ),
                "endToEndMs": latency_summary(
                    [sample["endToEndMs"] for sample in timings]
                ),
                "inferenceThroughputLinesPerSecond": (
                    batch_size * measured_samples / total_inference_seconds
                ),
                "raw": timings,
            }
        )
    return records


def bubble_latency_evidence(
    session: Any,
    samples: Sequence[BubbleSample],
    characters: Sequence[str],
    warmups: int,
    measured_samples: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4, 8):
        for batch in exact_batches(samples, batch_size, warmups):
            recognize_bubbles(session, batch, characters, resegment=True)
        timings: list[dict[str, float]] = []
        for batch in exact_batches(
            samples,
            batch_size,
            measured_samples,
            start=warmups * batch_size,
        ):
            _, sample_timing = recognize_bubbles(
                session, batch, characters, resegment=True
            )
            timings.append(sample_timing)
        total_seconds = sum(sample["endToEndMs"] for sample in timings) / 1000
        records.append(
            {
                "batchSize": batch_size,
                "warmups": warmups,
                "samples": measured_samples,
                "segmentationMs": latency_summary(
                    [sample["segmentationMs"] for sample in timings]
                ),
                "preprocessMs": latency_summary(
                    [sample["preprocessMs"] for sample in timings]
                ),
                "inferenceMs": latency_summary(
                    [sample["inferenceMs"] for sample in timings]
                ),
                "postprocessMs": latency_summary(
                    [sample["postprocessMs"] for sample in timings]
                ),
                "endToEndMs": latency_summary(
                    [sample["endToEndMs"] for sample in timings]
                ),
                "endToEndThroughputBubblesPerSecond": (
                    batch_size * measured_samples / total_seconds
                ),
                "raw": timings,
            }
        )
    return records


def package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "onnxruntime-gpu",
        "nvidia-cublas",
        "nvidia-cuda-nvrtc",
        "nvidia-cuda-runtime",
        "nvidia-cudnn-cu13",
        "nvidia-cufft",
        "nvidia-curand",
        "nvidia-nvjitlink",
        "Pillow",
        "psutil",
        "PyYAML",
        "nvidia-ml-py",
    )
    return {name: importlib.metadata.version(name) for name in names}


def repository_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_worker(
    model_key: str,
    output: Path,
    warmups: int,
    samples_count: int,
    clearance: str,
) -> None:
    if clearance != CLEARANCE_TOKEN:
        raise RuntimeError(
            "GPU inference refused: explicit serialized-run clearance is absent"
        )
    spec = MODELS[model_key]
    model_identity = prepare_model(spec, allow_download=False)
    regions, corpus_identity = load_and_validate_corpus(require_complete_gold=True)
    bubbles = build_bubble_samples(english_translation_targets(regions))
    splitter_evidence = segmentation_summary(bubbles)
    characters = load_characters(spec)

    import onnxruntime as ort

    ort.preload_dlls(directory="")
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            f"onnxruntime-gpu lacks CUDAExecutionProvider: {available_providers}"
        )
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True

    sampler = ResourceSampler()
    sampler.start()
    started_at = utc_now()
    try:
        session_started = time.perf_counter_ns()
        session = ort.InferenceSession(
            str(spec.artifact_path("inference.onnx")),
            sess_options=session_options,
            providers=[
                ("CUDAExecutionProvider", {"device_id": "0"}),
                "CPUExecutionProvider",
            ],
        )
        session_load_ms = (time.perf_counter_ns() - session_started) * 1e-6
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"CUDA is not the primary provider: {session.get_providers()}"
            )
        output_shape = session.get_outputs()[0].shape
        if output_shape[-1] != spec.expected_classes:
            raise RuntimeError(
                f"unexpected output class dimension: {output_shape}"
            )

        accuracy = accuracy_evidence(session, bubbles, characters)
        line_latency = line_latency_evidence(
            session, bubbles, characters, warmups, samples_count
        )
        bubble_latency = bubble_latency_evidence(
            session, bubbles, characters, warmups, samples_count
        )
        ended_at = utc_now()
        evidence = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "ocr-model-worker",
            "startedAtUtc": started_at,
            "endedAtUtc": ended_at,
            "repositoryRoot": str(REPOSITORY_ROOT),
            "repositoryRevision": repository_revision(),
            "corpus": corpus_identity,
            "model": model_identity,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": package_versions(),
                "onnxRuntimeAvailableProviders": available_providers,
                "sessionProviders": session.get_providers(),
                "sessionProviderOptions": session.get_provider_options(),
                "sessionLoadMs": session_load_ms,
                "input": {
                    "name": session.get_inputs()[0].name,
                    "shape": session.get_inputs()[0].shape,
                    "type": session.get_inputs()[0].type,
                },
                "output": {
                    "name": session.get_outputs()[0].name,
                    "shape": output_shape,
                    "type": session.get_outputs()[0].type,
                },
            },
            "segmentation": splitter_evidence,
            "accuracy": accuracy,
            "latency": {
                "recognizerLineBatches": line_latency,
                "productionContractBubbleBatches": bubble_latency,
                "lineBatchLimit": OCR_LINE_BATCH_LIMIT,
            },
        }
    finally:
        resources = sampler.stop()
        gc.collect()
    evidence["resources"] = resources
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_parent(
    output: Path,
    model_keys: Sequence[str],
    warmups: int,
    samples_count: int,
    clearance: str,
) -> None:
    if clearance != CLEARANCE_TOKEN:
        raise RuntimeError(
            "GPU inference refused. The caller must obtain explicit clearance "
            f"and pass --gpu-clearance {CLEARANCE_TOKEN}"
        )
    regions, corpus_identity = load_and_validate_corpus(require_complete_gold=True)
    del regions
    for key in model_keys:
        prepare_model(MODELS[key], allow_download=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    worker_dir = output.parent / (output.stem + ".workers")
    worker_dir.mkdir(parents=True, exist_ok=True)

    worker_evidence: list[dict[str, Any]] = []
    for key in model_keys:
        worker_output = worker_dir / f"{key}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model",
            key,
            "--output",
            str(worker_output),
            "--warmups",
            str(warmups),
            "--samples",
            str(samples_count),
            "--gpu-clearance",
            clearance,
        ]
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        worker_evidence.append(json.loads(worker_output.read_text(encoding="utf-8")))

    qualifying = [
        row["model"]["key"]
        for row in worker_evidence
        if row["accuracy"]["cer"] <= SELECTION_CER_LIMIT
    ]
    combined = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "ocr-model-comparison",
        "createdAtUtc": utc_now(),
        "repositoryRoot": str(REPOSITORY_ROOT),
        "repositoryRevision": repository_revision(),
        "corpus": corpus_identity,
        "selection": {
            "maximumCer": SELECTION_CER_LIMIT,
            "qualifyingModels": qualifying,
            "selectedModel": (
                min(
                    (
                        row
                        for row in worker_evidence
                        if row["model"]["key"] in qualifying
                    ),
                    key=lambda row: row["accuracy"]["cer"],
                )["model"]["key"]
                if qualifying
                else None
            ),
        },
        "models": worker_evidence,
    }
    output.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())


def prepare_command(inspect_montage: Path | None) -> None:
    regions, corpus = load_and_validate_corpus()
    model_records = [
        prepare_model(spec, allow_download=True) for spec in MODELS.values()
    ]
    bubbles = build_bubble_samples(english_translation_targets(regions))
    segmentation = segmentation_summary(bubbles)
    if inspect_montage is not None:
        write_inspection_montage(bubbles, inspect_montage)
    print(
        json.dumps(
            {
                "corpus": corpus,
                "models": model_records,
                "segmentation": {
                    key: value
                    for key, value in segmentation.items()
                    if key != "regions"
                },
                "inspectionMontage": (
                    str(inspect_montage.resolve())
                    if inspect_montage is not None
                    else None
                ),
                "gpuModelExecuted": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def assert_synthetic_segmentation_cases() -> None:
    blank = Image.new("RGB", (80, 40), "white")
    assert segment_text_line_bounds(blank) == [(0, 0, 80, 40)]

    punctuation = Image.new("RGB", (60, 70), "white")
    punctuation_draw = ImageDraw.Draw(punctuation)
    punctuation_draw.rectangle((24, 10, 34, 37), fill="black")
    punctuation_draw.ellipse((24, 47, 34, 57), fill="black")
    assert len(segment_text_lines(punctuation)) == 1

    # These line components overlap in horizontal projection, but their glyph
    # centers remain distinct. The long first-line component is a descender.
    close_lines = Image.new("RGB", (120, 55), "white")
    close_draw = ImageDraw.Draw(close_lines)
    close_draw.rectangle((8, 8, 33, 24), fill="black")
    close_draw.rectangle((40, 8, 48, 31), fill="black")
    close_draw.rectangle((65, 26, 95, 42), fill="black")
    close_draw.rectangle((100, 26, 110, 42), fill="black")
    close_bounds = segment_text_line_bounds(close_lines)
    assert len(close_bounds) == 2
    assert close_bounds == segment_text_line_bounds(close_lines)

    decorated = Image.new("RGB", (120, 70), "white")
    decorated_draw = ImageDraw.Draw(decorated)
    decorated_draw.line((0, 0, 24, 0), fill="black", width=2)
    decorated_draw.line((95, 69, 119, 69), fill="black", width=2)
    decorated_draw.line((0, 5, 0, 15), fill="black", width=2)
    decorated_draw.rectangle((15, 13, 100, 25), fill="black")
    decorated_draw.rectangle((20, 43, 105, 55), fill="black")
    assert len(segment_text_lines(decorated)) == 2


def segmentation_self_test() -> None:
    assert_synthetic_segmentation_cases()
    regions, _ = load_and_validate_corpus()
    samples = build_bubble_samples(regions)
    assert all(
        sample.line_bounds == segment_text_line_bounds(sample.block_crop)
        for sample in samples
    )
    print(
        "segmentation self-test passed "
        f"({sum(len(sample.line_crops) for sample in samples)} fixture lines; "
        "no ONNX session created)"
    )


def self_test() -> None:
    assert canonical_text("  A\nB  ") == "A B"
    assert edit_distance("kitten", "sitting") == 3
    characters = ["blank", "A", "B", " "]
    probabilities = np.zeros((1, 7, 4), dtype=np.float32)
    for column, token in enumerate((1, 1, 0, 2, 3, 3, 0)):
        probabilities[0, column, token] = 1.0
    decoded = decode_ctc(probabilities, characters)
    assert decoded[0].text == "AB "
    synthetic = Image.new("RGB", (120, 60), "white")
    draw = ImageDraw.Draw(synthetic)
    draw.rectangle((10, 8, 100, 20), fill="black")
    draw.rectangle((20, 38, 110, 50), fill="black")
    assert len(segment_text_lines(synthetic)) == 2
    assert_synthetic_segmentation_cases()
    for spec in MODELS.values():
        identity = prepare_model(spec, allow_download=False)
        assert len(identity["artifacts"]) == 2
        assert len(load_characters(spec)) == spec.expected_classes
    print("self-test passed (no ONNX session created)")


def default_output_path() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_EVIDENCE_ROOT / stamp / "comparison.json"


def default_segmentation_audit_path() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_EVIDENCE_ROOT / f"segmentation-audit-{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="validate/download only; never execute a model"
    )
    prepare_parser.add_argument("--inspect-montage", type=Path)

    subparsers.add_parser(
        "self-test", help="test scoring/preprocessing without creating an ONNX session"
    )

    subparsers.add_parser(
        "segmentation-self-test",
        help="test synthetic and audited fixture segmentation; never load a model",
    )

    segmentation_audit_parser = subparsers.add_parser(
        "audit-segmentation",
        help="write segmentation-only line-crop evidence; never load a model",
    )
    segmentation_audit_parser.add_argument("--output-directory", type=Path)
    segmentation_audit_parser.add_argument(
        "--baseline-evidence",
        type=Path,
        help=(
            "optional prior JSON used only for offline before/after diagnostics"
        ),
    )

    audit_parser = subparsers.add_parser(
        "audit-fixture",
        help="preserve a manually identified gold crop/reference mismatch",
    )
    audit_parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPOSITORY_ROOT
        / ".cache"
        / "ocr-benchmark"
        / "fixture-audit",
    )
    audit_parser.add_argument(
        "--region-id",
        action="append",
        required=True,
        help="mismatched region ID; pass exactly twice",
    )

    run_parser = subparsers.add_parser(
        "run", help="run candidates sequentially in isolated CUDA worker processes"
    )
    run_parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODELS),
        dest="models",
        help="candidate to run; repeat for multiple (default: both)",
    )
    run_parser.add_argument("--output", type=Path, default=None)
    run_parser.add_argument("--warmups", type=int, default=10)
    run_parser.add_argument("--samples", type=int, default=50)
    run_parser.add_argument("--gpu-clearance", required=True)

    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--model", choices=tuple(MODELS), required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.add_argument("--warmups", type=int, required=True)
    worker_parser.add_argument("--samples", type=int, required=True)
    worker_parser.add_argument("--gpu-clearance", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_command(args.inspect_montage)
    elif args.command == "self-test":
        self_test()
    elif args.command == "segmentation-self-test":
        segmentation_self_test()
    elif args.command == "audit-segmentation":
        output_directory = (
            args.output_directory or default_segmentation_audit_path()
        )
        print(
            write_segmentation_audit(
                output_directory.resolve(),
                (
                    args.baseline_evidence.resolve()
                    if args.baseline_evidence is not None
                    else None
                ),
            )
        )
    elif args.command == "audit-fixture":
        if len(args.region_id) != 2:
            raise ValueError("--region-id must be passed exactly twice")
        print(
            audit_fixture_pair(
                args.output_directory.resolve(),
                args.region_id[0],
                args.region_id[1],
            )
        )
    elif args.command == "run":
        if args.warmups < 1 or args.samples < 1:
            raise ValueError("warmups and samples must both be positive")
        run_parent(
            output=(args.output or default_output_path()).resolve(),
            model_keys=args.models or tuple(MODELS),
            warmups=args.warmups,
            samples_count=args.samples,
            clearance=args.gpu_clearance,
        )
    elif args.command == "worker":
        run_worker(
            model_key=args.model,
            output=args.output.resolve(),
            warmups=args.warmups,
            samples_count=args.samples,
            clearance=args.gpu_clearance,
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
