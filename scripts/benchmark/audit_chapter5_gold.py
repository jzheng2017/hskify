#!/usr/bin/env python3
"""Render and validate the 30 Years Since the Prologue chapter 5 OCR gold fixture.

This helper performs no OCR or model inference. It creates deterministic visual
audit artifacts from the committed polygons and checks the fixture's schema,
hashes, dimensions, counts, identifiers, and reading-order invariants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont
from jsonschema import Draft202012Validator


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
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / ".cache" / "ocr-benchmark" / "fixture-gold-audit"
)

TILE_WIDTH = 700
TILE_HEIGHT = 540
TILE_COLUMNS = 3
IMAGE_INSET = 18
IMAGE_MAX_WIDTH = TILE_WIDTH - (IMAGE_INSET * 2)
IMAGE_MAX_HEIGHT = 390
CONTEXT_PADDING_PIXELS = 42
DETECTOR_GOLD_KINDS = frozenset(("dialogue", "thought"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def polygon_pixels(
    polygon: Iterable[Iterable[float]], width: int, height: int
) -> list[tuple[int, int]]:
    return [
        (round(float(point[0]) * width), round(float(point[1]) * height))
        for point in polygon
    ]


def bounds_for_polygons(
    polygons: Iterable[Iterable[Iterable[float]]],
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    points = [
        point
        for polygon in polygons
        for point in polygon_pixels(polygon, width, height)
    ]
    left = max(0, min(x for x, _ in points) - padding)
    top = max(0, min(y for _, y in points) - padding)
    right = min(width, max(x for x, _ in points) + padding + 1)
    bottom = min(height, max(y for _, y in points) + padding + 1)
    return left, top, right, bottom


def shifted(
    points: Iterable[tuple[int, int]], left: int, top: int
) -> list[tuple[int, int]]:
    return [(x - left, y - top) for x, y in points]


def fit_inside(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    target = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(target, Image.Resampling.LANCZOS)


def wrap_words(draw: ImageDraw.ImageDraw, text: str, text_font: Any, width: int) -> list[str]:
    output: list[str] = []
    for logical_line in text.splitlines() or [""]:
        words = logical_line.split()
        if not words:
            output.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=text_font)[2] <= width:
                current = candidate
            else:
                output.append(current)
                current = word
        output.append(current)
    return output


def draw_region_tile(
    source: Image.Image,
    region: dict[str, Any],
    page_width: int,
    page_height: int,
) -> tuple[Image.Image, dict[str, Any]]:
    polygons = [
        region["textPolygon"],
        region["eraseMask"]["polygon"],
    ]
    if "bubblePolygon" in region:
        polygons.append(region["bubblePolygon"])
    bounds = bounds_for_polygons(
        polygons, page_width, page_height, CONTEXT_PADDING_PIXELS
    )
    left, top, right, bottom = bounds
    crop = source.crop(bounds).convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    styles = (
        ("bubblePolygon", (25, 90, 255, 225), 3),
        ("textPolygon", (255, 35, 35, 255), 3),
    )
    for key, color, line_width in styles:
        if key not in region:
            continue
        points = shifted(polygon_pixels(region[key], page_width, page_height), left, top)
        overlay_draw.line(points + points[:1], fill=color, width=line_width, joint="curve")
    erase_points = shifted(
        polygon_pixels(region["eraseMask"]["polygon"], page_width, page_height),
        left,
        top,
    )
    overlay_draw.line(
        erase_points + erase_points[:1],
        fill=(0, 160, 65, 225),
        width=2,
        joint="curve",
    )
    crop.alpha_composite(overlay)
    crop = fit_inside(crop.convert("RGB"), IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT)

    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), "white")
    tile_draw = ImageDraw.Draw(tile)
    x = (TILE_WIDTH - crop.width) // 2
    y = IMAGE_INSET + (IMAGE_MAX_HEIGHT - crop.height) // 2
    tile.paste(crop, (x, y))
    tile_draw.rectangle(
        (0, 0, TILE_WIDTH - 1, TILE_HEIGHT - 1),
        outline=(175, 175, 175),
        width=2,
    )

    id_font = font(23, bold=True)
    body_font = font(19)
    label_top = IMAGE_INSET + IMAGE_MAX_HEIGHT + 9
    header = (
        f"{region['id']}  order={region['readingOrder']}  "
        f"{region['kind']}/{region['containerStyle']}"
    )
    tile_draw.text((IMAGE_INSET, label_top), header, fill="black", font=id_font)
    lines = wrap_words(
        tile_draw,
        region["sourceEnglish"].replace("\n", " / "),
        body_font,
        TILE_WIDTH - (IMAGE_INSET * 2),
    )
    line_y = label_top + 31
    for line in lines[:4]:
        tile_draw.text((IMAGE_INSET, line_y), line, fill=(35, 35, 35), font=body_font)
        line_y += 23

    audit_record: dict[str, Any] = {
        "regionId": region["id"],
        "readingOrder": region["readingOrder"],
        "sourceEnglish": region["sourceEnglish"],
        "contextPixelBounds": list(bounds),
        "textPolygonPixelBounds": list(
            bounds_for_polygons(
                [region["textPolygon"]], page_width, page_height, padding=0
            )
        ),
    }
    if "bubblePolygon" in region:
        audit_record["bubblePolygonPixelBounds"] = list(
            bounds_for_polygons(
                [region["bubblePolygon"]], page_width, page_height, padding=0
            )
        )
    return tile, audit_record


def draw_page_overlay(
    source: Image.Image,
    regions: list[dict[str, Any]],
    page_width: int,
    page_height: int,
) -> Image.Image:
    output = source.convert("RGBA")
    draw = ImageDraw.Draw(output)
    label_font = font(16, bold=True)
    for region in regions:
        text_points = polygon_pixels(region["textPolygon"], page_width, page_height)
        bubble_points = (
            polygon_pixels(region["bubblePolygon"], page_width, page_height)
            if "bubblePolygon" in region
            else None
        )
        if bubble_points is not None:
            draw.line(
                bubble_points + bubble_points[:1],
                fill=(25, 90, 255, 240),
                width=3,
                joint="curve",
            )
        draw.line(
            text_points + text_points[:1],
            fill=(255, 35, 35, 255),
            width=3,
            joint="curve",
        )
        anchor_points = bubble_points or text_points
        anchor_x = min(x for x, _ in anchor_points)
        anchor_y = min(y for _, y in anchor_points)
        label = f"{region['readingOrder']:02d}"
        box = draw.textbbox((anchor_x, anchor_y), label, font=label_font, stroke_width=2)
        draw.rectangle(box, fill=(255, 255, 255, 235))
        draw.text(
            (anchor_x, anchor_y),
            label,
            font=label_font,
            fill=(0, 0, 0, 255),
            stroke_width=1,
            stroke_fill=(255, 255, 255, 255),
        )
    return output.convert("RGB")


def render_audit(
    manifest: dict[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    contact_root = output_root / "contact-sheets"
    overlay_root = output_root / "page-overlays"
    crop_root = output_root / "region-crops"
    for directory in (contact_root, overlay_root, crop_root):
        directory.mkdir(parents=True, exist_ok=True)

    audit_pages: list[dict[str, Any]] = []
    for image_entry in manifest["images"]:
        annotation = load_json(FIXTURE_ROOT / image_entry["annotation"])
        page_width = annotation["page"]["width"]
        page_height = annotation["page"]["height"]
        with Image.open(SOURCE_ROOT / image_entry["file"]) as opened:
            source = opened.convert("RGB")

        tiles: list[Image.Image] = []
        audit_regions: list[dict[str, Any]] = []
        for region in annotation["regions"]:
            tile, audit_record = draw_region_tile(
                source, region, page_width, page_height
            )
            tiles.append(tile)
            audit_regions.append(audit_record)
            crop_path = crop_root / f"{region['id']}.png"
            tile.save(crop_path, format="PNG", optimize=False, compress_level=9)

        rows = max(1, math.ceil(len(tiles) / TILE_COLUMNS))
        sheet = Image.new(
            "RGB", (TILE_COLUMNS * TILE_WIDTH, rows * TILE_HEIGHT), (232, 232, 232)
        )
        for index, tile in enumerate(tiles):
            sheet.paste(
                tile,
                (
                    (index % TILE_COLUMNS) * TILE_WIDTH,
                    (index // TILE_COLUMNS) * TILE_HEIGHT,
                ),
            )
        sheet.save(
            contact_root / f"{image_entry['order']:03d}.png",
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        draw_page_overlay(
            source, annotation["regions"], page_width, page_height
        ).save(
            overlay_root / f"{image_entry['order']:03d}.png",
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        audit_pages.append(
            {
                "order": image_entry["order"],
                "pageFile": image_entry["file"],
                "annotationFile": image_entry["annotation"],
                "regions": audit_regions,
            }
        )

    index = {"schemaVersion": 1, "pages": audit_pages}
    (output_root / "audit-index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit_pages


def validate_fixture(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schemaVersion") != 3:
        errors.append("fixture manifest schemaVersion must be 3")
    if manifest.get("id") != "30-years-since-the-prologue-chapter-5":
        errors.append("unexpected benchmark id")

    schema_path = FIXTURE_ROOT / manifest["annotationSchema"]
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema)
    if schema_path.stat().st_size != manifest["annotationSchemaBytes"]:
        errors.append("annotation schema byte count does not match manifest")
    if sha256(schema_path) != manifest["annotationSchemaSha256"]:
        errors.append("annotation schema SHA-256 does not match manifest")

    expected_page_order = list(range(1, manifest["pageCount"] + 1))
    actual_page_order = [entry["order"] for entry in manifest["images"]]
    if actual_page_order != expected_page_order:
        errors.append(
            f"manifest page order {actual_page_order!r} != {expected_page_order!r}"
        )

    source_bytes = sum(entry["bytes"] for entry in manifest["images"])
    source_pixels = sum(
        entry["width"] * entry["height"] for entry in manifest["images"]
    )
    if source_bytes != manifest["totalSourceBytes"]:
        errors.append(f"source bytes {source_bytes} != {manifest['totalSourceBytes']}")
    if source_pixels != manifest["totalSourcePixels"]:
        errors.append(
            f"source pixels {source_pixels} != {manifest['totalSourcePixels']}"
        )

    annotation_status = manifest.get("annotationStatus")
    if not isinstance(annotation_status, dict):
        errors.append("manifest annotationStatus must be an object")
        return errors
    completed_pages = annotation_status.get("completedPageCount")
    required_pages = annotation_status.get("requiredPageCount")
    if (
        annotation_status.get("status") not in {"complete", "incomplete"}
        or annotation_status.get("reviewedPageCount") != manifest["pageCount"]
        or annotation_status.get("generatedPageCount") != manifest["pageCount"]
        or required_pages != manifest["pageCount"]
        or type(completed_pages) is not int
        or not 0 <= completed_pages <= required_pages
    ):
        errors.append("manifest annotationStatus fields are inconsistent")
        return errors

    region_total = 0
    detector_gold_total = 0
    narration_total = 0
    target_total = 0
    exclusion_total = 0
    seen_ids: set[str] = set()
    for image_entry in manifest["images"]:
        source_path = SOURCE_ROOT / image_entry["file"]
        annotation_path = FIXTURE_ROOT / image_entry["annotation"]
        for label, path in (("source", source_path), ("annotation", annotation_path)):
            if not path.is_file():
                errors.append(f"missing {label}: {path}")
        if not source_path.is_file() or not annotation_path.is_file():
            continue

        if source_path.stat().st_size != image_entry["bytes"]:
            errors.append(f"{image_entry['file']}: source byte count mismatch")
        if sha256(source_path) != image_entry["sha256"]:
            errors.append(f"{image_entry['file']}: source SHA-256 mismatch")
        if annotation_path.stat().st_size != image_entry["annotationBytes"]:
            errors.append(f"{image_entry['annotation']}: byte count mismatch")
        if sha256(annotation_path) != image_entry["annotationSha256"]:
            errors.append(f"{image_entry['annotation']}: SHA-256 mismatch")

        annotation = load_json(annotation_path)
        schema_errors = sorted(
            validator.iter_errors(annotation), key=lambda error: list(error.path)
        )
        for error in schema_errors:
            path = ".".join(str(part) for part in error.absolute_path) or "$"
            errors.append(f"{image_entry['annotation']}:{path}: {error.message}")

        with Image.open(source_path) as source:
            dimensions = source.size
        expected_dimensions = (image_entry["width"], image_entry["height"])
        if dimensions != expected_dimensions:
            errors.append(
                f"{image_entry['file']}: dimensions {dimensions} != {expected_dimensions}"
            )
        page = annotation["page"]
        if (page["width"], page["height"]) != expected_dimensions:
            errors.append(f"{image_entry['annotation']}: page dimensions mismatch")
        if page["file"] != image_entry["file"]:
            errors.append(f"{image_entry['annotation']}: source file mismatch")
        if page["sourceSha256"] != image_entry["sha256"]:
            errors.append(f"{image_entry['annotation']}: source SHA-256 mismatch")

        regions = annotation["regions"]
        if len(regions) != image_entry["expectedRegionCount"]:
            errors.append(
                f"{image_entry['annotation']}: {len(regions)} reviewed regions != "
                f"{image_entry['expectedRegionCount']}"
            )
        bubble_regions = [
            region
            for region in regions
            if region.get("kind") in DETECTOR_GOLD_KINDS
        ]
        expected_count = image_entry["expectedDialogueBubbleCount"]
        if len(bubble_regions) != expected_count:
            errors.append(
                f"{image_entry['annotation']}: {len(bubble_regions)} "
                f"dialogue/thought regions != {expected_count}"
            )
        narration_count = sum(
            region.get("kind") == "narration" for region in regions
        )
        if narration_count != image_entry["expectedNarrationCount"]:
            errors.append(
                f"{image_entry['annotation']}: {narration_count} narration regions != "
                f"{image_entry['expectedNarrationCount']}"
            )
        target_count = sum(
            region.get("translationTarget") is not False for region in regions
        )
        exclusion_count = len(regions) - target_count
        if (
            target_count != image_entry["expectedEnglishTranslationTargetCount"]
            or exclusion_count != image_entry["expectedUntouchedExclusionCount"]
        ):
            errors.append(
                f"{image_entry['annotation']}: translation target/exclusion counts mismatch"
            )
        orders = [region["readingOrder"] for region in regions]
        expected_orders = list(range(len(regions)))
        if orders != expected_orders:
            errors.append(
                f"{image_entry['annotation']}: reading order {orders!r} "
                f"!= {expected_orders!r}"
            )
        for index, region in enumerate(regions):
            expected_id = (
                f"30ysp-ch5-p{image_entry['order']:03d}-r{index:02d}"
            )
            if region["id"] != expected_id:
                errors.append(
                    f"{image_entry['annotation']}: region {index} id "
                    f"{region['id']!r} != {expected_id!r}"
                )
            if region["id"] in seen_ids:
                errors.append(f"duplicate region id: {region['id']}")
            seen_ids.add(region["id"])

        region_total += len(regions)
        detector_gold_total += len(bubble_regions)
        narration_total += narration_count
        target_total += target_count
        exclusion_total += exclusion_count

    totals = (
        ("region count", region_total, manifest["totalExpectedRegionCount"]),
        (
            "dialogue/thought detector-gold count",
            detector_gold_total,
            manifest["totalExpectedDialogueBubbleCount"],
        ),
        (
            "narration count",
            narration_total,
            manifest["totalExpectedNarrationCount"],
        ),
        (
            "translation target count",
            target_total,
            manifest["totalExpectedEnglishTranslationTargetCount"],
        ),
        (
            "untouched exclusion count",
            exclusion_total,
            manifest["totalExpectedUntouchedExclusionCount"],
        ),
    )
    for label, actual, expected in totals:
        if actual != expected:
            errors.append(f"{label} {actual} != {expected}")

    for asset in manifest["replicaAssets"]:
        path = FIXTURE_ROOT / asset["path"]
        if not path.is_file():
            errors.append(f"missing replica asset: {asset['path']}")
            continue
        if path.stat().st_size != asset["bytes"]:
            errors.append(f"{asset['path']}: byte count mismatch")
        if sha256(path) != asset["sha256"]:
            errors.append(f"{asset['path']}: SHA-256 mismatch")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="audit artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate without rendering contact sheets or overlays",
    )
    parser.add_argument(
        "--require-complete-gold",
        action="store_true",
        help="fail if any Chinese, pinyin, or HSK gold field remains missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_json(FIXTURE_ROOT / "manifest.json")
    errors = validate_fixture(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    status = manifest["annotationStatus"]
    complete_gold = (
        status["status"] == "complete"
        and status["completedPageCount"] == status["requiredPageCount"]
        and status["totalMissingFieldCount"] == 0
        and status["missingPages"] == []
    )
    if args.require_complete_gold and not complete_gold:
        print(
            "ERROR: release measurement is blocked by incomplete translation gold: "
            f"completedPageCount={status['completedPageCount']}/"
            f"{status['requiredPageCount']}, reasonCode={status['reasonCode']}, "
            f"missingFieldCounts={status['missingFieldCounts']}"
        )
        return 1
    print(
        "Validated "
        f"{manifest['pageCount']} pages and "
        f"{manifest['totalExpectedRegionCount']} reviewed regions "
        f"({manifest['totalExpectedDialogueBubbleCount']} dialogue/thought detector gold, "
        f"{manifest['totalExpectedNarrationCount']} narration)."
    )
    if not complete_gold:
        print(
            "WARNING: structural fixture is valid, but release measurement remains blocked "
            f"by {status['totalMissingFieldCount']} missing translation-gold fields."
        )
    if not args.validate_only:
        pages = render_audit(manifest, args.output.resolve())
        region_count = sum(len(page["regions"]) for page in pages)
        print(
            f"Rendered {region_count} labeled region crops, "
            f"{len(pages)} contact sheets, and {len(pages)} page overlays "
            f"under {args.output.resolve()}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
