# 30 Years Since the Prologue chapter 5 benchmark fixture

This is the sole canonical Hskify benchmark fixture for the Asura series
"30 Years Since the Prologue", chapter 5.

The 36 exact source WebP files are intentionally ignored and must exist at:

```text
.cache/benchmarks/30-years-since-the-prologue-chapter-5/source
```

`manifest.json` freezes the image order, byte sizes, decoded dimensions, and
SHA-256 hashes. Source image URLs are omitted because neither the retained
source files nor the audit script record them; no URLs were inferred.

`replica/index.html` references the ignored source directory and eagerly
decodes all 36 images before setting `data-benchmark-ready="true"` on the
document element. Serve the replica from the repository root so its absolute
source paths resolve.

## Annotation status

All 36 pages have reviewed annotation documents. The inclusion review is
authoritative: all 158 accepted detector proposals and all 60 manually noted
detector misses are included, while the five rejected proposals are excluded.
The generated files use deterministic page-local IDs and reading order.

Regenerate them with:

```powershell
python scripts/benchmark/build_chapter5_annotations.py
```

The generator reads the ignored independent review and the committed
`geometry-corrections.json`. That correction source replaces all 60 deliberately
approximate `missedStoryTextRegion` rectangles with source-pixel bounds verified
against the original pages: 57 are unions of recorded PP-OCRv5 detections and
three punctuation-only regions use visually measured glyph bounds. Corrected
misses do not inherit bubble contours or erase patches from product output; the
verified text polygon is the detector-gold fallback and its three-pixel expansion
is the erase mask.

The generator may also copy translations from uniquely geometry-matched daemon
output and use raw detector output for bubble geometry on other review proposal
types. It never invents Chinese, pinyin, or HSK tokens.

The region inventory and translation gold are complete. `manifest.json`
records exact global and per-page missing-field counts. The machine-readable
status is:

```json
{
  "annotationStatus": {
    "status": "complete",
    "reasonCode": "all-gold-fields-present",
    "reviewedPageCount": 36,
    "generatedPageCount": 36,
    "completedPageCount": 36,
    "requiredPageCount": 36,
    "missingFieldCounts": {
      "simplifiedChinese": 0,
      "pinyin": 0,
      "hskTokens": 0
    }
  }
}
```

`annotation.schema.json` reserves region IDs in the form
`30ysp-ch5-p001-r00`. Translation fields are optional at the schema level so
reviewed geometry can be committed without false data; manifest completeness
is the acceptance gate.

Release tooling rejects the fixture if `annotationStatus.status` is not
`complete` or any required field becomes missing.
