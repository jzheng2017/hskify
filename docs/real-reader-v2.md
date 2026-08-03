# Real-reader v2 release corpus

The browser release gate uses only complete, content-addressed local chapters.
The tracked manifest at `fixtures/real-reader-corpus/manifest.json` is the
capture contract for the ten core chapters and three stress chapters. Until
the image objects and exhaustive annotations are captured, the gate is
intentionally `capture-required` and cannot report a pass.

Each page is stored as `objects/<sha256>.<ext>` below
`local-corpus/real-reader-v2`. An annotation records every readable story
target, exclusion, reading-order/continuation group, entity type, style run,
protected artwork region, cleanup allowance, and reviewed HSK alternatives.
Provenance URLs are discovery metadata only; the release runner never fetches
them and never depends on a live site.

Run the manifest audit with:

```powershell
node scripts/real-reader-corpus.mjs manifest
```

Run the packaged Firefox release qualification with:

```powershell
node scripts/run-real-reader-regression.mjs --selection core
node scripts/run-real-reader-regression.mjs --selection all
```

The browser runner serves a local replica for image, lazy, iframe, canvas,
WebGL/background, continuous-scroll, and paged readers. It checks terminal
job order, final-only DOM publication, patch-before-text ordering, exact
resource replay, OCR/translation coverage, HSK differentials, safe fitting,
and the five-minute chapter limit. Missing corpus objects or annotations are
release failures, not reasons to fall back to synthetic pages or a daemon-only
smoke test.
