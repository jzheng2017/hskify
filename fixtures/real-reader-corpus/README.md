# Local real-reader-v2 regression corpus

The release corpus is a complete, local-only set of ordered chapters. Every
page is stored as a content-addressed object and every page has a tracked,
exhaustive annotation file. Chapter URLs are provenance/discovery metadata
only; no regression command fetches them or uses them as a fallback.

The required layout is:

```text
local-corpus/real-reader-v2/objects/<sha256>.<extension>
fixtures/real-reader-corpus/annotations/<chapter-id>/0001.json
```

`manifest.json` must use schema version 2, list all ten core chapters and
three stress chapters, cover the continuous, paged, iframe, canvas, and WebGL
reader adapters, and declare `completeness.state` as `complete`. Each
page annotation records every story target and exclusion, exact source
English, geometry/reading order, continuation groups, typed entities, style
runs, protected artwork, normalized cleanup-allowance polygons, and reviewed
natural/strict alternatives. The verifier checks annotation bytes, SHA-256, page order,
dimensions, coverage totals, and annotation shape before a daemon is started.

The tracked manifest is a v2 capture contract and currently declares
`capture-required` until the complete authorized local objects and annotations
are supplied. Sparse samples and synthetic renderer regions must never produce
a release-quality green result.

Verify the manifest without reading image objects:

```powershell
node scripts/real-reader-corpus.mjs manifest --json
```

Verify a local selection (network remains forbidden):

```powershell
node scripts/real-reader-corpus.mjs verify --selection core --json
```

Missing or mismatched objects/annotations exit with code 2 and a
machine-readable `requiredPaths` list. The runner never downloads live URLs,
uses sampled pages, or accepts synthetic image fixtures as chapter data.

Once the v2 corpus is present, run the packaged reader regression:

```powershell
node scripts/run-real-reader-regression.mjs --selection core --config <packaged-firefox-driver-config.json>
```

The config is the isolated release-driver record produced by the packaged
Firefox benchmark setup. It identifies the signed extension archive, Firefox,
Playwright, profile directory, extension version, and exact model resource
identities. Missing config or any missing identity fails closed; there is no
daemon-only fallback. The runner serves the verified objects through local
reader replicas, then checks canonical page order, terminal jobs, final-only
DOM publication, patch/resource evidence, and reader adapter coverage. Firefox
reader replicas must resolve their pages through the same local v2 object
store; renderer-only contract fixtures are not pipeline evidence.
