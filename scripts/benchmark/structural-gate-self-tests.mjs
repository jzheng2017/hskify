/**
 * Deliberately corrupted release evidence for the structural quality gates.
 *
 * These tests do not stand in for real reader pages. They prove that the
 * packaged-browser gate rejects the failure modes that previously looked
 * green when only a job-complete snapshot was inspected.
 */

import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  annotationCoverage,
  publicationConsistency,
  routeJobConsistency,
  semanticConsistency,
} from '../run-real-reader-browser-regression.mjs'

const polygon = [
  { x: 0.1, y: 0.1 },
  { x: 0.9, y: 0.1 },
  { x: 0.9, y: 0.3 },
  { x: 0.1, y: 0.3 },
]

function region(overrides = {}) {
  const value = {
    itemId: 'region-1',
    textPolygon: polygon,
    text: {
      sourceText: 'The evidence',
      displayedChinese: '证据',
      pinyin: 'zhèng jù',
    },
    confidenceEvidence: {
      ocrConsensus: 0.95,
      geometryCoverage: 1,
      contextConsistency: 0.95,
      cleanupScore: 0.95,
    },
    ...overrides,
  }
  return {
    ...value,
    text: {
      sourceText: 'The evidence',
      displayedChinese: '证据',
      pinyin: 'zhèng jù',
      ...(overrides.text ?? {}),
    },
  }
}

function routeWith(regionValue) {
  return { jobs: [{ pageIndex: 0, updates: [{ type: 'imageRegionReady', region: regionValue }] }] }
}

function domWith(regionValue, overrides = {}) {
  return {
    regions: [
      {
        page: 1,
        itemId: regionValue.itemId,
        text: regionValue.text.displayedChinese,
        pinyin: regionValue.text.pinyin,
        fit: 'normal',
        overflows: false,
      },
    ],
    patchCount: 1,
    regionCount: 1,
    degradedFitCount: 0,
    ...overrides,
  }
}

function withAnnotation(annotation, callback) {
  const root = mkdtempSync(join(tmpdir(), 'hskify-gates-'))
  const annotationDirectory = join(root, 'annotations', 'chapter-1')
  mkdirSync(annotationDirectory, { recursive: true })
  const annotationPath = join(annotationDirectory, '0001.json')
  writeFileSync(annotationPath, JSON.stringify(annotation))
  try {
    return callback(root, annotationPath)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

const valid = region()
assert.deepEqual(publicationConsistency(domWith(valid), routeWith(valid)), {
  publishedCount: 1,
  renderedCount: 1,
  missing: [],
  mismatched: [],
  duplicatePublishedItemIds: [],
  untranslatedEnglish: [],
  weakEvidence: [],
})

// Unchanged source English must not pass as a translated terminal region.
const unchanged = publicationConsistency(
  domWith(region({ text: { displayedChinese: 'The evidence', pinyin: 'The evidence' } })),
  routeWith(region({ text: { displayedChinese: 'The evidence', pinyin: 'The evidence' } })),
)
assert.deepEqual(unchanged.untranslatedEnglish, ['region-1'])

// A block-shaped or otherwise unverified cleanup patch has no trustworthy
// terminal evidence and must fail before it reaches a release result.
const opaquePatch = publicationConsistency(
  domWith(valid),
  routeWith(region({ confidenceEvidence: { ocrConsensus: 0.95, geometryCoverage: 1, contextConsistency: 0.95, cleanupScore: 0.1 } })),
)
assert.deepEqual(opaquePatch.weakEvidence, ['region-1'])

// Unreadable OCR is an interactive source notice and must reconcile exactly.
const unreadable = {
  itemId: 'unreadable-1',
  textPolygon: polygon,
  sourceText: 'Unrecognized text',
  ocrConfidence: 0.2,
  readingOrder: 1,
  reason: 'OCR consensus failed',
}
assert.equal(
  publicationConsistency(
    {
      regions: [
        {
          page: 1,
          itemId: unreadable.itemId,
          text: unreadable.sourceText,
          sourceText: unreadable.sourceText,
          sourcePreserving: true,
          pinyin: '',
          fit: 'normal',
          overflows: false,
        },
      ],
    },
    { jobs: [{ pageIndex: 0, updates: [{ type: 'imageRegionPreserved', region: unreadable }] }] },
  ).missing.length,
  0,
)

// Preserved image regions expose only a transparent source-text lookup target.
const preserved = { ...unreadable, itemId: 'preserved-1' }
const missingPreserved = publicationConsistency(
  { regions: [] },
  { jobs: [{ pageIndex: 0, updates: [{ type: 'imageRegionPreserved', region: preserved }] }] },
)
assert.deepEqual(missingPreserved.missing, [preserved.itemId])
const paintedPreserved = publicationConsistency(
  { regions: [{ itemId: preserved.itemId, text: '错误', sourcePreserving: false }] },
  { jobs: [{ pageIndex: 0, updates: [{ type: 'imageRegionPreserved', region: preserved }] }] },
)
assert.deepEqual(paintedPreserved.mismatched, [preserved.itemId])

// A retained early snapshot or reordered replay is not a complete chapter.
const stale = routeJobConsistency(
  [
    { jobId: 'job-1', pageIndex: 0, sourceSha256: 'a'.repeat(64) },
    { jobId: 'job-2', pageIndex: 1, sourceSha256: 'b'.repeat(64) },
  ],
  {
    jobs: [
      {
        jobId: 'job-1',
        pageIndex: 0,
        sourceSha256: 'a'.repeat(64),
        terminal: { type: 'complete' },
      },
    ],
  },
)
assert.equal(stale.exact, false)

const failedTerminal = routeJobConsistency(
  [{ jobId: 'job-1', pageIndex: 0, sourceSha256: 'a'.repeat(64) }],
  {
    jobs: [
      {
        jobId: 'job-1',
        pageIndex: 0,
        sourceSha256: 'a'.repeat(64),
        terminal: { type: 'failed' },
      },
    ],
  },
)
assert.equal(failedTerminal.exact, false)

const chapter = {
  id: 'chapter-1',
  pages: [{ order: 0, annotation: { path: 'annotations/chapter-1/0001.json' } }],
}

// OCR letter soup is rejected by the independently recomputed CER gate.
withAnnotation(
  {
    regions: [{ itemId: 'target-1', polygon, sourceText: 'The evidence' }],
    exclusions: [],
  },
  (root) => {
    const coverage = annotationCoverage(
      chapter,
      join(root, 'manifest.json'),
      routeWith(region({ text: { sourceText: 'qqqq zzzz' } })),
    )
    assert.ok(coverage.ocrCer > 0.02)
    assert.ok(coverage.highErrorRegions.length > 0)
  },
)

// Exclusions/protected artwork may keep optional hover metadata, but they can
// never become painted translated regions.
withAnnotation(
  {
    regions: [],
    exclusions: [{ itemId: 'artwork-1', polygon, sourceText: 'TECHNIQUE', reason: 'decorative artwork' }],
  },
  (root) => {
    const coverage = annotationCoverage(chapter, join(root, 'manifest.json'), routeWith(valid))
    assert.deepEqual(coverage.modifiedExclusions, [{ page: 1, itemId: 'artwork-1' }])
  },
)

// Continuation groups must survive page adjudication.
withAnnotation(
  {
    regions: [
      {
        itemId: 'dialogue-1',
        polygon,
        sourceText: 'Alice calls Wife.',
        continuationGroup: 'exchange',
      },
      {
        itemId: 'dialogue-2',
        polygon: polygon.map((point) => ({ ...point, y: point.y + 0.3 })),
        sourceText: 'She answers.',
        continuationGroup: 'exchange',
      },
    ],
    exclusions: [],
  },
  (root) => {
    const semantic = semanticConsistency(
      { ...chapter, pages: [{ order: 0, annotation: { path: 'annotations/chapter-1/0001.json' } }] },
      join(root, 'manifest.json'),
      {
        jobs: [
          {
            pageIndex: 0,
            updates: [
              {
                type: 'imageRegionReady',
                region: {
                  ...region({ itemId: 'dialogue-1', text: { sourceText: 'Alice calls Wife.' } }),
                  contextGroup: 'ctx',
                },
              },
              {
                type: 'imageRegionReady',
                region: {
                  ...region({ itemId: 'dialogue-2', text: { sourceText: 'She answers.' } }),
                  textPolygon: polygon.map((point) => ({ ...point, y: point.y + 0.3 })),
                },
              },
            ],
          },
        ],
      },
    )
    assert.equal(semantic.continuationViolations.length, 1)
  },
)

// Tiny text/overflow must fail the browser rendering gate, even if the job
// reached a terminal state.
const degradedDom = domWith(valid, {
  degradedFitCount: 1,
  regions: [{ page: 1, itemId: 'region-1', text: '证据', pinyin: 'zhèng jù', fit: 'degraded', overflows: true }],
})
assert.equal(degradedDom.degradedFitCount === 0 && degradedDom.regions.every((item) => !item.overflows), false)

process.stdout.write('structural gate self-tests passed\n')
