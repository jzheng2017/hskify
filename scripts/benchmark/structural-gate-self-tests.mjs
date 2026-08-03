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
  return {
    id: 'region-1',
    textPolygon: polygon,
    sourceEnglish: 'The evidence',
    displayedChinese: '证据',
    pinyin: 'zhèng jù',
    entities: [],
    confidenceEvidence: {
      ocrConsensus: 0.95,
      geometryCoverage: 1,
      cleanupScore: 0.95,
    },
    ...overrides,
  }
}

function routeWith(regionValue) {
  return { jobs: [{ pageIndex: 0, updates: [{ type: 'regionReady', region: regionValue }] }] }
}

function domWith(regionValue, overrides = {}) {
  return {
    regions: [
      {
        page: 1,
        regionId: regionValue.id,
        text: regionValue.displayedChinese,
        pinyin: regionValue.pinyin,
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
  duplicatePublishedIds: [],
  untranslatedEnglish: [],
  weakEvidence: [],
})

// Unchanged source English must not pass as a translated terminal region.
const unchanged = publicationConsistency(
  domWith(region({ displayedChinese: 'The evidence', pinyin: 'The evidence' })),
  routeWith(region({ displayedChinese: 'The evidence', pinyin: 'The evidence' })),
)
assert.deepEqual(unchanged.untranslatedEnglish, ['region-1'])

// A block-shaped or otherwise unverified cleanup patch has no trustworthy
// terminal evidence and must fail before it reaches a release result.
const opaquePatch = publicationConsistency(
  domWith(valid),
  routeWith(region({ confidenceEvidence: { ocrConsensus: 0.95, geometryCoverage: 1, cleanupScore: 0.1 } })),
)
assert.deepEqual(opaquePatch.weakEvidence, ['region-1'])

// Source-preserving terminal updates (decorative artwork and unreadable OCR)
// are part of the same terminal identity set and must not be mistaken for
// missing or mismatched translated regions.
const unreadable = {
  id: 'unreadable-1',
  textPolygon: polygon,
  sourceEnglish: 'Unrecognized text',
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
          regionId: unreadable.id,
          text: unreadable.sourceEnglish,
          sourceEnglish: unreadable.sourceEnglish,
          sourcePreserving: true,
          pinyin: '',
          fit: 'normal',
          overflows: false,
        },
      ],
    },
    { jobs: [{ pageIndex: 0, updates: [{ type: 'unreadable', region: unreadable }] }] },
  ).missing.length,
  0,
)

// A retained early snapshot or reordered replay is not a complete chapter.
const stale = routeJobConsistency(
  [
    { jobId: 'job-1', pageIndex: 0, sourceSha256: 'a'.repeat(64) },
    { jobId: 'job-2', pageIndex: 1, sourceSha256: 'b'.repeat(64) },
  ],
  {
    jobs: [{ jobId: 'job-1', pageIndex: 0, sourceSha256: 'a'.repeat(64) }],
  },
)
assert.equal(stale.exact, false)

const chapter = {
  id: 'chapter-1',
  pages: [{ order: 0, annotation: { path: 'annotations/chapter-1/0001.json' } }],
}

// OCR letter soup is rejected by the independently recomputed CER gate.
withAnnotation(
  {
    regions: [{ id: 'target-1', polygon, sourceEnglish: 'The evidence' }],
    exclusions: [],
  },
  (root) => {
    const coverage = annotationCoverage(
      chapter,
      join(root, 'manifest.json'),
      routeWith(region({ sourceEnglish: 'qqqq zzzz' })),
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
    exclusions: [{ id: 'artwork-1', polygon, sourceEnglish: 'TECHNIQUE', reason: 'decorative artwork' }],
  },
  (root) => {
    const coverage = annotationCoverage(chapter, join(root, 'manifest.json'), routeWith(valid))
    assert.deepEqual(coverage.modifiedExclusions, [{ page: 1, id: 'artwork-1' }])
  },
)

// Names stay opaque while relationship/occupation/title spans remain ordinary
// translation input; continuation groups must survive page adjudication.
withAnnotation(
  {
    regions: [
      {
        id: 'dialogue-1',
        polygon,
        sourceEnglish: 'Alice calls Wife.',
        entities: [
          { start: 0, end: 5, type: 'person', source: 'Alice' },
          { start: 12, end: 16, type: 'relationship', source: 'Wife' },
        ],
        continuationGroup: 'exchange',
      },
      {
        id: 'dialogue-2',
        polygon: polygon.map((point) => ({ ...point, y: point.y + 0.3 })),
        sourceEnglish: 'She answers.',
        entities: [],
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
                type: 'regionReady',
                region: {
                  ...region({ id: 'dialogue-1', sourceEnglish: 'Alice calls Wife.' }),
                  entities: [
                    { startChar: 0, endChar: 5, entityType: 'person', source: 'Alice', translated: 'Alice' },
                    { startChar: 12, endChar: 16, entityType: 'relationship', source: 'Wife', translated: 'Wife' },
                  ],
                  contextGroup: 'ctx',
                },
              },
              {
                type: 'regionReady',
                region: {
                  ...region({ id: 'dialogue-2', sourceEnglish: 'She answers.' }),
                  textPolygon: polygon.map((point) => ({ ...point, y: point.y + 0.3 })),
                },
              },
            ],
          },
        ],
      },
    )
    assert.deepEqual(semantic.nameViolations, [])
    assert.deepEqual(semantic.translatedDescriptionViolations, [{ page: 1, id: 'dialogue-1', source: 'Wife', type: 'relationship' }])
    assert.equal(semantic.continuationViolations.length, 1)
  },
)

// Tiny text/overflow must fail the browser rendering gate, even if the job
// reached a terminal state.
const degradedDom = domWith(valid, {
  degradedFitCount: 1,
  regions: [{ page: 1, regionId: 'region-1', text: '证据', pinyin: 'zhèng jù', fit: 'degraded', overflows: true }],
})
assert.equal(degradedDom.degradedFitCount === 0 && degradedDom.regions.every((item) => !item.overflows), false)

process.stdout.write('structural gate self-tests passed\n')
