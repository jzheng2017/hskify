import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import {
  DEFAULT_MANIFEST_PATH,
  auditCorpus,
  validateManifest,
} from './real-reader-corpus.mjs'
import {
  assertCompletedJob,
  assertHskDifferential,
  assertSemanticExpectations,
} from './run-real-reader-regression.mjs'

function region(overrides = {}) {
  return {
    id: 'region-1',
    sourceEnglish: 'Thanks, Seongeun.',
    displayedChinese: '谢谢，Seongeun。',
    readingOrder: 1,
    textPolygon: [
      { x: 0.1, y: 0.1 },
      { x: 0.3, y: 0.1 },
      { x: 0.3, y: 0.2 },
      { x: 0.1, y: 0.2 },
    ],
    patch: {
      blobId: 'patch-1',
      mimeType: 'image/png',
      rect: { x: 0.09, y: 0.09, width: 0.22, height: 0.12 },
    },
    hsk: { requestedLevel: 2, strictlyValid: true, repairState: 'accepted' },
    ...overrides,
  }
}

test('tracked manifest is deterministic, offline-only, and structurally complete', () => {
  const manifest = JSON.parse(readFileSync(DEFAULT_MANIFEST_PATH, 'utf8'))
  const failures = validateManifest(manifest).filter((item) => !item.passed)
  assert.deepEqual(failures, [])
  assert.equal(manifest.cases.length, 27)
  assert.equal(new Set(manifest.cases.map((item) => item.chapterId)).size, 9)
  assert.equal(manifest.execution.networkPolicy, 'forbidden')
})

test('missing local copyrighted objects are explicit machine-readable failures', () => {
  const emptyCorpus = mkdtempSync(join(tmpdir(), 'hskify-real-reader-empty-'))
  try {
    const result = auditCorpus({ corpusRoot: emptyCorpus, selection: 'smoke' })
    assert.equal(result.status, 'failed')
    assert.equal(result.caseCount, 10)
    assert.equal(result.missingCount, 10)
    assert.equal(result.remediation.requiredPaths.length, 10)
    assert.match(result.remediation.message, /will not download/u)
  } finally {
    rmSync(emptyCorpus, { recursive: true, force: true })
  }
})

test('semantic expectations catch missed names and forbidden SFX regions', () => {
  const item = {
    id: 'dense-page',
    expectations: {
      requiredSourceFragments: ['Seongeun'],
      preserveNamesWhenDetected: ['Seongeun'],
      excludedSourceTexts: ['DING'],
    },
  }
  const passing = assertSemanticExpectations(item, [region()])
  assert.equal(passing.every((item) => item.passed), true)

  const failures = assertSemanticExpectations(item, [
    region({ sourceEnglish: 'Thanks.', displayedChinese: '谢谢。' }),
    region({ id: 'region-2', sourceEnglish: 'DING', displayedChinese: '叮' }),
  ]).filter((item) => !item.passed)
  assert.deepEqual(
    failures.map((item) => item.id),
    [
      'semantic.dense-page.required-source.Seongeun',
      'semantic.dense-page.excluded-source.DING',
    ],
  )
})

test('decorative artwork expectations tolerate OCR noise but reject translated overlays', () => {
  const item = {
    id: 'technique-page',
    expectations: {
      preservedArtworkSourceFragments: ['MYUNGWANG SWORD AUTHORITY'],
    },
  }
  const passing = assertSemanticExpectations(
    item,
    [region({ sourceEnglish: 'Ordinary dialogue.' })],
    [{ sourceEnglish: 'MyUNGWANG SHORd AUTHORITY' }],
  )
  assert.equal(passing.every((assertion) => assertion.passed), true)

  const translated = assertSemanticExpectations(
    item,
    [region({ sourceEnglish: 'MYUNGWANG SWORD AUTHORITY' })],
    [{ sourceEnglish: 'MyUNGWANG SHORd AUTHORITY' }],
  )
  assert.equal(translated.some((assertion) => !assertion.passed), true)
})

test('decorative artwork expectations do not match scattered letters across dialogue', () => {
  const item = {
    id: 'technique-page',
    expectations: {
      preservedArtworkSourceFragments: ['THIRD SWORD'],
    },
  }
  const evaluated = assertSemanticExpectations(
    item,
    [
      region({ sourceEnglish: "THIS FEAR IS IMPRINTED IN MY BLOOD." }),
      region({
        id: 'region-2',
        sourceEnglish: "THERE'S NO WAY THE WHITE TIGER TRIBE WOULD KNOW THE FEAR OF DEATH.",
      }),
    ],
    [{ sourceEnglish: 'THIRD SWORD' }],
  )
  assert.equal(evaluated.every((assertion) => assertion.passed), true)
})

test('completed job assertions require terminal repairs and real PNG patches', () => {
  const item = { id: 'page', expectations: { minimumRegionCount: 1 } }
  const ready = region()
  const evaluated = assertCompletedJob(
    item,
    2,
    { type: 'complete' },
    [{ type: 'regionReady', region: ready }],
    [{ blobId: 'patch-1', validPng: true }],
  )
  assert.equal(evaluated.assertions.every((item) => item.passed), true)

  const pending = assertCompletedJob(
    item,
    2,
    { type: 'complete' },
    [
      {
        type: 'regionReady',
        region: region({ hsk: { requestedLevel: 2, repairState: 'pending' } }),
      },
    ],
    [{ blobId: 'patch-1', validPng: false }],
  )
  assert.equal(
    pending.assertions.some((item) => item.id.endsWith('.repair-terminal') && !item.passed),
    true,
  )
  assert.equal(
    pending.assertions.some((item) => item.id.endsWith('.patch-png') && !item.passed),
    true,
  )
})

test('protected artwork rectangles reject cleanup patches even without readable OCR', () => {
  const item = {
    id: 'illustrated-label',
    expectations: {
      minimumRegionCount: 1,
      protectedArtworkRects: [{ x: 0.1, y: 0.4, width: 0.8, height: 0.2 }],
    },
  }
  const safe = assertCompletedJob(
    item,
    2,
    { type: 'complete' },
    [
      {
        type: 'regionReady',
        region: region({ patch: { blobId: 'patch-1', mimeType: 'image/png', rect: { x: 0.1, y: 0.1, width: 0.2, height: 0.1 } } }),
      },
    ],
    [{ blobId: 'patch-1', validPng: true }],
  )
  assert.equal(safe.assertions.every((assertion) => assertion.passed), true)

  const damaged = assertCompletedJob(
    item,
    2,
    { type: 'complete' },
    [
      {
        type: 'regionReady',
        region: region({ patch: { blobId: 'patch-1', mimeType: 'image/png', rect: { x: 0.2, y: 0.45, width: 0.2, height: 0.1 } } }),
      },
    ],
    [{ blobId: 'patch-1', validPng: true }],
  )
  assert.equal(
    damaged.assertions.some(
      (assertion) => assertion.id.endsWith('protected-artwork-rect.1') && !assertion.passed,
    ),
    true,
  )
})

test('completed job assertions support exact zero-region non-story controls', () => {
  const evaluated = assertCompletedJob(
    { id: 'credit-splash', expectations: { exactRegionCount: 0 } },
    3,
    { type: 'complete' },
    [],
    [],
  )
  assert.equal(evaluated.assertions.every((item) => item.passed), true)

  const falsePositive = assertCompletedJob(
    { id: 'credit-splash', expectations: { exactRegionCount: 0 } },
    3,
    { type: 'complete' },
    [{ type: 'regionReady', region: region() }],
    [{ blobId: 'patch-1', validPng: true }],
  )
  assert.equal(
    falsePositive.assertions.some((item) => item.id.endsWith('.regions') && !item.passed),
    true,
  )
})

test('HSK2 and HSK5 differential rejects identical output and accepts simpler divergence', () => {
  const low = { regions: [region({ displayedChinese: '你过来。' })] }
  const identicalHigh = {
    regions: [
      region({
        displayedChinese: '你过来。',
        hsk: { requestedLevel: 5, strictlyValid: true, repairState: 'accepted' },
      }),
    ],
  }
  assert.equal(
    assertHskDifferential(low, identicalHigh).find(
      (item) => item.id === 'differential.hsk-2-vs-5.changed-output',
    ).passed,
    false,
  )

  const differentHigh = {
    regions: [
      region({
        displayedChinese: '请到这里来。',
        hsk: { requestedLevel: 5, strictlyValid: true, repairState: 'accepted' },
      }),
    ],
  }
  assert.equal(assertHskDifferential(low, differentHigh).every((item) => item.passed), true)
})
