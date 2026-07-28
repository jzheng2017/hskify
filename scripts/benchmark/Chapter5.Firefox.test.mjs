import assert from 'node:assert/strict'
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import Ajv2020 from 'ajv/dist/2020.js'

import {
  BENCHMARK_LIMITS,
  BENCHMARK_QUALITY_LIMITS,
  BUILD_FINGERPRINT,
  assertCompleteTranslationGold,
  assertRequiredGates,
  buildJobRequestEvidence,
  buildPatchCommitOrderingEvidence,
  buildPatchQualityEvidence,
  buildQualityEvidence,
  cancellationTiming,
  clearExactResultCache,
  evaluateOverflowEvidence,
  exactChapterSnapshotMatch,
  reconcileCompleteJobTerminals,
  requestedZoomApplied,
  selectViewportPlan,
  validateBenchmarkManifest,
  validateExpectedResourceIdentities,
} from './Chapter5.Firefox.mjs'

const BENCHMARK_ID = '30-years-since-the-prologue-chapter-5'
const PAGE_COUNT = 36
const SYNTHETIC_MANIFEST = Object.freeze({
  schemaVersion: 3,
  id: BENCHMARK_ID,
  pageCount: PAGE_COUNT,
  totalExpectedRegionCount: 218,
  totalExpectedDialogueBubbleCount: 165,
  totalExpectedNarrationCount: 53,
  totalExpectedEnglishTranslationTargetCount: 214,
  totalExpectedUntouchedExclusionCount: 4,
  annotationStatus: {
    status: 'complete',
    reasonCode: 'all-gold-fields-present',
    reviewedPageCount: PAGE_COUNT,
    generatedPageCount: PAGE_COUNT,
    completedPageCount: PAGE_COUNT,
    requiredPageCount: PAGE_COUNT,
    missingPages: [],
    missingFieldCounts: {},
    totalMissingFieldCount: 0,
  },
  images: Array.from({ length: PAGE_COUNT }, (_, index) => ({
    order: index + 1,
    expectedRegionCount: index === 0 ? 218 : 0,
    expectedDialogueBubbleCount: index === 0 ? 165 : 0,
    expectedNarrationCount: index === 0 ? 53 : 0,
    expectedEnglishTranslationTargetCount: index === 0 ? 214 : 0,
    expectedUntouchedExclusionCount: index === 0 ? 4 : 0,
  })),
})

test('terminal evidence follows daemon acceptance order rather than page order', () => {
  const jobMonitor = {
    observations: [
      { jobId: 'page-1', pageIndex: 0, createdAtUnixMs: 200 },
      { jobId: 'visible-page-4', pageIndex: 3, createdAtUnixMs: 100 },
      { jobId: 'page-2', pageIndex: 1, createdAtUnixMs: 300 },
    ],
  }
  const routes = {
    jobs: [
      { jobId: 'page-1', terminal: { type: 'complete' } },
      { jobId: 'visible-page-4', terminal: { type: 'complete' } },
      { jobId: 'page-2', terminal: { type: 'complete' } },
    ],
  }

  reconcileCompleteJobTerminals(jobMonitor, routes, {
    events: [{ type: 'hudComplete', epochMs: 400 }],
  })

  assert.equal(jobMonitor.observations[1].terminalObservedAtEpochMs, 200)
  assert.equal(jobMonitor.observations[0].terminalObservedAtEpochMs, 300)
  assert.equal(jobMonitor.observations[2].terminalObservedAtEpochMs, 400)
})

test('terminal evidence ignores stale HUD completion from a preceding run', () => {
  const jobMonitor = {
    observations: [
      {
        jobId: 'last-job',
        pageIndex: 9,
        createdAtUnixMs: 700,
      },
    ],
  }
  const routes = {
    jobs: [{ jobId: 'last-job', terminal: { type: 'complete' } }],
  }

  reconcileCompleteJobTerminals(jobMonitor, routes, {
    events: [
      { type: 'hudComplete', epochMs: 600 },
      { type: 'hudComplete', epochMs: 800 },
    ],
  })

  assert.equal(jobMonitor.observations[0].terminalObservedAtEpochMs, 800)
})

test('warm inference reset removes only exact isolated result-cache entries', () => {
  const outputDirectory = mkdtempSync(join(tmpdir(), 'hskify-benchmark-cache-'))
  try {
    const stateDirectory = join(outputDirectory, 'isolated-state')
    const resultDirectory = join(stateDirectory, 'browser-cache', 'results')
    mkdirSync(resultDirectory, { recursive: true })
    const first = `${'a'.repeat(64)}.json`
    const second = `${'b'.repeat(64)}.json`
    writeFileSync(join(resultDirectory, first), '{}')
    writeFileSync(join(resultDirectory, second), '{}')

    const evidence = clearExactResultCache({ outputDirectory, stateDirectory })

    assert.equal(evidence.removedEntryCount, 2)
    assert.equal(evidence.removedBytes, 4)
    assert.equal(evidence.measuredPhaseExcluded, true)
    assert.equal(existsSync(join(resultDirectory, first)), false)
    assert.equal(existsSync(join(resultDirectory, second)), false)
  } finally {
    rmSync(outputDirectory, { recursive: true, force: true })
  }
})

test('committed chapter gold has a deterministic real viewport', () => {
  const fixtureRoot = new URL(
    '../../fixtures/benchmarks/30-years-since-the-prologue-chapter-5/',
    import.meta.url,
  )
  const manifest = JSON.parse(readFileSync(new URL('manifest.json', fixtureRoot), 'utf8'))
  const goldPages = manifest.images.map((image) => {
    assert.equal(
      typeof image.annotation,
      'string',
      `chapter-5 fixture page ${image.order} is missing its annotation path`,
    )
    return {
      order: image.order,
      file: image.file,
      regions: JSON.parse(readFileSync(new URL(image.annotation, fixtureRoot), 'utf8')).regions,
    }
  })

  const plan = selectViewportPlan(manifest, goldPages)

  assert.ok(plan.expectedVisibleRegionCount >= 3)
  assert.ok(plan.expectedVisibleRegionCount <= 6)
  assert.equal(plan.expectedVisibleRegionIds.length, plan.expectedVisibleRegionCount)
})

function polygon(left, top, right, bottom) {
  return [
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom],
  ]
}

function region(id, sourceEnglish, textPolygon) {
  return {
    id,
    textPolygon: textPolygon.map(([x, y]) => ({ x, y })),
    sourceEnglish,
    displayedChinese: '你好',
    pinyin: 'nǐ hǎo',
    hsk: {},
  }
}

function fixture(observedText = 'Hello', observedPolygon = polygon(0.1, 0.1, 0.3, 0.2)) {
  return {
    gold: [
      {
        order: 1,
        regions: [
          {
            id: 'gold-1',
            kind: 'dialogue',
            sourceEnglish: 'Hello',
            textPolygon: polygon(0.1, 0.1, 0.3, 0.2),
          },
        ],
      },
    ],
    routes: {
      jobs: [
        {
          pageIndex: 0,
          updates: [
            {
              type: 'regionReady',
              region: region('observed-1', observedText, observedPolygon),
            },
          ],
        },
      ],
    },
  }
}

test('accepted translation quality uses browser point objects and explicit OCR/false-translation denominators', () => {
  const input = fixture()
  const evidence = buildQualityEvidence(input.routes, input.gold)

  assert.equal(evidence.totals.expectedRegionCount, 1)
  assert.equal(evidence.totals.detectorGoldBubbleCount, 1)
  assert.equal(evidence.totals.expectedNarrationRegionCount, 0)
  assert.equal(evidence.totals.expectedEnglishTranslationTargetCount, 1)
  assert.equal(evidence.totals.matchedEnglishTargetCount, 1)
  assert.equal(evidence.totals.ocrCharacterErrorNumerator, 0)
  assert.equal(evidence.totals.ocrReferenceCharacterDenominator, 5)
  assert.equal(evidence.totals.englishOcrCer, 0)
  assert.equal(evidence.totals.falseTranslationNumerator, 0)
  assert.equal(evidence.totals.falseTranslationDenominator, 1)
  assert.equal(evidence.totals.falseTranslationRate, 0)
  assert.ok(evidence.gates.every((gate) => gate.status === 'pass'))
})

test('narration is a translation target but not detector bubble gold', () => {
  const input = fixture()
  input.gold[0].regions[0].kind = 'narration'

  const evidence = buildQualityEvidence(input.routes, input.gold)
  assert.equal(evidence.totals.expectedRegionCount, 1)
  assert.equal(evidence.totals.detectorGoldBubbleCount, 0)
  assert.equal(evidence.totals.expectedNarrationRegionCount, 1)
  assert.equal(evidence.totals.expectedEnglishTranslationTargetCount, 1)
  assert.equal(evidence.totals.matchedEnglishTargetCount, 1)
})

test('punctuation-only gold stays untouched and outside translation correctness', () => {
  const input = fixture()
  input.gold[0].regions.push({
    id: 'gold-punctuation',
    kind: 'dialogue',
    sourceEnglish: '?!!!',
    translationTarget: false,
    textPolygon: polygon(0.5, 0.5, 0.6, 0.6),
  })

  const evidence = buildQualityEvidence(input.routes, input.gold)
  assert.equal(evidence.totals.detectorGoldBubbleCount, 2)
  assert.equal(evidence.totals.expectedEnglishTranslationTargetCount, 1)
  assert.equal(evidence.totals.expectedUntouchedExclusionCount, 1)
  assert.equal(evidence.totals.untouchedExclusions, 1)
  assert.equal(evidence.totals.modifiedExclusions, 0)
  assert.deepEqual(evidence.pages[0].untouchedExclusionRegionIds, [
    'gold-punctuation',
  ])
  assert.equal(evidence.totals.matchedEnglishTargetCount, 1)
  assert.equal(evidence.totals.englishOcrCer, 0)
  assert.equal(
    evidence.gates.find((gate) => gate.id === 'ambiguous-punctuation-left-untouched')
      ?.status,
    'pass',
  )
  assert.ok(evidence.gates.every((gate) => gate.status === 'pass'))

  input.routes.jobs[0].updates.push({
    type: 'regionReady',
    region: region(
      'observed-punctuation',
      '?!!!',
      polygon(0.5, 0.5, 0.6, 0.6),
    ),
  })
  const modified = buildQualityEvidence(input.routes, input.gold)
  assert.equal(modified.totals.matchedEnglishTargetCount, 1)
  assert.equal(modified.totals.ocrUnmatchedInsertionErrors, 0)
  assert.equal(modified.totals.englishOcrCer, 0)
  assert.equal(modified.totals.falseTranslationNumerator, 1)
  assert.equal(modified.totals.untouchedExclusions, 0)
  assert.equal(modified.totals.modifiedExclusions, 1)
  assert.equal(
    modified.gates.find((gate) => gate.id === 'ambiguous-punctuation-left-untouched')
      ?.status,
    'fail',
  )
})

test('OCR CER measures recognition only after spatial matching', () => {
  const ocr = fixture('Hxllo')
  const ocrEvidence = buildQualityEvidence(ocr.routes, ocr.gold)
  assert.equal(ocrEvidence.totals.ocrCharacterErrorNumerator, 1)
  assert.equal(ocrEvidence.totals.ocrReferenceCharacterDenominator, 5)
  assert.equal(ocrEvidence.totals.englishOcrCer, 0.2)
  assert.equal(
    ocrEvidence.gates.find((gate) => gate.id === 'english-ocr-cer')?.status,
    'fail',
  )

  const geometry = fixture('Hello', polygon(0.7, 0.7, 0.9, 0.8))
  const geometryEvidence = buildQualityEvidence(geometry.routes, geometry.gold)
  assert.equal(geometryEvidence.totals.missingEnglishTargetCount, 1)
  assert.equal(geometryEvidence.totals.unmatchedAcceptedTranslationCount, 1)
  assert.equal(geometryEvidence.totals.ocrMissingCharacterErrors, 0)
  assert.equal(geometryEvidence.totals.ocrUnmatchedInsertionErrors, 0)
  assert.equal(geometryEvidence.totals.ocrReferenceCharacterDenominator, 0)
  assert.equal(geometryEvidence.totals.englishOcrCer, 1)
  assert.equal(
    geometryEvidence.gates.find((gate) => gate.id === 'story-region-publication-recall')
      ?.status,
    'fail',
  )
  assert.equal(
    geometryEvidence.gates.find(
      (gate) => gate.id === 'non-english-non-dialogue-false-translation-rate',
    )?.status,
    'fail',
  )
})

test('English OCR CER accepts exactly three percent and recall charges missing output', () => {
  const reference = 'a'.repeat(100)
  const atLimit = fixture(`${'b'.repeat(3)}${'a'.repeat(97)}`)
  atLimit.gold[0].regions[0].sourceEnglish = reference
  const atLimitEvidence = buildQualityEvidence(atLimit.routes, atLimit.gold)
  assert.equal(atLimitEvidence.totals.englishOcrCer, 0.03)
  assert.equal(
    atLimitEvidence.gates.find((gate) => gate.id === 'english-ocr-cer')?.status,
    'pass',
  )

  const missing = fixture()
  missing.gold[0].regions[0].sourceEnglish = reference
  missing.routes.jobs[0].updates = []
  const missingEvidence = buildQualityEvidence(missing.routes, missing.gold)
  assert.equal(missingEvidence.totals.ocrMissingCharacterErrors, 0)
  assert.equal(missingEvidence.totals.ocrReferenceCharacterDenominator, 0)
  assert.equal(missingEvidence.totals.englishOcrCer, 1)
  assert.equal(missingEvidence.totals.storyRegionRecall, 0)
  assert.equal(missingEvidence.totals.falseTranslationDenominator, 0)
  assert.equal(missingEvidence.totals.falseTranslationRate, 0)
  assert.equal(
    missingEvidence.gates.find((gate) => gate.id === 'story-region-publication-recall')
      ?.status,
    'fail',
  )
})

test('spatial components tolerate OCR splits and merges without hiding text errors', () => {
  const input = fixture()
  input.gold[0].regions.push({
    id: 'gold-2',
    kind: 'dialogue',
    sourceEnglish: 'world',
    textPolygon: polygon(0.3, 0.1, 0.5, 0.2),
  })
  input.routes.jobs[0].updates = [
    {
      type: 'regionReady',
      region: region('observed-merged', 'Hello world', polygon(0.1, 0.1, 0.5, 0.2)),
    },
  ]

  const merged = buildQualityEvidence(input.routes, input.gold)
  assert.equal(merged.totals.matchedEnglishTargetCount, 2)
  assert.equal(merged.totals.storyRegionRecall, 1)
  assert.equal(merged.totals.englishOcrCer, 0)
  assert.equal(merged.components.length, 1)
  assert.deepEqual(merged.components[0].expectedRegionIds, ['gold-1', 'gold-2'])
  assert.deepEqual(merged.components[0].observedRegionIds, ['observed-merged'])

  input.routes.jobs[0].updates = [
    {
      type: 'regionReady',
      region: region('observed-left', 'Hello', polygon(0.1, 0.1, 0.3, 0.2)),
    },
    {
      type: 'regionReady',
      region: region('observed-right', 'wurld', polygon(0.3, 0.1, 0.5, 0.2)),
    },
  ]
  const split = buildQualityEvidence(input.routes, input.gold)
  assert.equal(split.totals.matchedEnglishTargetCount, 2)
  assert.equal(split.totals.storyRegionRecall, 1)
  assert.equal(split.totals.ocrCharacterErrorNumerator, 1)
  assert.equal(split.totals.englishOcrCer, 0.1)
})

test('false translation rate is unmatched accepted output over all accepted output', () => {
  const gold = []
  const updates = []
  for (let index = 0; index < 100; index += 1) {
    const column = index % 10
    const row = Math.floor(index / 10)
    const geometry = polygon(
      column * 0.09,
      row * 0.09,
      column * 0.09 + 0.04,
      row * 0.09 + 0.04,
    )
    gold.push({
      id: `gold-${index}`,
      kind: 'dialogue',
      sourceEnglish: 'a',
      textPolygon: geometry,
    })
    updates.push({
      type: 'regionReady',
      region: region(`observed-${index}`, 'a', geometry),
    })
  }
  updates.push({
    type: 'regionReady',
    region: region('unmatched', 'x', polygon(0.95, 0.95, 0.99, 0.99)),
  })
  const evidence = buildQualityEvidence(
    { jobs: [{ pageIndex: 0, updates }] },
    [{ order: 1, regions: gold }],
  )
  assert.equal(evidence.totals.falseTranslationNumerator, 1)
  assert.equal(evidence.totals.falseTranslationDenominator, 101)
  assert.equal(evidence.totals.falseTranslationRate, 1 / 101)
  assert.equal(
    evidence.gates.find(
      (gate) => gate.id === 'non-english-non-dialogue-false-translation-rate',
    )?.status,
    'pass',
  )
})

function patchFixture(alphaRows) {
  const goldRegion = {
    id: 'gold-1',
    sourceEnglish: 'Hello',
    textPolygon: polygon(0.1, 0.1, 0.3, 0.3),
    bubblePolygon: polygon(0.1, 0.1, 0.3, 0.3),
    eraseMask: { polygon: polygon(0.1, 0.1, 0.3, 0.3) },
  }
  return {
    goldPages: [{ order: 1, regions: [goldRegion] }],
    sourceGlyphs: [
      {
        page: 1,
        id: 'observed-1',
        originX: 1,
        originY: 1,
        width: 2,
        height: 2,
        pixels: 4,
        rows: [
          { y: 0, runs: [[0, 2]] },
          { y: 1, runs: [[0, 2]] },
        ],
      },
    ],
    matches: [
      {
        page: 1,
        expectedRegionIds: ['gold-1'],
        observedRegionIds: ['observed-1'],
      },
    ],
    routes: {
      jobs: [
        {
          pageIndex: 0,
          sourceWidth: 10,
          sourceHeight: 10,
          sourceSha256: 'b'.repeat(64),
          patches: [
            {
              patchId: 'patch-1',
              regionId: 'observed-1',
              rect: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 },
              bubblePolygon: polygon(0.1, 0.1, 0.3, 0.3),
              width: 2,
              height: 2,
              sha256: 'c'.repeat(64),
              decodedRgbaSha256: 'd'.repeat(64),
              alphaNonZeroPixelCount: alphaRows.reduce(
                (sum, row) =>
                  sum +
                  row.runs.reduce(
                    (rowSum, [start, end]) => rowSum + end - start,
                    0,
                  ),
                0,
              ),
              alphaRows,
            },
          ],
        },
      ],
    },
  }
}

test('decoded fetched PNG alpha covers source glyphs and stays in its accepted region', () => {
  const input = patchFixture([
    { y: 0, runs: [[0, 2]] },
    { y: 1, runs: [[0, 2]] },
  ])
  const evidence = buildPatchQualityEvidence(
    input.routes,
    input.goldPages,
    input.matches,
    input.sourceGlyphs,
  )
  assert.equal(evidence.totals.eraseMaskPixelDenominator, 4)
  assert.equal(evidence.totals.coveredEraseMaskPixelNumerator, 4)
  assert.equal(evidence.totals.coveredGlyphPixelNumerator, 4)
  assert.equal(evidence.totals.alphaOutsideAcceptedRegionPixels, 0)
  assert.ok(evidence.gates.every((gate) => gate.status === 'pass'))

  input.routes.jobs[0].patches[0].textPolygon =
    input.routes.jobs[0].patches[0].bubblePolygon
  delete input.routes.jobs[0].patches[0].bubblePolygon
  const freeText = buildPatchQualityEvidence(
    input.routes,
    input.goldPages,
    input.matches,
    input.sourceGlyphs,
  )
  assert.equal(freeText.totals.alphaOutsideAcceptedRegionPixels, 0)
  assert.ok(freeText.gates.every((gate) => gate.status === 'pass'))

  input.routes.jobs[0].patches[0].bubblePolygon =
    input.routes.jobs[0].patches[0].textPolygon
  input.routes.jobs[0].patches[0].width = 3
  input.routes.jobs[0].patches[0].rect.width = 0.3
  input.routes.jobs[0].patches[0].alphaRows = [
    { y: 0, runs: [[0, 3]] },
    { y: 1, runs: [[0, 3]] },
  ]
  input.routes.jobs[0].patches[0].alphaNonZeroPixelCount = 6
  const outside = buildPatchQualityEvidence(
    input.routes,
    input.goldPages,
    input.matches,
    input.sourceGlyphs,
  )
  assert.equal(outside.totals.alphaOutsideAcceptedRegionPixels, 2)
  assert.equal(
    outside.gates.find(
      (gate) => gate.id === 'patch-changes-outside-runtime-accepted-region',
    )
      ?.status,
    'fail',
  )

  const split = patchFixture([
    { y: 0, runs: [[0, 1]] },
    { y: 1, runs: [[0, 1]] },
  ])
  const left = split.routes.jobs[0].patches[0]
  left.rect.width = 0.1
  left.width = 1
  left.bubblePolygon = polygon(0.1, 0.1, 0.2, 0.3)
  left.alphaNonZeroPixelCount = 2
  const right = structuredClone(left)
  right.patchId = 'patch-2'
  right.regionId = 'observed-2'
  right.rect.x = 0.2
  right.bubblePolygon = polygon(0.2, 0.1, 0.3, 0.3)
  split.routes.jobs[0].patches.push(right)
  split.matches[0].observedRegionIds.push('observed-2')
  split.sourceGlyphs[0] = {
    ...split.sourceGlyphs[0],
    width: 1,
    pixels: 2,
    rows: [
      { y: 0, runs: [[0, 1]] },
      { y: 1, runs: [[0, 1]] },
    ],
  }
  split.sourceGlyphs.push({
    ...split.sourceGlyphs[0],
    id: 'observed-2',
    originX: 2,
  })
  const splitEvidence = buildPatchQualityEvidence(
    split.routes,
    split.goldPages,
    split.matches,
    split.sourceGlyphs,
  )
  assert.equal(splitEvidence.totals.patchRegionCount, 2)
  assert.equal(splitEvidence.totals.coveredGlyphPixelNumerator, 4)
  assert.ok(splitEvidence.gates.every((gate) => gate.status === 'pass'))
})

test('Chinese DOM ordering requires the corresponding decoded patch installation first', () => {
  const routes = {
    jobs: [
      {
        pageIndex: 0,
        updates: [
          {
            type: 'regionReady',
            region: {
              id: 'region-1',
              patch: { blobId: 'patch-1' },
            },
          },
        ],
      },
    ],
  }
  const patch = {
    type: 'patchDomCommitted',
    patchId: 'patch-1',
    index: 1,
    epochMs: 10,
    complete: true,
    naturalWidth: 2,
    naturalHeight: 2,
    decodedAndInstalled: true,
  }
  const text = {
    type: 'selectableTextDomCommitted',
    regionId: 'region-1',
    index: 2,
    epochMs: 10,
  }
  const pass = buildPatchCommitOrderingEvidence(routes, { events: [patch, text] })
  assert.ok(pass.gates.every((gate) => gate.status === 'pass'))

  const fail = buildPatchCommitOrderingEvidence(routes, {
    events: [{ ...text, index: 0 }, patch],
  })
  assert.equal(
    fail.gates.find(
      (gate) => gate.id === 'decoded-patch-installed-before-corresponding-chinese-dom',
    )?.status,
    'fail',
  )
})

test('overflow evidence checks every translated region in every supported zoom/resize case', () => {
  const translatedRegionCount = 7
  const cases = [1, 1.25, 1.5].map((zoom) => ({
    zoom,
    cssZoomSupported: true,
    inlineZoom: String(zoom),
    computedZoom: String(zoom),
    regionCount: translatedRegionCount,
    overflowRegionIds: [],
  }))
  const evidence = evaluateOverflowEvidence(cases, translatedRegionCount)
  assert.equal(evidence.checkedRegionDenominator, translatedRegionCount * cases.length)
  assert.ok(evidence.gates.every((gate) => gate.status === 'pass'))

  cases[1].overflowRegionIds.push('region-7')
  const overflow = evaluateOverflowEvidence(cases, translatedRegionCount)
  assert.equal(overflow.overflowRegionNumerator, 1)
  assert.equal(
    overflow.gates.find((gate) => gate.id === 'zero-overflow-under-supported-zoom-resize')
      ?.status,
    'fail',
  )
})

test('missing measurements cannot be asserted as passing evidence', () => {
  assert.throws(
    () =>
      assertRequiredGates(
        [
          {
            id: 'peak-vram',
            status: 'missing',
            reason: 'process VRAM unavailable',
          },
        ],
        'resource',
      ),
    /peak-vram: process VRAM unavailable/u,
  )
  assert.deepEqual(BENCHMARK_LIMITS, {
    hudAcknowledgementMs: 100,
    exactCachedFirstViewportMs: 250,
    firstVisibleRegionMs: 2_000,
    visibleRegionGroupMs: 5_000,
    firstLongImageCompleteMs: 12_000,
    allImagesCompleteMs: 90_000,
    cancellationMs: 500,
    installedColdFirstVisibleBubbleMs: 8_000,
    installedColdFirstLongImageCompleteMs: 20_000,
    installedColdAllImagesCompleteMs: 120_000,
  })
  assert.deepEqual(BENCHMARK_QUALITY_LIMITS, {
    storyRegionRecall: 0.95,
    englishOcrCer: 0.03,
    falseTranslationRate: 0.01,
  })
})

test('cancellation timing is causal and excludes post-hoc route evidence', () => {
  const timing = cancellationTiming({
    cancelIssuedAtEpochMs: 1_000,
    pageRestoredAtEpochMs: 1_037,
    daemonTerminalObservedAtEpochMs: 1_052,
    postHocHealthAndPatchEvidenceEndedAtEpochMs: 9_999,
  })

  assert.equal(timing.pageCancellationLatencyMs, 37)
  assert.equal(timing.daemonCancellationLatencyMs, 52)
  assert.equal(timing.measuredPhaseStartedAtEpochMs, 1_000)
  assert.equal(timing.measuredPhaseEndedAtEpochMs, 1_052)
  assert.match(timing.timestampDefinition.pageRestoredAt, /exact DOM snapshot/u)
  assert.match(timing.timestampDefinition.daemonTerminalObservedAt, /cancelled/u)
  assert.match(timing.timestampDefinition.excluded, /health.*patch/iu)
  assert.throws(
    () =>
      cancellationTiming({
        cancelIssuedAtEpochMs: 1_000,
        pageRestoredAtEpochMs: 999,
        daemonTerminalObservedAtEpochMs: 1_001,
      }),
    /must not precede cancel issuance/u,
  )
})

test('exact chapter snapshots detect any attribute, sibling, or order difference', () => {
  const expected = {
    outerHTML:
      '<main id="chapter"><span>before</span><img src="1.webp" srcset="1.webp 1x" sizes="100vw" class="page" style="width: 100%"><!--edge--><img src="2.webp"></main>',
    attributes: [['id', 'chapter']],
    childNodes: [
      { nodeType: 1, nodeName: 'SPAN', outerHTML: '<span>before</span>' },
      {
        nodeType: 1,
        nodeName: 'IMG',
        outerHTML:
          '<img src="1.webp" srcset="1.webp 1x" sizes="100vw" class="page" style="width: 100%">',
      },
      { nodeType: 8, nodeName: '#comment', data: 'edge' },
      { nodeType: 1, nodeName: 'IMG', outerHTML: '<img src="2.webp">' },
    ],
    images: [
      {
        path: [1],
        parentPath: [],
        previousSibling: {
          nodeType: 1,
          nodeName: 'SPAN',
          outerHTML: '<span>before</span>',
        },
        nextSibling: { nodeType: 8, nodeName: '#comment', data: 'edge' },
        outerHTML:
          '<img src="1.webp" srcset="1.webp 1x" sizes="100vw" class="page" style="width: 100%">',
        attributes: [
          ['src', '1.webp'],
          ['srcset', '1.webp 1x'],
          ['sizes', '100vw'],
          ['class', 'page'],
          ['style', 'width: 100%'],
        ],
        src: '1.webp',
        srcset: '1.webp 1x',
        sizes: '100vw',
        class: 'page',
        style: 'width: 100%',
      },
    ],
  }
  assert.equal(exactChapterSnapshotMatch(expected, structuredClone(expected)), true)

  const changedAttribute = structuredClone(expected)
  changedAttribute.images[0].srcset = '2.webp 2x'
  assert.equal(exactChapterSnapshotMatch(expected, changedAttribute), false)

  const changedOrder = structuredClone(expected)
  changedOrder.childNodes.reverse()
  assert.equal(exactChapterSnapshotMatch(expected, changedOrder), false)
})

test('zoom evidence passes only when inline and computed zoom equal the request', () => {
  assert.equal(requestedZoomApplied(1.25, '1.25', '1.25'), true)
  assert.equal(requestedZoomApplied(1.5, '1.5', '1'), false)
  assert.equal(requestedZoomApplied(1.25, '', '1.25'), false)
})

test('pinned detector and OCR identities are strict, unique, and sorted', () => {
  const identity = (id) => ({
    id,
    repository: 'publisher/model',
    repositoryRevision: 'a'.repeat(40),
    filename: `${id}.onnx`,
    bytes: 123,
    sha256: 'b'.repeat(64),
  })
  const identities = [identity('detector'), identity('ocr')]
  assert.equal(validateExpectedResourceIdentities(identities), identities)
  assert.throws(
    () => validateExpectedResourceIdentities([...identities].reverse()),
    /must be sorted by id/u,
  )
  assert.throws(
    () => validateExpectedResourceIdentities([{ ...identities[0], unexpected: true }]),
    /invalid resource identity/u,
  )
  assert.throws(
    () => validateExpectedResourceIdentities([identities[0], { ...identities[0] }]),
    /invalid resource identity/u,
  )
  assert.throws(() => validateExpectedResourceIdentities([]), /missing or empty/u)
})

test('job request evidence gates the direct request contract and rolling six-item context', () => {
  const hashes = ['a'.repeat(64), 'b'.repeat(64)]
  const manifest = {
    pageCount: 2,
    images: [
      { order: 1, sha256: hashes[0], bytes: 101, width: 800, height: 1000 },
      { order: 2, sha256: hashes[1], bytes: 202, width: 800, height: 1100 },
    ],
  }
  const routes = {
    jobs: [
      {
        pageIndex: 0,
        sourceSha256: hashes[0],
        updates: [
          {
            type: 'regionReady',
            region: {
              ...region('p1-r1', 'First', polygon(0.1, 0.1, 0.2, 0.2)),
              readingOrder: 0,
              displayedChinese: '第一',
            },
          },
        ],
      },
      {
        pageIndex: 1,
        sourceSha256: hashes[1],
        updates: [
          {
            type: 'regionReady',
            region: {
              ...region('p2-r1', 'Second', polygon(0.1, 0.1, 0.2, 0.2)),
              readingOrder: 0,
              displayedChinese: '第二',
            },
          },
        ],
      },
    ],
  }
  const action = 1_000
  const session = 'page-session'
  const settings = {
    sourceLanguage: 'en',
    targetLanguage: 'zh-CN',
    hskStandard: '2.0',
    hskLevel: 5,
    readingDirection: 'auto',
    translateSoundEffects: false,
  }
  const makeRequest = (pageIndex, precedingContext) => ({
    buildFingerprint: BUILD_FINGERPRINT,
    clientImageId: `${session}-${pageIndex}-${hashes[pageIndex].slice(0, 16)}`,
    sourceSha256: hashes[pageIndex],
    sourceMimeType: 'image/webp',
    naturalWidth: manifest.images[pageIndex].width,
    naturalHeight: manifest.images[pageIndex].height,
    pageSessionId: session,
    pageIndex,
    settings,
    visibleRects: [],
    ...(precedingContext ? { precedingContext } : {}),
  })
  const records = [
    {
      pageIndex: 0,
      sourceSha256: hashes[0],
      submittedRequest: makeRequest(0),
      uploadedImageBytes: 101,
      submittedAtUnixMs: 1_010,
      createdAtUnixMs: 1_015,
    },
    {
      pageIndex: 1,
      sourceSha256: hashes[1],
      submittedRequest: makeRequest(1, [
        { sourceEnglish: 'First', chinese: '第一' },
      ]),
      uploadedImageBytes: 202,
      submittedAtUnixMs: 1_020,
      createdAtUnixMs: 1_025,
    },
  ]
  const evidence = buildJobRequestEvidence(
    manifest,
    routes,
    records,
    5,
    action,
  )
  assert.equal(evidence.gates[0].status, 'pass')

  const outOfPageOrderRecords = structuredClone(records)
  outOfPageOrderRecords[0].submittedAtUnixMs = 1_020
  outOfPageOrderRecords[0].createdAtUnixMs = 1_025
  outOfPageOrderRecords[0].submittedRequest = makeRequest(0, [
    {
      sourceEnglish: 'Second',
      chinese: routes.jobs[1].updates[0].region.displayedChinese,
    },
  ])
  outOfPageOrderRecords[1].submittedAtUnixMs = 1_010
  outOfPageOrderRecords[1].createdAtUnixMs = 1_015
  outOfPageOrderRecords[1].submittedRequest = makeRequest(1)
  const visibleFirst = buildJobRequestEvidence(
    manifest,
    routes,
    outOfPageOrderRecords,
    5,
    action,
  )
  assert.equal(visibleFirst.gates[0].status, 'pass', JSON.stringify(visibleFirst.mismatches))

  records[1].submittedRequest.properNameGlossary = []
  const mismatch = buildJobRequestEvidence(
    manifest,
    routes,
    records,
    5,
    action,
  )
  assert.equal(mismatch.gates[0].status, 'fail')
  assert.match(mismatch.mismatches[0], /exactKeys/u)
})

test('benchmark evidence schema compiles in strict draft-2020-12 mode', () => {
  const fixtureRoot = new URL(
    '../../fixtures/benchmarks/30-years-since-the-prologue-chapter-5/',
    import.meta.url,
  )
  const manifest = JSON.parse(readFileSync(new URL('manifest.json', fixtureRoot), 'utf8'))
  const schema = JSON.parse(
    readFileSync(
      new URL('benchmark-evidence.schema.json', fixtureRoot),
      'utf8',
    ),
  )
  assert.doesNotThrow(() => new Ajv2020({ strict: true, allErrors: true }).compile(schema))
  assert.ok(
    schema.$defs.correctness.properties.totals.required.includes(
      'ocrReferenceCharacterDenominator',
    ),
  )
  assert.ok(
    schema.$defs.correctness.properties.totals.required.includes(
      'falseTranslationDenominator',
    ),
  )
  assert.ok(schema.$defs.correctness.required.includes('patchPng'))
  assert.ok(schema.$defs.correctness.required.includes('commitOrdering'))
  assert.ok(!schema.$defs.summary.required.includes('detectorCorrectness'))
  assert.equal(
    schema.$defs.correctness.properties.totals.properties.expectedRegionCount.const,
    manifest.totalExpectedRegionCount,
  )
  assert.equal(
    schema.$defs.correctness.properties.totals.properties.expectedNarrationRegionCount.const,
    manifest.totalExpectedNarrationCount,
  )
  assert.ok(
    !schema.$defs.jobRequests.required.includes('expectedProperNameGlossary'),
  )

})

test('chapter-5 fixture has 36 manifest-counted pages and canonical region IDs', () => {
  const fixtureRoot = new URL(
    '../../fixtures/benchmarks/30-years-since-the-prologue-chapter-5/',
    import.meta.url,
  )
  const manifest = JSON.parse(readFileSync(new URL('manifest.json', fixtureRoot), 'utf8'))
  assert.equal(manifest.id, BENCHMARK_ID)
  assert.equal(manifest.pageCount, PAGE_COUNT)
  assert.equal(manifest.images.length, manifest.pageCount)
  assert.deepEqual(validateBenchmarkManifest(manifest), {
    regionCount: 218,
    goldBubbleCount: 165,
    narrationRegionCount: 53,
    englishTranslationTargetCount: 214,
    untouchedExclusionCount: 4,
  })
  assert.equal(manifest.annotationStatus.status, 'complete')
  assert.equal(manifest.annotationStatus.completedPageCount, 36)
  assert.equal(manifest.annotationStatus.totalMissingFieldCount, 0)
  assert.doesNotThrow(() => assertCompleteTranslationGold(manifest))
  const schema = JSON.parse(
    readFileSync(new URL(manifest.annotationSchema, fixtureRoot), 'utf8'),
  )
  const validate = new Ajv2020({ strict: true, allErrors: true }).compile(schema)
  const regions = []

  for (const image of manifest.images) {
    assert.equal(
      typeof image.annotation,
      'string',
      `chapter-5 fixture page ${image.order} is missing its annotation path`,
    )
    const annotation = JSON.parse(
      readFileSync(new URL(image.annotation, fixtureRoot), 'utf8'),
    )
    assert.equal(
      validate(annotation),
      true,
      `${image.annotation}: ${JSON.stringify(validate.errors)}`,
    )
    assert.equal(annotation.regions.length, image.expectedRegionCount)
    assert.equal(
      annotation.regions.filter((region) => ['dialogue', 'thought'].includes(region.kind)).length,
      image.expectedDialogueBubbleCount,
    )
    assert.equal(
      annotation.regions.filter((region) => region.kind === 'narration').length,
      image.expectedNarrationCount,
    )
    assert.equal(
      annotation.regions.filter((region) => region.translationTarget !== false).length,
      image.expectedEnglishTranslationTargetCount,
    )
    assert.equal(
      annotation.regions.filter((region) => region.translationTarget === false).length,
      image.expectedUntouchedExclusionCount,
    )
    const pagePrefix = `30ysp-ch5-p${String(image.order).padStart(3, '0')}-r`
    assert.ok(
      annotation.regions.every(
        (region) =>
          region.id.startsWith(pagePrefix) && /^30ysp-ch5-p\d{3}-r\d{2}$/u.test(region.id),
      ),
    )
    regions.push(...annotation.regions)
  }

  const targets = regions.filter((region) => region.translationTarget !== false)
  const exclusions = regions.filter((region) => region.translationTarget === false)
  const detectorGold = regions.filter((region) => ['dialogue', 'thought'].includes(region.kind))
  const narration = regions.filter((region) => region.kind === 'narration')
  assert.equal(regions.length, manifest.totalExpectedRegionCount)
  assert.equal(detectorGold.length, manifest.totalExpectedDialogueBubbleCount)
  assert.equal(narration.length, manifest.totalExpectedNarrationCount)
  assert.equal(targets.length, manifest.totalExpectedEnglishTranslationTargetCount)
  assert.equal(exclusions.length, manifest.totalExpectedUntouchedExclusionCount)
  assert.ok(
    targets.every((region) =>
      /[A-Za-z\u00c0-\u024f\u1e00-\u1eff]/u.test(region.sourceEnglish),
    ),
  )
  assert.ok(regions.every((region) => region.textPolygon && region.eraseMask?.polygon))
  assert.ok(regions.some((region) => region.bubblePolygon === undefined))
})
