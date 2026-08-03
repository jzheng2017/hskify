import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import {
  CORE_CHAPTER_IDS,
  DEFAULT_MANIFEST_PATH,
  REQUIRED_READER_KINDS,
  STRESS_CHAPTER_IDS,
  auditCorpus,
  validateManifest,
} from './real-reader-corpus.mjs'
import {
  annotationCoverage,
  chapterMarkup,
  requiredBrowserConfig,
} from './run-real-reader-browser-regression.mjs'
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

function completeManifest() {
  const chapters = [...CORE_CHAPTER_IDS, ...STRESS_CHAPTER_IDS].map((id, index) => {
    const objectSha = `${String(index + 1).padStart(2, '0')}${'a'.repeat(62)}`
    const annotationSha = `${String(index + 1).padStart(2, '0')}${'b'.repeat(62)}`
    return {
      id,
      provenance: {
        provider: index < CORE_CHAPTER_IDS.length ? 'webtoon' : 'asura',
        chapterUrl: `https://reader.test/${id}`,
        capturedAtUtc: '2026-08-02T00:00:00Z',
      },
      reader: { kind: REQUIRED_READER_KINDS[index % REQUIRED_READER_KINDS.length] },
      pageCount: 1,
      pages: [
        {
          order: 0,
          object: {
            path: `objects/${objectSha}.png`,
            sha256: objectSha,
            bytes: 1,
            mimeType: 'image/png',
            width: 1,
            height: 1,
          },
          annotation: {
            path: `annotations/${id}/0001.json`,
            sha256: annotationSha,
            bytes: 1,
          },
        },
      ],
      coverage: { annotatedPageCount: 1, storyTargetCount: 1, exclusionCount: 0 },
    }
  })
  return {
    schemaVersion: 2,
    corpusId: 'real-reader-v2',
    completeness: {
      state: 'complete',
      chapterCount: chapters.length,
      pageCount: chapters.length,
      annotationCount: chapters.length,
    },
    execution: {
      networkPolicy: 'forbidden',
      defaultCorpusRoot: 'local-corpus/real-reader-v2',
    },
    coverageRequirements: {
      coreChapterIds: CORE_CHAPTER_IDS,
      stressChapterIds: STRESS_CHAPTER_IDS,
      readerKinds: REQUIRED_READER_KINDS,
    },
    selections: {
      core: CORE_CHAPTER_IDS,
      stress: STRESS_CHAPTER_IDS,
      all: [...CORE_CHAPTER_IDS, ...STRESS_CHAPTER_IDS],
    },
    chapters,
  }
}

function writeCompleteCorpus(manifestRoot, corpusRoot) {
  const manifest = completeManifest()
  const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    'base64',
  )
  const pageAnnotation = (chapterId, objectSha) => ({
    schemaVersion: 2,
    chapterId,
    pageOrder: 0,
    sourceSha256: objectSha,
    regions: [
      {
        id: 'region-1',
        role: 'dialogue',
        sourceEnglish: 'Hello',
        polygon: [
          { x: 0.1, y: 0.1 },
          { x: 0.9, y: 0.1 },
          { x: 0.9, y: 0.9 },
          { x: 0.1, y: 0.9 },
        ],
        readingOrder: 0,
        continuationGroup: null,
        entities: [],
        styleRuns: [{ start: 0, end: 5, fontCategory: 'dialogue' }],
        protectedArtwork: false,
        cleanupAllowance: null,
        reviewedTranslations: {
          natural: [{ chinese: '你好', teachingTerms: [] }],
          strict: [{ chinese: '你好', teachingTerms: [] }],
        },
      },
    ],
    exclusions: [],
  })
  for (const chapter of manifest.chapters) {
    const page = chapter.pages[0]
    const objectBytes = png
    const objectSha = createHash('sha256').update(objectBytes).digest('hex')
    page.object = {
      path: `objects/${objectSha}.png`,
      sha256: objectSha,
      bytes: objectBytes.length,
      mimeType: 'image/png',
      width: 1,
      height: 1,
    }
    const annotation = pageAnnotation(chapter.id, objectSha)
    const annotationBytes = Buffer.from(`${JSON.stringify(annotation)}\n`)
    const annotationSha = createHash('sha256').update(annotationBytes).digest('hex')
    page.annotation = {
      path: `annotations/${chapter.id}/0001.json`,
      sha256: annotationSha,
      bytes: annotationBytes.length,
    }
    mkdirSync(join(corpusRoot, 'objects'), { recursive: true })
    mkdirSync(join(manifestRoot, 'annotations', chapter.id), { recursive: true })
    writeFileSync(join(corpusRoot, page.object.path), objectBytes)
    writeFileSync(join(manifestRoot, page.annotation.path), annotationBytes)
  }
  const manifestPath = join(manifestRoot, 'manifest.json')
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
  return manifestPath
}

test('the tracked v2 manifest is capture-required and cannot be green', () => {
  const manifest = JSON.parse(readFileSync(DEFAULT_MANIFEST_PATH, 'utf8'))
  const failures = validateManifest(manifest).filter((item) => !item.passed)
  assert.equal(
    failures.some((item) => item.id === 'manifest.capture-complete'),
    true,
  )
  const result = auditCorpus({ selection: 'smoke' })
  assert.equal(result.status, 'failed')
  assert.equal(result.captureRequired, true)
  assert.match(result.remediation.message, /real-reader-v2 corpus is not complete/u)
})

test('v2 requires all core/stress chapters, ordered pages, and coverage metadata', () => {
  const manifest = completeManifest()
  assert.deepEqual(
    validateManifest(manifest).filter((item) => !item.passed),
    [],
  )
  manifest.chapters[0].pages[0].order = 1
  assert.equal(
    validateManifest(manifest).some((item) => item.id === 'chapter.1.page-order' && !item.passed),
    true,
  )
  manifest.chapters[0].pages[0].order = 0
  manifest.chapters.pop()
  assert.equal(
    validateManifest(manifest).some((item) => item.id === 'coverage.chapter-set' && !item.passed),
    true,
  )
})

test('missing local v2 objects and annotations are explicit machine-readable failures', () => {
  const emptyCorpus = mkdtempSync(join(tmpdir(), 'hskify-real-reader-empty-'))
  const manifestRoot = mkdtempSync(join(tmpdir(), 'hskify-real-reader-manifest-'))
  try {
    const manifestPath = join(manifestRoot, 'manifest.json')
    writeFileSync(manifestPath, `${JSON.stringify(completeManifest(), null, 2)}\n`)
    const result = auditCorpus({ manifestPath, corpusRoot: emptyCorpus, selection: 'core' })
    assert.equal(result.status, 'failed')
    assert.equal(result.caseCount, CORE_CHAPTER_IDS.length)
    assert.equal(result.missingCount, CORE_CHAPTER_IDS.length)
    assert.equal(result.missingAnnotationCount, CORE_CHAPTER_IDS.length)
    assert.equal(result.remediation.requiredPaths.length, CORE_CHAPTER_IDS.length * 2)
    assert.match(result.remediation.message, /will not download/u)
  } finally {
    rmSync(emptyCorpus, { recursive: true, force: true })
    rmSync(manifestRoot, { recursive: true, force: true })
  }
})

test('v2 verifies annotation bytes and rejects incomplete page evidence', () => {
  const corpusRoot = mkdtempSync(join(tmpdir(), 'hskify-real-reader-valid-'))
  const manifestRoot = mkdtempSync(join(tmpdir(), 'hskify-real-reader-valid-manifest-'))
  try {
    const manifestPath = writeCompleteCorpus(manifestRoot, corpusRoot)
    const passing = auditCorpus({ manifestPath, corpusRoot, selection: 'core' })
    assert.equal(passing.status, 'passed')
    assert.equal(passing.verifiedCount, CORE_CHAPTER_IDS.length)
    assert.equal(passing.verifiedAnnotationCount, CORE_CHAPTER_IDS.length)

    const coverageMismatch = JSON.parse(readFileSync(manifestPath, 'utf8'))
    coverageMismatch.chapters[0].coverage.storyTargetCount = 2
    writeFileSync(manifestPath, `${JSON.stringify(coverageMismatch, null, 2)}\n`)
    const mismatch = auditCorpus({ manifestPath, corpusRoot, selection: 'core' })
    assert.equal(mismatch.status, 'failed')
    assert.equal(
      mismatch.failures.some((item) => item.id === `coverage.${CORE_CHAPTER_IDS[0]}.recomputed`),
      true,
    )

    writeCompleteCorpus(manifestRoot, corpusRoot)

    const malformedPath = join(manifestRoot, 'annotations', CORE_CHAPTER_IDS[0], '0001.json')
    const malformed = JSON.parse(readFileSync(malformedPath, 'utf8'))
    malformed.regions[0].reviewedTranslations = undefined
    writeFileSync(malformedPath, `${JSON.stringify(malformed)}\n`)
    const failed = auditCorpus({ manifestPath, corpusRoot, selection: 'core' })
    assert.equal(failed.status, 'failed')
    assert.equal(
      failed.failures.some(
        (item) => item.id === `annotation.${CORE_CHAPTER_IDS[0]}-page-0001.shape`,
      ),
      true,
    )
  } finally {
    rmSync(corpusRoot, { recursive: true, force: true })
    rmSync(manifestRoot, { recursive: true, force: true })
  }
})

test('v2 corpus validation fails closed for malformed region records', () => {
  const manifest = completeManifest()
  const chapter = manifest.chapters[0]
  // The release gate must report bad annotation evidence, not throw while
  // trying to spread a missing sourceEnglish value.
  const malformed = {
    schemaVersion: 2,
    chapterId: chapter.id,
    pageOrder: 0,
    sourceSha256: chapter.pages[0].object.sha256,
    regions: [
      {
        id: 'region-1',
        role: 'dialogue',
        polygon: [
          { x: 0.1, y: 0.1 },
          { x: 0.9, y: 0.1 },
          { x: 0.9, y: 0.9 },
        ],
        readingOrder: 0,
        reviewedTranslations: { natural: [], strict: [] },
      },
    ],
    exclusions: [],
  }
  const result = validateManifest({
    ...manifest,
    chapters: manifest.chapters.map((candidate, index) =>
      index === 0
        ? {
            ...candidate,
            // The manifest remains structurally complete; the malformed
            // annotation is exercised through the exported page validator
            // path in the audit test below.
          }
        : candidate,
    ),
  })
  assert.equal(result.every((item) => item.passed), true)
  // `validPageAnnotation` is intentionally exercised via a temporary local
  // corpus so the malformed record is handled as a normal shape failure.
  const corpusRoot = mkdtempSync(join(tmpdir(), 'hskify-real-reader-malformed-'))
  const manifestRoot = mkdtempSync(join(tmpdir(), 'hskify-real-reader-malformed-manifest-'))
  try {
    const manifestPath = writeCompleteCorpus(manifestRoot, corpusRoot)
    const annotationPath = join(manifestRoot, 'annotations', chapter.id, '0001.json')
    writeFileSync(annotationPath, `${JSON.stringify(malformed)}\n`)
    const failed = auditCorpus({ manifestPath, corpusRoot, selection: 'core' })
    assert.equal(failed.status, 'failed')
    assert.equal(
      failed.failures.some(
        (item) => item.id === `annotation.${chapter.id}-page-0001.shape`,
      ),
      true,
    )
  } finally {
    rmSync(corpusRoot, { recursive: true, force: true })
    rmSync(manifestRoot, { recursive: true, force: true })
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
  assert.equal(
    passing.every((item) => item.passed),
    true,
  )

  const failures = assertSemanticExpectations(item, [
    region({ sourceEnglish: 'Thanks.', displayedChinese: '谢谢。' }),
    region({ id: 'region-2', sourceEnglish: 'DING', displayedChinese: '叮' }),
  ]).filter((item) => !item.passed)
  assert.deepEqual(
    failures.map((item) => item.id),
    ['semantic.dense-page.required-source.Seongeun', 'semantic.dense-page.excluded-source.DING'],
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
  assert.equal(
    passing.every((assertion) => assertion.passed),
    true,
  )

  const translated = assertSemanticExpectations(
    item,
    [region({ sourceEnglish: 'MYUNGWANG SWORD AUTHORITY' })],
    [{ sourceEnglish: 'MyUNGWANG SHORd AUTHORITY' }],
  )
  assert.equal(
    translated.some((assertion) => !assertion.passed),
    true,
  )
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
      region({ sourceEnglish: 'THIS FEAR IS IMPRINTED IN MY BLOOD.' }),
      region({
        id: 'region-2',
        sourceEnglish: "THERE'S NO WAY THE WHITE TIGER TRIBE WOULD KNOW THE FEAR OF DEATH.",
      }),
    ],
    [{ sourceEnglish: 'THIRD SWORD' }],
  )
  assert.equal(
    evaluated.every((assertion) => assertion.passed),
    true,
  )
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
  assert.equal(
    evaluated.assertions.every((item) => item.passed),
    true,
  )

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
        region: region({
          patch: {
            blobId: 'patch-1',
            mimeType: 'image/png',
            rect: { x: 0.1, y: 0.1, width: 0.2, height: 0.1 },
          },
        }),
      },
    ],
    [{ blobId: 'patch-1', validPng: true }],
  )
  assert.equal(
    safe.assertions.every((assertion) => assertion.passed),
    true,
  )

  const damaged = assertCompletedJob(
    item,
    2,
    { type: 'complete' },
    [
      {
        type: 'regionReady',
        region: region({
          patch: {
            blobId: 'patch-1',
            mimeType: 'image/png',
            rect: { x: 0.2, y: 0.45, width: 0.2, height: 0.1 },
          },
        }),
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
  assert.equal(
    evaluated.assertions.every((item) => item.passed),
    true,
  )

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
  assert.equal(
    assertHskDifferential(low, differentHigh).every((item) => item.passed),
    true,
  )
})

test('packaged browser reader replica preserves canonical page order and surface adapters', () => {
  const chapter = {
    id: 'fixture-reader',
    reader: { kind: 'canvas' },
    pages: [
      {
        order: 0,
        object: { width: 800, height: 1200 },
      },
      {
        order: 1,
        object: { width: 900, height: 18000 },
      },
    ],
  }
  const html = chapterMarkup(chapter, [...chapter.pages], '')
  assert.match(html, /id="chapter"/u)
  assert.match(html, /data-page="1"/u)
  assert.match(html, /data-page="2"/u)
  assert.match(html, /data-reader-surface="canvas"/u)
  assert.match(html, /__hskifyReaderCanvasPromises/u)
})

test('packaged browser reader replica emits each manifest reader kind without synthetic daemon regions', () => {
  const page = { order: 0, object: { width: 640, height: 960 } }
  for (const kind of REQUIRED_READER_KINDS) {
    const html = chapterMarkup(
      { id: `fixture-${kind}`, reader: { kind }, pages: [page] },
      [page],
      '',
    )
    assert.match(html, new RegExp(`data-reader-kind="${kind}"`, 'u'))
    if (kind === 'iframe-image') assert.match(html, /<iframe/u)
    else if (kind === 'background') assert.match(html, /background-image:/u)
    else if (kind === 'canvas' || kind === 'webgl') {
      assert.match(html, new RegExp(`data-reader-surface="${kind}"`, 'u'))
    } else assert.match(html, /<img/u)
  }
})

test('packaged browser runner fails closed when its release prerequisites are absent', () => {
  const error = requiredBrowserConfig(undefined)
  assert.match(error ?? '', /Packaged Firefox config is missing/u)
  assert.equal(requiredBrowserConfig({}), error)
})

test('release regression entry point cannot regress to daemon-only HTTP jobs', () => {
  const source = readFileSync(new URL('./run-real-reader-regression.mjs', import.meta.url), 'utf8')
  assert.equal(source.includes('/jobs'), false)
  assert.match(source, /runBrowserRegression/u)
})

test('packaged browser quality evidence matches accepted polygons to local annotations', () => {
  const corpusRoot = mkdtempSync(join(tmpdir(), 'hskify-reader-browser-corpus-'))
  const manifestRoot = mkdtempSync(join(tmpdir(), 'hskify-reader-browser-manifest-'))
  try {
    const manifestPath = writeCompleteCorpus(manifestRoot, corpusRoot)
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))
    const chapter = manifest.chapters[0]
    const coverage = annotationCoverage(chapter, manifestPath, {
      jobs: [
        {
          pageIndex: 0,
          updates: [
            {
              type: 'regionReady',
              region: {
                sourceEnglish: 'Hello',
                textPolygon: [
                  { x: 0.1, y: 0.1 },
                  { x: 0.9, y: 0.1 },
                  { x: 0.9, y: 0.9 },
                  { x: 0.1, y: 0.9 },
                ],
              },
            },
          ],
        },
      ],
    })
    assert.equal(coverage.matchedTargetCount, coverage.expectedTargetCount)
    assert.equal(coverage.modifiedExclusions.length, 0)
    assert.equal(coverage.ocrCer, 0)
  } finally {
    rmSync(corpusRoot, { recursive: true, force: true })
    rmSync(manifestRoot, { recursive: true, force: true })
  }
})
