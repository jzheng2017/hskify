import assert from 'node:assert/strict'
import { isAbsolute } from 'node:path'
import test from 'node:test'

import { evaluateImageRuntimeBenchmark } from './benchmark/document-chapter-gates.mjs'
import {
  baselineArtifact,
  freshImageRuntimeLayout,
  imageLanguageMeasurement,
  imageRuntimeEvidence,
  imageWorkload,
  parseImageRuntimeArguments,
  parseNvidiaSmiSample,
  workloadSha256,
} from './run-image-runtime-benchmark.mjs'

function summary(settings = {
  sourceLanguage: 'en',
  targetLanguage: 'zh-CN',
  hskStandard: '2.0',
  hskLevel: 3,
  learningMode: 'natural',
}) {
  return {
    status: 'passed',
    corpusId: 'real-reader-v2',
    selection: 'core',
    chapterRuns: [{
      chapterId: 'chapter-one',
      hskLevel: 3,
      readerKind: 'continuous-image',
      monitor: {
        observations: [{
          jobId: 'job-one',
          sourceKind: 'image',
          pageIndex: 0,
          sourceSha256: 'a'.repeat(64),
          sourceWidth: 800,
          sourceHeight: 1_200,
          submittedRequest: {
            buildFingerprint: 'hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10',
            clientImageId: 'chapter-one-page-one',
            sourceSha256: 'a'.repeat(64),
            sourceMimeType: 'image/png',
            naturalWidth: 800,
            naturalHeight: 1_200,
            pageSessionId: '00000000-0000-4000-8000-000000000000',
            sourceIndex: 0,
            chapterSourceOrder: [0],
            surfaceKind: 'image',
            visibleRects: [{ x: 0, y: 0, width: 1, height: 1 }],
            readingDirection: 'ltr',
            settings,
          },
        }],
      },
    }],
  }
}

function nativeEvidence() {
  return {
    samples: [{
      kind: 'image',
      jobId: 'job-one',
      tokenizerIdentity: 'tokenizer-sha256',
      languageUnitCount: 90,
      languageGenerationDurationMs: 9_000,
    }],
  }
}

test('builds a settings-sensitive canonical same-workload identity', () => {
  const natural = imageWorkload(summary())
  const strict = imageWorkload(summary({
    sourceLanguage: 'en',
    targetLanguage: 'zh-CN',
    hskStandard: '2.0',
    hskLevel: 3,
    learningMode: 'strict',
  }))
  assert.match(workloadSha256(natural), /^[a-f0-9]{64}$/u)
  assert.notEqual(workloadSha256(natural), workloadSha256(strict))
})

test('rejects malformed or oversized image request workload arrays', () => {
  const malformed = summary()
  malformed.chapterRuns[0].monitor.observations[0].submittedRequest.chapterSourceOrder = [0, 0]
  assert.throws(() => imageWorkload(malformed), /unique/u)

  const oversized = summary()
  oversized.chapterRuns[0].monitor.observations[0].submittedRequest.chapterSourceOrder =
    Array.from({ length: 100_001 }, (_, index) => index)
  assert.throws(() => imageWorkload(oversized), /1-100000 entries/u)
})

test('derives a fresh profile and empty result-cache location from only the installed marker', () => {
  const layout = freshImageRuntimeLayout(
    {
      stateDirectory: 'C:\\installed-state',
      profileDirectory: 'C:\\persistent-profile',
      extensionPackagePath: 'extension.zip',
    },
    'C:\\benchmark-output',
    'fixed-run',
  )
  assert.equal(layout.benchmarkConfig.stateDirectory, 'C:\\benchmark-output\\runs\\fixed-run\\native-state')
  assert.equal(layout.benchmarkConfig.profileDirectory, 'C:\\benchmark-output\\runs\\fixed-run\\firefox-profile')
  assert.equal(layout.sourceReadinessMarker, 'C:\\installed-state\\browser-cache\\browser-runtime\\models.ready')
  assert.equal(layout.targetReadinessMarker, 'C:\\benchmark-output\\runs\\fixed-run\\native-state\\browser-cache\\browser-runtime\\models.ready')
  assert.equal(layout.resultCacheDirectory, 'C:\\benchmark-output\\runs\\fixed-run\\native-state\\browser-cache\\results')
})

test('joins measured native image language work through exact browser job ids', () => {
  const measurement = imageLanguageMeasurement(summary(), nativeEvidence())
  assert.deepEqual(measurement, {
    tokenizerIdentity: 'tokenizer-sha256',
    nativeJobCount: 1,
    matchedNativeJobCount: 1,
    languageUnitCount: 90,
    languageGenerationDurationMs: 9_000,
    languageUnitsPerSecond: 10,
    jobs: [{
      jobId: 'job-one',
      languageUnitCount: 90,
      languageGenerationDurationMs: 9_000,
    }],
  })
  assert.throws(
    () => imageLanguageMeasurement(summary(), { samples: [] }),
    /did not exactly match browser jobs/u,
  )
})

test('parses one exact nvidia-smi memory sample', () => {
  assert.deepEqual(
    parseNvidiaSmiSample('0, GPU-0123, 16384, 7421\r\n', 123),
    {
      epochMs: 123,
      gpuIndex: 0,
      gpuUuid: 'GPU-0123',
      memoryTotalMiB: 16_384,
      memoryUsedMiB: 7_421,
    },
  )
  assert.throws(() => parseNvidiaSmiSample('0, GPU-a, 1, 1\n1, GPU-b, 1, 1'), /row count/u)
})

test('runner-generated baseline feeds the fail-closed image gate', () => {
  const browserSummary = summary()
  const workload = imageWorkload(browserSummary)
  const measurement = imageLanguageMeasurement(browserSummary, nativeEvidence())
  const gpu = { peakVramMiB: 15_000, sampleCount: 4 }
  const baseline = baselineArtifact(workload, {
    ...measurement,
    languageUnitCount: 100,
    languageGenerationDurationMs: 10_000,
  }, gpu)
  const evidence = imageRuntimeEvidence({
    summary: browserSummary,
    workload,
    measurement,
    gpu,
    baseline,
  })
  assert.equal(evaluateImageRuntimeBenchmark(evidence).status, 'passed')
})

test('requires an explicit measured baseline mode', () => {
  const previous = process.env.HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH
  delete process.env.HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH
  try {
    assert.throws(
      () => parseImageRuntimeArguments([]),
      /Provide a measured baseline/u,
    )
    const path = parseImageRuntimeArguments(['--record-baseline', 'baseline.json'])
      .recordBaselinePath
    assert.equal(isAbsolute(path), true)
    assert.match(path, /baseline\.json$/u)
  } finally {
    if (previous === undefined) delete process.env.HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH
    else process.env.HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH = previous
  }
})
