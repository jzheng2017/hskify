import assert from 'node:assert/strict'
import test from 'node:test'

import {
  evaluateDocumentChapterBenchmark,
  evaluateImageRuntimeBenchmark,
} from './document-chapter-gates.mjs'

function runtimeCounters(overrides = {}) {
  return {
    vision: 0,
    ocr: 0,
    projector: 0,
    segmentation: 0,
    inpainting: 0,
    patch: 0,
    font: 0,
    ...overrides,
  }
}

function sample(overrides = {}) {
  return {
    sourceBlockCount: 300,
    sourceCharacterCount: 100_000,
    mounted: true,
    blockCount: 300,
    measures: { 'hskify-document-extraction-and-skeleton': 75 },
    longTaskSupported: true,
    longTaskObservationMode: 'event-loop-probe',
    scrollLongTasks: [{ durationMs: 40 }],
    scroll: { completed: true, steps: 12 },
    nativeEvidenceMatched: true,
    tokenizerIdentity: 'qwen3.5-tokenizer-test-identity',
    dispatches: [
      { reason: 'visible', itemIds: ['block-1'], tokenCount: 90 },
      { reason: 'ordered', itemIds: ['block-2', 'block-3'], tokenCount: 180 },
    ],
    runtimeInitializations: runtimeCounters(),
    runtimeInvocations: runtimeCounters(),
    ...overrides,
  }
}

test('accepts raw evidence satisfying every document performance bound', () => {
  assert.equal(evaluateDocumentChapterBenchmark([sample(), sample()]).status, 'passed')
})

test('rejects p95, long-task, batch, workload, and vision violations', () => {
  const result = evaluateDocumentChapterBenchmark([
    sample({
      sourceBlockCount: 299,
      measures: { 'hskify-document-extraction-and-skeleton': 100 },
      scrollLongTasks: [{ durationMs: 51 }],
      dispatches: [{ reason: 'ordered', itemIds: Array.from({ length: 7 }, (_, index) => `block-${index}`), tokenCount: 500 }],
      runtimeInvocations: runtimeCounters({ vision: 1 }),
    }),
  ])
  assert.equal(result.status, 'failed')
  assert.deepEqual(
    result.gates.filter((entry) => !entry.passed).map((entry) => entry.id),
    [
      'representative-workload',
      'extraction-and-skeleton-p95',
      'maximum-long-task',
      'first-visible-dispatch-alone',
      'token-aware-batch-bound',
      'language-only-runtime',
    ],
  )
})

test('fails closed without browser long-task support or job-matched native counters', () => {
  const result = evaluateDocumentChapterBenchmark([
    sample({
      longTaskSupported: false,
      longTaskObservationMode: undefined,
      nativeEvidenceMatched: false,
      tokenizerIdentity: undefined,
      dispatches: undefined,
      runtimeInitializations: undefined,
      runtimeInvocations: undefined,
    }),
  ])
  assert.equal(result.status, 'failed')
  assert.deepEqual(
    result.gates.filter((entry) => !entry.passed).map((entry) => entry.id),
    [
      'complete-long-task-observation',
      'complete-native-evidence',
      'first-visible-dispatch-alone',
      'token-aware-batch-bound',
      'language-only-runtime',
    ],
  )
})

test('rejects a trace that stops after the first visible dispatch', () => {
  const result = evaluateDocumentChapterBenchmark([
    sample({
      dispatches: [{ reason: 'visible', itemIds: ['block-1'], tokenCount: 90 }],
    }),
  ])
  assert.equal(result.status, 'failed')
  assert.deepEqual(
    result.gates.filter((entry) => !entry.passed).map((entry) => entry.id),
    ['first-visible-dispatch-alone'],
  )
})

test('enforces the image throughput and 16 GiB VRAM regression bounds', () => {
  assert.equal(
    evaluateImageRuntimeBenchmark({
      measurementSource: 'packaged-firefox-native-evidence',
      browserRegressionPassed: true,
      nativeJobCount: 2,
      matchedNativeJobCount: 2,
      workloadSha256: 'a'.repeat(64),
      baselineWorkloadSha256: 'a'.repeat(64),
      tokenizerIdentity: 'tokenizer',
      baselineTokenizerIdentity: 'tokenizer',
      baselineLanguageUnitCount: 100,
      baselineLanguageGenerationDurationMs: 10_000,
      languageUnitCount: 90,
      languageGenerationDurationMs: 10_000,
      gpuSampleCount: 10,
      peakVramMiB: 16_384,
    }).status,
    'passed',
  )
  assert.equal(
    evaluateImageRuntimeBenchmark({
      measurementSource: 'packaged-firefox-native-evidence',
      browserRegressionPassed: true,
      nativeJobCount: 2,
      matchedNativeJobCount: 2,
      workloadSha256: 'a'.repeat(64),
      baselineWorkloadSha256: 'a'.repeat(64),
      tokenizerIdentity: 'tokenizer',
      baselineTokenizerIdentity: 'tokenizer',
      baselineLanguageUnitCount: 100,
      baselineLanguageGenerationDurationMs: 10_000,
      languageUnitCount: 899,
      languageGenerationDurationMs: 100_000,
      gpuSampleCount: 10,
      peakVramMiB: 16_385,
    }).status,
    'failed',
  )
})

test('rejects hand-entered image rates without raw native and GPU measurements', () => {
  const result = evaluateImageRuntimeBenchmark({
    baselineLanguageUnitsPerSecond: 10,
    currentLanguageUnitsPerSecond: 10,
    peakVramMiB: 1,
  })
  assert.equal(result.status, 'failed')
  assert.deepEqual(
    result.gates.filter((entry) => !entry.passed).map((entry) => entry.id),
    [
      'measured-image-runtime',
      'same-image-workload-baseline',
      'image-language-throughput',
      'image-runtime-vram',
    ],
  )
})
