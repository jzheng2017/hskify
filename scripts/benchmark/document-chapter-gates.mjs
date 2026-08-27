import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const DOCUMENT_BENCHMARK_BLOCKS = 300
export const DOCUMENT_BENCHMARK_CHARACTERS = 100_000
export const DOCUMENT_SKELETON_P95_MS = 100
export const MAX_SCROLL_TASK_MS = 50
export const MAX_LANGUAGE_BATCH_UNITS = 6
export const MAX_IMAGE_LANGUAGE_REGRESSION = 0.1
export const MAX_IMAGE_RUNTIME_VRAM_MIB = 16 * 1024

const DOCUMENT_FORBIDDEN_RUNTIME_PATHS = [
  'vision',
  'ocr',
  'projector',
  'segmentation',
  'inpainting',
  'patch',
  'font',
]

function percentile(values, fraction) {
  if (values.length === 0) return undefined
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)]
}

function gate(id, passed, expected, actual) {
  return { id, passed, expected, actual }
}

function exactRuntimeCounters(value) {
  return value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    DOCUMENT_FORBIDDEN_RUNTIME_PATHS.every(
      (name) => Number.isInteger(value[name]) && value[name] >= 0,
    )
}

/** Evaluate raw packaged-Firefox samples without smoothing or inferred data. */
export function evaluateDocumentChapterBenchmark(samples) {
  if (!Array.isArray(samples) || samples.length === 0)
    return { status: 'failed', gates: [gate('samples', false, 'at least one raw sample', 0)] }

  const durations = samples.map(
    (sample) => sample.measures?.['hskify-document-extraction-and-skeleton'],
  )
  const validDurations = durations.filter(Number.isFinite)
  const longTasks = samples.flatMap((sample) => sample.scrollLongTasks ?? [])
  const dispatches = samples.flatMap((sample) => sample.dispatches ?? [])
  const nativeEvidenceComplete = samples.every(
    (sample) =>
      sample.nativeEvidenceMatched === true &&
      typeof sample.tokenizerIdentity === 'string' &&
      sample.tokenizerIdentity.length > 0 &&
      Array.isArray(sample.dispatches) &&
      sample.dispatches.length > 0 &&
      exactRuntimeCounters(sample.runtimeInitializations) &&
      exactRuntimeCounters(sample.runtimeInvocations),
  )
  const runtimeCounts = Object.fromEntries(
    DOCUMENT_FORBIDDEN_RUNTIME_PATHS.map((name) => [
      name,
      {
        initializations: samples.reduce(
          (total, sample) => total + Number(sample.runtimeInitializations?.[name] ?? 0),
          0,
        ),
        invocations: samples.reduce(
          (total, sample) => total + Number(sample.runtimeInvocations?.[name] ?? 0),
          0,
        ),
      },
    ]),
  )

  const p95 = percentile(validDurations, 0.95)
  const maximumLongTask = longTasks.length
    ? Math.max(...longTasks.map((entry) => Number(entry.durationMs)))
    : 0
  const gates = [
    gate(
      'representative-workload',
      samples.every(
        (sample) =>
          sample.sourceBlockCount === DOCUMENT_BENCHMARK_BLOCKS &&
          sample.sourceCharacterCount === DOCUMENT_BENCHMARK_CHARACTERS,
      ),
      `${DOCUMENT_BENCHMARK_BLOCKS} blocks and ${DOCUMENT_BENCHMARK_CHARACTERS} normalized characters per sample`,
      samples.map((sample) => ({
        blocks: sample.sourceBlockCount,
        characters: sample.sourceCharacterCount,
      })),
    ),
    gate(
      'reader-skeleton',
      samples.every(
        (sample) => sample.mounted === true && sample.blockCount === DOCUMENT_BENCHMARK_BLOCKS,
      ),
      `mounted in-place translation with ${DOCUMENT_BENCHMARK_BLOCKS} placeholders per sample`,
      samples.map((sample) => ({ mounted: sample.mounted, blockCount: sample.blockCount })),
    ),
    gate('complete-user-timing', validDurations.length === samples.length, samples.length, validDurations.length),
    gate('extraction-and-skeleton-p95', Number.isFinite(p95) && p95 < DOCUMENT_SKELETON_P95_MS, `< ${DOCUMENT_SKELETON_P95_MS} ms`, p95),
    gate(
      'scroll-exercise',
      samples.every(
        (sample) =>
          sample.scroll?.completed === true &&
          Number.isInteger(sample.scroll.steps) &&
          sample.scroll.steps > 0,
      ),
      'completed deterministic chapter scroll in every sample',
      samples.map((sample) => sample.scroll),
    ),
    gate(
      'complete-long-task-observation',
      samples.every(
        (sample) =>
          sample.longTaskSupported === true &&
          ['performance-observer', 'event-loop-probe'].includes(sample.longTaskObservationMode) &&
          Array.isArray(sample.scrollLongTasks),
      ),
      'native Long Tasks or conservative event-loop observation for every scroll sample',
      samples.map((sample) => ({
        supported: sample.longTaskSupported,
        mode: sample.longTaskObservationMode,
        entries: sample.scrollLongTasks?.length,
      })),
    ),
    gate('maximum-long-task', maximumLongTask <= MAX_SCROLL_TASK_MS, `<= ${MAX_SCROLL_TASK_MS} ms`, maximumLongTask),
    gate(
      'complete-native-evidence',
      nativeEvidenceComplete,
      'job-matched dispatch, tokenizer, initialization, and invocation counters per sample',
      samples.map((sample) => ({
        jobId: sample.jobId,
        matched: sample.nativeEvidenceMatched,
        tokenizerIdentity: sample.tokenizerIdentity,
        dispatchCount: sample.dispatches?.length,
        runtimeInitializations: sample.runtimeInitializations,
        runtimeInvocations: sample.runtimeInvocations,
      })),
    ),
    gate(
      'first-visible-dispatch-alone',
      samples.every(
        (sample) =>
          sample.dispatches?.length >= 2 &&
          sample.dispatches[0]?.reason === 'visible' &&
          sample.dispatches[0]?.itemIds?.length === 1,
      ),
      'one visible block first, followed by at least one measured batch in every sample',
      samples.map((sample) => ({
        dispatchCount: sample.dispatches?.length,
        firstDispatch: sample.dispatches?.[0],
      })),
    ),
    gate(
      'token-aware-batch-bound',
      dispatches.length > 0 &&
        dispatches.every(
          (dispatch) =>
            Array.isArray(dispatch.itemIds) &&
            dispatch.itemIds.length >= 1 &&
            dispatch.itemIds.length <= MAX_LANGUAGE_BATCH_UNITS &&
            Number.isInteger(dispatch.tokenCount) &&
            dispatch.tokenCount > 0,
        ),
      `1-${MAX_LANGUAGE_BATCH_UNITS} units with measured tokenizer counts`,
      dispatches,
    ),
    gate(
      'language-only-runtime',
      nativeEvidenceComplete && Object.values(runtimeCounts).every(
        (counts) => counts.initializations === 0 && counts.invocations === 0,
      ),
      'zero vision/OCR/projector/segmentation/inpainting/patch/font initializations or invocations',
      runtimeCounts,
    ),
  ]
  return {
    status: gates.every((entry) => entry.passed) ? 'passed' : 'failed',
    sampleCount: samples.length,
    extractionAndSkeletonP95Ms: p95,
    maximumLongTaskMs: maximumLongTask,
    runtimeCounters: runtimeCounts,
    gates,
  }
}

export function evaluateImageRuntimeBenchmark(evidence) {
  const baselineUnits = Number(evidence?.baselineLanguageUnitCount)
  const baselineDurationMs = Number(evidence?.baselineLanguageGenerationDurationMs)
  const currentUnits = Number(evidence?.languageUnitCount)
  const currentDurationMs = Number(evidence?.languageGenerationDurationMs)
  const baseline = baselineDurationMs > 0 ? baselineUnits / (baselineDurationMs / 1_000) : Number.NaN
  const current = currentDurationMs > 0 ? currentUnits / (currentDurationMs / 1_000) : Number.NaN
  const peakVram = Number(evidence?.peakVramMiB)
  const retainedRatio = baseline > 0 ? current / baseline : Number.NaN
  const completeMeasurement =
    evidence?.measurementSource === 'packaged-firefox-native-evidence' &&
    evidence?.browserRegressionPassed === true &&
    Number.isSafeInteger(evidence?.nativeJobCount) && evidence.nativeJobCount > 0 &&
    evidence?.matchedNativeJobCount === evidence.nativeJobCount &&
    Number.isSafeInteger(currentUnits) && currentUnits > 0 &&
    Number.isSafeInteger(currentDurationMs) && currentDurationMs > 0 &&
    Number.isSafeInteger(baselineUnits) && baselineUnits > 0 &&
    Number.isSafeInteger(baselineDurationMs) && baselineDurationMs > 0 &&
    Number.isSafeInteger(evidence?.gpuSampleCount) && evidence.gpuSampleCount > 0
  const sameWorkload =
    typeof evidence?.workloadSha256 === 'string' &&
    /^[a-f0-9]{64}$/u.test(evidence.workloadSha256) &&
    evidence.baselineWorkloadSha256 === evidence.workloadSha256 &&
    typeof evidence?.tokenizerIdentity === 'string' &&
    evidence.tokenizerIdentity.length > 0 &&
    evidence.baselineTokenizerIdentity === evidence.tokenizerIdentity
  const gates = [
    gate(
      'measured-image-runtime',
      completeMeasurement,
      'packaged-Firefox jobs matched to native timings and sampled GPU memory',
      {
        source: evidence?.measurementSource,
        browserRegressionPassed: evidence?.browserRegressionPassed,
        nativeJobCount: evidence?.nativeJobCount,
        matchedNativeJobCount: evidence?.matchedNativeJobCount,
        languageUnitCount: currentUnits,
        languageGenerationDurationMs: currentDurationMs,
        gpuSampleCount: evidence?.gpuSampleCount,
      },
    ),
    gate(
      'same-image-workload-baseline',
      sameWorkload,
      'exact workload and tokenizer identities match the measured baseline',
      {
        workloadSha256: evidence?.workloadSha256,
        baselineWorkloadSha256: evidence?.baselineWorkloadSha256,
        tokenizerIdentity: evidence?.tokenizerIdentity,
        baselineTokenizerIdentity: evidence?.baselineTokenizerIdentity,
      },
    ),
    gate(
      'image-language-throughput',
      completeMeasurement && sameWorkload && Number.isFinite(retainedRatio) &&
        retainedRatio >= 1 - MAX_IMAGE_LANGUAGE_REGRESSION,
      `>= ${(1 - MAX_IMAGE_LANGUAGE_REGRESSION) * 100}% of the same-workload pre-refactor baseline`,
      { baseline, current, retainedRatio },
    ),
    gate(
      'image-runtime-vram',
      completeMeasurement && Number.isFinite(peakVram) && peakVram > 0 &&
        peakVram <= MAX_IMAGE_RUNTIME_VRAM_MIB,
      `<= ${MAX_IMAGE_RUNTIME_VRAM_MIB} MiB`,
      peakVram,
    ),
  ]
  return { status: gates.every((entry) => entry.passed) ? 'passed' : 'failed', gates }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const path = process.argv[2]
  if (!path) throw new Error('usage: node document-chapter-gates.mjs <raw-samples.json>')
  const input = JSON.parse(readFileSync(path, 'utf8'))
  const document = evaluateDocumentChapterBenchmark(input.samples)
  const result = {
    status: document.status,
    document,
  }
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
  if (result.status !== 'passed') process.exitCode = 1
}
