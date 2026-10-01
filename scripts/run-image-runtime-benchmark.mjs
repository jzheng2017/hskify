/*
 * Local image-runtime benchmark. It drives the existing packaged-Firefox
 * real-reader regression, joins native language timings by jobId, and samples
 * the selected NVIDIA GPU for the complete run.
 */

import { execFile } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import { writeJsonSync } from './benchmark/browser-harness.mjs'
import { evaluateImageRuntimeBenchmark } from './benchmark/document-chapter-gates.mjs'
import {
  initializeNativeEvidence,
  readNativeEvidence,
  resolveNativeEvidencePath,
  waitForNativeSample,
} from './benchmark/native-evidence.mjs'
import {
  DEFAULT_CORPUS_ROOT,
  DEFAULT_MANIFEST_PATH,
} from './real-reader-corpus.mjs'
import {
  requiredBrowserConfig,
  runBrowserRegression,
} from './run-real-reader-browser-regression.mjs'

const execFileAsync = promisify(execFile)
const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const DEFAULT_OUTPUT = resolve(REPOSITORY_ROOT, '.cache/image-runtime-benchmark')
const BUILD_FINGERPRINT = 'hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10'
const GPU_SAMPLE_INTERVAL_MS = 100
const IMAGE_BASELINE_BENCHMARK = 'hskify-image-language-baseline'
const MAX_U32 = 0xffff_ffff
const MAX_CHAPTER_SOURCE_ORDER = 100_000
const IMAGE_REQUEST_FIELDS = [
  'buildFingerprint',
  'chapterSourceOrder',
  'clientImageId',
  'naturalHeight',
  'naturalWidth',
  'pageSessionId',
  'readingDirection',
  'settings',
  'sourceIndex',
  'sourceMimeType',
  'sourceSha256',
  'surfaceKind',
  'visibleRects',
]

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds))
}

function positiveNumber(value, name) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`${name} must be positive.`)
  return parsed
}

function nonNegativeNumber(value, name) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${name} must be non-negative.`)
  return parsed
}

function nonNegativeInteger(value, name) {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer.`)
  }
  return parsed
}

function nextValue(args, argument) {
  const value = args.shift()
  if (!value) throw new Error(`${argument} requires a value.`)
  return value
}

export function parseImageRuntimeArguments(argv) {
  const options = {
    manifestPath: DEFAULT_MANIFEST_PATH,
    corpusRoot: DEFAULT_CORPUS_ROOT,
    selection: 'core',
    caseId: undefined,
    configPath: process.env.HSKIFY_REAL_READER_BROWSER_CONFIG
      ? resolve(process.env.HSKIFY_REAL_READER_BROWSER_CONFIG)
      : undefined,
    outputDirectory: DEFAULT_OUTPUT,
    baselinePath: process.env.HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH
      ? resolve(process.env.HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH)
      : undefined,
    recordBaselinePath: undefined,
    nvidiaSmiPath: process.env.HSKIFY_NVIDIA_SMI_PATH,
    gpuIndex: process.env.HSKIFY_BENCH_GPU_INDEX
      ? nonNegativeInteger(process.env.HSKIFY_BENCH_GPU_INDEX, 'HSKIFY_BENCH_GPU_INDEX')
      : 0,
    timeoutMs: 5 * 60_000,
    headed: false,
  }
  const args = [...argv]
  while (args.length > 0) {
    const argument = args.shift()
    if (argument === '--manifest') options.manifestPath = resolve(nextValue(args, argument))
    else if (argument === '--corpus') options.corpusRoot = resolve(nextValue(args, argument))
    else if (argument === '--selection') options.selection = nextValue(args, argument)
    else if (argument === '--case') options.caseId = nextValue(args, argument)
    else if (argument === '--config' || argument === '--browser-config') {
      options.configPath = resolve(nextValue(args, argument))
    } else if (argument === '--output') {
      options.outputDirectory = resolve(nextValue(args, argument))
    } else if (argument === '--baseline') {
      options.baselinePath = resolve(nextValue(args, argument))
    } else if (argument === '--record-baseline') {
      options.recordBaselinePath = resolve(nextValue(args, argument))
    } else if (argument === '--nvidia-smi') {
      options.nvidiaSmiPath = nextValue(args, argument)
    } else if (argument === '--gpu-index') {
      options.gpuIndex = nonNegativeInteger(nextValue(args, argument), argument)
    } else if (argument === '--timeout-minutes') {
      options.timeoutMs = positiveNumber(nextValue(args, argument), argument) * 60_000
    } else if (argument === '--headed') options.headed = true
    else throw new Error(`Unknown image runtime benchmark argument: ${argument}.`)
  }
  if (options.baselinePath && options.recordBaselinePath) {
    throw new Error('--baseline and --record-baseline are mutually exclusive.')
  }
  if (!options.baselinePath && !options.recordBaselinePath) {
    throw new Error(
      'Provide a measured baseline with --baseline or HSKIFY_IMAGE_LANGUAGE_BASELINE_PATH; use --record-baseline only on the baseline build.',
    )
  }
  return options
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, canonical(value[key])]),
    )
  }
  return value
}

function exactKeys(value, expected, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${name} must be an object.`)
  }
  const actual = Object.keys(value).sort()
  const fields = [...expected].sort()
  if (actual.length !== fields.length || actual.some((field, index) => field !== fields[index])) {
    throw new Error(`${name} must contain the exact image request fields.`)
  }
}

function boundedString(value, name, maximum) {
  if (typeof value !== 'string' || value.length < 1 || value.length > maximum) {
    throw new Error(`${name} must contain 1-${maximum} characters.`)
  }
  return value
}

function boundedU32(value, name, minimum = 0) {
  if (!Number.isInteger(value) || value < minimum || value > MAX_U32) {
    throw new Error(`${name} is outside its integer bound.`)
  }
  return value
}

function exactSettings(settings) {
  exactKeys(
    settings,
    ['sourceLanguage', 'targetLanguage', 'hskStandard', 'hskLevel', 'learningMode'],
    'settings',
  )
  if (
    settings.sourceLanguage !== 'en' ||
    settings.targetLanguage !== 'zh-CN' ||
    settings.hskStandard !== '2.0' ||
    ![1, 2, 3, 4, 5, 6].includes(settings.hskLevel) ||
    !['natural', 'strict'].includes(settings.learningMode)
  ) throw new Error('settings contain an unsupported translation value.')
  return { ...settings }
}

function exactSourceOrder(value, sourceIndex) {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_CHAPTER_SOURCE_ORDER) {
    throw new Error(`chapterSourceOrder must contain 1-${MAX_CHAPTER_SOURCE_ORDER} entries.`)
  }
  const order = value.map((item, index) => boundedU32(item, `chapterSourceOrder[${index}]`))
  if (!order.includes(sourceIndex)) throw new Error('chapterSourceOrder must include sourceIndex.')
  if (new Set(order).size !== order.length) throw new Error('chapterSourceOrder must contain unique source identities in DOM order.')
  return order
}

function exactVisibleRects(value) {
  if (!Array.isArray(value) || value.length > 64) {
    throw new Error('visibleRects must contain at most 64 rectangles.')
  }
  return value.map((candidate, index) => {
    exactKeys(candidate, ['x', 'y', 'width', 'height'], `visibleRects[${index}]`)
    const { x, y, width, height } = candidate
    if (
      ![x, y, width, height].every(Number.isFinite) ||
      x < 0 || x > 1 || y < 0 || y > 1 ||
      width <= 0 || width > 1 || height <= 0 || height > 1 ||
      x + width > 1 + Number.EPSILON || y + height > 1 + Number.EPSILON
    ) throw new Error(`visibleRects[${index}] is outside the normalized source.`)
    return { x, y, width, height }
  })
}

export function workloadSha256(workload) {
  return createHash('sha256').update(JSON.stringify(canonical(workload)), 'utf8').digest('hex')
}

export function freshImageRuntimeLayout(config, outputDirectory, runId) {
  const runDirectory = resolve(outputDirectory, 'runs', runId)
  const stateDirectory = resolve(runDirectory, 'native-state')
  return {
    runDirectory,
    benchmarkConfig: {
      ...config,
      profileDirectory: resolve(runDirectory, 'firefox-profile'),
      stateDirectory,
    },
    sourceReadinessMarker: resolve(
      config.stateDirectory,
      'browser-cache',
      'browser-runtime',
      'models.ready',
    ),
    targetReadinessMarker: resolve(
      stateDirectory,
      'browser-cache',
      'browser-runtime',
      'models.ready',
    ),
    resultCacheDirectory: resolve(stateDirectory, 'browser-cache', 'results'),
  }
}

function exactSourceWorkload(record) {
  const request = record?.submittedRequest
  exactKeys(request, IMAGE_REQUEST_FIELDS, 'submitted image request')
  if (
    record?.sourceKind !== 'image' ||
    typeof record.jobId !== 'string' || record.jobId.length === 0 ||
    typeof record.sourceSha256 !== 'string' ||
    !/^[a-f0-9]{64}$/u.test(record.sourceSha256) ||
    !Number.isInteger(record.pageIndex) || record.pageIndex < 0 || record.pageIndex > MAX_U32 ||
    !Number.isSafeInteger(record.sourceWidth) || record.sourceWidth < 1 ||
    !Number.isSafeInteger(record.sourceHeight) || record.sourceHeight < 1 ||
    request.buildFingerprint !== BUILD_FINGERPRINT ||
    request.sourceSha256 !== record.sourceSha256 ||
    request.naturalWidth !== record.sourceWidth ||
    request.naturalHeight !== record.sourceHeight ||
    request.sourceIndex !== record.pageIndex
  ) throw new Error(`Image job ${record?.jobId ?? '<unknown>'} lacks canonical workload metadata.`)
  const sourceIndex = boundedU32(request.sourceIndex, 'sourceIndex')
  const sourceMimeType = boundedString(request.sourceMimeType, 'sourceMimeType', 128)
  if (!['image/png', 'image/jpeg', 'image/webp', 'image/gif'].includes(sourceMimeType)) {
    throw new Error('sourceMimeType is unsupported.')
  }
  boundedString(request.clientImageId, 'clientImageId', 512)
  boundedString(request.pageSessionId, 'pageSessionId', 256)
  if (!['image', 'background', 'canvas', 'webgl', 'frame'].includes(request.surfaceKind)) {
    throw new Error('surfaceKind is unsupported.')
  }
  if (!['ltr', 'rtl'].includes(request.readingDirection)) {
    throw new Error('readingDirection is unsupported.')
  }
  return {
    sourceIndex,
    sourceSha256: record.sourceSha256,
    sourceMimeType,
    naturalWidth: record.sourceWidth,
    naturalHeight: record.sourceHeight,
    chapterSourceOrder: exactSourceOrder(request.chapterSourceOrder, sourceIndex),
    surfaceKind: request.surfaceKind,
    readingDirection: request.readingDirection,
    settings: exactSettings(request.settings),
    visibleRects: exactVisibleRects(request.visibleRects),
  }
}

export function imageWorkload(summary) {
  if (
    typeof summary?.corpusId !== 'string' || summary.corpusId.length === 0 ||
    typeof summary?.selection !== 'string' || summary.selection.length === 0 ||
    !Array.isArray(summary.chapterRuns) || summary.chapterRuns.length === 0
  ) throw new Error('The real-reader summary contains no canonical image workload.')
  return {
    corpusId: summary.corpusId,
    selection: summary.selection,
    runs: summary.chapterRuns.map((run) => {
      const records = run?.monitor?.observations
      if (!Array.isArray(records) || records.length === 0) {
        throw new Error(`Chapter ${run?.chapterId ?? '<unknown>'} contains no observed image jobs.`)
      }
      return {
        chapterId: run.chapterId,
        hskLevel: run.hskLevel,
        readerKind: run.readerKind,
        sources: records.map(exactSourceWorkload),
      }
    }),
  }
}

function imageJobIds(summary) {
  const ids = summary.chapterRuns.flatMap((run) =>
    (run.monitor?.observations ?? [])
      .filter((record) => record.sourceKind === 'image')
      .map((record) => record.jobId),
  )
  if (ids.length === 0 || ids.some((id) => typeof id !== 'string' || id.length === 0)) {
    throw new Error('The packaged-Firefox regression observed no valid image job IDs.')
  }
  if (new Set(ids).size !== ids.length) throw new Error('The image benchmark observed a duplicate jobId.')
  return ids
}

export function imageLanguageMeasurement(summary, nativeEvidence) {
  const ids = imageJobIds(summary)
  const expected = new Set(ids)
  const nativeSamples = nativeEvidence.samples.filter((sample) => sample.kind === 'image')
  const matched = nativeSamples.filter((sample) => expected.has(sample.jobId))
  if (matched.length !== ids.length || nativeSamples.length !== ids.length) {
    throw new Error(
      `Native image evidence did not exactly match browser jobs (${matched.length}/${ids.length}).`,
    )
  }
  const tokenizers = new Set()
  let languageUnitCount = 0
  let languageGenerationDurationMs = 0
  for (const sample of matched) {
    if (typeof sample.tokenizerIdentity !== 'string' || sample.tokenizerIdentity.length === 0) {
      throw new Error(`Native image job ${sample.jobId} has no tokenizer identity.`)
    }
    if (!Number.isSafeInteger(sample.languageUnitCount) || sample.languageUnitCount < 0) {
      throw new Error(`Native image job ${sample.jobId} has an invalid languageUnitCount.`)
    }
    if (
      !Number.isSafeInteger(sample.languageGenerationDurationMs) ||
      sample.languageGenerationDurationMs < 0
    ) throw new Error(`Native image job ${sample.jobId} has an invalid language duration.`)
    tokenizers.add(sample.tokenizerIdentity)
    languageUnitCount += sample.languageUnitCount
    languageGenerationDurationMs += sample.languageGenerationDurationMs
  }
  if (tokenizers.size !== 1) throw new Error('Image jobs used different tokenizer identities.')
  if (languageUnitCount < 1 || languageGenerationDurationMs <= 0) {
    throw new Error('Native image evidence contains no completed language generation work.')
  }
  return {
    tokenizerIdentity: [...tokenizers][0],
    nativeJobCount: ids.length,
    matchedNativeJobCount: matched.length,
    languageUnitCount,
    languageGenerationDurationMs,
    languageUnitsPerSecond: languageUnitCount / (languageGenerationDurationMs / 1_000),
    jobs: matched.map((sample) => ({
      jobId: sample.jobId,
      languageUnitCount: sample.languageUnitCount,
      languageGenerationDurationMs: sample.languageGenerationDurationMs,
    })),
  }
}

export function parseNvidiaSmiSample(output, epochMs = Date.now()) {
  const lines = String(output).trim().split(/\r?\n/u).filter(Boolean)
  if (lines.length !== 1) throw new Error('nvidia-smi returned an unexpected GPU row count.')
  const fields = lines[0].split(',').map((field) => field.trim())
  if (fields.length !== 4) throw new Error('nvidia-smi returned an unexpected memory row.')
  const index = nonNegativeInteger(fields[0], 'nvidia-smi GPU index')
  const memoryTotalMiB = positiveNumber(fields[2], 'nvidia-smi total memory')
  const memoryUsedMiB = nonNegativeNumber(fields[3], 'nvidia-smi used memory')
  if (!fields[1] || memoryUsedMiB > memoryTotalMiB) {
    throw new Error('nvidia-smi returned invalid GPU identity or memory usage.')
  }
  return { epochMs, gpuIndex: index, gpuUuid: fields[1], memoryTotalMiB, memoryUsedMiB }
}

export async function queryNvidiaSmi(executable, gpuIndex) {
  const { stdout } = await execFileAsync(executable, [
    '-i',
    String(gpuIndex),
    '--query-gpu=index,uuid,memory.total,memory.used',
    '--format=csv,noheader,nounits',
  ], { timeout: 10_000, windowsHide: true, maxBuffer: 64 * 1024 })
  return parseNvidiaSmiSample(stdout)
}

export async function startNvidiaMemorySampler(
  { executable, gpuIndex, intervalMs = GPU_SAMPLE_INTERVAL_MS },
  query = queryNvidiaSmi,
) {
  const samples = [await query(executable, gpuIndex)]
  let stopped = false
  let failure
  const loop = (async () => {
    while (!stopped) {
      await delay(intervalMs)
      if (stopped) break
      try {
        samples.push(await query(executable, gpuIndex))
      } catch (error) {
        failure = error
        stopped = true
      }
    }
  })()
  return {
    async stop() {
      stopped = true
      await loop
      if (failure) throw failure
      return samples
    },
  }
}

function gpuMeasurement(samples) {
  if (!Array.isArray(samples) || samples.length === 0) {
    throw new Error('The image benchmark captured no GPU memory samples.')
  }
  const identity = samples[0]
  if (samples.some(
    (sample) => sample.gpuIndex !== identity.gpuIndex ||
      sample.gpuUuid !== identity.gpuUuid ||
      sample.memoryTotalMiB !== identity.memoryTotalMiB,
  )) throw new Error('nvidia-smi changed GPU identity during the benchmark.')
  return {
    gpuIndex: identity.gpuIndex,
    gpuUuid: identity.gpuUuid,
    memoryTotalMiB: identity.memoryTotalMiB,
    sampleIntervalMs: GPU_SAMPLE_INTERVAL_MS,
    sampleCount: samples.length,
    peakVramMiB: Math.max(...samples.map((sample) => sample.memoryUsedMiB)),
    samples,
  }
}

export function baselineArtifact(workload, measurement, gpu) {
  return {
    schemaVersion: 1,
    benchmark: IMAGE_BASELINE_BENCHMARK,
    transport: 'packaged-firefox-local-reader',
    cacheState: 'fresh-empty-result-cache',
    workloadSha256: workloadSha256(workload),
    workload,
    measurement: {
      tokenizerIdentity: measurement.tokenizerIdentity,
      nativeJobCount: measurement.nativeJobCount,
      languageUnitCount: measurement.languageUnitCount,
      languageGenerationDurationMs: measurement.languageGenerationDurationMs,
      peakVramMiB: gpu.peakVramMiB,
      gpuSampleCount: gpu.sampleCount,
    },
  }
}

function readBaseline(path) {
  const baseline = JSON.parse(readFileSync(path, 'utf8'))
  if (
    baseline?.schemaVersion !== 1 ||
    baseline?.benchmark !== IMAGE_BASELINE_BENCHMARK ||
    baseline?.transport !== 'packaged-firefox-local-reader' ||
    baseline?.cacheState !== 'fresh-empty-result-cache' ||
    typeof baseline?.workloadSha256 !== 'string' ||
    baseline.workloadSha256 !== workloadSha256(baseline.workload) ||
    typeof baseline?.measurement?.tokenizerIdentity !== 'string' ||
    baseline.measurement.tokenizerIdentity.length === 0 ||
    !Number.isSafeInteger(baseline.measurement.languageUnitCount) ||
    baseline.measurement.languageUnitCount < 1 ||
    !Number.isSafeInteger(baseline.measurement.languageGenerationDurationMs) ||
    baseline.measurement.languageGenerationDurationMs <= 0
  ) throw new Error('The image language baseline is not a runner-generated measured artifact.')
  return baseline
}

export function imageRuntimeEvidence({
  summary,
  workload,
  measurement,
  gpu,
  baseline,
}) {
  return {
    measurementSource: 'packaged-firefox-native-evidence',
    browserRegressionPassed: summary.status === 'passed',
    workloadSha256: workloadSha256(workload),
    baselineWorkloadSha256: baseline.workloadSha256,
    tokenizerIdentity: measurement.tokenizerIdentity,
    baselineTokenizerIdentity: baseline.measurement.tokenizerIdentity,
    nativeJobCount: measurement.nativeJobCount,
    matchedNativeJobCount: measurement.matchedNativeJobCount,
    languageUnitCount: measurement.languageUnitCount,
    languageGenerationDurationMs: measurement.languageGenerationDurationMs,
    baselineLanguageUnitCount: baseline.measurement.languageUnitCount,
    baselineLanguageGenerationDurationMs:
      baseline.measurement.languageGenerationDurationMs,
    gpuSampleCount: gpu.sampleCount,
    peakVramMiB: gpu.peakVramMiB,
  }
}

export async function runImageRuntimeBenchmark(options) {
  if (!options.configPath || !existsSync(options.configPath)) {
    throw new Error(
      'A packaged-Firefox config is required via --config or HSKIFY_REAL_READER_BROWSER_CONFIG.',
    )
  }
  const config = JSON.parse(readFileSync(options.configPath, 'utf8'))
  const configError = requiredBrowserConfig(config)
  if (configError) throw new Error(configError)
  mkdirSync(options.outputDirectory, { recursive: true })
  const rawPath = resolve(options.outputDirectory, 'raw-samples.json')
  const evaluationPath = resolve(options.outputDirectory, 'evaluation.json')
  const realReaderOutput = resolve(options.outputDirectory, 'real-reader')
  const nativeEvidencePath = resolveNativeEvidencePath({
    configuredPath: config.nativeEvidencePath,
    outputDirectory: options.outputDirectory,
  })
  if (nativeEvidencePath === rawPath) {
    throw new Error('Native evidence and image benchmark output must use different files.')
  }
  const baselinePath = options.baselinePath ? resolve(options.baselinePath) : undefined
  const recordBaselinePath = options.recordBaselinePath
    ? resolve(options.recordBaselinePath)
    : undefined
  if ((baselinePath && baselinePath === rawPath) || (recordBaselinePath && recordBaselinePath === rawPath)) {
    throw new Error('The image baseline and current raw output must use different files.')
  }
  if (
    (baselinePath && baselinePath === nativeEvidencePath) ||
    (recordBaselinePath && recordBaselinePath === nativeEvidencePath)
  ) throw new Error('The image baseline and native evidence must use different files.')
  if ((baselinePath === undefined) === (recordBaselinePath === undefined)) {
    throw new Error('Select exactly one measured baseline input or baseline-recording output.')
  }
  const baseline = baselinePath ? readBaseline(baselinePath) : undefined
  initializeNativeEvidence(nativeEvidencePath)

  const freshRuntime = freshImageRuntimeLayout(
    config,
    options.outputDirectory,
    `${Date.now()}-${process.pid}-${randomUUID()}`,
  )
  const { runDirectory, benchmarkConfig, sourceReadinessMarker, targetReadinessMarker } =
    freshRuntime
  if (existsSync(runDirectory)) throw new Error('Fresh image benchmark run directory already exists.')
  mkdirSync(runDirectory, { recursive: true })
  if (!existsSync(sourceReadinessMarker) || !statSync(sourceReadinessMarker).isFile()) {
    throw new Error(
      `The configured state has no installed-resource marker: ${sourceReadinessMarker}.`,
    )
  }
  mkdirSync(dirname(targetReadinessMarker), { recursive: true })
  copyFileSync(sourceReadinessMarker, targetReadinessMarker)
  if (existsSync(freshRuntime.resultCacheDirectory)) {
    throw new Error('Fresh image benchmark state unexpectedly contains a result cache.')
  }
  const readinessMarkerSha256 = createHash('sha256')
    .update(readFileSync(targetReadinessMarker))
    .digest('hex')
  const benchmarkConfigPath = resolve(runDirectory, 'browser-config.json')
  writeJsonSync(benchmarkConfigPath, benchmarkConfig)

  const nvidiaSmiPath = options.nvidiaSmiPath ?? config.nvidiaSmiPath ?? 'nvidia-smi.exe'
  if (typeof nvidiaSmiPath !== 'string' || nvidiaSmiPath.trim().length === 0) {
    throw new Error('HSKIFY_NVIDIA_SMI_PATH or nvidiaSmiPath must be a non-empty string.')
  }
  const gpuIndex = nonNegativeInteger(options.gpuIndex ?? config.gpuIndex ?? 0, 'GPU index')

  const sampler = await startNvidiaMemorySampler({ executable: nvidiaSmiPath, gpuIndex })
  let summary
  let browserError
  let gpuSamples
  let gpuError
  try {
    summary = await runBrowserRegression({
      manifestPath: options.manifestPath,
      corpusRoot: options.corpusRoot,
      selection: options.selection,
      caseId: options.caseId,
      configPath: benchmarkConfigPath,
      outputDirectory: realReaderOutput,
      timeoutMs: options.timeoutMs,
      headed: options.headed,
      requireInstalledResources: true,
    })
  } catch (error) {
    browserError = error
  } finally {
    try {
      gpuSamples = await sampler.stop()
    } catch (error) {
      gpuError = error
    }
  }
  if (browserError) throw browserError
  if (gpuError) throw gpuError
  if (!summary || summary.status !== 'passed') {
    throw new Error(
      `The packaged-Firefox real-reader workload did not pass: ${summary?.message ?? summary?.status ?? 'unknown failure'}.`,
    )
  }

  const ids = imageJobIds(summary)
  for (const jobId of ids) {
    await waitForNativeSample(nativeEvidencePath, {
      kind: 'image',
      jobId,
      timeoutMs: options.timeoutMs,
      ready: (sample) =>
        Number.isSafeInteger(sample.languageUnitCount) && sample.languageUnitCount > 0 &&
        Number.isSafeInteger(sample.languageGenerationDurationMs) &&
        sample.languageGenerationDurationMs > 0,
    })
  }
  const nativeEvidence = readNativeEvidence(nativeEvidencePath)
  const workload = imageWorkload(summary)
  const measurement = imageLanguageMeasurement(summary, nativeEvidence)
  const gpu = gpuMeasurement(gpuSamples)
  const rawEvidence = {
    schemaVersion: 1,
    benchmark: 'hskify-image-runtime',
    buildFingerprint: BUILD_FINGERPRINT,
    offline: true,
    realReaderSummary: resolve(realReaderOutput, 'summary.json'),
    nativeEvidenceFile: nativeEvidencePath,
    freshRuntime: {
      runDirectory,
      profileDirectory: benchmarkConfig.profileDirectory,
      stateDirectory: benchmarkConfig.stateDirectory,
      installedResourcesReused: true,
      resultCacheStartedEmpty: true,
      readinessMarkerSha256,
    },
    workloadSha256: workloadSha256(workload),
    workload,
    language: measurement,
    gpu,
  }

  if (recordBaselinePath) {
    const artifact = baselineArtifact(workload, measurement, gpu)
    mkdirSync(dirname(recordBaselinePath), { recursive: true })
    writeJsonSync(recordBaselinePath, artifact)
    rawEvidence.status = 'baseline-recorded'
    rawEvidence.baselineOutput = recordBaselinePath
    writeJsonSync(rawPath, rawEvidence)
    writeJsonSync(evaluationPath, {
      status: 'baseline-recorded',
      baselinePath: recordBaselinePath,
      workloadSha256: artifact.workloadSha256,
    })
    return { rawEvidence, evaluation: { status: 'baseline-recorded' }, rawPath, evaluationPath }
  }

  const imageRuntime = imageRuntimeEvidence({ summary, workload, measurement, gpu, baseline })
  const evaluation = evaluateImageRuntimeBenchmark(imageRuntime)
  rawEvidence.imageRuntime = imageRuntime
  rawEvidence.baselineInput = baselinePath
  rawEvidence.status = evaluation.status
  writeJsonSync(rawPath, rawEvidence)
  writeJsonSync(evaluationPath, evaluation)
  return { rawEvidence, evaluation, rawPath, evaluationPath }
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : undefined
if (invokedPath === fileURLToPath(import.meta.url)) {
  try {
    const result = await runImageRuntimeBenchmark(
      parseImageRuntimeArguments(process.argv.slice(2)),
    )
    process.stdout.write(`${JSON.stringify({
      status: result.evaluation.status,
      rawPath: result.rawPath,
      evaluationPath: result.evaluationPath,
    }, null, 2)}\n`)
    if (!['passed', 'baseline-recorded'].includes(result.evaluation.status)) process.exitCode = 1
  } catch (error) {
    process.stderr.write(`${JSON.stringify({
      status: 'failed',
      stage: 'image-runtime-benchmark',
      message: error instanceof Error ? error.message : String(error),
    }, null, 2)}\n`)
    process.exitCode = 1
  }
}
