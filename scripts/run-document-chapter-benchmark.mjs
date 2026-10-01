/*
 * Local packaged-Firefox benchmark for the representative light-novel chapter.
 * The browser performs real extraction, reader mounting, focus observation,
 * and scrolling. Native scheduler/runtime counters are joined by exact jobId
 * from the deterministic evidence file inherited by the native host.
 */

import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  beginContentStart,
  documentDomEvidence,
  extensionMessage,
  installDomObserver,
  jobMonitorSnapshot,
  launchPackagedFirefox,
  prepareContentRuntime,
  startJobMonitor,
  stopJobMonitor,
  writeJsonSync,
} from './benchmark/browser-harness.mjs'
import {
  DOCUMENT_BENCHMARK_BLOCKS,
  DOCUMENT_BENCHMARK_CHARACTERS,
  evaluateDocumentChapterBenchmark,
} from './benchmark/document-chapter-gates.mjs'
import {
  initializeNativeEvidence,
  readNativeEvidence,
  resolveNativeEvidencePath,
  validateNativeEvidence,
  waitForNativeSample,
} from './benchmark/native-evidence.mjs'
import {
  requiredBrowserConfig,
  waitForPackagedSetup,
} from './run-real-reader-browser-regression.mjs'

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const DEFAULT_OUTPUT = resolve(
  REPOSITORY_ROOT,
  '.cache/document-chapter-benchmark/raw-samples.json',
)
const BUILD_FINGERPRINT = 'hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10'
const TITLE = 'The Lantern Road: A Representative Chapter'
const RUNTIME_COUNTER_NAMES = [
  'vision',
  'ocr',
  'projector',
  'segmentation',
  'inpainting',
  'patch',
  'font',
]

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds))
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function englishText(index, targetLength) {
  const words = [
    'traveler', 'lantern', 'quiet', 'village', 'remembered', 'promise', 'morning',
    'companion', 'answered', 'carefully', 'river', 'mountain', 'letter', 'journey',
    'wondered', 'friend', 'story', 'footsteps', 'window', 'garden', 'returned',
  ]
  let value = `Block ${String(index + 1).padStart(3, '0')} follows the chapter in canonical order.`
  let cursor = index
  while (value.length < targetLength) {
    const remaining = targetLength - value.length
    const word = words[cursor % words.length]
    cursor += 1
    if (remaining > word.length) {
      value += ` ${word}`
      continue
    }
    value += remaining === 1 ? 'x' : ` ${'x'.repeat(remaining - 1)}`
  }
  if (value.length !== targetLength) {
    throw new Error(`Could not generate exact English block length ${targetLength}.`)
  }
  return value
}

export function representativeChapterBlocks() {
  const separatorCharacters = (DOCUMENT_BENCHMARK_BLOCKS - 1) * 2
  const textCharacters = DOCUMENT_BENCHMARK_CHARACTERS - separatorCharacters
  const paragraphCharacters = textCharacters - TITLE.length
  const paragraphCount = DOCUMENT_BENCHMARK_BLOCKS - 1
  const baseLength = Math.floor(paragraphCharacters / paragraphCount)
  const longerCount = paragraphCharacters % paragraphCount
  const blocks = [TITLE]
  for (let index = 0; index < paragraphCount; index += 1) {
    blocks.push(englishText(index, baseLength + (index < longerCount ? 1 : 0)))
  }
  const chapterLength = [...blocks.join('\n\n')].length
  if (blocks.length !== DOCUMENT_BENCHMARK_BLOCKS || chapterLength !== DOCUMENT_BENCHMARK_CHARACTERS) {
    throw new Error(
      `Representative chapter generation failed: ${blocks.length} blocks/${chapterLength} characters.`,
    )
  }
  return blocks
}

export function representativeChapterMarkup() {
  const [title, ...paragraphs] = representativeChapterBlocks()
  const content = [
    `<h1>${escapeHtml(title)}</h1>`,
    ...paragraphs.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`),
  ].join('\n')
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <style>
    html { color-scheme: light; background: #f4efe5; }
    body { margin: 0; color: #24211d; font: 18px/1.65 Georgia, serif; }
    #benchmark-shell { width: min(760px, calc(100% - 48px)); margin: 48px auto; }
    h1 { font-size: 2rem; line-height: 1.2; }
    p { margin: 1.1em 0; }
  </style>
</head>
<body>
  <main id="benchmark-shell">
    <article id="chapter" data-benchmark="hskify-document-300x100000">${content}</article>
  </main>
  <script>globalThis.__hskifyBenchmarkReady = true</script>
</body>
</html>`
}

export function createDocumentBenchmarkServer() {
  const markup = Buffer.from(representativeChapterMarkup(), 'utf8')
  const server = createServer((request, response) => {
    const url = new URL(request.url ?? '/', 'http://127.0.0.1')
    if (url.pathname === '/chapter') {
      response.writeHead(200, {
        'content-type': 'text/html; charset=utf-8',
        'content-length': markup.length,
        'cache-control': 'no-store',
      })
      response.end(markup)
      return
    }
    if (url.pathname === '/health') {
      response.writeHead(204, { 'cache-control': 'no-store' })
      response.end()
      return
    }
    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' })
    response.end('Not found')
  })
  return new Promise((resolvePromise, rejectPromise) => {
    server.once('error', rejectPromise)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        rejectPromise(new Error('Document benchmark server did not bind.'))
        return
      }
      resolvePromise({ server, port: address.port })
    })
  })
}

function positiveInteger(value, name, maximum = Number.MAX_SAFE_INTEGER) {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > maximum) {
    throw new Error(`${name} must be an integer between 1 and ${maximum}.`)
  }
  return parsed
}

export function parseDocumentBenchmarkArguments(argv) {
  const options = {
    configPath: process.env.HSKIFY_REAL_READER_BROWSER_CONFIG
      ? resolve(process.env.HSKIFY_REAL_READER_BROWSER_CONFIG)
      : undefined,
    nativeEvidencePath: process.env.HSKIFY_BENCH_EVIDENCE_PATH
      ? resolve(process.env.HSKIFY_BENCH_EVIDENCE_PATH)
      : undefined,
    outputPath: DEFAULT_OUTPUT,
    sampleCount: 30,
    learningMode: 'natural',
    hskLevel: 3,
    startPosition: 'top',
    timeoutMs: 60_000,
    scrollSteps: 12,
    scrollDwellMs: 120,
    headed: false,
  }
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    const next = () => {
      const value = argv[++index]
      if (!value) throw new Error(`${argument} requires a value.`)
      return value
    }
    switch (argument) {
      case '--config': options.configPath = resolve(next()); break
      case '--output': options.outputPath = resolve(next()); break
      case '--samples': options.sampleCount = positiveInteger(next(), '--samples', 100); break
      case '--timeout-ms': options.timeoutMs = positiveInteger(next(), '--timeout-ms', 10 * 60_000); break
      case '--scroll-steps': options.scrollSteps = positiveInteger(next(), '--scroll-steps', 100); break
      case '--scroll-dwell-ms': options.scrollDwellMs = positiveInteger(next(), '--scroll-dwell-ms', 5_000); break
      case '--mode': options.learningMode = next(); if (!['natural', 'strict'].includes(options.learningMode)) throw new Error('--mode must be natural or strict.'); break
      case '--hsk-level': options.hskLevel = positiveInteger(next(), '--hsk-level', 6); break
      case '--start-position': options.startPosition = next(); if (!['top', 'middle'].includes(options.startPosition)) throw new Error('--start-position must be top or middle.'); break
      case '--headed': options.headed = true; break
      default: throw new Error(`Unknown document benchmark argument: ${argument}.`)
    }
  }
  return options
}

async function waitForDetectedDocument(extensionPage, chapterPage, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  let state
  while (Date.now() < deadline) {
    await chapterPage.bringToFront()
    state = await extensionMessage(extensionPage, { type: 'popup:state' })
    if (state.contentKind === 'document') return state
    await delay(50)
  }
  throw new Error(`Timed out waiting for document detection (last state: ${JSON.stringify(state)}).`)
}

async function waitForDocumentJob(extensionPage, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  let snapshot
  while (Date.now() < deadline) {
    snapshot = await jobMonitorSnapshot(extensionPage)
    const documents = snapshot.observations.filter((entry) => entry.sourceKind === 'document')
    if (documents.length === 1) return { snapshot, job: documents[0] }
    if (documents.length > 1) throw new Error('One document benchmark sample created multiple jobs.')
    await delay(50)
  }
  throw new Error(`Timed out waiting for one document job: ${JSON.stringify(snapshot)}.`)
}

async function exerciseDocumentScroll(page, steps, dwellMs) {
  return page.evaluate(async ({ stepCount, waitMs }) => {
    performance.clearMarks('hskify-document-scroll-start')
    performance.clearMarks('hskify-document-scroll-end')
    performance.clearMeasures('hskify-document-scroll')
    const scrolling = document.scrollingElement ?? document.documentElement
    const maximum = Math.max(0, scrolling.scrollHeight - innerHeight)
    performance.mark('hskify-document-scroll-start')
    for (let index = 0; index <= stepCount; index += 1) {
      scrollTo(0, Math.round(maximum * index / stepCount))
      await new Promise((resolvePromise) => setTimeout(resolvePromise, waitMs))
    }
    performance.mark('hskify-document-scroll-end')
    performance.measure(
      'hskify-document-scroll',
      'hskify-document-scroll-start',
      'hskify-document-scroll-end',
    )
    scrollTo(0, 0)
    await new Promise((resolvePromise) => setTimeout(resolvePromise, waitMs))
    return { steps: stepCount, dwellMs: waitMs, maximumScrollTop: maximum }
  }, { stepCount: steps, waitMs: dwellMs })
}

function exactCounterObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const result = {}
  for (const name of RUNTIME_COUNTER_NAMES) {
    const count = value[name]
    if (!Number.isInteger(count) || count < 0) return undefined
    result[name] = count
  }
  return result
}

export function mergeNativeDocumentEvidence(browserSamples, nativeEvidence) {
  if (!nativeEvidence) return browserSamples.map((sample) => ({ ...sample, nativeEvidenceMatched: false }))
  validateNativeEvidence(nativeEvidence)
  const byJobId = new Map()
  for (const sample of nativeEvidence.samples) {
    if (sample.kind === 'document') byJobId.set(sample.jobId, sample)
  }
  return browserSamples.map((browserSample) => {
    const matched = byJobId.get(browserSample.jobId)
    if (!matched) return { ...browserSample, nativeEvidenceMatched: false }
    return {
      ...browserSample,
      nativeEvidenceMatched: true,
      tokenizerIdentity: matched.tokenizerIdentity,
      dispatches: Array.isArray(matched.dispatches) ? matched.dispatches : undefined,
      runtimeInitializations: exactCounterObject(matched.runtimeInitializations),
      runtimeInvocations: exactCounterObject(matched.runtimeInvocations),
    }
  })
}

export async function waitForNativeDocumentEvidence(path, jobId, timeoutMs) {
  return waitForNativeSample(path, {
    kind: 'document',
    jobId,
    timeoutMs,
    ready: (sample) => Array.isArray(sample.dispatches) && sample.dispatches.length >= 2,
  })
}

async function runBrowserSample({ launched, port, options, sampleIndex, nativeEvidencePath }) {
  const chapterPage = await launched.context.newPage()
  const pageUrl = `http://127.0.0.1:${port}/chapter?sample=${sampleIndex}`
  const runId = `document-chapter-benchmark-${sampleIndex}`
  let monitor
  let documentEvidence
  let job
  let scroll
  let error
  let monitorInstalled = false
  let startPerformanceMs
  try {
    await chapterPage.goto(pageUrl, { waitUntil: 'domcontentloaded' })
    await chapterPage.waitForFunction(
      () => globalThis.__hskifyBenchmarkReady === true,
      undefined,
      { timeout: options.timeoutMs },
    )
    await installDomObserver(chapterPage, runId)
    await chapterPage.bringToFront()
    await prepareContentRuntime(launched.extensionPage, pageUrl)
    await extensionMessage(launched.extensionPage, { type: 'popup:prepare' })
    await waitForDetectedDocument(launched.extensionPage, chapterPage, options.timeoutMs)
    await startJobMonitor(launched.extensionPage, pageUrl, runId)
    monitorInstalled = true
    startPerformanceMs = await chapterPage.evaluate(position => {
      if (position === 'middle') window.scrollTo(0, Math.max(0, (document.scrollingElement.scrollHeight - innerHeight) / 2))
      return performance.now()
    }, options.startPosition)
    await beginContentStart(launched.extensionPage, options.hskLevel, pageUrl, 'ltr', options.learningMode)
    await chapterPage.waitForFunction(
      (expectedBlocks) => {
        const host = document.querySelector('[data-hskify-document-reader="true"]')
        return host && document.querySelectorAll('[data-hskify-item-id][data-hskify-state]').length === expectedBlocks
      },
      DOCUMENT_BENCHMARK_BLOCKS,
      { timeout: options.timeoutMs },
    )
    const observed = await waitForDocumentJob(launched.extensionPage, options.timeoutMs)
    job = observed.job
    await chapterPage.waitForFunction(() => globalThis.__hskifyRuntimeEvidence?.events.some(event =>
      event.type === 'documentBlockDomCommitted' && event.state === 'translated' && event.visible && event.readable), undefined, {timeout: options.timeoutMs})
    scroll = await exerciseDocumentScroll(
      chapterPage,
      options.scrollSteps,
      options.scrollDwellMs,
    )
    await delay(100)
    documentEvidence = await documentDomEvidence(chapterPage)
    await waitForNativeDocumentEvidence(nativeEvidencePath, job.jobId, options.timeoutMs)
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught)
    documentEvidence = await documentDomEvidence(chapterPage).catch(() => undefined)
  } finally {
    await chapterPage.bringToFront().catch(() => undefined)
    await extensionMessage(launched.extensionPage, { type: 'popup:cancel' }).catch(() => undefined)
    if (monitorInstalled) {
      monitor = await stopJobMonitor(launched.extensionPage).catch((caught) => ({
        observations: [],
        errors: [caught instanceof Error ? caught.message : String(caught)],
      }))
    }
    await chapterPage.close().catch(() => undefined)
  }
  const documentJobs = monitor?.observations?.filter((entry) => entry.sourceKind === 'document') ?? []
  job ??= documentJobs.length === 1 ? documentJobs[0] : undefined
  return {
    sampleIndex,
    modality: 'document', mode: options.learningMode, hskLevel: options.hskLevel, startPosition: options.startPosition,
    startPerformanceMs,
    firstReadableResultMs: documentEvidence?.events.find(event => event.performanceMs >= startPerformanceMs && event.type === 'documentBlockDomCommitted' && event.state === 'translated' && event.visible && event.readable)?.performanceMs - startPerformanceMs,
    pageUrl,
    jobId: job?.jobId,
    sourceSha256: documentEvidence?.sourceSha256,
    sourceBlockCount: documentEvidence?.sourceBlockCount,
    sourceCharacterCount: documentEvidence?.sourceCharacterCount,
    mounted: documentEvidence?.mounted === true,
    blockCount: documentEvidence?.blockCount,
    measures: documentEvidence?.measures ?? {},
    longTaskSupported: documentEvidence?.longTaskSupported === true,
    longTaskObservationMode: documentEvidence?.longTaskObservationMode,
    longTasks: documentEvidence?.longTasks ?? [],
    scrollLongTasks: documentEvidence?.scrollLongTasks ?? [],
    scroll: documentEvidence?.scroll
      ? { ...documentEvidence.scroll, ...scroll }
      : { completed: false, ...scroll },
    documentDom: documentEvidence,
    monitor,
    ...(error ? { error } : {}),
  }
}

function evaluationFor(rawEvidence) {
  const document = evaluateDocumentChapterBenchmark(rawEvidence.samples)
  return {
    status: document.status,
    document,
  }
}

export async function runDocumentChapterBenchmark(options) {
  if (!options.configPath || !existsSync(options.configPath)) {
    throw new Error(
      'A packaged-Firefox config is required via --config or HSKIFY_REAL_READER_BROWSER_CONFIG.',
    )
  }
  const config = JSON.parse(readFileSync(options.configPath, 'utf8'))
  const configError = requiredBrowserConfig(config)
  if (configError) throw new Error(configError)
  process.env.HSKIFY_STATE_DIR = resolve(config.stateDirectory)
  mkdirSync(dirname(options.outputPath), { recursive: true })
  const nativeEvidencePath = resolveNativeEvidencePath({
    configuredPath: options.nativeEvidencePath ?? config.nativeEvidencePath,
    outputDirectory: dirname(options.outputPath),
  })
  if (nativeEvidencePath === resolve(options.outputPath)) {
    throw new Error('Native evidence and raw browser output must use different files.')
  }
  initializeNativeEvidence(nativeEvidencePath)

  const reader = await createDocumentBenchmarkServer()
  let launched
  const browserSamples = []
  try {
    launched = await launchPackagedFirefox({ ...config, headed: options.headed })
    await waitForPackagedSetup(launched.extensionPage, Math.min(options.timeoutMs, 5 * 60_000))
    for (let sampleIndex = 1; sampleIndex <= options.sampleCount; sampleIndex += 1) {
      browserSamples.push(
        await runBrowserSample({
          launched,
          port: reader.port,
          options,
          sampleIndex,
          nativeEvidencePath,
        }),
      )
      writeJsonSync(options.outputPath, {
        schemaVersion: 1,
        benchmark: 'hskify-document-chapter',
        buildFingerprint: BUILD_FINGERPRINT,
        status: 'collecting',
        workload: {
          blocks: DOCUMENT_BENCHMARK_BLOCKS,
          normalizedCharacters: DOCUMENT_BENCHMARK_CHARACTERS,
        },
        samples: browserSamples,
      })
    }
  } finally {
    launched?.extensionPage?.close()
    await launched?.context?.close().catch(() => undefined)
    await new Promise((resolvePromise) => reader.server.close(resolvePromise))
  }

  const nativeEvidence = readNativeEvidence(nativeEvidencePath)
  const samples = mergeNativeDocumentEvidence(browserSamples, nativeEvidence)
  const rawEvidence = {
    schemaVersion: 1,
    benchmark: 'hskify-document-chapter',
    buildFingerprint: BUILD_FINGERPRINT,
    offline: true,
    workload: {
      blocks: DOCUMENT_BENCHMARK_BLOCKS,
      normalizedCharacters: DOCUMENT_BENCHMARK_CHARACTERS,
    },
    browser: {
      extensionId: launched?.identity?.id,
      extensionVersion: launched?.identity?.manifest?.version,
    },
    nativeEvidenceFile: nativeEvidencePath,
    samples,
  }
  const evaluation = evaluationFor(rawEvidence)
  rawEvidence.status = evaluation.status
  writeJsonSync(options.outputPath, rawEvidence)
  const evaluationPath = resolve(dirname(options.outputPath), 'evaluation.json')
  writeJsonSync(evaluationPath, evaluation)
  return { rawEvidence, evaluation, outputPath: options.outputPath, evaluationPath }
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : undefined
if (invokedPath === fileURLToPath(import.meta.url)) {
  try {
    const result = await runDocumentChapterBenchmark(
      parseDocumentBenchmarkArguments(process.argv.slice(2)),
    )
    process.stdout.write(`${JSON.stringify({
      status: result.evaluation.status,
      outputPath: result.outputPath,
      evaluationPath: result.evaluationPath,
      sampleCount: result.rawEvidence.samples.length,
    }, null, 2)}\n`)
    if (result.evaluation.status !== 'passed') process.exitCode = 1
  } catch (error) {
    process.stderr.write(`${JSON.stringify({
      status: 'failed',
      stage: 'document-chapter-benchmark',
      message: error instanceof Error ? error.message : String(error),
    }, null, 2)}\n`)
    process.exitCode = 1
  }
}
