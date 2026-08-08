/*
 * Observational packaged-Firefox probe for newly captured, unannotated chapters.
 *
 * This intentionally does not share the release status of real-reader-v2:
 * captured pixels are integrity checked, browser behavior is measured, and
 * model output is preserved for review, but quality remains unreviewed until
 * exhaustive human annotations are added to the release corpus.
 */

import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  chapterDomEvidence,
  installDomObserver,
  launchPackagedFirefox,
  prepareContentRuntime,
  routeEvidence,
  startJobMonitor,
  stopJobMonitor,
  timedContentStart,
  waitForPageState,
  writeJsonSync,
} from './benchmark/browser-harness.mjs'
import {
  DEFAULT_CAPTURE_RESULT_PATH,
  DEFAULT_CORPUS_ROOT,
  auditCapture,
  captureReaderChapters,
} from './real-reader-corpus.mjs'
import {
  committedResourceIdentities,
  createReaderServer,
  requiredBrowserConfig,
  waitForPackagedSetup,
} from './run-real-reader-browser-regression.mjs'

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const DEFAULT_OUTPUT = resolve(REPOSITORY_ROOT, '.cache/capture-browser-probe')

function parseArguments(argv) {
  const options = {
    capturePath: DEFAULT_CAPTURE_RESULT_PATH,
    corpusRoot: DEFAULT_CORPUS_ROOT,
    configPath: process.env.HSKIFY_REAL_READER_BROWSER_CONFIG
      ? resolve(process.env.HSKIFY_REAL_READER_BROWSER_CONFIG)
      : undefined,
    outputDirectory: DEFAULT_OUTPUT,
    timeoutMs: 5 * 60_000,
    dwellMs: 6_000,
    warmupWaitMs: 20_000,
    hskLevel: 3,
    headed: false,
  }
  const args = [...argv]
  while (args.length > 0) {
    const argument = args.shift()
    if (argument === '--capture') options.capturePath = resolve(args.shift() ?? '')
    else if (argument === '--corpus') options.corpusRoot = resolve(args.shift() ?? '')
    else if (argument === '--config' || argument === '--browser-config')
      options.configPath = resolve(args.shift() ?? '')
    else if (argument === '--output') options.outputDirectory = resolve(args.shift() ?? '')
    else if (argument === '--timeout-minutes') options.timeoutMs = Number(args.shift()) * 60_000
    else if (argument === '--dwell-ms') options.dwellMs = Number(args.shift())
    else if (argument === '--warmup-wait-ms') options.warmupWaitMs = Number(args.shift())
    else if (argument === '--hsk-level') options.hskLevel = Number(args.shift())
    else if (argument === '--headed') options.headed = true
    else throw new Error(`Unknown argument: ${argument}`)
  }
  for (const [name, value] of [
    ['--timeout-minutes', options.timeoutMs],
    ['--dwell-ms', options.dwellMs],
    ['--warmup-wait-ms', options.warmupWaitMs],
  ]) {
    if (!Number.isFinite(value) || value < 0) throw new Error(`${name} must be non-negative.`)
  }
  if (!Number.isSafeInteger(options.hskLevel) || options.hskLevel < 1 || options.hskLevel > 6)
    throw new Error('--hsk-level must be an integer from 1 through 6.')
  return options
}

function percentile(values, fraction) {
  if (values.length === 0) return undefined
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[Math.ceil(sorted.length * fraction) - 1]
}

async function startReaderSimulation(page, pageCount, dwellMs) {
  await page.evaluate(
    ({ count, dwell }) => {
      if (globalThis.__hskifyReaderSimulation?.timer)
        clearInterval(globalThis.__hskifyReaderSimulation.timer)
      const state = { startedAtEpochMs: Date.now(), dwellMs: dwell, entries: [], timer: 0 }
      const enter = (pageNumber) => {
        const image = document.querySelector(`[data-page="${pageNumber}"]`)
        const surface = image?.closest('.hmt-wrapper') ?? image
        if (surface) scrollTo({ top: surface.getBoundingClientRect().top + scrollY, behavior: 'instant' })
        state.entries.push({ page: pageNumber, epochMs: Date.now() })
      }
      enter(1)
      let nextPage = 2
      if (count > 1 && dwell > 0) {
        state.timer = setInterval(() => {
          enter(nextPage)
          nextPage += 1
          if (nextPage > count) {
            clearInterval(state.timer)
            state.timer = 0
          }
        }, dwell)
      }
      globalThis.__hskifyReaderSimulation = state
    },
    { count: pageCount, dwell: dwellMs },
  )
}

async function stopReaderSimulation(page) {
  return page.evaluate(() => {
    const state = globalThis.__hskifyReaderSimulation
    if (!state) return { startedAtEpochMs: Date.now(), dwellMs: 0, entries: [] }
    if (state.timer) clearInterval(state.timer)
    const result = {
      startedAtEpochMs: state.startedAtEpochMs,
      dwellMs: state.dwellMs,
      entries: [...state.entries],
    }
    delete globalThis.__hskifyReaderSimulation
    return result
  })
}

function readingMetrics(dom, simulation, pageCount) {
  const finalEvents = dom.events.filter(
    (event) => event.type === 'selectableTextDomCommitted' && event.sourcePreserving !== true,
  )
  const byPage = new Map()
  for (const event of finalEvents) {
    const events = byPage.get(event.page) ?? []
    events.push(event)
    byPage.set(event.page, events)
  }
  const actualEntry = new Map(simulation.entries.map((entry) => [entry.page, entry.epochMs]))
  const pages = []
  for (let page = 1; page <= pageCount; page += 1) {
    const events = byPage.get(page) ?? []
    if (events.length === 0) continue
    const entryAtEpochMs =
      actualEntry.get(page) ??
      simulation.startedAtEpochMs + (page - 1) * simulation.dwellMs
    const finalAtEpochMs = Math.max(...events.map((event) => event.epochMs))
    pages.push({
      page,
      regionCount: events.length,
      entryAtEpochMs,
      finalAtEpochMs,
      readyBeforeViewport: finalAtEpochMs <= entryAtEpochMs,
      stallMs: Math.max(0, finalAtEpochMs - entryAtEpochMs),
    })
  }
  const stalls = pages.map((page) => page.stallMs)
  const readyCount = pages.filter((page) => page.readyBeforeViewport).length
  return {
    translatedPageCount: pages.length,
    readyBeforeViewportCount: readyCount,
    readyBeforeViewportRatio: pages.length > 0 ? readyCount / pages.length : undefined,
    p95StallMs: percentile(stalls, 0.95),
    maximumStallMs: stalls.length > 0 ? Math.max(...stalls) : undefined,
    pagesOverOneSecond: pages.filter((page) => page.stallMs > 1_000).map((page) => page.page),
    pages,
  }
}

export async function runCaptureBrowserProbe(options) {
  const integrity = auditCapture({
    capturePath: options.capturePath,
    corpusRoot: options.corpusRoot,
  })
  if (integrity.status !== 'passed') {
    return {
      schemaVersion: 1,
      status: 'failed',
      evidenceKind: 'unannotated-capture-probe',
      qualityStatus: 'unreviewed',
      stage: 'capture-integrity',
      integrity,
    }
  }
  const config = options.configPath
    ? JSON.parse(readFileSync(options.configPath, 'utf8'))
    : undefined
  const configError = requiredBrowserConfig(config)
  if (configError) {
    return {
      schemaVersion: 1,
      status: 'failed',
      evidenceKind: 'unannotated-capture-probe',
      qualityStatus: 'unreviewed',
      stage: 'packaged-firefox-prerequisites',
      message: configError,
      integrity,
    }
  }
  const expectedResourceIdentities = committedResourceIdentities()
  mkdirSync(options.outputDirectory, { recursive: true })
  process.env.HSK_MANGA_STATE_DIR = resolve(config.stateDirectory)
  const capture = JSON.parse(readFileSync(options.capturePath, 'utf8'))
  const chapters = captureReaderChapters(capture)
  const reader = await createReaderServer({
    manifest: { schemaVersion: 1, evidenceKind: 'captured-images', chapters },
    corpusRoot: options.corpusRoot,
    chapters,
  })
  const runs = []
  const failures = []
  let launched
  try {
    launched = await launchPackagedFirefox({ ...config, headed: options.headed })
    await waitForPackagedSetup(launched.extensionPage, Math.min(options.timeoutMs, 5 * 60_000))
    if (options.warmupWaitMs > 0)
      await new Promise((resolvePromise) => setTimeout(resolvePromise, options.warmupWaitMs))
    for (const chapter of chapters) {
      const chapterPage = await launched.context.newPage()
      const pageUrl = `http://127.0.0.1:${reader.port}/chapter/${encodeURIComponent(chapter.id)}`
      const startedAt = Date.now()
      try {
        await chapterPage.goto(pageUrl, { waitUntil: 'domcontentloaded' })
        await chapterPage.waitForFunction(
          () => globalThis.__hskifyReaderReady instanceof Promise,
          undefined,
          { timeout: 30_000 },
        )
        await chapterPage.evaluate(() => globalThis.__hskifyReaderReady)
        await installDomObserver(chapterPage, `capture-probe-${chapter.id}`)
        await prepareContentRuntime(launched.extensionPage, pageUrl)
        await startJobMonitor(launched.extensionPage, pageUrl, `capture-probe-${chapter.id}`)
        const action = await timedContentStart(
          launched.extensionPage,
          options.hskLevel,
          pageUrl,
          chapter.reader.direction,
        )
        await startReaderSimulation(chapterPage, chapter.pageCount, options.dwellMs)
        const state = await waitForPageState(
          launched.extensionPage,
          chapterPage,
          ['complete', 'failed', 'cancelled'],
          options.timeoutMs,
        )
        const simulation = await stopReaderSimulation(chapterPage)
        const monitor = await stopJobMonitor(launched.extensionPage)
        const dom = await chapterDomEvidence(chapterPage)
        const route = await routeEvidence(
          launched.extensionPage,
          monitor.observations,
          true,
          expectedResourceIdentities,
        )
        const expectedPages = chapter.pageCount
        const records = monitor.observations
        const orderedCompleteJobs =
          records.length === expectedPages &&
          records.every((record, index) => record.pageIndex === index) &&
          route.jobs.length === expectedPages &&
          route.jobs.every(
            (job, index) =>
              job.jobId === records[index]?.jobId &&
              job.pageIndex === index &&
              job.terminal?.type === 'complete',
          )
        const textEvents = dom.events.filter(
          (event) => event.type === 'selectableTextDomCommitted',
        )
        const uniqueRegionIds = new Set(textEvents.map((event) => event.regionId))
        const firstTranslatedVisible = textEvents.find(
          (event) => event.visible && event.sourcePreserving !== true,
        )
        const reading = readingMetrics(dom, simulation, expectedPages)
        const assertions = [
          {
            id: `${chapter.id}.terminal`,
            passed: state.state === 'complete',
            expected: 'complete',
            actual: state.state,
          },
          {
            id: `${chapter.id}.ordered-complete-pages`,
            passed: orderedCompleteJobs,
            expected: `pageIndex 0..${expectedPages - 1}, all complete`,
            actual: route.jobs.map((job) => ({
              pageIndex: job.pageIndex,
              terminalType: job.terminal?.type,
            })),
          },
          {
            id: `${chapter.id}.wrapper-count`,
            passed: dom.wrappedImageCount === expectedPages,
            expected: expectedPages,
            actual: dom.wrappedImageCount,
          },
          {
            id: `${chapter.id}.single-final-publication`,
            passed:
              uniqueRegionIds.size === textEvents.length &&
              dom.regions.every((region) => region.repairState !== 'pending'),
            expected: 'one terminal DOM commit per region',
            actual: {
              commits: textEvents.length,
              uniqueRegionIds: uniqueRegionIds.size,
              pending: dom.regions.filter((region) => region.repairState === 'pending').length,
            },
          },
          {
            id: `${chapter.id}.patch-before-translated-text`,
            passed:
              dom.regions.every((region) => region.sourcePreserving) || dom.patchBeforeText,
            expected: 'decoded patch precedes translated selectable text',
            actual: dom.patchBeforeText,
          },
          {
            id: `${chapter.id}.readable-fit`,
            passed:
              dom.degradedFitCount === 0 &&
              dom.regions
                .filter((region) => !region.sourcePreserving)
                .every((region) => !region.overflows),
            expected: 'no degraded fit or overflow',
            actual: {
              degradedFitCount: dom.degradedFitCount,
              overflowRegionIds: dom.regions
                .filter((region) => !region.sourcePreserving && region.overflows)
                .map((region) => region.regionId),
            },
          },
          {
            id: `${chapter.id}.viewport-readiness`,
            passed:
              reading.translatedPageCount > 0 &&
              reading.readyBeforeViewportRatio >= 0.95 &&
              reading.p95StallMs <= 500 &&
              reading.pagesOverOneSecond.length === 0,
            expected: '>=95% translated pages ready before viewport; p95 stall <=500 ms; none >1 s',
            actual: reading,
          },
          {
            id: `${chapter.id}.resource-identities`,
            passed: route.resourceIdentityEvidence.gates.every((gate) => gate.status === 'pass'),
            expected: 'exact packaged resource identity set',
            actual: route.resourceIdentityEvidence.gates,
          },
        ]
        const run = {
          chapterId: chapter.id,
          hskLevel: options.hskLevel,
          pageCount: expectedPages,
          terminalState: state.state,
          jobCount: records.length,
          regionCount: dom.regionCount,
          translatedRegionCount: dom.regions.filter((region) => !region.sourcePreserving).length,
          sourcePreservingRegionCount: dom.regions.filter((region) => region.sourcePreserving).length,
          firstTranslatedVisibleMs: firstTranslatedVisible?.epochMs - action.issuedAtEpochMs,
          durationMs: Date.now() - startedAt,
          reading,
          simulation,
          dom,
          route,
          assertions,
          status: assertions.every((assertion) => assertion.passed) ? 'passed' : 'failed',
        }
        runs.push(run)
        failures.push(...assertions.filter((assertion) => !assertion.passed))
      } catch (error) {
        const failure = {
          id: `${chapter.id}.browser-run`,
          passed: false,
          expected: 'terminal packaged-Firefox capture probe',
          actual: error instanceof Error ? error.message : String(error),
        }
        failures.push(failure)
        runs.push({ chapterId: chapter.id, status: 'failed', assertions: [failure] })
      } finally {
        await chapterPage.close().catch(() => undefined)
      }
    }
  } catch (error) {
    failures.push({
      id: 'packaged-firefox.launch',
      passed: false,
      expected: 'packaged Firefox launch',
      actual: error instanceof Error ? error.message : String(error),
    })
  } finally {
    launched?.extensionPage?.close()
    await launched?.context?.close().catch(() => undefined)
    await new Promise((resolvePromise) => reader.server.close(resolvePromise))
  }
  const summary = {
    schemaVersion: 1,
    status: failures.length === 0 ? 'passed' : 'failed',
    evidenceKind: 'unannotated-capture-probe',
    qualityStatus: 'unreviewed',
    releaseReady: false,
    transport: 'packaged-firefox-local-reader',
    integrity,
    outputDirectory: options.outputDirectory,
    runs,
    failures,
  }
  writeJsonSync(resolve(options.outputDirectory, 'summary.json'), summary)
  return summary
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : undefined
if (invokedPath === fileURLToPath(import.meta.url)) {
  try {
    const options = parseArguments(process.argv.slice(2))
    if (!options.configPath || !existsSync(options.configPath))
      throw new Error('A packaged Firefox browser config is required.')
    const summary = await runCaptureBrowserProbe(options)
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`)
    if (summary.status !== 'passed') process.exitCode = 1
  } catch (error) {
    process.stderr.write(
      `${JSON.stringify({ schemaVersion: 1, status: 'error', message: error instanceof Error ? error.message : String(error) }, null, 2)}\n`,
    )
    process.exitCode = 1
  }
}
