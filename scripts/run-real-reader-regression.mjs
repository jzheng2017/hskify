import { closeSync, mkdirSync, openSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { spawn } from 'node:child_process'
import { setTimeout as delay } from 'node:timers/promises'
import { fileURLToPath } from 'node:url'

import {
  DEFAULT_CORPUS_ROOT,
  DEFAULT_MANIFEST_PATH,
  auditCorpus,
  selectedCases,
} from './real-reader-corpus.mjs'

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const DEFAULT_DAEMON = resolve(
  REPOSITORY_ROOT,
  'target/release/hsk-manga-browser-daemon.exe',
)
const DENSE_DIFFERENTIAL_CASE = 'webtoon-thirty-minutes-1-page-50'
const EXTENSION_ORIGIN = 'moz-extension://00000000-0000-4000-8000-000000000006'
const PNG_SIGNATURE = '89504e470d0a1a0a'

function check(id, passed, expected, actual, detail) {
  return { id, passed, expected, actual, ...(detail ? { detail } : {}) }
}

function polygonBounds(points) {
  if (!Array.isArray(points) || points.length < 3) return undefined
  const xs = points.map((point) => point?.x)
  const ys = points.map((point) => point?.y)
  if (![...xs, ...ys].every((value) => Number.isFinite(value))) return undefined
  return {
    x0: Math.min(...xs),
    y0: Math.min(...ys),
    x1: Math.max(...xs),
    y1: Math.max(...ys),
  }
}

function validRect(rect) {
  return (
    rect &&
    [rect.x, rect.y, rect.width, rect.height].every(Number.isFinite) &&
    rect.x >= 0 &&
    rect.y >= 0 &&
    rect.width > 0 &&
    rect.height > 0 &&
    rect.x + rect.width <= 1.000001 &&
    rect.y + rect.height <= 1.000001
  )
}

function rectsOverlap(left, right) {
  if (!validRect(left) || !validRect(right)) return false
  return (
    Math.min(left.x + left.width, right.x + right.width) > Math.max(left.x, right.x) &&
    Math.min(left.y + left.height, right.y + right.height) > Math.max(left.y, right.y)
  )
}

function rectOverlapsPolygon(rect, polygon) {
  if (!validRect(rect)) return false
  const bounds = polygonBounds(polygon)
  if (!bounds) return false
  const overlapWidth = Math.min(rect.x + rect.width, bounds.x1) - Math.max(rect.x, bounds.x0)
  const overlapHeight = Math.min(rect.y + rect.height, bounds.y1) - Math.max(rect.y, bounds.y0)
  return overlapWidth > 0 && overlapHeight > 0
}

function finalRegions(updates) {
  const regions = new Map()
  for (const update of updates) {
    if (update.type === 'regionReady' && update.region?.id) {
      regions.set(update.region.id, structuredClone(update.region))
    }
  }
  return [...regions.values()]
}

function preservedArtwork(updates) {
  const regions = new Map()
  for (const update of updates) {
    if (update.type === 'artworkPreserved' && update.region?.id) {
      regions.set(update.region.id, structuredClone(update.region))
    }
  }
  return [...regions.values()]
}

function normalizedOcrText(value) {
  return String(value ?? '').toLocaleUpperCase().replaceAll(/[^A-Z0-9]+/gu, '')
}

function orderedCharacterCoverage(expected, actual) {
  const left = normalizedOcrText(expected)
  const right = normalizedOcrText(actual)
  if (!left || !right) return 0
  if (right.includes(left)) return 1
  if (right.length > left.length * 2) return 0
  let previous = new Uint16Array(right.length + 1)
  for (let leftIndex = 0; leftIndex < left.length; leftIndex += 1) {
    const current = new Uint16Array(right.length + 1)
    for (let rightIndex = 0; rightIndex < right.length; rightIndex += 1) {
      current[rightIndex + 1] =
        left[leftIndex] === right[rightIndex]
          ? previous[rightIndex] + 1
          : Math.max(previous[rightIndex + 1], current[rightIndex])
    }
    previous = current
  }
  return previous[right.length] / left.length
}

export function assertSemanticExpectations(item, regions, preserved = []) {
  const assertions = []
  const combinedSource = regions.map((region) => region.sourceEnglish ?? '').join('\n')
  for (const fragment of item.expectations?.requiredSourceFragments ?? []) {
    assertions.push(
      check(
        `semantic.${item.id}.required-source.${fragment}`,
        combinedSource.toLocaleLowerCase().includes(fragment.toLocaleLowerCase()),
        `sourceEnglish contains ${fragment}`,
        combinedSource,
        'Required OCR fragments make missed-text and partial-translation regressions terminal failures.',
      ),
    )
  }
  for (const sourceText of item.expectations?.excludedSourceTexts ?? []) {
    const matches = regions.filter(
      (region) => region.sourceEnglish?.trim().toLocaleUpperCase() === sourceText.toUpperCase(),
    )
    assertions.push(
      check(
        `semantic.${item.id}.excluded-source.${sourceText}`,
        matches.length === 0,
        0,
        matches.length,
        'translateSoundEffects=false must keep excluded SFX out of the translation regions.',
      ),
    )
  }
  const preservedTexts = preserved.map((region) => region.sourceEnglish ?? '').filter(Boolean)
  const translatedTexts = regions.map((region) => region.sourceEnglish ?? '').filter(Boolean)
  const preservedSource = preservedTexts.join('\n')
  for (const fragment of item.expectations?.preservedArtworkSourceFragments ?? []) {
    const preservedCoverage = Math.max(
      0,
      ...preservedTexts.map((source) => orderedCharacterCoverage(fragment, source)),
    )
    const translatedCoverage = Math.max(
      0,
      ...translatedTexts.map((source) => orderedCharacterCoverage(fragment, source)),
    )
    const preservedMatches = preservedCoverage >= 0.68
    const translatedMatches = translatedCoverage >= 0.8
    assertions.push(
      check(
        `semantic.${item.id}.preserved-artwork.${fragment}`,
        preservedMatches && !translatedMatches,
        'preserved source artwork without a translated overlay',
        {
          preservedMatches,
          translatedMatches,
          preservedCoverage,
          translatedCoverage,
          preservedSource,
          combinedSource,
        },
        'Illustrated technique lettering must remain source artwork instead of receiving a cleanup patch and standard-font overlay.',
      ),
    )
  }
  for (const name of item.expectations?.preserveNamesWhenDetected ?? []) {
    const detected = regions.flatMap((region) => {
      const source = region.sourceEnglish ?? ''
      const start = source.toLocaleLowerCase().indexOf(name.toLocaleLowerCase())
      return start < 0
        ? []
        : [{ region, exactSourceSpelling: source.slice(start, start + name.length) }]
    })
    const preserved = detected.filter(({ region, exactSourceSpelling }) =>
      region.displayedChinese?.includes(exactSourceSpelling),
    )
    assertions.push({
      ...check(
        `semantic.${item.id}.preserve-name.${name}`,
        detected.length === 0 || preserved.length === detected.length,
        detected.length,
        preserved.length,
        'When OCR detects an annotated name, keep-original must preserve it in every corresponding Chinese region.',
      ),
      skipped: detected.length === 0,
    })
  }
  return assertions
}

export function assertCompletedJob(item, hskLevel, terminal, updates, patchRecords) {
  const regions = finalRegions(updates)
  const preserved = preservedArtwork(updates)
  const exactRegionCount = item.expectations?.exactRegionCount
  const expectedRegionCount =
    exactRegionCount === undefined ? `>= ${item.expectations?.minimumRegionCount ?? 1}` : exactRegionCount
  const regionCountPassed =
    exactRegionCount === undefined
      ? regions.length >= (item.expectations?.minimumRegionCount ?? 1)
      : regions.length === exactRegionCount
  const assertions = [
    check(`job.${item.id}.hsk-${hskLevel}.terminal`, terminal?.type === 'complete', 'complete', terminal?.type),
    check(
      `job.${item.id}.hsk-${hskLevel}.regions`,
      regionCountPassed,
      expectedRegionCount,
      regions.length,
    ),
  ]
  const patchById = new Map(patchRecords.map((record) => [record.blobId, record]))
  for (const region of regions) {
    const prefix = `job.${item.id}.hsk-${hskLevel}.region.${region.id}`
    assertions.push(
      check(`${prefix}.source`, Boolean(region.sourceEnglish?.trim()), 'non-empty', region.sourceEnglish),
      check(
        `${prefix}.translation`,
        Boolean(region.displayedChinese?.trim()),
        'non-empty',
        region.displayedChinese,
      ),
      check(
        `${prefix}.repair-terminal`,
        Boolean(region.hsk?.repairState) && region.hsk.repairState !== 'pending',
        'accepted or exhausted',
        region.hsk?.repairState,
      ),
      check(
        `${prefix}.requested-level`,
        region.hsk?.requestedLevel === hskLevel,
        hskLevel,
        region.hsk?.requestedLevel,
      ),
      check(`${prefix}.patch-rect`, validRect(region.patch?.rect), 'normalized non-empty rect', region.patch?.rect),
      check(
        `${prefix}.patch-source-overlap`,
        rectOverlapsPolygon(region.patch?.rect, region.textPolygon),
        true,
        rectOverlapsPolygon(region.patch?.rect, region.textPolygon),
      ),
    )
    const patch = patchById.get(region.patch?.blobId)
    assertions.push(
      check(`${prefix}.patch-present`, Boolean(patch), true, Boolean(patch)),
      check(`${prefix}.patch-mime`, region.patch?.mimeType === 'image/png', 'image/png', region.patch?.mimeType),
      check(`${prefix}.patch-png`, patch?.validPng === true, true, patch?.validPng),
    )
  }
  for (const [index, protectedRect] of (
    item.expectations?.protectedArtworkRects ?? []
  ).entries()) {
    const overlapping = regions
      .filter((region) => rectsOverlap(region.patch?.rect, protectedRect))
      .map((region) => ({
        id: region.id,
        sourceEnglish: region.sourceEnglish,
        patchRect: region.patch?.rect,
      }))
    assertions.push(
      check(
        `semantic.${item.id}.protected-artwork-rect.${index + 1}`,
        overlapping.length === 0,
        'no cleanup patch or translated overlay intersects the annotated source artwork',
        overlapping,
        'Locally annotated illustrated lettering must remain pixel-identical even when OCR can read only a fragment of its stylized glyphs.',
      ),
    )
  }
  assertions.push(...assertSemanticExpectations(item, regions, preserved))
  return { regions, preserved, assertions }
}

function regionComparisonKey(region) {
  return `${region.readingOrder ?? ''}\u0000${region.sourceEnglish?.trim() ?? ''}`
}

export function assertHskDifferential(lowRun, highRun) {
  const highByKey = new Map(highRun.regions.map((region) => [regionComparisonKey(region), region]))
  const shared = lowRun.regions
    .map((region) => [region, highByKey.get(regionComparisonKey(region))])
    .filter((pair) => pair[1])
  const changed = shared.filter(
    ([low, high]) => low.displayedChinese?.trim() !== high.displayedChinese?.trim(),
  )
  return [
    check('differential.hsk-2-vs-5.shared-regions', shared.length > 0, '> 0', shared.length),
    check(
      'differential.hsk-2-vs-5.changed-output',
      changed.length > 0,
      '> 0 shared translations changed',
      changed.length,
    ),
    check(
      'differential.hsk-2.low-level-validator',
      lowRun.regions.length > 0 &&
        lowRun.regions.every(
          (region) =>
            region.hsk?.requestedLevel === 2 &&
            region.hsk?.repairState &&
            region.hsk.repairState !== 'pending',
        ),
      'every HSK2 region used requestedLevel=2 and reached terminal repair state',
      lowRun.regions.map((region) => ({
        id: region.id,
        requestedLevel: region.hsk?.requestedLevel,
        repairState: region.hsk?.repairState,
      })),
    ),
  ]
}

function parseArguments(argv) {
  const timestamp = new Date().toISOString().replaceAll(':', '').replaceAll('.', '-')
  const options = {
    manifestPath: DEFAULT_MANIFEST_PATH,
    corpusRoot: DEFAULT_CORPUS_ROOT,
    daemonPath: DEFAULT_DAEMON,
    selection: 'quality',
    caseId: undefined,
    outputDirectory: resolve(REPOSITORY_ROOT, `runs/real-reader-${timestamp}`),
    timeoutMinutes: 20,
  }
  const args = [...argv]
  while (args.length > 0) {
    const argument = args.shift()
    if (argument === '--manifest') options.manifestPath = resolve(args.shift() ?? '')
    else if (argument === '--corpus') options.corpusRoot = resolve(args.shift() ?? '')
    else if (argument === '--daemon') options.daemonPath = resolve(args.shift() ?? '')
    else if (argument === '--selection') options.selection = args.shift() ?? ''
    else if (argument === '--case') options.caseId = args.shift() ?? ''
    else if (argument === '--output') options.outputDirectory = resolve(args.shift() ?? '')
    else if (argument === '--timeout-minutes') options.timeoutMinutes = Number(args.shift())
    else throw new Error(`Unknown argument: ${argument}`)
  }
  if (!Number.isFinite(options.timeoutMinutes) || options.timeoutMinutes <= 0) {
    throw new Error('--timeout-minutes must be positive.')
  }
  return options
}

async function waitForState(child, statePath, stderrPath) {
  const deadline = Date.now() + 30_000
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(
        `Daemon exited during startup (${child.exitCode}): ${readFileSync(stderrPath, 'utf8')}`,
      )
    }
    try {
      const state = JSON.parse(readFileSync(statePath, 'utf8'))
      if (state.pid === child.pid && state.port && state.controlSecret) return state
    } catch {
      // State publication is atomic but may not have happened yet.
    }
    await delay(50)
  }
  throw new Error('Daemon startup timed out.')
}

async function responseJson(response) {
  const text = await response.text()
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${text}`)
  return text ? JSON.parse(text) : undefined
}

async function issueSession(baseUrl, controlSecret) {
  return responseJson(
    await fetch(`${baseUrl}/browser-internal/session`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-hsk-manga-control': controlSecret,
      },
      body: JSON.stringify({ extensionOrigin: EXTENSION_ORIGIN }),
    }),
  )
}

function authorizedHeaders(token) {
  return {
    authorization: `Bearer ${token}`,
    'x-hsk-manga-extension-origin': EXTENSION_ORIGIN,
  }
}

async function ensureReady(baseUrl, token, timeoutMinutes) {
  const headers = authorizedHeaders(token)
  const health = await responseJson(await fetch(`${baseUrl}/health`, { headers }))
  if (health.setupState === 'ready') return
  await responseJson(await fetch(`${baseUrl}/setup/models`, { method: 'POST', headers }))
  const deadline = Date.now() + timeoutMinutes * 60_000
  while (Date.now() < deadline) {
    const setup = await responseJson(await fetch(`${baseUrl}/setup`, { headers }))
    if (setup.state === 'ready') return
    if (setup.state === 'failed') throw new Error(`Model setup failed: ${setup.message}`)
    await delay(250)
  }
  throw new Error('Model setup timed out.')
}

async function runJob({
  item,
  hskLevel,
  precedingContext,
  corpusRoot,
  baseUrl,
  session,
  outputDirectory,
  timeoutMinutes,
}) {
  const objectPath = resolve(corpusRoot, item.object.path)
  const bytes = readFileSync(objectPath)
  const request = {
    buildFingerprint: session.buildFingerprint,
    clientImageId: `${item.id}-hsk-${hskLevel}`,
    sourceSha256: item.object.sha256,
    sourceMimeType: item.object.mimeType,
    naturalWidth: item.object.width,
    naturalHeight: item.object.height,
    pageSessionId: `real-reader-${item.chapterId}-hsk-${hskLevel}`,
    pageIndex: item.provenance.pageIndex - 1,
    settings: {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel,
      learningMode: 'natural',
      readingDirection: 'auto',
      translateSoundEffects: false,
      nameTranslation: 'keep-original',
    },
    visibleRects: item.expectations?.initialVisibleRects ?? [
      { x: 0, y: 0, width: 1, height: 1 },
    ],
    precedingContext,
    properNameGlossary: [],
  }
  const form = new FormData()
  form.append('image', new Blob([bytes], { type: item.object.mimeType }), item.object.path)
  form.append('request', new Blob([JSON.stringify(request)], { type: 'application/json' }), 'request.json')
  const headers = authorizedHeaders(session.token)
  const created = await responseJson(
    await fetch(`${baseUrl}/jobs`, { method: 'POST', headers, body: form }),
  )
  const acceptedAt = Date.now()
  const updates = []
  const updateTimeline = []
  let sequence = 0
  let terminal
  let firstRegionReadyMs
  const deadline = Date.now() + timeoutMinutes * 60_000
  while (!terminal && Date.now() < deadline) {
    const batch = await responseJson(
      await fetch(`${baseUrl}/jobs/${encodeURIComponent(created.jobId)}/updates?after=${sequence}&waitMs=20000`, {
        headers,
      }),
    )
    for (const update of batch.updates ?? []) {
      updates.push(update)
      updateTimeline.push({
        sequence: update.sequence,
        type: update.type,
        ...(update.stage ? { stage: update.stage } : {}),
        receivedAfterMs: Date.now() - acceptedAt,
      })
      sequence = Math.max(sequence, update.sequence)
      if (update.type === 'regionReady' && firstRegionReadyMs === undefined) {
        firstRegionReadyMs = Date.now() - acceptedAt
      }
      if (['complete', 'failed', 'cancelled'].includes(update.type)) terminal = update
    }
  }
  if (!terminal) throw new Error(`Job ${created.jobId} timed out.`)

  const regions = finalRegions(updates)
  const patchRecords = []
  for (const blobId of new Set(regions.map((region) => region.patch?.blobId).filter(Boolean))) {
    const response = await fetch(`${baseUrl}/blobs/${encodeURIComponent(blobId)}`, { headers })
    const patchBytes = Buffer.from(await response.arrayBuffer())
    patchRecords.push({
      blobId,
      status: response.status,
      contentType: response.headers.get('content-type'),
      bytes: patchBytes.length,
      validPng:
        response.ok &&
        response.headers.get('content-type')?.startsWith('image/png') &&
        patchBytes.subarray(0, 8).toString('hex') === PNG_SIGNATURE,
    })
  }
  const evaluated = assertCompletedJob(item, hskLevel, terminal, updates, patchRecords)
  if (item.expectations?.maximumFirstRegionReadyMs !== undefined) {
    evaluated.assertions.push(
      check(
        `performance.${item.id}.first-region-ready`,
        Number.isFinite(firstRegionReadyMs) &&
          firstRegionReadyMs <= item.expectations.maximumFirstRegionReadyMs,
        `<= ${item.expectations.maximumFirstRegionReadyMs} ms`,
        firstRegionReadyMs,
        'A real partial-viewport request must publish a completed local bubble before the rest of the tall image finishes.',
      ),
    )
  }
  const evidence = {
    caseId: item.id,
    hskLevel,
    objectPath,
    jobId: created.jobId,
    terminal,
    firstRegionReadyMs,
    updateTimeline,
    updates,
    regions: evaluated.regions,
    preservedArtwork: evaluated.preserved,
    patches: patchRecords,
    assertions: evaluated.assertions,
  }
  writeFileSync(
    resolve(outputDirectory, `${item.id}-hsk-${hskLevel}.json`),
    `${JSON.stringify(evidence, null, 2)}\n`,
  )
  return evidence
}

export async function runRegression(options) {
  const integrity = auditCorpus({
    manifestPath: options.manifestPath,
    corpusRoot: options.corpusRoot,
    selection: options.selection,
  })
  if (integrity.status !== 'passed') {
    return {
      schemaVersion: 1,
      status: 'failed',
      stage: 'corpus-integrity',
      integrity,
      assertions: integrity.assertions,
    }
  }
  const manifest = JSON.parse(readFileSync(options.manifestPath, 'utf8'))
  const selected = selectedCases(manifest, options.selection)
  const cases = options.caseId
    ? selected.filter((item) => item.id === options.caseId)
    : selected
  if (options.caseId && cases.length !== 1) {
    throw new Error(`Case ${options.caseId} is not present in selection ${options.selection}.`)
  }
  mkdirSync(options.outputDirectory, { recursive: true })
  const stateRoot = resolve(options.outputDirectory, 'daemon-state')
  mkdirSync(stateRoot, { recursive: true })
  const stdoutPath = resolve(options.outputDirectory, 'daemon.stdout.log')
  const stderrPath = resolve(options.outputDirectory, 'daemon.stderr.log')
  const stdout = openSync(stdoutPath, 'w')
  const stderr = openSync(stderrPath, 'w')
  const child = spawn(
    options.daemonPath,
    ['--state-dir', stateRoot, '--idle-milliseconds', '3600000'],
    { windowsHide: true, stdio: ['ignore', stdout, stderr] },
  )
  const jobRuns = []
  const chapterContext = new Map()
  const assertions = [...integrity.assertions]
  let buildFingerprint
  try {
    const state = await waitForState(child, resolve(stateRoot, 'daemon-state.json'), stderrPath)
    const baseUrl = `http://127.0.0.1:${state.port}`
    const session = await issueSession(baseUrl, state.controlSecret)
    buildFingerprint = session.buildFingerprint
    await ensureReady(baseUrl, session.token, options.timeoutMinutes)
    for (const item of cases) {
      const levels = item.id === DENSE_DIFFERENTIAL_CASE ? [2, 5] : [3]
      for (const hskLevel of levels) {
        const contextKey = `${item.chapterId}\0${hskLevel}`
        const precedingContext = chapterContext.get(contextKey) ?? []
        const run = await runJob({
          item,
          hskLevel,
          precedingContext,
          corpusRoot: options.corpusRoot,
          baseUrl,
          session,
          outputDirectory: options.outputDirectory,
          timeoutMinutes: options.timeoutMinutes,
        })
        jobRuns.push(run)
        const completedContext = run.regions
          .filter((region) => region.sourceEnglish?.trim() && region.displayedChinese?.trim())
          .sort(
            (left, right) =>
              (left.readingOrder ?? 0) - (right.readingOrder ?? 0) ||
              String(left.id).localeCompare(String(right.id)),
          )
          .map((region) => ({
            sourceEnglish: region.sourceEnglish,
            chinese: region.displayedChinese,
          }))
        chapterContext.set(
          contextKey,
          [...precedingContext, ...completedContext].slice(-6),
        )
        assertions.push(...run.assertions)
      }
    }
    const lowRun = jobRuns.find(
      (run) => run.caseId === DENSE_DIFFERENTIAL_CASE && run.hskLevel === 2,
    )
    const highRun = jobRuns.find(
      (run) => run.caseId === DENSE_DIFFERENTIAL_CASE && run.hskLevel === 5,
    )
    if (lowRun && highRun) assertions.push(...assertHskDifferential(lowRun, highRun))
    else if (cases.some((item) => item.id === DENSE_DIFFERENTIAL_CASE)) {
      assertions.push(
        check(
          'differential.hsk-2-vs-5.runs',
          false,
          'both HSK2 and HSK5 dense-page runs',
          jobRuns.map((run) => ({ caseId: run.caseId, hskLevel: run.hskLevel })),
        ),
      )
    }
  } finally {
    if (child.exitCode === null) child.kill()
    closeSync(stdout)
    closeSync(stderr)
  }
  const failures = assertions.filter((item) => !item.passed)
  const summary = {
    schemaVersion: 1,
    recordedAtUtc: new Date().toISOString(),
    status: failures.length === 0 ? 'passed' : 'failed',
    offline: true,
    selection: options.selection,
    buildFingerprint,
    corpusId: integrity.corpusId,
    caseCount: cases.length,
    jobCount: jobRuns.length,
    outputDirectory: options.outputDirectory,
    assertions,
    failures,
    jobs: jobRuns.map((run) => ({
      caseId: run.caseId,
      hskLevel: run.hskLevel,
      jobId: run.jobId,
      terminal: run.terminal,
      regionCount: run.regions.length,
      assertionFailures: run.assertions.filter((item) => !item.passed).length,
    })),
  }
  writeFileSync(
    resolve(options.outputDirectory, 'summary.json'),
    `${JSON.stringify(summary, null, 2)}\n`,
  )
  return summary
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : undefined
if (invokedPath === fileURLToPath(import.meta.url)) {
  try {
    const options = parseArguments(process.argv.slice(2))
    const summary = await runRegression(options)
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`)
    if (summary.status !== 'passed') process.exitCode = 1
  } catch (error) {
    process.stderr.write(
      `${JSON.stringify(
        {
          schemaVersion: 1,
          status: 'error',
          offline: true,
          message: error instanceof Error ? error.message : String(error),
        },
        null,
        2,
      )}\n`,
    )
    process.exitCode = 1
  }
}
