import { createHash } from 'node:crypto'
import {
  closeSync,
  createReadStream,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  statSync,
  unlinkSync,
  writeSync,
} from 'node:fs'
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { basename, dirname, extname, join, normalize, resolve, sep } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { pathToFileURL } from 'node:url'
import { isDeepStrictEqual } from 'node:util'

const EXTENSION_ID = 'hsk-manga-translator@local.hskify'
const EXTENSION_UUID = '7e9a74d0-34ad-4ff7-9c2c-1ea555945100'
export const BUILD_FINGERPRINT = 'hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-27-r6'
const SESSION_STORAGE_KEY = 'hmt.nativeSession'
const ACTIVE_JOB_PREFIX = 'hmt.activeJob.'
const LONG_IMAGE_MIN_HEIGHT_PX = 10_000
const REGION_MATCH_MINIMUM_IOU = 0.5
const BENCHMARK_VIEWPORT_WIDTH = 1280
const BENCHMARK_VIEWPORT_HEIGHT = 1080
const MIN_VISIBLE_REGIONS = 3
const MAX_VISIBLE_REGIONS = 6
const MAX_PRECEDING_CONTEXT = 6
const BENCHMARK_ID = '30-years-since-the-prologue-chapter-5'
const EXPECTED_PAGE_COUNT = 36
const STORY_REGION_MINIMUM_RECALL = 0.95
const STORY_REGION_MINIMUM_OVERLAP = 0.5
const MAXIMUM_ENGLISH_OCR_CER = 0.03
const MAXIMUM_FALSE_TRANSLATION_RATE = 0.01
const INFERENCE_PROGRESS_STAGES = new Set([
  'decoding',
  'detecting',
  'ocr',
  'translating',
  'hsk-validating',
  'packaging',
])

export const BENCHMARK_LIMITS = Object.freeze({
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

export const BENCHMARK_QUALITY_LIMITS = Object.freeze({
  storyRegionRecall: STORY_REGION_MINIMUM_RECALL,
  englishOcrCer: MAXIMUM_ENGLISH_OCR_CER,
  falseTranslationRate: MAXIMUM_FALSE_TRANSLATION_RATE,
})

function fail(message) {
  throw new Error(message)
}

export function writeJsonSync(path, value) {
  const bytes = Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8')
  const handle = openSync(path, 'w')
  try {
    writeSync(handle, bytes)
    fsyncSync(handle)
  } finally {
    closeSync(handle)
  }
}

export function nowIso() {
  return new Date().toISOString()
}

function finiteNonNegative(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
}

function sameJson(left, right) {
  return isDeepStrictEqual(left, right)
}

export function exactChapterSnapshotMatch(expected, actual) {
  return Boolean(expected && actual && sameJson(expected, actual))
}

export function requestedZoomApplied(requested, inlineValue, computedValue) {
  const inline = Number.parseFloat(String(inlineValue))
  const computed = Number.parseFloat(String(computedValue))
  return (
    Number.isFinite(requested) &&
    requested > 0 &&
    Number.isFinite(inline) &&
    Number.isFinite(computed) &&
    Math.abs(inline - requested) <= 0.001 &&
    Math.abs(computed - requested) <= 0.001
  )
}

export function cancellationTiming({
  cancelIssuedAtEpochMs,
  pageRestoredAtEpochMs,
  daemonTerminalObservedAtEpochMs,
}) {
  for (const [name, value] of Object.entries({
    cancelIssuedAtEpochMs,
    pageRestoredAtEpochMs,
    daemonTerminalObservedAtEpochMs,
  })) {
    if (!finiteNonNegative(value)) fail(`${name} must be a finite non-negative timestamp.`)
  }
  if (
    pageRestoredAtEpochMs < cancelIssuedAtEpochMs ||
    daemonTerminalObservedAtEpochMs < cancelIssuedAtEpochMs
  ) {
    fail('Cancellation completion timestamps must not precede cancel issuance.')
  }
  return {
    cancelIssuedAtEpochMs,
    pageRestoredAtEpochMs,
    daemonTerminalObservedAtEpochMs,
    pageCancellationLatencyMs: pageRestoredAtEpochMs - cancelIssuedAtEpochMs,
    daemonCancellationLatencyMs: daemonTerminalObservedAtEpochMs - cancelIssuedAtEpochMs,
    measuredPhaseStartedAtEpochMs: cancelIssuedAtEpochMs,
    measuredPhaseEndedAtEpochMs: Math.max(pageRestoredAtEpochMs, daemonTerminalObservedAtEpochMs),
    timestampDefinition: {
      cancelIssuedAt:
        'Date.now() immediately before browser.runtime.sendMessage({type:"popup:cancel"}).',
      pageRestoredAt:
        'The first chapter MutationObserver callback whose exact DOM snapshot equals the pre-translation snapshot.',
      daemonTerminalObservedAt:
        'The first authenticated GET /jobs/{job_id}/updates response containing cancelled for every pre-cancel in-flight job.',
      excluded:
        'Later health, setup, full replay, patch download/decode, DOM evidence, and artifact collection.',
    },
  }
}

const RESOURCE_IDENTITY_KEYS = Object.freeze([
  'bytes',
  'filename',
  'id',
  'repository',
  'repositoryRevision',
  'sha256',
])

export function validateExpectedResourceIdentities(identities) {
  if (!Array.isArray(identities) || identities.length === 0) {
    fail('The pinned model manifest resourceIdentities array is missing or empty.')
  }
  const ids = new Set()
  for (const identity of identities) {
    if (
      !identity ||
      typeof identity !== 'object' ||
      Array.isArray(identity) ||
      !sameJson(Object.keys(identity).sort(), RESOURCE_IDENTITY_KEYS) ||
      typeof identity.id !== 'string' ||
      identity.id.length === 0 ||
      typeof identity.repository !== 'string' ||
      identity.repository.length === 0 ||
      typeof identity.filename !== 'string' ||
      identity.filename.length === 0 ||
      !/^[0-9a-f]{40}$/u.test(identity.repositoryRevision) ||
      !/^[0-9a-f]{64}$/u.test(identity.sha256) ||
      !Number.isSafeInteger(identity.bytes) ||
      identity.bytes <= 0 ||
      ids.has(identity.id)
    ) {
      fail('The pinned model manifest contains an invalid resource identity.')
    }
    ids.add(identity.id)
  }
  const declaredOrder = identities.map((identity) => identity.id)
  const sortedOrder = [...declaredOrder].sort((left, right) =>
    left < right ? -1 : left > right ? 1 : 0,
  )
  if (!sameJson(declaredOrder, sortedOrder)) {
    fail('The pinned model manifest resourceIdentities array must be sorted by id.')
  }
  return identities
}

function fileIdentity(path, file = basename(path)) {
  const bytes = readFileSync(path)
  return {
    file,
    bytes: bytes.byteLength,
    sha256: createHash('sha256').update(bytes).digest('hex'),
  }
}

function minimumRatioGate(id, actual, required, numerator, denominator, definition) {
  const measured = finiteNonNegative(actual)
  const passed = measured && actual >= required
  return {
    id,
    status: measured ? (passed ? 'pass' : 'fail') : 'missing',
    actual,
    required,
    operator: 'atLeast',
    numerator,
    denominator,
    definition,
    ...(!measured
      ? { reason: `${id} has no finite non-negative automatic measurement.` }
      : passed
        ? {}
        : { reason: `${actual} is below the required ${required}.` }),
  }
}

function requireFileIdentity(actual, expected, label) {
  if (
    !actual ||
    actual.file !== expected.file ||
    actual.bytes !== expected.bytes ||
    actual.sha256 !== expected.sha256
  ) {
    fail(
      `${label} identity mismatch: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}.`,
    )
  }
}

export function validateBenchmarkManifest(manifest) {
  if (
    !manifest ||
    typeof manifest !== 'object' ||
    Array.isArray(manifest) ||
    manifest.schemaVersion !== 3 ||
    manifest.id !== BENCHMARK_ID ||
    manifest.pageCount !== EXPECTED_PAGE_COUNT ||
    !Array.isArray(manifest.images) ||
    manifest.images.length !== manifest.pageCount
  ) {
    fail(`The benchmark requires the ${BENCHMARK_ID} ${EXPECTED_PAGE_COUNT}-image manifest.`)
  }
  const totals = {
    regionCount: manifest.totalExpectedRegionCount,
    goldBubbleCount: manifest.totalExpectedDialogueBubbleCount,
    narrationRegionCount: manifest.totalExpectedNarrationCount,
    englishTranslationTargetCount: manifest.totalExpectedEnglishTranslationTargetCount,
    untouchedExclusionCount: manifest.totalExpectedUntouchedExclusionCount,
  }
  if (
    !Object.values(totals).every((value) => Number.isInteger(value) && value >= 0) ||
    totals.regionCount < 1 ||
    totals.regionCount !== totals.goldBubbleCount + totals.narrationRegionCount ||
    totals.regionCount !==
      totals.englishTranslationTargetCount + totals.untouchedExclusionCount
  ) {
    fail('The benchmark manifest has inconsistent canonical region totals.')
  }
  const pageOrders = manifest.images.map((image) => image.order)
  if (
    new Set(pageOrders).size !== manifest.pageCount ||
    pageOrders.some((order) => !Number.isInteger(order) || order < 1 || order > manifest.pageCount)
  ) {
    fail('The benchmark manifest must identify every image with one unique in-range page order.')
  }
  const pageTotals = manifest.images.reduce(
    (sum, image) => {
      const values = [
        image.expectedRegionCount,
        image.expectedDialogueBubbleCount,
        image.expectedNarrationCount,
        image.expectedEnglishTranslationTargetCount,
        image.expectedUntouchedExclusionCount,
      ]
      if (
        !values.every((value) => Number.isInteger(value) && value >= 0) ||
        image.expectedRegionCount !==
          image.expectedDialogueBubbleCount + image.expectedNarrationCount ||
        image.expectedRegionCount !==
          image.expectedEnglishTranslationTargetCount + image.expectedUntouchedExclusionCount
      ) {
        fail(`Page ${image.order} has inconsistent canonical region counts.`)
      }
      return {
        regionCount: sum.regionCount + image.expectedRegionCount,
        goldBubbleCount: sum.goldBubbleCount + image.expectedDialogueBubbleCount,
        narrationRegionCount: sum.narrationRegionCount + image.expectedNarrationCount,
        englishTranslationTargetCount:
          sum.englishTranslationTargetCount + image.expectedEnglishTranslationTargetCount,
        untouchedExclusionCount:
          sum.untouchedExclusionCount + image.expectedUntouchedExclusionCount,
      }
    },
    {
      regionCount: 0,
      goldBubbleCount: 0,
      narrationRegionCount: 0,
      englishTranslationTargetCount: 0,
      untouchedExclusionCount: 0,
    },
  )
  if (!sameJson(pageTotals, totals)) {
    fail('The benchmark page counts do not sum to the canonical manifest totals.')
  }
  const status = manifest.annotationStatus
  if (
    !status ||
    typeof status !== 'object' ||
    Array.isArray(status) ||
    !['complete', 'incomplete'].includes(status.status) ||
    status.reviewedPageCount !== manifest.pageCount ||
    status.generatedPageCount !== manifest.pageCount ||
    status.requiredPageCount !== manifest.pageCount ||
    !Number.isInteger(status.completedPageCount) ||
    status.completedPageCount < 0 ||
    status.completedPageCount > manifest.pageCount ||
    !Number.isInteger(status.totalMissingFieldCount) ||
    status.totalMissingFieldCount < 0 ||
    !Array.isArray(status.missingPages)
  ) {
    fail('The benchmark manifest has an inconsistent annotationStatus object.')
  }
  if (
    (status.status === 'complete' &&
      (status.completedPageCount !== manifest.pageCount ||
        status.totalMissingFieldCount !== 0 ||
        status.missingPages.length !== 0)) ||
    (status.status === 'incomplete' &&
      (status.completedPageCount >= manifest.pageCount ||
        status.totalMissingFieldCount < 1 ||
        status.missingPages.length < 1))
  ) {
    fail('The benchmark annotationStatus completeness fields contradict each other.')
  }
  return totals
}

export function assertCompleteTranslationGold(manifest) {
  validateBenchmarkManifest(manifest)
  const status = manifest.annotationStatus
  if (
    status.status !== 'complete' ||
    status.completedPageCount !== manifest.pageCount ||
    status.totalMissingFieldCount !== 0 ||
    status.missingPages.length !== 0
  ) {
    fail(
      `Chapter 5 release measurement is blocked by incomplete translation gold: status=${status.status}, completedPageCount=${status.completedPageCount}/${status.requiredPageCount}, reasonCode=${status.reasonCode}, missingFieldCounts=${JSON.stringify(status.missingFieldCounts)}.`,
    )
  }
}

function measuredGate(id, actual, limit, unit = 'ms') {
  if (!finiteNonNegative(actual)) {
    return {
      id,
      status: 'missing',
      limit,
      unit,
      reason: `${id} has no finite non-negative automatic measurement.`,
    }
  }
  return {
    id,
    status: actual <= limit ? 'pass' : 'fail',
    actual,
    limit,
    unit,
    ...(actual <= limit ? {} : { reason: `${actual} ${unit} exceeds ${limit} ${unit}.` }),
  }
}

function exactGate(id, actual, expected, evidence = {}) {
  const passed = actual === expected
  return {
    id,
    status: passed ? 'pass' : 'fail',
    actual,
    expected,
    ...evidence,
    ...(passed
      ? {}
      : { reason: `Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}.` }),
  }
}

function booleanGate(id, actual, missingReason, evidence = {}) {
  if (typeof actual !== 'boolean') {
    return { id, status: 'missing', reason: missingReason, ...evidence }
  }
  return {
    id,
    status: actual ? 'pass' : 'fail',
    actual,
    ...evidence,
    ...(actual ? {} : { reason: missingReason }),
  }
}

export function assertRequiredGates(gates, scope) {
  const failures = gates.filter((gate) => gate.status !== 'pass')
  if (failures.length === 0) return
  fail(
    `${scope} gate failure:\n${failures
      .map((gate) => ` - ${gate.id}: ${gate.reason ?? `status ${gate.status}`}`)
      .join('\n')}`,
  )
}

function mimeType(path) {
  switch (extname(path).toLowerCase()) {
    case '.html':
      return 'text/html; charset=utf-8'
    case '.css':
      return 'text/css; charset=utf-8'
    case '.js':
    case '.mjs':
      return 'text/javascript; charset=utf-8'
    case '.json':
      return 'application/json; charset=utf-8'
    case '.webp':
      return 'image/webp'
    default:
      return 'application/octet-stream'
  }
}

function startReplicaServer(repositoryRoot, requestedPort) {
  const root = resolve(repositoryRoot)
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1')
      const relative = decodeURIComponent(url.pathname).replace(/^\/+/, '')
      const candidate = resolve(root, normalize(relative))
      if (candidate !== root && !candidate.startsWith(`${root}${sep}`)) {
        response.writeHead(403).end('Forbidden')
        return
      }
      if (!existsSync(candidate) || !statSync(candidate).isFile()) {
        response.writeHead(404).end('Not found')
        return
      }
      const immutable = extname(candidate).toLowerCase() === '.webp'
      response.writeHead(200, {
        'Content-Type': mimeType(candidate),
        'Content-Length': statSync(candidate).size,
        'Cache-Control': immutable ? 'private, max-age=31536000, immutable' : 'no-cache',
        'Cross-Origin-Resource-Policy': 'same-origin',
      })
      await pipeline(createReadStream(candidate), response)
    } catch (error) {
      if (!response.headersSent) response.writeHead(500)
      response.end(error instanceof Error ? error.message : String(error))
    }
  })
  return new Promise((resolvePromise, rejectPromise) => {
    server.once('error', rejectPromise)
    server.listen(requestedPort, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        rejectPromise(new Error('Replica server did not bind an IPv4 TCP port.'))
        return
      }
      resolvePromise({ server, port: address.port })
    })
  })
}

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds))
}

function decodeBidiRemoteValue(remote) {
  if (!remote || typeof remote !== 'object') return remote
  if (remote.type === 'undefined') return undefined
  if (remote.type === 'null') return null
  if (remote.type === 'number') {
    if (remote.value === 'NaN') return Number.NaN
    if (remote.value === '-0') return -0
    if (remote.value === 'Infinity') return Number.POSITIVE_INFINITY
    if (remote.value === '-Infinity') return Number.NEGATIVE_INFINITY
    return remote.value
  }
  if (
    remote.type === 'string' ||
    remote.type === 'boolean' ||
    remote.type === 'bigint'
  ) {
    return remote.value
  }
  if (remote.type === 'array') {
    return (remote.value ?? []).map((item) => decodeBidiRemoteValue(item))
  }
  if (remote.type === 'object') {
    return Object.fromEntries(
      (remote.value ?? []).map(([key, value]) => [
        typeof key === 'string' ? key : decodeBidiRemoteValue(key),
        decodeBidiRemoteValue(value),
      ]),
    )
  }
  return remote.value
}

function evaluationExpression(pageFunction, argument) {
  const source =
    typeof pageFunction === 'function' ? pageFunction.toString() : String(pageFunction)
  const serialized = JSON.stringify(argument)
  return `(${source})(${serialized === undefined ? 'undefined' : serialized})`
}

async function waitForBidiServer(profileDirectory, timeoutMs = 30_000) {
  const path = join(profileDirectory, 'WebDriverBiDiServer.json')
  const deadline = Date.now() + timeoutMs
  let lastError
  while (Date.now() < deadline) {
    try {
      const parsed = JSON.parse(readFileSync(path, 'utf8'))
      if (
        typeof parsed.ws_host === 'string' &&
        parsed.ws_host.length > 0 &&
        Number.isInteger(parsed.ws_port) &&
        parsed.ws_port > 0 &&
        parsed.ws_port <= 65_535
      ) {
        return parsed
      }
      lastError = new Error('WebDriverBiDiServer.json has no valid ws_host/ws_port.')
    } catch (error) {
      lastError = error
    }
    await delay(50)
  }
  throw new Error(
    `Firefox WebDriver BiDi did not become ready: ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }`,
  )
}

class FirefoxBidiClient {
  constructor(socket) {
    this.socket = socket
    this.nextId = 0
    this.pending = new Map()
    this.closed = false
    socket.addEventListener('message', (event) => this.handleMessage(event))
    socket.addEventListener('close', () => this.handleClose())
    socket.addEventListener('error', () => this.handleClose())
  }

  handleMessage(event) {
    let message
    try {
      message = JSON.parse(String(event.data))
    } catch {
      return
    }
    if (!Number.isInteger(message.id)) return
    const pending = this.pending.get(message.id)
    if (!pending) return
    this.pending.delete(message.id)
    clearTimeout(pending.timer)
    if (message.type === 'success') {
      pending.resolve(message.result)
      return
    }
    pending.reject(
      new Error(
        `${pending.method} failed: ${message.error ?? 'unknown error'}: ${
          message.message ?? 'Firefox returned no detail.'
        }`,
      ),
    )
  }

  handleClose() {
    if (this.closed) return
    this.closed = true
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer)
      pending.reject(new Error(`${pending.method} failed because Firefox BiDi closed.`))
    }
    this.pending.clear()
  }

  command(method, params = {}, timeoutMs = 30_000) {
    if (this.closed) {
      return Promise.reject(new Error(`${method} cannot run because Firefox BiDi is closed.`))
    }
    return new Promise((resolvePromise, rejectPromise) => {
      const id = ++this.nextId
      const timer = setTimeout(() => {
        if (!this.pending.delete(id)) return
        rejectPromise(new Error(`${method} timed out after ${timeoutMs} ms.`))
      }, timeoutMs)
      this.pending.set(id, {
        method,
        resolve: resolvePromise,
        reject: rejectPromise,
        timer,
      })
      this.socket.send(JSON.stringify({ id, method, params }))
    })
  }

  close() {
    if (this.closed) return
    this.socket.close()
    this.handleClose()
  }
}

async function connectFirefoxBidi(profileDirectory) {
  if (typeof WebSocket !== 'function') {
    fail('The Firefox benchmark requires Node.js with the built-in WebSocket API.')
  }
  const server = await waitForBidiServer(profileDirectory)
  const host = server.ws_host.includes(':') ? `[${server.ws_host}]` : server.ws_host
  const socket = new WebSocket(`ws://${host}:${server.ws_port}/session`)
  await new Promise((resolvePromise, rejectPromise) => {
    const timer = setTimeout(
      () => rejectPromise(new Error('Timed out connecting to Firefox WebDriver BiDi.')),
      30_000,
    )
    socket.addEventListener(
      'open',
      () => {
        clearTimeout(timer)
        resolvePromise()
      },
      { once: true },
    )
    socket.addEventListener(
      'error',
      () => {
        clearTimeout(timer)
        rejectPromise(new Error('Firefox WebDriver BiDi connection failed.'))
      },
      { once: true },
    )
  })
  return new FirefoxBidiClient(socket)
}

class BidiExtensionPage {
  constructor(client, browsingContext) {
    this.client = client
    this.browsingContext = browsingContext
  }

  async evaluate(pageFunction, argument) {
    const result = await this.client.command('script.evaluate', {
      expression: evaluationExpression(pageFunction, argument),
      target: { context: this.browsingContext },
      awaitPromise: true,
      resultOwnership: 'none',
    })
    if (result.type !== 'success') {
      const detail = result.exceptionDetails?.text ?? 'Extension evaluation failed.'
      throw new Error(detail)
    }
    return decodeBidiRemoteValue(result.result)
  }

  async waitForFunction(pageFunction, argument, options = {}) {
    const timeoutMs = options.timeout ?? 30_000
    const pollingMs =
      typeof options.polling === 'number' && options.polling > 0 ? options.polling : 100
    const deadline = Date.now() + timeoutMs
    let value
    while (Date.now() < deadline) {
      value = await this.evaluate(pageFunction, argument)
      if (value) return { jsonValue: async () => value }
      await delay(pollingMs)
    }
    throw new Error(`Extension condition timed out after ${timeoutMs} ms.`)
  }

  locator(selector) {
    return {
      click: async () => {
        const point = await this.evaluate((query) => {
          const element = document.querySelector(query)
          if (!(element instanceof HTMLElement)) {
            throw new Error(`Extension element not found: ${query}`)
          }
          const rect = element.getBoundingClientRect()
          return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }
        }, selector)
        await this.client.command('input.performActions', {
          context: this.browsingContext,
          actions: [
            {
              type: 'pointer',
              id: 'hskify-benchmark-mouse',
              parameters: { pointerType: 'mouse' },
              actions: [
                {
                  type: 'pointerMove',
                  x: Math.round(point.x),
                  y: Math.round(point.y),
                  duration: 0,
                  origin: 'viewport',
                },
                { type: 'pointerDown', button: 0 },
                { type: 'pointerUp', button: 0 },
              ],
            },
          ],
        })
        await this.client
          .command('input.releaseActions', { context: this.browsingContext })
          .catch(() => undefined)
      },
    }
  }

  close() {
    this.client.close()
  }
}

export async function launchPackagedFirefox(config) {
  mkdirSync(config.profileDirectory, { recursive: true })
  const require = createRequire(import.meta.url)
  const { firefox } = require(config.playwrightModule)
  let context
  let bidi
  try {
    context = await firefox.launchPersistentContext(config.profileDirectory, {
      executablePath: config.firefoxExecutable,
      headless: !config.headed,
      args: ['--remote-debugging-port=0'],
      firefoxUserPrefs: {
        'extensions.webextensions.uuids': JSON.stringify({
          [EXTENSION_ID]: EXTENSION_UUID,
        }),
        ...(config.firefoxUserPrefs ?? {}),
      },
    })
    const bootstrap = context.pages()[0] ?? (await context.newPage())
    bidi = await connectFirefoxBidi(config.profileDirectory)
    await bidi.command('session.new', { capabilities: {} })
    const installed = await bidi.command('webExtension.install', {
      extensionData: {
        type: 'archivePath',
        path: resolve(config.extensionPackagePath),
      },
      'moz:permanent': false,
    })
    if (installed.extension !== EXTENSION_ID) {
      fail(`Packaged Firefox installed the wrong extension ID: ${installed.extension}.`)
    }
    const realms = await bidi.command('script.getRealms')
    const bootstrapRealm = realms.realms?.find(
      (realm) => realm.type === 'window' && typeof realm.context === 'string',
    )
    if (!bootstrapRealm) fail('Firefox BiDi exposed no bootstrap browsing context.')
    await bidi.command('browsingContext.navigate', {
      context: bootstrapRealm.context,
      url: `moz-extension://${EXTENSION_UUID}/popup.html`,
      wait: 'none',
    })
    const extensionPage = new BidiExtensionPage(bidi, bootstrapRealm.context)
    await extensionPage.waitForFunction(
      () => document.readyState === 'complete' && typeof globalThis.browser === 'object',
      undefined,
      { timeout: 30_000, polling: 50 },
    )
    const identity = await extensionPage.evaluate(async () => ({
      id: globalThis.browser.runtime.id,
      manifest: globalThis.browser.runtime.getManifest(),
      origin: new URL(globalThis.browser.runtime.getURL('')).origin,
      commands: globalThis.browser.commands?.getAll
        ? await globalThis.browser.commands.getAll()
        : [],
    }))
    if (identity.id !== EXTENSION_ID) {
      fail(`Packaged Firefox extension ID mismatch: ${identity.id}.`)
    }
    if (identity.manifest.version !== config.extensionVersion) {
      fail(
        `Packaged Firefox extension version mismatch: ${identity.manifest.version} != ${config.extensionVersion}.`,
      )
    }
    return { context, extensionPage, identity }
  } catch (error) {
    bidi?.close()
    await context?.close().catch(() => undefined)
    throw error
  }
}

export async function extensionMessage(extensionPage, message) {
  const response = await extensionPage.evaluate(
    async (payload) => globalThis.browser.runtime.sendMessage(payload),
    message,
  )
  if (!response || response.ok !== true) {
    const code = response?.error?.code ?? 'EXTENSION_MESSAGE_FAILED'
    const detail = response?.error?.message ?? 'The packaged extension returned no response.'
    throw new Error(`${code}: ${detail}`)
  }
  return response.value
}

async function startJobMonitor(extensionPage, pageUrl, runId) {
  await extensionPage.evaluate(
    ({ prefix, expectedPageUrl, id }) => {
      if (globalThis.__hskifyJobMonitor?.timer) {
        clearInterval(globalThis.__hskifyJobMonitor.timer)
      }
      const monitor = {
        runId: id,
        pageUrl: expectedPageUrl,
        actionIssuedAtEpochMs: 0,
        observations: new Map(),
        errors: [],
        timer: 0,
        onStorageChanged: undefined,
      }
      const observe = (key, value) => {
        if (
          !key.startsWith(prefix) ||
          value?.pageUrl !== expectedPageUrl ||
          typeof value?.createdAtUnixMs !== 'number' ||
          value.createdAtUnixMs < monitor.actionIssuedAtEpochMs
        ) {
          return
        }
        const now = Date.now()
        const previous = monitor.observations.get(value.jobId)
        monitor.observations.set(value.jobId, {
          jobId: value.jobId,
          pageIndex: value.pageIndex,
          sourceSha256: value.sourceSha256,
          sourceWidth: value.sourceWidth,
          sourceHeight: value.sourceHeight,
          submittedRequest: value.submittedRequest,
          uploadedImageBytes: value.uploadedImageBytes,
          submittedAtUnixMs: value.submittedAtUnixMs,
          createdAtUnixMs: value.createdAtUnixMs,
          firstObservedAtEpochMs: previous?.firstObservedAtEpochMs ?? now,
          terminalType: value.terminalType,
          terminalObservedAtEpochMs:
            previous?.terminalObservedAtEpochMs ?? (value.terminalType ? now : undefined),
        })
      }
      const sample = async () => {
        try {
          const values = await globalThis.browser.storage.local.get(null)
          for (const [key, value] of Object.entries(values)) observe(key, value)
        } catch (error) {
          monitor.errors.push(error instanceof Error ? error.message : String(error))
        }
      }
      monitor.onStorageChanged = (changes, areaName) => {
        if (areaName !== 'local') return
        for (const [key, change] of Object.entries(changes)) {
          observe(key, change.newValue ?? change.oldValue)
        }
      }
      globalThis.browser.storage.onChanged.addListener(monitor.onStorageChanged)
      monitor.sample = sample
      monitor.timer = setInterval(() => void sample(), 10)
      globalThis.__hskifyJobMonitor = monitor
    },
    { prefix: ACTIVE_JOB_PREFIX, expectedPageUrl: pageUrl, id: runId },
  )
}

async function timedExtensionMessage(extensionPage, message) {
  const timed = await extensionPage.evaluate(async (payload) => {
    const issuedAtEpochMs = Date.now()
    if (
      globalThis.__hskifyJobMonitor &&
      globalThis.__hskifyJobMonitor.actionIssuedAtEpochMs === 0
    ) {
      globalThis.__hskifyJobMonitor.actionIssuedAtEpochMs = issuedAtEpochMs
    }
    const response = await globalThis.browser.runtime.sendMessage(payload)
    return { issuedAtEpochMs, responseAtEpochMs: Date.now(), response }
  }, message)
  if (!timed.response || timed.response.ok !== true) {
    const code = timed.response?.error?.code ?? 'EXTENSION_MESSAGE_FAILED'
    const detail = timed.response?.error?.message ?? 'The packaged extension returned no response.'
    throw new Error(`${code}: ${detail}`)
  }
  return {
    issuedAtEpochMs: timed.issuedAtEpochMs,
    responseAtEpochMs: timed.responseAtEpochMs,
    value: timed.response.value,
  }
}

export async function timedContentStart(
  extensionPage,
  hskLevel,
  expectedPageUrl,
  nameTranslation = 'keep-original',
) {
  const timed = await extensionPage.evaluate(
    async ({ level, pageUrl, names }) => {
      const tabs = await globalThis.browser.tabs.query({})
      const tab = tabs.find((candidate) => candidate.url === pageUrl)
      if (!Number.isInteger(tab?.id)) {
        throw new Error(`Packaged Firefox has no chapter tab for ${pageUrl}.`)
      }
      const issuedAtEpochMs = Date.now()
      if (
        globalThis.__hskifyJobMonitor &&
        globalThis.__hskifyJobMonitor.actionIssuedAtEpochMs === 0
      ) {
        globalThis.__hskifyJobMonitor.actionIssuedAtEpochMs = issuedAtEpochMs
      }
      const response = await globalThis.browser.tabs.sendMessage(tab.id, {
        type: 'content:start',
        scope: 'all',
        hskLevel: level,
        nameTranslation: names,
      })
      return {
        issuedAtEpochMs,
        responseAtEpochMs: Date.now(),
        response,
      }
    },
    { level: hskLevel, pageUrl: expectedPageUrl, names: nameTranslation },
  )
  if (
    !timed.response ||
    typeof timed.response !== 'object' ||
    typeof timed.response.state !== 'string'
  ) {
    fail('The packaged content runtime returned no valid start state.')
  }
  return {
    issuedAtEpochMs: timed.issuedAtEpochMs,
    responseAtEpochMs: timed.responseAtEpochMs,
    value: timed.response,
  }
}

export async function prepareContentRuntime(extensionPage, expectedPageUrl) {
  return extensionPage.evaluate(async (pageUrl) => {
    const tabs = await globalThis.browser.tabs.query({})
    const tab = tabs.find((candidate) => candidate.url === pageUrl)
    if (!Number.isInteger(tab?.id)) {
      throw new Error(`Packaged Firefox has no chapter tab for ${pageUrl}.`)
    }
    await globalThis.browser.scripting.executeScript({
      target: { tabId: tab.id, allFrames: false },
      files: ['translator.js'],
    })
    const state = await globalThis.browser.tabs.sendMessage(tab.id, {
      type: 'content:state',
    })
    if (!state || typeof state !== 'object' || typeof state.state !== 'string') {
      throw new Error(`The packaged content runtime did not initialize for ${pageUrl}.`)
    }
    return {
      tabId: tab.id,
      state,
    }
  }, expectedPageUrl)
}

async function stopJobMonitor(extensionPage) {
  return extensionPage.evaluate(async () => {
    const monitor = globalThis.__hskifyJobMonitor
    if (!monitor) return { observations: [], errors: ['job monitor was not installed'] }
    clearInterval(monitor.timer)
    if (monitor.onStorageChanged) {
      globalThis.browser.storage.onChanged.removeListener(monitor.onStorageChanged)
    }
    await monitor.sample()
    const result = {
      actionIssuedAtEpochMs: monitor.actionIssuedAtEpochMs,
      observations: [...monitor.observations.values()].sort(
        (left, right) => left.pageIndex - right.pageIndex,
      ),
      errors: [...monitor.errors],
    }
    delete globalThis.__hskifyJobMonitor
    return result
  })
}

export async function waitForPageState(extensionPage, chapterPage, expected, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  let last
  while (Date.now() < deadline) {
    await chapterPage.bringToFront()
    last = await extensionMessage(extensionPage, { type: 'popup:state' })
    if (expected.includes(last.state)) return last
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 200))
  }
  throw new Error(
    `Timed out waiting for extension state ${expected.join('/')} (last: ${JSON.stringify(last)}).`,
  )
}

export async function installDomObserver(page, runId) {
  await page.evaluate((id) => {
    for (const observer of globalThis.__hskifyRuntimeEvidence?.observers ?? []) {
      observer.disconnect()
    }
    const state = {
      runId: id,
      observerInstalledAtEpochMs: Date.now(),
      nextEventIndex: 1,
      events: [],
      observedShadowRoots: 0,
      observers: [],
      lastHudState: '',
    }
    globalThis.__hskifyRuntimeEvidence = state
    const observed = new WeakSet()
    const emit = (type, details = {}) => {
      state.events.push({
        index: state.nextEventIndex++,
        type,
        epochMs: Date.now(),
        performanceMs: performance.now(),
        ...details,
      })
    }
    const pageFor = (element) => {
      const root = element.getRootNode()
      const host = root instanceof ShadowRoot ? root.host : element
      return Number(host.closest('.hmt-wrapper')?.dataset.hmtSourcePage ?? 0)
    }
    const isVisible = (element) => {
      const rect = element.getBoundingClientRect()
      const style = getComputedStyle(element)
      return (
        rect.width > 0 &&
        rect.height > 0 &&
        rect.bottom > 0 &&
        rect.right > 0 &&
        rect.top < innerHeight &&
        rect.left < innerWidth &&
        style.visibility !== 'hidden' &&
        style.display !== 'none'
      )
    }
    const recordHudState = (root) => {
      const title = root.querySelector('.title')?.textContent?.trim() ?? ''
      const detail = root.querySelector('.detail')?.textContent?.trim() ?? ''
      if (!title || title === state.lastHudState) return
      state.lastHudState = title
      if (title === 'Hskify' || title.startsWith('Image ')) {
        emit('hudAcknowledged', { title, detail })
      }
      if (title === 'Translation complete') emit('hudComplete', { title, detail })
      if (title === 'Translation cancelled') emit('hudCancelled', { title, detail })
      if (title === 'Translation needs attention') emit('hudFailed', { title, detail })
    }
    const observeShadow = (root) => {
      if (observed.has(root)) return
      observed.add(root)
      state.observedShadowRoots += 1
      const observer = new MutationObserver((records) => {
        for (const record of records) {
          for (const node of record.addedNodes) recordElement(node)
          for (const node of record.removedNodes) {
            if (node instanceof Element && node.matches('.hmt-patch, .hmt-region')) {
              emit('translatedNodeRemoved', {
                className: node.className,
                page: pageFor(node),
              })
            }
          }
        }
        recordHudState(root)
      })
      observer.observe(root, {
        childList: true,
        subtree: true,
        characterData: true,
        attributes: true,
        attributeFilter: ['hidden', 'aria-pressed', 'aria-busy'],
      })
      state.observers.push(observer)
      for (const child of root.children) recordElement(child)
      recordHudState(root)
    }
    const recordElement = (element) => {
      if (!(element instanceof Element)) return
      const patchNodes = [
        ...(element.matches('.hmt-patch') ? [element] : []),
        ...element.querySelectorAll('.hmt-patch'),
      ]
      for (const patch of patchNodes) {
        emit('patchDomCommitted', {
          patchId: patch.dataset.patchId ?? '',
          complete: patch.complete,
          naturalWidth: patch.naturalWidth,
          naturalHeight: patch.naturalHeight,
          decodedAndInstalled: patch.complete && patch.naturalWidth > 0 && patch.naturalHeight > 0,
          page: pageFor(patch),
          visible: isVisible(patch),
        })
      }
      const regionNodes = [
        ...(element.matches('.hmt-region') ? [element] : []),
        ...element.querySelectorAll('.hmt-region'),
      ]
      for (const region of regionNodes) {
        emit('selectableTextDomCommitted', {
          regionId: region.dataset.regionId ?? '',
          hskValid: region.dataset.hskValid ?? '',
          repairState: region.dataset.hskRepairState ?? '',
          text: region.textContent ?? '',
          pinyin: region.dataset.pinyin ?? '',
          page: pageFor(region),
          visible: isVisible(region),
        })
      }
      const ownedNodes = [
        ...(element.matches('[data-hmt-owned="true"]') ? [element] : []),
        ...element.querySelectorAll('[data-hmt-owned="true"]'),
      ]
      for (const owned of ownedNodes) {
        if (owned.classList.contains('hmt-wrapper')) {
          emit('imageWrapperCommitted', {
            page: Number(owned.querySelector('img[data-page]')?.dataset.page ?? 0),
          })
        }
        if (owned.shadowRoot) observeShadow(owned.shadowRoot)
      }
    }
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.addedNodes) recordElement(node)
      }
    })
    observer.observe(document.documentElement, { childList: true, subtree: true })
    state.observers.push(observer)
    for (const child of document.documentElement.children) recordElement(child)
    emit('observerReady')
  }, runId)
}

export async function captureChapterSnapshot(page, serializedChapterOuterHtml) {
  return page.evaluate((serialized) => {
    let chapter
    if (typeof serialized === 'string') {
      const template = document.createElement('template')
      template.innerHTML = serialized
      chapter = template.content.firstElementChild
    } else {
      chapter = document.querySelector('#chapter')
    }
    if (!(chapter instanceof HTMLElement)) {
      throw new Error('The chapter root is missing while capturing an exact DOM snapshot.')
    }
    const nodePath = (node) => {
      if (!node) return null
      const path = []
      let current = node
      while (current && current !== chapter) {
        const parent = current.parentNode
        if (!parent) return null
        path.unshift([...parent.childNodes].indexOf(current))
        current = parent
      }
      return current === chapter ? path : null
    }
    const nodeSnapshot = (node) => {
      if (!node) return null
      if (node instanceof Element) {
        return {
          nodeType: node.nodeType,
          nodeName: node.nodeName,
          outerHTML: node.outerHTML,
        }
      }
      return {
        nodeType: node.nodeType,
        nodeName: node.nodeName,
        data: node.data,
      }
    }
    return {
      outerHTML: chapter.outerHTML,
      attributes: [...chapter.attributes].map((attribute) => [attribute.name, attribute.value]),
      childNodes: [...chapter.childNodes].map(nodeSnapshot),
      images: [...chapter.querySelectorAll('img')].map((image) => ({
        path: nodePath(image),
        parentPath: nodePath(image.parentNode),
        previousSibling: nodeSnapshot(image.previousSibling),
        nextSibling: nodeSnapshot(image.nextSibling),
        outerHTML: image.outerHTML,
        attributes: [...image.attributes].map((attribute) => [attribute.name, attribute.value]),
        src: image.getAttribute('src'),
        srcset: image.getAttribute('srcset'),
        sizes: image.getAttribute('sizes'),
        class: image.getAttribute('class'),
        style: image.getAttribute('style'),
      })),
    }
  }, serializedChapterOuterHtml)
}

async function armExactChapterRestoration(page, expectedSnapshot) {
  await page.evaluate((expected) => {
    globalThis.__hskifyExactRestorationProbe?.observer?.disconnect()
    const chapter = document.querySelector('#chapter')
    if (!(chapter instanceof HTMLElement)) {
      throw new Error('The chapter root is missing while arming restoration timing.')
    }
    const state = {
      expectedOuterHTML: expected.outerHTML,
      armedAtEpochMs: Date.now(),
      restoredAtEpochMs: undefined,
      mutationBatches: 0,
      observer: undefined,
    }
    const check = () => {
      if (state.restoredAtEpochMs === undefined && chapter.outerHTML === state.expectedOuterHTML) {
        state.restoredAtEpochMs = Date.now()
        state.observer?.disconnect()
      }
    }
    state.observer = new MutationObserver(() => {
      state.mutationBatches += 1
      check()
    })
    state.observer.observe(chapter, {
      attributes: true,
      childList: true,
      subtree: true,
      characterData: true,
    })
    globalThis.__hskifyExactRestorationProbe = state
    check()
  }, expectedSnapshot)
}

async function waitForExactChapterRestoration(page, timeoutMs = 5_000) {
  await page.waitForFunction(
    () => Number.isFinite(globalThis.__hskifyExactRestorationProbe?.restoredAtEpochMs),
    undefined,
    { timeout: timeoutMs, polling: 10 },
  )
  return page.evaluate(() => {
    const probe = globalThis.__hskifyExactRestorationProbe
    if (!probe) throw new Error('The exact restoration probe disappeared.')
    probe.observer?.disconnect()
    return {
      expectedOuterHTML: probe.expectedOuterHTML,
      armedAtEpochMs: probe.armedAtEpochMs,
      restoredAtEpochMs: probe.restoredAtEpochMs,
      mutationBatches: probe.mutationBatches,
    }
  })
}

export async function chapterDomEvidence(page) {
  return page.evaluate(() => {
    const hosts = [...document.querySelectorAll('[data-hmt-owned="true"]')].filter(
      (node) => node.shadowRoot,
    )
    const patches = []
    const regions = []
    let degradedFitCount = 0
    for (const host of hosts) {
      const pageNumber = Number(host.closest('.hmt-wrapper')?.dataset.hmtSourcePage ?? 0)
      for (const patch of host.shadowRoot.querySelectorAll('.hmt-patch')) {
        patches.push({
          page: pageNumber,
          patchId: patch.dataset.patchId ?? '',
          complete: patch.complete,
          naturalWidth: patch.naturalWidth,
          naturalHeight: patch.naturalHeight,
        })
      }
      for (const region of host.shadowRoot.querySelectorAll('.hmt-region')) {
        if (region.dataset.fit === 'degraded') degradedFitCount += 1
        regions.push({
          page: pageNumber,
          regionId: region.dataset.regionId ?? '',
          text: region.textContent ?? '',
          pinyin: region.dataset.pinyin ?? '',
          hskValid: region.dataset.hskValid ?? '',
          repairState: region.dataset.hskRepairState ?? '',
          fit: region.dataset.fit ?? 'normal',
          overflows:
            region.scrollWidth > region.clientWidth + 0.5 ||
            region.scrollHeight > region.clientHeight + 0.5,
        })
      }
    }
    const observer = globalThis.__hskifyRuntimeEvidence
    const events = observer?.events ?? []
    const firstPatch = events.find((event) => event.type === 'patchDomCommitted')
    const firstText = events.find((event) => event.type === 'selectableTextDomCommitted')
    return {
      sourceImageCount: document.querySelectorAll('#chapter > img').length,
      wrappedImageCount: document.querySelectorAll('.hmt-wrapper').length,
      patchCount: patches.length,
      regionCount: regions.length,
      degradedFitCount,
      patches,
      regions,
      events,
      patchBeforeText:
        Boolean(firstPatch) && Boolean(firstText) && firstPatch.index < firstText.index,
      observerInstalledAtEpochMs: observer?.observerInstalledAtEpochMs,
    }
  })
}

async function sourceAcquisitionEvidence(page, actionIssuedAtEpochMs) {
  return page.evaluate((actionEpochMs) => {
    const imageEntries = performance
      .getEntriesByType('resource')
      .filter((entry) => /\/\d{3}\.webp(?:\?|$)/u.test(entry.name))
      .filter((entry) => performance.timeOrigin + entry.startTime >= actionEpochMs)
      .map((entry) => ({
        name: new URL(entry.name).pathname.split('/').pop(),
        startTimeMs: entry.startTime,
        durationMs: entry.duration,
        responseEndMs: entry.responseEnd,
        transferSize: entry.transferSize,
        encodedBodySize: entry.encodedBodySize,
        decodedBodySize: entry.decodedBodySize,
        initiatorType: entry.initiatorType,
      }))
    return {
      replica: globalThis.__hskifyBenchmark,
      imageEntries,
      actionIssuedAtEpochMs: actionEpochMs,
      note: 'Entries cover content-world source fetches visible to the document timeline; ActiveJobRecord.createdAtUnixMs separately bounds acquisition, SHA-256, upload, and daemon acceptance.',
    }
  }, actionIssuedAtEpochMs)
}

export async function sourceGlyphEvidence(page, routes) {
  const regions = routes.jobs.flatMap((job) => {
    const accepted = new Map(
      job.updates
        .filter((update) => update.type === 'regionReady')
        .map((update) => [update.region.id, update.region]),
    )
    return job.patches.map((patch) => {
      const region = accepted.get(patch.regionId)
      if (!region?.sourceEnglish) {
        fail(`Missing accepted source text for glyph audit region ${patch.regionId}.`)
      }
      return {
        page: job.pageIndex + 1,
        id: patch.regionId,
        sourceEnglish: region.sourceEnglish,
        textPolygon: patch.textPolygon,
      }
    })
  })
  return page.evaluate((expectedRegions) => {
    const images = [...document.querySelectorAll('#chapter img[data-page]')].sort(
      (left, right) => Number(left.dataset.page) - Number(right.dataset.page),
    )
    const point = (value) => (Array.isArray(value) ? value : [value.x, value.y])
    const otsu = (grayscale) => {
      const histogram = new Uint32Array(256)
      for (const value of grayscale) histogram[value] += 1
      let weightedTotal = 0
      for (let index = 0; index < histogram.length; index += 1) {
        weightedTotal += index * histogram[index]
      }
      let backgroundWeight = 0
      let backgroundSum = 0
      let bestThreshold = 0
      let bestVariance = -1
      for (let threshold = 0; threshold < 255; threshold += 1) {
        backgroundWeight += histogram[threshold]
        if (backgroundWeight === 0) continue
        const foregroundWeight = grayscale.length - backgroundWeight
        if (foregroundWeight === 0) break
        backgroundSum += threshold * histogram[threshold]
        const backgroundMean = backgroundSum / backgroundWeight
        const foregroundMean = (weightedTotal - backgroundSum) / foregroundWeight
        const variance =
          backgroundWeight *
          foregroundWeight *
          (backgroundMean - foregroundMean) *
          (backgroundMean - foregroundMean)
        if (variance > bestVariance) {
          bestVariance = variance
          bestThreshold = threshold
        }
      }
      return bestThreshold
    }
    return expectedRegions.map((expected) => {
      const image = images[expected.page - 1]
      if (!image || image.naturalWidth < 1 || image.naturalHeight < 1) {
        throw new Error(`Missing decoded source image for glyph audit page ${expected.page}.`)
      }
      const coordinates = expected.textPolygon.map(point)
      const x0 = Math.max(
        0,
        Math.floor(Math.min(...coordinates.map(([x]) => x)) * image.naturalWidth),
      )
      const y0 = Math.max(
        0,
        Math.floor(Math.min(...coordinates.map(([, y]) => y)) * image.naturalHeight),
      )
      const x1 = Math.min(
        image.naturalWidth,
        Math.ceil(Math.max(...coordinates.map(([x]) => x)) * image.naturalWidth),
      )
      const y1 = Math.min(
        image.naturalHeight,
        Math.ceil(Math.max(...coordinates.map(([, y]) => y)) * image.naturalHeight),
      )
      const width = x1 - x0
      const height = y1 - y0
      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height
      const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true })
      if (!context) throw new Error(`Could not create source glyph audit canvas for ${expected.id}.`)
      context.drawImage(image, x0, y0, width, height, 0, 0, width, height)
      const rgba = context.getImageData(0, 0, width, height).data
      const grayscale = new Uint8Array(width * height)
      for (let index = 0; index < grayscale.length; index += 1) {
        const offset = index * 4
        grayscale[index] = Math.round(
          0.299 * rgba[offset] + 0.587 * rgba[offset + 1] + 0.114 * rgba[offset + 2],
        )
      }
      const border = []
      for (let x = 0; x < width; x += 1) {
        border.push(grayscale[x], grayscale[(height - 1) * width + x])
      }
      for (let y = 0; y < height; y += 1) {
        border.push(grayscale[y * width], grayscale[y * width + width - 1])
      }
      border.sort((left, right) => left - right)
      const background = border[Math.floor(border.length / 2)]
      const threshold = otsu(grayscale)
      const darkInk = background >= threshold
      const core = new Uint8Array(width * height)
      for (let index = 0; index < grayscale.length; index += 1) {
        core[index] = Number(
          darkInk
            ? grayscale[index] <= Math.min(threshold, background - 8)
            : grayscale[index] >= Math.max(threshold, background + 8),
        )
      }
      let retained = core.slice()
      const queue = new Int32Array(retained.length)
      let queueHead = 0
      let queueTail = 0
      const rejectBorderPixel = (x, y) => {
        const index = y * width + x
        if (retained[index] === 0) return
        retained[index] = 0
        queue[queueTail] = index
        queueTail += 1
      }
      for (let x = 0; x < width; x += 1) {
        rejectBorderPixel(x, 0)
        rejectBorderPixel(x, height - 1)
      }
      for (let y = 0; y < height; y += 1) {
        rejectBorderPixel(0, y)
        rejectBorderPixel(width - 1, y)
      }
      while (queueHead < queueTail) {
        const index = queue[queueHead]
        queueHead += 1
        const x = index % width
        const y = Math.floor(index / width)
        for (let dy = -1; dy <= 1; dy += 1) {
          const neighborY = y + dy
          if (neighborY < 0 || neighborY >= height) continue
          for (let dx = -1; dx <= 1; dx += 1) {
            const neighborX = x + dx
            if (neighborX < 0 || neighborX >= width) continue
            rejectBorderPixel(neighborX, neighborY)
          }
        }
      }
      // Tight text polygons can put every glyph component on the crop border
      // (for example, large title lettering). In that case the normal contour
      // rejection would erase the entire independent audit mask, so retain
      // the polarity-selected core instead of producing unverifiable evidence.
      if (!retained.some((value) => value !== 0)) retained = core
      const mask = new Uint8Array(core.length)
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          let foreground = false
          for (let dy = -1; dy <= 1 && !foreground; dy += 1) {
            const sourceY = y + dy
            if (sourceY < 0 || sourceY >= height) continue
            for (let dx = -1; dx <= 1; dx += 1) {
              const sourceX = x + dx
              if (
                sourceX >= 0 &&
                sourceX < width &&
                retained[sourceY * width + sourceX] !== 0
              ) {
                foreground = true
                break
              }
            }
          }
          mask[y * width + x] = Number(foreground)
        }
      }
      const seen = new Uint8Array(mask.length)
      const componentQueue = new Int32Array(mask.length)
      const preserveHorizontalMarks = /[-\u2014\u2013_]|\.{2,}|\u2026/u.test(
        expected.sourceEnglish,
      )
      for (let start = 0; start < mask.length; start += 1) {
        if (mask[start] === 0 || seen[start] !== 0) continue
        let head = 0
        let tail = 0
        let left = width
        let top = height
        let right = 0
        let bottom = 0
        componentQueue[tail] = start
        tail += 1
        seen[start] = 1
        while (head < tail) {
          const index = componentQueue[head]
          head += 1
          const x = index % width
          const y = Math.floor(index / width)
          left = Math.min(left, x)
          top = Math.min(top, y)
          right = Math.max(right, x + 1)
          bottom = Math.max(bottom, y + 1)
          for (let dy = -1; dy <= 1; dy += 1) {
            const neighborY = y + dy
            if (neighborY < 0 || neighborY >= height) continue
            for (let dx = -1; dx <= 1; dx += 1) {
              const neighborX = x + dx
              if (neighborX < 0 || neighborX >= width) continue
              const neighbor = neighborY * width + neighborX
              if (mask[neighbor] === 0 || seen[neighbor] !== 0) continue
              seen[neighbor] = 1
              componentQueue[tail] = neighbor
              tail += 1
            }
          }
        }
        const componentWidth = right - left
        const componentHeight = bottom - top
        const detachedSpeck = tail < 25
        const shallowWideArtwork =
          !preserveHorizontalMarks &&
          componentHeight > 0 &&
          componentWidth / componentHeight >= 2.75 &&
          componentHeight <= Math.max(24, height * 0.06)
        if (detachedSpeck || shallowWideArtwork) {
          for (let index = 0; index < tail; index += 1) {
            mask[componentQueue[index]] = 0
          }
        }
      }
      const rows = []
      let pixels = 0
      for (let y = 0; y < height; y += 1) {
        const runs = []
        let start = -1
        for (let x = 0; x < width; x += 1) {
          if (mask[y * width + x] !== 0) {
            pixels += 1
            if (start < 0) start = x
          } else if (start >= 0) {
            runs.push([start, x])
            start = -1
          }
        }
        if (start >= 0) runs.push([start, width])
        if (runs.length > 0) rows.push({ y, runs })
      }
      return {
        page: expected.page,
        id: expected.id,
        originX: x0,
        originY: y0,
        width,
        height,
        pixels,
        rows,
        method:
          'accepted source-text crop grayscale Otsu against border-median polarity, rejecting border-connected foreground, detached sub-25-pixel specks, and isolated shallow-wide artwork before one-pixel audit dilation',
      }
    })
  }, regions)
}

export async function activeJobs(extensionPage, pageUrl) {
  return extensionPage.evaluate(
    async ({ prefix, expectedPageUrl }) => {
      const values = await globalThis.browser.storage.local.get(null)
      return Object.entries(values)
        .filter(([key]) => key.startsWith(prefix))
        .map(([, value]) => value)
        .filter((value) => value?.pageUrl === expectedPageUrl)
        .sort((left, right) => left.pageIndex - right.pageIndex)
    },
    { prefix: ACTIVE_JOB_PREFIX, expectedPageUrl: pageUrl },
  )
}

async function waitForInFlightJobs(extensionPage, pageUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  let records = []
  while (Date.now() < deadline) {
    records = (await activeJobs(extensionPage, pageUrl)).filter((record) => !record.terminalType)
    if (records.length > 0) return records
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 10))
  }
  throw new Error(`Expected a pre-cancel in-flight extension-owned job, found ${records.length}.`)
}

async function armDaemonCancellationProbe(extensionPage, records) {
  const targets = records
    .filter((record) => !record.terminalType)
    .map((record) => ({
      jobId: record.jobId,
      pageIndex: record.pageIndex,
      sourceSha256: record.sourceSha256,
    }))
  if (targets.length === 0) {
    fail('Cancellation timing requires a pre-cancel in-flight extension-owned job.')
  }
  return extensionPage.evaluate(
    async ({ jobs, sessionKey }) => {
      if (globalThis.__hskifyDaemonCancellationProbe) {
        globalThis.__hskifyDaemonCancellationProbe.stopped = true
      }
      const stored = await globalThis.browser.storage.session.get(sessionKey)
      const session = stored[sessionKey]
      if (!session || typeof session.token !== 'string' || typeof session.port !== 'number') {
        throw new Error('The packaged extension has no authenticated daemon session.')
      }
      const headers = {
        Authorization: `Bearer ${session.token}`,
        'X-HSK-Manga-Extension-Origin': new URL(globalThis.browser.runtime.getURL('')).origin,
      }
      const observe = async (job) => {
        const response = await fetch(
          `http://127.0.0.1:${session.port}/jobs/${encodeURIComponent(
            job.jobId,
          )}/updates?after=0&waitMs=0`,
          {
            headers,
            cache: 'no-store',
            redirect: 'error',
          },
        )
        const batch = await response.json()
        if (!response.ok || batch.jobId !== job.jobId) {
          throw new Error(
            `Cancellation terminal probe failed for ${job.jobId}: HTTP ${response.status}.`,
          )
        }
        const terminal = [...batch.updates]
          .reverse()
          .find((update) => ['complete', 'failed', 'cancelled'].includes(update.type))
        return {
          jobId: job.jobId,
          pageIndex: job.pageIndex,
          httpStatus: response.status,
          nextSequence: batch.nextSequence,
          terminalType: terminal?.type,
        }
      }
      const initial = await Promise.all(jobs.map(observe))
      const activeJobIds = new Set(
        initial
          .filter((observation) => observation.terminalType === undefined)
          .map((observation) => observation.jobId),
      )
      const activeJobs = jobs.filter((job) => activeJobIds.has(job.jobId))
      if (activeJobs.length === 0) {
        throw new Error('Cancellation timing found no daemon-confirmed nonterminal job.')
      }
      const state = {
        armedAtEpochMs: Date.now(),
        targets: activeJobs,
        observations: new Map(
          initial
            .filter((observation) => activeJobIds.has(observation.jobId))
            .map((observation) => [observation.jobId, observation]),
        ),
        terminalObservedAtEpochMs: undefined,
        stopped: false,
        errors: [],
      }
      const sample = async () => {
        const observations = await Promise.all(activeJobs.map(observe))
        for (const observation of observations) {
          state.observations.set(observation.jobId, observation)
        }
        if (
          observations.length === activeJobs.length &&
          observations.every((observation) => observation.terminalType === 'cancelled')
        ) {
          state.terminalObservedAtEpochMs = Date.now()
          state.stopped = true
        }
      }
      const loop = async () => {
        while (!state.stopped) {
          try {
            await sample()
          } catch (error) {
            state.errors.push(error instanceof Error ? error.message : String(error))
          }
          if (!state.stopped) {
            await new Promise((resolvePromise) => setTimeout(resolvePromise, 5))
          }
        }
      }
      globalThis.__hskifyDaemonCancellationProbe = state
      void loop()
      return activeJobs
    },
    { jobs: targets, sessionKey: SESSION_STORAGE_KEY },
  )
}

async function waitForDaemonCancellationProbe(extensionPage, timeoutMs = 30_000) {
  await extensionPage.waitForFunction(
    () => Number.isFinite(globalThis.__hskifyDaemonCancellationProbe?.terminalObservedAtEpochMs),
    undefined,
    { timeout: timeoutMs, polling: 10 },
  )
  return extensionPage.evaluate(() => {
    const probe = globalThis.__hskifyDaemonCancellationProbe
    if (!probe) throw new Error('The daemon cancellation probe disappeared.')
    probe.stopped = true
    return {
      armedAtEpochMs: probe.armedAtEpochMs,
      targets: probe.targets,
      observations: [...probe.observations.values()].sort(
        (left, right) => left.pageIndex - right.pageIndex,
      ),
      terminalObservedAtEpochMs: probe.terminalObservedAtEpochMs,
      errors: probe.errors,
      evidenceRoutes: ['GET /jobs/{job_id}/updates?after=0&waitMs=0'],
      healthRequests: 0,
      patchRequests: 0,
    }
  })
}

export async function routeEvidence(
  extensionPage,
  records,
  terminalRequired,
  expectedResourceIdentities,
) {
  return extensionPage.evaluate(
    async ({ jobs, sessionKey, buildFingerprint, requireTerminal, expectedIdentities }) => {
      const stored = await globalThis.browser.storage.session.get(sessionKey)
      const session = stored[sessionKey]
      if (!session || typeof session.token !== 'string' || typeof session.port !== 'number') {
        throw new Error('The packaged extension has no authenticated daemon session.')
      }
      const extensionOrigin = new URL(globalThis.browser.runtime.getURL('')).origin
      const headers = {
        Authorization: `Bearer ${session.token}`,
        'X-HSK-Manga-Extension-Origin': extensionOrigin,
      }
      const request = async (path) => {
        const started = performance.now()
        const response = await fetch(`http://127.0.0.1:${session.port}${path}`, {
          headers,
          cache: 'no-store',
          redirect: 'error',
        })
        return { response, durationMs: performance.now() - started }
      }
      const healthFetch = await request('/health')
      const health = await healthFetch.response.json()
      if (!healthFetch.response.ok || health.buildFingerprint !== buildFingerprint) {
        throw new Error(`Unversioned /health failed: HTTP ${healthFetch.response.status}.`)
      }
      const actualIdentities = Array.isArray(health.resourceIdentities)
        ? health.resourceIdentities
        : undefined
      const canonicalJson = (value) => {
        if (Array.isArray(value)) return value.map(canonicalJson)
        if (value && typeof value === 'object') {
          return Object.fromEntries(
            Object.keys(value)
              .sort()
              .map((key) => [key, canonicalJson(value[key])]),
          )
        }
        return value
      }
      const exactResourceIdentities =
        Array.isArray(expectedIdentities) &&
        expectedIdentities.length > 0 &&
        JSON.stringify(canonicalJson(actualIdentities)) ===
          JSON.stringify(canonicalJson(expectedIdentities))
      const setupFetch = await request('/setup')
      const setup = await setupFetch.response.json()
      if (!setupFetch.response.ok || setup.state !== 'ready') {
        throw new Error(`Installed resources are not ready: ${JSON.stringify(setup)}.`)
      }
      const jobsEvidence = []
      for (const job of jobs) {
        const updateFetch = await request(
          `/jobs/${encodeURIComponent(job.jobId)}/updates?after=0&waitMs=0`,
        )
        const batch = await updateFetch.response.json()
        if (!updateFetch.response.ok || batch.jobId !== job.jobId) {
          throw new Error(`Update replay failed for ${job.jobId}.`)
        }
        const terminal = [...batch.updates]
          .reverse()
          .find((update) => ['complete', 'failed', 'cancelled'].includes(update.type))
        if (requireTerminal && !terminal) {
          throw new Error(`Job ${job.jobId} has no terminal replay update.`)
        }
        const patches = []
        for (const update of batch.updates) {
          if (update.type !== 'regionReady') continue
          const patchId = update.region.patch.blobId
          const patchFetch = await request(`/blobs/${encodeURIComponent(patchId)}`)
          const contentType =
            patchFetch.response.headers.get('content-type')?.split(';', 1)[0]?.trim() ?? ''
          const bytes = await patchFetch.response.arrayBuffer()
          if (!patchFetch.response.ok || contentType !== 'image/png') {
            throw new Error(`Patch GET failed for ${patchId}.`)
          }
          const digest = await crypto.subtle.digest('SHA-256', bytes)
          const sha256 = [...new Uint8Array(digest)]
            .map((value) => value.toString(16).padStart(2, '0'))
            .join('')
          const blob = new Blob([bytes], { type: contentType })
          const url = URL.createObjectURL(blob)
          const image = document.createElement('img')
          image.src = url
          const decodeStarted = performance.now()
          let decodedPixels
          try {
            await image.decode()
            const canvas = document.createElement('canvas')
            canvas.width = image.naturalWidth
            canvas.height = image.naturalHeight
            const context = canvas.getContext('2d', {
              alpha: true,
              willReadFrequently: true,
            })
            if (!context) throw new Error(`Could not inspect decoded PNG pixels for ${patchId}.`)
            context.clearRect(0, 0, canvas.width, canvas.height)
            context.drawImage(image, 0, 0)
            decodedPixels = context.getImageData(0, 0, canvas.width, canvas.height)
          } finally {
            URL.revokeObjectURL(url)
          }
          if (!decodedPixels) {
            throw new Error(`Decoded PNG pixels are missing for ${patchId}.`)
          }
          const rgbaDigest = await crypto.subtle.digest('SHA-256', decodedPixels.data)
          const decodedRgbaSha256 = [...new Uint8Array(rgbaDigest)]
            .map((value) => value.toString(16).padStart(2, '0'))
            .join('')
          const alphaRows = []
          let alphaNonZeroPixelCount = 0
          let partialAlphaPixelCount = 0
          for (let y = 0; y < image.naturalHeight; y += 1) {
            const runs = []
            let start = -1
            for (let x = 0; x < image.naturalWidth; x += 1) {
              const alpha = decodedPixels.data[(y * image.naturalWidth + x) * 4 + 3]
              if (alpha > 0) {
                alphaNonZeroPixelCount += 1
                if (alpha < 255) partialAlphaPixelCount += 1
                if (start < 0) start = x
              } else if (start >= 0) {
                runs.push([start, x])
                start = -1
              }
            }
            if (start >= 0) runs.push([start, image.naturalWidth])
            if (runs.length > 0) alphaRows.push({ y, runs })
          }
          patches.push({
            route: `/blobs/${patchId}`,
            patchId,
            regionId: update.region.id,
            // Copy the scalars instead of retaining a second reference to the
            // update object. Firefox BiDi serializes that repeated reference
            // as an empty remote object even though the replayed update keeps
            // the original rectangle intact.
            rect: {
              x: update.region.patch.rect.x,
              y: update.region.patch.rect.y,
              width: update.region.patch.rect.width,
              height: update.region.patch.rect.height,
            },
            bubblePolygon: update.region.bubblePolygon?.map((point) => ({
              x: point.x,
              y: point.y,
            })),
            textPolygon: update.region.textPolygon.map((point) => ({
              x: point.x,
              y: point.y,
            })),
            httpStatus: patchFetch.response.status,
            getDurationMs: patchFetch.durationMs,
            decodeDurationMs: performance.now() - decodeStarted,
            bytes: bytes.byteLength,
            sha256,
            decodedRgbaSha256,
            decodedPixelMethod:
              'Firefox HTMLImageElement.decode then CanvasRenderingContext2D.getImageData',
            width: image.naturalWidth,
            height: image.naturalHeight,
            alphaNonZeroPixelCount,
            partialAlphaPixelCount,
            alphaRows,
          })
        }
        const stageCounts = {}
        for (const update of batch.updates) {
          if (update.type !== 'progress') continue
          stageCounts[update.stage] = (stageCounts[update.stage] ?? 0) + 1
        }
        const regionUpdates = batch.updates.filter((update) => update.type === 'regionReady')
        const detectorProgress = batch.updates.filter(
          (update) => update.type === 'progress' && update.stage === 'detecting',
        )
        const repairStateCounts = {}
        for (const update of regionUpdates) {
          const repairState = update.region.hsk.repairState
          repairStateCounts[repairState] = (repairStateCounts[repairState] ?? 0) + 1
        }
        jobsEvidence.push({
          jobId: job.jobId,
          pageIndex: job.pageIndex,
          sourceSha256: job.sourceSha256,
          sourceWidth: job.sourceWidth,
          sourceHeight: job.sourceHeight,
          route: `/jobs/${job.jobId}/updates`,
          updatesHttpStatus: updateFetch.response.status,
          updatesDurationMs: updateFetch.durationMs,
          nextSequence: batch.nextSequence,
          terminal,
          stageCounts,
          detectorTilesProcessed: Math.max(
            0,
            ...detectorProgress.map((update) => update.current ?? 0),
          ),
          detectorTilesTotal: Math.max(0, ...detectorProgress.map((update) => update.total ?? 0)),
          acceptedRegionCount: regionUpdates.length,
          strictHskValidCount: regionUpdates.filter((update) => update.region.hsk.strictlyValid)
            .length,
          repairStateCounts,
          updates: batch.updates,
          patches,
        })
      }
      return {
        session: {
          buildFingerprint: session.buildFingerprint,
          engineVersion: session.engineVersion,
          port: session.port,
          sessionExpiresAtUnixMs: session.sessionExpiresAtUnixMs,
          capabilities: session.capabilities,
          tokenRedacted: true,
        },
        health: {
          route: '/health',
          httpStatus: healthFetch.response.status,
          durationMs: healthFetch.durationMs,
          body: health,
        },
        resourceIdentityEvidence: {
          comparison: 'JSON-deep-exact ordered projection from committed model manifest',
          expected: expectedIdentities,
          actual: actualIdentities ?? [],
          gates: [
            {
              id: 'runtime-model-resource-identities',
              status: exactResourceIdentities ? 'pass' : 'fail',
              ...(exactResourceIdentities
                ? {}
                : {
                    reason:
                      'GET /health.resourceIdentities is missing or differs from the committed sorted detector/OCR resource identity array.',
                  }),
            },
          ],
        },
        setup: {
          route: '/setup',
          httpStatus: setupFetch.response.status,
          durationMs: setupFetch.durationMs,
          body: setup,
          downloadsInvoked: false,
        },
        actualExtensionRouteContract: {
          health: 'GET /health',
          setup: 'GET /setup',
          createJob: 'POST /jobs',
          viewport: 'PUT /jobs/{job_id}/viewport',
          updates: 'GET /jobs/{job_id}/updates',
          cancel: 'DELETE /jobs/{job_id}',
          patch: 'GET /blobs/{patch_id}',
        },
        jobs: jobsEvidence,
      }
    },
    {
      jobs: records,
      sessionKey: SESSION_STORAGE_KEY,
      buildFingerprint: BUILD_FINGERPRINT,
      requireTerminal: terminalRequired,
      expectedIdentities: expectedResourceIdentities,
    },
  )
}

function pointCoordinates(point) {
  const x = Array.isArray(point) ? point[0] : point?.x
  const y = Array.isArray(point) ? point[1] : point?.y
  return [Number(x), Number(y)]
}

function bounds(points) {
  if (!Array.isArray(points) || points.length < 4) return undefined
  const coordinates = points.map(pointCoordinates)
  const xs = coordinates.map(([x]) => x)
  const ys = coordinates.map(([, y]) => y)
  if ([...xs, ...ys].some((value) => !Number.isFinite(value))) return undefined
  return {
    left: Math.min(...xs),
    top: Math.min(...ys),
    right: Math.max(...xs),
    bottom: Math.max(...ys),
  }
}

function rectangleIou(leftPoints, rightPoints) {
  const left = bounds(leftPoints)
  const right = bounds(rightPoints)
  if (!left || !right) return 0
  const intersectionWidth = Math.max(
    0,
    Math.min(left.right, right.right) - Math.max(left.left, right.left),
  )
  const intersectionHeight = Math.max(
    0,
    Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top),
  )
  const intersection = intersectionWidth * intersectionHeight
  const leftArea = Math.max(0, left.right - left.left) * Math.max(0, left.bottom - left.top)
  const rightArea = Math.max(0, right.right - right.left) * Math.max(0, right.bottom - right.top)
  const union = leftArea + rightArea - intersection
  return union > 0 ? intersection / union : 0
}

function rectangleOverlapOverSmaller(leftPoints, rightPoints) {
  const left = bounds(leftPoints)
  const right = bounds(rightPoints)
  if (!left || !right) return 0
  const intersectionWidth = Math.max(
    0,
    Math.min(left.right, right.right) - Math.max(left.left, right.left),
  )
  const intersectionHeight = Math.max(
    0,
    Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top),
  )
  const intersection = intersectionWidth * intersectionHeight
  const leftArea = Math.max(0, left.right - left.left) * Math.max(0, left.bottom - left.top)
  const rightArea = Math.max(0, right.right - right.left) * Math.max(0, right.bottom - right.top)
  const smaller = Math.min(leftArea, rightArea)
  return smaller > 0 ? intersection / smaller : 0
}

function canonicalOcrText(value) {
  return String(value).normalize('NFKC').toLocaleLowerCase('en').replace(/\s+/gu, ' ').trim()
}

function isMainlandMandarinLanguage(value) {
  const language = String(value).trim().replaceAll('_', '-').toLowerCase()
  try {
    const locale = new Intl.Locale(language).maximize()
    return (
      (locale.language === 'zh' || locale.language === 'cmn') &&
      locale.script === 'Hans' &&
      locale.region === 'CN'
    )
  } catch {
    return /^(?:zh|cmn)(?:-(?:hans-)?cn|-cn-hans)?$/u.test(language)
  }
}

function editDistance(left, right) {
  const a = [...left]
  const b = [...right]
  let previous = Array.from({ length: b.length + 1 }, (_, index) => index)
  for (let i = 0; i < a.length; i += 1) {
    const current = [i + 1]
    for (let j = 0; j < b.length; j += 1) {
      current.push(
        Math.min(current[j] + 1, previous[j + 1] + 1, previous[j] + Number(a[i] !== b[j])),
      )
    }
    previous = current
  }
  return previous[b.length]
}

function finalRegions(job) {
  const regions = new Map()
  for (const update of job.updates) {
    if (update.type === 'regionReady') regions.set(update.region.id, structuredClone(update.region))
    if (update.type === 'regionRefined' && regions.has(update.regionId)) {
      const previous = regions.get(update.regionId)
      regions.set(update.regionId, {
        ...previous,
        displayedChinese: update.displayedChinese,
        pinyin: update.pinyin,
        hsk: update.hsk,
      })
    }
  }
  return [...regions.values()]
}

function isEnglishTranslationTarget(region) {
  return region.translationTarget !== false
}

function maximumRatioGate(id, actual, limit, numerator, denominator, definition) {
  const gate = measuredGate(id, actual, limit, 'ratio')
  return {
    ...gate,
    numerator,
    denominator,
    definition,
  }
}

export function buildQualityEvidence(routes, goldPages) {
  const pages = []
  const allMatches = []
  const allComponents = []
  const unmatchedAccepted = []
  const missingTargets = []
  let totalAccepted = 0
  let totalReviewedRegions = 0
  let totalDetectedBubbleGold = 0
  let totalNarrationRegions = 0
  let totalTranslationTargets = 0
  let totalExpectedUntouchedExclusions = 0
  let totalUntouchedExclusions = 0
  let totalModifiedExclusions = 0
  let matchedCharacterErrors = 0
  let matchedReferenceCharacters = 0
  let coveredTranslationTargets = 0
  for (const goldPage of goldPages) {
    const job = routes.jobs.find((candidate) => candidate.pageIndex === goldPage.order - 1)
    const accepted = job ? finalRegions(job) : []
    const reviewedRegions = goldPage.regions
    const detectedBubbleGold = reviewedRegions.filter((region) =>
      ['dialogue', 'thought'].includes(region.kind),
    )
    const narrationGold = reviewedRegions.filter((region) => region.kind === 'narration')
    const translationTargets = reviewedRegions.filter(isEnglishTranslationTarget)
    const untouchedExclusionGold = reviewedRegions.filter(
      (region) => !isEnglishTranslationTarget(region),
    )
    totalAccepted += accepted.length
    totalReviewedRegions += reviewedRegions.length
    totalDetectedBubbleGold += detectedBubbleGold.length
    totalNarrationRegions += narrationGold.length
    totalTranslationTargets += translationTargets.length
    totalExpectedUntouchedExclusions += untouchedExclusionGold.length
    const edges = []
    for (let observedIndex = 0; observedIndex < accepted.length; observedIndex += 1) {
      for (let expectedIndex = 0; expectedIndex < translationTargets.length; expectedIndex += 1) {
        const observed = accepted[observedIndex]
        const expected = translationTargets[expectedIndex]
        const overlap = rectangleOverlapOverSmaller(observed.textPolygon, expected.textPolygon)
        if (overlap >= STORY_REGION_MINIMUM_OVERLAP) {
          edges.push({ observed, observedIndex, expected, expectedIndex, overlap })
        }
      }
    }
    const parent = Array.from(
      { length: accepted.length + translationTargets.length },
      (_, index) => index,
    )
    const find = (index) => {
      let root = index
      while (parent[root] !== root) root = parent[root]
      while (parent[index] !== index) {
        const next = parent[index]
        parent[index] = root
        index = next
      }
      return root
    }
    const union = (left, right) => {
      const leftRoot = find(left)
      const rightRoot = find(right)
      if (leftRoot !== rightRoot) parent[rightRoot] = leftRoot
    }
    for (const edge of edges) {
      union(edge.expectedIndex, translationTargets.length + edge.observedIndex)
    }
    const componentByRoot = new Map()
    for (const edge of edges) {
      const root = find(edge.expectedIndex)
      if (!componentByRoot.has(root)) {
        componentByRoot.set(root, {
          expectedIndices: new Set(),
          observedIndices: new Set(),
          maximumOverlap: 0,
        })
      }
      const component = componentByRoot.get(root)
      component.expectedIndices.add(edge.expectedIndex)
      component.observedIndices.add(edge.observedIndex)
      component.maximumOverlap = Math.max(component.maximumOverlap, edge.overlap)
    }
    const readingOrder = (left, right) => {
      const leftBounds = bounds(left.textPolygon)
      const rightBounds = bounds(right.textPolygon)
      return (
        (leftBounds?.top ?? 0) - (rightBounds?.top ?? 0) ||
        (leftBounds?.left ?? 0) - (rightBounds?.left ?? 0) ||
        String(left.id).localeCompare(String(right.id), 'en')
      )
    }
    const components = [...componentByRoot.values()].map((component) => {
      const expected = [...component.expectedIndices]
        .map((index) => translationTargets[index])
        .sort(readingOrder)
      const observed = [...component.observedIndices]
        .map((index) => accepted[index])
        .sort(readingOrder)
      const expectedText = expected.map((region) => canonicalOcrText(region.sourceEnglish)).join(' ')
      const observedText = observed.map((region) => canonicalOcrText(region.sourceEnglish)).join(' ')
      const characterErrors = editDistance(expectedText, observedText)
      const referenceCharacters = [...expectedText].length
      matchedCharacterErrors += characterErrors
      matchedReferenceCharacters += referenceCharacters
      return {
        page: goldPage.order,
        expectedRegionIds: expected.map((region) => region.id),
        observedRegionIds: observed.map((region) => region.id),
        expectedSourceEnglish: expected.map((region) => region.sourceEnglish),
        observedSourceEnglish: observed.map((region) => region.sourceEnglish),
        maximumOverlapOverSmaller: component.maximumOverlap,
        characterErrors,
        referenceCharacters,
      }
    })
    allComponents.push(...components)
    const coveredExpectedIds = new Set(
      components.flatMap((component) => component.expectedRegionIds),
    )
    const coveredObservedIds = new Set(
      components.flatMap((component) => component.observedRegionIds),
    )
    coveredTranslationTargets += coveredExpectedIds.size

    const candidates = [...edges].sort(
      (left, right) =>
        right.overlap - left.overlap ||
        String(left.observed.id).localeCompare(String(right.observed.id), 'en') ||
        String(left.expected.id).localeCompare(String(right.expected.id), 'en'),
    )
    const observedIds = new Set()
    const expectedIds = new Set()
    const matches = []
    for (const candidate of candidates) {
      if (
        observedIds.has(candidate.observed.id) ||
        expectedIds.has(candidate.expected.id)
      ) {
        continue
      }
      observedIds.add(candidate.observed.id)
      expectedIds.add(candidate.expected.id)
      const match = {
        expectedRegionId: candidate.expected.id,
        observedRegionId: candidate.observed.id,
        textRectangleOverlapOverSmaller: candidate.overlap,
        expectedSourceEnglish: candidate.expected.sourceEnglish,
        observedSourceEnglish: candidate.observed.sourceEnglish,
        pinyin: candidate.observed.pinyin,
      }
      matches.push(match)
      allMatches.push({ page: goldPage.order, ...match })
    }
    const pageMissingTargets = translationTargets
      .filter((expected) => !coveredExpectedIds.has(expected.id))
      .map((expected) => {
        const evidence = {
          page: goldPage.order,
          expectedRegionId: expected.id,
          expectedSourceEnglish: expected.sourceEnglish,
          semantics: 'No accepted story region overlaps this target; charged to publication recall, not OCR CER.',
        }
        missingTargets.push(evidence)
        return evidence
      })
    const pageUnmatchedAccepted = accepted
      .filter((observed) => !coveredObservedIds.has(observed.id))
      .map((observed) => {
        const evidence = {
          page: goldPage.order,
          observedRegionId: observed.id,
          observedSourceEnglish: observed.sourceEnglish,
          semantics: 'No committed English story target overlaps this accepted translation.',
        }
        unmatchedAccepted.push(evidence)
        return evidence
      })
    const modifiedExclusionRegionIds = untouchedExclusionGold
      .filter((expected) =>
        accepted.some(
          (observed) =>
            rectangleIou(observed.textPolygon, expected.textPolygon) >= REGION_MATCH_MINIMUM_IOU,
        ),
      )
      .map((region) => region.id)
    const modifiedExclusionIds = new Set(modifiedExclusionRegionIds)
    const untouchedExclusionRegionIds = untouchedExclusionGold
      .filter((region) => !modifiedExclusionIds.has(region.id))
      .map((region) => region.id)
    const untouchedExclusionCount = untouchedExclusionRegionIds.length
    totalUntouchedExclusions += untouchedExclusionCount
    totalModifiedExclusions += modifiedExclusionRegionIds.length
    pages.push({
      page: goldPage.order,
      expectedRegionCount: reviewedRegions.length,
      expectedDetectorGoldBubbleCount: detectedBubbleGold.length,
      expectedNarrationRegionCount: narrationGold.length,
      expectedEnglishTranslationTargetCount: translationTargets.length,
      expectedUntouchedExclusionCount: untouchedExclusionGold.length,
      acceptedTranslationCount: accepted.length,
      matchedEnglishTargetCount: coveredExpectedIds.size,
      missingEnglishTargetCount: pageMissingTargets.length,
      unmatchedAcceptedTranslationCount: pageUnmatchedAccepted.length,
      untouchedExclusionCount,
      untouchedExclusionRegionIds,
      modifiedExclusionCount: modifiedExclusionRegionIds.length,
      modifiedExclusionRegionIds,
      matches,
      components,
      missingTargets: pageMissingTargets,
      unmatchedAccepted: pageUnmatchedAccepted,
    })
  }
  const ocrCharacterErrors = matchedCharacterErrors
  const ocrReferenceCharacters = matchedReferenceCharacters
  const ocrCer = ocrReferenceCharacters > 0 ? ocrCharacterErrors / ocrReferenceCharacters : 1
  const storyRegionRecall =
    totalTranslationTargets > 0 ? coveredTranslationTargets / totalTranslationTargets : 0
  const falseTranslationNumerator = unmatchedAccepted.length
  const falseTranslationDenominator = totalAccepted
  const falseTranslationRate =
    falseTranslationDenominator > 0 ? falseTranslationNumerator / falseTranslationDenominator : 0
  const pinyinComplete = pages.every((page) =>
    page.components
      .flatMap((component) => component.observedRegionIds)
      .every((observedId) => {
        const job = routes.jobs.find((candidate) => candidate.pageIndex === page.page - 1)
        const observed = job ? finalRegions(job).find((region) => region.id === observedId) : undefined
        return String(observed?.pinyin ?? '').trim().length > 0
      }),
  )
  const exclusionsUntouched =
    totalUntouchedExclusions === totalExpectedUntouchedExclusions && totalModifiedExclusions === 0
  const gates = [
    minimumRatioGate(
      'story-region-publication-recall',
      storyRegionRecall,
      STORY_REGION_MINIMUM_RECALL,
      coveredTranslationTargets,
      totalTranslationTargets,
      'Committed English story targets with any accepted overlapping region, allowing detector/OCR splits and merges.',
    ),
    maximumRatioGate(
      'english-ocr-cer',
      ocrCer,
      MAXIMUM_ENGLISH_OCR_CER,
      ocrCharacterErrors,
      ocrReferenceCharacters,
      'Levenshtein character errors within spatially connected matched components after Unicode NFKC, English lowercase, and whitespace collapse. Detector/publication misses and false regions are measured separately.',
    ),
    maximumRatioGate(
      'non-english-non-dialogue-false-translation-rate',
      falseTranslationRate,
      MAXIMUM_FALSE_TRANSLATION_RATE,
      falseTranslationNumerator,
      falseTranslationDenominator,
      `Accepted regionReady translations with no spatial overlap to any of the ${totalTranslationTargets} committed English translation targets divided by all accepted regionReady translations.`,
    ),
    booleanGate(
      'pinyin-present-for-every-translation-target',
      pinyinComplete,
      'At least one matched committed English translation target has no displayed pinyin.',
    ),
    booleanGate(
      'ambiguous-punctuation-left-untouched',
      exclusionsUntouched,
      'At least one language-ambiguous punctuation-only exclusion emitted a geometrically matching regionReady update.',
      {
        expectedUntouchedExclusions: totalExpectedUntouchedExclusions,
        untouchedExclusions: totalUntouchedExclusions,
        modifiedExclusions: totalModifiedExclusions,
      },
    ),
  ]
  return {
    matcher: {
      geometry: 'axis-aligned bounding rectangle of committed and observed textPolygon',
      minimumOverlapOverSmaller: STORY_REGION_MINIMUM_OVERLAP,
      assignment:
        'bipartite spatial connected components for OCR/recall; descending-overlap one-to-one representatives only for per-patch audit',
      acceptedOutputSource: 'final regionReady/regionRefined browser job updates',
      detectorOutputUsed: false,
    },
    metricDefinitions: {
      englishOcrCer: {
        numerator:
          'Levenshtein errors after concatenating expected and observed text in reading order inside each spatially connected component',
        denominator:
          'normalized committed reference characters only inside spatially matched components',
        maximum: MAXIMUM_ENGLISH_OCR_CER,
      },
      storyRegionRecall: {
        numerator: 'committed English story targets covered by at least one accepted region',
        denominator: `all ${totalTranslationTargets} committed English story targets`,
        minimum: STORY_REGION_MINIMUM_RECALL,
      },
      falseTranslationRate: {
        numerator: 'accepted regionReady translations with no spatial edge to English target gold',
        denominator: 'all accepted regionReady translations',
        maximum: MAXIMUM_FALSE_TRANSLATION_RATE,
      },
    },
    pages,
    totals: {
      expectedRegionCount: totalReviewedRegions,
      detectorGoldBubbleCount: totalDetectedBubbleGold,
      expectedNarrationRegionCount: totalNarrationRegions,
      expectedEnglishTranslationTargetCount: totalTranslationTargets,
      expectedUntouchedExclusionCount: totalExpectedUntouchedExclusions,
      acceptedTranslationCount: totalAccepted,
      matchedEnglishTargetCount: coveredTranslationTargets,
      missingEnglishTargetCount: missingTargets.length,
      unmatchedAcceptedTranslationCount: unmatchedAccepted.length,
      untouchedExclusions: totalUntouchedExclusions,
      modifiedExclusions: totalModifiedExclusions,
      ocrMatchedCharacterErrors: matchedCharacterErrors,
      ocrMatchedReferenceCharacters: matchedReferenceCharacters,
      storyRegionRecall,
      ocrMissingCharacterErrors: 0,
      ocrMissingReferenceCharacters: 0,
      ocrUnmatchedInsertionErrors: 0,
      ocrCharacterErrorNumerator: ocrCharacterErrors,
      ocrReferenceCharacterDenominator: ocrReferenceCharacters,
      englishOcrCer: ocrCer,
      falseTranslationNumerator,
      falseTranslationDenominator,
      falseTranslationRate,
    },
    allMatches,
    components: allComponents,
    missingTargets,
    unmatchedAccepted,
    gates,
  }
}

function polygonPixelEnvelope(polygon, sourceWidth, sourceHeight, label) {
  const normalizedBounds = bounds(polygon)
  if (!normalizedBounds) fail(`${label} is not a valid normalized polygon.`)
  const coordinates = polygon.map(pointCoordinates)
  if (
    coordinates.some(
      ([x, y]) =>
        !(
          Math.abs(x - normalizedBounds.left) <= 0.000001 ||
          Math.abs(x - normalizedBounds.right) <= 0.000001
        ) ||
        !(
          Math.abs(y - normalizedBounds.top) <= 0.000001 ||
          Math.abs(y - normalizedBounds.bottom) <= 0.000001
        ),
    )
  ) {
    fail(`${label} is not the committed axis-aligned pixel-envelope geometry.`)
  }
  const x0 = Math.max(0, Math.floor(normalizedBounds.left * sourceWidth))
  const y0 = Math.max(0, Math.floor(normalizedBounds.top * sourceHeight))
  const x1 = Math.min(sourceWidth, Math.ceil(normalizedBounds.right * sourceWidth))
  const y1 = Math.min(sourceHeight, Math.ceil(normalizedBounds.bottom * sourceHeight))
  if (x1 <= x0 || y1 <= y0) fail(`${label} has an empty pixel envelope.`)
  return {
    x0,
    y0,
    x1,
    y1,
    pixels: (x1 - x0) * (y1 - y0),
  }
}

function validatedAlphaRows(patch) {
  if (!Array.isArray(patch?.alphaRows)) return { rows: [], pixels: 0, valid: false }
  const rows = []
  let pixels = 0
  let previousY = -1
  for (const row of patch.alphaRows) {
    if (
      !Number.isInteger(row?.y) ||
      row.y < 0 ||
      row.y >= patch.height ||
      row.y <= previousY ||
      !Array.isArray(row.runs)
    ) {
      return { rows: [], pixels: 0, valid: false }
    }
    previousY = row.y
    let previousEnd = -1
    const runs = []
    for (const run of row.runs) {
      const start = run?.[0]
      const end = run?.[1]
      if (
        !Number.isInteger(start) ||
        !Number.isInteger(end) ||
        start < 0 ||
        end <= start ||
        end > patch.width ||
        start < previousEnd
      ) {
        return { rows: [], pixels: 0, valid: false }
      }
      previousEnd = end
      pixels += end - start
      runs.push([start, end])
    }
    if (runs.length < 1) return { rows: [], pixels: 0, valid: false }
    rows.push({ y: row.y, runs })
  }
  return {
    rows,
    pixels,
    valid:
      Number.isInteger(patch.alphaNonZeroPixelCount) && patch.alphaNonZeroPixelCount === pixels,
  }
}

function coveredAlphaPixels(alphaRows, patchOrigin, envelope) {
  let covered = 0
  const localX0 = envelope.x0 - patchOrigin.x
  const localX1 = envelope.x1 - patchOrigin.x
  for (const row of alphaRows) {
    const globalY = patchOrigin.y + row.y
    if (globalY < envelope.y0 || globalY >= envelope.y1) continue
    for (const [start, end] of row.runs) {
      covered += Math.max(0, Math.min(end, localX1) - Math.max(start, localX0))
    }
  }
  return covered
}

function coveredEnvelopePixels(patches, envelope) {
  let covered = 0
  for (let globalY = envelope.y0; globalY < envelope.y1; globalY += 1) {
    const intervals = []
    for (const patch of patches) {
      const row = patch.alpha.rows.find(
        (candidate) => patch.patchOrigin.y + candidate.y === globalY,
      )
      for (const [alphaStart, alphaEnd] of row?.runs ?? []) {
        const start = Math.max(envelope.x0, patch.patchOrigin.x + alphaStart)
        const end = Math.min(envelope.x1, patch.patchOrigin.x + alphaEnd)
        if (end > start) intervals.push([start, end])
      }
    }
    intervals.sort((left, right) => left[0] - right[0] || left[1] - right[1])
    let mergedEnd = -1
    for (const [start, end] of intervals) {
      if (end <= mergedEnd) continue
      covered += end - Math.max(start, mergedEnd)
      mergedEnd = end
    }
  }
  return covered
}

function coveredGlyphMaskPixels(patches, glyph) {
  let covered = 0
  for (const row of glyph.rows) {
    for (const [glyphStart, glyphEnd] of row.runs) {
      const globalStart = glyph.originX + glyphStart
      const globalEnd = glyph.originX + glyphEnd
      const intervals = []
      for (const patch of patches) {
        const alphaRow = patch.alpha.rows.find(
          (candidate) => patch.patchOrigin.y + candidate.y === glyph.originY + row.y,
        )
        for (const [alphaStart, alphaEnd] of alphaRow?.runs ?? []) {
          const start = Math.max(globalStart, patch.patchOrigin.x + alphaStart)
          const end = Math.min(globalEnd, patch.patchOrigin.x + alphaEnd)
          if (end > start) intervals.push([start, end])
        }
      }
      intervals.sort((left, right) => left[0] - right[0] || left[1] - right[1])
      let mergedEnd = -1
      for (const [start, end] of intervals) {
        if (end <= mergedEnd) continue
        covered += end - Math.max(start, mergedEnd)
        mergedEnd = end
      }
    }
  }
  return covered
}

export function buildPatchQualityEvidence(routes, goldPages, components, sourceGlyphs) {
  const matches = components.flatMap((component) =>
    component.expectedRegionIds.map((expectedRegionId) => ({
      page: component.page,
      expectedRegionId,
      observedRegionIds: component.observedRegionIds,
    })),
  )
  const regions = []
  let eraseMaskPixelDenominator = 0
  let coveredEraseMaskPixelNumerator = 0
  let glyphPixelDenominator = 0
  let coveredGlyphPixelNumerator = 0
  let alphaPixelDenominator = 0
  let alphaOutsideAcceptedRegionPixels = 0
  let dimensionMatchCount = 0
  let decodedAlphaEvidenceCount = 0
  const auditedPatchIds = new Set()
  const auditedGlyphIds = new Set()
  for (const match of matches) {
    const page = goldPages.find((candidate) => candidate.order === match.page)
    const gold = page?.regions.find((candidate) => candidate.id === match.expectedRegionId)
    const job = routes.jobs.find((candidate) => candidate.pageIndex === match.page - 1)
    const glyphs = match.observedRegionIds.map((observedRegionId) =>
      sourceGlyphs.find(
        (candidate) =>
          candidate.page === match.page && candidate.id === observedRegionId,
      ),
    )
    if (
      !gold ||
      !job ||
      glyphs.some(
        (glyph) => !glyph || !Number.isInteger(glyph.pixels) || glyph.pixels < 1,
      )
    ) {
      fail(`Patch audit cannot resolve matched region ${match.expectedRegionId}.`)
    }
    const patches = job.patches.filter((candidate) =>
      match.observedRegionIds.includes(candidate.regionId),
    )
    if (patches.length !== match.observedRegionIds.length) {
      fail(`Patch audit cannot resolve every observed patch for ${match.expectedRegionId}.`)
    }
    const sourceWidth = Number(job.sourceWidth)
    const sourceHeight = Number(job.sourceHeight)
    if (
      !Number.isInteger(sourceWidth) ||
      sourceWidth < 1 ||
      !Number.isInteger(sourceHeight) ||
      sourceHeight < 1
    ) {
      fail(`Patch audit has invalid source dimensions for page ${match.page}.`)
    }
    const eraseEnvelope = polygonPixelEnvelope(
      gold.eraseMask?.polygon,
      sourceWidth,
      sourceHeight,
      `${gold.id}.eraseMask.polygon`,
    )
    eraseMaskPixelDenominator += eraseEnvelope.pixels
    const patchEvidence = patches.map((patch) => {
      const rect = patch.rect
      const rectX = Number(rect?.x)
      const rectY = Number(rect?.y)
      const rectWidth = Number(rect?.width)
      const rectHeight = Number(rect?.height)
      const patchOrigin = {
        x: Number.isFinite(rectX) ? Math.round(rectX * sourceWidth) : 0,
        y: Number.isFinite(rectY) ? Math.round(rectY * sourceHeight) : 0,
      }
      const expectedWidth = Number.isFinite(rectWidth)
        ? Math.max(1, Math.round(rectWidth * sourceWidth))
        : 0
      const expectedHeight = Number.isFinite(rectHeight)
        ? Math.max(1, Math.round(rectHeight * sourceHeight))
        : 0
      const dimensionsMatch =
        Number.isFinite(rectX) &&
        Number.isFinite(rectY) &&
        Number.isFinite(rectWidth) &&
        Number.isFinite(rectHeight) &&
        patch.width === expectedWidth &&
        patch.height === expectedHeight
      const alpha = validatedAlphaRows(patch)
      const acceptedRegionPolygon = patch.bubblePolygon ?? [
        { x: rectX, y: rectY },
        { x: rectX + rectWidth, y: rectY },
        { x: rectX + rectWidth, y: rectY + rectHeight },
        { x: rectX, y: rectY + rectHeight },
      ]
      const acceptedRegionEnvelope = polygonPixelEnvelope(
        acceptedRegionPolygon,
        sourceWidth,
        sourceHeight,
        `${patch.regionId}.acceptedRegionPolygon`,
      )
      const coveredAcceptedRegionPixels = alpha.valid
        ? coveredAlphaPixels(alpha.rows, patchOrigin, acceptedRegionEnvelope)
        : 0
      const outsideAcceptedRegionPixels = alpha.valid
        ? Math.max(0, alpha.pixels - coveredAcceptedRegionPixels)
        : 0
      if (!auditedPatchIds.has(patch.patchId)) {
        auditedPatchIds.add(patch.patchId)
        if (dimensionsMatch) dimensionMatchCount += 1
        if (alpha.valid) decodedAlphaEvidenceCount += 1
        alphaPixelDenominator += alpha.pixels
        alphaOutsideAcceptedRegionPixels += outsideAcceptedRegionPixels
      }
      return {
        patch,
        patchOrigin,
        expectedWidth,
        expectedHeight,
        dimensionsMatch,
        alpha,
        outsideAcceptedRegionPixels,
      }
    })
    const glyphEvidence = glyphs.map((glyph) => {
      const evidence = patchEvidence.find(
        (candidate) => candidate.patch.regionId === glyph.id,
      )
      if (!evidence) {
        fail(`Patch audit cannot resolve source glyph evidence ${glyph.id}.`)
      }
      return {
        glyph,
        coveredPixels: coveredGlyphMaskPixels([evidence], glyph),
      }
    })
    const regionGlyphPixels = glyphEvidence.reduce(
      (sum, evidence) => sum + evidence.glyph.pixels,
      0,
    )
    const coveredGlyphPixels = glyphEvidence.reduce(
      (sum, evidence) => sum + evidence.coveredPixels,
      0,
    )
    for (const evidence of glyphEvidence) {
      if (auditedGlyphIds.has(evidence.glyph.id)) continue
      auditedGlyphIds.add(evidence.glyph.id)
      glyphPixelDenominator += evidence.glyph.pixels
      coveredGlyphPixelNumerator += evidence.coveredPixels
    }
    const coveredErasePixels = coveredEnvelopePixels(patchEvidence, eraseEnvelope)
    coveredEraseMaskPixelNumerator += coveredErasePixels
    regions.push({
      page: match.page,
      expectedRegionId: match.expectedRegionId,
      observedRegionIds: match.observedRegionIds,
      sourceSha256: job.sourceSha256,
      patches: patchEvidence.map((evidence) => ({
        patchId: evidence.patch.patchId,
        observedRegionId: evidence.patch.regionId,
        pngSha256: evidence.patch.sha256,
        decodedRgbaSha256: evidence.patch.decodedRgbaSha256,
        patchRect: evidence.patch.rect,
        decodedWidth: evidence.patch.width,
        decodedHeight: evidence.patch.height,
        expectedWidth: evidence.expectedWidth,
        expectedHeight: evidence.expectedHeight,
        dimensionsMatch: evidence.dimensionsMatch,
        alphaEvidenceValid: evidence.alpha.valid,
        alphaPixels: evidence.alpha.pixels,
        alphaOutsideAcceptedRegionPixels: evidence.outsideAcceptedRegionPixels,
      })),
      eraseMaskPixelDenominator: eraseEnvelope.pixels,
      coveredEraseMaskPixelNumerator: coveredErasePixels,
      eraseMaskCoverage: coveredErasePixels / eraseEnvelope.pixels,
      glyphPixelDenominator: regionGlyphPixels,
      coveredGlyphPixelNumerator: coveredGlyphPixels,
      glyphCoverage: coveredGlyphPixels / regionGlyphPixels,
    })
  }
  const matchedRegionDenominator = matches.length
  const eraseMaskCoverage =
    eraseMaskPixelDenominator > 0 ? coveredEraseMaskPixelNumerator / eraseMaskPixelDenominator : 0
  const glyphCoverage =
    glyphPixelDenominator > 0 ? coveredGlyphPixelNumerator / glyphPixelDenominator : 0
  return {
    provenance: {
      patchBytes:
        'authenticated packaged-extension GET /blobs/{patch_id} response with recorded PNG SHA-256',
      decodedPixels:
        'Firefox HTMLImageElement.decode followed by CanvasRenderingContext2D.getImageData',
      gold:
        'hash-pinned source pixels localized by each gold-matched accepted text proposal; committed eraseMask geometry remains the maximum-change diagnostic',
      containment:
        'Every non-zero-alpha decoded PNG pixel is conservatively treated as a potential changed composite pixel.',
      glyphMask:
        'Foreground pixels are independently derived inside each gold-matched accepted source-text proposal using grayscale Otsu and border-median polarity. Border-connected components, detached sub-25-pixel specks, and isolated shallow-wide artwork are rejected; when contour rejection would empty a tight text crop, the polarity-selected core is retained before the same component filters and one-pixel audit dilation.',
      eraseMask:
        'The committed eraseMask is a maximum allowed change area, not a rectangle production is required to fill.',
    },
    rasterization:
      'Pixel-envelope bounds floor each committed normalized polygon minimum and ceil its maximum in source pixels.',
    totals: {
      matchedRegionDenominator,
      patchRegionCount: auditedPatchIds.size,
      decodedAlphaEvidenceCount,
      dimensionMatchCount,
      eraseMaskPixelDenominator,
      coveredEraseMaskPixelNumerator,
      eraseMaskCoverage,
      glyphPixelDenominator,
      coveredGlyphPixelNumerator,
      glyphCoverage,
      alphaPixelDenominator,
      alphaOutsideAcceptedRegionPixels,
    },
    regions,
    gates: [
      exactGate(
        'matched-patch-decoded-alpha-evidence',
        decodedAlphaEvidenceCount,
        auditedPatchIds.size,
        { denominator: auditedPatchIds.size },
      ),
      exactGate('matched-patch-dimensions', dimensionMatchCount, auditedPatchIds.size, {
        denominator: auditedPatchIds.size,
      }),
      exactGate(
        'matched-patch-alpha-covers-independent-source-glyph-mask',
        coveredGlyphPixelNumerator,
        glyphPixelDenominator,
        {
          numerator: coveredGlyphPixelNumerator,
          denominator: glyphPixelDenominator,
        },
      ),
      exactGate('patch-changes-outside-runtime-accepted-region', alphaOutsideAcceptedRegionPixels, 0, {
        numerator: alphaOutsideAcceptedRegionPixels,
        denominator: alphaPixelDenominator,
        conservativeProof:
          'Zero non-zero-alpha pixels outside the runtime bubble or the bounded free-text patch proves zero possible composite changes there.',
      }),
    ],
  }
}

export function buildPatchCommitOrderingEvidence(routes, dom, actionEpochMs = 0) {
  const expected = routes.jobs.flatMap((job) =>
    job.updates
      .filter((update) => update.type === 'regionReady')
      .map((update) => ({
        page: job.pageIndex + 1,
        patchId: update.region.patch.blobId,
        regionId: update.region.id,
      })),
  )
  const patchEvents = new Map()
  const textEvents = new Map()
  for (const event of dom.events ?? []) {
    if (!finiteNonNegative(event.epochMs) || event.epochMs < actionEpochMs) continue
    if (
      event.type === 'patchDomCommitted' &&
      (!patchEvents.has(event.patchId) || event.index < patchEvents.get(event.patchId).index)
    ) {
      patchEvents.set(event.patchId, event)
    }
    if (
      event.type === 'selectableTextDomCommitted' &&
      (!textEvents.has(event.regionId) || event.index < textEvents.get(event.regionId).index)
    ) {
      textEvents.set(event.regionId, event)
    }
  }
  const pairs = expected.map((item) => {
    const patch = patchEvents.get(item.patchId)
    const text = textEvents.get(item.regionId)
    const passed = Boolean(
      patch?.decodedAndInstalled &&
      patch.complete &&
      patch.naturalWidth > 0 &&
      patch.naturalHeight > 0 &&
      text &&
      patch.index < text.index,
    )
    return {
      ...item,
      patchEventIndex: patch?.index ?? -1,
      chineseTextEventIndex: text?.index ?? -1,
      patchDecodedAndInstalled: patch?.decodedAndInstalled === true,
      passed,
    }
  })
  const expectedRegionIds = new Set(expected.map((item) => item.regionId))
  const unexpectedChineseCommitRegionIds = [...textEvents.keys()].filter(
    (regionId) => !expectedRegionIds.has(regionId),
  )
  const orderedPairNumerator = pairs.filter((pair) => pair.passed).length
  const orderedPairDenominator = pairs.length
  return {
    eventSource:
      'chapter MutationObserver installed before content:start; patch event records complete/natural dimensions at DOM installation',
    invariant:
      'For every accepted region, the corresponding decoded patch installation event index is strictly lower than the first selectable Chinese DOM event index.',
    orderedPairNumerator,
    orderedPairDenominator,
    unexpectedChineseCommitRegionIds,
    pairs,
    gates: [
      exactGate(
        'decoded-patch-installed-before-corresponding-chinese-dom',
        orderedPairNumerator,
        orderedPairDenominator,
        {
          numerator: orderedPairNumerator,
          denominator: orderedPairDenominator,
        },
      ),
      exactGate('unexpected-chinese-dom-commits', unexpectedChineseCommitRegionIds.length, 0, {
        unexpectedChineseCommitRegionIds,
      }),
    ],
  }
}

export function buildJobRequestEvidence(
  manifest,
  routes,
  jobRecords,
  hskLevel,
  actionIssuedAtEpochMs,
) {
  const rollingContext = []
  const pages = []
  const mismatches = []
  const pageIndexes = new Set()
  const pageSessionIds = new Set()
  const submittedRecords = [...jobRecords].sort(
    (left, right) =>
      left.submittedAtUnixMs - right.submittedAtUnixMs || left.pageIndex - right.pageIndex,
  )
  for (const record of submittedRecords) {
    const request = record.submittedRequest
    const image = manifest.images.find((candidate) => candidate.order - 1 === request.pageIndex)
    const job = routes.jobs.find((candidate) => candidate.pageIndex === request.pageIndex)
    const expectedContext = rollingContext.slice(-MAX_PRECEDING_CONTEXT)
    const expectedKeys = [
      'buildFingerprint',
      'clientImageId',
      'sourceSha256',
      'sourceMimeType',
      'naturalWidth',
      'naturalHeight',
      'pageSessionId',
      'pageIndex',
      'settings',
      'visibleRects',
      ...(expectedContext.length ? ['precedingContext'] : []),
    ].sort()
    const actualKeys = request && typeof request === 'object' ? Object.keys(request).sort() : []
    const visibleRectsValid =
      Array.isArray(request?.visibleRects) &&
      request.visibleRects.length <= 64 &&
      request.visibleRects.every(
        (rect) =>
          rect &&
          ['x', 'y', 'width', 'height'].every((key) => Number.isFinite(rect[key])) &&
          rect.x >= 0 &&
          rect.y >= 0 &&
          rect.width > 0 &&
          rect.height > 0 &&
          rect.x + rect.width <= 1 + 1e-6 &&
          rect.y + rect.height <= 1 + 1e-6,
      )
    const expectedSettings = {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel,
      readingDirection: 'auto',
      translateSoundEffects: false,
    }
    const checks = {
      knownPage: Boolean(image && job && record),
      uniquePage: !pageIndexes.has(request?.pageIndex),
      exactKeys: sameJson(actualKeys, expectedKeys),
      buildFingerprint: request?.buildFingerprint === BUILD_FINGERPRINT,
      clientImageId:
        typeof request?.pageSessionId === 'string' &&
        request.clientImageId ===
          `${request.pageSessionId}-${request.pageIndex}-${request.sourceSha256?.slice(0, 16)}`,
      sourceIdentity:
        request?.sourceSha256 === image?.sha256 &&
        record.sourceSha256 === image?.sha256 &&
        record.uploadedImageBytes === image?.bytes,
      sourceShape:
        request?.sourceMimeType === 'image/webp' &&
        request?.naturalWidth === image?.width &&
        request?.naturalHeight === image?.height,
      routeIdentity:
        request?.sourceSha256 === job?.sourceSha256 &&
        request?.sourceSha256 === record?.sourceSha256,
      settings: sameJson(request?.settings, expectedSettings),
      visibleRects: visibleRectsValid,
      precedingContext:
        expectedContext.length > 0
          ? sameJson(request?.precedingContext, expectedContext)
          : request?.precedingContext === undefined,
      causality:
        finiteNonNegative(record.submittedAtUnixMs) &&
        record.submittedAtUnixMs >= actionIssuedAtEpochMs &&
        finiteNonNegative(record?.createdAtUnixMs) &&
        record.submittedAtUnixMs <= record.createdAtUnixMs,
    }
    if (Number.isInteger(request?.pageIndex)) pageIndexes.add(request.pageIndex)
    if (typeof request?.pageSessionId === 'string') {
      pageSessionIds.add(request.pageSessionId)
    }
    const failedChecks = Object.entries(checks)
      .filter(([, passed]) => !passed)
      .map(([name]) => name)
    if (failedChecks.length) {
      mismatches.push(
        `page ${Number.isInteger(request?.pageIndex) ? request.pageIndex + 1 : '?'}: ${failedChecks.join(', ')}`,
      )
    }
    pages.push({
      page: Number.isInteger(request?.pageIndex) ? request.pageIndex + 1 : 0,
      submittedAtUnixMs: record.submittedAtUnixMs,
      acceptedAtUnixMs: record.createdAtUnixMs,
      uploadedImageBytes: record.uploadedImageBytes,
      uploadedImageSha256: record.sourceSha256,
      request,
      checks,
    })
    if (job) {
      for (const region of finalRegions(job).sort(
        (left, right) => left.readingOrder - right.readingOrder,
      )) {
        if (!region.sourceEnglish || !region.displayedChinese) continue
        rollingContext.push({
          sourceEnglish: region.sourceEnglish,
          chinese: region.displayedChinese,
        })
        if (rollingContext.length > MAX_PRECEDING_CONTEXT) rollingContext.shift()
      }
    }
  }
  const complete =
    submittedRecords.length === manifest.pageCount &&
    pageIndexes.size === manifest.pageCount &&
    pageSessionIds.size === 1 &&
    mismatches.length === 0
  return {
    captureMethod:
      'Packaged-extension ActiveJobRecord written only after POST /jobs returns; it retains the exact immutable submitted request, uploaded byte count, source SHA-256, pre-submit timestamp, and post-accept timestamp.',
    maximumPrecedingContext: MAX_PRECEDING_CONTEXT,
    capturedRequestCount: submittedRecords.length,
    uniquePageCount: pageIndexes.size,
    pageSessionCount: pageSessionIds.size,
    mismatches,
    pages,
    gates: [
      booleanGate(
        'exact-post-jobs-request',
        complete,
        'The captured packaged-extension POST /jobs requests did not exactly match the fixture identities, settings, bounded preceding context, or causal job records.',
      ),
    ],
  }
}

function timingEvidence(result, manifest, viewportPlan) {
  const action = result.action.issuedAtEpochMs
  const events = result.dom.events.filter((event) => event.epochMs >= action)
  const hud = events.find((event) => event.type === 'hudAcknowledged')
  const expectedViewportIds = new Set(viewportPlan.expectedVisibleRegionIds)
  const observedViewportIds = new Set(
    result.correctness.allMatches
      .filter(
        (match) =>
          match.page === viewportPlan.page && expectedViewportIds.has(match.expectedRegionId),
      )
      .map((match) => match.observedRegionId),
  )
  const visibleTextEvents = events
    .filter(
      (event) =>
        event.type === 'selectableTextDomCommitted' &&
        event.visible &&
        event.page === viewportPlan.page &&
        observedViewportIds.has(event.regionId),
    )
    .filter(
      (event, index, values) =>
        values.findIndex((candidate) => candidate.regionId === event.regionId) === index,
    )
  const visibleTarget = visibleTextEvents[viewportPlan.expectedVisibleRegionCount - 1]
  const firstVisible = visibleTextEvents[0]
  const longTerminals = result.jobMonitor.observations.filter((observation) => {
    const expected = manifest.images.find((image) => image.order - 1 === observation.pageIndex)
    return expected?.height >= LONG_IMAGE_MIN_HEIGHT_PX && observation.terminalType === 'complete'
  })
  const firstLongTerminal = longTerminals
    .filter((item) => finiteNonNegative(item.terminalObservedAtEpochMs))
    .sort((left, right) => left.terminalObservedAtEpochMs - right.terminalObservedAtEpochMs)[0]
  const allComplete = events.find((event) => event.type === 'hudComplete')
  return {
    scopeStart: {
      event: 'content:start browser.tabs.sendMessage invocation',
      issuedAtEpochMs: action,
    },
    includedProductPath: [
      'content source-byte acquisition',
      'source validation and SHA-256',
      'multipart upload and daemon acceptance',
      'daemon detector/OCR/translation/patch publication',
      'extension patch GET and validation',
      'PNG decode',
      'decoded patch DOM commit',
      'selectable-text DOM commit',
    ],
    hudAcknowledgementMs: hud?.epochMs - action,
    firstVisibleRegionMs: firstVisible?.epochMs - action,
    visibleRegionGroupMs: visibleTarget?.epochMs - action,
    visibleRegionGroupCount: viewportPlan.expectedVisibleRegionCount,
    expectedViewportRegionIds: viewportPlan.expectedVisibleRegionIds,
    observedViewportRegionIds: [...observedViewportIds],
    firstLongImageCompleteMs: firstLongTerminal?.terminalObservedAtEpochMs - action,
    firstLongImagePage: finiteNonNegative(firstLongTerminal?.pageIndex)
      ? firstLongTerminal.pageIndex + 1
      : undefined,
    allImagesCompleteMs: allComplete?.epochMs - action,
    jobCreatedAfterAcquisitionHashUpload: result.jobMonitor.observations.map((observation) => ({
      page: observation.pageIndex + 1,
      sourceSha256: observation.sourceSha256,
      daemonAcceptedAtEpochMs: observation.createdAtUnixMs,
      actionToDaemonAcceptedMs: observation.createdAtUnixMs - action,
    })),
  }
}

function performanceGates(kind, timing, exactCache) {
  const gates = [
    measuredGate(
      'hud-acknowledgement',
      timing.hudAcknowledgementMs,
      BENCHMARK_LIMITS.hudAcknowledgementMs,
    ),
  ]
  if (kind === 'installed-cold') {
    gates.push(
      measuredGate(
        'installed-cold-first-visible-bubble',
        timing.firstVisibleRegionMs,
        BENCHMARK_LIMITS.installedColdFirstVisibleBubbleMs,
      ),
      measuredGate(
        'installed-cold-first-long-image-complete',
        timing.firstLongImageCompleteMs,
        BENCHMARK_LIMITS.installedColdFirstLongImageCompleteMs,
      ),
      measuredGate(
        'installed-cold-all-images-complete',
        timing.allImagesCompleteMs,
        BENCHMARK_LIMITS.installedColdAllImagesCompleteMs,
      ),
    )
  } else if (kind === 'warm') {
    gates.push(
      measuredGate(
        'warm-first-visible-bubble',
        timing.firstVisibleRegionMs,
        BENCHMARK_LIMITS.firstVisibleRegionMs,
      ),
      measuredGate(
        'warm-visible-bubble-group',
        timing.visibleRegionGroupMs,
        BENCHMARK_LIMITS.visibleRegionGroupMs,
      ),
      measuredGate(
        'warm-first-long-image-complete',
        timing.firstLongImageCompleteMs,
        BENCHMARK_LIMITS.firstLongImageCompleteMs,
      ),
      measuredGate(
        'warm-all-images-complete',
        timing.allImagesCompleteMs,
        BENCHMARK_LIMITS.allImagesCompleteMs,
      ),
    )
  } else if (kind === 'cache-replay') {
    gates.push(
      booleanGate(
        'cache-replay-is-exact',
        exactCache,
        'Every cache-replay job must terminate with "Exact cached translation replayed" and emit no inference progress stages.',
      ),
      measuredGate(
        'exact-cached-first-viewport',
        timing.visibleRegionGroupMs,
        BENCHMARK_LIMITS.exactCachedFirstViewportMs,
      ),
    )
  }
  return gates
}

export function reconcileCompleteJobTerminals(jobMonitor, routes, dom) {
  const observations = [...jobMonitor.observations].sort(
    (left, right) =>
      left.createdAtUnixMs - right.createdAtUnixMs || left.pageIndex - right.pageIndex,
  )
  const hudCompletions = dom.events.filter((event) => event.type === 'hudComplete')
  for (let index = 0; index < observations.length; index += 1) {
    const observation = observations[index]
    const route = routes.jobs.find((job) => job.jobId === observation.jobId)
    const terminalType = route?.terminal?.type
    if (!terminalType) continue
    const nextJob = observations[index + 1]
    const causalHudComplete = hudCompletions.find(
      (event) => event.epochMs >= observation.createdAtUnixMs,
    )
    const observedAt =
      observation.terminalObservedAtEpochMs ??
      nextJob?.createdAtUnixMs ??
      causalHudComplete?.epochMs
    observation.terminalType = terminalType
    if (finiteNonNegative(observedAt)) {
      observation.terminalObservedAtEpochMs = observedAt
      observation.terminalEvidence =
        nextJob === undefined
          ? 'authoritative daemon terminal replay plus HUD completion'
          : 'authoritative daemon terminal replay bounded by next sequential daemon acceptance'
    }
  }
}

function validateCompleteRun(result, manifest) {
  if (result.extensionState.state !== 'complete') {
    fail(`Run ${result.runId} did not complete: ${JSON.stringify(result.extensionState)}.`)
  }
  if (result.jobRecords.length !== manifest.pageCount) {
    fail(
      `Run ${result.runId} produced ${result.jobRecords.length} jobs, expected ${manifest.pageCount}.`,
    )
  }
  if (
    (result.kind === 'warmup' || result.kind === 'warm') &&
    (!result.resultCacheReset ||
      result.resultCacheReset.removedEntryCount !== manifest.pageCount ||
      result.resultCacheReset.measuredPhaseExcluded !== true ||
      !finiteNonNegative(result.resultCacheReset.completedAtEpochMs) ||
      result.resultCacheReset.completedAtEpochMs > result.action.issuedAtEpochMs)
  ) {
    fail(
      `Run ${result.runId} did not prove an exact ${manifest.pageCount}-entry result-cache reset before its measured phase.`,
    )
  }
  if (
    result.jobMonitor.errors.length > 0 ||
    result.jobMonitor.observations.length !== manifest.pageCount
  ) {
    fail(
      `Run ${result.runId} job monitor retained ${result.jobMonitor.observations.length}/${manifest.pageCount} observations with errors ${JSON.stringify(result.jobMonitor.errors)}.`,
    )
  }
  for (const expected of manifest.images) {
    const record = result.jobRecords.find((item) => item.pageIndex === expected.order - 1)
    const observation = result.jobMonitor.observations.find(
      (item) => item.pageIndex === expected.order - 1,
    )
    if (!record) fail(`Run ${result.runId} is missing page ${expected.order}.`)
    if (!observation) {
      fail(`Run ${result.runId} job monitor is missing page ${expected.order}.`)
    }
    if (record.sourceSha256 !== expected.sha256) {
      fail(`Run ${result.runId} page ${expected.order} source hash does not match the fixture.`)
    }
    if (record.terminalType !== 'complete') {
      fail(`Run ${result.runId} page ${expected.order} is not terminal complete.`)
    }
    if (
      observation.sourceSha256 !== expected.sha256 ||
      observation.terminalType !== 'complete' ||
      !finiteNonNegative(observation.createdAtUnixMs) ||
      observation.createdAtUnixMs < result.action.issuedAtEpochMs ||
      !finiteNonNegative(observation.terminalObservedAtEpochMs) ||
      observation.terminalObservedAtEpochMs < observation.createdAtUnixMs
    ) {
      fail(
        `Run ${result.runId} page ${expected.order} lacks causal acquisition/upload/daemon/terminal monitor evidence.`,
      )
    }
  }
  if (result.dom.wrappedImageCount !== manifest.pageCount) {
    fail(
      `Run ${result.runId} committed ${result.dom.wrappedImageCount} image overlays, expected ${manifest.pageCount}.`,
    )
  }
  if (result.dom.patchCount !== result.dom.regionCount) {
    fail(`Run ${result.runId} committed different patch and selectable-text counts.`)
  }
  if (
    result.dom.patches.some(
      (patch) => !patch.complete || patch.naturalWidth < 1 || patch.naturalHeight < 1,
    )
  ) {
    fail(`Run ${result.runId} committed an undecoded or dimensionless patch.`)
  }
  const replayPatches = result.routes.jobs.flatMap((job) => job.patches)
  if (replayPatches.length !== result.dom.patchCount) {
    fail(
      `Run ${result.runId} patch replay count ${replayPatches.length} does not match DOM count ${result.dom.patchCount}.`,
    )
  }
  const requiredStages = ['decoding', 'detecting', 'ocr', 'translating', 'packaging']
  const stages = new Set(result.routes.jobs.flatMap((job) => Object.keys(job.stageCounts)))
  if (result.exactCache) {
    const emittedInferenceStages = [...stages].filter((stage) =>
      INFERENCE_PROGRESS_STAGES.has(stage),
    )
    if (emittedInferenceStages.length !== 0) {
      fail(
        `Run ${result.runId} emitted inference progress during exact cache replay: ${emittedInferenceStages.join(', ')}.`,
      )
    }
  } else {
    for (const stage of requiredStages) {
      if (!stages.has(stage)) fail(`Run ${result.runId} has no ${stage} progress evidence.`)
    }
  }
  const expectedTotals = validateBenchmarkManifest(manifest)
  if (
    result.correctness.totals.expectedRegionCount !== expectedTotals.regionCount ||
    result.correctness.totals.detectorGoldBubbleCount !== expectedTotals.goldBubbleCount ||
    result.correctness.totals.expectedNarrationRegionCount !==
      expectedTotals.narrationRegionCount ||
    result.correctness.totals.expectedEnglishTranslationTargetCount !==
      expectedTotals.englishTranslationTargetCount ||
    result.correctness.totals.expectedUntouchedExclusionCount !==
      expectedTotals.untouchedExclusionCount
  ) {
    fail(`Run ${result.runId} correctness totals do not match the canonical fixture partitions.`)
  }
  assertRequiredGates(result.correctness.gates, `Run ${result.runId} correctness`)
  assertRequiredGates(result.jobRequests.gates, `Run ${result.runId} job request`)
  assertRequiredGates(
    result.routes.resourceIdentityEvidence.gates,
    `Run ${result.runId} runtime model identity`,
  )
  assertRequiredGates(result.performanceGates, `Run ${result.runId} performance`)
}

export function selectViewportPlan(
  manifest,
  goldPages,
  viewportHeight = BENCHMARK_VIEWPORT_HEIGHT,
) {
  const candidates = []
  for (const image of manifest.images) {
    const page = goldPages.find((candidate) => candidate.order === image.order)
    const translationTargets = page.regions.filter(isEnglishTranslationTarget)
    for (const region of translationTargets) {
      const regionBounds = bounds(region.bubblePolygon ?? region.textPolygon)
      if (!regionBounds) continue
      const center = (regionBounds.top + regionBounds.bottom) / 2
      const halfWindow = viewportHeight / image.height / 2
      const visible = translationTargets.filter((candidate) => {
        const candidateBounds = bounds(candidate.bubblePolygon ?? candidate.textPolygon)
        if (!candidateBounds) return false
        const candidateCenter = (candidateBounds.top + candidateBounds.bottom) / 2
        return candidateCenter >= center - halfWindow && candidateCenter <= center + halfWindow
      })
      if (visible.length >= MIN_VISIBLE_REGIONS && visible.length <= MAX_VISIBLE_REGIONS) {
        candidates.push({
          page: image.order,
          centerY: center,
          expectedVisibleRegionIds: visible.map((candidate) => candidate.id),
          expectedVisibleRegionCount: visible.length,
        })
      }
    }
  }
  candidates.sort(
    (left, right) =>
      right.expectedVisibleRegionCount - left.expectedVisibleRegionCount ||
      left.page - right.page ||
      left.centerY - right.centerY,
  )
  const selected = candidates[0]
  if (!selected) {
    fail(
      `Committed annotations do not contain a deterministic viewport intersecting ${MIN_VISIBLE_REGIONS}-${MAX_VISIBLE_REGIONS} story regions.`,
    )
  }
  return selected
}

async function positionBenchmarkViewport(page, plan) {
  await page.setViewportSize({
    width: BENCHMARK_VIEWPORT_WIDTH,
    height: BENCHMARK_VIEWPORT_HEIGHT,
  })
  const positioned = await page.evaluate((target) => {
    const image = document.querySelector(`#chapter img[data-page="${target.page}"]`)
    if (!(image instanceof HTMLImageElement) || !image.complete || image.naturalHeight < 1) {
      throw new Error(`Benchmark viewport page ${target.page} is unavailable.`)
    }
    const top =
      image.getBoundingClientRect().top +
      scrollY +
      target.centerY * image.getBoundingClientRect().height -
      innerHeight / 2
    scrollTo({ top: Math.max(0, top), behavior: 'instant' })
    return {
      page: target.page,
      requestedCenterY: target.centerY,
      scrollY,
      viewportWidth: innerWidth,
      viewportHeight: innerHeight,
    }
  }, plan)
  await page.evaluate(
    () =>
      new Promise((resolvePromise) =>
        requestAnimationFrame(() => requestAnimationFrame(resolvePromise)),
      ),
  )
  return positioned
}

async function prepareChapter(chapterPage, replicaUrl, runId, viewportPlan, navigate = true) {
  if (navigate) {
    await chapterPage.goto(`${replicaUrl}?run=${encodeURIComponent(runId)}`, {
      waitUntil: 'domcontentloaded',
    })
    await chapterPage.waitForFunction(
      () =>
        globalThis.__hskifyBenchmark?.ready === true ||
        Boolean(globalThis.__hskifyBenchmark?.error),
      undefined,
      { timeout: 60_000 },
    )
    const decodeError = await chapterPage.evaluate(() => globalThis.__hskifyBenchmark?.error)
    if (decodeError) fail(`Firefox failed to decode the local replica: ${decodeError}.`)
  }
  const viewport = await positionBenchmarkViewport(chapterPage, viewportPlan)
  await chapterPage.evaluate(() => performance.clearResourceTimings())
  await installDomObserver(chapterPage, runId)
  await chapterPage.bringToFront()
  return viewport
}

async function verifyPackagedResources(extensionPage, timeoutMs = 300_000) {
  const startedAtEpochMs = Date.now()
  const observed = []
  let status = await extensionMessage(extensionPage, { type: 'setup:status' })
  observed.push(status)
  if (status.state !== 'ready') {
    status = await extensionMessage(extensionPage, { type: 'setup:start' })
    observed.push(status)
  }
  while (status.state !== 'ready') {
    if (status.state === 'failed') {
      fail(`Packaged resource verification failed: ${status.errorCode}: ${status.message}`)
    }
    if (Date.now() - startedAtEpochMs > timeoutMs) {
      fail(`Packaged resource verification exceeded ${timeoutMs} ms.`)
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100))
    status = await extensionMessage(extensionPage, { type: 'setup:status' })
    const previous = observed.at(-1)
    if (
      previous?.state !== status.state ||
      previous?.currentFile !== status.currentFile ||
      previous?.completedBytes !== status.completedBytes
    ) {
      observed.push(status)
    }
  }
  return {
    startedAtEpochMs,
    readyAtEpochMs: Date.now(),
    durationMs: Date.now() - startedAtEpochMs,
    modelsLoaded: false,
    measuredRunExcluded: true,
    observed,
  }
}

export function clearExactResultCache(config) {
  const outputRoot = resolve(config.outputDirectory)
  const stateRoot = resolve(config.stateDirectory)
  const resultRoot = resolve(stateRoot, 'browser-cache', 'results')
  const relativeResultRoot = normalize(resultRoot.slice(outputRoot.length))
  if (
    stateRoot === outputRoot ||
    !stateRoot.startsWith(`${outputRoot}${sep}`) ||
    resultRoot === stateRoot ||
    !resultRoot.startsWith(`${stateRoot}${sep}`) ||
    relativeResultRoot.startsWith(`..${sep}`)
  ) {
    fail(`Refusing to clear result cache outside the isolated benchmark state: ${resultRoot}.`)
  }
  if (!existsSync(resultRoot)) {
    fail(`Expected isolated result cache does not exist: ${resultRoot}.`)
  }
  const removed = []
  for (const entry of readdirSync(resultRoot, { withFileTypes: true })) {
    if (!entry.isFile() || !/^[a-f0-9]{64}\.json$/u.test(entry.name)) {
      fail(`Unexpected entry in isolated result cache: ${entry.name}.`)
    }
    const path = join(resultRoot, entry.name)
    const bytes = statSync(path).size
    unlinkSync(path)
    removed.push({ file: entry.name, bytes })
  }
  const remaining = readdirSync(resultRoot)
  if (remaining.length !== 0) {
    fail(`Isolated result cache still contains ${remaining.length} entries after reset.`)
  }
  return {
    reason: 'Force a result-cache miss while preserving the live daemon and resident models.',
    isolatedResultCache: resultRoot,
    removedEntryCount: removed.length,
    removedBytes: removed.reduce((sum, entry) => sum + entry.bytes, 0),
    removed,
    completedAtEpochMs: Date.now(),
    completedAtUtc: nowIso(),
    measuredPhaseExcluded: true,
  }
}

async function executeCompleteRun(context, extensionPage, replicaUrl, config, descriptor) {
  const chapterPage = descriptor.chapterPage ?? (await context.newPage())
  const closePage = !descriptor.keepPage
  try {
    const viewport = await prepareChapter(
      chapterPage,
      replicaUrl,
      descriptor.runId,
      config.viewportPlan,
      !descriptor.chapterPage,
    )
    await chapterPage.bringToFront()
    const setup = await extensionMessage(extensionPage, { type: 'setup:status' })
    if (setup.state !== 'ready') {
      fail(
        `Installed-but-cold benchmark excludes downloads; setup is ${setup.state}: ${setup.message}`,
      )
    }
    await extensionMessage(extensionPage, { type: 'popup:prepare' })
    const pageUrl = chapterPage.url()
    await startJobMonitor(extensionPage, pageUrl, descriptor.runId)
    const action = await timedContentStart(
      extensionPage,
      config.hskLevel,
      pageUrl,
    )
    if (action.value.state === 'failed') {
      fail(`Content start failed before job submission: ${action.value.message}`)
    }
    const extensionState = await waitForPageState(
      extensionPage,
      chapterPage,
      ['complete', 'failed', 'cancelled'],
      config.runTimeoutMs,
    )
    if (extensionState.state !== 'complete') {
      const imageFailures = await chapterPage.evaluate(() =>
        [...document.querySelectorAll('[data-hmt-owned="true"]')]
          .map((host) => host.shadowRoot?.querySelector('.badge > span')?.textContent?.trim())
          .filter(Boolean),
      )
      fail(
        `Content runtime terminated before job collection: ${extensionState.state}: ${extensionState.message}; image failures: ${JSON.stringify(imageFailures)}`,
      )
    }
    const jobMonitor = await stopJobMonitor(extensionPage)
    const jobRecords = jobMonitor.observations
    const dom = await chapterDomEvidence(chapterPage)
    const acquisition = await sourceAcquisitionEvidence(chapterPage, action.issuedAtEpochMs)
    const routes = await routeEvidence(
      extensionPage,
      jobRecords,
      true,
      config.expectedResourceIdentities,
    )
    reconcileCompleteJobTerminals(jobMonitor, routes, dom)
    const translationCorrectness = buildQualityEvidence(routes, config.goldPages)
    const sourceGlyphs = await sourceGlyphEvidence(chapterPage, routes)
    const patchPng = buildPatchQualityEvidence(
      routes,
      config.goldPages,
      translationCorrectness.components,
      sourceGlyphs,
    )
    const commitOrdering = buildPatchCommitOrderingEvidence(routes, dom, action.issuedAtEpochMs)
    const correctness = {
      ...translationCorrectness,
      patchPng,
      commitOrdering,
      gates: [...translationCorrectness.gates, ...patchPng.gates, ...commitOrdering.gates],
    }
    const jobRequests = buildJobRequestEvidence(
      config.manifest,
      routes,
      jobRecords,
      config.hskLevel,
      action.issuedAtEpochMs,
    )
    const exactCache =
      routes.jobs.length === config.manifest.pageCount &&
      routes.jobs.every(
        (job) =>
          job.terminal?.type === 'complete' &&
          job.terminal?.message === 'Exact cached translation replayed' &&
          Object.keys(job.stageCounts).every(
            (stage) => !INFERENCE_PROGRESS_STAGES.has(stage),
          ),
      )
    const result = {
      benchmarkId: config.manifest.id,
      runId: descriptor.runId,
      sequence: descriptor.sequence,
      kind: descriptor.kind,
      measuredWarmRun: descriptor.kind === 'warm',
      ...(descriptor.resultCacheReset
        ? { resultCacheReset: descriptor.resultCacheReset }
        : {}),
      startedAtUtc: new Date(action.issuedAtEpochMs).toISOString(),
      endedAtUtc: nowIso(),
      measuredPhaseStartedAtEpochMs: action.issuedAtEpochMs,
      measuredPhaseEndedAtEpochMs: Math.max(
        action.responseAtEpochMs,
        ...dom.events.filter((event) => event.type === 'hudComplete').map((event) => event.epochMs),
      ),
      action: {
        type: 'content:start',
        issuedAtEpochMs: action.issuedAtEpochMs,
        responseAtEpochMs: action.responseAtEpochMs,
        responseLatencyMs: action.responseAtEpochMs - action.issuedAtEpochMs,
      },
      extensionState,
      viewport: {
        ...viewport,
        ...config.viewportPlan,
      },
      acquisition,
      jobRecords,
      jobMonitor,
      routes,
      dom,
      correctness,
      jobRequests,
      exactCache,
    }
    result.timing = timingEvidence(result, config.manifest, config.viewportPlan)
    result.performanceGates = performanceGates(descriptor.kind, result.timing, exactCache)
    return { result, chapterPage }
  } finally {
    if (closePage) await chapterPage.close().catch(() => undefined)
  }
}

async function measureReaderFeatures(page, routes) {
  const expected = new Map(
    routes.jobs.flatMap((job) =>
      finalRegions(job).map((region) => [
        region.id,
        { pinyin: region.pinyin, displayedChinese: region.displayedChinese },
      ]),
    ),
  )
  const pinyinAndComparison = await page.evaluate(
    (expectedEntries) => {
      const expectedById = new Map(expectedEntries)
      const hosts = [...document.querySelectorAll('[data-hmt-owned="true"]')].filter((node) =>
        node.shadowRoot?.querySelector('.hmt-region'),
      )
      const regions = hosts.flatMap((host) => [...host.shadowRoot.querySelectorAll('.hmt-region')])
      const pinyinMismatches = regions
        .filter((region) => {
          const expectedRegion = expectedById.get(region.dataset.regionId ?? '')
          return (
            !expectedRegion ||
            !region.dataset.pinyin?.trim() ||
            region.dataset.pinyin !== expectedRegion.pinyin ||
            region.textContent !== expectedRegion.displayedChinese
          )
        })
        .map((region) => region.dataset.regionId ?? '')
      const host = hosts[0]
      const viewport = host?.shadowRoot?.querySelector('.hmt-viewport')
      const modeControls = document.querySelector('[data-hmt-mode-controls="true"]')
      const buttons = [
        ...(modeControls?.shadowRoot?.querySelectorAll('.hmt-controls button') ?? []),
      ]
      const original = buttons.find((button) => button.textContent === 'Original')
      const chinese = buttons.find((button) => button.textContent === 'Chinese')
      const compare = buttons.find((button) => button.textContent === 'Hold to compare')
      const states = []
      if (original && chinese && compare && viewport) {
        original.click()
        states.push({ action: 'original', overlayHidden: viewport.hidden })
        chinese.click()
        states.push({ action: 'chinese', overlayHidden: viewport.hidden })
        compare.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
        states.push({ action: 'hold', overlayHidden: viewport.hidden })
        compare.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }))
        states.push({ action: 'release', overlayHidden: viewport.hidden })
      }
      const region = regions[0]
      if (region) {
        region.dispatchEvent(
          new KeyboardEvent('keydown', {
            key: 'a',
            code: 'KeyA',
            ctrlKey: true,
            bubbles: true,
            cancelable: true,
          }),
        )
      }
      return {
        regionCount: regions.length,
        pinyinMismatches,
        comparisonStates: states,
        selectionRegionId: region?.dataset.regionId ?? '',
      }
    },
    [...expected.entries()],
  )
  await page.waitForFunction(
    () =>
      [...document.querySelectorAll('[data-hmt-owned="true"]')].some((host) =>
        host.shadowRoot?.querySelector('.hmt-lookup:not([hidden]) .hmt-lookup-entry'),
      ),
    undefined,
    { timeout: 5_000 },
  )
  const dictionary = await page.evaluate(() => {
    const host = [...document.querySelectorAll('[data-hmt-owned="true"]')].find((candidate) =>
      candidate.shadowRoot?.querySelector('.hmt-lookup:not([hidden]) .hmt-lookup-entry'),
    )
    const popover = host?.shadowRoot?.querySelector('.hmt-lookup:not([hidden])')
    const entries = [...(popover?.querySelectorAll('.hmt-lookup-entry') ?? [])].map((entry) => ({
      text: entry.children[0]?.textContent ?? '',
      pinyinAndHsk: entry.children[1]?.textContent ?? '',
      definitions: entry.children[2]?.textContent ?? '',
    }))
    const context = [...(popover?.querySelector('.hmt-lookup-context')?.children ?? [])].map(
      (node) => node.textContent ?? '',
    )
    const button = popover?.querySelector('.hmt-speak')
    globalThis.__hskifySpeechProbe = { states: [] }
    if (button) {
      const capture = () =>
        globalThis.__hskifySpeechProbe.states.push({
          atEpochMs: Date.now(),
          text: button.textContent ?? '',
          ariaPressed: button.getAttribute('aria-pressed'),
          ariaLabel: button.getAttribute('aria-label'),
          disabled: button.disabled,
          voiceName: button.dataset.hmtVoiceName ?? '',
          voiceLang: button.dataset.hmtVoiceLang ?? '',
          voiceLocalService: button.dataset.hmtVoiceLocalService ?? '',
        })
      const observer = new MutationObserver(capture)
      observer.observe(button, {
        attributes: true,
        childList: true,
        subtree: true,
        characterData: true,
      })
      globalThis.__hskifySpeechProbe.observer = observer
      capture()
      button.click()
    }
    const voices = speechSynthesis
      .getVoices()
      .filter((voice) => voice.localService)
      .map((voice) => ({ name: voice.name, lang: voice.lang, voiceURI: voice.voiceURI }))
    return {
      entries,
      context,
      speechButtonPresent: Boolean(button),
      speechButtonInitiallyDisabled: Boolean(button?.disabled),
      localVoices: voices,
    }
  })
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 2_500))
  const speech = await page.evaluate(() => {
    const probe = globalThis.__hskifySpeechProbe
    probe?.observer?.disconnect()
    const states = probe?.states ?? []
    const selectedVoice = states.find(
      (state) => state.ariaPressed === 'true' && state.text === 'Stop',
    )
    const button = [...document.querySelectorAll('[data-hmt-owned="true"]')]
      .map((host) => host.shadowRoot?.querySelector('.hmt-speak'))
      .find(Boolean)
    if (button?.getAttribute('aria-pressed') === 'true') button.click()
    delete globalThis.__hskifySpeechProbe
    return {
      states,
      speakingObserved: Boolean(selectedVoice),
      selectedVoice: selectedVoice
        ? {
            name: selectedVoice.voiceName,
            lang: selectedVoice.voiceLang,
            localService: selectedVoice.voiceLocalService === 'true',
          }
        : undefined,
    }
  })
  const comparisonFunctional =
    pinyinAndComparison.comparisonStates.length === 4 &&
    pinyinAndComparison.comparisonStates[0].overlayHidden === true &&
    pinyinAndComparison.comparisonStates[1].overlayHidden === false &&
    pinyinAndComparison.comparisonStates[2].overlayHidden === true &&
    pinyinAndComparison.comparisonStates[3].overlayHidden === false
  const dictionaryFunctional =
    dictionary.entries.length > 0 &&
    dictionary.entries.every(
      (entry) => entry.text.trim() && entry.pinyinAndHsk.trim() && entry.definitions.trim(),
    ) &&
    dictionary.context.length === 2 &&
    dictionary.context.every((value) => value.trim())
  const gates = [
    booleanGate(
      'pinyin-functional',
      pinyinAndComparison.regionCount === expected.size &&
        pinyinAndComparison.pinyinMismatches.length === 0,
      'Rendered pinyin/text is missing or differs from the daemon result.',
    ),
    booleanGate(
      'dictionary-functional',
      dictionaryFunctional,
      'The bound local dictionary selection did not return pinyin, definitions, HSK state, and region context.',
    ),
    booleanGate(
      'comparison-functional',
      comparisonFunctional,
      'Original, Chinese, and hold-to-compare controls did not toggle only the overlay.',
    ),
    booleanGate(
      'mandarin-speech-functional',
      speech.speakingObserved &&
        speech.selectedVoice?.localService === true &&
        Boolean(speech.selectedVoice?.name) &&
        isMainlandMandarinLanguage(speech.selectedVoice?.lang),
      dictionary.speechButtonPresent
        ? 'Firefox did not expose and start an identified local Mainland Mandarin voice.'
        : 'The Mandarin speech control was not rendered.',
      {
        selectedVoice: speech.selectedVoice,
        enumeratedLocalVoices: dictionary.localVoices,
      },
    ),
  ]
  return { pinyinAndComparison, dictionary, speech, gates }
}

export function evaluateOverflowEvidence(cases, expectedRegionCount) {
  const supportedScenarioDenominator = cases.length
  const appliedSupportedScenarioNumerator = cases.filter(
    (item) =>
      item.cssZoomSupported && requestedZoomApplied(item.zoom, item.inlineZoom, item.computedZoom),
  ).length
  const completeRegionScenarioNumerator = cases.filter(
    (item) => item.regionCount === expectedRegionCount,
  ).length
  const checkedRegionDenominator = expectedRegionCount * supportedScenarioDenominator
  const overflowRegionNumerator = cases.reduce(
    (sum, item) => sum + item.overflowRegionIds.length,
    0,
  )
  return {
    supportedScenarioDenominator,
    appliedSupportedScenarioNumerator,
    expectedRegionCount,
    completeRegionScenarioNumerator,
    checkedRegionDenominator,
    overflowRegionNumerator,
    cases,
    gates: [
      exactGate(
        'supported-zoom-resize-scenarios-applied',
        appliedSupportedScenarioNumerator,
        supportedScenarioDenominator,
        {
          numerator: appliedSupportedScenarioNumerator,
          denominator: supportedScenarioDenominator,
        },
      ),
      exactGate(
        'all-translated-regions-checked-for-overflow',
        completeRegionScenarioNumerator,
        supportedScenarioDenominator,
        {
          expectedRegionCount,
          checkedRegionDenominator,
        },
      ),
      exactGate('zero-overflow-under-supported-zoom-resize', overflowRegionNumerator, 0, {
        numerator: overflowRegionNumerator,
        denominator: checkedRegionDenominator,
      }),
    ],
  }
}

async function measureOverflow(page, expectedRegionCount) {
  const scenarios = [
    { name: 'baseline', width: 1280, height: 900, zoom: 1 },
    { name: 'narrow-resize', width: 760, height: 720, zoom: 1 },
    { name: 'zoom-125', width: 1024, height: 768, zoom: 1.25 },
    { name: 'zoom-150-narrow', width: 760, height: 720, zoom: 1.5 },
  ]
  const cases = []
  for (const scenario of scenarios) {
    await page.setViewportSize({ width: scenario.width, height: scenario.height })
    const measured = await page.evaluate(async ({ zoom }) => {
      document.documentElement.style.zoom = String(zoom)
      await new Promise((resolvePromise) =>
        requestAnimationFrame(() => requestAnimationFrame(resolvePromise)),
      )
      const regions = [...document.querySelectorAll('[data-hmt-owned="true"]')].flatMap((host) => [
        ...(host.shadowRoot?.querySelectorAll('.hmt-region') ?? []),
      ])
      const inlineZoom = document.documentElement.style.zoom
      const computedZoom = getComputedStyle(document.documentElement).zoom
      return {
        cssZoomSupported: CSS.supports('zoom', String(zoom)),
        inlineZoom,
        computedZoom,
        regionCount: regions.length,
        regionMeasurements: regions.map((region) => ({
          regionId: region.dataset.regionId ?? '',
          clientWidth: region.clientWidth,
          clientHeight: region.clientHeight,
          scrollWidth: region.scrollWidth,
          scrollHeight: region.scrollHeight,
          overflows:
            region.scrollWidth > region.clientWidth + 0.5 ||
            region.scrollHeight > region.clientHeight + 0.5,
        })),
        overflowRegionIds: regions
          .filter(
            (region) =>
              region.scrollWidth > region.clientWidth + 0.5 ||
              region.scrollHeight > region.clientHeight + 0.5,
          )
          .map((region) => region.dataset.regionId ?? ''),
      }
    }, scenario)
    cases.push({ ...scenario, ...measured })
  }
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.evaluate(async () => {
    document.documentElement.style.zoom = '1'
    await new Promise((resolvePromise) =>
      requestAnimationFrame(() => requestAnimationFrame(resolvePromise)),
    )
  })
  return evaluateOverflowEvidence(cases, expectedRegionCount)
}

async function executeSourceReplacementProbe(page, extensionPage, replicaUrl, config) {
  await prepareChapter(page, replicaUrl, 'source-replacement-probe', config.viewportPlan)
  const originalSnapshot = await captureChapterSnapshot(page)
  await extensionMessage(extensionPage, { type: 'popup:prepare' })
  const pageUrl = page.url()
  await startJobMonitor(extensionPage, pageUrl, 'source-replacement-probe')
  const action = await timedContentStart(
    extensionPage,
    config.hskLevel,
    pageUrl,
  )
  await page.waitForFunction(
    (issuedAt) =>
      globalThis.__hskifyRuntimeEvidence?.events.some(
        (event) => event.epochMs >= issuedAt && event.type === 'patchDomCommitted' && event.visible,
      ),
    action.issuedAtEpochMs,
    { timeout: 30_000 },
  )
  const replacementPage = await page.evaluate((issuedAtEpochMs) => {
    const event = globalThis.__hskifyRuntimeEvidence.events.find(
      (candidate) =>
        candidate.epochMs >= issuedAtEpochMs &&
        candidate.type === 'patchDomCommitted' &&
        candidate.visible,
    )
    if (!event) {
      throw new Error('Could not identify a patch committed by the source-replacement action.')
    }
    const image = document.querySelector(`#chapter img[data-page="${event.page}"]`)
    if (!(image instanceof HTMLImageElement)) {
      throw new Error('Could not identify the actively rendered source image.')
    }
    return event.page
  }, action.issuedAtEpochMs)
  const replacementSource = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs='
  const expectedOuterHTML = await page.evaluate(
    ({ originalOuterHTML, pageNumber, source }) => {
      const template = document.createElement('template')
      template.innerHTML = originalOuterHTML
      const image = template.content.querySelector(`#chapter img[data-page="${pageNumber}"]`)
      if (!(image instanceof HTMLImageElement)) {
        throw new Error('The original snapshot is missing the replacement image.')
      }
      image.setAttribute('src', source)
      const chapter = template.content.querySelector('#chapter')
      if (!(chapter instanceof HTMLElement)) {
        throw new Error('The original snapshot is missing the chapter root.')
      }
      return chapter.outerHTML
    },
    {
      originalOuterHTML: originalSnapshot.outerHTML,
      pageNumber: replacementPage,
      source: replacementSource,
    },
  )
  const expectedSnapshot = await captureChapterSnapshot(page, expectedOuterHTML)
  await armExactChapterRestoration(page, expectedSnapshot)
  const replacement = await page.evaluate(
    ({ pageNumber, source }) => {
      const image = document.querySelector(`#chapter img[data-page="${pageNumber}"]`)
      if (!(image instanceof HTMLImageElement)) {
        throw new Error('Could not replace the active source image.')
      }
      const issuedAtEpochMs = Date.now()
      image.src = source
      return { page: pageNumber, source, issuedAtEpochMs }
    },
    { pageNumber: replacementPage, source: replacementSource },
  )
  const restorationProbe = await waitForExactChapterRestoration(page)
  const actualSnapshot = await captureChapterSnapshot(page)
  await extensionMessage(extensionPage, { type: 'popup:cancel' }).catch(() => undefined)
  const jobMonitor = await stopJobMonitor(extensionPage)
  const passed = exactChapterSnapshotMatch(expectedSnapshot, actualSnapshot)
  const evidence = {
    action,
    replacement,
    exactRestoration: {
      expected: expectedSnapshot,
      actual: actualSnapshot,
      exactMatch: passed,
      probe: restorationProbe,
    },
    jobMonitor,
    gates: [
      booleanGate(
        'source-replacement-restores-original',
        passed,
        'Replacing an active source did not restore the exact expected chapter DOM, attributes, siblings, and image order.',
      ),
    ],
  }
  return evidence
}

async function executeSameTabNavigationProbe(page, extensionPage, replicaUrl, config) {
  await prepareChapter(page, replicaUrl, 'same-tab-navigation-probe', config.viewportPlan)
  const expectedSnapshot = await captureChapterSnapshot(page)
  await extensionMessage(extensionPage, { type: 'popup:prepare' })
  const pageUrl = page.url()
  await startJobMonitor(extensionPage, pageUrl, 'same-tab-navigation-probe')
  const action = await timedContentStart(
    extensionPage,
    config.hskLevel,
    pageUrl,
  )
  await page.waitForFunction(
    (issuedAt) =>
      globalThis.__hskifyRuntimeEvidence?.events.some(
        (event) => event.epochMs >= issuedAt && event.type === 'patchDomCommitted' && event.visible,
      ),
    action.issuedAtEpochMs,
    { timeout: 30_000 },
  )
  await armExactChapterRestoration(page, expectedSnapshot)
  const navigation = await page.evaluate(() => {
    const token = crypto.randomUUID()
    globalThis.__hskifySameTabNavigationToken = token
    const beforeUrl = location.href
    const navigationEntryCount = performance.getEntriesByType('navigation').length
    const issuedAtEpochMs = Date.now()
    const next = new URL(location.href)
    next.searchParams.set('same-tab-cleanup', String(issuedAtEpochMs))
    history.pushState({ hskifyProbe: true }, '', next)
    return {
      token,
      issuedAtEpochMs,
      beforeUrl,
      afterUrl: location.href,
      navigationEntryCount,
      historyLength: history.length,
    }
  })
  const restorationProbe = await waitForExactChapterRestoration(page)
  const actualSnapshot = await captureChapterSnapshot(page)
  const continuity = await page.evaluate(
    ({ token, navigationEntryCount }) => ({
      sameWindowGlobal: globalThis.__hskifySameTabNavigationToken === token,
      navigationEntryCount: performance.getEntriesByType('navigation').length,
      navigationEntryCountUnchanged:
        performance.getEntriesByType('navigation').length === navigationEntryCount,
      currentUrl: location.href,
    }),
    {
      token: navigation.token,
      navigationEntryCount: navigation.navigationEntryCount,
    },
  )
  const state = await extensionMessage(extensionPage, { type: 'popup:state' })
  const jobMonitor = await stopJobMonitor(extensionPage)
  const exactMatch = exactChapterSnapshotMatch(expectedSnapshot, actualSnapshot)
  const sameDocumentNavigation =
    navigation.beforeUrl !== navigation.afterUrl &&
    continuity.currentUrl === navigation.afterUrl &&
    continuity.sameWindowGlobal &&
    continuity.navigationEntryCountUnchanged
  const passed = exactMatch && sameDocumentNavigation && state.state === 'idle'
  return {
    action,
    navigation,
    continuity,
    extensionState: state,
    exactRestoration: {
      expected: expectedSnapshot,
      actual: actualSnapshot,
      exactMatch,
      probe: restorationProbe,
    },
    jobMonitor,
    gates: [
      booleanGate(
        'same-tab-navigation-restores-exact-original',
        passed,
        'A same-document history navigation did not preserve the browsing context, reset the controller, and restore the exact original chapter DOM.',
      ),
    ],
  }
}

async function executeCancellationRun(
  context,
  extensionPage,
  replicaUrl,
  config,
  sequence,
  resultCacheReset,
) {
  if (resultCacheReset?.removedEntryCount !== config.manifest.pageCount) {
    fail(
      `Cancellation requires an exact ${config.manifest.pageCount}-entry result-cache reset.`,
    )
  }
  const descriptor = {
    runId: 'cancellation',
    kind: 'cancellation',
    sequence,
    resultCacheReset,
  }
  const chapterPage = await context.newPage()
  try {
    await prepareChapter(chapterPage, replicaUrl, descriptor.runId, config.viewportPlan)
    const expectedSnapshot = await captureChapterSnapshot(chapterPage)
    await chapterPage.bringToFront()
    await extensionMessage(extensionPage, { type: 'popup:prepare' })
    const pageUrl = chapterPage.url()
    await startJobMonitor(extensionPage, pageUrl, descriptor.runId)
    const startAction = await timedContentStart(
      extensionPage,
      config.hskLevel,
      pageUrl,
    )
    const records = await waitForInFlightJobs(extensionPage, pageUrl, 30_000)
    await armExactChapterRestoration(chapterPage, expectedSnapshot)
    const cancellationTargets = await armDaemonCancellationProbe(extensionPage, records)
    const cancellationTargetIds = new Set(
      cancellationTargets.map((target) => target.jobId),
    )
    const cancellationRecords = records.filter((record) =>
      cancellationTargetIds.has(record.jobId),
    )
    const cancelAction = await timedExtensionMessage(extensionPage, { type: 'popup:cancel' })
    const [restorationProbe, daemonProbe] = await Promise.all([
      waitForExactChapterRestoration(chapterPage, 30_000),
      waitForDaemonCancellationProbe(extensionPage, 30_000),
    ])
    const state =
      cancelAction.value?.state === 'cancelled'
        ? cancelAction.value
        : await waitForPageState(extensionPage, chapterPage, ['cancelled', 'failed'], 30_000)
    const timing = cancellationTiming({
      cancelIssuedAtEpochMs: cancelAction.issuedAtEpochMs,
      pageRestoredAtEpochMs: restorationProbe.restoredAtEpochMs,
      daemonTerminalObservedAtEpochMs: daemonProbe.terminalObservedAtEpochMs,
    })
    const postHocEvidenceStartedAtEpochMs = Date.now()
    const terminalRoutes = await routeEvidence(
      extensionPage,
      cancellationRecords,
      true,
      config.expectedResourceIdentities,
    )
    const allDaemonJobsCancelled =
      terminalRoutes.jobs.length === cancellationRecords.length &&
      terminalRoutes.jobs.every((job) => job.terminal?.type === 'cancelled')
    if (state.state !== 'cancelled' || !allDaemonJobsCancelled) {
      fail(
        `Cancellation did not win both page and daemon terminal state: ${JSON.stringify(state)}.`,
      )
    }
    const jobMonitor = await stopJobMonitor(extensionPage)
    const dom = await chapterDomEvidence(chapterPage)
    const actualSnapshot = await captureChapterSnapshot(chapterPage)
    const originalRestored = exactChapterSnapshotMatch(expectedSnapshot, actualSnapshot)
    const restoration = {
      expected: expectedSnapshot,
      actual: actualSnapshot,
      exactMatch: originalRestored,
      probe: restorationProbe,
    }
    const postHocEvidenceEndedAtEpochMs = Date.now()
    const gates = [
      measuredGate(
        'page-cancellation',
        timing.pageCancellationLatencyMs,
        BENCHMARK_LIMITS.cancellationMs,
      ),
      measuredGate(
        'daemon-cancellation',
        timing.daemonCancellationLatencyMs,
        BENCHMARK_LIMITS.cancellationMs,
      ),
      booleanGate(
        'cancellation-restores-original',
        originalRestored,
        'Cancellation did not restore the exact pre-translation chapter DOM, image attributes, siblings, and order.',
      ),
      ...terminalRoutes.resourceIdentityEvidence.gates,
    ]
    const result = {
      benchmarkId: config.manifest.id,
      ...descriptor,
      measuredWarmRun: false,
      startedAtUtc: new Date(cancelAction.issuedAtEpochMs).toISOString(),
      endedAtUtc: nowIso(),
      measuredPhaseStartedAtEpochMs: timing.measuredPhaseStartedAtEpochMs,
      measuredPhaseEndedAtEpochMs: timing.measuredPhaseEndedAtEpochMs,
      startAction,
      cancelAction,
      cancelIssuedAtEpochMs: timing.cancelIssuedAtEpochMs,
      pageCancelledAtEpochMs: timing.pageRestoredAtEpochMs,
      pageRestoredAtEpochMs: timing.pageRestoredAtEpochMs,
      daemonCancelledAtEpochMs: timing.daemonTerminalObservedAtEpochMs,
      daemonTerminalObservedAtEpochMs: timing.daemonTerminalObservedAtEpochMs,
      pageCancellationLatencyMs: timing.pageCancellationLatencyMs,
      daemonCancellationLatencyMs: timing.daemonCancellationLatencyMs,
      cancellationTimestampDefinition: timing.timestampDefinition,
      cancellationTargets,
      daemonCancellationProbe: daemonProbe,
      extensionState: state,
      preCancelJobRecords: records,
      terminalRoutes,
      postHocEvidence: {
        startedAtEpochMs: postHocEvidenceStartedAtEpochMs,
        endedAtEpochMs: postHocEvidenceEndedAtEpochMs,
        excludedFromMeasuredPhase: true,
        includes: [
          'GET /health',
          'GET /setup',
          'full update replay',
          'patch download, hash, and decode',
          'DOM and job-monitor evidence',
        ],
      },
      jobMonitor,
      dom,
      restoration,
      gates,
    }
    return result
  } finally {
    await chapterPage.close().catch(() => undefined)
  }
}

async function main() {
  const configPath = process.argv[2]
  if (!configPath) fail('Usage: node Chapter5.Firefox.mjs <config.json>')
  const config = JSON.parse(readFileSync(configPath, 'utf8'))
  config.manifest = JSON.parse(readFileSync(config.manifestPath, 'utf8'))
  validateBenchmarkManifest(config.manifest)
  assertCompleteTranslationGold(config.manifest)
  config.expectedResourceIdentities = validateExpectedResourceIdentities(
    config.expectedResourceIdentities,
  )
  const fixtureDirectory = dirname(config.manifestPath)
  config.goldPages = config.manifest.images
    .sort((left, right) => left.order - right.order)
    .map((image) => {
      const annotation = JSON.parse(readFileSync(join(fixtureDirectory, image.annotation), 'utf8'))
      const regions = annotation.regions
      if (
        annotation.schemaVersion !== 1 ||
        annotation.page?.order !== image.order ||
        annotation.page?.file !== image.file ||
        annotation.page?.width !== image.width ||
        annotation.page?.height !== image.height ||
        annotation.page?.sourceSha256 !== image.sha256 ||
        !Array.isArray(regions) ||
        regions.length !== image.expectedRegionCount ||
        regions.some(
          (region, index) =>
            region.id !==
              `30ysp-ch5-p${String(image.order).padStart(3, '0')}-r${String(index).padStart(2, '0')}` ||
            region.readingOrder !== index ||
            !['dialogue', 'thought', 'narration'].includes(region.kind),
        )
      ) {
        fail(`Page ${image.order} annotation does not match the canonical manifest structure.`)
      }
      const detectorGoldCount = regions.filter((region) =>
        ['dialogue', 'thought'].includes(region.kind),
      ).length
      const narrationCount = regions.filter((region) => region.kind === 'narration').length
      const translationTargets = regions.filter(isEnglishTranslationTarget)
      const exclusions = regions.filter((region) => !isEnglishTranslationTarget(region))
      if (
        detectorGoldCount !== image.expectedDialogueBubbleCount ||
        narrationCount !== image.expectedNarrationCount ||
        translationTargets.length !== image.expectedEnglishTranslationTargetCount ||
        exclusions.length !== image.expectedUntouchedExclusionCount ||
        translationTargets.some(
          (region) =>
            typeof region.simplifiedChinese !== 'string' ||
            region.simplifiedChinese.trim().length === 0 ||
            typeof region.pinyin !== 'string' ||
            region.pinyin.trim().length === 0 ||
            !Array.isArray(region.hskTokens) ||
            region.hskTokens.length === 0,
        )
      ) {
        fail(`Page ${image.order} annotation lacks canonical complete translation gold.`)
      }
      return { order: image.order, file: image.file, regions }
    })
  config.viewportPlan = selectViewportPlan(config.manifest, config.goldPages)
  mkdirSync(config.outputDirectory, { recursive: true })
  const { server, port } = await startReplicaServer(config.repositoryRoot, config.port)
  const replicaUrl = `http://127.0.0.1:${port}/fixtures/benchmarks/${BENCHMARK_ID}/replica/index.html`
  let context
  let extensionPage
  let identity
  const results = []
  let readerFeatures
  let overflow
  let sourceReplacement
  let sameTabNavigation
  let setupEvidence
  let failure
  try {
    const launched = await launchPackagedFirefox(config)
    context = launched.context
    extensionPage = launched.extensionPage
    identity = launched.identity
    setupEvidence = await verifyPackagedResources(extensionPage)
    let sequence = 1
    const cold = await executeCompleteRun(context, extensionPage, replicaUrl, config, {
      runId: 'installed-cold',
      kind: 'installed-cold',
      sequence: sequence++,
    })
    results.push(cold.result)
    validateCompleteRun(cold.result, config.manifest)
    const warmupCacheReset = clearExactResultCache(config)
    const warmup = await executeCompleteRun(context, extensionPage, replicaUrl, config, {
      runId: 'warmup',
      kind: 'warmup',
      sequence: sequence++,
      keepPage: true,
      resultCacheReset: warmupCacheReset,
    })
    results.push(warmup.result)
    validateCompleteRun(warmup.result, config.manifest)
    for (let index = 1; index <= config.iterations; index += 1) {
      const resultCacheReset = clearExactResultCache(config)
      const warm = await executeCompleteRun(context, extensionPage, replicaUrl, config, {
        runId: `warm-${String(index).padStart(3, '0')}`,
        kind: 'warm',
        sequence: sequence++,
        resultCacheReset,
      })
      results.push(warm.result)
      validateCompleteRun(warm.result, config.manifest)
    }
    const cacheReplay = await executeCompleteRun(context, extensionPage, replicaUrl, config, {
      runId: 'cache-replay',
      kind: 'cache-replay',
      sequence: sequence++,
      chapterPage: warmup.chapterPage,
      keepPage: true,
    })
    results.push(cacheReplay.result)
    validateCompleteRun(cacheReplay.result, config.manifest)
    readerFeatures = await measureReaderFeatures(warmup.chapterPage, cacheReplay.result.routes)
    overflow = await measureOverflow(warmup.chapterPage, cacheReplay.result.dom.regionCount)
    sameTabNavigation = await executeSameTabNavigationProbe(
      warmup.chapterPage,
      extensionPage,
      replicaUrl,
      config,
    )
    const sourceReplacementEvidence = await executeSourceReplacementProbe(
      warmup.chapterPage,
      extensionPage,
      replicaUrl,
      config,
    )
    sourceReplacement = {
      ...sourceReplacementEvidence,
      sameTabNavigation,
      gates: [...sourceReplacementEvidence.gates, ...sameTabNavigation.gates],
    }
    cacheReplay.result.readerFeatures = readerFeatures
    cacheReplay.result.overflow = overflow
    cacheReplay.result.sourceReplacement = sourceReplacement
    cacheReplay.result.sameTabNavigation = sameTabNavigation
    await warmup.chapterPage.close().catch(() => undefined)
    const cancellationCacheReset = clearExactResultCache(config)
    const cancellation = await executeCancellationRun(
      context,
      extensionPage,
      replicaUrl,
      config,
      sequence++,
      cancellationCacheReset,
    )
    results.push(cancellation)
    assertRequiredGates(cancellation.gates, 'Cancellation')
    assertRequiredGates(
      [...readerFeatures.gates, ...overflow.gates, ...sourceReplacement.gates],
      'Reader feature correctness',
    )
  } catch (error) {
    failure = {
      failedAtUtc: nowIso(),
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : '',
    }
  } finally {
    extensionPage?.close()
    await context?.close().catch(() => undefined)
    await new Promise((resolvePromise) => server.close(resolvePromise))
  }
  for (const result of results) {
    writeJsonSync(join(config.outputDirectory, `${result.runId}.raw.json`), result)
  }
  const runIndex = {
    benchmarkId: config.manifest.id,
    evidenceWrittenAtUtc: nowIso(),
    extensionIdentity: identity,
    setupEvidence,
    replicaUrl,
    viewportPlan: config.viewportPlan,
    liveNetworkSmokeIncluded: false,
    writesDuringMeasuredPhases: 0,
    evidenceWritePolicy:
      'All driver raw artifacts are synchronously written and fsynced only after Firefox closes and every measured phase has ended.',
    results: results.map((result) => ({
      runId: result.runId,
      kind: result.kind,
      sequence: result.sequence,
      rawFile: `${result.runId}.raw.json`,
    })),
  }
  writeJsonSync(join(config.outputDirectory, 'run-index.json'), runIndex)
  if (failure) {
    writeJsonSync(join(config.outputDirectory, 'driver-failure.json'), failure)
    throw new Error(failure.message)
  }
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : ''
if (import.meta.url === invokedPath) await main()
