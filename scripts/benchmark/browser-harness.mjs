/*
 * Reader-neutral packaged Firefox harness.
 *
 * This module contains only browser transport and evidence collection. It
 * never creates daemon jobs and it has no knowledge of a particular title,
 * page count, OCR fixture, or benchmark. The release runner supplies the
 * local reader and its annotations.
 */

import { createRequire } from 'node:module'
import { closeSync, mkdirSync, openSync, writeSync, fsyncSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const EXTENSION_ID = 'hsk-manga-translator@local.hskify'
const EXTENSION_UUID = '7e9a74d0-34ad-4ff7-9c2c-1ea555945100'
const ACTIVE_JOB_PREFIX = 'hmt.activeJob.'
const SESSION_STORAGE_KEY = 'hmt.nativeSession'

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
  if (remote.type === 'string' || remote.type === 'boolean' || remote.type === 'bigint') {
    return remote.value
  }
  if (remote.type === 'array') return (remote.value ?? []).map(decodeBidiRemoteValue)
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
  const source = typeof pageFunction === 'function' ? pageFunction.toString() : String(pageFunction)
  const serialized = JSON.stringify(argument)
  return `(${source})(${serialized === undefined ? 'undefined' : serialized})`
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
    if (this.closed) return Promise.reject(new Error(`${method} cannot run after Firefox closed.`))
    return new Promise((resolvePromise, rejectPromise) => {
      const id = ++this.nextId
      const timer = setTimeout(() => {
        if (!this.pending.delete(id)) return
        rejectPromise(new Error(`${method} timed out after ${timeoutMs} ms.`))
      }, timeoutMs)
      this.pending.set(id, { method, resolve: resolvePromise, reject: rejectPromise, timer })
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
    throw new Error('The packaged Firefox harness requires Node.js WebSocket support.')
  }
  const path = join(profileDirectory, 'WebDriverBiDiServer.json')
  const deadline = Date.now() + 30_000
  let lastError
  while (Date.now() < deadline) {
    try {
      const parsed = JSON.parse(readFileSync(path, 'utf8'))
      if (
        typeof parsed.ws_host === 'string' &&
        Number.isInteger(parsed.ws_port) &&
        parsed.ws_port > 0 &&
        parsed.ws_port <= 65_535
      ) {
        const host = parsed.ws_host.includes(':') ? `[${parsed.ws_host}]` : parsed.ws_host
        const socket = new WebSocket(`ws://${host}:${parsed.ws_port}/session`)
        await new Promise((resolvePromise, rejectPromise) => {
          const timer = setTimeout(() => rejectPromise(new Error('Timed out connecting to Firefox BiDi.')), 30_000)
          socket.addEventListener('open', () => {
            clearTimeout(timer)
            resolvePromise()
          }, { once: true })
          socket.addEventListener('error', () => {
            clearTimeout(timer)
            rejectPromise(new Error('Firefox BiDi connection failed.'))
          }, { once: true })
        })
        return new FirefoxBidiClient(socket)
      }
      lastError = new Error('Firefox BiDi metadata is incomplete.')
    } catch (error) {
      lastError = error
    }
    await delay(50)
  }
  throw new Error(`Firefox WebDriver BiDi did not become ready: ${lastError?.message ?? 'unknown error'}`)
}

class ExtensionPage {
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
      throw new Error(result.exceptionDetails?.text ?? 'Extension evaluation failed.')
    }
    return decodeBidiRemoteValue(result.result)
  }

  async waitForFunction(pageFunction, argument, options = {}) {
    const timeoutMs = options.timeout ?? 30_000
    const pollingMs = typeof options.polling === 'number' && options.polling > 0 ? options.polling : 100
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      const value = await this.evaluate(pageFunction, argument)
      if (value) return { jsonValue: async () => value }
      await delay(pollingMs)
    }
    throw new Error(`Extension condition timed out after ${timeoutMs} ms.`)
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
        'extensions.webextensions.uuids': JSON.stringify({ [EXTENSION_ID]: EXTENSION_UUID }),
        ...(config.firefoxUserPrefs ?? {}),
      },
    })
    if (!context.pages()[0]) await context.newPage()
    bidi = await connectFirefoxBidi(config.profileDirectory)
    await bidi.command('session.new', { capabilities: {} })
    const installed = await bidi.command('webExtension.install', {
      extensionData: { type: 'archivePath', path: resolve(config.extensionPackagePath) },
      'moz:permanent': false,
    })
    if (installed.extension !== EXTENSION_ID) throw new Error(`Installed extension ID mismatch: ${installed.extension}.`)
    const realms = await bidi.command('script.getRealms')
    const realm = realms.realms?.find((candidate) => candidate.type === 'window' && typeof candidate.context === 'string')
    if (!realm) throw new Error('Firefox BiDi exposed no extension browsing context.')
    await bidi.command('browsingContext.navigate', {
      context: realm.context,
      url: `moz-extension://${EXTENSION_UUID}/popup.html`,
      wait: 'none',
    })
    const extensionPage = new ExtensionPage(bidi, realm.context)
    await extensionPage.waitForFunction(
      () => document.readyState === 'complete' && typeof globalThis.browser === 'object',
      undefined,
      { timeout: 30_000, polling: 50 },
    )
    const identity = await extensionPage.evaluate(async () => ({
      id: globalThis.browser.runtime.id,
      manifest: globalThis.browser.runtime.getManifest(),
      origin: new URL(globalThis.browser.runtime.getURL('')).origin,
    }))
    if (identity.id !== EXTENSION_ID) throw new Error(`Extension runtime ID mismatch: ${identity.id}.`)
    if (identity.manifest.version !== config.extensionVersion) {
      throw new Error(`Extension version mismatch: ${identity.manifest.version} != ${config.extensionVersion}.`)
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
    throw new Error(`${response?.error?.code ?? 'EXTENSION_MESSAGE_FAILED'}: ${response?.error?.message ?? 'No response.'}`)
  }
  return response.value
}

export async function prepareContentRuntime(extensionPage, expectedPageUrl) {
  return extensionPage.evaluate(async (pageUrl) => {
    const tabs = await globalThis.browser.tabs.query({})
    const tab = tabs.find((candidate) => candidate.url === pageUrl)
    if (!Number.isInteger(tab?.id)) throw new Error(`No chapter tab for ${pageUrl}.`)
    await globalThis.browser.scripting.executeScript({ target: { tabId: tab.id, allFrames: false }, files: ['translator.js'] })
    const state = await globalThis.browser.tabs.sendMessage(tab.id, { type: 'content:state' })
    if (!state || typeof state.state !== 'string') throw new Error(`Content runtime did not initialize for ${pageUrl}.`)
    return { tabId: tab.id, state }
  }, expectedPageUrl)
}

async function timedExtensionMessage(extensionPage, message) {
  const timed = await extensionPage.evaluate(async (payload) => {
    const issuedAtEpochMs = Date.now()
    if (globalThis.__hskifyJobMonitor && globalThis.__hskifyJobMonitor.actionIssuedAtEpochMs === 0) {
      globalThis.__hskifyJobMonitor.actionIssuedAtEpochMs = issuedAtEpochMs
    }
    const response = await globalThis.browser.runtime.sendMessage(payload)
    return { issuedAtEpochMs, responseAtEpochMs: Date.now(), response }
  }, message)
  if (!timed.response || timed.response.ok !== true) throw new Error(`${timed.response?.error?.code ?? 'EXTENSION_MESSAGE_FAILED'}: ${timed.response?.error?.message ?? 'No response.'}`)
  return { issuedAtEpochMs: timed.issuedAtEpochMs, responseAtEpochMs: timed.responseAtEpochMs, value: timed.response.value }
}

export async function timedContentStart(extensionPage, hskLevel, expectedPageUrl, nameTranslation = 'keep-original') {
  const timed = await extensionPage.evaluate(async ({ level, pageUrl, names }) => {
    const tabs = await globalThis.browser.tabs.query({})
    const tab = tabs.find((candidate) => candidate.url === pageUrl)
    if (!Number.isInteger(tab?.id)) throw new Error(`No chapter tab for ${pageUrl}.`)
    const issuedAtEpochMs = Date.now()
    if (globalThis.__hskifyJobMonitor && globalThis.__hskifyJobMonitor.actionIssuedAtEpochMs === 0) {
      globalThis.__hskifyJobMonitor.actionIssuedAtEpochMs = issuedAtEpochMs
    }
    const response = await globalThis.browser.tabs.sendMessage(tab.id, {
      type: 'content:start', scope: 'all', hskLevel: level, learningMode: 'natural', nameTranslation: names,
    })
    return { issuedAtEpochMs, responseAtEpochMs: Date.now(), response }
  }, { level: hskLevel, pageUrl: expectedPageUrl, names: nameTranslation })
  if (!timed.response || typeof timed.response.state !== 'string') throw new Error('Content runtime returned no valid start state.')
  return { issuedAtEpochMs: timed.issuedAtEpochMs, responseAtEpochMs: timed.responseAtEpochMs, value: timed.response }
}

export async function startJobMonitor(extensionPage, pageUrl, runId) {
  await extensionPage.evaluate(({ prefix, expectedPageUrl, id }) => {
    if (globalThis.__hskifyJobMonitor?.timer) clearInterval(globalThis.__hskifyJobMonitor.timer)
    const monitor = {
      runId: id, pageUrl: expectedPageUrl, actionIssuedAtEpochMs: 0, observations: new Map(), errors: [], timer: 0,
      onStorageChanged: undefined,
    }
    const observe = (key, value) => {
      if (!key.startsWith(prefix) || value?.pageUrl !== expectedPageUrl || typeof value?.createdAtUnixMs !== 'number' || value.createdAtUnixMs < monitor.actionIssuedAtEpochMs) return
      const now = Date.now()
      const previous = monitor.observations.get(value.jobId)
      monitor.observations.set(value.jobId, {
        jobId: value.jobId, pageIndex: value.pageIndex, sourceSha256: value.sourceSha256, sourceUrl: value.sourceUrl,
        sourceWidth: value.sourceWidth, sourceHeight: value.sourceHeight, submittedRequest: value.submittedRequest,
        uploadedImageBytes: value.uploadedImageBytes, submittedAtUnixMs: value.submittedAtUnixMs, createdAtUnixMs: value.createdAtUnixMs,
        firstObservedAtEpochMs: previous?.firstObservedAtEpochMs ?? now,
        terminalType: value.terminalType,
        terminalObservedAtEpochMs: previous?.terminalObservedAtEpochMs ?? (value.terminalType ? now : undefined),
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
      for (const [key, change] of Object.entries(changes)) observe(key, change.newValue ?? change.oldValue)
    }
    globalThis.browser.storage.onChanged.addListener(monitor.onStorageChanged)
    monitor.sample = sample
    monitor.timer = setInterval(() => void sample(), 10)
    globalThis.__hskifyJobMonitor = monitor
  }, { prefix: ACTIVE_JOB_PREFIX, expectedPageUrl: pageUrl, id: runId })
}

export async function stopJobMonitor(extensionPage) {
  return extensionPage.evaluate(async () => {
    const monitor = globalThis.__hskifyJobMonitor
    if (!monitor) return { observations: [], errors: ['job monitor was not installed'] }
    clearInterval(monitor.timer)
    if (monitor.onStorageChanged) globalThis.browser.storage.onChanged.removeListener(monitor.onStorageChanged)
    await monitor.sample()
    const result = {
      actionIssuedAtEpochMs: monitor.actionIssuedAtEpochMs,
      observations: [...monitor.observations.values()].sort((left, right) => left.pageIndex - right.pageIndex),
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
    await delay(200)
  }
  throw new Error(`Timed out waiting for extension state ${expected.join('/')} (last: ${JSON.stringify(last)}).`)
}

export async function installDomObserver(page, runId) {
  await page.evaluate((id) => {
    for (const observer of globalThis.__hskifyRuntimeEvidence?.observers ?? []) observer.disconnect()
    const state = { runId: id, observerInstalledAtEpochMs: Date.now(), nextEventIndex: 1, events: [], observedShadowRoots: 0, observers: [], lastHudState: '' }
    globalThis.__hskifyRuntimeEvidence = state
    const observed = new WeakSet()
    const emit = (type, details = {}) => state.events.push({ index: state.nextEventIndex++, type, epochMs: Date.now(), performanceMs: performance.now(), ...details })
    const pageFor = (element) => {
      const root = element.getRootNode()
      const host = root instanceof ShadowRoot ? root.host : element
      return Number(host.closest('.hmt-wrapper')?.dataset.hmtSourcePage ?? 0)
    }
    const visible = (element) => {
      const rect = element.getBoundingClientRect()
      const style = getComputedStyle(element)
      return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth && style.visibility !== 'hidden' && style.display !== 'none'
    }
    const recordHud = (root) => {
      const title = root.querySelector('.title')?.textContent?.trim() ?? ''
      const detail = root.querySelector('.detail')?.textContent?.trim() ?? ''
      if (!title || title === state.lastHudState) return
      state.lastHudState = title
      if (title === 'Hskify' || title === 'Translating chapter' || title === 'Preparing chapter') emit('hudAcknowledged', { title, detail })
      if (title === 'Translation complete') emit('hudComplete', { title, detail })
      if (title === 'Translation cancelled') emit('hudCancelled', { title, detail })
      if (title === 'Translation needs attention') emit('hudFailed', { title, detail })
    }
    let recordElement
    const observeShadow = (root) => {
      if (observed.has(root)) return
      observed.add(root)
      state.observedShadowRoots += 1
      const observer = new MutationObserver((records) => {
        for (const record of records) {
          for (const node of record.addedNodes) recordElement(node)
          for (const node of record.removedNodes) if (node instanceof Element && node.matches('.hmt-patch, .hmt-region')) emit('translatedNodeRemoved', { className: node.className, page: pageFor(node) })
        }
        recordHud(root)
      })
      observer.observe(root, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ['hidden', 'aria-pressed', 'aria-busy'] })
      state.observers.push(observer)
      for (const child of root.children) recordElement(child)
      recordHud(root)
    }
    recordElement = (element) => {
      if (!(element instanceof Element)) return
      for (const patch of [...(element.matches('.hmt-patch') ? [element] : []), ...element.querySelectorAll('.hmt-patch')]) emit('patchDomCommitted', { patchId: patch.dataset.patchId ?? '', complete: patch.complete, naturalWidth: patch.naturalWidth, naturalHeight: patch.naturalHeight, decodedAndInstalled: patch.complete && patch.naturalWidth > 0 && patch.naturalHeight > 0, page: pageFor(patch), visible: visible(patch) })
      for (const region of [...(element.matches('.hmt-region') ? [element] : []), ...element.querySelectorAll('.hmt-region')]) emit('selectableTextDomCommitted', { regionId: region.dataset.regionId ?? '', hskValid: region.dataset.hskValid ?? '', repairState: region.dataset.hskRepairState ?? '', text: region.textContent ?? '', pinyin: region.dataset.pinyin ?? '', page: pageFor(region), visible: visible(region) })
      for (const owned of [...(element.matches('[data-hmt-owned="true"]') ? [element] : []), ...element.querySelectorAll('[data-hmt-owned="true"]')]) {
        if (owned.classList.contains('hmt-wrapper')) emit('imageWrapperCommitted', { page: Number(owned.querySelector('img[data-page]')?.dataset.page ?? 0) })
        if (owned.shadowRoot) observeShadow(owned.shadowRoot)
      }
    }
    const observer = new MutationObserver((records) => { for (const record of records) for (const node of record.addedNodes) recordElement(node) })
    observer.observe(document.documentElement, { childList: true, subtree: true })
    state.observers.push(observer)
    for (const child of document.documentElement.children) recordElement(child)
    emit('observerReady')
  }, runId)
}

export async function chapterDomEvidence(page) {
  return page.evaluate(() => {
    const hosts = [...document.querySelectorAll('[data-hmt-owned="true"]')].filter((node) => node.shadowRoot)
    const patches = []
    const regions = []
    const wrapperNodes = [...document.querySelectorAll('.hmt-wrapper')]
    const wrapperIndexes = new Map(wrapperNodes.map((wrapper, index) => [wrapper, index + 1]))
    const wrapperPage = (wrapper) => {
      const explicit = Number(wrapper?.dataset.hmtSourcePage ?? 0)
      return explicit > 0 ? explicit : wrapperIndexes.get(wrapper) ?? 0
    }
    const wrappers = wrapperNodes.map((wrapper, index) => ({ index, page: wrapperPage(wrapper), ownedHostCount: wrapper.querySelectorAll('[data-hmt-owned="true"]').length, patchCount: wrapper.querySelectorAll('.hmt-patch').length, regionCount: wrapper.querySelectorAll('.hmt-region').length }))
    let degradedFitCount = 0
    for (const host of hosts) {
      const pageNumber = wrapperPage(host.closest('.hmt-wrapper'))
      for (const patch of host.shadowRoot.querySelectorAll('.hmt-patch')) patches.push({ page: pageNumber, patchId: patch.dataset.patchId ?? '', complete: patch.complete, naturalWidth: patch.naturalWidth, naturalHeight: patch.naturalHeight })
      for (const region of host.shadowRoot.querySelectorAll('.hmt-region')) {
        if (region.dataset.fit === 'degraded') degradedFitCount += 1
        const fontSizePx = Number.parseFloat(getComputedStyle(region).fontSize)
        const shortSidePx = Math.min(region.clientWidth, region.clientHeight)
        regions.push({ page: pageNumber, regionId: region.dataset.regionId ?? '', text: region.textContent ?? '', sourceEnglish: region.dataset.sourceEnglish ?? '', translatedChinese: region.dataset.translatedChinese ?? '', sourcePreserving: region.classList.contains('hmt-source-notice'), pinyin: region.dataset.pinyin ?? '', hskValid: region.dataset.hskValid ?? '', repairState: region.dataset.hskRepairState ?? '', fit: region.dataset.fit ?? 'normal', fontSizePx, boxWidthPx: region.clientWidth, boxHeightPx: region.clientHeight, fontToBoxShortSide: Number.isFinite(fontSizePx) && shortSidePx > 0 ? fontSizePx / shortSidePx : undefined, overflows: region.scrollWidth > region.clientWidth + 0.5 || region.scrollHeight > region.clientHeight + 0.5 })
      }
    }
    const events = globalThis.__hskifyRuntimeEvidence?.events ?? []
    const firstPatch = events.find((event) => event.type === 'patchDomCommitted')
    const firstText = events.find((event) => event.type === 'selectableTextDomCommitted')
    return { sourceImageCount: document.querySelectorAll('#chapter > img').length, wrappedImageCount: wrapperNodes.length, patchCount: patches.length, regionCount: regions.length, degradedFitCount, wrappers, patches, regions, events, patchBeforeText: Boolean(firstPatch) && Boolean(firstText) && firstPatch.index < firstText.index, observerInstalledAtEpochMs: globalThis.__hskifyRuntimeEvidence?.observerInstalledAtEpochMs }
  })
}

export async function routeEvidence(extensionPage, records, terminalRequired, expectedResourceIdentities) {
  return extensionPage.evaluate(async ({ jobs, sessionKey, expectedIdentities, requireTerminal }) => {
    const stored = await globalThis.browser.storage.session.get(sessionKey)
    const session = stored[sessionKey]
    if (!session || typeof session.token !== 'string' || typeof session.port !== 'number') throw new Error('The extension has no authenticated daemon session.')
    const headers = { Authorization: `Bearer ${session.token}`, 'X-HSK-Manga-Extension-Origin': new URL(globalThis.browser.runtime.getURL('')).origin }
    const request = async (path) => {
      const started = performance.now()
      const response = await fetch(`http://127.0.0.1:${session.port}${path}`, { headers, cache: 'no-store', redirect: 'error' })
      return { response, durationMs: performance.now() - started }
    }
    const healthFetch = await request('/health')
    const health = await healthFetch.response.json()
    if (!healthFetch.response.ok) throw new Error(`GET /health failed: HTTP ${healthFetch.response.status}.`)
    const setupFetch = await request('/setup')
    const setup = await setupFetch.response.json()
    if (!setupFetch.response.ok || setup.state !== 'ready') throw new Error(`Installed resources are not ready: ${JSON.stringify(setup)}.`)
    const canonical = (value) => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value
    const actualIdentities = Array.isArray(health.resourceIdentities) ? health.resourceIdentities : []
    const exact = JSON.stringify(canonical(actualIdentities)) === JSON.stringify(canonical(expectedIdentities))
    const jobsEvidence = []
    for (const job of jobs) {
      const replayFetch = await request(`/jobs/${encodeURIComponent(job.jobId)}/updates?after=0&waitMs=0`)
      const batch = await replayFetch.response.json()
      if (!replayFetch.response.ok || batch.jobId !== job.jobId || !Array.isArray(batch.updates)) throw new Error(`Update replay failed for ${job.jobId}.`)
      const terminal = [...batch.updates].reverse().find((update) => ['complete', 'failed', 'cancelled'].includes(update.type))
      if (requireTerminal && !terminal) throw new Error(`Job ${job.jobId} has no terminal replay update.`)
      const patches = []
      for (const update of batch.updates) {
        if (update.type !== 'regionReady') continue
        const patchId = update.region.patch.blobId
        const patchFetch = await request(`/blobs/${encodeURIComponent(patchId)}`)
        const contentType = patchFetch.response.headers.get('content-type')?.split(';', 1)[0]?.trim() ?? ''
        const bytes = await patchFetch.response.arrayBuffer()
        if (!patchFetch.response.ok || contentType !== 'image/png' || bytes.byteLength === 0) throw new Error(`Patch replay failed for ${patchId}.`)
        const digest = await crypto.subtle.digest('SHA-256', bytes)
        patches.push({ patchId, regionId: update.region.id, route: `/blobs/${patchId}`, httpStatus: patchFetch.response.status, bytes: bytes.byteLength, sha256: [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join(''), rect: update.region.patch.rect, textPolygon: update.region.textPolygon, bubblePolygon: update.region.bubblePolygon })
      }
      jobsEvidence.push({ jobId: job.jobId, pageIndex: job.pageIndex, sourceSha256: job.sourceSha256, sourceWidth: job.sourceWidth, sourceHeight: job.sourceHeight, route: `/jobs/${job.jobId}/updates`, updatesHttpStatus: replayFetch.response.status, updatesDurationMs: replayFetch.durationMs, nextSequence: batch.nextSequence, terminal, updates: batch.updates, patches })
    }
    return {
      session: { buildFingerprint: session.buildFingerprint, engineVersion: session.engineVersion, port: session.port, sessionExpiresAtUnixMs: session.sessionExpiresAtUnixMs, capabilities: session.capabilities, tokenRedacted: true },
      health: { route: '/health', httpStatus: healthFetch.response.status, durationMs: healthFetch.durationMs, body: health },
      resourceIdentityEvidence: { comparison: 'exact canonical projection from the committed model manifest', expected: expectedIdentities, actual: actualIdentities, gates: [{ id: 'runtime-model-resource-identities', status: exact ? 'pass' : 'fail', ...(exact ? {} : { reason: 'Runtime resource identities differ from the committed set.' }) }] },
      setup: { route: '/setup', httpStatus: setupFetch.response.status, durationMs: setupFetch.durationMs, body: setup, downloadsInvoked: false },
      jobs: jobsEvidence,
    }
  }, { jobs: records, sessionKey: SESSION_STORAGE_KEY, expectedIdentities: expectedResourceIdentities, requireTerminal: terminalRequired })
}
