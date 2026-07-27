import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import {
  activeJobs,
  chapterDomEvidence,
  extensionMessage,
  installDomObserver,
  launchPackagedFirefox,
  nowIso,
  routeEvidence,
  timedContentStart,
  validateBenchmarkManifest,
  writeJsonSync,
} from './Chapter5.Firefox.mjs'

const EVIDENCE_FILE = 'live-asura-smoke.json'

function fail(message) {
  throw new Error(message)
}

function finiteTiming(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

function publicUrl(value) {
  const url = new URL(value)
  url.username = ''
  url.password = ''
  url.hash = ''
  return url.href
}

export function validateLiveChapterUrl(value) {
  let url
  try {
    url = new URL(value)
  } catch {
    fail('The live smoke requires an explicit absolute chapter URL.')
  }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
    fail('The live chapter URL must be credential-free HTTP or HTTPS.')
  }
  url.hash = ''
  return url.href
}

function isLoopback(value) {
  try {
    const hostname = new URL(value).hostname.toLowerCase()
    return hostname === '127.0.0.1' || hostname === 'localhost' || hostname === '::1'
  } catch {
    return false
  }
}

function installNetworkTimingCapture(context) {
  const pending = new Map()
  const records = []
  let phase = 'startup'
  let nextId = 1
  const onRequest = (request) => {
    if (!/^https?:/iu.test(request.url())) return
    let frameUrl = ''
    try {
      frameUrl = request.frame().url()
    } catch {
      // Background extension fetches intentionally have no document frame.
    }
    pending.set(request, {
      id: nextId++,
      phase,
      url: publicUrl(request.url()),
      method: request.method(),
      resourceType: request.resourceType(),
      frameUrl: frameUrl && /^https?:/iu.test(frameUrl) ? publicUrl(frameUrl) : '',
      observedStartEpochMs: Date.now(),
    })
  }
  const onResponse = (response) => {
    const record = pending.get(response.request())
    if (!record) return
    record.status = response.status()
    record.fromServiceWorker = response.fromServiceWorker()
  }
  const finish = (request, failure) => {
    const record = pending.get(request)
    if (!record) return
    pending.delete(request)
    const timing = request.timing()
    records.push({
      ...record,
      ...(failure ? { failure } : {}),
      timing: {
        startTimeEpochMs: finiteTiming(timing.startTime),
        domainLookupMs:
          finiteTiming(timing.domainLookupEnd) === undefined ||
          finiteTiming(timing.domainLookupStart) === undefined
            ? undefined
            : timing.domainLookupEnd - timing.domainLookupStart,
        connectMs:
          finiteTiming(timing.connectEnd) === undefined ||
          finiteTiming(timing.connectStart) === undefined
            ? undefined
            : timing.connectEnd - timing.connectStart,
        tlsMs:
          finiteTiming(timing.secureConnectionStart) === undefined ||
          finiteTiming(timing.connectEnd) === undefined
            ? undefined
            : timing.connectEnd - timing.secureConnectionStart,
        requestToResponseStartMs:
          finiteTiming(timing.responseStart) === undefined ||
          finiteTiming(timing.requestStart) === undefined
            ? undefined
            : timing.responseStart - timing.requestStart,
        responseTransferMs:
          finiteTiming(timing.responseEnd) === undefined ||
          finiteTiming(timing.responseStart) === undefined
            ? undefined
            : timing.responseEnd - timing.responseStart,
        totalMs: finiteTiming(timing.responseEnd),
      },
    })
  }
  const onFinished = (request) => finish(request)
  const onFailed = (request) => finish(request, request.failure()?.errorText ?? 'request failed')
  context.on('request', onRequest)
  context.on('response', onResponse)
  context.on('requestfinished', onFinished)
  context.on('requestfailed', onFailed)
  return {
    setPhase(value) {
      phase = value
    },
    snapshot() {
      context.off('request', onRequest)
      context.off('response', onResponse)
      context.off('requestfinished', onFinished)
      context.off('requestfailed', onFailed)
      for (const [request] of pending) finish(request, 'request still pending when evidence closed')
      return records.sort((left, right) => left.id - right.id)
    },
  }
}

async function requestOptionalOrigins(extensionPage, origins, timeoutMs) {
  const requested = [...new Set(origins)].sort()
  if (requested.length === 0) return { requested, missing: [], granted: true }
  const missing = await extensionPage.evaluate(async (patterns) => {
    const unresolved = []
    for (const origin of patterns) {
      if (!(await globalThis.browser.permissions.contains({ origins: [origin] }))) {
        unresolved.push(origin)
      }
    }
    return unresolved
  }, requested)
  if (missing.length === 0) return { requested, missing, granted: true }
  const buttonId = '__hskify_live_permission_request'
  await extensionPage.evaluate(
    ({ id, patterns }) => {
      document.getElementById(id)?.remove()
      const button = document.createElement('button')
      button.id = id
      button.type = 'button'
      button.textContent = 'Grant live image access'
      globalThis.__hskifyLivePermissionResult = undefined
      button.addEventListener(
        'click',
        () => {
          void globalThis.browser.permissions
            .request({ origins: patterns })
            .then((granted) => {
              globalThis.__hskifyLivePermissionResult = { granted }
            })
            .catch((error) => {
              globalThis.__hskifyLivePermissionResult = {
                granted: false,
                error: error instanceof Error ? error.message : String(error),
              }
            })
        },
        { once: true },
      )
      document.body.append(button)
    },
    { id: buttonId, patterns: missing },
  )
  await extensionPage.locator(`#${buttonId}`).click()
  const handle = await extensionPage.waitForFunction(
    () => globalThis.__hskifyLivePermissionResult !== undefined,
    undefined,
    { timeout: timeoutMs },
  )
  const result = await handle.jsonValue()
  await extensionPage.evaluate((id) => {
    document.getElementById(id)?.remove()
    delete globalThis.__hskifyLivePermissionResult
  }, buttonId)
  if (!result?.granted) {
    fail(`Firefox did not grant live image origins: ${result?.error ?? missing.join(', ')}`)
  }
  return { requested, missing, granted: true }
}

async function primeLazyImages(page, timeoutMs) {
  return page.evaluate(async (maximumMs) => {
    const started = performance.now()
    let bottomPasses = 0
    let steps = 0
    scrollTo({ top: 0, behavior: 'instant' })
    while (performance.now() - started < maximumMs && bottomPasses < 3) {
      const before = scrollY
      const maximum = Math.max(0, document.documentElement.scrollHeight - innerHeight)
      scrollTo({ top: Math.min(maximum, before + Math.max(320, innerHeight * 0.8)), behavior: 'instant' })
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 100))
      steps += 1
      if (scrollY >= maximum - 1) bottomPasses += 1
      else bottomPasses = 0
    }
    const reachedBottom = bottomPasses >= 3
    scrollTo({ top: 0, behavior: 'instant' })
    await new Promise((resolvePromise) =>
      requestAnimationFrame(() => requestAnimationFrame(resolvePromise)),
    )
    return {
      durationMs: performance.now() - started,
      steps,
      reachedBottom,
      documentImageCount: document.images.length,
    }
  }, timeoutMs)
}

async function navigationTiming(page) {
  return page.evaluate(() => {
    const entry = performance.getEntriesByType('navigation')[0]
    if (!entry) return undefined
    return {
      type: entry.type,
      redirectCount: entry.redirectCount,
      redirectMs: entry.redirectEnd - entry.redirectStart,
      dnsMs: entry.domainLookupEnd - entry.domainLookupStart,
      connectMs: entry.connectEnd - entry.connectStart,
      tlsMs:
        entry.secureConnectionStart > 0
          ? entry.connectEnd - entry.secureConnectionStart
          : undefined,
      requestToResponseStartMs: entry.responseStart - entry.requestStart,
      responseTransferMs: entry.responseEnd - entry.responseStart,
      domInteractiveMs: entry.domInteractive,
      domContentLoadedMs: entry.domContentLoadedEventEnd,
      loadEventMs: entry.loadEventEnd,
      durationMs: entry.duration,
      transferSize: entry.transferSize,
      encodedBodySize: entry.encodedBodySize,
      decodedBodySize: entry.decodedBodySize,
    }
  })
}

async function discoverySnapshot(page) {
  return page.evaluate(() => {
    const discovery = globalThis.__hmtPageController?.discovery
    if (!discovery || typeof discovery.current !== 'function') return []
    return discovery.current().map((candidate) => ({
      domIndex: candidate.domIndex,
      sourceUrl: new URL(candidate.sourceUrl, location.href).href,
      naturalWidth: candidate.element.naturalWidth,
      naturalHeight: candidate.element.naturalHeight,
      visible: candidate.visible,
    }))
  })
}

export function buildLiveTranslationProof(dom, routes) {
  for (const job of routes.jobs ?? []) {
    for (const update of job.updates ?? []) {
      if (
        update.type !== 'regionReady' ||
        !/[A-Za-z]/u.test(update.region?.sourceEnglish ?? '') ||
        !update.region?.displayedChinese?.trim() ||
        update.region?.hsk?.strictlyValid !== true
      ) {
        continue
      }
      const patchId = update.region.patch?.blobId
      const patch = dom.patches.find(
        (item) =>
          item.patchId === patchId &&
          item.complete === true &&
          item.naturalWidth > 0 &&
          item.naturalHeight > 0,
      )
      const region = dom.regions.find(
        (item) =>
          item.regionId === update.region.id &&
          item.text.trim() &&
          item.hskValid === 'true',
      )
      const patchEvent = dom.events.find(
        (event) => event.type === 'patchDomCommitted' && event.patchId === patchId,
      )
      const textEvent = dom.events.find(
        (event) =>
          event.type === 'selectableTextDomCommitted' &&
          event.regionId === update.region.id,
      )
      if (!patch || !region || !patchEvent || !textEvent || patchEvent.index >= textEvent.index) {
        continue
      }
      return {
        passed: true,
        jobId: job.jobId,
        pageIndex: job.pageIndex,
        regionId: update.region.id,
        sourceEnglish: update.region.sourceEnglish,
        displayedChinese: update.region.displayedChinese,
        hskStrictlyValid: true,
        patch: {
          patchId,
          mimeType: update.region.patch.mimeType,
          decodedWidth: patch.naturalWidth,
          decodedHeight: patch.naturalHeight,
        },
        domOrdering: {
          patchEventIndex: patchEvent.index,
          selectableTextEventIndex: textEvent.index,
          patchBeforeText: true,
        },
      }
    }
  }
  return {
    passed: false,
    reason:
      'No Latin-English, strict-HSK-valid translated region had a decoded patch committed before selectable Chinese text.',
  }
}

async function waitForTranslationProof(
  chapterPage,
  extensionPage,
  pageUrl,
  expectedResourceIdentities,
  timeoutMs,
) {
  const deadline = Date.now() + timeoutMs
  let lastState
  while (Date.now() < deadline) {
    const dom = await chapterDomEvidence(chapterPage)
    if (dom.regionCount > 0) {
      const records = await activeJobs(extensionPage, pageUrl)
      if (records.length > 0) {
        const routes = await routeEvidence(
          extensionPage,
          records,
          false,
          expectedResourceIdentities,
        )
        const proof = buildLiveTranslationProof(dom, routes)
        if (proof.passed) return { proof, dom, routes, records }
      }
    }
    lastState = await extensionMessage(extensionPage, { type: 'popup:state' })
    if (lastState.state === 'failed' || lastState.state === 'cancelled') {
      fail(`Live translation reached ${lastState.state}: ${lastState.message}`)
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250))
  }
  fail(`Timed out waiting for one valid live translation region (last state: ${JSON.stringify(lastState)}).`)
}

function timingSections(records, discovery, requestedUrl, finalUrl) {
  const discoveredUrls = new Set(discovery.map((item) => publicUrl(item.sourceUrl)))
  const remote = records.filter((record) => !isLoopback(record.url))
  const loopback = records.filter((record) => isLoopback(record.url))
  return {
    liveNetwork: {
      aggregationPolicy:
        'Live remote timings are smoke evidence only and are forbidden from deterministic local-replica percentiles or gates.',
      navigationRequests: remote.filter(
        (record) =>
          record.resourceType === 'document' ||
          record.url === publicUrl(requestedUrl) ||
          record.url === publicUrl(finalUrl),
      ),
      chapterImageAcquisitionRequests: remote.filter(
        (record) => discoveredUrls.has(record.url),
      ),
      otherRemoteRequestCount: remote.filter(
        (record) =>
          record.resourceType !== 'document' && !discoveredUrls.has(record.url),
      ).length,
    },
    localExtensionDaemon: {
      scope:
        'Loopback HTTP timing only; excludes live page/CDN requests and is not a deterministic benchmark sample.',
      requests: loopback,
    },
  }
}

async function main() {
  const configPath = process.argv[2]
  if (!configPath) {
    fail('Usage: node Chapter5.LiveAsura.Firefox.mjs <live-config.json>')
  }
  const config = JSON.parse(readFileSync(configPath, 'utf8'))
  const requestedChapterUrl = validateLiveChapterUrl(config.chapterUrl)
  const manifest = JSON.parse(readFileSync(config.manifestPath, 'utf8'))
  validateBenchmarkManifest(manifest)
  const expectedImageCount = manifest.pageCount
  let context
  let chapterPage
  let network
  let identity
  let finalChapterUrl = requestedChapterUrl
  let evidence
  let failure
  const runStartedAtEpochMs = Date.now()
  try {
    const launched = await launchPackagedFirefox(config)
    context = launched.context
    identity = launched.identity
    const extensionPage = launched.extensionPage
    network = installNetworkTimingCapture(context)
    chapterPage = await context.newPage()
    await chapterPage.setViewportSize({ width: 1280, height: 900 })
    network.setPhase('page-navigation')
    const navigationStartedAtEpochMs = Date.now()
    const response = await chapterPage.goto(requestedChapterUrl, {
      waitUntil: 'domcontentloaded',
      timeout: config.runTimeoutMs,
    })
    const navigationEndedAtEpochMs = Date.now()
    finalChapterUrl = chapterPage.url()
    network.setPhase('page-image-load')
    const lazyLoad = await primeLazyImages(chapterPage, Math.min(config.runTimeoutMs, 60_000))
    await chapterPage.bringToFront()
    const permissionPlan = await extensionMessage(extensionPage, { type: 'popup:prepare' })
    const discovery = await discoverySnapshot(chapterPage)
    const permissions = await requestOptionalOrigins(
      extensionPage,
      permissionPlan.allOrigins,
      Math.min(config.runTimeoutMs, 30_000),
    )
    await installDomObserver(chapterPage, 'live-asura-smoke')
    network.setPhase('extension-translation')
    await chapterPage.bringToFront()
    const action = await timedContentStart(
      extensionPage,
      config.hskLevel,
      finalChapterUrl,
    )
    if (action.value.total !== expectedImageCount || discovery.length !== expectedImageCount) {
      fail(
        `Expected exactly ${expectedImageCount} discovered chapter images; content start reported ${action.value.total} and discovery inspection found ${discovery.length}.`,
      )
    }
    const translated = await waitForTranslationProof(
      chapterPage,
      extensionPage,
      finalChapterUrl,
      config.expectedResourceIdentities,
      config.runTimeoutMs,
    )
    const stateAtProof = await extensionMessage(extensionPage, { type: 'popup:state' })
    const cancelledAfterProof = await extensionMessage(extensionPage, { type: 'popup:cancel' })
    const capturedNetwork = network.snapshot()
    network = undefined
    const timings = timingSections(
      capturedNetwork,
      discovery,
      requestedChapterUrl,
      finalChapterUrl,
    )
    const firstPatch = translated.dom.events.find(
      (event) => event.index === translated.proof.domOrdering.patchEventIndex,
    )
    const firstText = translated.dom.events.find(
      (event) => event.index === translated.proof.domOrdering.selectableTextEventIndex,
    )
    evidence = {
      schemaVersion: 1,
      evidenceKind: 'live-asura-packaged-firefox-smoke',
      status: 'pass',
      recordedAtUtc: nowIso(),
      deterministicLocalReplicaAggregationEligible: false,
      requestedChapterUrl,
      finalChapterUrl,
      redirectObserved: finalChapterUrl !== requestedChapterUrl,
      extensionIdentity: identity,
      navigation: {
        httpStatus: response?.status(),
        wallDurationMs: navigationEndedAtEpochMs - navigationStartedAtEpochMs,
        browserTiming: await navigationTiming(chapterPage),
        lazyImagePriming: lazyLoad,
      },
      discovery: {
        expectedImageCount,
        contentStartImageCount: action.value.total,
        inspectedImageCount: discovery.length,
        images: discovery,
      },
      permissions,
      translationProof: translated.proof,
      extensionWorkflow: {
        actionIssuedAtEpochMs: action.issuedAtEpochMs,
        actionResponseLatencyMs: action.responseAtEpochMs - action.issuedAtEpochMs,
        firstPatchAfterActionMs:
          firstPatch === undefined ? undefined : firstPatch.epochMs - action.issuedAtEpochMs,
        firstSelectableTextAfterActionMs:
          firstText === undefined ? undefined : firstText.epochMs - action.issuedAtEpochMs,
        stateAtProof,
        cancelledAfterProof,
        note:
          'These end-to-end milestones can include live image acquisition; they are not local-only benchmark timings.',
      },
      timings,
      localRouteReplay: translated.routes,
      gates: [
        {
          id: 'ten-live-chapter-images-discovered',
          status: 'pass',
          actual: action.value.total,
          expected: expectedImageCount,
        },
        {
          id: 'valid-english-dialogue-region-translated',
          status: 'pass',
          regionId: translated.proof.regionId,
        },
        {
          id: 'decoded-patch-before-selectable-text',
          status: 'pass',
          ...translated.proof.domOrdering,
        },
      ],
    }
  } catch (error) {
    failure = error
    const capturedNetwork = network?.snapshot() ?? []
    network = undefined
    const observedUrl = chapterPage?.url()
    const failedFinalUrl =
      observedUrl && /^https?:/iu.test(observedUrl) ? observedUrl : finalChapterUrl
    evidence = {
      schemaVersion: 1,
      evidenceKind: 'live-asura-packaged-firefox-smoke',
      status: 'fail',
      recordedAtUtc: nowIso(),
      deterministicLocalReplicaAggregationEligible: false,
      requestedChapterUrl,
      finalChapterUrl: failedFinalUrl,
      extensionIdentity: identity,
      timings: timingSections(
        capturedNetwork,
        [],
        requestedChapterUrl,
        failedFinalUrl,
      ),
      failure: {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : '',
      },
    }
  } finally {
    network?.snapshot()
    await context?.close().catch(() => undefined)
  }
  evidence.runStartedAtUtc = new Date(runStartedAtEpochMs).toISOString()
  evidence.runEndedAtUtc = nowIso()
  writeJsonSync(join(config.outputDirectory, EVIDENCE_FILE), evidence)
  if (failure) throw failure
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : ''
if (import.meta.url === invokedPath) await main()
