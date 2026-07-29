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
  prepareContentRuntime,
  routeEvidence,
  timedContentStart,
  waitForPageState,
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

export function resolveExpectedImageCount(configuredCount, discoveredCount) {
  if (!Number.isInteger(discoveredCount) || discoveredCount < 1) {
    fail('The live reader exposed no translatable chapter images.')
  }
  if (configuredCount === undefined) return discoveredCount
  if (!Number.isInteger(configuredCount) || configuredCount < 1) {
    fail('expectedImageCount must be a positive integer when provided.')
  }
  return configuredCount
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

async function contentDiscoverySnapshot(extensionPage, pageUrl) {
  return extensionPage.evaluate(async (expectedUrl) => {
    const tabs = await globalThis.browser.tabs.query({})
    const tab = tabs.find((candidate) => candidate.url === expectedUrl)
    if (!Number.isInteger(tab?.id)) {
      throw new Error(`Packaged Firefox has no chapter tab for ${expectedUrl}.`)
    }
    const executions = await globalThis.browser.scripting.executeScript({
      target: { tabId: tab.id, allFrames: false },
      func: () => {
        const discovery = globalThis.__hmtPageController?.discovery
        if (
          !discovery ||
          typeof discovery.current !== 'function' ||
          typeof discovery.deferred !== 'function' ||
          typeof discovery.completionKey !== 'function'
        ) {
          return {
            controllerPresent: false,
            completionKey: '',
            candidates: [],
            deferredCount: 0,
            documentImageCount: document.images.length,
          }
        }
        return {
          controllerPresent: true,
          completionKey: discovery.completionKey(),
          candidates: discovery.current().map((candidate) => ({
            domIndex: candidate.domIndex,
            sourceUrl: new URL(candidate.sourceUrl, location.href).href,
            naturalWidth: candidate.element.naturalWidth,
            naturalHeight: candidate.element.naturalHeight,
            visible: candidate.visible,
          })),
          deferredCount: discovery.deferred().length,
          documentImageCount: document.images.length,
        }
      },
    })
    return executions[0]?.result
  }, pageUrl)
}

async function contentRuntimeDiagnostics(extensionPage, pageUrl) {
  return extensionPage.evaluate(async (expectedUrl) => {
    const tabs = await globalThis.browser.tabs.query({})
    const tab = tabs.find((candidate) => candidate.url === expectedUrl)
    if (!Number.isInteger(tab?.id)) return undefined
    const executions = await globalThis.browser.scripting.executeScript({
      target: { tabId: tab.id, allFrames: false },
      func: () => globalThis.__hmtPageController?.diagnostics(),
    })
    return executions[0]?.result
  }, pageUrl)
}

async function waitForSetupReady(extensionPage, timeoutMs) {
  const startedAtEpochMs = Date.now()
  const deadline = Date.now() + timeoutMs
  const observed = []
  let status = await extensionMessage(extensionPage, { type: 'setup:status' })
  observed.push(status)
  if (status.state !== 'ready') {
    status = await extensionMessage(extensionPage, { type: 'setup:start' })
    observed.push(status)
  }
  while (Date.now() < deadline) {
    if (status.state === 'ready') {
      return {
        startedAtEpochMs,
        readyAtEpochMs: Date.now(),
        durationMs: Date.now() - startedAtEpochMs,
        observed,
      }
    }
    if (status.state === 'failed') {
      fail(`Hskify setup reached ${status.state}: ${JSON.stringify(status)}`)
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250))
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
  fail(`Timed out waiting for Hskify setup readiness: ${JSON.stringify(status)}`)
}

async function primeLazyImages(page, extensionPage, pageUrl, timeoutMs) {
  const startedAt = Date.now()
  let bottomPasses = 0
  let steps = 0
  let stableSince = Date.now()
  let previousCompletionKey = ''
  let snapshot
  await page.evaluate(() => scrollTo({ top: 0, behavior: 'instant' }))
  while (Date.now() - startedAt < timeoutMs) {
    snapshot = await contentDiscoverySnapshot(extensionPage, pageUrl)
    if (snapshot?.completionKey !== previousCompletionKey) {
      previousCompletionKey = snapshot?.completionKey ?? ''
      stableSince = Date.now()
    }
    if (!snapshot?.controllerPresent || snapshot.candidates.length + snapshot.deferredCount === 0) {
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 250))
      continue
    }
    const scroll = await page.evaluate(() => {
      const maximum = Math.max(0, document.documentElement.scrollHeight - innerHeight)
      scrollTo({
        top: Math.min(maximum, scrollY + Math.max(640, innerHeight * 1.5)),
        behavior: 'instant',
      })
      return {
        atBottom: scrollY >= maximum - 1,
      }
    })
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100))
    steps += 1
    if (scroll.atBottom) bottomPasses += 1
    else bottomPasses = 0
    if (
      bottomPasses >= 5 &&
      snapshot.deferredCount === 0 &&
      Date.now() - stableSince >= 1_500
    ) {
      break
    }
  }
  snapshot = await contentDiscoverySnapshot(extensionPage, pageUrl)
  await page.evaluate(
    () =>
      new Promise((resolvePromise) => {
        scrollTo({ top: 0, behavior: 'instant' })
        requestAnimationFrame(() => requestAnimationFrame(resolvePromise))
      }),
  )
  return {
    durationMs: Date.now() - startedAt,
    steps,
    reachedBottom: bottomPasses >= 5,
    controllerPresent: snapshot?.controllerPresent === true,
    documentImageCount: snapshot?.documentImageCount ?? 0,
    discoveredCount: snapshot?.candidates.length ?? 0,
    deferredCount: snapshot?.deferredCount ?? 0,
  }
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

export function buildLiveTranslationProof(dom, routes) {
  for (const job of routes.jobs ?? []) {
    for (const update of job.updates ?? []) {
      const displayedChinese = update.region?.displayedChinese
      const hsk = update.region?.hsk
      const strictlyValid = hsk?.strictlyValid === true
      const target = hsk?.requestedLevel <= 3 ? 0.9 : hsk?.requestedLevel === 4 ? 0.93 : 0.95
      const naturalAccepted =
        hsk?.learningMode === 'natural' &&
        hsk?.repairState !== 'pending' &&
        hsk?.repairState !== 'rejected' &&
        Number.isFinite(hsk?.levelCoverage) &&
        hsk.levelCoverage >= target
      if (
        update.type !== 'regionReady' ||
        !/[A-Za-z]/u.test(update.region?.sourceEnglish ?? '') ||
        !displayedChinese?.trim() ||
        displayedChinese.trim() === update.region.sourceEnglish.trim() ||
        (!strictlyValid && !naturalAccepted)
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
            item.hskValid === String(strictlyValid),
        )
        const patchEvent = dom.events.find(
          (event) =>
            event.type === 'patchDomCommitted' && event.patchId === patchId,
        )
        const textEvent = dom.events.find(
          (event) =>
            event.type === 'selectableTextDomCommitted' &&
            event.regionId === update.region.id,
        )
        if (
          !patch ||
          !region ||
          !patchEvent ||
          !textEvent ||
          patchEvent.index >= textEvent.index
        ) {
          continue
        }
        return {
          passed: true,
          jobId: job.jobId,
          pageIndex: job.pageIndex,
          regionId: update.region.id,
          sourceEnglish: update.region.sourceEnglish,
          displayedChinese,
          hskStrictlyValid: strictlyValid,
          hskAssessment: {
            repairState: hsk?.repairState,
            levelCoverage: hsk?.levelCoverage,
            aboveLevelTokens: hsk?.aboveLevelTokens ?? [],
            teachingTerms: hsk?.teachingTerms ?? [],
          },
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
      'No finalized HSK-policy-accepted Latin-English region had a decoded patch committed before selectable Chinese text.',
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
  let fallback
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
        if (proof.passed) {
          const evidence = { proof, dom, routes, records }
          if (proof.hskStrictlyValid) return evidence
          fallback = evidence
        }
      }
    }
    lastState = await extensionMessage(extensionPage, { type: 'popup:state' })
    if (lastState.state === 'complete' && fallback) return fallback
    if (lastState.state === 'failed' || lastState.state === 'cancelled') {
      const diagnostics = await contentRuntimeDiagnostics(extensionPage, pageUrl)
      fail(
        `Live translation reached ${lastState.state}: ${lastState.message}; diagnostics: ${JSON.stringify(diagnostics)}`,
      )
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
  let context
  let chapterPage
  let network
  let identity
  let finalChapterUrl = requestedChapterUrl
  let readerDiagnostics
  let discoveredImages = []
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
    const activation = await prepareContentRuntime(extensionPage, finalChapterUrl)
    const setup = await waitForSetupReady(
      extensionPage,
      Math.min(config.runTimeoutMs, 5 * 60_000),
    )
    const lazyLoad = await primeLazyImages(
      chapterPage,
      extensionPage,
      finalChapterUrl,
      Math.min(config.runTimeoutMs, 60_000),
    )
    readerDiagnostics = lazyLoad
    await installDomObserver(chapterPage, 'live-asura-smoke')
    const discovery = (await contentDiscoverySnapshot(extensionPage, finalChapterUrl)).candidates
    discoveredImages = discovery
    const expectedImageCount = resolveExpectedImageCount(
      config.expectedImageCount,
      discovery.length,
    )
    network.setPhase('extension-translation')
    const action = await timedContentStart(extensionPage, config.hskLevel, finalChapterUrl)
    const permissions = {
      installationScope: ['http://*/*', 'https://*/*'],
      granted: true,
    }
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
    const finalState = await waitForPageState(
      extensionPage,
      chapterPage,
      ['complete', 'failed', 'cancelled'],
      config.runTimeoutMs,
    )
    if (finalState.state !== 'complete') {
      const diagnostics = await contentRuntimeDiagnostics(
        extensionPage,
        finalChapterUrl,
      )
      fail(
        `Live translation reached ${finalState.state}: ${finalState.message}; diagnostics: ${JSON.stringify(diagnostics)}`,
      )
    }
    if (finalState.current !== expectedImageCount || finalState.total !== expectedImageCount) {
      fail(
        `Live translation completed with ${finalState.current}/${finalState.total} images; expected ${expectedImageCount}/${expectedImageCount}.`,
      )
    }
    const finalDom = await chapterDomEvidence(chapterPage)
    const finalRoutes = await routeEvidence(
      extensionPage,
      translated.records,
      false,
      config.expectedResourceIdentities,
    )
    const finalProof = buildLiveTranslationProof(finalDom, finalRoutes)
    if (!finalProof.passed) {
      fail(`Live translation completed without final HSK-policy proof: ${finalProof.reason}`)
    }
    const finalized = {
      ...translated,
      proof: finalProof,
      dom: finalDom,
      routes: finalRoutes,
    }
    const capturedNetwork = network.snapshot()
    network = undefined
    const timings = timingSections(
      capturedNetwork,
      discovery,
      requestedChapterUrl,
      finalChapterUrl,
    )
    const proofPatch = finalized.dom.events.find(
      (event) => event.index === finalized.proof.domOrdering.patchEventIndex,
    )
    const proofText = finalized.dom.events.find(
      (event) => event.index === finalized.proof.domOrdering.selectableTextEventIndex,
    )
    const firstPatch = finalized.dom.events.find(
      (event) =>
        event.epochMs >= action.issuedAtEpochMs && event.type === 'patchDomCommitted',
    )
    const firstText = finalized.dom.events.find(
      (event) =>
        event.epochMs >= action.issuedAtEpochMs &&
        event.type === 'selectableTextDomCommitted',
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
      readerDiagnostics,
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
      activation: {
        method: 'packaged-extension-explicit-chapter-tab',
        tabId: activation.tabId,
        initialState: activation.state,
      },
      setup,
      permissions,
      translationProof: finalized.proof,
      extensionWorkflow: {
        actionIssuedAtEpochMs: action.issuedAtEpochMs,
        actionResponseLatencyMs: action.responseAtEpochMs - action.issuedAtEpochMs,
        firstPatchAfterActionMs:
          firstPatch === undefined ? undefined : firstPatch.epochMs - action.issuedAtEpochMs,
        firstSelectableTextAfterActionMs:
          firstText === undefined ? undefined : firstText.epochMs - action.issuedAtEpochMs,
        proofPatchAfterActionMs:
          proofPatch === undefined
            ? undefined
            : proofPatch.epochMs - action.issuedAtEpochMs,
        proofSelectableTextAfterActionMs:
          proofText === undefined ? undefined : proofText.epochMs - action.issuedAtEpochMs,
        stateAtProof,
        finalState,
        note:
          'First-patch milestones measure the first DOM commit of any translated region. Proof milestones identify the separate policy-audited dialogue region. These end-to-end milestones can include live image acquisition and are not local-only benchmark timings.',
      },
      timings,
      localRouteReplay: finalized.routes,
      gates: [
        {
          id: 'all-live-chapter-images-discovered',
          status: 'pass',
          actual: action.value.total,
          expected: expectedImageCount,
        },
        {
          id: 'english-dialogue-region-translated',
          status: 'pass',
          regionId: finalized.proof.regionId,
        },
        {
          id: 'hsk-assessment-recorded',
          status: 'pass',
          strictlyValid: finalized.proof.hskStrictlyValid,
          ...finalized.proof.hskAssessment,
        },
        {
          id: 'decoded-patch-before-selectable-text',
          status: 'pass',
          ...finalized.proof.domOrdering,
        },
        {
          id: 'entire-live-chapter-completed',
          status: 'pass',
          actual: {
            state: finalState.state,
            current: finalState.current,
            total: finalState.total,
          },
          expected: {
            state: 'complete',
            current: expectedImageCount,
            total: expectedImageCount,
          },
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
      readerDiagnostics,
      timings: timingSections(
        capturedNetwork,
        discoveredImages,
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
