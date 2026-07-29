import { sha256Hex } from '../acquisition/hash'
import type {
  BrowserJobRequest,
  JobUpdateBatch,
  JobUpdate,
  LearningMode,
  LookupRequest,
  NameTranslation,
} from '../contracts/browser'
import {
  deferredImageSourceUrl,
  ImageDiscovery,
  isRectVisible,
  visibleFirst,
  type DiscoveredImage,
  type DiscoveryEvent,
} from '../discovery/images'
import { VisibleFirstQueue, type QueueItem } from '../discovery/queue'
import {
  parseContentRequest,
  sendBackgroundMessage,
  RuntimeMessageError,
  type PageState,
  type RecoveredJob,
  type RecoveryCandidate,
  type TranslationScope,
} from '../messaging/messages'
import { ImageStatusBadge, PageHud } from '../progress/hud'
import { visibleImageRects } from '../rendering/geometry'
import { SelectableRenderer, type RenderedImage } from '../rendering/renderer'
import { ChapterRunState } from './run-state'
import { ChapterContextLedger } from './chapter-context'

const PAGE_SESSION_KEY = 'hmt.pageSessionId'
const CONTENT_BYTE_LIMIT = 25 * 1024 * 1024
const NAVIGATION_CHECK_INTERVAL_MS = 250
const VIEWPORT_THROTTLE_MS = 100
export const AUTOMATIC_IMAGE_RETRY_LIMIT = 2
export const COMPLETION_SETTLE_MS = 300

type TranslationCandidate = {
  candidate: DiscoveredImage
  recovered?: RecoveredJob
}

type ImageFailureDiagnostic = {
  sourceUrl: string
  code: string
  message: string
  retryable: boolean
}

type SourceSnapshot = {
  generation: number
  pageSessionId: string
  navigationUrl: string
  sourceUrl: string
  naturalWidth: number
  naturalHeight: number
}

function abortError(): Error {
  const error = new Error('The operation was cancelled.')
  error.name = 'AbortError'
  return error
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortError()
}

function createPageSessionId(reuseStored: boolean): string {
  const pageKey = `${PAGE_SESSION_KEY}:${location.href}`
  const existing = reuseStored ? sessionStorage.getItem(pageKey) : null
  if (existing) return existing
  const generated =
    typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  sessionStorage.setItem(pageKey, generated)
  return generated
}

function normalizedSourceUrl(value: string): string {
  const url = new URL(value, location.href)
  url.hash = ''
  return url.href
}

function currentSourceUrl(image: HTMLImageElement): string {
  return normalizedSourceUrl(image.currentSrc || image.src)
}

function candidateKey(candidate: DiscoveredImage): string {
  return `${candidate.domIndex}:${normalizedSourceUrl(candidate.sourceUrl)}:${candidate.element.naturalWidth}x${candidate.element.naturalHeight}`
}

function recoveryKey(sourceUrl: string, width: number, height: number, pageIndex: number): string {
  return `${normalizedSourceUrl(sourceUrl)}:${width}x${height}:${pageIndex}`
}

function sourceMimeType(sourceUrl: string): string | undefined {
  if (!sourceUrl.startsWith('data:')) return undefined
  const match = /^data:([^;,]+)/i.exec(sourceUrl)
  return match?.[1]?.toLowerCase()
}

async function readBoundedBody(
  response: Response,
  maximumBytes: number,
): Promise<ArrayBuffer | undefined> {
  const contentLength = response.headers.get('content-length')
  if (contentLength !== null) {
    const parsed = Number(contentLength)
    if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > maximumBytes) return undefined
  }
  if (!response.body) {
    const bytes = await response.arrayBuffer()
    return bytes.byteLength <= maximumBytes ? bytes : undefined
  }
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    while (true) {
      const item = await reader.read()
      if (item.done) break
      total += item.value.byteLength
      if (total > maximumBytes) {
        await reader.cancel()
        return undefined
      }
      chunks.push(item.value)
    }
  } finally {
    reader.releaseLock()
  }
  const merged = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.byteLength
  }
  return merged.buffer
}

export async function tryContentBytes(candidate: DiscoveredImage): Promise<
  | {
      bytes: ArrayBuffer
      mimeType?: string
    }
  | undefined
> {
  let source: URL
  try {
    source = new URL(candidate.sourceUrl, location.href)
  } catch {
    return undefined
  }
  if (
    source.protocol !== 'data:' &&
    source.protocol !== 'blob:' &&
    source.origin !== location.origin
  ) {
    return undefined
  }
  try {
    const response = await fetch(candidate.sourceUrl, {
      credentials: source.origin === location.origin ? 'include' : 'omit',
      cache: 'no-store',
    })
    if (!response.ok) return undefined
    const bytes = await readBoundedBody(response, CONTENT_BYTE_LIMIT)
    if (!bytes) return undefined
    const mimeType =
      response.headers.get('content-type')?.split(';', 1)[0]?.trim() ||
      sourceMimeType(candidate.sourceUrl)
    return {
      bytes,
      ...(mimeType ? { mimeType } : {}),
    }
  } catch {
    return undefined
  }
}

class ViewportReporter {
  private timer: number | undefined
  private lastPayload = ''
  private stopped = false
  private inFlight: Promise<void> = Promise.resolve()
  private readonly resizeObserver: ResizeObserver | undefined

  constructor(
    private readonly jobId: string,
    private readonly image: HTMLImageElement,
    private readonly sourceWidth: number,
    private readonly sourceHeight: number,
  ) {
    addEventListener('scroll', this.schedule, true)
    addEventListener('resize', this.schedule)
    document.addEventListener('visibilitychange', this.schedule)
    this.resizeObserver =
      typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(this.schedule)
    this.resizeObserver?.observe(image)
    this.send(true, true)
  }

  private readonly schedule = (): void => {
    if (this.stopped || this.timer !== undefined) return
    this.timer = window.setTimeout(() => {
      this.timer = undefined
      this.send(true)
    }, VIEWPORT_THROTTLE_MS)
  }

  private send(active: boolean, force = false): void {
    const visibleRects = visibleImageRects(this.image, this.sourceWidth, this.sourceHeight)
    const payloadKey = JSON.stringify({ visibleRects, active })
    if (!force && payloadKey === this.lastPayload) return
    this.lastPayload = payloadKey
    this.inFlight = this.inFlight
      .catch(() => undefined)
      .then(() =>
        sendBackgroundMessage({
          type: 'job:viewport',
          jobId: this.jobId,
          visibleRects,
          active,
        }),
      )
  }

  async stop(): Promise<void> {
    if (this.stopped) return
    this.stopped = true
    if (this.timer !== undefined) window.clearTimeout(this.timer)
    this.timer = undefined
    removeEventListener('scroll', this.schedule, true)
    removeEventListener('resize', this.schedule)
    document.removeEventListener('visibilitychange', this.schedule)
    this.resizeObserver?.disconnect()
    this.send(false, true)
    await this.inFlight
  }
}

function errorMessage(error: unknown): string {
  if (error instanceof RuntimeMessageError && error.code === 'IMAGE_PERMISSION_DENIED') {
    return 'Allow image access, then try again.'
  }
  return 'This image couldn’t be translated. Try again.'
}

export function shouldAutomaticallyRetryImage(error: unknown, attempts: number): boolean {
  return (
    error instanceof RuntimeMessageError &&
    error.retryable &&
    attempts < AUTOMATIC_IMAGE_RETRY_LIMIT
  )
}

export class PageTranslationController {
  private sessionId = createPageSessionId(true)
  private navigationUrl = location.href
  private generation = 0
  private readonly discovery: ImageDiscovery
  private readonly renderer: SelectableRenderer
  private readonly queue: VisibleFirstQueue<TranslationCandidate>
  private readonly rendered = new Map<HTMLImageElement, RenderedImage>()
  private readonly badges = new Map<HTMLImageElement, ImageStatusBadge>()
  private readonly queueIds = new Map<HTMLImageElement, string>()
  private readonly processed = new Set<HTMLImageElement>()
  private readonly runState = new ChapterRunState<HTMLImageElement>()
  private readonly failures = new Map<HTMLImageElement, ImageFailureDiagnostic>()
  private readonly context = new ChapterContextLedger()
  private readonly pageIndexByImage = new Map<HTMLImageElement, number>()
  private properNameGlossary: NonNullable<BrowserJobRequest['properNameGlossary']> = []
  private readonly navigationTimer: number
  private hud: PageHud | undefined
  private scope: TranslationScope | undefined
  private hskLevel: 1 | 2 | 3 | 4 | 5 | 6 = 5
  private learningMode: LearningMode = 'natural'
  private nameTranslation: NameTranslation = 'keep-original'
  private readonly activeJobIds = new Set<string>()
  private prefetchTargetId: string | undefined
  private prefetchEnabled = false
  private completionTimer: number | undefined
  private completionPublished = false
  private cancelledState = false
  private destroyed = false

  constructor() {
    this.renderer = new SelectableRenderer({
      fetchFont: async (fontId, jobId) =>
        (
          await sendBackgroundMessage({
            type: 'font:get',
            jobId,
            fontId,
          })
        ).bytes,
      lookup: async (request: LookupRequest) =>
        sendBackgroundMessage({
          type: 'dictionary:lookup',
          request,
        }),
      onFitDegraded: () => {
        // The region remains selectable and non-clipping; diagnostics can read
        // data-fit="degraded" without interrupting the normal workflow.
      },
    })
    this.queue = new VisibleFirstQueue(
      (item, signal) => this.process(item, signal),
      {
        onStart: (item) => {
          this.cancelCompletion()
          this.runState.start(item.value.candidate.element)
        },
        onSuccess: (item) => {
          const image = item.value.candidate.element
          this.queueIds.delete(image)
          this.failures.delete(image)
          this.runState.complete(image)
          this.scheduleFinish()
        },
        onFailure: (item, error) => {
          const image = item.value.candidate.element
          const attempts = this.runState.automaticRetries(image)
          if (shouldAutomaticallyRetryImage(error, attempts)) {
            const retryNumber = this.runState.automaticRetryQueued(image)
            if (
              this.requeueFailedImage(
                image,
                `Trying again automatically (${retryNumber}/${AUTOMATIC_IMAGE_RETRY_LIMIT})`,
              )
            ) {
              return
            }
            this.runState.start(image)
          }
          this.runState.fail(image)
          this.failures.set(image, {
            sourceUrl: item.value.candidate.sourceUrl,
            code:
              error instanceof RuntimeMessageError
                ? error.code
                : typeof (error as { code?: unknown })?.code === 'string'
                  ? (error as { code: string }).code
                  : error instanceof Error
                    ? error.name
                    : 'UNKNOWN_ERROR',
            message: error instanceof Error ? error.message : String(error),
            retryable: error instanceof RuntimeMessageError ? error.retryable : false,
          })
          this.badge(image).failure(errorMessage(error))
          this.scheduleFinish()
        },
        onIdle: () => this.scheduleFinish(),
      },
    )
    this.discovery = new ImageDiscovery((event) => this.onDiscovery(event))
    this.discovery.start()
    this.navigationTimer = window.setInterval(
      () => this.checkNavigation(),
      NAVIGATION_CHECK_INTERVAL_MS,
    )
  }

  async start(
    scope: TranslationScope,
    hskLevel: 1 | 2 | 3 | 4 | 5 | 6,
    learningMode: LearningMode,
    nameTranslation: NameTranslation,
    properNameGlossary: BrowserJobRequest['properNameGlossary'] = [],
  ): Promise<PageState> {
    const replacingRun = this.scope !== undefined
    this.generation += 1
    if (replacingRun) this.restoreAll()
    this.scope = scope
    this.hskLevel = hskLevel
    this.learningMode = learningMode
    this.nameTranslation = nameTranslation
    this.properNameGlossary = properNameGlossary.slice()
    this.cancelledState = false
    this.completionPublished = false
    this.runState.reset()
    this.failures.clear()
    this.cancelCompletion()
    this.context.clear()
    this.hud?.destroy()
    this.hud = new PageHud(() => this.cancel())

    const generation = this.generation
    if (replacingRun) {
      await sendBackgroundMessage({
        type: 'jobs:cancel-page',
        pageSessionId: this.sessionId,
      })
      if (generation !== this.generation || this.navigationUrl !== location.href) {
        throw abortError()
      }
    }

    const candidates = visibleFirst(this.discovery.current()).filter(
      (candidate) => scope === 'all' || candidate.visible,
    )
    if (candidates.length === 0) {
      const deferred = scope === 'all' ? this.discovery.deferred().length : 0
      if (deferred > 0) {
        this.hud.update({
          current: 0,
          total: deferred,
          message: 'Waiting for the chapter images',
        })
        this.scheduleFinish()
        return this.snapshot()
      }
      this.hud.fail('No manga images were found on this page.', 0, 0)
      return this.snapshot()
    }

    const recoveryCandidates = await Promise.all(
      candidates.map((candidate) => this.buildRecoveryCandidate(candidate, generation)),
    )
    if (generation !== this.generation || this.navigationUrl !== location.href) {
      throw abortError()
    }
    const recovered = await sendBackgroundMessage({
      type: 'jobs:recover',
      pageSessionId: this.sessionId,
      pageUrl: location.href,
      candidates: recoveryCandidates,
    })
    if (generation !== this.generation || this.navigationUrl !== location.href) {
      throw abortError()
    }
    const recoveredByIdentity = new Map(
      recovered.map((job) => [
        recoveryKey(job.sourceUrl, job.sourceWidth, job.sourceHeight, job.pageIndex),
        job,
      ]),
    )
    for (const candidate of candidates) {
      this.enqueue(
        candidate,
        recoveredByIdentity.get(
          recoveryKey(
            candidate.sourceUrl,
            candidate.element.naturalWidth,
            candidate.element.naturalHeight,
            candidate.domIndex,
          ),
        ),
      )
    }
    this.hud.update({
      current: 0,
      total: this.runState.snapshot().total,
      message: 'Waiting to start',
    })
    return this.snapshot()
  }

  cancel(): PageState {
    const run = this.runState.snapshot()
    this.generation += 1
    this.restoreAll()
    this.cancelledState = true
    this.hud?.cancelled(run.completed, run.total)
    return this.snapshot()
  }

  snapshot(): PageState {
    return (
      this.hud?.snapshot() ?? {
        state: 'idle',
        current: 0,
        total: 0,
        message: 'Ready',
      }
    )
  }

  diagnostics(): {
    run: ReturnType<ChapterRunState<HTMLImageElement>['snapshot']>
    failures: ImageFailureDiagnostic[]
  } {
    return {
      run: this.runState.snapshot(),
      failures: [...this.failures.values()],
    }
  }

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.generation += 1
    this.discovery.stop()
    window.clearInterval(this.navigationTimer)
    this.restoreAll()
    this.hud?.destroy()
    this.hud = undefined
  }

  private async buildRecoveryCandidate(
    candidate: DiscoveredImage,
    generation: number,
  ): Promise<RecoveryCandidate> {
    const sourceUrl = normalizedSourceUrl(candidate.sourceUrl)
    const identity = {
      sourceUrl,
      naturalWidth: candidate.element.naturalWidth,
      naturalHeight: candidate.element.naturalHeight,
    }
    const protocol = new URL(sourceUrl).protocol
    if (protocol === 'http:' || protocol === 'https:') return identity
    const inline = await tryContentBytes(candidate)
    if (generation !== this.generation || currentSourceUrl(candidate.element) !== sourceUrl) {
      throw abortError()
    }
    if (!inline) return identity
    const sourceSha256 = await sha256Hex(inline.bytes)
    if (generation !== this.generation || currentSourceUrl(candidate.element) !== sourceUrl) {
      throw abortError()
    }
    return { ...identity, sourceSha256 }
  }

  private restoreAll(): void {
    this.cancelCompletion()
    this.queue.cancelAll()
    this.clearPrefetch()
    for (const jobId of this.activeJobIds) {
      void sendBackgroundMessage({
        type: 'job:cancel',
        jobId,
      }).catch(() => undefined)
    }
    this.activeJobIds.clear()
    const renderedImages = [...this.rendered.values()]
    this.rendered.clear()
    for (const rendered of renderedImages) rendered.destroy()
    const badges = [...this.badges.values()]
    this.badges.clear()
    for (const badge of badges) badge.destroy()
    this.queueIds.clear()
    this.processed.clear()
    this.runState.reset()
    this.failures.clear()
    this.completionPublished = false
    this.context.clear()
    this.pageIndexByImage.clear()
  }

  private clearPrefetch(): void {
    this.prefetchEnabled = false
    this.prefetchTargetId = undefined
    void sendBackgroundMessage({
      type: 'image:prefetch-cancel',
      pageSessionId: this.sessionId,
      pageUrl: this.navigationUrl,
    }).catch(() => undefined)
  }

  private refreshPrefetch(): void {
    if (!this.prefetchEnabled || this.cancelledState) return
    const next = this.queue.next
    const candidate = next?.value.recovered ? undefined : next?.value.candidate
    let supported = false
    if (candidate) {
      try {
        const protocol = new URL(candidate.sourceUrl, location.href).protocol
        supported = protocol === 'http:' || protocol === 'https:'
      } catch {
        supported = false
      }
    }
    const targetId = supported ? next?.id : undefined
    if (targetId === this.prefetchTargetId) return
    this.prefetchTargetId = targetId
    if (!candidate || !targetId) {
      void sendBackgroundMessage({
        type: 'image:prefetch-cancel',
        pageSessionId: this.sessionId,
        pageUrl: this.navigationUrl,
      }).catch(() => undefined)
      return
    }
    void sendBackgroundMessage({
      type: 'image:prefetch',
      pageSessionId: this.sessionId,
      pageIndex: candidate.domIndex,
      imageUrl: candidate.sourceUrl,
      pageUrl: this.navigationUrl,
      naturalWidth: candidate.element.naturalWidth,
      naturalHeight: candidate.element.naturalHeight,
    }).catch(() => undefined)
  }

  private enqueue(candidate: DiscoveredImage, recovered?: RecoveredJob): void {
    if (this.completionPublished) return
    if (
      this.processed.has(candidate.element) ||
      this.runState.phase(candidate.element) !== undefined ||
      this.rendered.has(candidate.element) ||
      this.queueIds.has(candidate.element)
    ) {
      return
    }
    if (
      recovered &&
      (normalizedSourceUrl(recovered.sourceUrl) !== normalizedSourceUrl(candidate.sourceUrl) ||
        recovered.sourceWidth !== candidate.element.naturalWidth ||
        recovered.sourceHeight !== candidate.element.naturalHeight)
    ) {
      return
    }
    const id = candidateKey(candidate)
    if (!this.runState.register(candidate.element)) return
    this.cancelCompletion()
    this.queueIds.set(candidate.element, id)
    this.pageIndexByImage.set(candidate.element, candidate.domIndex)
    this.badge(candidate.element).update(
      recovered ? 'Picking up where you left off' : 'Waiting to start',
    )
    const accepted = this.queue.enqueue({
      id,
      value: {
        candidate,
        ...(recovered ? { recovered } : {}),
      },
      visible: candidate.visible,
      order: candidate.domIndex,
    })
    if (!accepted) {
      this.queueIds.delete(candidate.element)
      this.pageIndexByImage.delete(candidate.element)
      this.runState.remove(candidate.element)
    } else {
      this.refreshPrefetch()
    }
  }

  private badge(image: HTMLImageElement): ImageStatusBadge {
    const existing = this.badges.get(image)
    if (existing) return existing
    const badge = new ImageStatusBadge(image, () => this.retry(image))
    this.badges.set(image, badge)
    return badge
  }

  private retry(image: HTMLImageElement): void {
    if (!this.runState.manualRetryQueued(image)) return
    this.failures.delete(image)
    if (this.requeueFailedImage(image, 'Trying again')) {
      const run = this.runState.snapshot()
      this.hud?.update({ current: run.resolved, total: run.total })
      return
    }
    this.runState.start(image)
    this.runState.fail(image)
    this.badge(image).failure('This image couldn’t be translated. Try again.')
  }

  private requeueFailedImage(image: HTMLImageElement, status: string): boolean {
    const candidate = this.discovery.current().find((item) => item.element === image)
    const failedId = this.queueIds.get(image)
    this.processed.delete(image)
    if (!candidate || !failedId) return false
    const queued = this.queue.retry({
      id: failedId,
      value: { candidate },
      visible: candidate.visible,
      order: candidate.domIndex,
    })
    if (!queued) return false
    this.badge(image).update(status)
    this.refreshPrefetch()
    return true
  }

  private sourceSnapshot(candidate: DiscoveredImage): SourceSnapshot {
    return {
      generation: this.generation,
      pageSessionId: this.sessionId,
      navigationUrl: this.navigationUrl,
      sourceUrl: normalizedSourceUrl(candidate.sourceUrl),
      naturalWidth: candidate.element.naturalWidth,
      naturalHeight: candidate.element.naturalHeight,
    }
  }

  private assertCurrent(
    candidate: DiscoveredImage,
    snapshot: SourceSnapshot,
    signal: AbortSignal,
  ): void {
    if (this.navigationUrl !== location.href) {
      this.checkNavigation()
      throw abortError()
    }
    throwIfAborted(signal)
    if (
      snapshot.generation !== this.generation ||
      snapshot.pageSessionId !== this.sessionId ||
      snapshot.navigationUrl !== this.navigationUrl ||
      !candidate.owner.isConnected ||
      !candidate.element.isConnected ||
      currentSourceUrl(candidate.element) !== snapshot.sourceUrl ||
      candidate.element.naturalWidth !== snapshot.naturalWidth ||
      candidate.element.naturalHeight !== snapshot.naturalHeight
    ) {
      throw abortError()
    }
  }

  private async process(item: QueueItem<TranslationCandidate>, signal: AbortSignal): Promise<void> {
    const { candidate, recovered } = item.value
    // Recovery is a one-shot attempt. If viewport preemption requeues this
    // item, its recovered daemon job has been cancelled and the retry must
    // submit a fresh job rather than polling a terminal identity.
    delete item.value.recovered
    const consumesPrefetch = this.prefetchTargetId === item.id
    if (consumesPrefetch) this.prefetchTargetId = undefined
    this.prefetchEnabled = false
    const snapshot = this.sourceSnapshot(candidate)
    const badge = this.badge(candidate.element)
    let jobId = recovered?.jobId
    let sourceSha256 = recovered?.sourceSha256
    let sourceUrl = recovered?.sourceUrl
    let sourceWidth = recovered?.sourceWidth
    let sourceHeight = recovered?.sourceHeight
    let after = recovered?.acknowledgedSequence ?? 0
    let rendered: RenderedImage | undefined
    let viewportReporter: ViewportReporter | undefined
    const cancelOnAbort = (): void => {
      if (jobId) {
        void sendBackgroundMessage({ type: 'job:cancel', jobId }).catch(() => undefined)
      }
    }
    signal.addEventListener('abort', cancelOnAbort, { once: true })
    try {
      this.assertCurrent(candidate, snapshot, signal)
      if (!jobId) {
        badge.update('Opening the image')
        const inline = consumesPrefetch ? undefined : await tryContentBytes(candidate)
        this.assertCurrent(candidate, snapshot, signal)
        const precedingContext = this.context.before(candidate.domIndex)
        const submitted = await sendBackgroundMessage({
          type: 'job:submit',
          pageSessionId: this.sessionId,
          pageIndex: candidate.domIndex,
          imageUrl: candidate.sourceUrl,
          pageUrl: location.href,
          naturalWidth: snapshot.naturalWidth,
          naturalHeight: snapshot.naturalHeight,
          ...(inline?.mimeType ? { sourceMimeType: inline.mimeType } : {}),
          ...(inline ? { sourceBytes: inline.bytes } : {}),
          hskLevel: this.hskLevel,
          learningMode: this.learningMode,
          nameTranslation: this.nameTranslation,
          visibleRects: visibleImageRects(
            candidate.element,
            snapshot.naturalWidth,
            snapshot.naturalHeight,
          ),
          ...(precedingContext.length ? { precedingContext } : {}),
          ...(this.properNameGlossary.length
            ? { properNameGlossary: this.properNameGlossary }
            : {}),
        })
        // Retain the identity before checking the live DOM so a navigation or
        // source replacement that happened during submission can cancel the
        // newly-created companion job instead of leaking it.
        jobId = submitted.jobId
        sourceSha256 = submitted.sourceSha256
        sourceUrl = submitted.sourceUrl
        sourceWidth = submitted.sourceWidth
        sourceHeight = submitted.sourceHeight
        after = submitted.acknowledgedSequence
        this.assertCurrent(candidate, snapshot, signal)
      }

      if (!jobId || !sourceSha256 || !sourceUrl || !sourceWidth || !sourceHeight) {
        throw new RuntimeMessageError(
          'JOB_SOURCE_IDENTITY_MISSING',
          'The translation job source identity is incomplete.',
          false,
        )
      }
      if (
        normalizedSourceUrl(sourceUrl) !== snapshot.sourceUrl ||
        sourceWidth !== snapshot.naturalWidth ||
        sourceHeight !== snapshot.naturalHeight
      ) {
        throw new RuntimeMessageError(
          'JOB_SOURCE_IDENTITY_MISMATCH',
          'The translation job no longer matches the live page image.',
          false,
        )
      }
      if (recovered?.terminalType) {
        throw new RuntimeMessageError(
          'TERMINAL_JOB_RECOVERED',
          'A completed translation was recovered without an active page overlay.',
          true,
        )
      }

      this.activeJobIds.add(jobId)
      this.prefetchEnabled = true
      this.refreshPrefetch()
      rendered = this.renderer.begin(
        candidate,
        {
          jobId,
          sourceWidth,
          sourceHeight,
        },
        {
          signal,
          validate: () => this.assertCurrent(candidate, snapshot, signal),
        },
      )
      this.rendered.set(candidate.element, rendered)
      viewportReporter = new ViewportReporter(jobId, candidate.element, sourceWidth, sourceHeight)

      let complete = false
      while (!complete) {
        this.assertCurrent(candidate, snapshot, signal)
        const batch: JobUpdateBatch = await sendBackgroundMessage({
          type: 'job:updates',
          jobId,
          after,
        })
        this.assertCurrent(candidate, snapshot, signal)
        if (batch.jobId !== jobId) {
          throw new RuntimeMessageError(
            'UPDATE_IDENTITY_MISMATCH',
            'The job updates did not match the active image.',
            false,
          )
        }
        if (batch.updates.length === 0) continue
        let terminal: Extract<JobUpdate, { type: 'complete' | 'failed' | 'cancelled' }> | undefined
        for (const update of batch.updates) {
          this.assertCurrent(candidate, snapshot, signal)
          switch (update.type) {
            case 'progress':
              badge.update(update)
              {
                const run = this.runState.snapshot()
                this.hud?.update({
                  current: run.resolved,
                  total: run.total,
                  status: update,
                })
              }
              break
            case 'regionReady': {
              badge.update('Adding translated text')
              const patch = await sendBackgroundMessage({
                type: 'job:patch',
                jobId,
                patchId: update.region.patch.blobId,
                mimeType: update.region.patch.mimeType,
              })
              this.assertCurrent(candidate, snapshot, signal)
              if (patch.patchId !== update.region.patch.blobId) {
                throw new RuntimeMessageError(
                  'PATCH_IDENTITY_MISMATCH',
                  'The translated patch did not match its region update.',
                  false,
                )
              }
              await rendered.installRegion(update.region, patch.bytes, {
                signal,
                validate: () => this.assertCurrent(candidate, snapshot, signal),
              })
              break
            }
            case 'artworkPreserved':
              // The source pixels remain untouched by design. This update is
              // retained for diagnostics and regression evidence.
              break
            case 'complete':
            case 'failed':
            case 'cancelled':
              terminal = update
              break
          }
        }
        after = batch.nextSequence
        await sendBackgroundMessage({
          type: 'job:ack',
          jobId,
          sequence: after,
          ...(terminal ? { terminalType: terminal.type } : {}),
        })
        if (terminal?.type === 'failed') {
          throw new RuntimeMessageError(terminal.code, terminal.message, terminal.retryable)
        }
        if (terminal?.type === 'cancelled') throw abortError()
        complete = terminal?.type === 'complete'
      }

      await viewportReporter.stop().catch(() => undefined)
      viewportReporter = undefined
      this.assertCurrent(candidate, snapshot, signal)
      this.processed.add(candidate.element)
      this.context.commitPage(candidate.domIndex, rendered.regionsInReadingOrder())
      badge.destroy()
      this.badges.delete(candidate.element)
    } catch (error) {
      await viewportReporter?.stop().catch(() => undefined)
      viewportReporter = undefined
      if (jobId) {
        await sendBackgroundMessage({ type: 'job:cancel', jobId }).catch(() => undefined)
      }
      if (rendered && !this.processed.has(candidate.element)) {
        rendered.destroy()
        if (this.rendered.get(candidate.element) === rendered) {
          this.rendered.delete(candidate.element)
        }
      }
      throw error
    } finally {
      this.prefetchEnabled = false
      signal.removeEventListener('abort', cancelOnAbort)
      if (jobId) this.activeJobIds.delete(jobId)
    }
  }

  private removeTracked(image: HTMLImageElement): void {
    let tracked = false
    const pageIndex = this.pageIndexByImage.get(image)
    if (pageIndex !== undefined) this.context.removePage(pageIndex)
    this.pageIndexByImage.delete(image)
    const id = this.queueIds.get(image)
    if (id) {
      this.queue.remove(id)
      this.queueIds.delete(image)
      tracked = true
    }
    const rendered = this.rendered.get(image)
    if (rendered) {
      this.processed.delete(image)
      rendered.destroy()
      this.rendered.delete(image)
      tracked = true
    }
    if (this.runState.remove(image)) tracked = true
    this.failures.delete(image)
    this.badges.get(image)?.destroy()
    this.badges.delete(image)
    if (tracked) this.scheduleFinish()
  }

  private onDiscovery(event: DiscoveryEvent): void {
    this.checkNavigation()
    const image = event.candidate.element
    if (this.completionPublished && event.type === 'added') return
    if (event.type === 'visibility') {
      const id = this.queueIds.get(image)
      if (id) {
        this.queue.reprioritize(id, event.candidate.visible, event.candidate.domIndex)
      }
      if (
        !this.cancelledState &&
        this.scope === 'visible' &&
        event.candidate.visible &&
        !this.processed.has(image) &&
        this.runState.phase(image) !== 'failed'
      ) {
        this.enqueue(event.candidate)
      }
      this.refreshPrefetch()
      return
    }
    if (event.type === 'removed') {
      this.removeTracked(image)
      this.refreshPrefetch()
      return
    }
    if (event.type === 'updated') {
      if (event.previousSourceUrl === event.candidate.sourceUrl) {
        const id = this.queueIds.get(image)
        if (id) {
          this.queue.reprioritize(id, event.candidate.visible, event.candidate.domIndex)
        }
        this.refreshPrefetch()
        return
      }
      if (this.scope === undefined) {
        this.removeTracked(image)
        return
      }
      const previousSession = this.sessionId
      this.generation += 1
      this.restoreAll()
      this.scope = undefined
      this.cancelledState = false
      this.hud?.destroy()
      this.hud = undefined
      this.sessionId = createPageSessionId(false)
      void sendBackgroundMessage({
        type: 'jobs:cancel-page',
        pageSessionId: previousSession,
      }).catch(() => undefined)
      return
    }
    if (
      !this.cancelledState &&
      this.runState.phase(image) !== 'failed' &&
      (this.scope === 'all' || (this.scope === 'visible' && event.candidate.visible))
    ) {
      this.enqueue(event.candidate)
    }
  }

  private checkNavigation(): void {
    if (location.href === this.navigationUrl) return
    const previousSession = this.sessionId
    this.generation += 1
    this.restoreAll()
    this.scope = undefined
    this.cancelledState = false
    this.hud?.destroy()
    this.hud = undefined
    this.navigationUrl = location.href
    this.sessionId = createPageSessionId(false)
    void sendBackgroundMessage({
      type: 'jobs:cancel-page',
      pageSessionId: previousSession,
    }).catch(() => undefined)
  }

  private cancelCompletion(): void {
    if (this.completionTimer !== undefined) {
      window.clearTimeout(this.completionTimer)
      this.completionTimer = undefined
    }
  }

  private scheduleFinish(): void {
    if (
      !this.scope ||
      this.cancelledState ||
      this.completionPublished ||
      this.queue.size > 0 ||
      this.completionTimer !== undefined
    ) {
      return
    }
    const generation = this.generation
    const completionKey = this.discovery.completionKey()
    this.completionTimer = window.setTimeout(() => {
      this.completionTimer = undefined
      if (
        generation !== this.generation ||
        !this.scope ||
        this.cancelledState ||
        this.completionPublished ||
        this.queue.size > 0
      ) {
        return
      }
      if (completionKey !== this.discovery.completionKey()) {
        this.scheduleFinish()
        return
      }
      this.finish()
    }, COMPLETION_SETTLE_MS)
  }

  private finish(): void {
    if (!this.scope || this.cancelledState) return
    const run = this.runState.snapshot()
    const deferred = this.scope === 'all' ? this.discovery.deferred().length : 0
    if (deferred > 0) {
      this.hud?.update({
        current: run.resolved,
        total: run.total + deferred,
        message: 'Waiting for the remaining chapter images',
      })
      this.scheduleFinish()
      return
    }
    if (this.queue.size > 0 || run.unresolved > 0) {
      this.hud?.update({
        current: run.resolved,
        total: run.total,
      })
      this.scheduleFinish()
      return
    }
    if (!run.allResolved) {
      this.hud?.fail('No chapter images remain to translate.', 0, 0)
      return
    }
    if (run.failed > 0) {
      this.hud?.fail(
        `${run.failed} image${run.failed === 1 ? '' : 's'} still needs attention.`,
        run.completed,
        run.total,
      )
    } else {
      this.completionPublished = true
      this.hud?.complete(run.completed, run.total)
    }
  }
}

declare global {
  var __hmtPageController: PageTranslationController | undefined
}

const CONTENT_MESSAGE_TYPES = new Set([
  'content:start',
  'content:cancel',
  'content:state',
])

export function bootContentRuntime(): void {
  if (globalThis.__hmtPageController) return
  const controller = new PageTranslationController()
  globalThis.__hmtPageController = controller
  document.documentElement.dataset.hmtInjected = 'true'
  browser.runtime.onMessage.addListener(async (raw: unknown, sender) => {
    if (
      typeof raw !== 'object' ||
      raw === null ||
      !CONTENT_MESSAGE_TYPES.has(String((raw as Record<string, unknown>).type))
    ) {
      return undefined
    }
    if (sender.id !== browser.runtime.id) {
      throw new RuntimeMessageError(
        'INVALID_MESSAGE_SENDER',
        'The content request did not come from this extension.',
        false,
      )
    }
    const message = parseContentRequest(raw)
    switch (message.type) {
      case 'content:start':
        return controller.start(
          message.scope,
          message.hskLevel,
          message.learningMode,
          message.nameTranslation,
          message.properNameGlossary,
        )
      case 'content:cancel':
        return controller.cancel()
      case 'content:state':
        return controller.snapshot()
    }
  })
}
