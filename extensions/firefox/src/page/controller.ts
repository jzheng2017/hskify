import type {
  BrowserJobRequest,
  LookupRequest,
} from '../contracts/browser'
import { sha256Hex } from '../acquisition/hash'
import { requiredCrossOriginPatterns } from '../acquisition/origin-permissions'
import {
  ImageDiscovery,
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
  type PermissionPlan,
  type RecoveredJob,
  type RecoveryCandidate,
  type TranslationScope,
} from '../messaging/messages'
import { ImageStatusBadge, PageHud } from '../progress/hud'
import {
  SelectableRenderer,
  type RenderedImage,
} from '../rendering/renderer'

const PAGE_SESSION_KEY = 'hmt.pageSessionId'
const CONTENT_BYTE_LIMIT = 25 * 1024 * 1024
const NAVIGATION_CHECK_INTERVAL_MS = 250

type TranslationCandidate = {
  candidate: DiscoveredImage
  recovered?: RecoveredJob
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

function delay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError())
      return
    }
    const timer = setTimeout(resolve, milliseconds)
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(abortError())
      },
      { once: true },
    )
  })
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

function recoveryKey(
  sourceUrl: string,
  width: number,
  height: number,
  pageIndex: number,
): string {
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

function errorMessage(error: unknown): string {
  if (error instanceof RuntimeMessageError || error instanceof Error) return error.message
  return 'This image could not be translated.'
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
  private readonly failedImages = new Set<HTMLImageElement>()
  private readonly context: NonNullable<BrowserJobRequest['precedingContext']> = []
  private readonly navigationTimer: number
  private hud: PageHud | undefined
  private scope: TranslationScope | undefined
  private hskLevel: 1 | 2 | 3 | 4 | 5 | 6 = 5
  private activeJobId: string | undefined
  private completed = 0
  private failed = 0
  private total = 0
  private current = 0
  private cancelledState = false

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
        onStart: () => {
          this.current = this.completed + this.failed
        },
        onSuccess: (item) => {
          const image = item.value.candidate.element
          this.queueIds.delete(image)
          this.completed += 1
          this.finishIfIdleSoon()
        },
        onFailure: (item, error) => {
          const image = item.value.candidate.element
          this.failedImages.add(image)
          this.failed += 1
          this.badge(image).failure(errorMessage(error))
          this.finishIfIdleSoon()
        },
        onIdle: () => this.finish(),
      },
    )
    this.discovery = new ImageDiscovery((event) => this.onDiscovery(event))
    this.discovery.start()
    this.navigationTimer = window.setInterval(
      () => this.checkNavigation(),
      NAVIGATION_CHECK_INTERVAL_MS,
    )
  }

  permissionPlan(): PermissionPlan {
    const candidates = this.discovery.current()
    return {
      visibleOrigins: requiredCrossOriginPatterns(
        location.href,
        candidates.filter((candidate) => candidate.visible).map((candidate) => candidate.sourceUrl),
      ),
      allOrigins: requiredCrossOriginPatterns(
        location.href,
        candidates.map((candidate) => candidate.sourceUrl),
      ),
    }
  }

  async start(scope: TranslationScope, hskLevel: 1 | 2 | 3 | 4 | 5 | 6): Promise<PageState> {
    const replacingRun = this.scope !== undefined
    this.generation += 1
    if (replacingRun) this.cancelIncomplete()
    this.scope = scope
    this.hskLevel = hskLevel
    this.cancelledState = false
    this.completed = 0
    this.failed = 0
    this.total = 0
    this.current = 0
    this.processed.clear()
    this.failedImages.clear()
    for (const rendered of this.rendered.values()) rendered.destroy()
    this.rendered.clear()
    for (const badge of this.badges.values()) badge.destroy()
    this.badges.clear()
    this.queueIds.clear()
    this.context.splice(0)
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
      this.hud.fail('No supported manga images are visible on this page.', 0, 0)
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
      total: this.total,
      message: 'Queued',
    })
    return this.snapshot()
  }

  cancel(): PageState {
    this.generation += 1
    this.cancelIncomplete()
    this.cancelledState = true
    this.hud?.cancelled(this.completed, this.total)
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

  destroy(): void {
    this.generation += 1
    this.cancelIncomplete()
    this.discovery.stop()
    window.clearInterval(this.navigationTimer)
    for (const rendered of this.rendered.values()) rendered.destroy()
    for (const badge of this.badges.values()) badge.destroy()
    this.rendered.clear()
    this.badges.clear()
    this.hud?.destroy()
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
    if (
      generation !== this.generation ||
      currentSourceUrl(candidate.element) !== sourceUrl
    ) {
      throw abortError()
    }
    if (!inline) return identity
    const sourceSha256 = await sha256Hex(inline.bytes)
    if (
      generation !== this.generation ||
      currentSourceUrl(candidate.element) !== sourceUrl
    ) {
      throw abortError()
    }
    return { ...identity, sourceSha256 }
  }

  private cancelIncomplete(): void {
    this.queue.cancelAll()
    if (this.activeJobId) {
      void sendBackgroundMessage({
        type: 'job:cancel',
        jobId: this.activeJobId,
      }).catch(() => undefined)
      this.activeJobId = undefined
    }
    for (const [image, badge] of this.badges) {
      if (!this.rendered.has(image)) {
        badge.destroy()
        this.badges.delete(image)
      }
    }
    this.queueIds.clear()
    this.failedImages.clear()
  }

  private enqueue(candidate: DiscoveredImage, recovered?: RecoveredJob): void {
    if (
      this.processed.has(candidate.element) ||
      this.failedImages.has(candidate.element) ||
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
    this.queueIds.set(candidate.element, id)
    this.total += 1
    this.badge(candidate.element).update(recovered ? recovered.status : 'Queued')
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
      this.total = Math.max(0, this.total - 1)
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
    if (!this.failedImages.delete(image)) return
    const candidate = this.discovery.current().find((item) => item.element === image)
    const failedId = this.queueIds.get(image)
    this.badges.get(image)?.destroy()
    this.badges.delete(image)
    this.failed = Math.max(0, this.failed - 1)
    this.total = Math.max(0, this.total - 1)
    this.processed.delete(image)
    if (!candidate || !failedId) {
      if (failedId) this.queue.remove(failedId)
      this.queueIds.delete(image)
      return
    }
    this.total += 1
    this.badge(image).update('Queued')
    const queued = this.queue.retry({
      id: failedId,
      value: { candidate },
      visible: candidate.visible,
      order: candidate.domIndex,
    })
    if (!queued) {
      this.queueIds.delete(image)
      this.total = Math.max(0, this.total - 1)
      return
    }
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
    throwIfAborted(signal)
    if (
      snapshot.generation !== this.generation ||
      snapshot.pageSessionId !== this.sessionId ||
      snapshot.navigationUrl !== this.navigationUrl ||
      this.navigationUrl !== location.href ||
      !candidate.owner.isConnected ||
      !candidate.element.isConnected ||
      currentSourceUrl(candidate.element) !== snapshot.sourceUrl ||
      candidate.element.naturalWidth !== snapshot.naturalWidth ||
      candidate.element.naturalHeight !== snapshot.naturalHeight
    ) {
      throw abortError()
    }
  }

  private async process(
    item: QueueItem<TranslationCandidate>,
    signal: AbortSignal,
  ): Promise<void> {
    const { candidate, recovered } = item.value
    const snapshot = this.sourceSnapshot(candidate)
    const badge = this.badge(candidate.element)
    let jobId = recovered?.jobId
    let sourceSha256 = recovered?.sourceSha256
    let sourceUrl = recovered?.sourceUrl
    let sourceWidth = recovered?.sourceWidth
    let sourceHeight = recovered?.sourceHeight
    let status = recovered?.status
    const cancelOnAbort = (): void => {
      if (jobId) {
        void sendBackgroundMessage({ type: 'job:cancel', jobId }).catch(() => undefined)
      }
    }
    signal.addEventListener('abort', cancelOnAbort, { once: true })
    try {
      this.assertCurrent(candidate, snapshot, signal)
      if (!jobId) {
        badge.update('Reading image bytes')
        const inline = await tryContentBytes(candidate)
        this.assertCurrent(candidate, snapshot, signal)
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
          ...(this.context.length ? { precedingContext: this.context.slice(-12) } : {}),
        })
        // Retain the identity before checking the live DOM so a navigation or
        // source replacement that happened during submission can cancel the
        // newly-created companion job instead of leaking it.
        jobId = submitted.jobId
        sourceSha256 = submitted.sourceSha256
        sourceUrl = submitted.sourceUrl
        sourceWidth = submitted.sourceWidth
        sourceHeight = submitted.sourceHeight
        this.assertCurrent(candidate, snapshot, signal)
      }

      if (!sourceSha256 || !sourceUrl || !sourceWidth || !sourceHeight) {
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

      this.activeJobId = jobId
      while (!status || status.state === 'running') {
        this.assertCurrent(candidate, snapshot, signal)
        status = await sendBackgroundMessage({ type: 'job:poll', jobId })
        this.assertCurrent(candidate, snapshot, signal)
        if (status.jobId !== jobId) {
          throw new RuntimeMessageError(
            'STATUS_IDENTITY_MISMATCH',
            'The job status did not match the active image.',
            false,
          )
        }
        badge.update(status)
        this.hud?.update({
          current: this.current,
          total: this.total,
          status,
        })
        if (status.state !== 'running') break
        await delay(
          document.visibilityState === 'visible' ? 1_000 : 4_000,
          signal,
        )
        this.assertCurrent(candidate, snapshot, signal)
      }
      if (status.state === 'failed') {
        throw new RuntimeMessageError(
          status.errorCode ?? 'JOB_FAILED',
          status.message,
          true,
        )
      }
      if (status.state === 'cancelled') throw abortError()
      this.assertCurrent(candidate, snapshot, signal)
      const delivered = await sendBackgroundMessage({
        type: 'job:result',
        jobId,
        pageSessionId: snapshot.pageSessionId,
        sourceUrl,
        sourceSha256,
        sourceWidth,
        sourceHeight,
      })
      this.assertCurrent(candidate, snapshot, signal)
      if (
        delivered.result.jobId !== jobId ||
        delivered.result.sourceSha256 !== sourceSha256 ||
        delivered.result.sourceWidth !== sourceWidth ||
        delivered.result.sourceHeight !== sourceHeight
      ) {
        throw new RuntimeMessageError(
          'RESULT_IDENTITY_MISMATCH',
          'The completed result did not match the live source image.',
          false,
        )
      }
      let rendered: RenderedImage | undefined
      try {
        rendered = await this.renderer.render(candidate, delivered, {
          signal,
          validate: () => this.assertCurrent(candidate, snapshot, signal),
        })
        this.assertCurrent(candidate, snapshot, signal)
        this.rendered.set(candidate.element, rendered)
      } catch (error) {
        rendered?.destroy()
        throw error
      }
      this.processed.add(candidate.element)
      for (const region of [...delivered.result.regions].sort(
        (left, right) => left.readingOrder - right.readingOrder,
      )) {
        if (region.kind === 'sfx' || !region.displayedChinese || !region.sourceEnglish) continue
        this.context.push({
          sourceEnglish: region.sourceEnglish,
          chinese: region.displayedChinese,
        })
      }
      if (this.context.length > 12) this.context.splice(0, this.context.length - 12)
      badge.destroy()
      this.badges.delete(candidate.element)
    } catch (error) {
      if (jobId) {
        await sendBackgroundMessage({ type: 'job:cancel', jobId }).catch(() => undefined)
      }
      throw error
    } finally {
      signal.removeEventListener('abort', cancelOnAbort)
      if (this.activeJobId === jobId) this.activeJobId = undefined
    }
  }

  private removeTracked(image: HTMLImageElement): void {
    let tracked = false
    const id = this.queueIds.get(image)
    if (id) {
      this.queue.remove(id)
      this.queueIds.delete(image)
      tracked = true
    }
    const rendered = this.rendered.get(image)
    if (rendered) {
      rendered.destroy()
      this.rendered.delete(image)
      this.processed.delete(image)
      this.completed = Math.max(0, this.completed - 1)
      tracked = true
    }
    if (this.failedImages.delete(image)) {
      this.failed = Math.max(0, this.failed - 1)
      tracked = true
    }
    this.badges.get(image)?.destroy()
    this.badges.delete(image)
    if (tracked) this.total = Math.max(0, this.total - 1)
  }

  private onDiscovery(event: DiscoveryEvent): void {
    this.checkNavigation()
    const image = event.candidate.element
    if (event.type === 'visibility') {
      const id = this.queueIds.get(image)
      if (id) this.queue.reprioritize(id, event.candidate.visible)
      if (
        !this.cancelledState &&
        this.scope === 'visible' &&
        event.candidate.visible &&
        !this.processed.has(image) &&
        !this.failedImages.has(image)
      ) {
        this.enqueue(event.candidate)
      }
      return
    }
    if (event.type === 'removed') {
      this.removeTracked(image)
      return
    }
    if (event.type === 'updated') {
      this.removeTracked(image)
      if (
        !this.cancelledState &&
        (this.scope === 'all' ||
          (this.scope === 'visible' && event.candidate.visible))
      ) {
        this.enqueue(event.candidate)
      }
      return
    }
    if (
      !this.cancelledState &&
      !this.failedImages.has(image) &&
      (this.scope === 'all' ||
        (this.scope === 'visible' && event.candidate.visible))
    ) {
      this.enqueue(event.candidate)
    }
  }

  private checkNavigation(): void {
    if (location.href === this.navigationUrl) return
    const previousSession = this.sessionId
    this.generation += 1
    this.cancelIncomplete()
    for (const rendered of this.rendered.values()) rendered.destroy()
    for (const badge of this.badges.values()) badge.destroy()
    this.rendered.clear()
    this.badges.clear()
    this.processed.clear()
    this.failedImages.clear()
    this.queueIds.clear()
    this.context.splice(0)
    this.completed = 0
    this.failed = 0
    this.total = 0
    this.current = 0
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

  private finishIfIdleSoon(): void {
    this.current = this.completed + this.failed
  }

  private finish(): void {
    if (!this.scope || this.cancelledState) return
    if (this.failed > 0) {
      this.hud?.fail(
        `${this.failed} image${this.failed === 1 ? '' : 's'} failed. Use Retry on the image.`,
        this.completed,
        this.total,
      )
    } else {
      this.hud?.complete(this.completed, this.total)
    }
  }
}

declare global {
  var __hmtPageController: PageTranslationController | undefined
}

const CONTENT_MESSAGE_TYPES = new Set([
  'content:prepare',
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
      case 'content:prepare':
        return controller.permissionPlan()
      case 'content:start':
        return controller.start(message.scope, message.hskLevel)
      case 'content:cancel':
        return controller.cancel()
      case 'content:state':
        return controller.snapshot()
    }
  })
}
