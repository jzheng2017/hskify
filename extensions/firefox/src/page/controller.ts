import type {
  BrowserJobRequest,
  BrowserJobStatus,
  LookupRequest,
} from '../contracts/browser'
import {
  ImageDiscovery,
  visibleFirst,
  type DiscoveredImage,
  type DiscoveryEvent,
} from '../discovery/images'
import { VisibleFirstQueue, type QueueItem } from '../discovery/queue'
import {
  sendBackgroundMessage,
  RuntimeMessageError,
  type ContentRequest,
  type PageState,
  type RecoveredJob,
  type TranslationScope,
} from '../messaging/messages'
import { ImageStatusBadge, PageHud } from '../progress/hud'
import {
  SelectableRenderer,
  type RenderedImage,
} from '../rendering/renderer'

const PAGE_SESSION_KEY = 'hmt.pageSessionId'
const CONTENT_BYTE_LIMIT = 25 * 1024 * 1024

type TranslationCandidate = {
  candidate: DiscoveredImage
  recovered?: RecoveredJob
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

function pageSessionId(): string {
  const pageKey = `${PAGE_SESSION_KEY}:${location.href.split('#', 1)[0]}`
  const existing = sessionStorage.getItem(pageKey)
  if (existing) return existing
  const generated =
    typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  sessionStorage.setItem(pageKey, generated)
  return generated
}

function isFixturePage(): boolean {
  return (
    document.documentElement.dataset.hmtFixture === 'true' ||
    new URL(location.href).searchParams.get('hmtFixture') === '1'
  )
}

function candidateKey(candidate: DiscoveredImage): string {
  return `${candidate.domIndex}:${candidate.sourceUrl}`
}

function sourceMimeType(sourceUrl: string): string | undefined {
  if (!sourceUrl.startsWith('data:')) return undefined
  const match = /^data:([^;,]+)/i.exec(sourceUrl)
  return match?.[1]?.toLowerCase()
}

async function tryContentBytes(candidate: DiscoveredImage): Promise<
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
    const declaredLength = Number(response.headers.get('content-length') ?? '0')
    if (declaredLength > CONTENT_BYTE_LIMIT) return undefined
    const bytes = await response.arrayBuffer()
    if (bytes.byteLength > CONTENT_BYTE_LIMIT) return undefined
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
  private readonly sessionId = pageSessionId()
  private readonly fixtureMode = isFixturePage()
  private readonly discovery: ImageDiscovery
  private readonly renderer: SelectableRenderer
  private readonly queue: VisibleFirstQueue<TranslationCandidate>
  private readonly rendered = new Map<HTMLImageElement, RenderedImage>()
  private readonly badges = new Map<HTMLImageElement, ImageStatusBadge>()
  private readonly queueIds = new Map<HTMLImageElement, string>()
  private readonly processed = new Set<HTMLImageElement>()
  private readonly context: NonNullable<BrowserJobRequest['precedingContext']> = []
  private hud?: PageHud
  private scope?: TranslationScope
  private hskLevel: 1 | 2 | 3 | 4 | 5 | 6 = 5
  private activeJobId: string | undefined
  private completed = 0
  private failed = 0
  private total = 0
  private current = 0
  private cancelledState = false

  constructor() {
    this.renderer = new SelectableRenderer({
      fetchFont: async (fontId) =>
        (
          await sendBackgroundMessage({
            type: 'font:get',
            fontId,
            fixtureMode: this.fixtureMode,
          })
        ).bytes,
      lookup: async (request: LookupRequest) =>
        sendBackgroundMessage({
          type: 'dictionary:lookup',
          request,
          fixtureMode: this.fixtureMode,
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
        onSuccess: () => {
          this.completed += 1
          this.finishIfIdleSoon()
        },
        onFailure: (item, error) => {
          this.failed += 1
          const image = item.value.candidate.element
          this.badge(image).failure(errorMessage(error))
          this.finishIfIdleSoon()
        },
        onIdle: () => this.finish(),
      },
    )
    this.discovery = new ImageDiscovery((event) => this.onDiscovery(event))
    this.discovery.start()
  }

  async start(scope: TranslationScope, hskLevel: 1 | 2 | 3 | 4 | 5 | 6): Promise<PageState> {
    if (this.scope) this.cancelIncomplete()
    this.scope = scope
    this.hskLevel = hskLevel
    this.cancelledState = false
    this.completed = 0
    this.failed = 0
    this.total = 0
    this.current = 0
    this.processed.clear()
    for (const rendered of this.rendered.values()) rendered.destroy()
    this.rendered.clear()
    for (const badge of this.badges.values()) badge.destroy()
    this.badges.clear()
    this.queueIds.clear()
    this.context.splice(0)
    this.hud?.destroy()
    this.hud = new PageHud(() => this.cancel())

    const candidates = visibleFirst(this.discovery.current()).filter(
      (candidate) => scope === 'all' || candidate.visible,
    )
    if (candidates.length === 0) {
      this.hud.fail('No supported manga images are visible on this page.', 0, 0)
      return this.snapshot()
    }

    const recovered = await sendBackgroundMessage({
      type: 'jobs:recover',
      pageSessionId: this.sessionId,
    })
    const recoveredByIndex = new Map(recovered.map((job) => [job.pageIndex, job]))
    for (const candidate of candidates) {
      this.enqueue(candidate, recoveredByIndex.get(candidate.domIndex))
    }
    this.hud.update({
      current: 0,
      total: this.total,
      message: 'Queued',
    })
    return this.snapshot()
  }

  cancel(): PageState {
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
    this.cancelIncomplete()
    this.discovery.stop()
    for (const rendered of this.rendered.values()) rendered.destroy()
    for (const badge of this.badges.values()) badge.destroy()
    this.rendered.clear()
    this.badges.clear()
    this.hud?.destroy()
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
  }

  private enqueue(candidate: DiscoveredImage, recovered?: RecoveredJob): void {
    if (
      this.processed.has(candidate.element) ||
      this.rendered.has(candidate.element) ||
      this.queueIds.has(candidate.element)
    ) {
      return
    }
    const id = candidateKey(candidate)
    this.queueIds.set(candidate.element, id)
    this.total += 1
    this.badge(candidate.element).update(recovered ? recovered.status : 'Queued')
    this.queue.enqueue({
      id,
      value: {
        candidate,
        ...(recovered ? { recovered } : {}),
      },
      visible: candidate.visible,
      order: candidate.domIndex,
    })
  }

  private badge(image: HTMLImageElement): ImageStatusBadge {
    const existing = this.badges.get(image)
    if (existing) return existing
    const badge = new ImageStatusBadge(image, () => this.retry(image))
    this.badges.set(image, badge)
    return badge
  }

  private retry(image: HTMLImageElement): void {
    const candidate = this.discovery.current().find((item) => item.element === image)
    if (!candidate) return
    this.failed = Math.max(0, this.failed - 1)
    this.total = Math.max(0, this.total - 1)
    this.processed.delete(image)
    this.queueIds.delete(image)
    this.enqueue(candidate)
  }

  private async process(
    item: QueueItem<TranslationCandidate>,
    signal: AbortSignal,
  ): Promise<void> {
    const { candidate, recovered } = item.value
    const badge = this.badge(candidate.element)
    let jobId = recovered?.jobId
    let status = recovered?.status
    this.queueIds.delete(candidate.element)
    throwIfAborted(signal)

    if (!jobId) {
      badge.update('Reading image bytes')
      const inline = this.fixtureMode ? undefined : await tryContentBytes(candidate)
      throwIfAborted(signal)
      const submitted = await sendBackgroundMessage({
        type: 'job:submit',
        pageSessionId: this.sessionId,
        pageIndex: candidate.domIndex,
        imageUrl: candidate.sourceUrl,
        pageOrigin: location.origin,
        naturalWidth: candidate.element.naturalWidth,
        naturalHeight: candidate.element.naturalHeight,
        ...(inline?.mimeType ? { sourceMimeType: inline.mimeType } : {}),
        ...(inline ? { sourceBytes: inline.bytes } : {}),
        hskLevel: this.hskLevel,
        ...(this.context.length ? { precedingContext: this.context.slice(-12) } : {}),
        fixtureMode: this.fixtureMode,
      })
      jobId = submitted.jobId
    }

    this.activeJobId = jobId
    const cancelOnAbort = (): void => {
      if (jobId) {
        void sendBackgroundMessage({ type: 'job:cancel', jobId }).catch(() => undefined)
      }
    }
    signal.addEventListener('abort', cancelOnAbort, { once: true })
    try {
      while (!status || status.state === 'running') {
        throwIfAborted(signal)
        status = await sendBackgroundMessage({ type: 'job:poll', jobId })
        badge.update(status)
        this.hud?.update({
          current: this.current,
          total: this.total,
          status,
        })
        if (status.state !== 'running') break
        await delay(
          this.fixtureMode ? 100 : document.visibilityState === 'visible' ? 1_000 : 4_000,
          signal,
        )
      }
      if (status.state === 'failed') {
        throw new RuntimeMessageError(
          status.errorCode ?? 'JOB_FAILED',
          status.message,
          true,
        )
      }
      if (status.state === 'cancelled') throw abortError()
      throwIfAborted(signal)
      const delivered = await sendBackgroundMessage({ type: 'job:result', jobId })
      throwIfAborted(signal)
      const rendered = await this.renderer.render(candidate, delivered)
      this.rendered.set(candidate.element, rendered)
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
      if (jobId && !signal.aborted) {
        await sendBackgroundMessage({ type: 'job:cancel', jobId }).catch(() => undefined)
      }
      throw error
    } finally {
      signal.removeEventListener('abort', cancelOnAbort)
      if (this.activeJobId === jobId) this.activeJobId = undefined
    }
  }

  private onDiscovery(event: DiscoveryEvent): void {
    const image = event.candidate.element
    if (event.type === 'visibility') {
      const id = this.queueIds.get(image)
      if (id) this.queue.reprioritize(id, event.candidate.visible)
      if (
        !this.cancelledState &&
        this.scope === 'visible' &&
        event.candidate.visible &&
        !this.processed.has(image)
      ) {
        this.enqueue(event.candidate)
      }
      return
    }
    if (event.type === 'removed') {
      const id = this.queueIds.get(image)
      if (id) this.queue.remove(id)
      this.queueIds.delete(image)
      const wasRendered = this.rendered.has(image)
      this.rendered.get(image)?.destroy()
      this.rendered.delete(image)
      this.badges.get(image)?.destroy()
      this.badges.delete(image)
      if (wasRendered) {
        this.processed.delete(image)
        this.completed = Math.max(0, this.completed - 1)
      }
      this.total = Math.max(0, this.total - 1)
      return
    }
    if (event.type === 'updated') {
      const id = this.queueIds.get(image)
      if (id) this.queue.remove(id)
      this.queueIds.delete(image)
      if (this.rendered.has(image)) {
        this.rendered.get(image)?.destroy()
        this.rendered.delete(image)
        this.processed.delete(image)
        this.completed = Math.max(0, this.completed - 1)
      }
      this.total = Math.max(0, this.total - 1)
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
      (this.scope === 'all' ||
        (this.scope === 'visible' && event.candidate.visible))
    ) {
      this.enqueue(event.candidate)
    }
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

function isContentRequest(value: unknown): value is ContentRequest {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const type = (value as Record<string, unknown>).type
  return type === 'content:start' || type === 'content:cancel' || type === 'content:state'
}

export function bootContentRuntime(): void {
  if (globalThis.__hmtPageController) return
  const controller = new PageTranslationController()
  globalThis.__hmtPageController = controller
  document.documentElement.dataset.hmtInjected = 'true'
  browser.runtime.onMessage.addListener(async (message: unknown) => {
    if (!isContentRequest(message)) return undefined
    switch (message.type) {
      case 'content:start':
        return controller.start(message.scope, message.hskLevel)
      case 'content:cancel':
        return controller.cancel()
      case 'content:state':
        return controller.snapshot()
    }
  })
}
