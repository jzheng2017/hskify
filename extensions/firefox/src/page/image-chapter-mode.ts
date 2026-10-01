import { sha256Hex } from '../acquisition/hash'
import { DEFAULT_IMAGE_LIMITS } from '../acquisition/image-format'
import { waitForVisibleSource } from '../acquisition/rendered-region'
import type {
  LearningMode,
  LookupRequest,
  ReadingDirection,
  TranslationSettings,
} from '../contracts/browser'
import { isRectVisible } from '../discovery/images'
import { VisibleFirstQueue, type QueueItem } from '../discovery/queue'
import {
  LiveSurfaceDiscovery,
  visibleFirst as visibleSurfaceFirst,
  type DiscoveredSurface,
  type SurfaceDiscoveryEvent,
} from '../discovery/surfaces'
import {
  parseContentRequest,
  sendBackgroundMessage,
  RuntimeMessageError,
  type PageState,
  type RecoveredImageJob,
  type RecoveryCandidate,
  type TranslationScope,
} from '../messaging/messages'
import { ImageStatusBadge } from '../progress/hud'
import { visibleImageRects } from '../rendering/geometry'
import { SelectableRenderer, type RenderedImage } from '../rendering/renderer'
import { ChapterRunController, chapterAbortError } from './chapter-run-controller'
import { ChapterRunState } from './run-state'

const VIEWPORT_THROTTLE_MS = 100
// The daemon already admits one Vision phase and one Language phase while
// prioritizing the live viewport. Two in-flight pages keep the independent
// Vision and Language lanes fed without making visible pages contend with a
// speculative tail for the same resident models.
const CHAPTER_PIPELINE_CONCURRENCY = 2
// Keep source retention inside the same two-page bound as execution.
const CHAPTER_PIPELINE_PIXEL_BUDGET = DEFAULT_IMAGE_LIMITS.maximumPixels * 2
export const COMPLETION_SETTLE_MS = 300

type TranslationCandidate = {
  candidate: DiscoveredSurface
  recovered?: RecoveredImageJob
  retryItemIds?: string[]
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
  contentRevision?: string
  capturedRevision?: boolean
  jobId?: string
}

function normalizedSourceUrl(value: string): string {
  const url = new URL(value, location.href)
  url.hash = ''
  return url.href
}

function currentSourceUrl(candidate: DiscoveredSurface): string {
  if (
    candidate.captureOnly ||
    candidate.kind !== 'image' ||
    candidate.element.tagName.toLowerCase() !== 'img' ||
    !('currentSrc' in candidate.element)
  ) {
    return normalizedSourceUrl(candidate.sourceUrl)
  }
  const image = candidate.element as HTMLImageElement
  return normalizedSourceUrl(image.currentSrc || image.src || candidate.sourceUrl)
}

function candidateKey(candidate: DiscoveredSurface): string {
  return `${candidate.id}:${normalizedSourceUrl(candidate.sourceUrl)}:${candidate.sourceWidth}x${candidate.sourceHeight}:${candidate.sourceRevision ?? ''}`
}

/**
 * Compare candidates in the reader's document order without using visibility
 * as a tie breaker.  The specialised image and generic surface adapters
 * maintain independent DOM indexes, so an element relationship is the only
 * reliable ordering signal when both adapters report a surface from the same
 * document.  Nested frame documents are disconnected from the parent DOM and
 * fall back to their adapter index plus stable identity.
 */
function compareDocumentCandidates(left: DiscoveredSurface, right: DiscoveredSurface): number {
  if (
    left.element !== right.element &&
    left.element.ownerDocument === right.element.ownerDocument
  ) {
    const position = left.element.compareDocumentPosition(right.element)
    if (position & 4) return -1
    if (position & 2) return 1
  }
  return left.domIndex - right.domIndex || left.id.localeCompare(right.id)
}

/**
 * Reserve page identities for unloaded lazy images before the first request
 * is submitted.  A reader is allowed to insert a decoded page before an
 * already translated page; assigning an index at load time would make the
 * chapter graph depend on network timing.  Cross-document surfaces cannot be
 * compared with DOM position, so their global viewport position is the
 * deterministic fallback after same-document relationships.
 */
function compareDocumentElements(left: Element, right: Element): number {
  if (left === right) return 0
  if (left.ownerDocument === right.ownerDocument) {
    const position = left.compareDocumentPosition(right)
    if (position & 4) return -1
    if (position & 2) return 1
  }
  const globalPosition = (element: Element): { top: number; left: number } => {
    let rect = element.getBoundingClientRect()
    let ownerDocument = element.ownerDocument
    while (ownerDocument !== document) {
      const frame = ownerDocument.defaultView?.frameElement
      if (!frame) break
      const frameRect = frame.getBoundingClientRect()
      rect = {
        top: rect.top + frameRect.top,
        left: rect.left + frameRect.left,
      } as DOMRect
      ownerDocument = frame.ownerDocument
    }
    return { top: rect.top, left: rect.left }
  }
  const leftPosition = globalPosition(left)
  const rightPosition = globalPosition(right)
  return (
    leftPosition.top - rightPosition.top ||
    leftPosition.left - rightPosition.left ||
    left.tagName.localeCompare(right.tagName)
  )
}

function recoveryKey(
  sourceUrl: string,
  width: number,
  height: number,
  sourceIndex: number,
): string {
  return `${normalizedSourceUrl(sourceUrl)}:${width}x${height}:${sourceIndex}`
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

export async function tryContentBytes(candidate: DiscoveredSurface): Promise<
  | {
      bytes: ArrayBuffer
      mimeType?: string
    }
  | undefined
> {
  if (candidate.capture) {
    try {
      const captured = await candidate.capture()
      if (!captured || captured.bytes.byteLength > DEFAULT_IMAGE_LIMITS.maximumBytes) {
        return undefined
      }
      return {
        bytes: captured.bytes,
        ...(captured.mimeType ? { mimeType: captured.mimeType } : {}),
      }
    } catch {
      return undefined
    }
  }
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
    const bytes = await readBoundedBody(response, DEFAULT_IMAGE_LIMITS.maximumBytes)
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

export class ImageFocusReporter {
  private timer: number | undefined
  private lastPayload = ''
  private stopped = false
  private inFlight: Promise<void> = Promise.resolve()
  private readonly resizeObserver: ResizeObserver | undefined

  constructor(
    private readonly jobId: string,
    private readonly image: HTMLElement,
    private readonly sourceWidth: number,
    private readonly sourceHeight: number,
    private readonly onViewport?: () => void,
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
    active = active && !this.image.ownerDocument.hidden
    const visibleRects = active
      ? visibleImageRects(this.image, this.sourceWidth, this.sourceHeight)
      : []
    const payloadKey = JSON.stringify({ visibleRects, active })
    if (!force && payloadKey === this.lastPayload) return
    this.lastPayload = payloadKey
    this.inFlight = this.inFlight
      .catch(() => undefined)
      .then(() =>
        sendBackgroundMessage({
          type: 'job:focus',
          jobId: this.jobId,
          focus: { kind: 'image', visibleRects, active },
        }),
      )
      .then(() => this.onViewport?.())
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
  return "This image couldn't be translated. Try again."
}

export class ImageChapterMode {
  private readonly run: ChapterRunController
  private readonly surfaceDiscovery: LiveSurfaceDiscovery
  private readonly renderer: SelectableRenderer
  private readonly queue: VisibleFirstQueue<TranslationCandidate>
  private readonly rendered = new Map<HTMLElement, RenderedImage>()
  private readonly badges = new Map<HTMLElement, ImageStatusBadge>()
  private readonly queueIds = new Map<HTMLElement, string>()
  private readonly processed = new Set<HTMLElement>()
  private readonly runState = new ChapterRunState<HTMLElement>()
  private readonly failures = new Map<HTMLElement, ImageFailureDiagnostic>()
  private readonly failedItems = new Map<HTMLElement, Set<string>>()
  // A reader may insert or reorder DOM nodes while lazy loading.  DOM indexes
  // are therefore observations, not page identities.  Assign one canonical
  // index to each admitted surface for the lifetime of this chapter run and
  // use it for every queue, recovery, daemon, and context-barrier message.
  private readonly canonicalSourceIndexByElement = new Map<HTMLElement, number>()
  private readonly submittedSourceIndexes = new Set<number>()
  private readonly submittingSourceIndexes = new Set<number>()
  private nextCanonicalSourceIndex = 0
  private scope: TranslationScope | undefined
  private hskLevel: 1 | 2 | 3 | 4 | 5 | 6 = 5
  private learningMode: LearningMode = 'natural'
  private readingDirection: ReadingDirection = 'ltr'
  // Stable page identities remain useful for recovery and available-context
  // ordering, but never form a wait barrier.
  private chapterSourceOrder: number[] = []
  private prefetchTargetId: string | undefined
  private prefetchEnabled = false
  private completionTimer: number | undefined
  private destroyed = false

  constructor(sourceRoot: ParentNode = document) {
    this.run = new ChapterRunController('image', () => this.restoreAll())
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
        // data-hskify-fit="degraded" without interrupting the normal workflow.
      },
      onRetryRegion: (itemId, candidate) => this.retry(candidate.element, itemId),
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
          this.run.completeItem(item.id)
          this.scheduleFinish()
        },
        onFailure: (item, error) => {
          const image = item.value.candidate.element
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
          this.run.completeItem(item.id)
          this.scheduleFinish()
        },
        onIdle: () => this.scheduleFinish(),
      },
      {
        maximumConcurrent: CHAPTER_PIPELINE_CONCURRENCY,
        maximumActiveCost: CHAPTER_PIPELINE_PIXEL_BUDGET,
      },
    )
    this.surfaceDiscovery = new LiveSurfaceDiscovery(
      (event) => this.onDiscovery(event),
      document,
      sourceRoot,
    )
    this.surfaceDiscovery.start()
  }

  private get sessionId(): string {
    return this.run.sessionId
  }
  private get navigationUrl(): string {
    return this.run.pageUrl
  }
  private get generation(): number {
    return this.run.generation
  }
  private get completionPublished(): boolean {
    return this.run.isComplete
  }
  private get cancelledState(): boolean {
    return this.run.isCancelled
  }

  private currentCandidates(): DiscoveredSurface[] {
    return visibleSurfaceFirst(this.surfaceDiscovery.current()).sort(
      (left, right) =>
        Number(right.visible) - Number(left.visible) ||
        this.orderingIndex(left) - this.orderingIndex(right),
    )
  }

  /** Return the frozen chapter index when admitted, otherwise the discovery hint. */
  private orderingIndex(candidate: DiscoveredSurface): number {
    const sourceIndex = this.canonicalSourceIndexByElement.get(candidate.element)
    return sourceIndex === undefined
      ? candidate.domIndex
      : this.chapterSourceOrder.indexOf(sourceIndex)
  }

  /**
   * Admit a surface to the chapter's immutable document stream.  DOM order is transmitted separately, so a lazy insertion cannot
   * rewrite the index of work already submitted or waiting in the queue.
   */
  private canonicalSourceIndex(candidate: DiscoveredSurface): number {
    const existing = this.canonicalSourceIndexByElement.get(candidate.element)
    if (existing !== undefined) {
      this.includeChapterPage(existing)
      return existing
    }
    const sourceIndex = this.nextCanonicalSourceIndex++
    this.canonicalSourceIndexByElement.set(candidate.element, sourceIndex)
    this.includeChapterPage(sourceIndex)
    return sourceIndex
  }

  private includeChapterPage(sourceIndex: number): void {
    const live = new Set(this.chapterSourceOrder)
    live.add(sourceIndex)
    this.chapterSourceOrder = [...this.canonicalSourceIndexByElement]
      .filter(([, index]) => live.has(index))
      .sort(([left], [right]) => compareDocumentElements(left, right))
      .map(([, index]) => index)
  }

  private removeUnsubmittedSource(element: HTMLElement): void {
    const sourceIndex = this.canonicalSourceIndexByElement.get(element)
    if (sourceIndex === undefined) return
    if (
      this.submittedSourceIndexes.has(sourceIndex) ||
      this.submittingSourceIndexes.has(sourceIndex)
    ) {
      return
    }
    this.chapterSourceOrder = this.chapterSourceOrder.filter((index) => index !== sourceIndex)
  }

  private establishCanonicalPageOrder(candidates: readonly DiscoveredSurface[]): void {
    const admitted = new Set<HTMLElement>(candidates.map((candidate) => candidate.element))
    const lazyImages = this.surfaceDiscovery
      .deferred()
      .filter(
        (image) =>
          !admitted.has(image) &&
          (this.scope === 'all' ||
            isRectVisible(
              image.getBoundingClientRect(),
              image.ownerDocument.defaultView ?? window,
            )),
      )
    const surfaces = [...candidates, ...lazyImages].sort((left, right) =>
      compareDocumentElements(
        'element' in left ? left.element : left,
        'element' in right ? right.element : right,
      ),
    )
    for (const surface of surfaces) {
      if ('sourceUrl' in surface) this.canonicalSourceIndex(surface)
      else {
        const element = surface as HTMLImageElement
        if (!this.canonicalSourceIndexByElement.has(element)) {
          const sourceIndex = this.nextCanonicalSourceIndex++
          this.canonicalSourceIndexByElement.set(element, sourceIndex)
          this.includeChapterPage(sourceIndex)
        }
      }
    }
  }

  private completionKey(): string {
    return this.surfaceDiscovery.completionKey()
  }

  async start(
    scope: TranslationScope,
    hskLevel: 1 | 2 | 3 | 4 | 5 | 6,
    learningMode: LearningMode,
    readingDirection: ReadingDirection,
  ): Promise<PageState> {
    this.scope = undefined
    this.hskLevel = hskLevel
    this.learningMode = learningMode
    this.readingDirection = readingDirection
    const token = await this.run.start(0, 'Preparing the manga or webtoon reader')
    this.surfaceDiscovery.setActive(true)
    try {
      this.scope = scope
      const candidates = this.currentCandidates().filter(
        (candidate) => scope === 'all' || candidate.visible,
      )
      const deferred = scope === 'all' ? this.surfaceDiscovery.deferred().length : 0
      this.run.update({
        current: 0,
        total: candidates.length + deferred,
        message:
          candidates.length === 0 && deferred > 0
            ? 'Waiting for the chapter images'
            : 'Preparing the manga or webtoon reader',
      })
      this.runState.reset()
      this.failures.clear()
      this.failedItems.clear()
      this.cancelCompletion()
      const generation = token.generation
      this.establishCanonicalPageOrder(candidates)
      const unsupportedSurfaceCount = this.surfaceDiscovery.unsupported().length
      if (candidates.length === 0) {
        if (deferred > 0) {
          this.run.update({
            current: 0,
            total: deferred,
            message: 'Waiting for the chapter images',
          })
          this.scheduleFinish()
          return this.snapshot()
        }
        return this.run.finish(
          { current: 0, total: 0 },
          unsupportedSurfaceCount > 0
            ? 'This reader hides its artwork from the extension.'
            : 'No manga images were found on this page.',
          true,
        )
      }
      if (unsupportedSurfaceCount > 0) {
        this.run.update({
          current: 0,
          total: candidates.length,
          message: 'Some reader content cannot be accessed safely',
        })
      }

      const recoveryCandidates = await Promise.all(
        candidates.map((candidate) => this.buildRecoveryCandidate(candidate, generation)),
      )
      if (generation !== this.generation || this.navigationUrl !== location.href) {
        throw chapterAbortError()
      }
      const recovered = await this.run.recover(token, recoveryCandidates)
      const recoveredByIdentity = new Map(
        recovered
          .filter((job): job is RecoveredImageJob => job.kind === 'image')
          .map((job) => [
            recoveryKey(job.sourceUrl, job.sourceWidth, job.sourceHeight, job.sourceIndex),
            job,
          ]),
      )
      this.queue.beginBatch()
      try {
        for (const candidate of candidates) {
          this.enqueue(
            candidate,
            recoveredByIdentity.get(
              recoveryKey(
                candidate.sourceUrl,
                candidate.sourceWidth,
                candidate.sourceHeight,
                this.canonicalSourceIndex(candidate),
              ),
            ),
          )
        }
      } finally {
        this.queue.endBatch()
      }
      this.run.update({
        current: 0,
        total: this.runState.snapshot().total,
        message: 'Waiting to start',
      })
      return this.snapshot()
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') throw error
      const status = this.runState.snapshot()
      this.scope = undefined
      this.run.fail(
        error instanceof Error ? error.message : 'The image chapter could not be started.',
        { current: status.resolved, total: status.total },
      )
      throw error
    }
  }

  cancel(): PageState {
    const run = this.runState.snapshot()
    return this.run.cancel({ current: run.completed, total: run.total })
  }

  snapshot(): PageState {
    return this.run.snapshot()
  }

  diagnostics(): {
    run: ReturnType<ChapterRunState<HTMLElement>['snapshot']>
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
    this.surfaceDiscovery.stop()
    this.run.destroy()
  }

  private async buildRecoveryCandidate(
    candidate: DiscoveredSurface,
    generation: number,
  ): Promise<RecoveryCandidate> {
    const sourceUrl = normalizedSourceUrl(candidate.sourceUrl)
    const identity = {
      kind: 'image' as const,
      sourceUrl,
      naturalWidth: candidate.sourceWidth,
      naturalHeight: candidate.sourceHeight,
      sourceIndex: this.canonicalSourceIndex(candidate),
      settings: this.translationSettings(),
      readingDirection: this.readingDirection,
    }
    const protocol = new URL(sourceUrl).protocol
    if ((protocol === 'http:' || protocol === 'https:') && !candidate.capture) return identity
    const inline = await tryContentBytes(candidate)
    if (generation !== this.generation || currentSourceUrl(candidate) !== sourceUrl) {
      throw chapterAbortError()
    }
    if (!inline) return identity
    const sourceSha256 = await sha256Hex(inline.bytes)
    if (generation !== this.generation || currentSourceUrl(candidate) !== sourceUrl) {
      throw chapterAbortError()
    }
    return { ...identity, sourceSha256 }
  }

  private translationSettings(): TranslationSettings {
    return {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel: this.hskLevel,
      learningMode: this.learningMode,
    }
  }

  private restoreAll(): void {
    this.surfaceDiscovery.setActive(false)
    this.cancelCompletion()
    this.queue.cancelAll()
    this.clearPrefetch()
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
    this.failedItems.clear()
    this.canonicalSourceIndexByElement.clear()
    this.submittedSourceIndexes.clear()
    this.submittingSourceIndexes.clear()
    this.nextCanonicalSourceIndex = 0
    this.chapterSourceOrder = []
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
        supported = candidate.kind === 'image' && (protocol === 'http:' || protocol === 'https:')
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
      sourceIndex: this.canonicalSourceIndex(candidate),
      imageUrl: candidate.sourceUrl,
      pageUrl: this.navigationUrl,
      naturalWidth: candidate.sourceWidth,
      naturalHeight: candidate.sourceHeight,
    }).catch(() => undefined)
  }

  private enqueue(candidate: DiscoveredSurface, recovered?: RecoveredImageJob): void {
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
        recovered.sourceWidth !== candidate.sourceWidth ||
        recovered.sourceHeight !== candidate.sourceHeight)
    ) {
      return
    }
    const sourceIndex = this.canonicalSourceIndex(candidate)
    const id = candidateKey(candidate)
    if (!this.runState.register(candidate.element)) return
    this.cancelCompletion()
    this.queueIds.set(candidate.element, id)
    // The chapter HUD owns queue progress. An image notice is mounted when
    // processing creates its document-anchored wrapper, so it never has to
    // chase the image during normal scrolling.
    const accepted = this.queue.enqueue({
      id,
      value: {
        candidate,
        ...(recovered ? { recovered } : {}),
      },
      visible: candidate.visible,
      order: this.orderingIndex(candidate),
      cost: candidate.sourceWidth * candidate.sourceHeight,
    })
    if (!accepted) {
      this.queueIds.delete(candidate.element)
      this.runState.remove(candidate.element)
    } else {
      this.refreshPrefetch()
    }
  }

  private badge(image: HTMLElement, anchor?: HTMLElement): ImageStatusBadge {
    const existing = this.badges.get(image)
    if (existing) {
      if (anchor) existing.attach(anchor)
      return existing
    }
    const badge = new ImageStatusBadge(image, () => this.retry(image))
    if (anchor) badge.attach(anchor)
    this.badges.set(image, badge)
    return badge
  }

  private retry(image: HTMLElement, itemId?: string): void {
    if (!this.runState.manualRetryQueued(image)) return
    this.failures.delete(image)
    if (this.requeueFailedImage(image, 'Trying again', itemId)) {
      const run = this.runState.snapshot()
      this.run.update({ current: run.resolved, total: run.total })
      return
    }
    this.runState.start(image)
    this.runState.fail(image)
    this.badge(image).failure("This image couldn't be translated. Try again.")
  }

  private requeueFailedImage(image: HTMLElement, status: string, itemId?: string): boolean {
    const candidate = this.currentCandidates().find((item) => item.element === image)
    const failedId = this.queueIds.get(image)
    this.processed.delete(image)
    if (!candidate || !failedId) return false
    const queued = this.queue.retry({
      id: failedId,
      value: {
        candidate,
        retryItemIds: itemId ? [itemId] : [...(this.failedItems.get(image) ?? [])],
      },
      visible: candidate.visible,
      order: this.orderingIndex(candidate),
      cost: candidate.sourceWidth * candidate.sourceHeight,
    })
    if (!queued) return false
    this.badge(image).update(status)
    this.refreshPrefetch()
    return true
  }

  private sourceSnapshot(candidate: DiscoveredSurface): SourceSnapshot {
    return {
      generation: this.generation,
      pageSessionId: this.sessionId,
      navigationUrl: this.navigationUrl,
      sourceUrl: normalizedSourceUrl(candidate.sourceUrl),
      naturalWidth: candidate.sourceWidth,
      naturalHeight: candidate.sourceHeight,
    }
  }

  private async verifySource(
    candidate: DiscoveredSurface,
    snapshot: SourceSnapshot,
    signal: AbortSignal,
  ): Promise<void> {
    this.assertCurrent(candidate, snapshot, signal)
    if (!snapshot.contentRevision) return
    await waitForVisibleSource(candidate.element, signal)
    this.assertCurrent(candidate, snapshot, signal)
    const capture = snapshot.capturedRevision ? await candidate.capture?.(signal) : undefined
    const revision = snapshot.capturedRevision
      ? capture
        ? await sha256Hex(capture.bytes)
        : 'unreadable'
      : await sendBackgroundMessage({ type: 'source:image-revision', jobId: snapshot.jobId! })
    this.assertCurrent(candidate, snapshot, signal)
    if (revision !== snapshot.contentRevision) {
      this.surfaceDiscovery.recordSourceRevision(candidate.id, snapshot.contentRevision)
      this.removeTracked(candidate.element)
      throw new RuntimeMessageError(
        'SOURCE_REVISION_CHANGED',
        'This page changed during translation. Retry its current content.',
        true,
      )
    }
  }

  private assertCurrent(
    candidate: DiscoveredSurface,
    snapshot: SourceSnapshot,
    signal: AbortSignal,
  ): void {
    this.run.assertCurrent(
      {
        generation: snapshot.generation,
        pageSessionId: snapshot.pageSessionId,
        pageUrl: snapshot.navigationUrl,
      },
      signal,
    )
    if (
      !candidate.owner.isConnected ||
      !candidate.element.isConnected ||
      currentSourceUrl(candidate) !== snapshot.sourceUrl ||
      candidate.sourceWidth !== snapshot.naturalWidth ||
      candidate.sourceHeight !== snapshot.naturalHeight
    ) {
      throw chapterAbortError()
    }
  }

  private async process(item: QueueItem<TranslationCandidate>, signal: AbortSignal): Promise<void> {
    const { candidate, recovered } = item.value
    const retryItemIds = item.value.retryItemIds ?? []
    const runToken = this.run.currentToken()
    const sourceIndex = this.canonicalSourceIndex(candidate)
    if (recovered?.jobId) this.submittedSourceIndexes.add(sourceIndex)
    // Recovery is a one-shot attempt. If viewport preemption requeues this
    // item, its recovered daemon job has been cancelled and the retry must
    // submit a fresh job rather than polling a terminal identity.
    delete item.value.recovered
    const consumesPrefetch = this.prefetchTargetId === item.id
    if (consumesPrefetch) this.prefetchTargetId = undefined
    this.prefetchEnabled = false
    const snapshot = this.sourceSnapshot(candidate)
    if (recovered) {
      snapshot.contentRevision = recovered.sourceSha256
      snapshot.capturedRevision = candidate.captureOnly === true
      snapshot.jobId = recovered.jobId
    }
    const badge = this.badge(candidate.element)
    let jobId = recovered?.jobId
    let sourceSha256 = recovered?.sourceSha256
    let sourceUrl = recovered?.sourceUrl
    let sourceWidth = recovered?.sourceWidth
    let sourceHeight = recovered?.sourceHeight
    let after = recovered?.acknowledgedSequence ?? 0
    let rendered: RenderedImage | undefined
    let viewportReporter: ImageFocusReporter | undefined
    const cancelOnAbort = (): void => {
      if (jobId) this.run.cancelJob(jobId)
    }
    signal.addEventListener('abort', cancelOnAbort, { once: true })
    try {
      this.assertCurrent(candidate, snapshot, signal)
      if (!jobId) {
        badge.update('Opening the image')
        const inline = consumesPrefetch ? undefined : await tryContentBytes(candidate)
        this.assertCurrent(candidate, snapshot, signal)
        if (inline && candidate.capture) {
          snapshot.capturedRevision = true
          snapshot.contentRevision = await sha256Hex(inline.bytes)
          this.surfaceDiscovery.recordSourceRevision(candidate.id, snapshot.contentRevision)
        }
        this.submittingSourceIndexes.add(sourceIndex)
        let submitted: Awaited<ReturnType<typeof sendBackgroundMessage<'job:submit-image'>>>
        try {
          submitted = await sendBackgroundMessage({
            type: 'job:submit-image',
            clientRequestId: crypto.randomUUID(),
            retryItemIds,
            pageSessionId: runToken.pageSessionId,
            sourceIndex,
            chapterSourceOrder: [...this.chapterSourceOrder],
            surfaceKind: candidate.kind,
            imageUrl: candidate.sourceUrl,
            pageUrl: runToken.pageUrl,
            naturalWidth: snapshot.naturalWidth,
            naturalHeight: snapshot.naturalHeight,
            ...(inline?.mimeType ? { sourceMimeType: inline.mimeType } : {}),
            ...(inline ? { sourceBytes: inline.bytes } : {}),
            hskLevel: this.hskLevel,
            learningMode: this.learningMode,
            readingDirection: this.readingDirection,
            visibleRects: visibleImageRects(
              candidate.element,
              snapshot.naturalWidth,
              snapshot.naturalHeight,
            ),
          })
        } finally {
          this.submittingSourceIndexes.delete(sourceIndex)
        }
        // Retain the identity before checking the live DOM so a navigation or
        // source replacement that happened during submission can cancel the
        // newly-created companion job instead of leaking it.
        jobId = submitted.jobId
        this.run.registerJob(runToken, jobId)
        sourceSha256 = submitted.sourceSha256
        snapshot.contentRevision ??= sourceSha256
        snapshot.jobId = jobId
        sourceUrl = submitted.sourceUrl
        sourceWidth = submitted.sourceWidth
        sourceHeight = submitted.sourceHeight
        after = submitted.acknowledgedSequence
        this.submittedSourceIndexes.add(sourceIndex)
        void sendBackgroundMessage({
          type: 'chapter:source',
          pageSessionId: runToken.pageSessionId,
          pageUrl: runToken.pageUrl,
          sourceIndex,
        }).catch(() => undefined)
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

      this.run.registerJob(runToken, jobId)
      this.prefetchEnabled = true
      this.refreshPrefetch()
      rendered =
        this.rendered.get(candidate.element) ??
        this.renderer.begin(
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
      badge.attach(rendered.wrapper)
      viewportReporter = new ImageFocusReporter(jobId, candidate.element, sourceWidth, sourceHeight)

      const activeJobId = jobId
      const activeRenderer = rendered
      const translatedItemIds = new Set<string>()
      const preservedItemIds = new Set<string>()
      const failedItemIds = new Set<string>()
      const terminal = await this.run.stream(
        runToken,
        activeJobId,
        after,
        signal,
        async (update) => {
          this.assertCurrent(candidate, snapshot, signal)
          switch (update.type) {
            case 'progress':
              badge.update(update)
              {
                const run = this.runState.snapshot()
                this.run.update({
                  current: run.resolved,
                  total: run.total,
                  key: item.id,
                  status: update,
                })
              }
              break
            case 'imageRegionReady': {
              if (
                translatedItemIds.has(update.region.itemId) ||
                preservedItemIds.has(update.region.itemId)
              ) {
                throw new RuntimeMessageError(
                  'DUPLICATE_IMAGE_ITEM',
                  'The image job published more than one terminal result for an item.',
                  false,
                )
              }
              badge.update('Adding translated text')
              if (retryItemIds.length && !retryItemIds.includes(update.region.itemId))
                throw new RuntimeMessageError(
                  'RETRY_ITEM_MISMATCH',
                  'A retry returned an unrequested source item.',
                  false,
                )
              const patch = await sendBackgroundMessage({
                type: 'job:patch',
                jobId: activeJobId,
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
              await activeRenderer.installRegion(update.region, patch.bytes, {
                signal,
                validate: () => this.assertCurrent(candidate, snapshot, signal),
                verify: () => this.verifySource(candidate, snapshot, signal),
              })
              translatedItemIds.add(update.region.itemId)
              break
            }
            case 'imageRegionPreserved':
              if (
                translatedItemIds.has(update.region.itemId) ||
                preservedItemIds.has(update.region.itemId)
              ) {
                throw new RuntimeMessageError(
                  'DUPLICATE_IMAGE_ITEM',
                  'The image job published more than one terminal result for an item.',
                  false,
                )
              }
              if (retryItemIds.length && !retryItemIds.includes(update.region.itemId))
                throw new RuntimeMessageError(
                  'RETRY_ITEM_MISMATCH',
                  'A retry returned an unrequested source item.',
                  false,
                )
              if (update.region.disposition === 'failed') {
                activeRenderer.installSourcePreservingRegion(update.region)
                failedItemIds.add(update.region.itemId)
              }
              preservedItemIds.add(update.region.itemId)
              break
            case 'documentBlockReady':
            case 'documentBlockPreserved':
              throw new RuntimeMessageError(
                'JOB_MODALITY_MISMATCH',
                'A document update was returned for an image job.',
                false,
              )
          }
        },
      )
      if (
        terminal.translatedCount !== translatedItemIds.size ||
        terminal.preservedCount !== preservedItemIds.size ||
        (retryItemIds.length > 0 &&
          translatedItemIds.size + preservedItemIds.size !== retryItemIds.length)
      ) {
        throw new RuntimeMessageError(
          'IMAGE_COMPLETION_MISMATCH',
          'The completed image job does not match its published item counts.',
          false,
        )
      }

      await viewportReporter.stop().catch(() => undefined)
      viewportReporter = undefined
      this.assertCurrent(candidate, snapshot, signal)
      const remainingFailures = retryItemIds.length
        ? new Set(this.failedItems.get(candidate.element))
        : new Set<string>()
      for (const id of translatedItemIds) remainingFailures.delete(id)
      for (const id of failedItemIds) remainingFailures.add(id)
      this.failedItems.set(candidate.element, remainingFailures)
      if (remainingFailures.size)
        throw new RuntimeMessageError(
          'IMAGE_ITEMS_FAILED',
          `${activeRenderer.regionCount} translated; ${remainingFailures.size} failed. Original and Compare remain available.`,
          true,
        )
      this.failedItems.delete(candidate.element)
      this.processed.add(candidate.element)
      badge.destroy()
      this.badges.delete(candidate.element)
    } catch (error) {
      await viewportReporter?.stop().catch(() => undefined)
      viewportReporter = undefined
      if (jobId) this.run.cancelJob(jobId)
      if (
        rendered &&
        rendered.regionCount === 0 &&
        !this.failedItems.get(candidate.element)?.size &&
        !this.processed.has(candidate.element)
      ) {
        // Move a failure notice out before the renderer removes its anchored
        // wrapper. The fallback is positioned once at document coordinates;
        // the next attempt attaches it to the new wrapper.
        badge.detach()
        rendered.destroy()
        if (this.rendered.get(candidate.element) === rendered) {
          this.rendered.delete(candidate.element)
        }
      }
      if (
        !this.submittedSourceIndexes.has(sourceIndex) &&
        !this.submittingSourceIndexes.has(sourceIndex)
      ) {
        this.chapterSourceOrder = this.chapterSourceOrder.filter((index) => index !== sourceIndex)
      }
      throw error
    } finally {
      this.prefetchEnabled = false
      signal.removeEventListener('abort', cancelOnAbort)
      if (jobId) this.run.forgetJob(jobId)
    }
  }

  private removeTracked(image: HTMLElement): void {
    let tracked = false
    if (this.runState.phase(image) !== 'running') this.removeUnsubmittedSource(image)
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
    this.failedItems.delete(image)
    this.badges.get(image)?.destroy()
    this.badges.delete(image)
    if (tracked) this.scheduleFinish()
  }

  private onDiscovery(event: SurfaceDiscoveryEvent): void {
    const image = event.candidate.element
    if (event.type === 'added' || event.type === 'updated')
      this.canonicalSourceIndex(event.candidate)
    if (event.type === 'visibility') {
      const id = this.queueIds.get(image)
      if (id) {
        this.queue.reprioritize(id, event.candidate.visible, this.orderingIndex(event.candidate))
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
      if (event.previousSourceUrl === event.candidate.sourceUrl && event.sourceChanged !== true) {
        const id = this.queueIds.get(image)
        if (id) {
          this.queue.reprioritize(id, event.candidate.visible, this.orderingIndex(event.candidate))
        }
        this.refreshPrefetch()
        return
      }
      this.removeTracked(image)
      if (this.scope && !this.cancelledState) this.enqueue(event.candidate)
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
    const completionKey = this.completionKey()
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
      if (completionKey !== this.completionKey()) {
        this.scheduleFinish()
        return
      }
      this.finish()
    }, COMPLETION_SETTLE_MS)
  }

  private finish(): void {
    if (!this.scope || this.cancelledState) return
    const run = this.runState.snapshot()
    const deferred = this.scope === 'all' ? this.surfaceDiscovery.deferred().length : 0
    if (deferred > 0) {
      this.run.update({
        current: run.resolved,
        total: run.total + deferred,
        message: 'Waiting for the remaining chapter images',
      })
      this.scheduleFinish()
      return
    }
    if (this.queue.size > 0 || run.unresolved > 0) {
      this.run.update({
        current: run.resolved,
        total: run.total,
      })
      this.scheduleFinish()
      return
    }
    if (!run.allResolved) {
      this.run.finish(
        { current: 0, total: 0 },
        'No discovered chapter images remain. Waiting for new content.',
        true,
      )
      return
    }
    if (run.failed > 0) {
      this.run.finish(
        { current: run.completed, total: run.total },
        `${run.failed} image${run.failed === 1 ? '' : 's'} still needs attention.`,
        true,
      )
    } else {
      this.run.finish({ current: run.completed, total: run.total }, undefined, true)
    }
  }
}
