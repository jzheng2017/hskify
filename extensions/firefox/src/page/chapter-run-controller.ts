import {
  sendBackgroundMessage,
  type PageState,
  type RecoveredJob,
  type RecoveryCandidate,
} from '../messaging/messages'
import { PageHud } from '../progress/hud'
import { createPageSessionId } from './chapter-session'
import { JobUpdateStream, type StreamingJobUpdate, type TerminalJobUpdate } from './job-stream'

type SupportedChapterKind = 'image' | 'document'

export type ChapterRunToken = Readonly<{
  generation: number
  pageSessionId: string
  pageUrl: string
  signal: AbortSignal
}>

export type ChapterRunIdentity = Pick<ChapterRunToken, 'generation' | 'pageSessionId' | 'pageUrl'>

export type ChapterRunCounts = Readonly<{
  current: number
  total: number
}>

export function chapterAbortError(): Error {
  const error = new Error('The operation was cancelled.')
  error.name = 'AbortError'
  return error
}

/**
 * Owns the protocol and UI lifetime that is identical for image and document
 * chapters. Mode adapters keep only source discovery and rendering details.
 */
export class ChapterRunController {
  private pageSessionIdValue = createPageSessionId(true)
  private pageUrlValue = location.href
  private generationValue = 0
  private abortController: AbortController | undefined
  private hudValue: PageHud | undefined
  private readonly activeJobIds = new Set<string>()
  private readonly updates = new JobUpdateStream()
  private hasRun = false
  private chapterClosed = false
  private chapterClosePromise: Promise<void> | undefined
  private pageReleased = false
  private releasing = false
  private running = false
  private cancelled = false
  private completed = false
  private destroyed = false
  private state: PageState

  constructor(
    private readonly contentKind: SupportedChapterKind,
    private readonly restoreMode: () => void,
    private readonly hudRoot: HTMLElement = document.documentElement,
  ) {
    this.state = {
      state: 'idle',
      contentKind,
      current: 0,
      total: 0,
      message: contentKind === 'document' ? 'Light-novel chapter detected' : 'Ready',
    }
  }

  get sessionId(): string {
    return this.pageSessionIdValue
  }
  get pageUrl(): string {
    return this.pageUrlValue
  }
  get generation(): number {
    return this.generationValue
  }
  get hud(): PageHud | undefined {
    return this.hudValue
  }
  get isRunning(): boolean {
    return this.running
  }
  get isCancelled(): boolean {
    return this.cancelled
  }
  get isComplete(): boolean {
    return this.completed
  }

  currentToken(): ChapterRunToken {
    const signal = this.abortController?.signal
    if (!this.running || !signal) throw chapterAbortError()
    return {
      generation: this.generationValue,
      pageSessionId: this.pageSessionIdValue,
      pageUrl: this.pageUrlValue,
      signal,
    }
  }

  async start(total: number, message: string): Promise<ChapterRunToken> {
    if (this.destroyed) throw new Error('The chapter mode has been destroyed.')

    const previousSessionId = this.hasRun ? this.pageSessionIdValue : undefined
    const previousChapterOpen = this.hasRun && !this.chapterClosed
    const previousPageReleased = this.pageReleased
    const previousClose = previousChapterOpen
      ? this.closeChapter('chapter:cancel')
      : this.chapterClosePromise
    this.releasing = true
    this.invalidateLocalRun()
    if (this.hasRun) {
      this.restoreMode()
      this.pageSessionIdValue = createPageSessionId(false)
    }
    this.pageUrlValue = location.href
    this.hasRun = true
    this.chapterClosed = false
    this.chapterClosePromise = undefined
    this.pageReleased = false
    this.releasing = false
    this.running = true
    this.cancelled = false
    this.completed = false
    const controller = new AbortController()
    this.abortController = controller
    const generation = ++this.generationValue
    const token: ChapterRunToken = {
      generation,
      pageSessionId: this.pageSessionIdValue,
      pageUrl: this.pageUrlValue,
      signal: controller.signal,
    }

    this.hudValue?.destroy()
    this.hudValue = new PageHud(
      () => this.cancel({ current: this.state.current, total: this.state.total }),
      this.hudRoot,
      this.contentKind,
    )
    this.update({ current: 0, total, message })

    try {
      if (previousSessionId) {
        await previousClose?.catch(() => undefined)
        if (!previousPageReleased) {
          await sendBackgroundMessage({
            type: 'jobs:cancel-page',
            pageSessionId: previousSessionId,
          })
        }
        this.assertCurrent(token)
      }
      await sendBackgroundMessage({
        type: 'chapter:start',
        pageSessionId: token.pageSessionId,
        pageUrl: token.pageUrl,
        contentKind: this.contentKind,
      })
      this.assertCurrent(token)
      return token
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') throw error
      this.fail(error instanceof Error ? error.message : 'The chapter could not be started.', {
        current: 0,
        total,
      })
      throw error
    }
  }

  async recover(token: ChapterRunToken, candidates: RecoveryCandidate[]): Promise<RecoveredJob[]> {
    this.assertCurrent(token)
    const recovered = await sendBackgroundMessage({
      type: 'jobs:recover',
      pageSessionId: token.pageSessionId,
      pageUrl: token.pageUrl,
      candidates,
    })
    this.assertCurrent(token)
    return recovered
  }

  registerJob(token: ChapterRunToken, jobId: string): void {
    this.activeJobIds.add(jobId)
    try {
      this.assertCurrent(token)
    } catch (error) {
      this.cancelJob(jobId)
      throw error
    }
  }

  forgetJob(jobId: string): void {
    this.activeJobIds.delete(jobId)
  }

  cancelJob(jobId: string): void {
    this.activeJobIds.delete(jobId)
    if (this.releasing || this.pageReleased) return
    void sendBackgroundMessage({ type: 'job:cancel', jobId }).catch(() => undefined)
  }

  async stream(
    token: ChapterRunToken,
    jobId: string,
    after: number,
    signal: AbortSignal,
    install: (update: StreamingJobUpdate) => void | Promise<void>,
  ): Promise<Extract<TerminalJobUpdate, { type: 'complete' }>> {
    this.registerJob(token, jobId)
    try {
      const terminal = await this.updates.run(jobId, after, signal, async (update) => {
        this.assertCurrent(token)
        await install(update)
      })
      this.assertCurrent(token)
      this.forgetJob(jobId)
      return terminal
    } catch (error) {
      // Keep a non-terminal job registered so cancel/fail can reliably stop it.
      throw error
    }
  }

  update(input: Parameters<PageHud['update']>[0]): void {
    this.hudValue?.update(input)
    this.state = this.hudValue?.snapshot() ?? {
      state: 'running',
      contentKind: this.contentKind,
      current: input.current,
      total: input.total,
      message: input.message ?? 'Translating chapter',
    }
  }

  completeItem(key: string): void {
    this.hudValue?.completeImage(key)
  }

  finish(counts: ChapterRunCounts, failureMessage?: string, keepAliveForRetry = false): PageState {
    if (!this.running) return this.snapshot()
    this.running = failureMessage !== undefined && keepAliveForRetry
    this.completed = failureMessage === undefined
    this.cancelled = false
    if (!this.running) this.abortController = undefined
    if (failureMessage) this.hudValue?.fail(failureMessage, counts.current, counts.total)
    else this.hudValue?.complete(counts.current, counts.total)
    this.state = this.hudValue?.snapshot() ?? {
      state: failureMessage ? 'failed' : 'complete',
      contentKind: this.contentKind,
      current: counts.current,
      total: counts.total,
      message: failureMessage ?? `${counts.current} of ${counts.total} items ready`,
    }
    if (!this.running) void this.closeChapter('chapter:finish')
    return this.snapshot()
  }

  fail(message: string, counts: ChapterRunCounts): PageState {
    ++this.generationValue
    this.releasing = true
    this.abortController?.abort()
    this.abortController = undefined
    this.restoreMode()
    this.running = false
    this.completed = false
    this.cancelled = false
    const close = this.hasRun ? this.closeChapter('chapter:cancel') : undefined
    this.releaseCurrentPage(close)
    this.hudValue?.fail(message, counts.current, counts.total)
    this.state = this.hudValue?.snapshot() ?? {
      state: 'failed',
      contentKind: this.contentKind,
      current: counts.current,
      total: counts.total,
      message,
    }
    return this.snapshot()
  }

  cancel(counts: ChapterRunCounts): PageState {
    ++this.generationValue
    this.releasing = true
    this.abortController?.abort()
    this.abortController = undefined
    this.restoreMode()
    this.running = false
    this.completed = false
    this.cancelled = true
    const close = this.hasRun ? this.closeChapter('chapter:cancel') : undefined
    this.releaseCurrentPage(close)
    this.hudValue?.cancelled(counts.current, counts.total)
    this.state = this.hudValue?.snapshot() ?? {
      state: 'cancelled',
      contentKind: this.contentKind,
      current: counts.current,
      total: counts.total,
      message: 'Anything unfinished was left unchanged',
    }
    return this.snapshot()
  }

  snapshot(): PageState {
    return { ...(this.hudValue?.snapshot() ?? this.state) }
  }

  assertCurrent(identity: ChapterRunIdentity, signal?: AbortSignal): void {
    if (
      this.destroyed ||
      !this.running ||
      this.generationValue !== identity.generation ||
      this.pageSessionIdValue !== identity.pageSessionId ||
      this.pageUrlValue !== identity.pageUrl ||
      location.href !== identity.pageUrl ||
      this.abortController?.signal.aborted ||
      signal?.aborted
    ) {
      throw chapterAbortError()
    }
  }

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    ++this.generationValue
    const wasRunning = this.running
    this.releasing = true
    this.abortController?.abort()
    this.abortController = undefined
    this.restoreMode()
    this.running = false
    const close = wasRunning && this.hasRun
      ? this.closeChapter('chapter:cancel')
      : this.chapterClosePromise
    this.releaseCurrentPage(close)
    this.hudValue?.destroy()
    this.hudValue = undefined
  }

  private invalidateLocalRun(): void {
    this.abortController?.abort()
    this.abortController = undefined
    this.activeJobIds.clear()
    this.running = false
  }

  private releaseCurrentPage(after?: Promise<unknown>): void {
    if (!this.hasRun || this.pageReleased) return
    this.pageReleased = true
    this.activeJobIds.clear()
    const pageSessionId = this.pageSessionIdValue
    void (after ?? Promise.resolve())
      .catch(() => undefined)
      .then(() => sendBackgroundMessage({ type: 'jobs:cancel-page', pageSessionId }))
      .catch(() => undefined)
  }

  private closeChapter(type: 'chapter:finish' | 'chapter:cancel'): Promise<void> {
    if (this.chapterClosed) return this.chapterClosePromise ?? Promise.resolve()
    this.chapterClosed = true
    const pageSessionId = this.pageSessionIdValue
    const pageUrl = this.pageUrlValue
    const close = sendBackgroundMessage({
      type,
      pageSessionId,
      pageUrl,
    })
    this.chapterClosePromise = close
    void close.catch(() => undefined)
    return close
  }
}
