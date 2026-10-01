import type { LearningMode, ReadingDirection } from '../contracts/browser'
import { classifyChapter } from '../discovery/chapter'
import { discoverPageSurfaces } from '../discovery/surfaces'
import { detectDocumentChapter } from '../document'
import { selectStoryRegion } from '../document/region-selection'
import {
  parseContentRequest,
  RuntimeMessageError,
  type PageState,
  type TranslationScope,
} from '../messaging/messages'
import { DocumentChapterMode } from './document-chapter-mode'
import { ImageChapterMode } from './image-chapter-mode'

export { ImageChapterMode, COMPLETION_SETTLE_MS, tryContentBytes } from './image-chapter-mode'
export { DocumentChapterMode } from './document-chapter-mode'

const NAVIGATION_CHECK_INTERVAL_MS = 250
const CONTENT_MESSAGE_TYPES = new Set(['content:start', 'content:cancel', 'content:state'])

type ChapterMode = {
  start(
    scope: TranslationScope,
    hskLevel: 1 | 2 | 3 | 4 | 5 | 6,
    learningMode: LearningMode,
    readingDirection: ReadingDirection,
  ): Promise<PageState>
  cancel(): PageState
  snapshot(): PageState
  destroy(): void
}

/**
 * Owns content classification and navigation. Exactly one mode adapter exists
 * for a document, and a valid Readability descriptor always wins over images.
 */
export class ChapterController {
  private navigationUrl = location.href
  private generation = 0
  private startRevision = 0
  private mode: ChapterMode | undefined
  private detection: Promise<void>
  private navigationTimer: number
  private readonly observer: MutationObserver
  private detectTimer: number | undefined
  private inputRejected = false
  private selection: AbortController | undefined
  private lastStart:
    | [TranslationScope, 1 | 2 | 3 | 4 | 5 | 6, LearningMode, ReadingDirection]
    | undefined
  private unsupportedState: PageState = {
    state: 'idle',
    contentKind: 'unsupported',
    current: 0,
    total: 0,
    message: 'No supported chapter content detected',
  }

  constructor() {
    this.detection = this.detect()
    this.observer = new MutationObserver((records) => {
      if (this.mode && !(this.mode instanceof DocumentChapterMode && this.mode.isInvalidated))
        return
      if (
        records.every((record) =>
          (record.target instanceof Element ? record.target : record.target.parentElement)?.closest(
            '[data-hskify-owned]',
          ),
        )
      )
        return
      this.scheduleDetection()
    })
    this.observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'data-src', 'style', 'class', 'hidden'],
    })
    this.navigationTimer = window.setInterval(
      () => this.checkNavigation(),
      NAVIGATION_CHECK_INTERVAL_MS,
    )
  }

  async start(
    scope: TranslationScope,
    hskLevel: 1 | 2 | 3 | 4 | 5 | 6,
    learningMode: LearningMode,
    readingDirection: ReadingDirection,
  ): Promise<PageState> {
    this.startRevision += 1
    if (this.detectTimer !== undefined) window.clearTimeout(this.detectTimer)
    this.detectTimer = undefined
    this.lastStart = [scope, hskLevel, learningMode, readingDirection]
    await this.detection
    if (!this.mode || (this.mode instanceof DocumentChapterMode && this.mode.isInvalidated)) {
      await this.redetect()
    }
    if (!this.mode) {
      this.selection?.abort()
      const selection = new AbortController()
      this.selection = selection
      const root = await selectStoryRegion(document, selection.signal)
      if (selection.signal.aborted) return this.snapshot()
      this.selection = undefined
      if (root) {
        const selected = await detectDocumentChapter(document, { root })
        if (selected.kind === 'document')
          this.mode = new DocumentChapterMode(selected.chapter, undefined, () =>
            this.scheduleDetection(),
          )
        else if (
          selected.kind !== 'rejected' &&
          discoverPageSurfaces(document, root).surfaces.length
        )
          this.mode = new ImageChapterMode(root)
      }
    }
    if (!this.mode) {
      this.unsupportedState = {
        state: 'failed',
        contentKind: 'unsupported',
        current: 0,
        total: 0,
        message: this.inputRejected
          ? 'This chapter exceeds the safe document input limits. Select a smaller story region.'
          : 'This page is not a confidently detected light-novel, manga, or webtoon chapter.',
      }
      return this.snapshot()
    }
    return this.mode.start(scope, hskLevel, learningMode, readingDirection)
  }

  cancel(): PageState {
    this.startRevision += 1
    this.lastStart = undefined
    this.selection?.abort()
    if (this.mode) return this.mode.cancel()
    this.unsupportedState = {
      state: 'cancelled',
      contentKind: 'unsupported',
      current: 0,
      total: 0,
      message: 'Cancelled',
    }
    return this.snapshot()
  }

  snapshot(): PageState {
    return this.mode?.snapshot() ?? { ...this.unsupportedState }
  }

  destroy(): void {
    this.selection?.abort()
    window.clearInterval(this.navigationTimer)
    this.observer.disconnect()
    if (this.detectTimer !== undefined) window.clearTimeout(this.detectTimer)
    this.generation += 1
    this.mode?.destroy()
    this.mode = undefined
  }

  private redetect(): Promise<void> {
    const pending = this.detection.then(() => this.detectReplacement())
    this.detection = pending
    return pending
  }

  private async detectReplacement(): Promise<void> {
    if (this.mode && !(this.mode instanceof DocumentChapterMode && this.mode.isInvalidated)) return
    const previous = this.mode
    const previousKind = previous?.snapshot().contentKind
    const generation = ++this.generation
    const navigationUrl = this.navigationUrl
    const result = await classifyChapter(document).catch(() => undefined)
    if (generation !== this.generation || navigationUrl !== this.navigationUrl) return
    this.inputRejected = result?.kind === 'rejected'
    let next: ChapterMode | undefined
    if (result?.kind === 'document') {
      const sameDocument =
        previousKind === 'document' &&
        previous instanceof DocumentChapterMode &&
        !previous.isInvalidated &&
        previous.chapter.snapshot.sourceSha256 === result.chapter.snapshot.sourceSha256
      if (sameDocument) return
      next = new DocumentChapterMode(
        result.chapter,
        previous instanceof DocumentChapterMode ? previous.retainedTranslations() : undefined,
        () => this.scheduleDetection(),
      )
    } else if (result?.kind === 'image') {
      if (previousKind === 'image') return
      next = new ImageChapterMode()
    }
    previous?.destroy()
    this.mode = next
    this.unsupportedState = {
      state: 'idle',
      contentKind: next?.snapshot().contentKind ?? 'unsupported',
      current: 0,
      total: 0,
      message: next
        ? next.snapshot().contentKind === 'document'
          ? 'Light-novel chapter detected'
          : 'Manga or webtoon chapter detected'
        : result?.kind === 'rejected'
          ? 'This chapter exceeds the safe document input limits'
          : 'No supported chapter content detected',
    }
  }

  private async detect(): Promise<void> {
    const generation = this.generation
    const navigationUrl = this.navigationUrl
    const result = await classifyChapter(document).catch(() => undefined)
    if (generation !== this.generation || navigationUrl !== this.navigationUrl) return
    this.inputRejected = result?.kind === 'rejected'
    if (result?.kind === 'document') {
      this.mode = new DocumentChapterMode(result.chapter, undefined, () => this.scheduleDetection())
      return
    }
    if (result?.kind === 'image') this.mode = new ImageChapterMode()
    this.unsupportedState = {
      ...this.unsupportedState,
      contentKind: this.mode?.snapshot().contentKind ?? 'unsupported',
      message: this.mode
        ? 'Manga or webtoon chapter detected'
        : result?.kind === 'rejected'
          ? 'This chapter exceeds the safe document input limits'
          : 'No supported chapter content detected',
    }
  }

  private scheduleDetection(): void {
    if (this.detectTimer !== undefined) return
    this.detectTimer = window.setTimeout(() => {
      this.detectTimer = undefined
      const restart =
        this.mode instanceof DocumentChapterMode && this.mode.isInvalidated
          ? this.lastStart
          : undefined
      const revision = this.startRevision
      void this.redetect().then(async () => {
        if (revision === this.startRevision && restart && this.mode)
          await this.mode.start(...restart).catch(() => undefined)
      })
    }, 100)
  }

  private checkNavigation(): void {
    if (location.href === this.navigationUrl) return
    this.selection?.abort()
    this.startRevision += 1
    this.lastStart = undefined
    this.generation += 1
    this.mode?.cancel()
    this.mode?.destroy()
    this.mode = undefined
    this.navigationUrl = location.href
    this.unsupportedState = {
      state: 'idle',
      contentKind: 'unsupported',
      current: 0,
      total: 0,
      message: 'Detecting chapter content',
    }
    this.detection = this.detect()
  }
}

declare global {
  var __hskifyContentRuntime:
    | {
        controller: ChapterController
        dispose(): void
      }
    | undefined
}

export function bootContentRuntime(): void {
  globalThis.__hskifyContentRuntime?.dispose()
  const controller = new ChapterController()
  document.documentElement.dataset.hskifyInjected = 'true'
  const listener = async (raw: unknown, sender: browser.runtime.MessageSender) => {
    if (
      typeof raw !== 'object' ||
      raw === null ||
      !CONTENT_MESSAGE_TYPES.has(String((raw as Record<string, unknown>).type))
    )
      return undefined
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
          message.readingDirection,
        )
      case 'content:cancel':
        return controller.cancel()
      case 'content:state':
        return controller.snapshot()
    }
  }
  browser.runtime.onMessage.addListener(listener)
  globalThis.__hskifyContentRuntime = {
    controller,
    dispose() {
      browser.runtime.onMessage.removeListener(listener)
      controller.destroy()
      delete document.documentElement.dataset.hskifyInjected
      if (globalThis.__hskifyContentRuntime?.controller === controller) {
        globalThis.__hskifyContentRuntime = undefined
      }
    },
  }
}
