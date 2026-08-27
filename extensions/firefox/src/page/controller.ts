import type { LearningMode, ReadingDirection } from '../contracts/browser'
import { looksLikeSequentialArtReader } from '../discovery/images'
import { discoverPageSurfaces } from '../discovery/surfaces'
import { detectDocumentChapter } from '../document'
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

function imageChapterDetected(): boolean {
  if (looksLikeSequentialArtReader()) return true
  return discoverPageSurfaces().surfaces.some(
    (surface) =>
      surface.kind !== 'image' &&
      (surface.continuous || surface.width >= 500) &&
      surface.height >= 700,
  )
}

/**
 * Owns content classification and navigation. Exactly one mode adapter exists
 * for a document, and a valid Readability descriptor always wins over images.
 */
export class ChapterController {
  private navigationUrl = location.href
  private generation = 0
  private mode: ChapterMode | undefined
  private detection: Promise<void>
  private navigationTimer: number
  private unsupportedState: PageState = {
    state: 'idle',
    contentKind: 'unsupported',
    current: 0,
    total: 0,
    message: 'No supported chapter content detected',
  }

  constructor() {
    this.detection = this.detect()
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
    await this.detection
    if (!this.mode || (this.mode instanceof DocumentChapterMode && this.mode.isInvalidated)) {
      await this.redetect()
    }
    if (!this.mode) {
      this.unsupportedState = {
        state: 'failed',
        contentKind: 'unsupported',
        current: 0,
        total: 0,
        message: 'This page is not a confidently detected light-novel, manga, or webtoon chapter.',
      }
      return this.snapshot()
    }
    return this.mode.start(scope, hskLevel, learningMode, readingDirection)
  }

  cancel(): PageState {
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
    window.clearInterval(this.navigationTimer)
    this.generation += 1
    this.mode?.destroy()
    this.mode = undefined
  }

  private async redetect(): Promise<void> {
    await this.detection
    const previous = this.mode
    const previousKind = previous?.snapshot().contentKind
    const generation = ++this.generation
    const navigationUrl = this.navigationUrl
    const result = await detectDocumentChapter(document).catch(() => undefined)
    if (generation !== this.generation || navigationUrl !== this.navigationUrl) return
    let next: ChapterMode | undefined
    if (result?.kind === 'document') {
      const sameDocument =
        previousKind === 'document' &&
        previous instanceof DocumentChapterMode &&
        !previous.isInvalidated &&
        previous.chapter.snapshot.sourceSha256 === result.chapter.snapshot.sourceSha256
      if (sameDocument) return
      next = new DocumentChapterMode(result.chapter)
    } else if (result?.kind !== 'rejected' && imageChapterDetected()) {
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
    const result = await detectDocumentChapter(document).catch(() => undefined)
    if (generation !== this.generation || navigationUrl !== this.navigationUrl) return
    if (result?.kind === 'document') {
      this.mode = new DocumentChapterMode(result.chapter)
      return
    }
    if (result?.kind !== 'rejected' && imageChapterDetected()) this.mode = new ImageChapterMode()
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

  private checkNavigation(): void {
    if (location.href === this.navigationUrl) return
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
