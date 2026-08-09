import type { ChapterKind } from '../contracts/browser'

export type ChapterLifecyclePhase = 'started' | 'active' | 'finished' | 'cancelled'
export type SupportedChapterKind = Exclude<ChapterKind, 'unsupported'>

export type ChapterLifecycleState = Readonly<{
  pageSessionId: string
  pageUrl: string
  contentKind: SupportedChapterKind
  phase: ChapterLifecyclePhase
  highestSourceIndex: number
  submittedSources: number
  focusRevision: number
}>

type MutableChapter = {
  pageSessionId: string
  pageUrl: string
  contentKind: SupportedChapterKind
  phase: ChapterLifecyclePhase
  highestSourceIndex: number
  submittedSources: Set<number>
  focusRevision: number
}

export class ChapterLifecycleStore {
  private readonly chapters = new Map<string, MutableChapter>()

  start(
    pageSessionId: string,
    pageUrl: string,
    contentKind: SupportedChapterKind,
  ): ChapterLifecycleState {
    const existing = this.chapters.get(pageSessionId)
    if (
      existing && existing.pageUrl === pageUrl && existing.contentKind === contentKind &&
      (existing.phase === 'started' || existing.phase === 'active')
    ) return this.snapshot(existing)
    const chapter: MutableChapter = {
      pageSessionId, pageUrl, contentKind, phase: 'started', highestSourceIndex: -1,
      submittedSources: new Set(), focusRevision: 0,
    }
    this.chapters.set(pageSessionId, chapter)
    return this.snapshot(chapter)
  }

  source(pageSessionId: string, pageUrl: string, sourceIndex: number): ChapterLifecycleState {
    const chapter = this.require(pageSessionId, pageUrl)
    if (chapter.phase === 'finished' || chapter.phase === 'cancelled') return this.snapshot(chapter)
    chapter.phase = 'active'
    chapter.highestSourceIndex = Math.max(chapter.highestSourceIndex, sourceIndex)
    chapter.submittedSources.add(sourceIndex)
    return this.snapshot(chapter)
  }

  focus(pageSessionId: string, pageUrl: string): ChapterLifecycleState {
    const chapter = this.require(pageSessionId, pageUrl)
    if (chapter.phase === 'started' || chapter.phase === 'active') {
      chapter.phase = 'active'
      chapter.focusRevision += 1
    }
    return this.snapshot(chapter)
  }

  finish(pageSessionId: string, pageUrl: string): ChapterLifecycleState {
    const chapter = this.require(pageSessionId, pageUrl)
    if (chapter.phase !== 'cancelled') chapter.phase = 'finished'
    return this.snapshot(chapter)
  }

  cancel(pageSessionId: string, pageUrl: string): ChapterLifecycleState {
    const chapter = this.require(pageSessionId, pageUrl)
    chapter.phase = 'cancelled'
    return this.snapshot(chapter)
  }

  remove(pageSessionId: string): void { this.chapters.delete(pageSessionId) }

  state(pageSessionId: string): ChapterLifecycleState | undefined {
    const chapter = this.chapters.get(pageSessionId)
    return chapter ? this.snapshot(chapter) : undefined
  }

  private require(pageSessionId: string, pageUrl: string): MutableChapter {
    const chapter = this.chapters.get(pageSessionId)
    if (!chapter || chapter.pageUrl !== pageUrl) throw new Error('The chapter session is not active for this document.')
    return chapter
  }

  private snapshot(chapter: MutableChapter): ChapterLifecycleState {
    return Object.freeze({
      pageSessionId: chapter.pageSessionId, pageUrl: chapter.pageUrl,
      contentKind: chapter.contentKind, phase: chapter.phase,
      highestSourceIndex: chapter.highestSourceIndex,
      submittedSources: chapter.submittedSources.size,
      focusRevision: chapter.focusRevision,
    })
  }
}
