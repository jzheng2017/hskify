export type ChapterLifecyclePhase = 'started' | 'active' | 'sealed' | 'cancelled'

export type ChapterLifecycleState = Readonly<{
  pageSessionId: string
  pageUrl: string
  phase: ChapterLifecyclePhase
  highestPageIndex: number
  submittedPages: number
  viewportRevision: number
}>

type MutableChapter = {
  pageSessionId: string
  pageUrl: string
  phase: ChapterLifecyclePhase
  highestPageIndex: number
  submittedPages: Set<number>
  viewportRevision: number
}

/**
 * One owner for chapter lifecycle state in the background process.  Image
 * jobs may complete out of order, but page registration and sealing are
 * monotonic and keyed by the chapter session rather than by a job callback.
 */
export class ChapterLifecycleStore {
  private readonly chapters = new Map<string, MutableChapter>()

  start(pageSessionId: string, pageUrl: string): ChapterLifecycleState {
    const existing = this.chapters.get(pageSessionId)
    if (
      existing &&
      (existing.phase === 'started' || existing.phase === 'active') &&
      existing.pageUrl === pageUrl
    ) {
      return this.snapshot(existing)
    }
    const chapter: MutableChapter = {
      pageSessionId,
      pageUrl,
      phase: 'started',
      highestPageIndex: -1,
      submittedPages: new Set(),
      viewportRevision: 0,
    }
    this.chapters.set(pageSessionId, chapter)
    return this.snapshot(chapter)
  }

  page(pageSessionId: string, pageUrl: string, pageIndex: number): ChapterLifecycleState {
    const chapter = this.require(pageSessionId, pageUrl)
    if (chapter.phase === 'sealed' || chapter.phase === 'cancelled') return this.snapshot(chapter)
    chapter.phase = 'active'
    chapter.highestPageIndex = Math.max(chapter.highestPageIndex, pageIndex)
    chapter.submittedPages.add(pageIndex)
    return this.snapshot(chapter)
  }

  viewport(pageSessionId: string, pageUrl: string): ChapterLifecycleState {
    const chapter = this.require(pageSessionId, pageUrl)
    if (chapter.phase === 'started' || chapter.phase === 'active') {
      chapter.phase = 'active'
      chapter.viewportRevision += 1
    }
    return this.snapshot(chapter)
  }

  seal(pageSessionId: string, pageUrl: string): ChapterLifecycleState {
    const chapter = this.require(pageSessionId, pageUrl)
    if (chapter.phase !== 'cancelled') chapter.phase = 'sealed'
    return this.snapshot(chapter)
  }

  cancel(pageSessionId: string, pageUrl: string): ChapterLifecycleState {
    const chapter = this.require(pageSessionId, pageUrl)
    chapter.phase = 'cancelled'
    return this.snapshot(chapter)
  }

  remove(pageSessionId: string): void {
    this.chapters.delete(pageSessionId)
  }

  state(pageSessionId: string): ChapterLifecycleState | undefined {
    const chapter = this.chapters.get(pageSessionId)
    return chapter ? this.snapshot(chapter) : undefined
  }

  private require(pageSessionId: string, pageUrl: string): MutableChapter {
    const chapter = this.chapters.get(pageSessionId)
    if (!chapter || chapter.pageUrl !== pageUrl) {
      throw new Error('The chapter session is not active for this document.')
    }
    return chapter
  }

  private snapshot(chapter: MutableChapter): ChapterLifecycleState {
    return Object.freeze({
      pageSessionId: chapter.pageSessionId,
      pageUrl: chapter.pageUrl,
      phase: chapter.phase,
      highestPageIndex: chapter.highestPageIndex,
      submittedPages: chapter.submittedPages.size,
      viewportRevision: chapter.viewportRevision,
    })
  }
}
