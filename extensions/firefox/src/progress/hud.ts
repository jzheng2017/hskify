import type { ProgressJobUpdate } from '../contracts/browser'
import type { PageState } from '../messaging/messages'

const HUD_CSS = `
:host {
  position: fixed;
  right: 18px;
  top: 18px;
  z-index: 2147483647;
}
.panel {
  background: rgb(15 23 42 / 94%);
  border: 1px solid rgb(255 255 255 / 18%);
  border-radius: 13px;
  box-shadow: 0 10px 34px rgb(0 0 0 / 32%);
  color: #f8fafc;
  display: grid;
  font: 13px/1.35 system-ui, sans-serif;
  gap: 8px;
  min-width: 230px;
  padding: 12px 14px;
}
.title { font-size: 13px; font-weight: 700; }
.detail { color: #cbd5e1; }
progress {
  accent-color: #60a5fa;
  height: 6px;
  width: 100%;
}
button {
  appearance: none;
  background: transparent;
  border: 1px solid #64748b;
  border-radius: 7px;
  color: #f8fafc;
  cursor: pointer;
  font: 600 12px/1 system-ui, sans-serif;
  justify-self: end;
  padding: 7px 10px;
}
button:hover { background: rgb(255 255 255 / 9%); }
`

const BADGE_CSS = `
:host {
  pointer-events: none;
  position: absolute;
  z-index: 2147483646;
}
.badge {
  align-items: center;
  background: rgb(15 23 42 / 88%);
  border-radius: 8px;
  box-shadow: 0 3px 12px rgb(0 0 0 / 24%);
  color: #fff;
  display: flex;
  font: 600 12px/1.2 system-ui, sans-serif;
  gap: 8px;
  max-width: 260px;
  padding: 8px 10px;
}
button {
  appearance: none;
  background: #f8fafc;
  border: 0;
  border-radius: 6px;
  color: #111827;
  cursor: pointer;
  font: 700 11px/1 system-ui, sans-serif;
  padding: 6px 8px;
  pointer-events: auto;
}
`

export type HudProgress = {
  current: number
  total: number
  /** Stable identity for the image that produced the update. */
  key?: string
  status?: ProgressJobUpdate
  message?: string
}

export type ChapterProgressPhase =
  | 'starting'
  | 'reading'
  | 'preparing'
  | 'translating'
  | 'finishing'

export type ChapterProgressSnapshot = {
  phase: ChapterProgressPhase
  message: string
  active: number
  overallProgress?: number
}

const CHAPTER_PHASES: readonly ChapterProgressPhase[] = [
  'starting',
  'reading',
  'preparing',
  'translating',
  'finishing',
]

const CHAPTER_PHASE_MESSAGES: Record<ChapterProgressPhase, string> = {
  starting: 'Starting translation',
  reading: 'Reading the page',
  preparing: 'Preparing the artwork',
  translating: 'Writing the Chinese text',
  finishing: 'Finishing the chapter',
}

const CHAPTER_STAGE_PHASE: Record<ProgressJobUpdate['stage'], ChapterProgressPhase> = {
  queued: 'starting',
  decoding: 'reading',
  detecting: 'reading',
  ocr: 'reading',
  inpainting: 'preparing',
  translating: 'translating',
  'hsk-validating': 'translating',
  styling: 'finishing',
  packaging: 'finishing',
}

function phaseIndex(phase: ChapterProgressPhase): number {
  return CHAPTER_PHASES.indexOf(phase)
}

/**
 * Reduces concurrent image progress into one monotonic chapter phase.
 *
 * A chapter can have several images in different engine stages at once. A
 * last-write-wins status therefore oscillates between "reading" and
 * "writing". The reducer advances each image independently and advances the
 * chapter only when a later phase has actually been observed. Completed
 * images remain terminal and cannot re-introduce an earlier phase.
 */
export class ChapterProgressReducer {
  private readonly jobs = new Map<
    string,
    { phase: ChapterProgressPhase; overallProgress?: number }
  >()
  private currentPhase: ChapterProgressPhase = 'starting'
  private currentProgress = 0

  update(
    key: string,
    status: Pick<ProgressJobUpdate, 'stage' | 'overallProgress'>,
  ): ChapterProgressSnapshot {
    const next = CHAPTER_STAGE_PHASE[status.stage]
    const previous = this.jobs.get(key)
    const phase =
      !previous || phaseIndex(next) >= phaseIndex(previous.phase)
        ? next
        : previous.phase
    const overallProgress =
      status.overallProgress === undefined
        ? previous?.overallProgress
        : Math.max(0, Math.min(1, status.overallProgress))
    this.jobs.set(key, {
      phase,
      ...(overallProgress === undefined ? {} : { overallProgress }),
    })
    if (overallProgress !== undefined) {
      // A late update from a newly started image must not make the chapter
      // progress bar jump backwards.
      this.currentProgress = Math.max(this.currentProgress, overallProgress)
    }
    this.advance()
    return this.snapshot()
  }

  complete(key: string): ChapterProgressSnapshot {
    this.jobs.delete(key)
    this.advance()
    return this.snapshot()
  }

  reset(): void {
    this.jobs.clear()
    this.currentPhase = 'starting'
    this.currentProgress = 0
  }

  snapshot(): ChapterProgressSnapshot {
    return {
      phase: this.currentPhase,
      message: CHAPTER_PHASE_MESSAGES[this.currentPhase],
      active: this.jobs.size,
      ...(this.currentProgress > 0 ? { overallProgress: this.currentProgress } : {}),
    }
  }

  private advance(): void {
    const observed = [...this.jobs.values()].reduce(
      (highest, job) => Math.max(highest, phaseIndex(job.phase)),
      phaseIndex(this.currentPhase),
    )
    this.currentPhase = CHAPTER_PHASES[observed] ?? this.currentPhase
  }
}

export function friendlyProgressMessage(
  status: Pick<ProgressJobUpdate, 'stage'>,
): string {
  switch (status.stage) {
    case 'queued':
      return 'Waiting to start'
    case 'decoding':
    case 'detecting':
    case 'ocr':
      return 'Reading the page'
    case 'inpainting':
      return 'Preparing the artwork'
    case 'translating':
      return 'Writing the Chinese text'
    case 'hsk-validating':
      return 'Matching your difficulty level'
    case 'styling':
      return 'Fitting the text'
    case 'packaging':
      return 'Finishing this image'
  }
}

export class PageHud {
  readonly host: HTMLElement
  private readonly title: HTMLElement
  private readonly detail: HTMLElement
  private readonly progress: HTMLProgressElement
  private readonly cancelButton: HTMLButtonElement
  private readonly chapterProgress = new ChapterProgressReducer()
  private state: PageState = {
    state: 'idle',
    current: 0,
    total: 0,
    message: 'Ready',
  }

  constructor(
    onCancel: () => void,
    private readonly root: HTMLElement = document.documentElement,
  ) {
    this.host = document.createElement('aside')
    this.host.dataset.hmtOwned = 'true'
    this.host.setAttribute('aria-live', 'polite')
    const shadow = this.host.attachShadow({ mode: 'open' })
    const style = document.createElement('style')
    style.textContent = HUD_CSS
    const panel = document.createElement('section')
    panel.className = 'panel'
    this.title = document.createElement('span')
    this.title.className = 'title'
    this.title.textContent = 'Hskify'
    this.detail = document.createElement('span')
    this.detail.className = 'detail'
    this.detail.textContent = 'Preparing the chapter…'
    this.progress = document.createElement('progress')
    this.progress.max = 1
    this.progress.removeAttribute('value')
    this.cancelButton = document.createElement('button')
    this.cancelButton.type = 'button'
    this.cancelButton.textContent = 'Cancel'
    this.cancelButton.addEventListener('click', onCancel)
    panel.append(this.title, this.detail, this.progress, this.cancelButton)
    shadow.append(style, panel)
    root.append(this.host)
  }

  update(input: HudProgress): void {
    const status = input.status
    const state = 'running'
    const progress = status
      ? this.chapterProgress.update(input.key ?? 'chapter', status)
      : this.chapterProgress.snapshot()
    // The queue can work on several images concurrently. Show one chapter
    // phase instead of letting whichever image reported last replace it.
    const message = status ? progress.message : input.message ?? progress.message
    this.state = {
      state,
      current: input.current,
      total: input.total,
      stage: progress.phase,
      message,
    }
    this.title.textContent = 'Translating chapter'
    const completionProgress =
      input.total > 0 ? Math.max(0, Math.min(1, input.current / input.total)) : 0
    const measurable =
      progress.overallProgress !== undefined || input.current > 0
        ? Math.max(progress.overallProgress ?? 0, completionProgress)
        : undefined
    if (measurable === undefined) this.progress.removeAttribute('value')
    else this.progress.value = measurable
    this.detail.textContent = message
    this.cancelButton.hidden = state !== 'running'
  }

  complete(completed: number, total: number): void {
    this.state = {
      state: 'complete',
      current: completed,
      total,
      message: `${completed} of ${total} images ready`,
    }
    this.title.textContent = 'Translation complete'
    this.detail.textContent = this.state.message
    this.progress.value = 1
    this.cancelButton.hidden = true
  }

  completeImage(key: string): void {
    this.chapterProgress.complete(key)
  }

  resetProgress(): void {
    this.chapterProgress.reset()
  }

  fail(message: string, current: number, total: number): void {
    this.state = { state: 'failed', current, total, message }
    this.title.textContent = 'Translation needs attention'
    this.detail.textContent = message
    this.progress.removeAttribute('value')
    this.cancelButton.hidden = true
  }

  cancelled(current: number, total: number): void {
    this.state = {
      state: 'cancelled',
      current,
      total,
      message: 'Anything unfinished was left unchanged',
    }
    this.title.textContent = 'Translation cancelled'
    this.detail.textContent = this.state.message
    this.progress.removeAttribute('value')
    this.cancelButton.hidden = true
  }

  snapshot(): PageState {
    return { ...this.state }
  }

  destroy(): void {
    this.host.remove()
  }
}

export class ImageStatusBadge {
  private readonly host: HTMLElement
  private readonly message: HTMLElement
  private readonly retryButton: HTMLButtonElement
  private anchor: HTMLElement | undefined

  constructor(
    private readonly image: HTMLElement,
    onRetry: () => void,
    root?: HTMLElement,
    anchor?: HTMLElement,
  ) {
    const documentRef = image.ownerDocument
    this.root = root ?? documentRef.documentElement
    this.host = documentRef.createElement('span')
    this.host.dataset.hmtOwned = 'true'
    this.host.style.position = 'absolute'
    const shadow = this.host.attachShadow({ mode: 'open' })
    const style = documentRef.createElement('style')
    style.textContent = BADGE_CSS
    const badge = documentRef.createElement('span')
    badge.className = 'badge'
    this.message = documentRef.createElement('span')
    this.message.textContent = 'Queued'
    this.retryButton = documentRef.createElement('button')
    this.retryButton.type = 'button'
    this.retryButton.textContent = 'Retry'
    this.retryButton.hidden = true
    this.retryButton.addEventListener('click', onRetry)
    badge.append(this.message, this.retryButton)
    shadow.append(style, badge)
    this.attach(anchor)
  }

  private readonly root: HTMLElement

  update(status: ProgressJobUpdate | string): void {
    this.message.textContent =
      typeof status === 'string' ? status : friendlyProgressMessage(status)
    this.retryButton.hidden = true
    this.host.style.pointerEvents = 'none'
  }

  failure(message: string): void {
    this.message.textContent = message
    this.retryButton.hidden = false
    this.host.style.pointerEvents = 'auto'
  }

  /**
   * Attach the notice to the document-anchored renderer wrapper. Once it is
   * inside that wrapper, normal scrolling is handled entirely by the browser
   * compositor and no scroll-time coordinate work is necessary.
   */
  attach(anchor?: HTMLElement): void {
    this.anchor = anchor
    ;(anchor ?? this.root).append(this.host)
    if (anchor) {
      this.host.style.left = '8px'
      this.host.style.top = '8px'
      this.host.style.transform = 'none'
      return
    }
    // Submission can fail before a renderer wrapper exists. Keep the retry
    // notice visible at the image's document position without following every
    // scroll frame. A subsequent retry attaches it to the real wrapper.
    const view = this.image.ownerDocument.defaultView
    const rect = this.image.getBoundingClientRect()
    this.host.style.left = `${Math.max(4, rect.left + (view?.scrollX ?? 0) + 8)}px`
    this.host.style.top = `${Math.max(4, rect.top + (view?.scrollY ?? 0) + 8)}px`
    this.host.style.transform = 'none'
  }

  detach(): void {
    if (!this.host.isConnected) return
    if (this.anchor) this.attach(undefined)
  }

  destroy(): void {
    this.host.remove()
  }
}
