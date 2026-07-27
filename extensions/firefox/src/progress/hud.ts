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
  left: 0;
  pointer-events: none;
  position: fixed;
  top: 0;
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
  status?: ProgressJobUpdate
  message?: string
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
    this.detail.textContent = 'Preparing page…'
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
    // The HUD represents one chapter-wide operation. Individual images can be
    // in different pipeline stages concurrently, so exposing whichever image
    // reported last makes the chapter status oscillate. Keep the chapter
    // message stable; per-image badges still show local progress and retries.
    const message = input.message ?? 'Translating this chapter'
    this.state = {
      state,
      current: input.current,
      total: input.total,
      ...(status ? { stage: status.stage } : {}),
      message,
    }
    this.title.textContent = `Image ${Math.min(input.current + 1, input.total)} of ${input.total}`
    const measurable = status?.overallProgress
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

  constructor(
    private readonly image: HTMLImageElement,
    onRetry: () => void,
    private readonly root: HTMLElement = document.documentElement,
  ) {
    this.host = document.createElement('span')
    this.host.dataset.hmtOwned = 'true'
    const shadow = this.host.attachShadow({ mode: 'open' })
    const style = document.createElement('style')
    style.textContent = BADGE_CSS
    const badge = document.createElement('span')
    badge.className = 'badge'
    this.message = document.createElement('span')
    this.message.textContent = 'Queued'
    this.retryButton = document.createElement('button')
    this.retryButton.type = 'button'
    this.retryButton.textContent = 'Retry'
    this.retryButton.hidden = true
    this.retryButton.addEventListener('click', onRetry)
    badge.append(this.message, this.retryButton)
    shadow.append(style, badge)
    root.append(this.host)
    addEventListener('scroll', this.position, true)
    addEventListener('resize', this.position)
    this.position()
  }

  update(status: ProgressJobUpdate | string): void {
    this.message.textContent =
      typeof status === 'string' ? status : friendlyProgressMessage(status)
    this.retryButton.hidden = true
    this.position()
  }

  failure(message: string): void {
    this.message.textContent = message
    this.retryButton.hidden = false
    this.host.style.pointerEvents = 'auto'
    this.position()
  }

  destroy(): void {
    removeEventListener('scroll', this.position, true)
    removeEventListener('resize', this.position)
    this.host.remove()
  }

  private readonly position = (): void => {
    if (!this.image.isConnected) return
    const rect = this.image.getBoundingClientRect()
    this.host.style.transform = `translate(${Math.max(4, rect.left + 8)}px, ${Math.max(
      4,
      rect.top + 8,
    )}px)`
  }
}
