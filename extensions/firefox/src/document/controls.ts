export type ChapterDisplayMode = 'original' | 'chinese'

export interface ComparisonRenderTarget {
  setMode(mode: ChapterDisplayMode): void
  showOriginalForComparison(): void
  restoreSelectedMode(): void
}

const CONTROLS_CSS = `
:host { all: initial; }
.hskify-controls {
  align-items: center;
  background: rgba(17, 24, 39, .94);
  border: 1px solid rgba(255, 255, 255, .2);
  border-radius: 999px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .3);
  display: flex;
  font: 600 12px/1 system-ui, sans-serif;
  gap: 4px;
  padding: 5px;
  pointer-events: auto;
}
button {
  appearance: none;
  background: transparent;
  border: 0;
  border-radius: 999px;
  color: #e5e7eb;
  cursor: pointer;
  font: inherit;
  padding: 8px 11px;
}
button:hover { background: rgba(255, 255, 255, .1); }
button[aria-pressed="true"] { background: #f3f4f6; color: #111827; }
button:focus-visible { outline: 2px solid #60a5fa; outline-offset: 1px; }
`

class ComparisonControls {
  private mode: ChapterDisplayMode = 'chinese'
  private comparing = false
  private readonly targets = new Set<ComparisonRenderTarget>()
  private readonly host: HTMLElement
  private readonly originalButton: HTMLButtonElement
  private readonly chineseButton: HTMLButtonElement
  private readonly compareButton: HTMLButtonElement

  constructor(
    private readonly documentRef: Document,
    private readonly onEmpty: () => void,
  ) {
    this.host = documentRef.createElement('div')
    this.host.dataset.hskifyOwned = 'true'
    this.host.dataset.hskifyModeControls = 'true'
    this.host.setAttribute('aria-label', 'Hskify chapter translation controls')
    Object.assign(this.host.style, {
      bottom: '12px',
      pointerEvents: 'none',
      position: 'fixed',
      right: '12px',
      zIndex: '2147483646',
    })

    const shadow = this.host.attachShadow({ mode: 'open' })
    const style = documentRef.createElement('style')
    style.textContent = CONTROLS_CSS
    const controls = documentRef.createElement('div')
    controls.className = 'hskify-controls'
    controls.setAttribute('role', 'group')
    controls.setAttribute('aria-label', 'Translated chapter mode')
    this.originalButton = this.button('Original')
    this.chineseButton = this.button('Chinese')
    this.compareButton = this.button('Hold to compare')
    this.compareButton.title = 'Press and hold to show the original chapter'
    this.compareButton.setAttribute('aria-pressed', 'false')
    controls.append(this.originalButton, this.chineseButton, this.compareButton)
    shadow.append(style, controls)

    this.originalButton.addEventListener('click', this.showOriginal)
    this.chineseButton.addEventListener('click', this.showChinese)
    this.compareButton.addEventListener('click', this.suppressNavigation)
    this.compareButton.addEventListener('pointerdown', this.pressCompare)
    this.compareButton.addEventListener('pointerup', this.releaseCompare)
    this.compareButton.addEventListener('pointercancel', this.releaseCompare)
    this.compareButton.addEventListener('blur', this.releaseCompare)
    this.compareButton.addEventListener('keydown', this.compareKeyDown)
    this.compareButton.addEventListener('keyup', this.compareKeyUp)
    documentRef.defaultView?.addEventListener('pointerup', this.releaseCompare)
    documentRef.defaultView?.addEventListener('pointercancel', this.releaseCompare)
    documentRef.defaultView?.addEventListener('blur', this.releaseCompare)
    this.updatePressedState()
    ;(documentRef.body ?? documentRef.documentElement).append(this.host)
  }

  attach(target: ComparisonRenderTarget): void {
    this.targets.add(target)
    target.setMode(this.mode)
    if (this.comparing) target.showOriginalForComparison()
  }

  detach(target: ComparisonRenderTarget): void {
    this.targets.delete(target)
    if (this.targets.size === 0) this.destroy()
  }

  private button(label: string): HTMLButtonElement {
    const button = this.documentRef.createElement('button')
    button.type = 'button'
    button.textContent = label
    return button
  }

  private readonly showOriginal = (event: Event): void => {
    this.suppressNavigation(event)
    this.setMode('original')
  }

  private readonly showChinese = (event: Event): void => {
    this.suppressNavigation(event)
    this.setMode('chinese')
  }

  private readonly suppressNavigation = (event: Event): void => {
    event.preventDefault()
    event.stopPropagation()
  }

  private readonly pressCompare = (event: Event): void => {
    this.suppressNavigation(event)
    if (this.comparing) return
    this.comparing = true
    this.compareButton.setAttribute('aria-pressed', 'true')
    for (const target of this.targets) target.showOriginalForComparison()
  }

  private readonly releaseCompare = (): void => {
    if (!this.comparing) return
    this.comparing = false
    this.compareButton.setAttribute('aria-pressed', 'false')
    for (const target of this.targets) target.restoreSelectedMode()
  }

  private readonly compareKeyDown = (event: KeyboardEvent): void => {
    if (event.key === ' ' || event.key === 'Enter') this.pressCompare(event)
  }

  private readonly compareKeyUp = (event: KeyboardEvent): void => {
    if (event.key !== ' ' && event.key !== 'Enter') return
    this.suppressNavigation(event)
    this.releaseCompare()
  }

  private setMode(mode: ChapterDisplayMode): void {
    this.mode = mode
    this.updatePressedState()
    for (const target of this.targets) target.setMode(mode)
  }

  private updatePressedState(): void {
    this.originalButton.setAttribute('aria-pressed', String(this.mode === 'original'))
    this.chineseButton.setAttribute('aria-pressed', String(this.mode === 'chinese'))
  }

  private destroy(): void {
    this.originalButton.removeEventListener('click', this.showOriginal)
    this.chineseButton.removeEventListener('click', this.showChinese)
    this.compareButton.removeEventListener('click', this.suppressNavigation)
    this.compareButton.removeEventListener('pointerdown', this.pressCompare)
    this.compareButton.removeEventListener('pointerup', this.releaseCompare)
    this.compareButton.removeEventListener('pointercancel', this.releaseCompare)
    this.compareButton.removeEventListener('blur', this.releaseCompare)
    this.compareButton.removeEventListener('keydown', this.compareKeyDown)
    this.compareButton.removeEventListener('keyup', this.compareKeyUp)
    this.documentRef.defaultView?.removeEventListener('pointerup', this.releaseCompare)
    this.documentRef.defaultView?.removeEventListener('pointercancel', this.releaseCompare)
    this.documentRef.defaultView?.removeEventListener('blur', this.releaseCompare)
    this.host.remove()
    this.onEmpty()
  }
}

const controlsByDocument = new WeakMap<Document, ComparisonControls>()

/** Attaches a target to the single comparison control instance in a document. */
export function attachComparisonControls(
  documentRef: Document,
  target: ComparisonRenderTarget,
): () => void {
  let controls = controlsByDocument.get(documentRef)
  if (!controls) {
    controls = new ComparisonControls(documentRef, () => controlsByDocument.delete(documentRef))
    controlsByDocument.set(documentRef, controls)
  }
  controls.attach(target)
  let attached = true
  return () => {
    if (!attached) return
    attached = false
    controls?.detach(target)
  }
}
