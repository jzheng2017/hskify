import type {
  BrowserRegion,
  LookupRequest,
  LookupResult,
  RegionRefinedJobUpdate,
} from '../contracts/browser'
import type { DiscoveredImage } from '../discovery/images'
import { SelectionController } from '../selection/popover'
import { MandarinSpeaker, type TextSpeaker } from '../selection/speech'
import { FontLoader, type FontFetcher } from './font-loader'
import {
  calculateImageGeometry,
  polygonBounds,
  type ImageGeometry,
} from './geometry'
import {
  fitPolygonForRegion,
  PolygonTextFitter,
} from './fitting'
import { applyRegionStyle } from './style'

const MEASUREMENT_SEARCH_STEPS = 16
const FIT_SAFETY_RATIO = 0.98
type TranslationMode = 'chinese' | 'original'

const RENDERER_CSS = `
:host {
  inset: 0;
  pointer-events: none;
  position: absolute;
  z-index: 1;
}
*, *::before, *::after { box-sizing: border-box; }
.hmt-viewport {
  overflow: hidden;
  pointer-events: none;
  position: absolute;
}
.hmt-image-space {
  position: absolute;
}
.hmt-patch-layer,
.hmt-text-layer {
  height: 100%;
  inset: 0;
  pointer-events: none;
  position: absolute;
  width: 100%;
}
.hmt-patch {
  object-fit: fill;
  pointer-events: none;
  position: absolute;
  user-select: none;
}
.hmt-region {
  align-items: center;
  cursor: text;
  display: flex;
  justify-content: center;
  overflow: hidden;
  pointer-events: auto;
  position: absolute;
  text-rendering: geometricPrecision;
  transform-origin: center;
  unicode-bidi: plaintext;
  user-select: text;
  white-space: pre;
  word-break: normal;
  overflow-wrap: break-word;
}
.hmt-region-text { display: block; }
.hmt-region-line { display: block; }
.hmt-region:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}
.hmt-lookup {
  background: #fff;
  border: 1px solid #d1d5db;
  border-radius: 9px;
  box-shadow: 0 8px 28px rgb(0 0 0 / 24%);
  color: #111827;
  display: grid;
  font: 13px/1.4 system-ui, sans-serif;
  gap: 7px;
  max-height: calc(100vh - 16px);
  max-width: min(320px, calc(100% - 8px));
  min-width: 190px;
  overflow: auto;
  padding: 10px 12px;
  pointer-events: auto;
  position: absolute;
  text-align: left;
  user-select: text;
  z-index: 6;
}
.hmt-lookup[hidden] { display: none; }
.hmt-lookup-heading {
  align-items: center;
  display: flex;
  gap: 10px;
  justify-content: space-between;
}
.hmt-speak {
  appearance: none;
  background: #eff6ff;
  border: 1px solid #93c5fd;
  border-radius: 999px;
  color: #1d4ed8;
  cursor: pointer;
  flex: none;
  font: 600 11px/1 system-ui, sans-serif;
  padding: 6px 9px;
}
.hmt-speak[aria-pressed="true"] {
  background: #1d4ed8;
  color: #fff;
}
.hmt-speak:focus-visible {
  outline: 2px solid #2563eb;
  outline-offset: 2px;
}
.hmt-speak:disabled {
  background: #f3f4f6;
  border-color: #d1d5db;
  color: #6b7280;
  cursor: not-allowed;
}
.hmt-lookup-entry,
.hmt-lookup-context {
  border-top: 1px solid #e5e7eb;
  display: grid;
  gap: 2px;
  padding-top: 6px;
}
.hmt-lookup-entry span:last-child,
.hmt-lookup-context span:last-child { color: #4b5563; }
`

const MODE_CONTROLS_CSS = `
:host {
  bottom: max(12px, env(safe-area-inset-bottom));
  display: block;
  pointer-events: none;
  position: fixed;
  right: max(12px, env(safe-area-inset-right));
  z-index: 2147483646;
}
*, *::before, *::after { box-sizing: border-box; }
.hmt-controls {
  align-items: center;
  background: rgb(17 24 39 / 92%);
  border: 1px solid rgb(255 255 255 / 22%);
  border-radius: 999px;
  box-shadow: 0 3px 14px rgb(0 0 0 / 28%);
  display: flex;
  gap: 2px;
  max-width: calc(100vw - 24px);
  padding: 3px;
  pointer-events: auto;
}
.hmt-controls button {
  appearance: none;
  background: transparent;
  border: 0;
  border-radius: 999px;
  color: #e5e7eb;
  cursor: pointer;
  font: 600 11px/1 system-ui, sans-serif;
  padding: 7px 9px;
  touch-action: none;
  white-space: nowrap;
}
.hmt-controls button[aria-pressed="true"] {
  background: #f8fafc;
  color: #111827;
}
.hmt-controls button:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 1px;
}
`

export type RenderJob = {
  jobId: string
  sourceWidth: number
  sourceHeight: number
}

export type RenderGuard = {
  signal?: AbortSignal
  validate(): void
}

export type PatchImageDecoder = (image: HTMLImageElement) => Promise<void>

export type RendererCallbacks = {
  fetchFont: FontFetcher
  lookup(request: LookupRequest): Promise<LookupResult>
  onRestore?: () => void
  onFitDegraded?: (regionId: string) => void
}

export class RendererError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message)
    this.name = 'RendererError'
  }
}

type RegionView = {
  region: BrowserRegion
  patch: HTMLImageElement
  patchUrl: string
  element: HTMLElement
  textElement: HTMLElement
  fontFamily: string
}

function px(value: number): string {
  return `${Number.isFinite(value) ? value : 0}px`
}

function setRect(
  element: HTMLElement,
  rect: { left: number; top: number; width: number; height: number },
): void {
  element.style.left = px(rect.left)
  element.style.top = px(rect.top)
  element.style.width = px(rect.width)
  element.style.height = px(rect.height)
}

function setPercentRegion(element: HTMLElement, region: BrowserRegion): void {
  const points = fitPolygonForRegion(region)
  const bounds = polygonBounds(points)
  element.style.left = `${bounds.minX * 100}%`
  element.style.top = `${bounds.minY * 100}%`
  element.style.width = `${bounds.width * 100}%`
  element.style.height = `${bounds.height * 100}%`
}

function setPercentPatch(element: HTMLElement, region: BrowserRegion): void {
  const rect = region.patch.rect
  element.style.left = `${rect.x * 100}%`
  element.style.top = `${rect.y * 100}%`
  element.style.width = `${rect.width * 100}%`
  element.style.height = `${rect.height * 100}%`
}

function createButton(
  label: string,
  documentRef: Document = document,
): HTMLButtonElement {
  const button = documentRef.createElement('button')
  button.type = 'button'
  button.textContent = label
  return button
}

class ModeControls {
  private mode: TranslationMode = 'chinese'
  private comparing = false
  private readonly targets = new Set<RenderedImage>()
  private readonly host: HTMLElement
  private readonly originalButton: HTMLButtonElement
  private readonly chineseButton: HTMLButtonElement
  private readonly compareButton: HTMLButtonElement

  constructor(
    private readonly documentRef: Document,
    private readonly onEmpty: (controls: ModeControls) => void,
  ) {
    this.host = documentRef.createElement('span')
    this.host.dataset.hmtOwned = 'true'
    this.host.dataset.hmtModeControls = 'true'
    this.host.setAttribute('aria-label', 'HSK manga translation mode controls')
    this.host.style.bottom = '12px'
    this.host.style.pointerEvents = 'none'
    this.host.style.position = 'fixed'
    this.host.style.right = '12px'
    this.host.style.zIndex = '2147483646'

    const shadow = this.host.attachShadow({ mode: 'open' })
    const style = documentRef.createElement('style')
    style.textContent = MODE_CONTROLS_CSS
    const controls = documentRef.createElement('span')
    controls.className = 'hmt-controls'
    controls.setAttribute('role', 'group')
    controls.setAttribute('aria-label', 'Translated image mode')
    this.originalButton = createButton('Original', documentRef)
    this.chineseButton = createButton('Chinese', documentRef)
    this.compareButton = createButton('Hold to compare', documentRef)
    this.compareButton.title = 'Press and hold to show the original'
    this.compareButton.setAttribute('aria-pressed', 'false')
    controls.append(this.originalButton, this.chineseButton, this.compareButton)
    shadow.append(style, controls)

    this.originalButton.addEventListener('click', this.showOriginal)
    this.chineseButton.addEventListener('click', this.showChinese)
    this.compareButton.addEventListener('click', this.suppressControlNavigation)
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
    const mount = documentRef.body ?? documentRef.documentElement
    mount.append(this.host)
  }

  attach(target: RenderedImage): void {
    this.targets.add(target)
    target.setMode(this.mode)
    if (this.comparing) target.showOriginalForComparison()
  }

  detach(target: RenderedImage): void {
    this.targets.delete(target)
    if (this.targets.size === 0) this.destroy()
  }

  private readonly showOriginal = (event: Event): void => {
    this.suppressControlNavigation(event)
    this.setMode('original')
  }

  private readonly showChinese = (event: Event): void => {
    this.suppressControlNavigation(event)
    this.setMode('chinese')
  }

  private readonly suppressControlNavigation = (event: Event): void => {
    event.preventDefault()
    event.stopPropagation()
  }

  private readonly pressCompare = (event: Event): void => {
    this.suppressControlNavigation(event)
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
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault()
      event.stopPropagation()
      this.releaseCompare()
    }
  }

  private setMode(mode: TranslationMode): void {
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
    this.compareButton.removeEventListener('click', this.suppressControlNavigation)
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
    this.onEmpty(this)
  }
}

async function decodePatchImage(image: HTMLImageElement): Promise<void> {
  if (typeof image.decode === 'function') {
    await image.decode()
    return
  }
  if (image.complete) {
    if (image.naturalWidth > 0 && image.naturalHeight > 0) return
    throw new RendererError('PATCH_DECODE_FAILED', 'The translated image patch could not be decoded.')
  }
  await new Promise<void>((resolve, reject) => {
    image.addEventListener('load', () => resolve(), { once: true })
    image.addEventListener(
      'error',
      () =>
        reject(
          new RendererError(
            'PATCH_DECODE_FAILED',
            'The translated image patch could not be decoded.',
          ),
        ),
      { once: true },
    )
  })
}

function applyChosenLines(element: HTMLElement, lines: readonly string[], text: string): void {
  const chosen = lines.length > 0 && lines.join('') === text ? lines : [text]
  const nodes: Node[] = []
  for (const line of chosen) {
    const lineElement = document.createElement('span')
    lineElement.className = 'hmt-region-line'
    lineElement.textContent = line
    nodes.push(lineElement)
  }
  element.replaceChildren(...nodes)
}

function regionElement(region: BrowserRegion): {
  element: HTMLElement
  textElement: HTMLElement
} {
  const element = document.createElement('span')
  element.className = 'hmt-region'
  element.lang = 'zh-CN'
  element.tabIndex = 0
  element.dataset.regionId = region.id
  element.dataset.pinyin = region.pinyin
  element.dataset.hskValid = String(region.hsk.strictlyValid)
  element.dataset.hskRepairState = region.hsk.repairState
  element.setAttribute(
    'aria-label',
    region.pinyin ? `${region.displayedChinese}; ${region.pinyin}` : region.displayedChinese,
  )
  const textElement = document.createElement('span')
  textElement.className = 'hmt-region-text'
  // Companion text always enters the page through text nodes.
  textElement.textContent = region.displayedChinese
  element.append(textElement)
  setPercentRegion(element, region)
  return { element, textElement }
}

function actualOverflow(element: HTMLElement): boolean {
  return (
    element.scrollWidth > element.clientWidth + 0.5 ||
    element.scrollHeight > element.clientHeight + 0.5
  )
}

export class RenderedImage {
  private mode: TranslationMode = 'chinese'
  private readonly regions = new Map<string, RegionView>()
  private readonly fitter = new PolygonTextFitter()
  private readonly selection: SelectionController
  private destroyed = false
  private geometry?: ImageGeometry

  constructor(
    readonly candidate: DiscoveredImage,
    readonly job: RenderJob,
    readonly wrapper: HTMLElement,
    private readonly viewport: HTMLElement,
    private readonly imageSpace: HTMLElement,
    private readonly patchLayer: HTMLElement,
    private readonly textLayer: HTMLElement,
    private readonly callbacks: RendererCallbacks,
    private readonly fontLoader: FontLoader,
    shadowRoot: ShadowRoot,
    popover: HTMLElement,
    speaker: TextSpeaker,
    private readonly patchImageDecoder: PatchImageDecoder,
    private readonly resizeObserver?: ResizeObserver,
    private readonly onDestroy?: (rendered: RenderedImage) => void,
  ) {
    this.selection = new SelectionController(
      shadowRoot,
      popover,
      callbacks.lookup,
      this.forwardPrimaryClick,
      speaker,
    )
    this.resizeObserver?.observe(candidate.element)
    candidate.element.ownerDocument.addEventListener('scroll', this.scheduleRefit, true)
    candidate.element.ownerDocument.defaultView?.addEventListener('resize', this.scheduleRefit)
    this.refit()
  }

  get currentMode(): TranslationMode {
    return this.mode
  }

  get regionCount(): number {
    return this.regions.size
  }

  regionsInReadingOrder(): BrowserRegion[] {
    return [...this.regions.values()]
      .map((view) => view.region)
      .sort((left, right) => left.readingOrder - right.readingOrder)
  }

  private readonly forwardPrimaryClick = (event: MouseEvent): void => {
    if (event.button !== 0 || event.defaultPrevented) return
    const forwarded = new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      composed: false,
      view: window,
      detail: event.detail,
      screenX: event.screenX,
      screenY: event.screenY,
      clientX: event.clientX,
      clientY: event.clientY,
      ctrlKey: event.ctrlKey,
      shiftKey: event.shiftKey,
      altKey: event.altKey,
      metaKey: event.metaKey,
      button: 0,
      buttons: event.buttons,
    })
    if (!this.candidate.element.dispatchEvent(forwarded)) event.preventDefault()
  }

  setMode(mode: TranslationMode): void {
    if (this.destroyed) return
    this.mode = mode
    if (mode === 'original') this.selection.dismiss()
    this.applyVisualMode(mode)
  }

  showOriginalForComparison(): void {
    if (!this.destroyed) this.applyVisualMode('original')
  }

  restoreSelectedMode(): void {
    if (!this.destroyed) this.applyVisualMode(this.mode)
  }

  private applyVisualMode(mode: TranslationMode): void {
    // The page's original image remains connected and visible at all times.
    // Comparison only toggles the transparent patch/text overlay.
    this.viewport.hidden = mode === 'original'
  }

  private updateRegionMetadata(view: RegionView): void {
    view.element.dataset.pinyin = view.region.pinyin
    view.element.dataset.hskValid = String(view.region.hsk.strictlyValid)
    view.element.dataset.hskRepairState = view.region.hsk.repairState
    view.element.setAttribute(
      'aria-label',
      view.region.pinyin
        ? `${view.region.displayedChinese}; ${view.region.pinyin}`
        : view.region.displayedChinese,
    )
  }

  async installRegion(
    region: BrowserRegion,
    patchBytes: ArrayBuffer,
    guard: RenderGuard = { validate: () => undefined },
  ): Promise<void> {
    guard.validate()
    if (this.destroyed) {
      throw new RendererError('RENDERER_DESTROYED', 'The translated image is no longer active.')
    }
    const patchUrl = URL.createObjectURL(
      new Blob([patchBytes], { type: region.patch.mimeType }),
    )
    const patch = document.createElement('img')
    patch.className = 'hmt-patch'
    patch.alt = ''
    patch.draggable = false
    patch.dataset.patchId = region.patch.blobId
    patch.src = patchUrl
    patch.style.zIndex = String(Math.max(0, region.readingOrder))
    setPercentPatch(patch, region)
    let fontFamily: string
    try {
      const [, loadedFontFamily] = await Promise.all([
        this.patchImageDecoder(patch),
        this.fontLoader.load(region.style.fontId, region.style.category, this.job.jobId),
      ])
      fontFamily = loadedFontFamily
      guard.validate()
      if (document.fonts?.ready) {
        await document.fonts.ready
        guard.validate()
      }
    } catch (error) {
      URL.revokeObjectURL(patchUrl)
      if (error instanceof Error && error.name === 'AbortError') throw error
      throw error instanceof RendererError
        ? error
        : new RendererError(
            'PATCH_DECODE_FAILED',
            'The translated image patch could not be decoded.',
          )
    }

    const expectedWidth = Math.max(1, Math.round(region.patch.rect.width * this.job.sourceWidth))
    const expectedHeight = Math.max(1, Math.round(region.patch.rect.height * this.job.sourceHeight))
    if (
      patch.naturalWidth > 0 &&
      patch.naturalHeight > 0 &&
      (Math.abs(patch.naturalWidth - expectedWidth) > 1 ||
        Math.abs(patch.naturalHeight - expectedHeight) > 1)
    ) {
      URL.revokeObjectURL(patchUrl)
      throw new RendererError(
        'PATCH_DIMENSIONS_MISMATCH',
        'The translated patch dimensions do not match its source rectangle.',
      )
    }

    const created = regionElement(region)
    created.element.style.zIndex = String(Math.max(0, region.readingOrder))
    const next: RegionView = {
      region,
      patch,
      patchUrl,
      element: created.element,
      textElement: created.textElement,
      fontFamily,
    }
    const previous = this.regions.get(region.id)

    // The decoded patch is inserted synchronously before its selectable text.
    // No page state can expose Chinese over an undecoded/absent inpaint.
    if (previous) {
      previous.patch.replaceWith(patch)
      previous.element.replaceWith(created.element)
      this.selection.unregister(previous.element)
    } else {
      this.patchLayer.append(patch)
      this.textLayer.append(created.element)
    }
    this.regions.set(region.id, next)
    this.selection.register(created.element, this.job.jobId, region.id)
    this.updateRegionMetadata(next)
    this.refitView(next)
    if (previous) URL.revokeObjectURL(previous.patchUrl)
  }

  refineRegion(update: RegionRefinedJobUpdate): void {
    const view = this.regions.get(update.regionId)
    if (this.destroyed) return
    if (!view) {
      throw new RendererError(
        'REGION_REFINEMENT_BEFORE_READY',
        'A region refinement arrived before its decoded patch was installed.',
      )
    }
    view.region = {
      ...view.region,
      displayedChinese: update.displayedChinese,
      pinyin: update.pinyin,
      hsk: update.hsk,
    }
    view.textElement.textContent = update.displayedChinese
    this.updateRegionMetadata(view)
    this.refitView(view)
  }

  private refitView(view: RegionView): void {
    if (!this.geometry) return
    const fit = this.fitter.fit(
      view.region,
      this.geometry.image.width,
      this.geometry.image.height,
    )
    applyChosenLines(view.textElement, fit.lines, view.region.displayedChinese)
    applyRegionStyle(view.element, view.region, fit.fontSize, view.fontFamily)
    view.element.style.alignItems =
      view.region.style.writingMode === 'vertical-rl' ? 'flex-start' : 'center'

    // Keep a small measured-layout margin so fractional CSS zoom and device
    // pixel rounding cannot turn an exact fit into a clipped final glyph.
    let measuredSize = fit.fontSize * FIT_SAFETY_RATIO
    applyRegionStyle(view.element, view.region, measuredSize, view.fontFamily)
    if (actualOverflow(view.element)) {
      let low = 0
      let high = measuredSize
      for (let iteration = 0; iteration < MEASUREMENT_SEARCH_STEPS; iteration += 1) {
        const midpoint = (low + high) / 2
        applyRegionStyle(view.element, view.region, midpoint, view.fontFamily)
        if (actualOverflow(view.element)) high = midpoint
        else low = midpoint
      }
      measuredSize = low * 0.995
      applyRegionStyle(view.element, view.region, measuredSize, view.fontFamily)
    }
    // A zero-size final fallback is only reachable for degenerate page
    // geometry. It preserves selectable text while strictly preventing
    // overflow; normal regions remain above zero through the binary search.
    if (actualOverflow(view.element)) {
      measuredSize = 0
      applyRegionStyle(view.element, view.region, measuredSize, view.fontFamily)
    }
    if (fit.degraded || measuredSize === 0) {
      view.element.dataset.fit = 'degraded'
      this.callbacks.onFitDegraded?.(view.region.id)
    } else {
      delete view.element.dataset.fit
    }
  }

  private refitFrame: number | null = null
  private readonly scheduleRefit = (): void => {
    if (this.destroyed || this.refitFrame !== null) return
    const view = this.candidate.element.ownerDocument.defaultView
    if (!view) {
      this.refit()
      return
    }
    this.refitFrame = view.requestAnimationFrame(() => {
      this.refitFrame = null
      this.refit()
    })
  }

  refit(): void {
    if (this.destroyed || !this.candidate.element.isConnected) return
    const imageRect = this.candidate.element.getBoundingClientRect()
    this.wrapper.style.left = px(imageRect.left)
    this.wrapper.style.top = px(imageRect.top)
    this.wrapper.style.width = px(imageRect.width)
    this.wrapper.style.height = px(imageRect.height)
    this.geometry = calculateImageGeometry(
      this.candidate.element,
      this.wrapper,
      this.job.sourceWidth,
      this.job.sourceHeight,
    )
    setRect(this.viewport, this.geometry.viewport)
    setRect(this.imageSpace, this.geometry.image)
    for (const view of this.regions.values()) this.refitView(view)
  }

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.resizeObserver?.disconnect()
    const documentRef = this.candidate.element.ownerDocument
    documentRef.removeEventListener('scroll', this.scheduleRefit, true)
    documentRef.defaultView?.removeEventListener('resize', this.scheduleRefit)
    if (this.refitFrame !== null) {
      documentRef.defaultView?.cancelAnimationFrame(this.refitFrame)
      this.refitFrame = null
    }
    this.selection.destroy()
    this.candidate.element.removeAttribute('data-hmt-original')
    this.wrapper.remove()
    for (const view of this.regions.values()) URL.revokeObjectURL(view.patchUrl)
    this.regions.clear()
    this.onDestroy?.(this)
    this.callbacks.onRestore?.()
  }
}

export class SelectableRenderer {
  private readonly fontLoader: FontLoader
  private readonly modeControls = new Map<Document, ModeControls>()

  constructor(
    private readonly callbacks: RendererCallbacks,
    private readonly ResizeObserverType:
      | typeof ResizeObserver
      | undefined = globalThis.ResizeObserver,
    private readonly patchImageDecoder: PatchImageDecoder = decodePatchImage,
    private readonly speaker: TextSpeaker = new MandarinSpeaker(),
  ) {
    this.fontLoader = new FontLoader(callbacks.fetchFont)
  }

  private controlsFor(documentRef: Document): ModeControls {
    const existing = this.modeControls.get(documentRef)
    if (existing) return existing
    const controls = new ModeControls(documentRef, (emptyControls) => {
      if (this.modeControls.get(documentRef) === emptyControls) {
        this.modeControls.delete(documentRef)
      }
    })
    this.modeControls.set(documentRef, controls)
    return controls
  }

  private readonly releaseControls = (rendered: RenderedImage): void => {
    this.modeControls.get(rendered.candidate.element.ownerDocument)?.detach(rendered)
  }

  begin(
    candidate: DiscoveredImage,
    job: RenderJob,
    guard: RenderGuard = { validate: () => undefined },
  ): RenderedImage {
    guard.validate()
    if (!candidate.owner.isConnected || !candidate.element.isConnected) {
      throw new RendererError(
        'IMAGE_REPLACED_DURING_PROCESSING',
        'The page replaced this image before translation started.',
      )
    }
    if (job.sourceWidth < 1 || job.sourceHeight < 1) {
      throw new RendererError('INVALID_RESULT_GEOMETRY', 'The source dimensions are invalid.')
    }
    if (
      job.sourceWidth !== candidate.element.naturalWidth ||
      job.sourceHeight !== candidate.element.naturalHeight
    ) {
      throw new RendererError(
        'RESULT_SOURCE_DIMENSIONS_MISMATCH',
        'The translation job dimensions do not match the live page image.',
      )
    }
    const transform = getComputedStyle(candidate.element).transform
    if (
      transform !== '' &&
      transform !== 'none' &&
      transform !== 'matrix(1, 0, 0, 1, 0, 0)'
    ) {
      throw new RendererError(
        'UNSUPPORTED_IMAGE_TRANSFORM',
        'Rotated or transformed page images cannot be aligned safely.',
      )
    }

    const originalParent = candidate.owner.parentNode
    if (!originalParent) {
      throw new RendererError(
        'IMAGE_REPLACED_DURING_PROCESSING',
        'The page removed this image before translation started.',
      )
    }
    const wrapper = document.createElement('span')
    wrapper.dataset.hmtOwned = 'true'
    if (candidate.element.dataset.page) {
      wrapper.dataset.hmtSourcePage = candidate.element.dataset.page
    }
    wrapper.className = 'hmt-wrapper'
    wrapper.style.contain = 'layout style'
    wrapper.style.display = 'block'
    wrapper.style.pointerEvents = 'none'
    wrapper.style.position = 'fixed'
    wrapper.style.zIndex = '2147483000'
    guard.validate()
    candidate.element.ownerDocument.body.append(wrapper)

    candidate.element.setAttribute('data-hmt-original', 'true')
    const host = document.createElement('span')
    host.dataset.hmtOwned = 'true'
    host.setAttribute('aria-label', 'HSK manga translation controls')
    wrapper.append(host)
    const shadow = host.attachShadow({ mode: 'open' })
    const style = document.createElement('style')
    style.textContent = RENDERER_CSS
    const viewport = document.createElement('span')
    viewport.className = 'hmt-viewport'
    const imageSpace = document.createElement('span')
    imageSpace.className = 'hmt-image-space'
    const patchLayer = document.createElement('span')
    patchLayer.className = 'hmt-patch-layer'
    const textLayer = document.createElement('span')
    textLayer.className = 'hmt-text-layer'
    imageSpace.append(patchLayer, textLayer)
    viewport.append(imageSpace)

    const popover = document.createElement('span')
    popover.className = 'hmt-lookup'
    popover.hidden = true
    popover.setAttribute('role', 'dialog')
    popover.setAttribute('aria-label', 'Chinese dictionary')
    popover.addEventListener('click', (event) => {
      event.preventDefault()
      event.stopPropagation()
    })
    shadow.append(style, viewport, popover)

    let rendered: RenderedImage | undefined
    const resizeObserver = this.ResizeObserverType
      ? new this.ResizeObserverType(() => rendered?.refit())
      : undefined
    try {
      guard.validate()
      rendered = new RenderedImage(
        candidate,
        job,
        wrapper,
        viewport,
        imageSpace,
        patchLayer,
        textLayer,
        this.callbacks,
        this.fontLoader,
        shadow,
        popover,
        this.speaker,
        this.patchImageDecoder,
        resizeObserver,
        this.releaseControls,
      )
      this.controlsFor(candidate.element.ownerDocument).attach(rendered)
    } catch (error) {
      resizeObserver?.disconnect()
      candidate.element.removeAttribute('data-hmt-original')
      wrapper.remove()
      throw error
    }
    return rendered
  }
}
