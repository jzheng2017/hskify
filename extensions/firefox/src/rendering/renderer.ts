import type { BrowserJobResult, BrowserRegion, LookupRequest, LookupResult } from '../contracts/browser'
import type { DiscoveredImage } from '../discovery/images'
import { SelectionController } from '../selection/popover'
import { FontLoader, type FontFetcher } from './font-loader'
import {
  calculateImageGeometry,
  polygonBounds,
  rectDifference,
  type ImageGeometry,
} from './geometry'
import {
  fitPolygonForRegion,
  minimumFontSizeForImage,
  PolygonTextFitter,
} from './fitting'
import { applyRegionStyle } from './style'

const MAX_LAYOUT_SHIFT_PX = 2

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
.hmt-clean-image,
.hmt-text-layer {
  height: 100%;
  inset: 0;
  position: absolute;
  width: 100%;
}
.hmt-clean-image {
  object-fit: fill;
  pointer-events: none;
  user-select: none;
}
.hmt-text-layer { pointer-events: none; }
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
.hmt-controls {
  align-items: center;
  background: rgb(17 24 39 / 88%);
  border: 1px solid rgb(255 255 255 / 22%);
  border-radius: 999px;
  box-shadow: 0 3px 14px rgb(0 0 0 / 28%);
  display: flex;
  gap: 2px;
  padding: 3px;
  pointer-events: auto;
  position: absolute;
  right: 8px;
  top: 8px;
  z-index: 4;
}
.hmt-controls button {
  appearance: none;
  background: transparent;
  border: 0;
  border-radius: 999px;
  color: #e5e7eb;
  cursor: pointer;
  font: 600 11px/1 system-ui, sans-serif;
  padding: 6px 8px;
}
.hmt-controls button[aria-pressed="true"] {
  background: #f8fafc;
  color: #111827;
}
.hmt-controls button:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 1px;
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
  max-width: min(320px, calc(100% - 8px));
  min-width: 190px;
  padding: 10px 12px;
  pointer-events: auto;
  position: absolute;
  text-align: left;
  user-select: text;
  z-index: 6;
}
.hmt-lookup[hidden] { display: none; }
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

export type RenderPayload = {
  result: BrowserJobResult
  cleanImage: ArrayBuffer
}

export type RenderGuard = {
  signal?: AbortSignal
  validate(): void
}

export type CleanImageDecoder = (image: HTMLImageElement) => Promise<void>

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
  element: HTMLElement
  textElement: HTMLElement
  fontFamily: string
}

function px(value: number): string {
  return `${Number.isFinite(value) ? value : 0}px`
}

function setRect(element: HTMLElement, rect: { left: number; top: number; width: number; height: number }): void {
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

function createButton(label: string): HTMLButtonElement {
  const button = document.createElement('button')
  button.type = 'button'
  button.textContent = label
  return button
}

async function decodeCleanImage(image: HTMLImageElement): Promise<void> {
  if (typeof image.decode === 'function') {
    await image.decode()
    return
  }
  if (image.complete) {
    if (image.naturalWidth > 0 && image.naturalHeight > 0) return
    throw new RendererError('CLEAN_IMAGE_DECODE_FAILED', 'The cleaned image could not be decoded.')
  }
  await new Promise<void>((resolve, reject) => {
    image.addEventListener('load', () => resolve(), { once: true })
    image.addEventListener(
      'error',
      () =>
        reject(
          new RendererError(
            'CLEAN_IMAGE_DECODE_FAILED',
            'The cleaned image could not be decoded.',
          ),
        ),
      { once: true },
    )
  })
}

function applyChosenLines(element: HTMLElement, lines: readonly string[], text: string): void {
  const chosen = lines.length > 0 && lines.join('') === text ? lines : [text]
  const nodes: Node[] = []
  chosen.forEach((line) => {
    const lineElement = document.createElement('span')
    lineElement.className = 'hmt-region-line'
    lineElement.textContent = line
    nodes.push(lineElement)
  })
  element.replaceChildren(...nodes)
}

function originalOwnerStyle(owner: HTMLElement): {
  opacity: string
  priority: string
} {
  return {
    opacity: owner.style.getPropertyValue('opacity'),
    priority: owner.style.getPropertyPriority('opacity'),
  }
}

function restoreOpacity(
  owner: HTMLElement,
  saved: { opacity: string; priority: string },
): void {
  if (saved.opacity) owner.style.setProperty('opacity', saved.opacity, saved.priority)
  else owner.style.removeProperty('opacity')
}

export class RenderedImage {
  private mode: 'chinese' | 'original' = 'chinese'
  private readonly regions: RegionView[] = []
  private readonly fitter = new PolygonTextFitter()
  private readonly selection: SelectionController
  private destroyed = false
  private geometry?: ImageGeometry

  constructor(
    readonly candidate: DiscoveredImage,
    readonly payload: RenderPayload,
    readonly wrapper: HTMLElement,
    private readonly viewport: HTMLElement,
    private readonly imageSpace: HTMLElement,
    private readonly originalButton: HTMLButtonElement,
    private readonly chineseButton: HTMLButtonElement,
    private readonly compareButton: HTMLButtonElement,
    private readonly cleanUrl: string,
    private readonly originalParent: Node,
    private readonly originalNextSibling: Node | null,
    private readonly savedOwnerOpacity: { opacity: string; priority: string },
    private readonly savedImageOpacity: { opacity: string; priority: string },
    private readonly callbacks: RendererCallbacks,
    shadowRoot: ShadowRoot,
    popover: HTMLElement,
    fontFamilies: Map<string, string>,
    private readonly resizeObserver?: ResizeObserver,
  ) {
    for (const region of payload.result.regions) {
      if (!region.displayedChinese) continue
      const element = document.createElement('span')
      element.className = 'hmt-region'
      element.lang = 'zh-CN'
      element.tabIndex = 0
      element.dataset.regionId = region.id
      const textElement = document.createElement('span')
      textElement.className = 'hmt-region-text'
      // Model output always enters the DOM through text nodes.
      textElement.textContent = region.displayedChinese
      element.append(textElement)
      setPercentRegion(element, region)
      this.imageSpace.querySelector('.hmt-text-layer')?.append(element)
      this.regions.push({
        region,
        element,
        textElement,
        fontFamily: fontFamilies.get(region.style.fontId) ?? 'sans-serif',
      })
    }
    this.selection = new SelectionController(
      shadowRoot,
      popover,
      callbacks.lookup,
      this.forwardPrimaryClick,
    )
    for (const view of this.regions) {
      this.selection.register(view.element, payload.result.jobId, view.region.id)
    }
    this.originalButton.addEventListener('click', this.showOriginal)
    this.chineseButton.addEventListener('click', this.showChinese)
    this.originalButton.addEventListener('click', this.suppressControlNavigation)
    this.chineseButton.addEventListener('click', this.suppressControlNavigation)
    this.compareButton.addEventListener('click', this.suppressControlNavigation)
    this.compareButton.addEventListener('pointerdown', this.pressCompare)
    this.compareButton.addEventListener('pointerup', this.releaseCompare)
    this.compareButton.addEventListener('pointercancel', this.releaseCompare)
    this.compareButton.addEventListener('blur', this.releaseCompare)
    this.compareButton.addEventListener('keydown', this.compareKeyDown)
    this.compareButton.addEventListener('keyup', this.compareKeyUp)
    this.resizeObserver?.observe(candidate.element)
    this.refit()
    this.setMode('chinese')
  }

  get currentMode(): 'chinese' | 'original' {
    return this.mode
  }

  private readonly showOriginal = (): void => this.setMode('original')
  private readonly showChinese = (): void => this.setMode('chinese')
  private readonly suppressControlNavigation = (event: Event): void => {
    event.preventDefault()
    event.stopPropagation()
  }
  private readonly pressCompare = (event: Event): void => {
    event.preventDefault()
    this.applyVisualMode('original')
  }
  private readonly releaseCompare = (): void => this.applyVisualMode(this.mode)
  private readonly compareKeyDown = (event: KeyboardEvent): void => {
    if (event.key === ' ' || event.key === 'Enter') this.pressCompare(event)
  }
  private readonly compareKeyUp = (event: KeyboardEvent): void => {
    if (event.key === ' ' || event.key === 'Enter') this.releaseCompare()
  }
  private readonly forwardPrimaryClick = (event: MouseEvent): void => {
    if (event.button !== 0 || event.defaultPrevented) return
    const forwarded = new MouseEvent('click', {
      bubbles: false,
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

  setMode(mode: 'chinese' | 'original'): void {
    this.mode = mode
    this.originalButton.setAttribute('aria-pressed', String(mode === 'original'))
    this.chineseButton.setAttribute('aria-pressed', String(mode === 'chinese'))
    this.applyVisualMode(mode)
  }

  private applyVisualMode(mode: 'chinese' | 'original'): void {
    if (mode === 'chinese') {
      this.candidate.owner.style.setProperty('opacity', '0', 'important')
      this.candidate.element.style.setProperty('opacity', '0', 'important')
      this.viewport.hidden = false
    } else {
      restoreOpacity(this.candidate.owner, this.savedOwnerOpacity)
      restoreOpacity(this.candidate.element, this.savedImageOpacity)
      this.viewport.hidden = true
    }
  }

  refit(): void {
    if (this.destroyed || !this.candidate.element.isConnected) return
    this.geometry = calculateImageGeometry(
      this.candidate.element,
      this.wrapper,
      this.payload.result.sourceWidth,
      this.payload.result.sourceHeight,
    )
    setRect(this.viewport, this.geometry.viewport)
    setRect(this.imageSpace, this.geometry.image)
    for (const view of this.regions) {
      const fit = this.fitter.fit(
        view.region,
        this.geometry.image.width,
        this.geometry.image.height,
      )
      applyChosenLines(view.textElement, fit.lines, view.region.displayedChinese)
      let measuredFontSize = fit.fontSize
      applyRegionStyle(view.element, view.region, measuredFontSize, view.fontFamily)
      if (view.region.style.writingMode === 'vertical-rl') {
        view.element.style.alignItems = 'flex-start'
      }
      const minimumFontSize = Math.min(
        measuredFontSize,
        minimumFontSizeForImage(this.geometry.image.width),
      )
      while (
        measuredFontSize > minimumFontSize &&
        view.element.clientWidth > 0 &&
        (view.element.scrollWidth > view.element.clientWidth + 1 ||
          view.element.scrollHeight > view.element.clientHeight + 1)
      ) {
        measuredFontSize = Math.max(minimumFontSize, measuredFontSize - 0.5)
        applyRegionStyle(view.element, view.region, measuredFontSize, view.fontFamily)
      }
      if (
        fit.degraded ||
        (view.element.clientWidth > 0 &&
          (view.element.scrollWidth > view.element.clientWidth + 1 ||
            view.element.scrollHeight > view.element.clientHeight + 1))
      ) {
        view.element.dataset.fit = 'degraded'
        this.callbacks.onFitDegraded?.(view.region.id)
      } else {
        delete view.element.dataset.fit
      }
    }
  }

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.resizeObserver?.disconnect()
    this.selection.destroy()
    this.originalButton.removeEventListener('click', this.showOriginal)
    this.chineseButton.removeEventListener('click', this.showChinese)
    this.originalButton.removeEventListener('click', this.suppressControlNavigation)
    this.chineseButton.removeEventListener('click', this.suppressControlNavigation)
    this.compareButton.removeEventListener('click', this.suppressControlNavigation)
    this.compareButton.removeEventListener('pointerdown', this.pressCompare)
    this.compareButton.removeEventListener('pointerup', this.releaseCompare)
    this.compareButton.removeEventListener('pointercancel', this.releaseCompare)
    this.compareButton.removeEventListener('blur', this.releaseCompare)
    this.compareButton.removeEventListener('keydown', this.compareKeyDown)
    this.compareButton.removeEventListener('keyup', this.compareKeyUp)
    restoreOpacity(this.candidate.owner, this.savedOwnerOpacity)
    restoreOpacity(this.candidate.element, this.savedImageOpacity)
    this.candidate.element.removeAttribute('data-hmt-original')
    if (this.wrapper.parentNode) {
      const reference =
        this.originalNextSibling?.parentNode === this.originalParent
          ? this.originalNextSibling
          : this.wrapper
      this.originalParent.insertBefore(this.candidate.owner, reference)
      this.wrapper.remove()
    }
    URL.revokeObjectURL(this.cleanUrl)
    this.callbacks.onRestore?.()
  }
}

export class SelectableRenderer {
  private readonly fontLoader: FontLoader

  constructor(
    private readonly callbacks: RendererCallbacks,
    private readonly ResizeObserverType:
      | typeof ResizeObserver
      | undefined = globalThis.ResizeObserver,
    private readonly cleanImageDecoder: CleanImageDecoder = decodeCleanImage,
  ) {
    this.fontLoader = new FontLoader(callbacks.fetchFont)
  }

  async render(
    candidate: DiscoveredImage,
    payload: RenderPayload,
    guard: RenderGuard = { validate: () => undefined },
  ): Promise<RenderedImage> {
    guard.validate()
    if (!candidate.owner.isConnected || !candidate.element.isConnected) {
      throw new RendererError(
        'IMAGE_REPLACED_DURING_PROCESSING',
        'The page replaced this image before translation completed.',
      )
    }
    if (payload.result.sourceWidth < 1 || payload.result.sourceHeight < 1) {
      throw new RendererError('INVALID_RESULT_GEOMETRY', 'The result dimensions are invalid.')
    }
    if (
      payload.result.sourceWidth !== candidate.element.naturalWidth ||
      payload.result.sourceHeight !== candidate.element.naturalHeight
    ) {
      throw new RendererError(
        'RESULT_SOURCE_DIMENSIONS_MISMATCH',
        'The result dimensions do not match the live page image.',
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

    const cleanUrl = URL.createObjectURL(
      new Blob([payload.cleanImage], { type: payload.result.cleanImageMimeType }),
    )
    const clean = document.createElement('img')
    clean.className = 'hmt-clean-image'
    clean.alt = ''
    clean.draggable = false
    clean.src = cleanUrl
    try {
      await this.cleanImageDecoder(clean)
    } catch (error) {
      URL.revokeObjectURL(cleanUrl)
      throw error instanceof RendererError
        ? error
        : new RendererError(
            'CLEAN_IMAGE_DECODE_FAILED',
            'The cleaned image could not be decoded.',
          )
    }
    guard.validate()
    if (
      clean.naturalWidth !== payload.result.sourceWidth ||
      clean.naturalHeight !== payload.result.sourceHeight
    ) {
      URL.revokeObjectURL(cleanUrl)
      throw new RendererError(
        'CLEAN_IMAGE_DIMENSIONS_MISMATCH',
        'The cleaned image dimensions do not match the translation result.',
      )
    }

    const fontFamilies = new Map<string, string>()
    try {
      await Promise.all(
        payload.result.regions.map(async (region) => {
          if (fontFamilies.has(region.style.fontId)) return
          fontFamilies.set(
            region.style.fontId,
            await this.fontLoader.load(
              region.style.fontId,
              region.style.category,
              payload.result.jobId,
            ),
          )
        }),
      )
      guard.validate()
      if (document.fonts?.ready) {
        await document.fonts.ready
        guard.validate()
      }
    } catch (error) {
      URL.revokeObjectURL(cleanUrl)
      throw error
    }

    const originalParent = candidate.owner.parentNode
    if (!originalParent) {
      URL.revokeObjectURL(cleanUrl)
      throw new RendererError(
        'IMAGE_REPLACED_DURING_PROCESSING',
        'The page removed this image before translation completed.',
      )
    }
    const originalNextSibling = candidate.owner.nextSibling
    const before = candidate.element.getBoundingClientRect()
    const parentRect =
      originalParent instanceof Element
        ? originalParent.getBoundingClientRect()
        : undefined
    const wrapper = document.createElement('span')
    wrapper.dataset.hmtOwned = 'true'
    wrapper.className = 'hmt-wrapper'
    const ownerStyle = getComputedStyle(candidate.owner)
    wrapper.style.position = 'relative'
    wrapper.style.display = ownerStyle.display === 'block' ? 'block' : 'inline-block'
    wrapper.style.width =
      parentRect && Math.abs(parentRect.width - before.width) <= MAX_LAYOUT_SHIFT_PX
        ? '100%'
        : `${before.width}px`
    wrapper.style.maxWidth = '100%'
    wrapper.style.verticalAlign = ownerStyle.verticalAlign || 'baseline'
    guard.validate()
    originalParent.insertBefore(wrapper, candidate.owner)
    wrapper.append(candidate.owner)
    const after = candidate.element.getBoundingClientRect()
    if (before.width > 0 && rectDifference(before, after) > MAX_LAYOUT_SHIFT_PX) {
      originalParent.insertBefore(candidate.owner, wrapper)
      wrapper.remove()
      URL.revokeObjectURL(cleanUrl)
      throw new RendererError(
        'UNSUPPORTED_PAGE_LAYOUT',
        'Wrapping this image changed the page layout, so the original was restored.',
      )
    }

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
    const textLayer = document.createElement('span')
    textLayer.className = 'hmt-text-layer'
    imageSpace.append(clean, textLayer)
    viewport.append(imageSpace)

    const controls = document.createElement('span')
    controls.className = 'hmt-controls'
    controls.setAttribute('role', 'group')
    controls.setAttribute('aria-label', 'Translated image mode')
    const originalButton = createButton('Original')
    const chineseButton = createButton('Chinese')
    const compareButton = createButton('Hold to compare')
    compareButton.title = 'Press and hold to show the original'
    compareButton.setAttribute('aria-pressed', 'false')
    controls.append(originalButton, chineseButton, compareButton)
    const popover = document.createElement('span')
    popover.className = 'hmt-lookup'
    popover.hidden = true
    popover.setAttribute('role', 'dialog')
    popover.setAttribute('aria-label', 'Chinese dictionary')
    popover.addEventListener('click', (event) => {
      event.preventDefault()
      event.stopPropagation()
    })
    shadow.append(style, viewport, controls, popover)

    const savedOwnerOpacity = originalOwnerStyle(candidate.owner)
    const savedImageOpacity = originalOwnerStyle(candidate.element)
    let rendered: RenderedImage | undefined
    const resizeObserver = this.ResizeObserverType
      ? new this.ResizeObserverType(() => rendered?.refit())
      : undefined
    try {
      guard.validate()
      rendered = new RenderedImage(
        candidate,
        payload,
        wrapper,
        viewport,
        imageSpace,
        originalButton,
        chineseButton,
        compareButton,
        cleanUrl,
        originalParent,
        originalNextSibling,
        savedOwnerOpacity,
        savedImageOpacity,
        this.callbacks,
        shadow,
        popover,
        fontFamilies,
        resizeObserver,
      )
    } catch (error) {
      resizeObserver?.disconnect()
      restoreOpacity(candidate.owner, savedOwnerOpacity)
      restoreOpacity(candidate.element, savedImageOpacity)
      candidate.element.removeAttribute('data-hmt-original')
      if (wrapper.parentNode) {
        originalParent.insertBefore(candidate.owner, wrapper)
        wrapper.remove()
      }
      URL.revokeObjectURL(cleanUrl)
      throw error
    }
    return rendered
  }
}
