import type {
  ImageRegion,
  LookupRequest,
  LookupResult,
  PreservedImageRegion,
  TeachingTerm,
  TranslatedText,
} from '../contracts/browser'
import type { DiscoveredSurface } from '../discovery/surfaces'
import { attachComparisonControls, type ChapterDisplayMode } from '../document/controls'
import { ExplanationController } from '../selection/popover'
import { LOOKUP_CSS } from '../selection/popover-style'
import { MandarinSpeaker, type TextSpeaker } from '../selection/speech'
import {
  buildTeachingTextRange,
  installTranslatedTextMetadata,
} from '../language/translated-text'
import { FontLoader, type FontFetcher } from './font-loader'
import {
  calculateImageGeometry,
  type LocalImageBox,
  polygonBounds,
  type ImageGeometry,
} from './geometry'
import {
  fitPolygonForRegion,
  minimumReadableFontSize,
  PolygonTextFitter,
} from './fitting'
import { applyRegionColorBands, applyRegionStyle } from './style'

const MEASUREMENT_SEARCH_STEPS = 16
const FIT_SAFETY_RATIO = 0.98

const RENDERER_CSS = `
:host {
  inset: 0;
  pointer-events: none;
  position: absolute;
  z-index: 1;
}
*, *::before, *::after { box-sizing: border-box; }
.hskify-viewport {
  overflow: hidden;
  pointer-events: none;
  position: absolute;
}
.hskify-image-space {
  position: absolute;
}
.hskify-patch-layer,
.hskify-text-layer {
  height: 100%;
  inset: 0;
  pointer-events: none;
  position: absolute;
  width: 100%;
}
.hskify-patch {
  object-fit: fill;
  pointer-events: none;
  position: absolute;
  user-select: none;
}
.hskify-region {
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
.hskify-region-text { display: block; }
.hskify-region-line { display: block; }
.hskify-source-notice {
  color: transparent;
  cursor: help;
  font: 600 1em/1.05 system-ui, sans-serif;
  pointer-events: auto;
  text-shadow: none;
}
.hskify-learning-term {
  text-decoration-line: underline;
  text-decoration-style: dotted;
  text-decoration-thickness: 0.06em;
  text-underline-offset: 0.12em;
}
.hskify-region:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}
${LOOKUP_CSS}
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
  onFitDegraded?: (itemId: string) => void
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
  region: ImageRegion
  patch: HTMLImageElement
  patchUrl: string
  element: HTMLElement
  textElement: HTMLElement
  fontFamily: string
}

type SourcePreservingView = {
  element: HTMLElement
  itemId: string
}

type SourcePreservingRegion = Pick<
  PreservedImageRegion,
  'itemId' | 'textPolygon' | 'sourceText' | 'itemOrder'
> & { translatedText?: TranslatedText }

function px(value: number): string {
  return `${Number.isFinite(value) ? value : 0}px`
}

type SurfaceTransform = Readonly<{
  box: LocalImageBox
  left: number
  top: number
  a: number
  b: number
  c: number
  d: number
  e: number
  f: number
}>

type SurfaceTransformResult = SurfaceTransform | null | 'unsupported'

type QuadPoint = Readonly<{ x: number; y: number }>
type ElementQuad = Readonly<{
  p1: QuadPoint
  p2: QuadPoint
  p3: QuadPoint
  p4: QuadPoint
}>

type BoxQuadElement = Element & {
  getBoxQuads?: () => readonly ElementQuad[]
}

function layoutBox(element: Element): LocalImageBox | undefined {
  const html = element as HTMLElement
  const style = element.ownerDocument.defaultView?.getComputedStyle(element)
  // `offsetWidth/offsetHeight` are the authoritative layout measurements in a
  // live browser, but they are zero for detached/virtualized reader surfaces
  // and for DOM realms used by packaged-reader tests.  The rendered quad is a
  // valid CSS-space measurement in both cases, so use it as the next source of
  // truth before falling back to intrinsic image dimensions.
  const rect = element.getBoundingClientRect()
  const intrinsic = element as HTMLImageElement
  const width =
    html.offsetWidth ||
    Number.parseFloat(style?.width || '') ||
    rect.width ||
    intrinsic.naturalWidth ||
    undefined
  const height =
    html.offsetHeight ||
    Number.parseFloat(style?.height || '') ||
    rect.height ||
    intrinsic.naturalHeight ||
    undefined
  if (
    typeof width !== 'number' ||
    typeof height !== 'number' ||
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return undefined
  }
  return Object.freeze({ width, height })
}

function cssMatrix(value: string):
  | Readonly<{ a: number; b: number; c: number; d: number; e: number; f: number }>
  | undefined {
  const normalized = value.trim()
  if (!normalized || normalized === 'none') return undefined
  const matrix = normalized.match(/^matrix\(([^)]+)\)$/i)
  if (matrix) {
    const values = matrix[1]!.split(',').map((entry) => Number.parseFloat(entry.trim()))
    if (values.length !== 6 || values.some((entry) => !Number.isFinite(entry))) return undefined
    return { a: values[0]!, b: values[1]!, c: values[2]!, d: values[3]!, e: values[4]!, f: values[5]! }
  }
  const matrix3d = normalized.match(/^matrix3d\(([^)]+)\)$/i)
  if (!matrix3d) return undefined
  const values = matrix3d[1]!.split(',').map((entry) => Number.parseFloat(entry.trim()))
  if (values.length !== 16 || values.some((entry) => !Number.isFinite(entry))) return undefined
  // A 3-D perspective transform cannot be represented by one overlay affine
  // matrix. Pure 2-D matrix3d output is safe to flatten.
  const unsupported = [2, 3, 6, 7, 8, 9, 11, 14].some((index) => Math.abs(values[index]!) > 1e-6)
  if (unsupported || Math.abs(values[10]! - 1) > 1e-6 || Math.abs(values[15]! - 1) > 1e-6) {
    return undefined
  }
  return { a: values[0]!, b: values[1]!, c: values[4]!, d: values[5]!, e: values[12]!, f: values[13]! }
}

function isIdentity(matrix: { a: number; b: number; c: number; d: number; e: number; f: number }): boolean {
  return (
    Math.abs(matrix.a - 1) < 1e-6 &&
    Math.abs(matrix.b) < 1e-6 &&
    Math.abs(matrix.c) < 1e-6 &&
    Math.abs(matrix.d - 1) < 1e-6 &&
    Math.abs(matrix.e) < 1e-6 &&
    Math.abs(matrix.f) < 1e-6
  )
}

function transformOrigin(value: string, box: LocalImageBox): { x: number; y: number } {
  const values = value.trim().split(/\s+/u)
  const resolve = (token: string | undefined, basis: number): number => {
    if (!token) return basis / 2
    if (token.endsWith('%')) {
      const percent = Number.parseFloat(token.slice(0, -1))
      return Number.isFinite(percent) ? basis * percent / 100 : basis / 2
    }
    const pixels = Number.parseFloat(token)
    return Number.isFinite(pixels) ? pixels : basis / 2
  }
  return { x: resolve(values[0], box.width), y: resolve(values[1], box.height) }
}

function hasTransformedAncestor(element: Element): boolean {
  const ownerWindow = element.ownerDocument.defaultView
  for (let ancestor = element.parentElement; ancestor; ancestor = ancestor.parentElement) {
    const style = ownerWindow?.getComputedStyle(ancestor)
    // Some DOM realms expose an empty string rather than CSS's explicit
    // `none`. Empty means “no declared transform”, not an unsupported one.
    if (style?.transform && style.transform !== 'none') return true
    if (style?.perspective && style.perspective !== 'none') return true
  }
  return false
}

function measureSurfaceTransform(element: Element): SurfaceTransformResult {
  const ownerWindow = element.ownerDocument.defaultView
  const style = ownerWindow?.getComputedStyle(element) ?? getComputedStyle(element)
  const box = layoutBox(element)
  if (!box) return 'unsupported'
  const rect = element.getBoundingClientRect()
  const quads = (element as BoxQuadElement).getBoxQuads?.()
  const quad = quads?.[0]
  if (quad) {
    const width = Math.hypot(quad.p2.x - quad.p1.x, quad.p2.y - quad.p1.y)
    const height = Math.hypot(quad.p4.x - quad.p1.x, quad.p4.y - quad.p1.y)
    if (width <= 0 || height <= 0) return 'unsupported'
    const expectedP3 = {
      x: quad.p2.x + quad.p4.x - quad.p1.x,
      y: quad.p2.y + quad.p4.y - quad.p1.y,
    }
    if (Math.hypot(expectedP3.x - quad.p3.x, expectedP3.y - quad.p3.y) > 2) {
      return 'unsupported'
    }
    const transform = {
      box,
      left: quad.p1.x,
      top: quad.p1.y,
      a: (quad.p2.x - quad.p1.x) / box.width,
      b: (quad.p2.y - quad.p1.y) / box.width,
      c: (quad.p4.x - quad.p1.x) / box.height,
      d: (quad.p4.y - quad.p1.y) / box.height,
      e: 0,
      f: 0,
    }
    if (
      isIdentity(transform) &&
      Math.abs(quad.p1.x - rect.left) < 0.5 &&
      Math.abs(quad.p1.y - rect.top) < 0.5
    ) {
      return null
    }
    return transform
  }
  const matrix = cssMatrix(style.transform || '')
  if (!matrix) {
    if (style.transform && style.transform !== 'none') return 'unsupported'
    // Without getBoxQuads an ancestor transform cannot be reproduced by a
    // body-mounted overlay. Fail visibly instead of painting a page-offset
    // block that only happens to align in the untransformed case.
    return hasTransformedAncestor(element) ? 'unsupported' : null
  }
  if (isIdentity(matrix)) return null
  // Browsers without getBoxQuads can still be handled for a pure 2-D
  // transform on the element itself. Ancestor transforms require quads and
  // remain an explicit visible unsupported state.
  const origin = transformOrigin(style.transformOrigin || '50% 50%', box)
  const corners = [
    [0, 0],
    [box.width, 0],
    [0, box.height],
    [box.width, box.height],
  ].map(([x = 0, y = 0]) => ({
    x: matrix.a * (x - origin.x) + matrix.c * (y - origin.y) + matrix.e + origin.x,
    y: matrix.b * (x - origin.x) + matrix.d * (y - origin.y) + matrix.f + origin.y,
  }))
  const minX = Math.min(...corners.map((point) => point.x))
  const minY = Math.min(...corners.map((point) => point.y))
  const baseLeft = rect.left - minX
  const baseTop = rect.top - minY
  return {
    box,
    left: baseLeft,
    top: baseTop,
    a: matrix.a,
    b: matrix.b,
    c: matrix.c,
    d: matrix.d,
    e: -matrix.a * origin.x - matrix.c * origin.y + matrix.e + origin.x,
    f: -matrix.b * origin.x - matrix.d * origin.y + matrix.f + origin.y,
  }
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

function setPercentRegion(element: HTMLElement, region: ImageRegion): void {
  const points = fitPolygonForRegion(region)
  const bounds = polygonBounds(points)
  element.style.left = `${bounds.minX * 100}%`
  element.style.top = `${bounds.minY * 100}%`
  element.style.width = `${bounds.width * 100}%`
  element.style.height = `${bounds.height * 100}%`
}

function setPercentPolygon(
  element: HTMLElement,
  points: readonly { x: number; y: number }[],
): void {
  const bounds = polygonBounds(points)
  element.style.left = `${bounds.minX * 100}%`
  element.style.top = `${bounds.minY * 100}%`
  element.style.width = `${bounds.width * 100}%`
  element.style.height = `${bounds.height * 100}%`
}

function setPercentPatch(element: HTMLElement, region: ImageRegion): void {
  const rect = region.patch.rect
  element.style.left = `${rect.x * 100}%`
  element.style.top = `${rect.y * 100}%`
  element.style.width = `${rect.width * 100}%`
  element.style.height = `${rect.height * 100}%`
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

function applyChosenLines(
  documentRef: Document,
  element: HTMLElement,
  lines: readonly string[],
  text: string,
  terms: readonly TeachingTerm[] = [],
): void {
  const chosen = lines.length > 0 && lines.join('') === text ? lines : [text]
  const nodes: Node[] = []
  let lineStart = 0
  for (const line of chosen) {
    const lineElement = documentRef.createElement('span')
    lineElement.className = 'hskify-region-line'
    const lineEnd = lineStart + [...line].length
    lineElement.append(
      buildTeachingTextRange(documentRef, text, terms, lineStart, lineEnd),
    )
    nodes.push(lineElement)
    lineStart = lineEnd
  }
  element.replaceChildren(...nodes)
}

function regionElement(region: ImageRegion, documentRef: Document): {
  element: HTMLElement
  textElement: HTMLElement
} {
  const element = documentRef.createElement('span')
  element.className = 'hskify-region'
  element.tabIndex = 0
  element.dataset.hskifyItemId = region.itemId
  installTranslatedTextMetadata(element, region.text)
  const textElement = documentRef.createElement('span')
  textElement.className = 'hskify-region-text'
  // Companion text always enters the page through text nodes.
  textElement.textContent = region.text.displayedChinese
  element.append(textElement)
  setPercentRegion(element, region)
  return { element, textElement }
}

function actualOverflow(element: HTMLElement, content?: HTMLElement): boolean {
  const scrollOverflow =
    element.scrollWidth > element.clientWidth + 0.5 ||
    element.scrollHeight > element.clientHeight + 0.5
  if (scrollOverflow || !content) return scrollOverflow
  const outer = element.getBoundingClientRect()
  const inner = content.getBoundingClientRect()
  if (outer.width <= 0 || outer.height <= 0 || inner.width <= 0 || inner.height <= 0) {
    return false
  }
  return (
    inner.left < outer.left + 0.5 ||
    inner.right > outer.right - 0.5 ||
    inner.top < outer.top + 0.5 ||
    inner.bottom > outer.bottom - 0.5
  )
}

export class RenderedImage {
  private mode: ChapterDisplayMode = 'chinese'
  private readonly regions = new Map<string, RegionView>()
  private readonly sourcePreserving = new Map<string, SourcePreservingView>()
  private readonly fitter = new PolygonTextFitter()
  private readonly explanation: ExplanationController
  private destroyed = false
  private geometry?: ImageGeometry
  private transformFrame: SurfaceTransform | undefined = undefined

  private markFitDegraded(view: RegionView): void {
    const alreadyReported = view.element.dataset.hskifyFit === 'degraded'
    view.element.dataset.hskifyFit = 'degraded'
    if (!alreadyReported) this.callbacks.onFitDegraded?.(view.region.itemId)
  }

  constructor(
    readonly candidate: DiscoveredSurface,
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
    private readonly transformedSurface = false,
    private readonly resizeObserver?: ResizeObserver,
    private readonly onDestroy?: (rendered: RenderedImage) => void,
  ) {
    this.explanation = new ExplanationController(
      shadowRoot,
      popover,
      callbacks.lookup,
      this.forwardPrimaryClick,
      speaker,
    )
    this.resizeObserver?.observe(candidate.element)
    candidate.element.ownerDocument.addEventListener('scroll', this.schedulePosition, true)
    candidate.element.ownerDocument.defaultView?.addEventListener('resize', this.scheduleRefit)
    this.refit()
  }

  get currentMode(): ChapterDisplayMode {
    return this.mode
  }

  get regionCount(): number {
    return this.regions.size
  }

  regionsInReadingOrder(): ImageRegion[] {
    return [...this.regions.values()]
      .map((view) => view.region)
      .sort((left, right) => left.itemOrder - right.itemOrder)
  }

  private readonly forwardPrimaryClick = (event: MouseEvent): void => {
    if (event.button !== 0 || event.defaultPrevented) return
    const forwarded = new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      composed: false,
      view: this.candidate.element.ownerDocument.defaultView ?? null,
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

  setMode(mode: ChapterDisplayMode): void {
    if (this.destroyed) return
    this.mode = mode
    if (mode === 'original') this.explanation.dismiss()
    this.applyVisualMode(mode)
  }

  showOriginalForComparison(): void {
    if (!this.destroyed) this.applyVisualMode('original')
  }

  restoreSelectedMode(): void {
    if (!this.destroyed) this.applyVisualMode(this.mode)
  }

  private applyVisualMode(mode: ChapterDisplayMode): void {
    // The page's original image remains connected and visible at all times.
    // Comparison only toggles the transparent patch/text overlay.
    this.viewport.hidden = mode === 'original'
  }

  private updateRegionMetadata(view: RegionView): void {
    installTranslatedTextMetadata(view.element, view.region.text)
  }

  async installRegion(
    region: ImageRegion,
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
    const patch = this.candidate.element.ownerDocument.createElement('img')
    patch.className = 'hskify-patch'
    patch.alt = ''
    patch.draggable = false
    patch.dataset.hskifyPatchId = region.patch.blobId
    patch.src = patchUrl
    patch.style.zIndex = String(Math.max(0, region.itemOrder))
    setPercentPatch(patch, region)
    let fontFamily: string
    try {
      const [, loadedFontFamily] = await Promise.all([
        this.patchImageDecoder(patch),
      this.fontLoader.load(
        region.style.fontId,
        region.style.category,
        this.job.jobId,
        this.candidate.element.ownerDocument,
      ),
      ])
      fontFamily = loadedFontFamily
      guard.validate()
      const fontSet = this.candidate.element.ownerDocument.fonts
      if (fontSet?.ready) {
        await fontSet.ready
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

    const created = regionElement(region, this.candidate.element.ownerDocument)
    created.element.style.zIndex = String(Math.max(0, region.itemOrder))
    const next: RegionView = {
      region,
      patch,
      patchUrl,
      element: created.element,
      textElement: created.textElement,
      fontFamily,
    }
    const previous = this.regions.get(region.itemId)

    // A mathematical fit below the readable floor is a preservation outcome,
    // not a reason to paint an unreadable patch over the source artwork.
    if (
      this.geometry &&
      this.geometry.image.width > 0 &&
      this.geometry.image.height > 0 &&
      this.fitter.fit(region, this.geometry.image.width, this.geometry.image.height).degraded
    ) {
      URL.revokeObjectURL(patchUrl)
      this.removeTranslatedRegion(previous)
      this.installSourcePreservingRegion({
        itemId: region.itemId,
        textPolygon: region.textPolygon,
        sourceText: region.text.sourceText,
        itemOrder: region.itemOrder,
        translatedText: region.text,
      })
      this.callbacks.onFitDegraded?.(region.itemId)
      return
    }

    // The decoded patch is inserted synchronously before its selectable text.
    // No page state can expose Chinese over an undecoded/absent inpaint.
    this.removeSourcePreservingRegion(region.itemId)
    if (previous) {
      previous.patch.replaceWith(patch)
      previous.element.replaceWith(created.element)
      this.explanation.unregister(previous.element)
    } else {
      this.patchLayer.append(patch)
      this.textLayer.append(created.element)
    }
    this.regions.set(region.itemId, next)
    this.explanation.register(created.element, this.job.jobId, region.itemId)
    this.updateRegionMetadata(next)
    if (!this.refitView(next)) {
      this.explanation.unregister(created.element)
      created.element.remove()
      patch.remove()
      URL.revokeObjectURL(patchUrl)
      this.removeTranslatedRegion(previous)
      this.installSourcePreservingRegion({
        itemId: region.itemId,
        textPolygon: region.textPolygon,
        sourceText: region.text.sourceText,
        itemOrder: region.itemOrder,
        translatedText: region.text,
      })
      return
    }
    if (previous) URL.revokeObjectURL(previous.patchUrl)
  }

  private removeTranslatedRegion(view: RegionView | undefined): void {
    if (!view) return
    this.explanation.unregister(view.element)
    view.element.remove()
    view.patch.remove()
    URL.revokeObjectURL(view.patchUrl)
    if (this.regions.get(view.region.itemId) === view) this.regions.delete(view.region.itemId)
  }

  private removeSourcePreservingRegion(itemId: string): void {
    const previous = this.sourcePreserving.get(itemId)
    if (!previous) return
    this.explanation.unregister(previous.element)
    previous.element.remove()
    this.sourcePreserving.delete(itemId)
  }

  /**
   * Keep stylized or low-confidence source pixels untouched while exposing
   * the recognized source span to the same hover dictionary route. The hit
   * target is transparent and therefore cannot create a guessed overlay.
   */
  installSourcePreservingRegion(region: SourcePreservingRegion): void {
    if (this.destroyed || region.textPolygon.length < 3 || !region.sourceText.trim()) return
    this.removeSourcePreservingRegion(region.itemId)
    const documentRef = this.candidate.element.ownerDocument
    const element = documentRef.createElement('span')
    element.className = 'hskify-region hskify-source-notice'
    element.lang = 'en'
    element.tabIndex = 0
    element.dataset.hskifyItemId = region.itemId
    element.dataset.hskifySourceText = region.sourceText
    if (region.translatedText) {
      installTranslatedTextMetadata(element, region.translatedText)
    }
    const displayText = region.translatedText?.displayedChinese.trim() || region.sourceText
    element.setAttribute(
      'aria-label',
      region.translatedText?.pinyin
        ? `${displayText}; ${region.translatedText.pinyin}`
        : displayText,
    )
    element.style.zIndex = String(Math.max(0, region.itemOrder))
    setPercentPolygon(element, region.textPolygon)
    const text = documentRef.createElement('span')
    text.className = 'hskify-region-text'
    // The transparent hit target follows the final hover translation when
    // available. The original artwork remains the only visible pixels.
    text.textContent = displayText
    element.append(text)
    this.textLayer.append(element)
    this.sourcePreserving.set(region.itemId, { element, itemId: region.itemId })
    this.explanation.register(element, this.job.jobId, region.itemId)
  }

  private refitView(view: RegionView): boolean {
    if (
      !this.geometry ||
      this.geometry.image.width <= 0 ||
      this.geometry.image.height <= 0
    ) {
      return true
    }
    const fit = this.fitter.fit(
      view.region,
      this.geometry.image.width,
      this.geometry.image.height,
    )
    const minimumFontSize = minimumReadableFontSize(this.geometry.image.width)
    if (fit.degraded || fit.fontSize < minimumFontSize) {
      this.markFitDegraded(view)
      return false
    }
    applyChosenLines(
      this.candidate.element.ownerDocument,
      view.textElement,
      fit.lines,
      view.region.text.displayedChinese,
      view.region.text.hsk.teachingTerms,
    )
    this.updateRegionMetadata(view)
    const applyStyle = (fontSize: number): void => {
      applyRegionStyle(view.element, view.region, fontSize, view.fontFamily)
      applyRegionColorBands(view.textElement, view.region, fontSize)
    }
    applyStyle(fit.fontSize)
    // Center both writing modes inside the safe polygon. In vertical-rl,
    // flex-start pins the glyph column to the bubble edge and makes the DOM
    // measurement report a false overflow even when the mathematical fit is
    // valid.
    view.element.style.alignItems = 'center'

    // Keep a small measured-layout margin so fractional CSS zoom and device
    // pixel rounding cannot turn an exact fit into a clipped final glyph.
    let measuredSize = Math.max(minimumFontSize, fit.fontSize * FIT_SAFETY_RATIO)
    applyStyle(measuredSize)
    if (actualOverflow(view.element, view.textElement)) {
      let low = minimumFontSize
      let high = measuredSize
      applyStyle(low)
      if (actualOverflow(view.element, view.textElement)) {
        this.markFitDegraded(view)
        return false
      }
      for (let iteration = 0; iteration < MEASUREMENT_SEARCH_STEPS; iteration += 1) {
        const midpoint = (low + high) / 2
        applyStyle(midpoint)
        if (actualOverflow(view.element, view.textElement)) high = midpoint
        else low = midpoint
      }
      measuredSize = Math.max(minimumFontSize, low * 0.995)
      applyStyle(measuredSize)
    }
    if (actualOverflow(view.element, view.textElement) || measuredSize < minimumFontSize) {
      this.markFitDegraded(view)
      return false
    } else {
      delete view.element.dataset.hskifyFit
    }
    return true
  }

  private positionFrame: number | null = null
  private refitFrame: number | null = null
  private readonly schedulePosition = (event: Event): void => {
    if (this.destroyed || this.positionFrame !== null) return
    const documentRef = this.candidate.element.ownerDocument
    // A document-anchored overlay moves with normal page scrolling in the
    // compositor. Only nested scrolling containers require a coordinate
    // refresh.
    if (event.target === documentRef || event.target === documentRef.defaultView) return
    const view = documentRef.defaultView
    if (!view) {
      this.positionWrapper()
      return
    }
    this.positionFrame = view.requestAnimationFrame(() => {
      this.positionFrame = null
      this.positionWrapper()
    })
  }

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

  private positionWrapper(): void {
    if (this.destroyed || !this.candidate.element.isConnected) return
    if (this.transformedSurface) {
      const measured = measureSurfaceTransform(this.candidate.element)
      if (measured === 'unsupported') return
      this.transformFrame = measured ?? undefined
      if (measured) {
        this.wrapper.style.transformOrigin = '0 0'
        this.wrapper.style.transform = `matrix(${measured.a}, ${measured.b}, ${measured.c}, ${measured.d}, ${measured.e}, ${measured.f})`
        this.positionWrapperAt(measured.left, measured.top, measured.box.width, measured.box.height)
        return
      }
    } else {
      this.transformFrame = undefined
      this.wrapper.style.transform = 'none'
      this.wrapper.style.transformOrigin = '0 0'
    }
    const imageRect = this.candidate.element.getBoundingClientRect()
    this.positionWrapperAt(imageRect.left, imageRect.top, imageRect.width, imageRect.height)
  }

  private positionWrapperAt(left: number, top: number, width: number, height: number): void {
    const offsetParent = this.wrapper.offsetParent
    if (offsetParent && offsetParent.nodeType === 1) {
      const parent = offsetParent as HTMLElement
      const parentRect = offsetParent.getBoundingClientRect()
      this.wrapper.style.left = px(
        left - parentRect.left + parent.scrollLeft - parent.clientLeft,
      )
      this.wrapper.style.top = px(
        top - parentRect.top + parent.scrollTop - parent.clientTop,
      )
    } else {
      const view = this.candidate.element.ownerDocument.defaultView
      this.wrapper.style.left = px(left + (view?.scrollX ?? 0))
      this.wrapper.style.top = px(top + (view?.scrollY ?? 0))
    }
    this.wrapper.style.width = px(width)
    this.wrapper.style.height = px(height)
  }

  refit(): void {
    if (this.destroyed || !this.candidate.element.isConnected) return
    this.positionWrapper()
    this.geometry = calculateImageGeometry(
      this.candidate.element,
      this.wrapper,
      this.job.sourceWidth,
      this.job.sourceHeight,
      this.transformFrame?.box,
    )
    setRect(this.viewport, this.geometry.viewport)
    setRect(this.imageSpace, this.geometry.image)
    for (const view of [...this.regions.values()]) {
      if (this.refitView(view)) continue
      const region = view.region
      this.removeTranslatedRegion(view)
      this.installSourcePreservingRegion({
        itemId: region.itemId,
        textPolygon: region.textPolygon,
        sourceText: region.text.sourceText,
        itemOrder: region.itemOrder,
        translatedText: region.text,
      })
    }
  }

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.resizeObserver?.disconnect()
    const documentRef = this.candidate.element.ownerDocument
    documentRef.removeEventListener('scroll', this.schedulePosition, true)
    documentRef.defaultView?.removeEventListener('resize', this.scheduleRefit)
    if (this.positionFrame !== null) {
      documentRef.defaultView?.cancelAnimationFrame(this.positionFrame)
      this.positionFrame = null
    }
    if (this.refitFrame !== null) {
      documentRef.defaultView?.cancelAnimationFrame(this.refitFrame)
      this.refitFrame = null
    }
    this.explanation.destroy()
    this.candidate.element.removeAttribute('data-hskify-original')
    this.wrapper.remove()
    for (const view of this.regions.values()) URL.revokeObjectURL(view.patchUrl)
    this.regions.clear()
    this.sourcePreserving.clear()
    this.onDestroy?.(this)
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
    private readonly patchImageDecoder: PatchImageDecoder = decodePatchImage,
    private readonly speaker: TextSpeaker = new MandarinSpeaker(),
  ) {
    this.fontLoader = new FontLoader(callbacks.fetchFont)
  }

  begin(
    candidate: DiscoveredSurface,
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
      job.sourceWidth !== candidate.sourceWidth ||
      job.sourceHeight !== candidate.sourceHeight
    ) {
      throw new RendererError(
        'RESULT_SOURCE_DIMENSIONS_MISMATCH',
        'The translation job dimensions do not match the live page image.',
      )
    }
    const documentRef = candidate.element.ownerDocument
    const surfaceTransform = measureSurfaceTransform(candidate.element)
    if (surfaceTransform === 'unsupported') {
      throw new RendererError(
        'UNSUPPORTED_IMAGE_TRANSFORM',
        'This transformed page surface cannot be aligned safely in this browser.',
      )
    }

    const originalParent = candidate.owner.parentNode
    if (!originalParent) {
      throw new RendererError(
        'IMAGE_REPLACED_DURING_PROCESSING',
        'The page removed this image before translation started.',
      )
    }
    const wrapper = documentRef.createElement('span')
    wrapper.dataset.hskifyOwned = 'true'
    if (candidate.element.dataset.page) {
      wrapper.dataset.hskifySourcePage = candidate.element.dataset.page
    }
    wrapper.className = 'hskify-wrapper'
    wrapper.style.contain = 'layout style'
    wrapper.style.display = 'block'
    wrapper.style.pointerEvents = 'none'
    // Document anchoring lets the browser compositor move the overlay with
    // the image during normal scrolling. Fixed positioning would require a
    // main-thread layout read and coordinate rewrite on every scroll frame.
    wrapper.style.position = 'absolute'
    wrapper.style.zIndex = '2147483000'
    guard.validate()
    ;(documentRef.body ?? documentRef.documentElement).append(wrapper)

    candidate.element.setAttribute('data-hskify-original', 'true')
    const host = documentRef.createElement('span')
    host.dataset.hskifyOwned = 'true'
    host.setAttribute('aria-label', 'Hskify chapter translation controls')
    wrapper.append(host)
    const shadow = host.attachShadow({ mode: 'open' })
    const style = documentRef.createElement('style')
    style.textContent = RENDERER_CSS
    const viewport = documentRef.createElement('span')
    viewport.className = 'hskify-viewport'
    const imageSpace = documentRef.createElement('span')
    imageSpace.className = 'hskify-image-space'
    const patchLayer = documentRef.createElement('span')
    patchLayer.className = 'hskify-patch-layer'
    const textLayer = documentRef.createElement('span')
    textLayer.className = 'hskify-text-layer'
    imageSpace.append(patchLayer, textLayer)
    viewport.append(imageSpace)

    const popover = documentRef.createElement('span')
    popover.className = 'hskify-lookup'
    popover.hidden = true
    popover.setAttribute('role', 'dialog')
    popover.setAttribute('aria-label', 'Chinese dictionary')
    popover.addEventListener('click', (event) => {
      event.preventDefault()
      event.stopPropagation()
    })
    shadow.append(style, viewport, popover)

    let rendered: RenderedImage | undefined
    let releaseControls: (() => void) | undefined
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
        surfaceTransform !== null,
        resizeObserver,
        () => releaseControls?.(),
      )
      releaseControls = attachComparisonControls(candidate.element.ownerDocument, rendered)
    } catch (error) {
      resizeObserver?.disconnect()
      candidate.element.removeAttribute('data-hskify-original')
      wrapper.remove()
      throw error
    }
    return rendered
  }
}
