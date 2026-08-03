/**
 * Reader-agnostic page surfaces.
 *
 * The translation pipeline must not care whether a publisher renders a page
 * as an <img>, a CSS background, a canvas, or an accessible same-origin
 * frame.  Adapters expose one immutable identity and one pixel capture
 * function.  Cross-origin/protected frames are reported as unsupported and
 * are never probed or bypassed.
 */

import { DEFAULT_IMAGE_LIMITS, normalizeMimeType } from '../acquisition/image-format'

export type PageSurfaceKind = 'image' | 'background' | 'canvas' | 'webgl' | 'frame'

export type SurfaceRect = Readonly<{
  x: number
  y: number
  width: number
  height: number
}>

export type SurfaceCapture = Readonly<{
  bytes: ArrayBuffer
  /** The response declaration is advisory; acquisition validates the bytes. */
  mimeType?: 'image/png' | 'image/jpeg' | 'image/webp' | 'image/gif'
  width: number
  height: number
}>

export type PageSurface = Readonly<{
  id: string
  kind: PageSurfaceKind
  element: Element
  pageIndex: number
  sourceUrl?: string
  /** True when the pixels are supplied by the capture callback rather than a fetchable URL. */
  captureOnly?: boolean
  width: number
  height: number
  rect: SurfaceRect
  visible: boolean
  continuous: boolean
  capture: (signal?: AbortSignal) => Promise<SurfaceCapture | undefined>
}>

/**
 * The browser pipeline's immutable page candidate.  Images, CSS surfaces,
 * canvases, and reader-owned surfaces all use this same contract after
 * acquisition; the daemon does not need to know which DOM primitive produced
 * the pixels.
 */
export type DiscoveredSurface = {
  id: string
  kind: PageSurfaceKind
  element: HTMLElement
  owner: HTMLElement
  sourceUrl: string
  /** Capture bytes are authoritative; sourceUrl is only a stable identity. */
  captureOnly?: boolean
  sourceWidth: number
  sourceHeight: number
  domIndex: number
  visible: boolean
  capture?: (signal?: AbortSignal) => Promise<SurfaceCapture | undefined>
}

export type SurfaceDiscoveryEvent =
  | { type: 'added'; candidate: DiscoveredSurface }
  | {
      type: 'updated'
      candidate: DiscoveredSurface
      previousSourceUrl: string
      previousDomIndex: number
    }
  | { type: 'removed'; candidate: DiscoveredSurface }
  | { type: 'visibility'; candidate: DiscoveredSurface }

export type UnsupportedSurface = Readonly<{
  kind: 'frame' | 'canvas' | 'background'
  element: Element
  reason: 'cross-origin' | 'protected' | 'not-readable' | 'empty'
}>

export type SurfaceDiscovery = Readonly<{
  surfaces: readonly PageSurface[]
  unsupported: readonly UnsupportedSurface[]
}>

// Keep the generic image adapter from promoting tiny lazy placeholders (or
// controls) into page surfaces. The specialised image adapter owns the same
// minimum intrinsic size and will rescan a deferred image when its real
// source becomes available.
const MIN_IMAGE_WIDTH = 320
const MIN_IMAGE_HEIGHT = 240

function rectOf(element: Element): SurfaceRect {
  const rect = element.getBoundingClientRect()
  return Object.freeze({ x: rect.left, y: rect.top, width: rect.width, height: rect.height })
}

// Same-origin reader frames have their own Window constructors.  `instanceof
// HTMLImageElement` therefore fails for otherwise ordinary frame content;
// surface discovery uses the DOM's stable tag/shape contract instead.
function isImageElement(element: Element): element is HTMLImageElement {
  return element.tagName.toLowerCase() === 'img' && 'naturalWidth' in element
}

function isCanvasElement(element: Element): element is HTMLCanvasElement {
  return element.tagName.toLowerCase() === 'canvas' && 'toDataURL' in element
}

function isHtmlElement(element: Element): element is HTMLElement {
  return element.nodeType === 1
}

function visibleInViewport(element: Element, ownerDocument: Document = element.ownerDocument): boolean {
  const rect = element.getBoundingClientRect()
  return rectVisibleInViewport(rect, ownerDocument)
}

function rectVisibleInViewport(
  rect: Pick<DOMRect, 'top' | 'right' | 'bottom' | 'left'>,
  ownerDocument: Document,
): boolean {
  const ownerWindow = ownerDocument.defaultView ?? window
  return (
    rect.bottom > 0 &&
    rect.right > 0 &&
    rect.top < (ownerWindow.innerHeight || ownerDocument.documentElement.clientHeight) &&
    rect.left < (ownerWindow.innerWidth || ownerDocument.documentElement.clientWidth)
  )
}

/** Convert a child-frame rect into the coordinate space of the top reader. */
function globalRect(element: Element, root: Document): SurfaceRect {
  let rect = element.getBoundingClientRect()
  let ownerDocument = element.ownerDocument
  while (ownerDocument !== root) {
    const frame = ownerDocument.defaultView?.frameElement
    if (!frame) break
    const frameRect = frame.getBoundingClientRect()
    rect = {
      left: rect.left + frameRect.left,
      top: rect.top + frameRect.top,
      right: rect.right + frameRect.left,
      bottom: rect.bottom + frameRect.top,
      width: rect.width,
      height: rect.height,
      x: rect.x + frameRect.left,
      y: rect.y + frameRect.top,
    } as DOMRect
    ownerDocument = frame.ownerDocument
  }
  return { x: rect.left, y: rect.top, width: rect.width, height: rect.height }
}

function surfaceIdentityUrl(surface: PageSurface): string {
  // A canvas/background may not have a fetchable URL.  Give it a stable,
  // same-origin identity while the capture bytes travel inline with submit.
  // The query (rather than a fragment) survives background normalization.
  const ownerDocument = surface.element.ownerDocument
  const url = new URL(ownerDocument.defaultView?.location.href ?? location.href)
  url.searchParams.set('hskify-surface', surface.id)
  return url.href
}

function captureOnlySurfaceIdentityUrl(element: Element, identity: string): string {
  const ownerDocument = element.ownerDocument
  const url = new URL(ownerDocument.defaultView?.location.href ?? location.href)
  url.searchParams.set('hskify-surface', identity)
  return url.href
}

function toCandidate(surface: PageSurface, domIndex: number): DiscoveredSurface | undefined {
  if (!isHtmlElement(surface.element)) return undefined
  // Preserve a real image URL when a same-origin frame exposes one. Canvas
  // and WebGL surfaces have no fetchable URL, so those retain a stable
  // same-origin identity and submit their captured pixels inline.
  const sourceUrl = surface.sourceUrl || surfaceIdentityUrl(surface)
  return Object.freeze({
    id: surface.id,
    kind: surface.kind,
    element: surface.element,
    owner: surface.element,
    sourceUrl,
    ...(surface.captureOnly || !surface.sourceUrl ? { captureOnly: true } : {}),
    sourceWidth: surface.width,
    sourceHeight: surface.height,
    domIndex,
    visible: surface.visible,
    capture: surface.capture,
  })
}

function dimensions(element: Element): { width: number; height: number } | undefined {
  if (isImageElement(element)) {
    return element.naturalWidth > 0 && element.naturalHeight > 0
      ? { width: element.naturalWidth, height: element.naturalHeight }
      : undefined
  }
  if (isCanvasElement(element)) {
    return element.width > 0 && element.height > 0
      ? { width: element.width, height: element.height }
      : undefined
  }
  const rect = element.getBoundingClientRect()
  return rect.width > 0 && rect.height > 0
    ? { width: Math.round(rect.width), height: Math.round(rect.height) }
    : undefined
}

function abortIfNeeded(signal?: AbortSignal): void {
  if (signal?.aborted) throw new DOMException('Surface capture cancelled.', 'AbortError')
}

async function fetchImage(
  sourceUrl: string,
  width: number,
  height: number,
  ownerDocument: Document,
  signal?: AbortSignal,
): Promise<SurfaceCapture | undefined> {
  abortIfNeeded(signal)
  let url: URL
  try {
    url = new URL(sourceUrl, ownerDocument.baseURI)
  } catch {
    return undefined
  }
  if (!['http:', 'https:', 'blob:', 'data:'].includes(url.protocol)) return undefined
  const ownerOrigin = ownerDocument.defaultView?.location.origin ?? location.origin
  if (url.origin !== ownerOrigin && !['blob:', 'data:'].includes(url.protocol)) return undefined
  try {
    const response = await fetch(url.href, {
      credentials: url.origin === ownerOrigin ? 'include' : 'omit',
      cache: 'no-store',
      ...(signal ? { signal } : {}),
    })
    if (!response.ok) return undefined
    const contentLength = response.headers.get('content-length')
    if (contentLength !== null) {
      const parsed = Number(contentLength)
      if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > DEFAULT_IMAGE_LIMITS.maximumBytes) {
        return undefined
      }
    }
    const bytes = response.body
      ? await readBoundedBody(response, DEFAULT_IMAGE_LIMITS.maximumBytes, signal)
      : await response.arrayBuffer()
    if (!bytes || bytes.byteLength > DEFAULT_IMAGE_LIMITS.maximumBytes) return undefined
    const contentType = normalizeMimeType(response.headers.get('content-type'))
    const mimeType =
      contentType === 'image/png' ||
      contentType === 'image/jpeg' ||
      contentType === 'image/webp' ||
      contentType === 'image/gif'
        ? contentType
        : undefined
    return { bytes, ...(mimeType ? { mimeType } : {}), width, height }
  } catch (error) {
    if (signal?.aborted) throw error
    return undefined
  }
}

async function readBoundedBody(
  response: Response,
  maximumBytes: number,
  signal?: AbortSignal,
): Promise<ArrayBuffer | undefined> {
  if (!response.body) return undefined
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    while (true) {
      abortIfNeeded(signal)
      const item = await reader.read()
      if (item.done) break
      total += item.value.byteLength
      if (total > maximumBytes) {
        await reader.cancel()
        return undefined
      }
      chunks.push(item.value)
    }
  } finally {
    reader.releaseLock()
  }
  const merged = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.byteLength
  }
  return merged.buffer
}

type BackgroundDrawPlan = Readonly<{
  width: number
  height: number
  offsetX: number
  offsetY: number
  repeatX: boolean
  repeatY: boolean
}>

function firstBackgroundLayer(value: string): string {
  // CSS background properties are comma-separated by layer. The page
  // adapter currently captures the first image layer, so use the matching
  // first size/position/repeat values consistently rather than mixing layers.
  return value.split(',')[0]?.trim() || ''
}

function cssLength(value: string, basis: number, intrinsic: number): number | undefined {
  const normalized = value.trim().toLowerCase()
  if (normalized === 'auto') return intrinsic
  if (normalized.endsWith('%')) {
    const percentage = Number.parseFloat(normalized.slice(0, -1))
    return Number.isFinite(percentage) ? basis * percentage / 100 : undefined
  }
  if (normalized.endsWith('px')) {
    const pixels = Number.parseFloat(normalized.slice(0, -2))
    return Number.isFinite(pixels) ? pixels : undefined
  }
  const pixels = Number.parseFloat(normalized)
  return Number.isFinite(pixels) ? pixels : undefined
}

function backgroundDrawPlan(
  style: Pick<CSSStyleDeclaration, 'backgroundSize' | 'backgroundPosition' | 'backgroundRepeat'>,
  targetWidth: number,
  targetHeight: number,
  intrinsicWidth: number,
  intrinsicHeight: number,
): BackgroundDrawPlan | undefined {
  if (
    targetWidth <= 0 ||
    targetHeight <= 0 ||
    intrinsicWidth <= 0 ||
    intrinsicHeight <= 0
  ) {
    return undefined
  }
  const size = firstBackgroundLayer(style.backgroundSize || 'auto')
  let width: number
  let height: number
  if (size === 'cover' || size === 'contain') {
    const scale =
      size === 'cover'
        ? Math.max(targetWidth / intrinsicWidth, targetHeight / intrinsicHeight)
        : Math.min(targetWidth / intrinsicWidth, targetHeight / intrinsicHeight)
    width = intrinsicWidth * scale
    height = intrinsicHeight * scale
  } else {
    const tokens = size.split(/\s+/u).filter(Boolean)
    const requestedWidth = cssLength(tokens[0] || 'auto', targetWidth, intrinsicWidth)
    const requestedHeight = cssLength(tokens[1] || 'auto', targetHeight, intrinsicHeight)
    if (requestedWidth === undefined || requestedHeight === undefined) return undefined
    if (tokens.length <= 1 || tokens[1] === 'auto') {
      width = requestedWidth
      height = width * intrinsicHeight / intrinsicWidth
    } else if (tokens[0] === 'auto') {
      height = requestedHeight
      width = height * intrinsicWidth / intrinsicHeight
    } else {
      width = requestedWidth
      height = requestedHeight
    }
  }
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return undefined
  }
  const remainingX = targetWidth - width
  const remainingY = targetHeight - height
  const positionToken = firstBackgroundLayer(style.backgroundPosition || '0% 0%')
    .split(/\s+/u)
    .filter(Boolean)
  const position = (token: string | undefined, remaining: number, axis: 'x' | 'y'): number => {
    const normalized = (token || (axis === 'x' ? '0%' : '0%')).toLowerCase()
    if (normalized === 'center') return remaining / 2
    if (axis === 'x' && normalized === 'right') return remaining
    if (axis === 'x' && normalized === 'left') return 0
    if (axis === 'y' && normalized === 'bottom') return remaining
    if (axis === 'y' && normalized === 'top') return 0
    const parsed = cssLength(normalized, remaining, 0)
    return parsed === undefined ? 0 : parsed
  }
  const repeat = firstBackgroundLayer(style.backgroundRepeat || 'repeat')
  const repeatTokens = repeat.split(/\s+/u).filter(Boolean)
  const repeatX = repeatTokens[0] !== 'no-repeat' && repeatTokens[0] !== 'repeat-y'
  const repeatY =
    (repeatTokens[1] || repeatTokens[0]) !== 'no-repeat' &&
    (repeatTokens[1] || repeatTokens[0]) !== 'repeat-x'
  return {
    width,
    height,
    offsetX: position(positionToken[0], remainingX, 'x'),
    offsetY: position(positionToken[1], remainingY, 'y'),
    repeatX,
    repeatY,
  }
}

async function renderedBackgroundCapture(
  element: Element,
  sourceUrl: string,
  width: number,
  height: number,
  ownerDocument: Document,
  signal?: AbortSignal,
): Promise<SurfaceCapture | undefined> {
  abortIfNeeded(signal)
  const raw = await fetchImage(sourceUrl, 1, 1, ownerDocument, signal)
  if (!raw) return undefined
  const style = ownerDocument.defaultView?.getComputedStyle(element) ?? getComputedStyle(element)
  if (typeof createImageBitmap !== 'function') return undefined
  let bitmap: ImageBitmap | undefined
  try {
    bitmap = await createImageBitmap(
      new Blob([raw.bytes], raw.mimeType ? { type: raw.mimeType } : undefined),
    )
    abortIfNeeded(signal)
    const plan = backgroundDrawPlan(
      style,
      width,
      height,
      bitmap.width,
      bitmap.height,
    )
    if (!plan) return undefined
    const canvas = ownerDocument.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d')
    if (!context) return undefined
    if (style.backgroundColor && style.backgroundColor !== 'transparent') {
      context.fillStyle = style.backgroundColor
      context.fillRect(0, 0, width, height)
    }
    const startX = plan.repeatX
      ? plan.offsetX - Math.ceil(plan.offsetX / plan.width) * plan.width
      : plan.offsetX
    const startY = plan.repeatY
      ? plan.offsetY - Math.ceil(plan.offsetY / plan.height) * plan.height
      : plan.offsetY
    const maxX = plan.repeatX ? width : startX + 1
    const maxY = plan.repeatY ? height : startY + 1
    for (let y = startY; y < maxY; y += plan.height) {
      for (let x = startX; x < maxX; x += plan.width) {
        context.drawImage(bitmap, x, y, plan.width, plan.height)
      }
    }
    const response = await fetch(canvas.toDataURL('image/png'), signal ? { signal } : undefined)
    if (!response.ok) return undefined
    const bytes = await response.arrayBuffer()
    return { bytes, mimeType: 'image/png', width, height }
  } catch (error) {
    if (signal?.aborted) throw error
    return undefined
  } finally {
    bitmap?.close()
  }
}

type ReadableWebGlContext = {
  RGBA: number
  UNSIGNED_BYTE: number
  readPixels(
    x: number,
    y: number,
    width: number,
    height: number,
    format: number,
    type: number,
    pixels: Uint8Array,
  ): void
}

function webGlPixels(canvas: HTMLCanvasElement, width: number, height: number): Uint8Array | undefined {
  let context: ReadableWebGlContext | null = null
  try {
    context =
      (canvas.getContext('webgl2') as ReadableWebGlContext | null) ??
      (canvas.getContext('webgl') as ReadableWebGlContext | null)
  } catch {
    return undefined
  }
  if (!context || typeof context.readPixels !== 'function') return undefined
  const pixels = new Uint8Array(width * height * 4)
  try {
    context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, pixels)
  } catch {
    return undefined
  }
  return pixels
}

function canvasCapture(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  ownerDocument: Document,
  kind: 'canvas' | 'webgl',
) {
  return async (signal?: AbortSignal): Promise<SurfaceCapture | undefined> => {
    abortIfNeeded(signal)
    try {
      let dataUrl: string
      if (kind === 'webgl') {
        const pixels = webGlPixels(canvas, width, height)
        if (pixels) {
          const restored = ownerDocument.createElement('canvas')
          restored.width = width
          restored.height = height
          const context = restored.getContext('2d')
          if (!context) return undefined
          const imageData = context.createImageData(width, height)
          // WebGL's origin is bottom-left while canvas/ImageData's origin is
          // top-left. Flip rows while copying so overlays use page order.
          const rowBytes = width * 4
          for (let row = 0; row < height; row += 1) {
            const source = (height - row - 1) * rowBytes
            imageData.data.set(pixels.subarray(source, source + rowBytes), row * rowBytes)
          }
          context.putImageData(imageData, 0, 0)
          dataUrl = restored.toDataURL('image/png')
        } else {
          // Some readers expose a WebGL context whose current framebuffer is
          // protected or has already been discarded. A normal canvas export
          // is still a valid, browser-owned capture when available.
          dataUrl = canvas.toDataURL('image/png')
        }
      } else {
        dataUrl = canvas.toDataURL('image/png')
      }
      const response = await fetch(dataUrl, signal ? { signal } : undefined)
      const bytes = await response.arrayBuffer()
      return { bytes, mimeType: 'image/png', width, height }
    } catch (error) {
      if (signal?.aborted) throw error
      return undefined
    }
  }
}

function imageCapture(image: HTMLImageElement, width: number, height: number) {
  return (signal?: AbortSignal) =>
    fetchImage(
      image.currentSrc || image.src || deferredSourceUrl(image, image.ownerDocument) || '',
      width,
      height,
      image.ownerDocument,
      signal,
    )
}

function deferredSourceUrl(
  image: HTMLImageElement,
  ownerDocument: Document = image.ownerDocument,
): string | undefined {
  for (const attribute of [...image.attributes]) {
    if (!/^data-(?:.*(?:src|url|image|original).*)$/i.test(attribute.name)) continue
    const value = attribute.value.trim()
    if (!value) continue
    try {
      const url = new URL(value, ownerDocument.baseURI)
      if (['http:', 'https:', 'blob:', 'data:'].includes(url.protocol)) return url.href
    } catch {
      // Ignore malformed lazy attributes and let the surface remain visible
      // as an unsupported/not-readable source.
    }
  }
  return undefined
}

function cssUrls(value: string): string[] {
  return [...value.matchAll(/url\((?:"([^"]+)"|'([^']+)'|([^)]*))\)/gi)]
    .map((match) => (match[1] || match[2] || match[3] || '').trim())
    .filter(Boolean)
}

function hasOnlyOneRenderableBackgroundImage(value: string): boolean {
  // A canvas reconstruction is deliberately limited to one URL image. CSS
  // gradients, image() functions, image-set(), and layered backgrounds carry
  // pixels that cannot be reproduced from the URL plus box geometry alone.
  // Reporting those surfaces as unreadable is safer than erasing text against
  // a subtly different background.
  if (cssUrls(value).length !== 1) return false
  const withoutUrl = value.replace(/url\((?:"[^"]*"|'[^']*'|[^)]*)\)/gi, '').trim()
  return withoutUrl === ''
}

function backgroundSurface(
  element: Element,
  pageIndex: number,
  ownerDocument: Document,
): PageSurface | undefined {
  const style = ownerDocument.defaultView?.getComputedStyle(element) ?? getComputedStyle(element)
  if (!hasOnlyOneRenderableBackgroundImage(style.backgroundImage || '')) return undefined
  const sourceUrl = cssUrls(style.backgroundImage)[0]
  const size = dimensions(element)
  if (!sourceUrl || !size) return undefined
  let resolvedUrl: URL
  try {
    resolvedUrl = new URL(sourceUrl, ownerDocument.baseURI)
  } catch {
    return undefined
  }
  if (
    ['http:', 'https:'].includes(resolvedUrl.protocol) &&
    resolvedUrl.origin !== (ownerDocument.defaultView?.location.origin ?? location.origin)
  ) {
    return undefined
  }
  const capture = (signal?: AbortSignal) =>
    renderedBackgroundCapture(
      element,
      resolvedUrl.href,
      size.width,
      size.height,
      ownerDocument,
      signal,
    )
  return Object.freeze({
    id: `background:${pageIndex}:${sourceUrl}`,
    kind: 'background' as const,
    element,
    pageIndex,
    sourceUrl: resolvedUrl.href,
    width: size.width,
    height: size.height,
    rect: rectOf(element),
    visible: visibleInViewport(element, ownerDocument),
    continuous: size.height >= size.width * 2.5,
    capture,
  })
}

function imageSurface(
  image: HTMLImageElement,
  pageIndex: number,
  ownerDocument: Document,
): PageSurface | undefined {
  // A lazy placeholder is a discovery signal, not page pixels.  The image
  // adapter will rescan it after its real source is loaded; submitting the
  // placeholder here would create a terminal job for the wrong bytes.
  if (
    !image.complete ||
    image.naturalWidth < MIN_IMAGE_WIDTH ||
    image.naturalHeight < MIN_IMAGE_HEIGHT
  ) {
    return undefined
  }
  const deferred = deferredSourceUrl(image, ownerDocument)
  const size = dimensions(image)
  if (!size) return undefined
  const sourceUrl = image.currentSrc || image.src || deferred
  return Object.freeze({
    id: `image:${pageIndex}:${sourceUrl || 'deferred'}`,
    kind: 'image' as const,
    element: image,
    pageIndex,
    ...(sourceUrl ? { sourceUrl } : {}),
    width: size.width,
    height: size.height,
    rect: rectOf(image),
    visible: visibleInViewport(image, ownerDocument),
    continuous: size.height >= size.width * 2.5,
    capture: imageCapture(image, size.width, size.height),
  })
}

function canvasSurface(
  canvas: HTMLCanvasElement,
  pageIndex: number,
  ownerDocument: Document,
): PageSurface | undefined {
  const size = dimensions(canvas)
  if (!size) return undefined
  let kind: 'canvas' | 'webgl' = 'canvas'
  try {
    if (canvas.getContext('webgl2') || canvas.getContext('webgl')) kind = 'webgl'
  } catch {
    // A tainted/protected context is still reported through capture failure.
  }
  return Object.freeze({
    id: `canvas:${pageIndex}:${size.width}x${size.height}`,
    kind,
    element: canvas,
    pageIndex,
    width: size.width,
    height: size.height,
    rect: rectOf(canvas),
    visible: visibleInViewport(canvas, ownerDocument),
    continuous: size.height >= size.width * 2.5,
    capture: canvasCapture(canvas, size.width, size.height, ownerDocument, kind),
  })
}

function sameOriginFrame(frame: HTMLIFrameElement): Document | 'cross-origin' | undefined {
  try {
    return frame.contentDocument ?? undefined
  } catch {
    return 'cross-origin'
  }
}

/** Discover all publicly rendered surfaces in document order. */
export function discoverPageSurfaces(root: Document = document): SurfaceDiscovery {
  const surfaces: PageSurface[] = []
  const unsupported: UnsupportedSurface[] = []
  let pageIndex = 0
  const add = (surface: PageSurface | undefined): void => {
    if (surface) surfaces.push(surface)
  }
  const seen = new Set<Element>()
  for (const element of root.querySelectorAll('img,canvas,*')) {
    if (seen.has(element)) continue
    seen.add(element)
    if (isImageElement(element)) {
      add(imageSurface(element, pageIndex++, root))
    } else if (isCanvasElement(element)) {
      add(canvasSurface(element, pageIndex++, root))
    } else if (isHtmlElement(element)) {
      const style = root.defaultView?.getComputedStyle(element) ?? getComputedStyle(element)
      if (style.backgroundImage === 'none' || !style.backgroundImage) continue
      const surface = backgroundSurface(element, pageIndex++, root)
      if (surface) {
        add(surface)
      } else {
        unsupported.push({ kind: 'background', element, reason: 'not-readable' })
      }
    }
  }
  for (const frame of root.querySelectorAll('iframe')) {
    const child = sameOriginFrame(frame)
    if (child === 'cross-origin') {
      unsupported.push({ kind: 'frame', element: frame, reason: 'cross-origin' })
      continue
    }
    if (!child) {
      unsupported.push({ kind: 'frame', element: frame, reason: 'not-readable' })
      continue
    }
    const frameVisible = visibleInViewport(frame, root)
    const nested = discoverPageSurfaces(child)
    surfaces.push(
      ...nested.surfaces.map((surface) => {
        const nestedRect = globalRect(surface.element, root)
        // The pixels come from an accessible child document. Keep the nested
        // element for geometry/capture, but express its rect/visibility in
        // the top reader's coordinate space so scrolling an iframe cannot
        // report an off-screen page as visible.
        return Object.freeze({
          ...surface,
          id: `frame:${frame.src || nestedRect.x + ':' + nestedRect.y}:${surface.id}`,
          kind: 'frame' as const,
          pageIndex: pageIndex++,
          rect: Object.freeze(nestedRect),
          visible:
            frameVisible &&
            rectVisibleInViewport(
              {
                top: nestedRect.y,
                right: nestedRect.x + nestedRect.width,
                bottom: nestedRect.y + nestedRect.height,
                left: nestedRect.x,
              },
              root,
            ),
        })
      }),
    )
    unsupported.push(...nested.unsupported)
  }
  return Object.freeze({ surfaces: Object.freeze(surfaces), unsupported: Object.freeze(unsupported) })
}

/**
 * Live discovery for non-`<img>` reader surfaces.  The image adapter keeps
 * its specialised lazy-load checks; this adapter owns every other publicly
 * rendered surface and emits the same candidate/event lifecycle.  A full
 * rescan is intentional: readers replace canvases and background hosts as
 * navigation advances, and comparing immutable identities is safer than
 * trying to infer publisher-specific mutation semantics.
 */
export class LiveSurfaceDiscovery {
  private readonly candidates = new Map<string, DiscoveredSurface>()
  private readonly candidateIdsByElement = new Map<Element, string>()
  // Reader DOMs insert/remove pages while scrolling.  Surface order is not an
  // identity: bind it to the actual element so a lazy page inserted before an
  // existing canvas/background/frame cannot create a second translation job.
  private readonly identities = new WeakMap<Element, string>()
  private nextIdentity = 0
  private unsupportedSurfaces: readonly UnsupportedSurface[] = []
  private readonly mutationObservers = new Map<Document, MutationObserver>()
  private intersectionObserver: IntersectionObserver | undefined
  private readonly viewportDocuments = new Set<Document>()
  private viewportListener: (() => void) | undefined
  private loadListener: (() => void) | undefined
  private scanScheduled = false

  constructor(
    private readonly onEvent: (event: SurfaceDiscoveryEvent) => void,
    private readonly root: Document = document,
  ) {}

  start(): void {
    this.viewportListener = () => this.refreshVisibility()
    this.loadListener = () => this.scheduleScan()
    this.observeDocument(this.root)
    this.scan()
    this.intersectionObserver =
      typeof IntersectionObserver === 'undefined'
        ? undefined
        : new IntersectionObserver((entries) => {
            for (const entry of entries) {
              const id = this.candidateIdsByElement.get(entry.target)
              const candidate = id ? this.candidates.get(id) : undefined
              if (!candidate) continue
              // IntersectionObserver is scoped to the reader document.  A
              // candidate discovered inside a same-origin frame is owned by a
              // different document, so its callback entry cannot be used as a
              // top-level viewport signal.  Recompute visibility from the
              // candidate's global transformed rectangle instead.
              const rect = globalRect(candidate.element, this.root)
              const visible = rectVisibleInViewport(
                {
                  top: rect.y,
                  right: rect.x + rect.width,
                  bottom: rect.y + rect.height,
                  left: rect.x,
                },
                this.root,
              )
              if (visible === candidate.visible) continue
              const next = Object.freeze({ ...candidate, visible })
              this.candidates.set(candidate.id, next)
              this.onEvent({ type: 'visibility', candidate: next })
            }
    })
    for (const candidate of this.candidates.values()) {
      if (candidate.element.ownerDocument === this.root) {
        this.intersectionObserver?.observe(candidate.element)
      }
    }
  }

  stop(): void {
    for (const observer of this.mutationObservers.values()) observer.disconnect()
    this.mutationObservers.clear()
    this.intersectionObserver?.disconnect()
    if (this.viewportListener) {
      for (const ownerDocument of this.viewportDocuments) {
        ownerDocument.removeEventListener('scroll', this.viewportListener, true)
        ownerDocument.defaultView?.removeEventListener('resize', this.viewportListener)
        if (this.loadListener) ownerDocument.removeEventListener('load', this.loadListener, true)
      }
    }
    this.viewportDocuments.clear()
    this.viewportListener = undefined
    this.loadListener = undefined
    this.intersectionObserver = undefined
    this.scanScheduled = false
    this.candidates.clear()
    this.candidateIdsByElement.clear()
    this.unsupportedSurfaces = []
  }

  private refreshVisibility(): void {
    for (const candidate of this.candidates.values()) {
      const rect = globalRect(candidate.element, this.root)
      const visible = rectVisibleInViewport(
        {
          top: rect.y,
          right: rect.x + rect.width,
          bottom: rect.y + rect.height,
          left: rect.x,
        },
        this.root,
      )
      if (visible === candidate.visible) continue
      const next = Object.freeze({ ...candidate, visible })
      this.candidates.set(candidate.id, next)
      this.onEvent({ type: 'visibility', candidate: next })
    }
  }

  current(): DiscoveredSurface[] {
    return [...this.candidates.values()].sort(
      (left, right) => Number(right.visible) - Number(left.visible) || left.domIndex - right.domIndex,
    )
  }

  unsupported(): readonly UnsupportedSurface[] {
    return this.unsupportedSurfaces
  }

  completionKey(): string {
    return this.current()
      .map(
        (candidate) =>
          `${candidate.id}:${candidate.sourceUrl}:${candidate.sourceWidth}x${candidate.sourceHeight}`,
      )
      .sort()
      .join('|')
  }

  private scan(): void {
    const discoveredResult = discoverPageSurfaces(this.root)
    this.unsupportedSurfaces = discoveredResult.unsupported
    const discovered = discoveredResult.surfaces
      .map((surface, domIndex) => toCandidate(surface, domIndex))
      .filter((candidate): candidate is DiscoveredSurface => candidate !== undefined)
      .map((candidate) => {
        let identity = this.identities.get(candidate.element)
        if (!identity) {
          identity = `surface-${this.nextIdentity++}`
          this.identities.set(candidate.element, identity)
        }
        const sourceUrl =
          candidate.captureOnly === true || candidate.kind === 'canvas' || candidate.kind === 'webgl'
            ? captureOnlySurfaceIdentityUrl(candidate.element, identity)
            : candidate.sourceUrl
        return identity === candidate.id && sourceUrl === candidate.sourceUrl
          ? candidate
          : { ...candidate, id: identity, sourceUrl }
      })
    const live = new Map(discovered.map((candidate) => [candidate.id, candidate]))
    if (this.viewportListener) {
      for (const candidate of discovered) {
        const ownerDocument = candidate.element.ownerDocument
        this.observeDocument(ownerDocument)
      }
    }
    for (const [id, previous] of this.candidates) {
      if (live.has(id)) continue
      this.candidates.delete(id)
      this.candidateIdsByElement.delete(previous.element)
      this.intersectionObserver?.unobserve(previous.element)
      this.onEvent({ type: 'removed', candidate: previous })
    }
    for (const candidate of discovered) {
      const previous = this.candidates.get(candidate.id)
      if (!previous) {
        this.candidates.set(candidate.id, candidate)
        this.candidateIdsByElement.set(candidate.element, candidate.id)
        if (candidate.element.ownerDocument === this.root) {
          this.intersectionObserver?.observe(candidate.element)
        }
        this.onEvent({ type: 'added', candidate })
        continue
      }
      const changed =
        previous.sourceUrl !== candidate.sourceUrl ||
        previous.sourceWidth !== candidate.sourceWidth ||
        previous.sourceHeight !== candidate.sourceHeight ||
        previous.element !== candidate.element
      if (previous.element !== candidate.element) {
        this.candidateIdsByElement.delete(previous.element)
        this.intersectionObserver?.unobserve(previous.element)
      }
      const next = Object.freeze({ ...candidate })
      this.candidates.set(candidate.id, next)
      this.candidateIdsByElement.set(candidate.element, candidate.id)
      if (previous.visible !== candidate.visible) {
        this.onEvent({ type: 'visibility', candidate: next })
      }
      if (changed || previous.domIndex !== candidate.domIndex) {
        this.onEvent({
          type: 'updated',
          candidate: next,
          previousSourceUrl: previous.sourceUrl,
          previousDomIndex: previous.domIndex,
        })
      }
    }
  }

  /** Collapse mutation bursts into one authoritative reader scan per turn. */
  private scheduleScan(): void {
    if (this.scanScheduled || this.mutationObservers.size === 0) return
    this.scanScheduled = true
    queueMicrotask(() => {
      this.scanScheduled = false
      if (this.mutationObservers.size > 0) this.scan()
    })
  }

  /**
   * Observe every accessible document that contributes a surface.  A root
   * MutationObserver cannot see pages inserted inside an iframe, so keeping
   * one observer per permitted document is part of the acquisition contract,
   * not a reader-specific workaround.
   */
  private observeDocument(ownerDocument: Document): void {
    if (this.viewportListener && !this.viewportDocuments.has(ownerDocument)) {
      this.viewportDocuments.add(ownerDocument)
      ownerDocument.addEventListener('scroll', this.viewportListener, true)
      ownerDocument.defaultView?.addEventListener('resize', this.viewportListener)
      if (this.loadListener) ownerDocument.addEventListener('load', this.loadListener, true)
    }
    if (this.mutationObservers.has(ownerDocument) || !ownerDocument.documentElement) return
    const observer = new MutationObserver(() => this.scheduleScan())
    observer.observe(ownerDocument.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'sizes', 'data-src', 'data-url', 'style', 'class'],
    })
    this.mutationObservers.set(ownerDocument, observer)
  }
}

export function visibleFirst(candidates: readonly DiscoveredSurface[]): DiscoveredSurface[] {
  return [...candidates].sort(
    (left, right) => Number(right.visible) - Number(left.visible) || left.domIndex - right.domIndex,
  )
}
