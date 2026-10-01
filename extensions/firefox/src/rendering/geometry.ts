import type { NormalizedRect, Point } from '../contracts/browser'

export type Rect = {
  left: number
  top: number
  width: number
  height: number
}

export type ImageGeometry = {
  viewport: Rect
  image: Rect
}

export type LocalImageBox = Readonly<{ width: number; height: number }>

export type Bounds = {
  minX: number
  minY: number
  maxX: number
  maxY: number
  width: number
  height: number
}

function finiteCssPixels(value: string): number {
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : 0
}

export function polygonBounds(points: readonly Point[]): Bounds {
  if (points.length === 0) {
    return { minX: 0, minY: 0, maxX: 0, maxY: 0, width: 0, height: 0 }
  }
  let minX = 1
  let minY = 1
  let maxX = 0
  let maxY = 0
  for (const point of points) {
    minX = Math.min(minX, point.x)
    minY = Math.min(minY, point.y)
    maxX = Math.max(maxX, point.x)
    maxY = Math.max(maxY, point.y)
  }
  return {
    minX,
    minY,
    maxX,
    maxY,
    width: Math.max(0, maxX - minX),
    height: Math.max(0, maxY - minY),
  }
}

function objectPositionTokens(value: string): [string | undefined, string | undefined] {
  const tokens = value.toLowerCase().match(/calc\([^)]*\)|[^\s]+/gu) ?? []
  if (tokens.length <= 2) {
    if (tokens.length === 1)
      return /^(top|bottom)$/u.test(tokens[0]!) ? ['center', tokens[0]] : [tokens[0], 'center']
    if (/^(top|bottom)$/u.test(tokens[0] ?? '') && /^(left|right|center)$/u.test(tokens[1] ?? ''))
      return [tokens[1], tokens[0]]
    return [tokens[0], tokens[1]]
  }
  let horizontal: string | undefined, vertical: string | undefined
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index]!
    const isX = token === 'left' || token === 'right'
    const isY = token === 'top' || token === 'bottom'
    if (!isX && !isY && token !== 'center') throw new RangeError('Unsupported object-position')
    let position = token
    const next = tokens[index + 1]
    if ((isX || isY) && next && /^(?:[-+]?\d|calc\()/u.test(next)) {
      index++
      position = token === 'right' || token === 'bottom' ? `calc(100% - ${next})` : next
    }
    if (isX || (token === 'center' && horizontal === undefined)) horizontal = position
    else vertical = position
  }
  return [horizontal, vertical]
}

function positionOffset(
  available: number,
  token: string | undefined,
  _horizontal: boolean,
): number {
  if (!token || token === 'center') return available / 2
  if (token === 'left' || token === 'top') return 0
  if (token === 'right' || token === 'bottom') return available
  let expression = token.replace(/^calc\((.*)\)$/u, '$1').replace(/\s/gu, '')
  // Computed length-percentage values are linear combinations of percentages and pixels.
  if (!/^[+-]?\d*\.?\d+(?:px|%)(?:[+-]\d*\.?\d+(?:px|%))*$/u.test(expression))
    throw new RangeError('Unsupported object-position')
  let offset = 0
  for (const term of expression.matchAll(/([+-]?\d*\.?\d+)(px|%)/gu)) {
    offset += Number(term[1]) * (term[2] === '%' ? available / 100 : 1)
  }
  return offset
}

export function objectFitRect(
  containerWidth: number,
  containerHeight: number,
  sourceWidth: number,
  sourceHeight: number,
  objectFit: string,
  objectPosition: string,
): Rect {
  const safeSourceWidth = Math.max(1, sourceWidth)
  const safeSourceHeight = Math.max(1, sourceHeight)
  const containScale = Math.min(
    containerWidth / safeSourceWidth,
    containerHeight / safeSourceHeight,
  )
  const coverScale = Math.max(containerWidth / safeSourceWidth, containerHeight / safeSourceHeight)
  let width = containerWidth
  let height = containerHeight
  switch (objectFit) {
    case 'contain':
      width = safeSourceWidth * containScale
      height = safeSourceHeight * containScale
      break
    case 'cover':
      width = safeSourceWidth * coverScale
      height = safeSourceHeight * coverScale
      break
    case 'none':
      width = safeSourceWidth
      height = safeSourceHeight
      break
    case 'scale-down': {
      const scale = Math.min(1, containScale)
      width = safeSourceWidth * scale
      height = safeSourceHeight * scale
      break
    }
    case 'fill':
    default:
      break
  }
  const [horizontal, vertical] = objectPositionTokens(objectPosition)
  return {
    left: positionOffset(containerWidth - width, horizontal, true),
    top: positionOffset(containerHeight - height, vertical, false),
    width,
    height,
  }
}

export function calculateImageGeometry(
  image: Element,
  wrapper: HTMLElement,
  sourceWidth: number,
  sourceHeight: number,
  localBox?: LocalImageBox,
  renderedBox = false,
): ImageGeometry {
  const imageRect = image.getBoundingClientRect()
  const wrapperRect = wrapper.getBoundingClientRect()
  const ownerWindow = image.ownerDocument.defaultView
  const style = ownerWindow?.getComputedStyle(image) ?? getComputedStyle(image)
  if (renderedBox)
    return {
      viewport: {
        left: imageRect.left - wrapperRect.left,
        top: imageRect.top - wrapperRect.top,
        width: imageRect.width,
        height: imageRect.height,
      },
      image: { left: 0, top: 0, width: imageRect.width, height: imageRect.height },
    }
  const borderLeft = finiteCssPixels(style.borderLeftWidth)
  const borderRight = finiteCssPixels(style.borderRightWidth)
  const borderTop = finiteCssPixels(style.borderTopWidth)
  const borderBottom = finiteCssPixels(style.borderBottomWidth)
  const paddingLeft = finiteCssPixels(style.paddingLeft)
  const paddingRight = finiteCssPixels(style.paddingRight)
  const paddingTop = finiteCssPixels(style.paddingTop)
  const paddingBottom = finiteCssPixels(style.paddingBottom)
  const viewport: Rect = {
    left: localBox
      ? borderLeft + paddingLeft
      : imageRect.left - wrapperRect.left + borderLeft + paddingLeft,
    top: localBox
      ? borderTop + paddingTop
      : imageRect.top - wrapperRect.top + borderTop + paddingTop,
    width: Math.max(
      0,
      (localBox?.width ?? imageRect.width) - borderLeft - borderRight - paddingLeft - paddingRight,
    ),
    height: Math.max(
      0,
      (localBox?.height ?? imageRect.height) -
        borderTop -
        borderBottom -
        paddingTop -
        paddingBottom,
    ),
  }
  const fitted = objectFitRect(
    viewport.width,
    viewport.height,
    sourceWidth,
    sourceHeight,
    style.objectFit || 'fill',
    style.objectPosition || '50% 50%',
  )
  return {
    viewport,
    image: {
      left: fitted.left,
      top: fitted.top,
      width: fitted.width,
      height: fitted.height,
    },
  }
}

function intersection(left: Rect, right: Rect): Rect | undefined {
  const x = Math.max(left.left, right.left)
  const y = Math.max(left.top, right.top)
  const rightEdge = Math.min(left.left + left.width, right.left + right.width)
  const bottomEdge = Math.min(left.top + left.height, right.top + right.height)
  if (rightEdge <= x || bottomEdge <= y) return undefined
  return {
    left: x,
    top: y,
    width: rightEdge - x,
    height: bottomEdge - y,
  }
}

function clampUnit(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function hasVisualTransform(element: Element): boolean {
  const ownerWindow = element.ownerDocument.defaultView
  for (let current: Element | null = element; current; current = current.parentElement) {
    const style = ownerWindow?.getComputedStyle(current)
    if (style?.transform && style.transform !== 'none') return true
    if (style?.perspective && style.perspective !== 'none') return true
  }
  return false
}

/**
 * Maps the currently visible portion of an <img> back into normalized source
 * coordinates. The rectangle accounts for borders, padding, object-fit,
 * object-position, cover cropping, and the browser viewport.
 */
export function visibleImageRects(
  image: Element,
  sourceWidth = image.tagName.toLowerCase() === 'img' && 'naturalWidth' in image
    ? Number((image as HTMLImageElement).naturalWidth)
    : Math.round(image.getBoundingClientRect().width),
  sourceHeight = image.tagName.toLowerCase() === 'img' && 'naturalHeight' in image
    ? Number((image as HTMLImageElement).naturalHeight)
    : Math.round(image.getBoundingClientRect().height),
): NormalizedRect[] {
  const ownerDocument = image.ownerDocument
  const ownerWindow = ownerDocument.defaultView
  if (
    !image.isConnected ||
    sourceWidth <= 0 ||
    sourceHeight <= 0 ||
    ownerDocument.visibilityState === 'hidden'
  ) {
    return []
  }
  const rect = image.getBoundingClientRect()
  const style = ownerWindow?.getComputedStyle(image) ?? getComputedStyle(image)
  // The normalized viewport contract is axis-aligned. A transformed surface
  // is rendered through the affine wrapper, but projecting a rotated quad to
  // a rectangle here would under-report visible text and starve the daemon's
  // visible-first scheduler. The bounding box is safe for priority: submit
  // the full source whenever any transformed pixels intersect the viewport.
  if (hasVisualTransform(image)) {
    const viewportWidth = Math.max(
      0,
      ownerWindow?.innerWidth || ownerDocument.documentElement.clientWidth,
    )
    const viewportHeight = Math.max(
      0,
      ownerWindow?.innerHeight || ownerDocument.documentElement.clientHeight,
    )
    if (
      rect.right <= 0 ||
      rect.bottom <= 0 ||
      rect.left >= viewportWidth ||
      rect.top >= viewportHeight
    ) {
      return []
    }
    return [{ x: 0, y: 0, width: 1, height: 1 }]
  }
  const borderLeft = finiteCssPixels(style.borderLeftWidth)
  const borderRight = finiteCssPixels(style.borderRightWidth)
  const borderTop = finiteCssPixels(style.borderTopWidth)
  const borderBottom = finiteCssPixels(style.borderBottomWidth)
  const paddingLeft = finiteCssPixels(style.paddingLeft)
  const paddingRight = finiteCssPixels(style.paddingRight)
  const paddingTop = finiteCssPixels(style.paddingTop)
  const paddingBottom = finiteCssPixels(style.paddingBottom)
  const content: Rect = {
    left: rect.left + borderLeft + paddingLeft,
    top: rect.top + borderTop + paddingTop,
    width: Math.max(0, rect.width - borderLeft - borderRight - paddingLeft - paddingRight),
    height: Math.max(0, rect.height - borderTop - borderBottom - paddingTop - paddingBottom),
  }
  if (content.width <= 0 || content.height <= 0) return []
  const fitted = objectFitRect(
    content.width,
    content.height,
    sourceWidth,
    sourceHeight,
    style.objectFit || 'fill',
    style.objectPosition || '50% 50%',
  )
  const drawn: Rect = {
    left: content.left + fitted.left,
    top: content.top + fitted.top,
    width: fitted.width,
    height: fitted.height,
  }
  if (drawn.width <= 0 || drawn.height <= 0) return []
  const browserViewport: Rect = {
    left: 0,
    top: 0,
    width: Math.max(0, ownerWindow?.innerWidth || ownerDocument.documentElement.clientWidth),
    height: Math.max(0, ownerWindow?.innerHeight || ownerDocument.documentElement.clientHeight),
  }
  const clippedToImage = intersection(drawn, content)
  const visible = clippedToImage && intersection(clippedToImage, browserViewport)
  if (!visible) return []
  const x = clampUnit((visible.left - drawn.left) / drawn.width)
  const y = clampUnit((visible.top - drawn.top) / drawn.height)
  const width = clampUnit(visible.width / drawn.width)
  const height = clampUnit(visible.height / drawn.height)
  if (width <= 0 || height <= 0) return []
  const containedWidth = Math.min(width, 1 - x)
  const containedHeight = Math.min(height, 1 - y)
  if (containedWidth <= 0 || containedHeight <= 0) return []
  return [
    {
      x,
      y,
      width: containedWidth,
      height: containedHeight,
    },
  ]
}

export function rectDifference(left: DOMRect, right: DOMRect): number {
  return Math.max(
    Math.abs(left.left - right.left),
    Math.abs(left.top - right.top),
    Math.abs(left.width - right.width),
    Math.abs(left.height - right.height),
  )
}

export function percentPolygon(
  points: readonly Point[],
  relativeBounds: Bounds = polygonBounds(points),
): string {
  const width = Math.max(Number.EPSILON, relativeBounds.width)
  const height = Math.max(Number.EPSILON, relativeBounds.height)
  return points
    .map(
      (point) =>
        `${((point.x - relativeBounds.minX) / width) * 100}% ${((point.y - relativeBounds.minY) / height) * 100}%`,
    )
    .join(', ')
}
