import type { Point } from '../contracts/browser'

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

function positionToken(token: string | undefined, horizontal: boolean): number | string {
  if (!token) return 0.5
  const normalized = token.toLowerCase()
  if (normalized === 'center') return 0.5
  if (normalized === 'left' || normalized === 'top') return 0
  if (normalized === 'right' || normalized === 'bottom') return 1
  if (normalized.endsWith('%')) {
    const percentage = Number.parseFloat(normalized)
    return Number.isFinite(percentage) ? percentage / 100 : 0.5
  }
  if (normalized.endsWith('px')) return normalized
  // A vertical keyword in the horizontal slot (or vice versa) is a valid CSS
  // reordering case. Fall back to center rather than guessing an edge.
  if (
    (horizontal && (normalized === 'top' || normalized === 'bottom')) ||
    (!horizontal && (normalized === 'left' || normalized === 'right'))
  ) {
    return 0.5
  }
  return 0.5
}

function objectPositionTokens(value: string): [string | undefined, string | undefined] {
  const tokens = value.trim().split(/\s+/).filter(Boolean)
  if (tokens.length === 0) return [undefined, undefined]
  if (tokens.length === 1) {
    const only = tokens[0]
    if (only === 'top' || only === 'bottom') return ['center', only]
    return [only, 'center']
  }
  const first = tokens[0]
  const second = tokens[1]
  if (
    (first === 'top' || first === 'bottom') &&
    (second === 'left' || second === 'right')
  ) {
    return [second, first]
  }
  return [first, second]
}

function positionOffset(
  available: number,
  token: string | undefined,
  horizontal: boolean,
): number {
  const parsed = positionToken(token, horizontal)
  if (typeof parsed === 'number') return available * parsed
  const pixels = Number.parseFloat(parsed)
  return Number.isFinite(pixels) ? pixels : available / 2
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
  const coverScale = Math.max(
    containerWidth / safeSourceWidth,
    containerHeight / safeSourceHeight,
  )
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
  image: HTMLImageElement,
  wrapper: HTMLElement,
  sourceWidth: number,
  sourceHeight: number,
): ImageGeometry {
  const imageRect = image.getBoundingClientRect()
  const wrapperRect = wrapper.getBoundingClientRect()
  const style = getComputedStyle(image)
  const borderLeft = finiteCssPixels(style.borderLeftWidth)
  const borderRight = finiteCssPixels(style.borderRightWidth)
  const borderTop = finiteCssPixels(style.borderTopWidth)
  const borderBottom = finiteCssPixels(style.borderBottomWidth)
  const paddingLeft = finiteCssPixels(style.paddingLeft)
  const paddingRight = finiteCssPixels(style.paddingRight)
  const paddingTop = finiteCssPixels(style.paddingTop)
  const paddingBottom = finiteCssPixels(style.paddingBottom)
  const viewport: Rect = {
    left: imageRect.left - wrapperRect.left + borderLeft + paddingLeft,
    top: imageRect.top - wrapperRect.top + borderTop + paddingTop,
    width: Math.max(
      0,
      imageRect.width - borderLeft - borderRight - paddingLeft - paddingRight,
    ),
    height: Math.max(
      0,
      imageRect.height - borderTop - borderBottom - paddingTop - paddingBottom,
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
