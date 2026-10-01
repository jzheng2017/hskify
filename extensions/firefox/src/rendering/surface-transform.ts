import type { LocalImageBox } from './geometry'

export type SurfaceTransform = Readonly<{
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

function cssMatrix(
  value: string,
): Readonly<{ a: number; b: number; c: number; d: number; e: number; f: number }> | undefined {
  const normalized = value.trim()
  if (!normalized || normalized === 'none') return undefined
  const matrix = normalized.match(/^matrix\(([^)]+)\)$/i)
  if (matrix) {
    const values = matrix[1]!.split(',').map((entry) => Number.parseFloat(entry.trim()))
    if (values.length !== 6 || values.some((entry) => !Number.isFinite(entry))) return undefined
    return {
      a: values[0]!,
      b: values[1]!,
      c: values[2]!,
      d: values[3]!,
      e: values[4]!,
      f: values[5]!,
    }
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
  return {
    a: values[0]!,
    b: values[1]!,
    c: values[4]!,
    d: values[5]!,
    e: values[12]!,
    f: values[13]!,
  }
}

function isIdentity(matrix: {
  a: number
  b: number
  c: number
  d: number
  e: number
  f: number
}): boolean {
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
      return Number.isFinite(percent) ? (basis * percent) / 100 : basis / 2
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

export function measureSurfaceTransform(element: Element): SurfaceTransformResult {
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
