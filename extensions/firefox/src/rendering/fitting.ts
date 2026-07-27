import type { BrowserRegion, Point } from '../contracts/browser'
import { polygonBounds, type Bounds } from './geometry'

const CLOSING_PUNCTUATION = new Set([...'，。！？；：、）》】」』”’…'])
const OPENING_PUNCTUATION = new Set([...'（《【「『“‘'])
const FIT_CONTENT_RATIO = 0.88
const MINIMUM_FONT_TO_IMAGE_WIDTH = 0.006
const ABSOLUTE_MINIMUM_FONT_PX = 1
const BINARY_SEARCH_STEPS = 20

export type TextFit = {
  fontSize: number
  lines: string[]
  degraded: boolean
  usedPolygon: boolean
}

export type FitBox = Bounds & {
  pixelWidth: number
  pixelHeight: number
}

function charUnits(character: string): number {
  if (/\s/u.test(character)) return 0.3
  if (/[\u3400-\u9fff\uf900-\ufaff]/u.test(character)) return 1
  if (/[A-Za-z0-9]/u.test(character)) return 0.58
  return 0.72
}

export function estimatedLineUnits(text: string): number {
  return [...text].reduce((total, character) => total + charUnits(character), 0)
}

export function isLegalLineBreak(left: string, right: string): boolean {
  const previous = [...left].at(-1)
  const next = [...right][0]
  return !(
    (previous !== undefined && /\s/u.test(previous)) ||
    (next !== undefined && /\s/u.test(next)) ||
    (previous !== undefined && OPENING_PUNCTUATION.has(previous)) ||
    (next !== undefined && CLOSING_PUNCTUATION.has(next))
  )
}

function breakAtIndices(text: string, indices: readonly number[]): string[] {
  const characters = [...text]
  const lines: string[] = []
  let start = 0
  for (const index of indices) {
    lines.push(characters.slice(start, index).join(''))
    start = index
  }
  lines.push(characters.slice(start).join(''))
  return lines.filter((line) => line.length > 0)
}

export function nearbyLineCandidates(
  text: string,
  suggested: readonly string[],
  maximumLines = 12,
): string[][] {
  const characters = [...text]
  const candidates: string[][] = []
  const seen = new Set<string>()
  const add = (lines: string[]): void => {
    if (lines.join('') !== text || lines.some((line) => line.length === 0)) return
    if (
      lines.some(
        (line, index) =>
          index < lines.length - 1 &&
          !isLegalLineBreak(line, lines[index + 1] ?? ''),
      )
    ) {
      return
    }
    const key = lines.join('\u0000')
    if (!seen.has(key)) {
      seen.add(key)
      candidates.push(lines)
    }
  }
  if (suggested.length > 0) add([...suggested])
  add([text])
  for (let lineCount = 2; lineCount <= Math.min(maximumLines, characters.length); lineCount += 1) {
    const base = characters.length / lineCount
    const indices: number[] = []
    for (let line = 1; line < lineCount; line += 1) {
      indices.push(Math.round(base * line))
    }
    for (let delta = -2; delta <= 2; delta += 1) {
      add(
        breakAtIndices(
          text,
          indices.map((index, position) =>
            Math.max(position + 1, Math.min(characters.length - 1, index + delta)),
          ),
        ),
      )
    }
  }
  return candidates
}

export function horizontalPolygonSpan(points: readonly Point[], y: number): number {
  const intersections: number[] = []
  for (let index = 0; index < points.length; index += 1) {
    const first = points[index]
    const second = points[(index + 1) % points.length]
    if (!first || !second || first.y === second.y) continue
    const minimum = Math.min(first.y, second.y)
    const maximum = Math.max(first.y, second.y)
    if (y < minimum || y >= maximum) continue
    const ratio = (y - first.y) / (second.y - first.y)
    intersections.push(first.x + ratio * (second.x - first.x))
  }
  intersections.sort((left, right) => left - right)
  let span = 0
  for (let index = 0; index + 1 < intersections.length; index += 2) {
    // A line cannot jump across a hole or excluded polygon area. Use the
    // widest contiguous interval instead of summing disjoint spans.
    span = Math.max(
      span,
      (intersections[index + 1] ?? 0) - (intersections[index] ?? 0),
    )
  }
  return Math.max(0, span)
}

function samePolygon(left: readonly Point[], right: readonly Point[]): boolean {
  return (
    left.length === right.length &&
    left.every(
      (point, index) =>
        Math.abs(point.x - (right[index]?.x ?? Number.NaN)) < 1e-6 &&
        Math.abs(point.y - (right[index]?.y ?? Number.NaN)) < 1e-6,
    )
  )
}

export function fitPolygonForRegion(region: BrowserRegion): readonly Point[] {
  const safe = region.layout.safePolygon
  const bubble = region.bubblePolygon
  // The current production companion may use the complete bubble hull as its
  // "safe" polygon. That includes outlines and tails, so keep translated text
  // in the original OCR text area until a genuinely inset polygon is supplied.
  if (safe && (!bubble || !samePolygon(safe, bubble))) return safe
  return region.textPolygon
}

function regionBox(region: BrowserRegion, imageWidth: number, imageHeight: number): FitBox {
  const points = fitPolygonForRegion(region)
  const bounds = polygonBounds(points)
  return {
    ...bounds,
    pixelWidth: bounds.width * imageWidth,
    pixelHeight: bounds.height * imageHeight,
  }
}

function rectangleFits(
  lines: readonly string[],
  fontSize: number,
  box: FitBox,
  region: BrowserRegion,
): boolean {
  const lineHeight = fontSize * region.style.lineHeight
  if (region.style.writingMode === 'vertical-rl') {
    const longest = Math.max(...lines.map((line) => [...line].length), 1)
    return (
      longest * lineHeight <= box.pixelHeight * FIT_CONTENT_RATIO &&
      lines.length * fontSize <= box.pixelWidth * FIT_CONTENT_RATIO
    )
  }
  const width = Math.max(...lines.map(estimatedLineUnits), 0) * fontSize
  const height = lines.length * lineHeight
  return (
    width <= box.pixelWidth * FIT_CONTENT_RATIO &&
    height <= box.pixelHeight * FIT_CONTENT_RATIO
  )
}

function polygonFits(
  lines: readonly string[],
  fontSize: number,
  points: readonly Point[],
  box: FitBox,
  region: BrowserRegion,
): boolean {
  if (region.style.writingMode === 'vertical-rl') {
    return rectangleFits(lines, fontSize, box, region)
  }
  const lineHeight = fontSize * region.style.lineHeight
  if (lines.length * lineHeight > box.pixelHeight * FIT_CONTENT_RATIO) return false
  const topInset = (box.pixelHeight - lines.length * lineHeight) / 2
  return lines.every((line, index) => {
    const normalizedY =
      box.minY +
      ((topInset + (index + 0.5) * lineHeight) / Math.max(1, box.pixelHeight)) *
        box.height
    const availablePixels = horizontalPolygonSpan(points, normalizedY) * imageWidthFromBox(box)
    return estimatedLineUnits(line) * fontSize <= availablePixels * FIT_CONTENT_RATIO
  })
}

function imageWidthFromBox(box: FitBox): number {
  return box.width > 0 ? box.pixelWidth / box.width : 0
}

export function minimumFontSizeForImage(imageWidth: number): number {
  return Math.max(
    ABSOLUTE_MINIMUM_FONT_PX,
    imageWidth * MINIMUM_FONT_TO_IMAGE_WIDTH,
  )
}

function chooseFit(
  region: BrowserRegion,
  imageWidth: number,
  imageHeight: number,
  usePolygon: boolean,
): TextFit {
  const text = region.displayedChinese
  const box = regionBox(region, imageWidth, imageHeight)
  const points = fitPolygonForRegion(region)
  const candidates = nearbyLineCandidates(text, region.layout.suggestedLines)
  const initial = Math.max(
    ABSOLUTE_MINIMUM_FONT_PX,
    region.layout.fontSizeToImageWidth * imageWidth,
  )
  let best: TextFit | undefined
  for (const lines of candidates) {
    const fits = (fontSize: number): boolean =>
      usePolygon
        ? polygonFits(lines, fontSize, points, box, region)
        : rectangleFits(lines, fontSize, box, region)
    let low = 0
    let high = initial
    if (fits(initial)) {
      low = initial
    } else {
      for (let iteration = 0; iteration < BINARY_SEARCH_STEPS; iteration += 1) {
        const midpoint = (low + high) / 2
        if (fits(midpoint)) low = midpoint
        else high = midpoint
      }
    }
    // Stay fractionally inside the mathematical boundary so subpixel
    // rounding cannot create a one-pixel scroll overflow.
    const fontSize = low === initial ? initial : low * 0.997
    const candidate = {
      fontSize,
      lines,
      degraded: fontSize + Number.EPSILON < minimumFontSizeForImage(imageWidth),
      usedPolygon: usePolygon,
    }
    if (!best || candidate.fontSize > best.fontSize) {
      best = candidate
    }
  }
  return (
    best ?? {
      fontSize: 0,
      lines: candidates[0] ?? [text],
      degraded: true,
      usedPolygon: usePolygon,
    }
  )
}

export class RectangleTextFitter {
  fit(region: BrowserRegion, imageWidth: number, imageHeight: number): TextFit {
    return chooseFit(region, imageWidth, imageHeight, false)
  }
}

export class PolygonTextFitter {
  fit(region: BrowserRegion, imageWidth: number, imageHeight: number): TextFit {
    return chooseFit(region, imageWidth, imageHeight, true)
  }
}
