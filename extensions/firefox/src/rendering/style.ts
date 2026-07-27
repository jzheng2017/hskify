import type { BrowserRegion } from '../contracts/browser'

const HEX_COLOR = /^#(?:[\da-f]{3}|[\da-f]{4}|[\da-f]{6}|[\da-f]{8})$/i

export function safeHexColor(value: string | undefined, fallback: string): string {
  return value && HEX_COLOR.test(value) ? value : fallback
}

export type AppliedRegionStyle = {
  color: string
  fontWeight: string
  fontStyle: string
  textAlign: 'left' | 'center' | 'right'
  writingMode: 'horizontal-tb' | 'vertical-rl'
  lineHeight: string
  letterSpacing: string
  strokeColor: string
  strokeWidthRatio: number
  shadowColor: string
  shadowXRatio: number
  shadowYRatio: number
  italicDegrees: number
}

function bounded(value: number, minimum: number, maximum: number, fallback: number): number {
  return Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, value)) : fallback
}

export function validateRegionStyle(region: BrowserRegion): AppliedRegionStyle {
  const style = region.style
  return {
    color: safeHexColor(style.foreground, '#111111'),
    fontWeight: String(Math.round(bounded(style.weight, 100, 900, 400))),
    fontStyle: Math.abs(style.italicDegrees) >= 4 ? 'italic' : 'normal',
    textAlign: ['left', 'center', 'right'].includes(style.alignment)
      ? style.alignment
      : 'center',
    writingMode:
      style.writingMode === 'vertical-rl' ? 'vertical-rl' : 'horizontal-tb',
    lineHeight: String(bounded(style.lineHeight, 0.8, 2.2, 1.1)),
    letterSpacing: `${bounded(style.letterSpacingEm, -0.08, 0.3, 0)}em`,
    strokeColor: safeHexColor(style.outlineColor, 'transparent'),
    strokeWidthRatio: bounded(style.outlineWidthRatio, 0, 0.2, 0),
    shadowColor: safeHexColor(style.shadowColor, 'transparent'),
    shadowXRatio: bounded(style.shadowXRatio, -0.3, 0.3, 0),
    shadowYRatio: bounded(style.shadowYRatio, -0.3, 0.3, 0),
    italicDegrees: bounded(style.italicDegrees, -30, 30, 0),
  }
}

export function applyRegionStyle(
  element: HTMLElement,
  region: BrowserRegion,
  fontSize: number,
  fontFamily: string,
): void {
  const style = validateRegionStyle(region)
  element.style.color = style.color
  element.style.fontFamily = fontFamily
  element.style.fontWeight = style.fontWeight
  element.style.fontStyle = style.fontStyle
  element.style.textAlign = style.textAlign
  element.style.writingMode = style.writingMode
  element.style.textOrientation = style.writingMode === 'vertical-rl' ? 'upright' : 'mixed'
  element.style.lineHeight = style.lineHeight
  element.style.letterSpacing = style.letterSpacing
  element.style.fontSize = `${fontSize}px`
  element.style.setProperty(
    '-webkit-text-stroke',
    `${fontSize * style.strokeWidthRatio}px ${style.strokeColor}`,
  )
  element.style.textShadow = `${fontSize * style.shadowXRatio}px ${
    fontSize * style.shadowYRatio
  }px ${Math.max(0, fontSize * 0.025)}px ${style.shadowColor}`
  element.style.transform = `skewX(${style.italicDegrees}deg)`
}

export function applyRegionColorBands(
  element: HTMLElement,
  region: BrowserRegion,
  fontSize: number,
): void {
  const bands = region.style.colorBands ?? []
  const lines = [...element.querySelectorAll<HTMLElement>('.hmt-region-line')]
  if (bands.length <= 1 || lines.length === 0) return
  for (const [index, line] of lines.entries()) {
    const position = (index + 0.5) / lines.length
    const band = bands.reduce((closest, candidate) =>
      Math.abs(candidate.position - position) < Math.abs(closest.position - position)
        ? candidate
        : closest,
    )
    line.style.color = safeHexColor(band.foreground, '#111111')
    const outline = safeHexColor(band.outlineColor, 'transparent')
    line.style.setProperty(
      '-webkit-text-stroke',
      `${fontSize * validateRegionStyle(region).strokeWidthRatio}px ${outline}`,
    )
  }
}
