import { describe, expect, it } from 'vitest'

import type { BrowserRegion } from '../../src/contracts/browser'
import { createFixtureRegions } from '../../src/messaging/fixture-service'
import {
  PolygonTextFitter,
  RectangleTextFitter,
  horizontalPolygonSpan,
  isLegalLineBreak,
  minimumFontSizeForImage,
  nearbyLineCandidates,
} from '../../src/rendering/fitting'

function fixtureRegion(): BrowserRegion {
  return createFixtureRegions({
    jobId: 'fixture',
    sourceSha256: 'a'.repeat(64),
    sourceWidth: 1200,
    sourceHeight: 1800,
  })[0] as BrowserRegion
}

describe('rectangle and polygon-aware text fitting', () => {
  it('never adds or drops Chinese while proposing nearby lines', () => {
    const text = '我们现在要走！'
    const candidates = nearbyLineCandidates(text, ['我们现在', '要走！'])
    expect(candidates.length).toBeGreaterThan(1)
    expect(candidates.every((lines) => lines.join('') === text)).toBe(true)
  })

  it('avoids illegal punctuation breaks', () => {
    expect(isLegalLineBreak('我们（', '现在')).toBe(false)
    expect(isLegalLineBreak('我们', '，现在')).toBe(false)
    expect(isLegalLineBreak('我们', '现在')).toBe(true)
    const candidates = nearbyLineCandidates('你好，世界！', [])
    expect(candidates.flat().some((line) => line.startsWith('，'))).toBe(false)
    const spaced = nearbyLineCandidates('面对那 6 个氏族', [])
    expect(
      spaced.flat().some((line) => /^\s|\s$/u.test(line)),
    ).toBe(false)
  })

  it('computes usable spans for irregular polygons', () => {
    const diamond = [
      { x: 0.5, y: 0 },
      { x: 1, y: 0.5 },
      { x: 0.5, y: 1 },
      { x: 0, y: 0.5 },
    ]
    expect(horizontalPolygonSpan(diamond, 0.5)).toBeCloseTo(1)
    expect(horizontalPolygonSpan(diamond, 0.1)).toBeCloseTo(0.2)

    const excludedMiddle = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0.7, y: 1 },
      { x: 0.7, y: 0.3 },
      { x: 0.3, y: 0.3 },
      { x: 0.3, y: 1 },
      { x: 0, y: 1 },
    ]
    expect(horizontalPolygonSpan(excludedMiddle, 0.8)).toBeCloseTo(0.3)
  })

  it('fits the frozen fixture in both rectangle and polygon modes', () => {
    const region = fixtureRegion()
    const rectangle = new RectangleTextFitter().fit(region, 600, 900)
    const polygon = new PolygonTextFitter().fit(region, 600, 900)
    expect(rectangle.fontSize).toBeGreaterThanOrEqual(8)
    expect(polygon.fontSize).toBeGreaterThanOrEqual(8)
    expect(rectangle.lines.join('')).toBe(region.displayedChinese)
    expect(polygon.lines.join('')).toBe(region.displayedChinese)
  })

  it('keeps font sizing proportional below the old eight-pixel floor', () => {
    const polygon = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ]
    const region: BrowserRegion = {
      ...fixtureRegion(),
      displayedChinese: '中',
      textPolygon: polygon,
      bubblePolygon: polygon,
      layout: {
        ...fixtureRegion().layout,
        safePolygon: polygon,
        suggestedLines: [],
        fontSizeToImageWidth: 0.02,
      },
    }

    const fit = new RectangleTextFitter().fit(region, 160, 160)
    expect(fit.fontSize).toBeCloseTo(3.2)
    expect(fit.fontSize).toBeLessThan(8)
    expect(minimumFontSizeForImage(160)).toBeLessThan(fit.fontSize)
  })

  it('reserves an inner margin instead of fitting against the bubble outline', () => {
    const bubble = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ]
    const textPolygon = [
      { x: 0.25, y: 0.2 },
      { x: 0.75, y: 0.2 },
      { x: 0.75, y: 0.8 },
      { x: 0.25, y: 0.8 },
    ]
    const region: BrowserRegion = {
      ...fixtureRegion(),
      displayedChinese: '中'.repeat(35),
      textPolygon,
      bubblePolygon: bubble,
      layout: {
        ...fixtureRegion().layout,
        safePolygon: bubble,
        suggestedLines: [],
        fontSizeToImageWidth: 0.1,
      },
    }

    const fit = new PolygonTextFitter().fit(region, 100, 100)
    expect(fit.fontSize).toBeLessThan(10)
    expect(
      Math.max(...fit.lines.map((line) => [...line].length)) * fit.fontSize,
    ).toBeLessThanOrEqual(44)
    expect(
      fit.lines.length * fit.fontSize * region.style.lineHeight,
    ).toBeLessThanOrEqual(52.8)
  })

  it('supports vertical text without converting all regions to vertical', () => {
    const horizontal = fixtureRegion()
    const vertical: BrowserRegion = {
      ...horizontal,
      style: { ...horizontal.style, writingMode: 'vertical-rl' },
      textPolygon: [
        { x: 0.2, y: 0.1 },
        { x: 0.32, y: 0.1 },
        { x: 0.32, y: 0.8 },
        { x: 0.2, y: 0.8 },
      ],
      layout: {
        ...horizontal.layout,
        safePolygon: [
          { x: 0.2, y: 0.1 },
          { x: 0.32, y: 0.1 },
          { x: 0.32, y: 0.8 },
          { x: 0.2, y: 0.8 },
        ],
      },
    }
    expect(new PolygonTextFitter().fit(vertical, 600, 900).fontSize).toBeGreaterThan(0)
    expect(horizontal.style.writingMode).toBe('horizontal-tb')
  })
})
