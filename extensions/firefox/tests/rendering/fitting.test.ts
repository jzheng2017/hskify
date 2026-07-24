import { describe, expect, it } from 'vitest'

import type { BrowserRegion } from '../../src/contracts/browser'
import { createFixtureResult } from '../../src/messaging/fixture-service'
import {
  PolygonTextFitter,
  RectangleTextFitter,
  horizontalPolygonSpan,
  isLegalLineBreak,
  nearbyLineCandidates,
} from '../../src/rendering/fitting'

function fixtureRegion(): BrowserRegion {
  return createFixtureResult({
    jobId: 'fixture',
    sourceSha256: 'a'.repeat(64),
    sourceWidth: 1200,
    sourceHeight: 1800,
  }).regions[0] as BrowserRegion
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
