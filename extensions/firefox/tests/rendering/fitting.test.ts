import { describe, expect, it } from 'vitest'

import type { BrowserRegion } from '../../src/contracts/browser'
import { createFixtureRegions } from '../support/fixture-service'
import {
  PolygonTextFitter,
  RectangleTextFitter,
  horizontalPolygonIntervals,
  horizontalPolygonSpan,
  isLegalLineBreak,
  minimumFontSizeForImage,
  minimumReadableFontSize,
  nearbyLineCandidates,
  sourceDensityScale,
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
      spaced.flat().some((line) => /^\s/u.test(line)),
    ).toBe(false)
  })

  it('never breaks a retained Latin name or title inside a word', () => {
    const text = '帝国称它为 SILVER HARBOR。'
    const candidates = nearbyLineCandidates(text, [])
    expect(candidates.length).toBeGreaterThan(1)
    expect(
      candidates.flat().some((line) => /(?:CO|ÉT)$|^(?:UP|TAT)/u.test(line)),
    ).toBe(false)
    expect(candidates.every((lines) => lines.join('') === text)).toBe(true)
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
    expect(horizontalPolygonIntervals(excludedMiddle, 0.8)).toEqual([
      { left: 0, right: 0.3 },
      { left: 0.7, right: 1 },
    ])
  })

  it('rejects centered lines that miss asymmetric and concave scanline intervals', () => {
    const shapes = [
      [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
      ],
      [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0.7, y: 1 },
        { x: 0.7, y: 0.3 },
        { x: 0.3, y: 0.3 },
        { x: 0.3, y: 1 },
        { x: 0, y: 1 },
      ],
    ]

    for (const polygon of shapes) {
      const region: BrowserRegion = {
        ...fixtureRegion(),
        displayedChinese: '中',
        textPolygon: polygon,
        bubblePolygon: polygon,
        layout: {
          ...fixtureRegion().layout,
          safePolygon: polygon,
          suggestedLines: ['中'],
          fontSizeToImageWidth: 0.2,
        },
      }

      expect(horizontalPolygonSpan(polygon, 0.5)).toBeGreaterThan(0)
      expect(new RectangleTextFitter().fit(region, 100, 100).fontSize).toBeGreaterThan(36)
      const fit = new PolygonTextFitter().fit(region, 100, 100)
      expect(fit.fontSize).toBeGreaterThanOrEqual(minimumReadableFontSize(region, 100))
      expect(fit.degraded).toBe(true)
    }
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

  it('scales concise Chinese by source-to-translation glyph density', () => {
    expect(sourceDensityScale('The tower administrator arrived', '管理员来了')).toBeGreaterThan(1)
    expect(sourceDensityScale('Wait', '请等一下')).toBe(1)
    expect(sourceDensityScale('A'.repeat(100), '中')).toBeGreaterThan(5)
  })

  it('does not force translated line count to equal the source color sample count', () => {
    const base = fixtureRegion()
    const region: BrowserRegion = {
      ...base,
      sourceEnglish: 'A'.repeat(100),
      displayedChinese: '我们现在必须离开这里',
      style: {
        ...base.style,
        colorBands: [
          { position: 0.2, foreground: '#111111' },
          { position: 0.5, foreground: '#b91c1c' },
          { position: 0.8, foreground: '#111111' },
        ],
      },
      layout: {
        ...base.layout,
        suggestedLines: ['我们现在', '必须离开', '这里'],
      },
    }

    region.textPolygon = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ]
    region.bubblePolygon = region.textPolygon
    region.layout.safePolygon = region.textPolygon
    const fit = new PolygonTextFitter().fit(region, 600, 120)

    expect(fit.lines.length).toBeLessThan(3)
    expect(fit.lines.join('')).toBe(region.displayedChinese)
  })

  it('keeps font sizing above the readable floor when a bubble is small', () => {
    const polygon = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ]
    const region: BrowserRegion = {
      ...fixtureRegion(),
      sourceEnglish: '中',
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
    expect(fit.fontSize).toBeGreaterThanOrEqual(minimumReadableFontSize(region, 160))
    expect(fit.fontSize).toBeGreaterThanOrEqual(minimumFontSizeForImage(160))
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
        // The daemon supplies an inset distance-field polygon; fitting
        // against the bubble outline would hide overflow until render time.
        safePolygon: [
          { x: 0.3, y: 0.25 },
          { x: 0.7, y: 0.25 },
          { x: 0.7, y: 0.75 },
          { x: 0.3, y: 0.75 },
        ],
        suggestedLines: [],
        fontSizeToImageWidth: 0.1,
      },
    }

    const fit = new PolygonTextFitter().fit(region, 100, 100)
    expect(fit.fontSize).toBeGreaterThanOrEqual(minimumReadableFontSize(region, 100))
    expect(fit.degraded).toBe(true)
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
