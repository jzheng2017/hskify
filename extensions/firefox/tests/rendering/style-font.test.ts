import { describe, expect, it, vi } from 'vitest'

import type { BrowserRegion } from '../../src/contracts/browser'
import { createFixtureResult } from '../../src/messaging/fixture-service'
import { FontLoader } from '../../src/rendering/font-loader'
import {
  applyRegionStyle,
  safeHexColor,
  validateRegionStyle,
} from '../../src/rendering/style'

function fixtureRegion(): BrowserRegion {
  return createFixtureResult({
    jobId: 'fixture',
    sourceSha256: 'a'.repeat(64),
    sourceWidth: 1200,
    sourceHeight: 1800,
  }).regions[0] as BrowserRegion
}

describe('validated browser typography', () => {
  it('accepts only hexadecimal model colours and clamps numeric style values', () => {
    const region = fixtureRegion()
    const unsafe = {
      ...region,
      rotationDegrees: Number.POSITIVE_INFINITY,
      style: {
        ...region.style,
        foreground: 'red; background:url(https://attacker.test)',
        outlineColor: 'rgb(1,2,3)',
        weight: 5_000,
        lineHeight: 99,
      },
    } as BrowserRegion
    const validated = validateRegionStyle(unsafe)
    expect(validated.color).toBe('#111111')
    expect(validated.strokeColor).toBe('transparent')
    expect(validated.fontWeight).toBe('900')
    expect(validated.lineHeight).toBe('2.2')
    expect(validated.rotationDegrees).toBe(0)
    expect(safeHexColor('#1234', 'fallback')).toBe('#1234')
  })

  it('applies horizontal and vertical CSS without inserting HTML', () => {
    const region = fixtureRegion()
    const element = document.createElement('span')
    element.textContent = region.displayedChinese
    applyRegionStyle(element, region, 24, '"Fixture Font", sans-serif')
    expect(element.textContent).toBe('我们现在要走！')
    expect(element.style.color).toBe('#151515')
    expect(element.style.fontSize).toBe('24px')
    expect(element.style.getPropertyValue('-webkit-text-stroke')).toContain('#ffffff')
  })

  it('caches a successfully loaded local font', async () => {
    const fetcher = vi.fn(async () => new Uint8Array([1, 2, 3]).buffer)
    const add = vi.fn()
    class TestFontFace {
      constructor(
        readonly family: string,
        readonly source: ArrayBuffer,
      ) {}
      async load() {
        return this
      }
    }
    const loader = new FontLoader(
      fetcher,
      { add } as unknown as FontFaceSet,
      TestFontFace as unknown as typeof FontFace,
    )
    const first = await loader.load('fixture-sans', 'sans', 'job-1')
    const second = await loader.load('fixture-sans', 'sans', 'job-1')
    expect(first).toContain('HMT-fixture-sans')
    expect(second).toBe(first)
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(add).toHaveBeenCalledTimes(1)
  })

  it('falls back visibly when font loading fails', async () => {
    class FailingFontFace {
      async load(): Promise<never> {
        throw new Error('bad font')
      }
    }
    const loader = new FontLoader(
      async () => new ArrayBuffer(1),
      { add: vi.fn() } as unknown as FontFaceSet,
      FailingFontFace as unknown as typeof FontFace,
    )
    await expect(loader.load('broken', 'handwritten', 'job-1')).resolves.toContain('KaiTi')
  })
})
