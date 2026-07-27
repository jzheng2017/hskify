import { describe, expect, it, vi } from 'vitest'

import {
  objectFitRect,
  polygonBounds,
  rectDifference,
  visibleImageRects,
} from '../../src/rendering/geometry'
import { loadedImage } from '../helpers/images'

describe('normalized and object-fit geometry', () => {
  it('maps contain with letterboxing and percentage object-position', () => {
    expect(objectFitRect(400, 400, 1600, 900, 'contain', '25% 50%')).toEqual({
      left: 0,
      top: 87.5,
      width: 400,
      height: 225,
    })
  })

  it('maps cover with crop offsets', () => {
    const rect = objectFitRect(400, 400, 1600, 900, 'cover', '70% 50%')
    expect(rect.width).toBeCloseTo(711.11, 2)
    expect(rect.height).toBe(400)
    expect(rect.left).toBeCloseTo(-217.78, 2)
    expect(rect.top).toBe(0)
  })

  it('handles none, scale-down, keyword position, and normalized bounds', () => {
    expect(objectFitRect(500, 500, 200, 100, 'none', 'right bottom')).toEqual({
      left: 300,
      top: 400,
      width: 200,
      height: 100,
    })
    expect(objectFitRect(500, 500, 200, 100, 'scale-down', 'center')).toEqual({
      left: 150,
      top: 200,
      width: 200,
      height: 100,
    })
    expect(
      polygonBounds([
        { x: 0.2, y: 0.4 },
        { x: 0.7, y: 0.3 },
        { x: 0.6, y: 0.9 },
      ]),
    ).toEqual({
      minX: 0.2,
      minY: 0.3,
      maxX: 0.7,
      maxY: 0.9,
      width: 0.49999999999999994,
      height: 0.6000000000000001,
    })
  })

  it('enforces the two-CSS-pixel layout shift threshold input', () => {
    const rect = (left: number, width = 600) =>
      ({
        left,
        top: 10,
        width,
        height: 900,
      }) as DOMRect
    expect(rectDifference(rect(0), rect(1.9))).toBeCloseTo(1.9)
    expect(rectDifference(rect(0), rect(0, 603))).toBe(3)
  })

  it('maps the browser viewport into normalized source coordinates', () => {
    const image = loadedImage()
    document.body.append(image)
    vi.spyOn(image, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      right: 500,
      top: -500,
      bottom: 1_500,
      width: 400,
      height: 2_000,
      x: 100,
      y: -500,
      toJSON: () => ({}),
    })
    const [visible] = visibleImageRects(image, 900, 16_000)
    expect(visible?.x).toBeCloseTo(0)
    expect(visible?.y).toBeCloseTo(0.25)
    expect(visible?.width).toBeCloseTo(1)
    expect(visible?.height).toBeCloseTo(window.innerHeight / 2_000)
    image.remove()
  })
})
