import { describe, expect, it } from 'vitest'

import {
  ImageValidationError,
  readImageDimensions,
  sniffImageMimeType,
  validateImageBytes,
} from '../../src/acquisition/image-format'
import { pngHeader } from '../helpers/images'

describe('safe image validation', () => {
  it('sniffs PNG bytes and reads dimensions instead of trusting the DOM', () => {
    const bytes = pngHeader(2048, 4096)
    expect(sniffImageMimeType(bytes)).toBe('image/png')
    expect(readImageDimensions(bytes, 'image/png')).toEqual({
      width: 2048,
      height: 4096,
    })
  })

  it('rejects MIME confusion', () => {
    expect(() => validateImageBytes(pngHeader(), 'image/jpeg')).toThrowError(
      /does not match/i,
    )
  })

  it('rejects byte, pixel, and dimension limits before upload', () => {
    expect(() =>
      validateImageBytes(pngHeader(100, 100, 100), 'image/png', {
        maximumBytes: 50,
        maximumWidth: 1_000,
        maximumHeight: 1_000,
        maximumPixels: 1_000_000,
      }),
    ).toThrow(ImageValidationError)
    expect(() =>
      validateImageBytes(pngHeader(20_000, 20_000), 'image/png'),
    ).toThrow(/dimensions/i)
  })

  it('accepts the daemon-aligned image limits and rejects values above them', () => {
    expect(() => validateImageBytes(pngHeader(1, 1, 20 * 1024 * 1024))).not.toThrow()
    expect(() => validateImageBytes(pngHeader(1, 1, 20 * 1024 * 1024 + 1))).toThrow(
      /between 1 byte/i,
    )

    expect(() => validateImageBytes(pngHeader(16_384, 1))).not.toThrow()
    expect(() => validateImageBytes(pngHeader(16_385, 1))).toThrow(/dimensions/i)

    expect(() => validateImageBytes(pngHeader(5_000, 5_000))).not.toThrow()
    expect(() => validateImageBytes(pngHeader(5_001, 5_000))).toThrow(/dimensions/i)
  })

  it('rejects unsupported and malformed content', () => {
    expect(() => sniffImageMimeType(new TextEncoder().encode('<svg/>').buffer)).toThrow(
      /Only PNG/i,
    )
    expect(() => readImageDimensions(pngHeader().slice(0, 20), 'image/png')).toThrow(
      /malformed/i,
    )
  })
})
