export const ALLOWED_IMAGE_MIME_TYPES = [
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
] as const

export type SupportedImageMimeType = (typeof ALLOWED_IMAGE_MIME_TYPES)[number]

export type ImageDimensions = {
  width: number
  height: number
}

export class ImageValidationError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable = false,
    options?: ErrorOptions,
  ) {
    super(message, options)
    this.name = 'ImageValidationError'
  }
}

function view(bytes: ArrayBuffer): DataView {
  return new DataView(bytes)
}

function ascii(bytes: ArrayBuffer, start: number, length: number): string {
  return String.fromCharCode(...new Uint8Array(bytes, start, length))
}

export function normalizeMimeType(value: string | null | undefined): string {
  return value?.split(';', 1)[0]?.trim().toLowerCase() ?? ''
}

export function sniffImageMimeType(bytes: ArrayBuffer): SupportedImageMimeType {
  const data = new Uint8Array(bytes)
  if (
    data.length >= 24 &&
    data[0] === 0x89 &&
    data[1] === 0x50 &&
    data[2] === 0x4e &&
    data[3] === 0x47 &&
    data[4] === 0x0d &&
    data[5] === 0x0a &&
    data[6] === 0x1a &&
    data[7] === 0x0a
  ) {
    return 'image/png'
  }
  if (data.length >= 4 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) {
    return 'image/jpeg'
  }
  if (
    data.length >= 16 &&
    ascii(bytes, 0, 4) === 'RIFF' &&
    ascii(bytes, 8, 4) === 'WEBP'
  ) {
    return 'image/webp'
  }
  if (
    data.length >= 10 &&
    (ascii(bytes, 0, 6) === 'GIF87a' || ascii(bytes, 0, 6) === 'GIF89a')
  ) {
    return 'image/gif'
  }
  throw new ImageValidationError(
    'UNSUPPORTED_IMAGE_TYPE',
    'Only PNG, JPEG, WebP, and GIF manga images are supported.',
  )
}

function pngDimensions(bytes: ArrayBuffer): ImageDimensions {
  if (bytes.byteLength < 24 || ascii(bytes, 12, 4) !== 'IHDR') {
    throw new ImageValidationError('INVALID_IMAGE', 'The PNG header is malformed.')
  }
  const data = view(bytes)
  return { width: data.getUint32(16), height: data.getUint32(20) }
}

function gifDimensions(bytes: ArrayBuffer): ImageDimensions {
  const data = view(bytes)
  return { width: data.getUint16(6, true), height: data.getUint16(8, true) }
}

function jpegDimensions(bytes: ArrayBuffer): ImageDimensions {
  const data = view(bytes)
  let offset = 2
  const startOfFrame = new Set([
    0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce,
    0xcf,
  ])
  while (offset + 4 <= bytes.byteLength) {
    while (offset < bytes.byteLength && data.getUint8(offset) !== 0xff) offset += 1
    while (offset < bytes.byteLength && data.getUint8(offset) === 0xff) offset += 1
    if (offset >= bytes.byteLength) break
    const marker = data.getUint8(offset)
    offset += 1
    if (marker === 0xd8 || marker === 0xd9) continue
    if (marker === 0xda) break
    if (offset + 2 > bytes.byteLength) break
    const segmentLength = data.getUint16(offset)
    if (segmentLength < 2 || offset + segmentLength > bytes.byteLength) break
    if (startOfFrame.has(marker)) {
      if (segmentLength < 7) break
      return {
        height: data.getUint16(offset + 3),
        width: data.getUint16(offset + 5),
      }
    }
    offset += segmentLength
  }
  throw new ImageValidationError('INVALID_IMAGE', 'The JPEG dimensions could not be read.')
}

function webpDimensions(bytes: ArrayBuffer): ImageDimensions {
  if (bytes.byteLength < 30) {
    throw new ImageValidationError('INVALID_IMAGE', 'The WebP header is malformed.')
  }
  const data = view(bytes)
  const chunk = ascii(bytes, 12, 4)
  if (chunk === 'VP8X') {
    return {
      width:
        1 + data.getUint8(24) + (data.getUint8(25) << 8) + (data.getUint8(26) << 16),
      height:
        1 + data.getUint8(27) + (data.getUint8(28) << 8) + (data.getUint8(29) << 16),
    }
  }
  if (chunk === 'VP8L') {
    if (data.getUint8(20) !== 0x2f) {
      throw new ImageValidationError('INVALID_IMAGE', 'The lossless WebP header is malformed.')
    }
    const bits = data.getUint32(21, true)
    return {
      width: 1 + (bits & 0x3fff),
      height: 1 + ((bits >>> 14) & 0x3fff),
    }
  }
  if (chunk === 'VP8 ' && bytes.byteLength >= 30) {
    if (
      data.getUint8(23) !== 0x9d ||
      data.getUint8(24) !== 0x01 ||
      data.getUint8(25) !== 0x2a
    ) {
      throw new ImageValidationError('INVALID_IMAGE', 'The lossy WebP header is malformed.')
    }
    return {
      width: data.getUint16(26, true) & 0x3fff,
      height: data.getUint16(28, true) & 0x3fff,
    }
  }
  throw new ImageValidationError('INVALID_IMAGE', 'The WebP dimensions could not be read.')
}

export function readImageDimensions(
  bytes: ArrayBuffer,
  mimeType: SupportedImageMimeType,
): ImageDimensions {
  switch (mimeType) {
    case 'image/png':
      return pngDimensions(bytes)
    case 'image/jpeg':
      return jpegDimensions(bytes)
    case 'image/webp':
      return webpDimensions(bytes)
    case 'image/gif':
      return gifDimensions(bytes)
  }
}

export type ImageLimits = {
  maximumBytes: number
  maximumWidth: number
  maximumHeight: number
  maximumPixels: number
}

export const DEFAULT_IMAGE_LIMITS: ImageLimits = {
  maximumBytes: 25 * 1024 * 1024,
  maximumWidth: 16_384,
  maximumHeight: 32_768,
  maximumPixels: 80_000_000,
}

export function validateImageBytes(
  bytes: ArrayBuffer,
  declaredMimeType?: string | null,
  limits: ImageLimits = DEFAULT_IMAGE_LIMITS,
): { mimeType: SupportedImageMimeType; dimensions: ImageDimensions } {
  if (bytes.byteLength === 0 || bytes.byteLength > limits.maximumBytes) {
    throw new ImageValidationError(
      'IMAGE_TOO_LARGE',
      `The image must be between 1 byte and ${limits.maximumBytes} bytes.`,
    )
  }
  const sniffed = sniffImageMimeType(bytes)
  const declared = normalizeMimeType(declaredMimeType)
  if (declared && declared !== 'application/octet-stream' && declared !== sniffed) {
    throw new ImageValidationError(
      'IMAGE_MIME_MISMATCH',
      'The image content does not match its declared MIME type.',
    )
  }
  const dimensions = readImageDimensions(bytes, sniffed)
  const pixels = dimensions.width * dimensions.height
  if (
    dimensions.width < 1 ||
    dimensions.height < 1 ||
    dimensions.width > limits.maximumWidth ||
    dimensions.height > limits.maximumHeight ||
    !Number.isSafeInteger(pixels) ||
    pixels > limits.maximumPixels
  ) {
    throw new ImageValidationError(
      'IMAGE_DIMENSIONS_UNSUPPORTED',
      'The decoded image dimensions exceed the safe processing limit.',
    )
  }
  return { mimeType: sniffed, dimensions }
}
