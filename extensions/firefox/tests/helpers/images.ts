export function pngHeader(width = 1200, height = 1800, totalBytes = 24): ArrayBuffer {
  const size = Math.max(24, totalBytes)
  const bytes = new Uint8Array(size)
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0)
  const chunk = new TextEncoder().encode('IHDR')
  bytes.set(chunk, 12)
  const view = new DataView(bytes.buffer)
  view.setUint32(16, width)
  view.setUint32(20, height)
  return bytes.buffer
}

export function loadedImage(
  source = 'https://reader.test/panel.png',
  naturalWidth = 1200,
  naturalHeight = 1800,
  rect: Partial<DOMRect> = {},
): HTMLImageElement {
  const image = document.createElement('img')
  image.src = source
  Object.defineProperties(image, {
    complete: { configurable: true, value: true },
    naturalWidth: { configurable: true, value: naturalWidth },
    naturalHeight: { configurable: true, value: naturalHeight },
    currentSrc: { configurable: true, value: source },
  })
  const fullRect = {
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: 600,
    bottom: 900,
    width: 600,
    height: 900,
    toJSON: () => ({}),
    ...rect,
  } satisfies DOMRect
  image.getBoundingClientRect = () => fullRect
  return image
}
