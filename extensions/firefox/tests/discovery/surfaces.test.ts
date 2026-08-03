import { beforeEach, describe, expect, it, vi } from 'vitest'

import { discoverPageSurfaces, LiveSurfaceDiscovery } from '../../src/discovery/surfaces'

function rect(width = 900, height = 1600): DOMRect {
  return {
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: width,
    bottom: height,
    width,
    height,
    toJSON: () => ({}),
  } as DOMRect
}

describe('reader-agnostic page surfaces', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('normalizes images, srcset-selected images, canvases, and CSS backgrounds', () => {
    const root = document.implementation.createHTMLDocument('reader')
    const image = root.createElement('img')
    image.src = 'https://reader.test/page.webp'
    Object.defineProperties(image, {
      complete: { value: true },
      naturalWidth: { value: 900 },
      naturalHeight: { value: 1600 },
      currentSrc: { value: 'https://reader.test/page@2x.webp' },
    })
    image.getBoundingClientRect = () => rect()
    root.body.append(image)
    const canvas = root.createElement('canvas')
    canvas.width = 900
    canvas.height = 3000
    canvas.toDataURL = () => 'data:image/png;base64,AA=='
    canvas.getBoundingClientRect = () => rect(900, 3000)
    root.body.append(canvas)
    const background = root.createElement('div')
    background.style.backgroundImage = 'url("/page-3.webp")'
    background.getBoundingClientRect = () => rect()
    root.body.append(background)
    vi.spyOn(window, 'getComputedStyle').mockImplementation((element) => {
      if (element === background) return { backgroundImage: 'url("/page-3.webp")' } as CSSStyleDeclaration
      return { backgroundImage: 'none' } as CSSStyleDeclaration
    })

    const result = discoverPageSurfaces(root)
    expect(result.unsupported).toEqual([])
    expect(result.surfaces.map((surface) => surface.kind)).toEqual([
      'image',
      'canvas',
      'background',
    ])
    expect(result.surfaces[0]?.sourceUrl).toBe('https://reader.test/page@2x.webp')
    expect(result.surfaces[0]?.continuous).toBe(false)
    expect(result.surfaces[1]?.continuous).toBe(true)
  })

  it('reports cross-origin frames instead of attempting to inspect them', () => {
    const root = document.implementation.createHTMLDocument('reader')
    const frame = root.createElement('iframe')
    Object.defineProperty(frame, 'contentDocument', {
      get() {
        throw new DOMException('Blocked', 'SecurityError')
      },
    })
    root.body.append(frame)
    const result = discoverPageSurfaces(root)
    expect(result.surfaces).toHaveLength(0)
    expect(result.unsupported).toEqual([
      { kind: 'frame', element: frame, reason: 'cross-origin' },
    ])
  })

  it('emits one shared candidate contract for canvas and background surfaces', async () => {
    const root = document.implementation.createHTMLDocument('reader')
    const canvas = root.createElement('canvas')
    canvas.width = 900
    canvas.height = 1800
    canvas.toDataURL = () => 'data:image/png;base64,AA=='
    canvas.getBoundingClientRect = () => rect(900, 1800)
    root.body.append(canvas)

    const events: string[] = []
    const discovery = new LiveSurfaceDiscovery(
      (event) => events.push(`${event.type}:${event.candidate.kind}`),
      root,
    )
    discovery.start()
    const candidate = discovery.current()[0]
    expect(candidate).toMatchObject({
      kind: 'canvas',
      sourceWidth: 900,
      sourceHeight: 1800,
    })
    expect(events).toEqual(['added:canvas'])
    await expect(candidate?.capture?.()).resolves.toMatchObject({
      mimeType: 'image/png',
      width: 900,
      height: 1800,
    })
    discovery.stop()
  })

  it('captures WebGL framebuffer pixels in page order before encoding the patch source', async () => {
    const root = document.implementation.createHTMLDocument('reader')
    const canvas = root.createElement('canvas')
    canvas.width = 2
    canvas.height = 2
    canvas.getBoundingClientRect = () => rect(2, 2)
    const framebuffer = new Uint8Array([
      // bottom row (WebGL origin)
      0, 0, 255, 255, 0, 255, 0, 255,
      // top row
      255, 0, 0, 255, 255, 255, 255, 255,
    ])
    const gl = {
      RGBA: 0x1908,
      UNSIGNED_BYTE: 0x1401,
      readPixels: vi.fn((_x, _y, _width, _height, _format, _type, target: Uint8Array) => {
        target.set(framebuffer)
      }),
    }
    const context = {
      createImageData: () => ({ data: new Uint8ClampedArray(16) }),
      putImageData: vi.fn(),
    }
    const getContext = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
      ((kind: string) => (kind === 'webgl2' ? gl : kind === '2d' ? context : null)) as typeof HTMLCanvasElement.prototype.getContext,
    )
    vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue(
      'data:image/png;base64,AA==',
    )
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Uint8Array([1, 2, 3]))))
    root.body.append(canvas)

    const discovery = new LiveSurfaceDiscovery(() => undefined, root)
    discovery.start()
    const candidate = discovery.current()[0]
    expect(candidate?.kind).toBe('webgl')
    await expect(candidate?.capture?.()).resolves.toMatchObject({
      mimeType: 'image/png',
      width: 2,
      height: 2,
    })
    expect(gl.readPixels).toHaveBeenCalledTimes(1)
    expect(context.putImageData).toHaveBeenCalledTimes(1)
    getContext.mockRestore()
    discovery.stop()
  })

  it('keeps surface identities stable when a lazy surface is inserted before them', async () => {
    const root = document.implementation.createHTMLDocument('reader')
    const first = root.createElement('canvas')
    first.width = 900
    first.height = 1800
    first.toDataURL = () => 'data:image/png;base64,AA=='
    first.getBoundingClientRect = () => rect(900, 1800)
    const second = root.createElement('canvas')
    second.width = 900
    second.height = 1800
    second.toDataURL = () => 'data:image/png;base64,AA=='
    second.getBoundingClientRect = () => rect(900, 1800)
    root.body.append(first, second)

    const discovery = new LiveSurfaceDiscovery(() => undefined, root)
    discovery.start()
    const before = new Map(
      discovery.current().map((candidate) => [candidate.element, { id: candidate.id, sourceUrl: candidate.sourceUrl }]),
    )
    const inserted = root.createElement('canvas')
    inserted.width = 900
    inserted.height = 1800
    inserted.toDataURL = () => 'data:image/png;base64,AA=='
    inserted.getBoundingClientRect = () => rect(900, 1800)
    root.body.prepend(inserted)
    await new Promise<void>((resolve) => queueMicrotask(resolve))

    expect(discovery.current().find((candidate) => candidate.element === first)?.id).toBe(
      before.get(first)?.id,
    )
    expect(discovery.current().find((candidate) => candidate.element === second)?.id).toBe(
      before.get(second)?.id,
    )
    const sourceUrls = new Map(
      discovery.current().map((candidate) => [candidate.element, candidate.sourceUrl]),
    )
    expect(sourceUrls.get(first)).toBe(before.get(first)?.sourceUrl)
    expect(sourceUrls.get(second)).toBe(before.get(second)?.sourceUrl)
    discovery.stop()
  })

  it('keeps capture-only identities stable for canvases inside same-origin frames', async () => {
    const root = document.implementation.createHTMLDocument('reader')
    const frame = root.createElement('iframe')
    const child = document.implementation.createHTMLDocument('embedded-reader')
    Object.defineProperty(frame, 'contentDocument', { configurable: true, value: child })
    root.body.append(frame)

    const first = child.createElement('canvas')
    first.width = 900
    first.height = 1800
    first.toDataURL = () => 'data:image/png;base64,AA=='
    first.getBoundingClientRect = () => rect(900, 1800)
    const second = child.createElement('canvas')
    second.width = 900
    second.height = 1800
    second.toDataURL = () => 'data:image/png;base64,AA=='
    second.getBoundingClientRect = () => rect(900, 1800)
    child.body.append(first, second)

    const discovery = new LiveSurfaceDiscovery(() => undefined, root)
    discovery.start()
    const before = new Map(
      discovery.current().map((candidate) => [candidate.element, { id: candidate.id, sourceUrl: candidate.sourceUrl }]),
    )
    expect(discovery.current().every((candidate) => candidate.captureOnly)).toBe(true)

    child.body.prepend(child.createElement('div'))
    await new Promise<void>((resolve) => queueMicrotask(resolve))

    expect(discovery.current().find((candidate) => candidate.element === first)?.id).toBe(
      before.get(first)?.id,
    )
    expect(discovery.current().find((candidate) => candidate.element === first)?.sourceUrl).toBe(
      before.get(first)?.sourceUrl,
    )
    discovery.stop()
  })
})
