import { describe, expect, it, vi } from 'vitest'

import type { DiscoveredImage } from '../../src/discovery/images'
import { tryContentBytes } from '../../src/page/controller'
import { loadedImage } from '../helpers/images'

describe('same-origin content byte fallback', () => {
  it('stops streaming at the byte cap before building an ArrayBuffer', async () => {
    const image = loadedImage('data:image/png;base64,fixture')
    const candidate: DiscoveredImage = {
      id: 'image:0:data:image/png;base64,fixture',
      kind: 'image',
      element: image,
      owner: image,
      sourceUrl: image.currentSrc,
      sourceWidth: image.naturalWidth,
      sourceHeight: image.naturalHeight,
      domIndex: 0,
      visible: true,
    }
    let produced = 0
    let cancelled = false
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        produced += 1
        controller.enqueue(new Uint8Array(1024 * 1024))
      },
      cancel() {
        cancelled = true
      },
    })
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(body, { status: 200 })),
    )
    await expect(tryContentBytes(candidate)).resolves.toBeUndefined()
    expect(produced).toBeGreaterThanOrEqual(21)
    expect(produced).toBeLessThan(25)
    expect(cancelled).toBe(true)
    vi.unstubAllGlobals()
  })
})
