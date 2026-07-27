import { describe, expect, it, vi } from 'vitest'

import {
  SingleImagePrefetch,
  type ImagePrefetchIdentity,
} from '../../src/acquisition/single-image-prefetch'

function identity(pageIndex: number): ImagePrefetchIdentity {
  return {
    tabId: 7,
    frameId: 0,
    pageSessionId: 'chapter',
    pageUrl: 'https://reader.test/chapter',
    pageIndex,
    sourceUrl: `https://cdn.test/${pageIndex}.webp`,
    naturalWidth: 900,
    naturalHeight: 16_000,
  }
}

describe('single-image acquisition prefetch', () => {
  it('aborts a reprioritized target and starts its replacement only after it settles', async () => {
    const prefetch = new SingleImagePrefetch<string>()
    let concurrent = 0
    let maximumConcurrent = 0
    let firstStarted = false
    const first = prefetch.prefetch(
      identity(1),
      (signal) =>
        new Promise<string>((resolve) => {
          concurrent += 1
          maximumConcurrent = Math.max(maximumConcurrent, concurrent)
          firstStarted = true
          signal.addEventListener(
            'abort',
            () => {
              concurrent -= 1
              resolve('stale')
            },
            { once: true },
          )
        }),
    )
    await vi.waitFor(() => expect(firstStarted).toBe(true))

    const second = prefetch.prefetch(identity(2), async () => {
      concurrent += 1
      maximumConcurrent = Math.max(maximumConcurrent, concurrent)
      concurrent -= 1
      return 'next'
    })
    await Promise.all([first, second])

    expect(maximumConcurrent).toBe(1)
    await expect(prefetch.consume(identity(2))).resolves.toBe('next')
    await expect(prefetch.consume(identity(2))).resolves.toBeUndefined()
  })

  it('clears retained bytes and aborts in-flight work on matching cancellation', async () => {
    const prefetch = new SingleImagePrefetch<{ bytes: ArrayBuffer }>()
    const retained = identity(1)
    await prefetch.prefetch(retained, async () => ({
      bytes: new ArrayBuffer(1024),
    }))
    await prefetch.cancelIf(
      (candidate) => candidate.pageSessionId === retained.pageSessionId,
    )
    await expect(prefetch.consume(retained)).resolves.toBeUndefined()

    let started = false
    let aborted = false
    const pending = prefetch.prefetch(
      identity(2),
      (signal) =>
        new Promise((resolve) => {
          started = true
          signal.addEventListener(
            'abort',
            () => {
              aborted = true
              resolve({ bytes: new ArrayBuffer(1) })
            },
            { once: true },
          )
        }),
    )
    await vi.waitFor(() => expect(started).toBe(true))
    await expect(prefetch.consume(identity(1))).resolves.toBeUndefined()
    await pending
    expect(aborted).toBe(true)
    await expect(prefetch.consume(identity(2))).resolves.toBeUndefined()
  })
})
