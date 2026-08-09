import { afterEach, describe, expect, it, vi } from 'vitest'

import { DocumentFocusTracker, type IntersectionObserverFactory } from '../../src/document'

describe('document focus tracking', () => {
  afterEach(() => vi.useRealTimers())

  it('orders and bounds visible focus to the 64-item wire limit', () => {
    vi.useFakeTimers()
    let notify: IntersectionObserverCallback | undefined
    const factory: IntersectionObserverFactory = (callback) => {
      notify = callback
      return {
        root: null,
        rootMargin: '200px 0px',
        scrollMargin: '0px',
        thresholds: [0],
        disconnect: vi.fn(),
        observe: vi.fn(),
        takeRecords: () => [],
        unobserve: vi.fn(),
      }
    }
    const ids = Array.from({ length: 80 }, (_, index) => `block-${index}`)
    const elements = ids.map(() => document.createElement('p'))
    const changed = vi.fn()
    const tracker = new DocumentFocusTracker(ids, changed, factory)
    elements.forEach((element, index) => tracker.observe(element, ids[index]!))

    notify?.(
      elements
        .toReversed()
        .map(
          (target) => ({ target, isIntersecting: true }) as unknown as IntersectionObserverEntry,
        ),
      {} as IntersectionObserver,
    )
    vi.advanceTimersByTime(100)

    expect(changed).toHaveBeenCalledTimes(1)
    expect(changed.mock.calls[0]?.[0]).toHaveLength(64)
    expect(changed.mock.calls[0]?.[0]).toEqual(ids.slice(0, 64))
    tracker.destroy()
  })
})
