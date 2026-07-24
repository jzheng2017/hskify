import { describe, expect, it, vi } from 'vitest'

import { VisibleFirstQueue } from '../../src/discovery/queue'

function deferred() {
  let resolve!: () => void
  const promise = new Promise<void>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

describe('visible-first page queue', () => {
  it('runs one item at a time and prioritizes visible pending items', async () => {
    const gate = deferred()
    const order: string[] = []
    let concurrent = 0
    let maximumConcurrent = 0
    const queue = new VisibleFirstQueue<string>(async (item) => {
      concurrent += 1
      maximumConcurrent = Math.max(maximumConcurrent, concurrent)
      order.push(item.id)
      if (item.id === 'active') await gate.promise
      concurrent -= 1
    })
    queue.enqueue({ id: 'active', value: 'active', visible: true, order: 0 })
    queue.enqueue({ id: 'offscreen', value: 'offscreen', visible: false, order: 1 })
    queue.enqueue({ id: 'visible-later', value: 'visible', visible: true, order: 2 })
    gate.resolve()
    await vi.waitFor(() => expect(queue.size).toBe(0))
    expect(order).toEqual(['active', 'visible-later', 'offscreen'])
    expect(maximumConcurrent).toBe(1)
  })

  it('cancels the active item and clears queued work', async () => {
    let aborted = false
    const queue = new VisibleFirstQueue<string>(async (_item, signal) => {
      await new Promise<void>((resolve) => {
        signal.addEventListener(
          'abort',
          () => {
            aborted = true
            resolve()
          },
          { once: true },
        )
      })
    })
    queue.enqueue({ id: 'one', value: 'one', visible: true, order: 0 })
    queue.enqueue({ id: 'two', value: 'two', visible: true, order: 1 })
    queue.cancelAll()
    await vi.waitFor(() => expect(aborted).toBe(true))
    expect(queue.size).toBe(0)
  })

  it('can remove a pending item without stopping the active pipeline', async () => {
    const gate = deferred()
    const processed: string[] = []
    const queue = new VisibleFirstQueue<string>(async (item) => {
      processed.push(item.id)
      if (item.id === 'active') await gate.promise
    })
    queue.enqueue({ id: 'active', value: 'a', visible: true, order: 0 })
    queue.enqueue({ id: 'removed', value: 'b', visible: true, order: 1 })
    queue.remove('removed')
    gate.resolve()
    await vi.waitFor(() => expect(queue.size).toBe(0))
    expect(processed).toEqual(['active'])
  })
})
