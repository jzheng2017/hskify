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
  it('pipelines visible images within both concurrency and source-pixel budgets', async () => {
    const gates = new Map([
      ['small', deferred()],
      ['story', deferred()],
      ['next', deferred()],
    ])
    const order: string[] = []
    const queue = new VisibleFirstQueue<string>(
      async (item) => {
        order.push(item.id)
        await gates.get(item.id)?.promise
      },
      {},
      { maximumConcurrent: 2, maximumActiveCost: 80 },
    )

    queue.enqueue({ id: 'small', value: 'small', visible: true, order: 0, cost: 5 })
    queue.enqueue({ id: 'story', value: 'story', visible: true, order: 1, cost: 70 })
    queue.enqueue({ id: 'next', value: 'next', visible: true, order: 2, cost: 70 })

    await vi.waitFor(() => expect(order).toEqual(['small', 'story']))
    gates.get('small')?.resolve()
    await Promise.resolve()
    expect(order).toEqual(['small', 'story'])
    gates.get('story')?.resolve()
    await vi.waitFor(() => expect(order).toEqual(['small', 'story', 'next']))
    gates.get('next')?.resolve()
    await vi.waitFor(() => expect(queue.size).toBe(0))
  })

  it('isolates the visible startup frontier until the first final result', async () => {
    const gates = new Map([
      ['cover', deferred()],
      ['story', deferred()],
      ['offscreen', deferred()],
    ])
    const order: string[] = []
    const queue = new VisibleFirstQueue<string>(
      async (item) => {
        order.push(item.id)
        await gates.get(item.id)?.promise
      },
      {},
      { maximumConcurrent: 3 },
    )

    queue.beginInteractiveStartup()
    queue.enqueue({ id: 'cover', value: 'cover', visible: true, order: 0 })
    queue.enqueue({ id: 'story', value: 'story', visible: true, order: 1 })
    queue.enqueue({ id: 'offscreen', value: 'offscreen', visible: false, order: 2 })

    await vi.waitFor(() => expect(order).toEqual(['cover']))
    gates.get('cover')?.resolve()
    await vi.waitFor(() => expect(order).toEqual(['cover', 'story']))

    queue.enableThroughput()
    await vi.waitFor(() => expect(order).toEqual(['cover', 'story', 'offscreen']))
    gates.get('story')?.resolve()
    gates.get('offscreen')?.resolve()
    await vi.waitFor(() => expect(queue.size).toBe(0))
  })

  it('opens throughput after a visible frontier with no final result is exhausted', async () => {
    const visible = deferred()
    const offscreen = deferred()
    const order: string[] = []
    const queue = new VisibleFirstQueue<string>(
      async (item) => {
        order.push(item.id)
        if (item.visible) await visible.promise
        else await offscreen.promise
      },
      {},
      { maximumConcurrent: 2 },
    )

    queue.beginInteractiveStartup()
    queue.enqueue({ id: 'visible', value: 'visible', visible: true, order: 0 })
    queue.enqueue({ id: 'offscreen-a', value: 'a', visible: false, order: 1 })
    queue.enqueue({ id: 'offscreen-b', value: 'b', visible: false, order: 2 })

    await vi.waitFor(() => expect(order).toEqual(['visible']))
    visible.resolve()
    await vi.waitFor(() =>
      expect(order).toEqual(['visible', 'offscreen-a', 'offscreen-b']),
    )
    offscreen.resolve()
    await vi.waitFor(() => expect(queue.size).toBe(0))
  })

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

  it('accepts a fresh run with the same id while cancelled work settles', async () => {
    const order: string[] = []
    const succeeded: string[] = []
    const queue = new VisibleFirstQueue<string>(
      async (item, signal) => {
        order.push(item.value)
        if (item.value === 'cancelled run') {
          await new Promise<void>((resolve) => {
            signal.addEventListener('abort', () => resolve(), { once: true })
          })
        }
      },
      { onSuccess: (item) => succeeded.push(item.value) },
    )

    queue.enqueue({
      id: 'same-image',
      value: 'cancelled run',
      visible: true,
      order: 0,
    })
    queue.cancelAll()
    expect(queue.activeId).toBeUndefined()
    expect(
      queue.enqueue({
        id: 'same-image',
        value: 'fresh run',
        visible: true,
        order: 0,
      }),
    ).toBe(true)

    await vi.waitFor(() => expect(queue.size).toBe(0))
    expect(order).toEqual(['cancelled run', 'fresh run'])
    expect(succeeded).toEqual(['fresh run'])
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

  it('preempts and requeues active offscreen work when a pending image becomes visible', async () => {
    const order: string[] = []
    const preempted: string[] = []
    let releaseVisible!: () => void
    const visibleGate = new Promise<void>((resolve) => {
      releaseVisible = resolve
    })
    const queue = new VisibleFirstQueue<string>(
      async (item, signal) => {
        order.push(item.id)
        if (item.id === 'offscreen' && order.length === 1) {
          await new Promise<void>((resolve) => {
            signal.addEventListener('abort', () => resolve(), { once: true })
          })
        }
        if (item.id === 'visible') await visibleGate
      },
      { onPreempt: (item) => preempted.push(item.id) },
    )

    queue.enqueue({ id: 'offscreen', value: 'offscreen', visible: true, order: 0 })
    queue.enqueue({ id: 'visible', value: 'visible', visible: false, order: 1 })
    queue.reprioritize('offscreen', false)
    queue.reprioritize('visible', true)

    await vi.waitFor(() => expect(order).toEqual(['offscreen', 'visible']))
    releaseVisible()
    await vi.waitFor(() => expect(queue.size).toBe(0))
    expect(order).toEqual(['offscreen', 'visible', 'offscreen'])
    expect(preempted).toEqual(['offscreen'])
  })

  it('preempts active offscreen work when a newly enqueued image is visible', async () => {
    const order: string[] = []
    const succeeded: string[] = []
    const failed: string[] = []
    const queue = new VisibleFirstQueue<string>(
      async (item, signal) => {
        order.push(item.id)
        if (item.id === 'offscreen' && order.length === 1) {
          await new Promise<void>((resolve) => {
            signal.addEventListener('abort', () => resolve(), { once: true })
          })
        }
      },
      {
        onSuccess: (item) => succeeded.push(item.id),
        onFailure: (item) => failed.push(item.id),
      },
    )

    queue.enqueue({ id: 'offscreen', value: 'offscreen', visible: false, order: 0 })
    queue.enqueue({ id: 'visible', value: 'visible', visible: true, order: 1 })

    await vi.waitFor(() => expect(queue.size).toBe(0))
    expect(order).toEqual(['offscreen', 'visible', 'offscreen'])
    expect(succeeded).toEqual(['visible', 'offscreen'])
    expect(failed).toEqual([])
  })

  it('does not preempt active visible work for another visible image', async () => {
    const gate = deferred()
    const order: string[] = []
    const queue = new VisibleFirstQueue<string>(async (item) => {
      order.push(item.id)
      if (item.id === 'first') await gate.promise
    })

    queue.enqueue({ id: 'first', value: 'first', visible: true, order: 0 })
    queue.enqueue({ id: 'second', value: 'second', visible: false, order: 1 })
    queue.reprioritize('second', true)
    await Promise.resolve()
    expect(order).toEqual(['first'])

    gate.resolve()
    await vi.waitFor(() => expect(queue.size).toBe(0))
    expect(order).toEqual(['first', 'second'])
  })

  it('updates queued document order without weakening visible-first priority', async () => {
    const gate = deferred()
    const order: string[] = []
    const queue = new VisibleFirstQueue<string>(async (item) => {
      order.push(item.id)
      if (item.id === 'active') await gate.promise
    })

    queue.enqueue({ id: 'active', value: 'active', visible: true, order: 0 })
    queue.enqueue({ id: 'offscreen-first', value: 'first', visible: false, order: 1 })
    queue.enqueue({ id: 'visible-later', value: 'visible', visible: true, order: 5 })
    queue.enqueue({ id: 'offscreen-moved', value: 'moved', visible: false, order: 4 })
    queue.reprioritize('offscreen-moved', false, 0)

    gate.resolve()
    await vi.waitFor(() => expect(queue.size).toBe(0))
    expect(order).toEqual([
      'active',
      'visible-later',
      'offscreen-moved',
      'offscreen-first',
    ])
  })

  it('keeps a complete chapter in document order even when later pages are visible', async () => {
    const gates = new Map([
      ['page-0', deferred()],
      ['page-1', deferred()],
      ['page-2', deferred()],
    ])
    const order: string[] = []
    const queue = new VisibleFirstQueue<string>(
      async (item) => {
        order.push(item.id)
        await gates.get(item.id)?.promise
      },
      {},
      { maximumConcurrent: 1 },
    )
    queue.setOrdering('document')
    queue.beginInteractiveStartup()
    queue.enqueueBatch([
      { id: 'page-0', value: '0', visible: false, order: 0 },
      { id: 'page-2', value: '2', visible: true, order: 2 },
      { id: 'page-1', value: '1', visible: true, order: 1 },
    ])

    await vi.waitFor(() => expect(order).toEqual(['page-0']))
    gates.get('page-0')?.resolve()
    await vi.waitFor(() => expect(order).toEqual(['page-0', 'page-1']))
    gates.get('page-1')?.resolve()
    await vi.waitFor(() => expect(order).toEqual(['page-0', 'page-1', 'page-2']))
    gates.get('page-2')?.resolve()
    await vi.waitFor(() => expect(queue.size).toBe(0))
  })

  it('does not automatically re-enqueue a failed item and requires explicit retry', async () => {
    let attempts = 0
    const failures: string[] = []
    const queue = new VisibleFirstQueue<string>(
      async () => {
        attempts += 1
        if (attempts === 1) throw new Error('fixture failure')
      },
      { onFailure: (item) => failures.push(item.id) },
    )
    const item = { id: 'failed', value: 'value', visible: true, order: 0 }
    expect(queue.enqueue(item)).toBe(true)
    await vi.waitFor(() => expect(failures).toEqual(['failed']))
    expect(queue.enqueue(item)).toBe(false)
    expect(attempts).toBe(1)
    expect(queue.retry(item)).toBe(true)
    await vi.waitFor(() => expect(attempts).toBe(2))
  })

  it('allows a bounded retry to be requested directly from the failure callback', async () => {
    let attempts = 0
    const item = { id: 'failed', value: 'value', visible: true, order: 0 }
    let queue!: VisibleFirstQueue<string>
    queue = new VisibleFirstQueue<string>(
      async () => {
        attempts += 1
        if (attempts === 1) throw new Error('temporary fixture failure')
      },
      {
        onFailure: (failedItem) => {
          expect(queue.retry(failedItem)).toBe(true)
        },
      },
    )

    expect(queue.enqueue(item)).toBe(true)
    await vi.waitFor(() => expect(attempts).toBe(2))
    expect(queue.size).toBe(0)
  })
})
