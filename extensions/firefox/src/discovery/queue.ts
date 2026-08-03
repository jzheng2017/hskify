export type QueueItem<T> = {
  id: string
  value: T
  visible: boolean
  order: number
  cost?: number
}

export type QueueCallbacks<T> = {
  onStart?: (item: QueueItem<T>) => void
  onPreempt?: (item: QueueItem<T>) => void
  onSuccess?: (item: QueueItem<T>) => void
  onFailure?: (item: QueueItem<T>, error: unknown) => void
  onIdle?: () => void
}

export type QueueCapacity = {
  maximumConcurrent?: number
  maximumActiveCost?: number
}

export type QueueOrdering = 'visible-first' | 'document'

export class VisibleFirstQueue<T> {
  private pending: QueueItem<T>[] = []
  private pendingIds = new Set<string>()
  private failedIds = new Set<string>()
  private active = new Map<string, { item: QueueItem<T>; controller: AbortController }>()
  private stopped = false
  private interactiveStartup = false
  private batchDepth = 0
  private ordering: QueueOrdering = 'visible-first'
  private readonly maximumConcurrent: number
  private readonly maximumActiveCost: number

  constructor(
    private readonly processor: (
      item: QueueItem<T>,
      signal: AbortSignal,
    ) => Promise<void>,
    private readonly callbacks: QueueCallbacks<T> = {},
    capacity: QueueCapacity = {},
  ) {
    this.maximumConcurrent = Math.max(1, Math.floor(capacity.maximumConcurrent ?? 1))
    this.maximumActiveCost = Math.max(1, capacity.maximumActiveCost ?? Number.POSITIVE_INFINITY)
  }

  enqueue(item: QueueItem<T>): boolean {
    if (
      this.failedIds.has(item.id) ||
      this.pendingIds.has(item.id) ||
      this.hasActive(item.id)
    ) {
      return false
    }
    this.stopped = false
    this.pending.push(item)
    this.pendingIds.add(item.id)
    this.sort()
    this.preemptOffscreenForVisible()
    if (this.batchDepth === 0) void this.drain()
    return true
  }

  /**
   * Atomically admit a discovered batch. Readers often discover the complete
   * initial chapter synchronously; draining after the first item would start
   * a later visible page before the lower document-order items are even in the
   * queue. The batch boundary makes canonical ordering observable to the
   * scheduler instead of depending on callback timing.
   */
  enqueueBatch(items: readonly QueueItem<T>[]): number {
    this.batchDepth += 1
    let accepted = 0
    try {
      for (const item of items) if (this.enqueue(item)) accepted += 1
    } finally {
      this.batchDepth -= 1
      if (this.batchDepth === 0) void this.drain()
    }
    return accepted
  }

  beginBatch(): void {
    this.batchDepth += 1
  }

  endBatch(): void {
    if (this.batchDepth === 0) return
    this.batchDepth -= 1
    if (this.batchDepth === 0) void this.drain()
  }

  retry(item: QueueItem<T>): boolean {
    if (!this.failedIds.delete(item.id)) return false
    return this.enqueue(item)
  }

  /**
   * Reserve startup capacity for the initially visible frontier.
   *
   * Model generations that have already started are not preemptible. Starting
   * throughput work beside the first visible image can therefore turn a
   * one-second interactive result into a multi-second queue wait. Startup
   * admits one visible image at a time until the consumer reports that the
   * first final result is installed. If the visible frontier contains no
   * result, throughput opens automatically after that frontier is exhausted.
   */
  beginInteractiveStartup(): void {
    this.interactiveStartup = true
  }

  enableThroughput(): void {
    if (!this.interactiveStartup) return
    this.interactiveStartup = false
    this.drain()
  }

  /**
   * Select the ordering policy for the current chapter run.
   *
   * A visible-first queue is useful for a viewport-only request.  A complete
   * chapter must instead submit pages in document order: the daemon owns one
   * ordered language stream and later pages must not consume context before
   * their predecessors have been admitted.  This is a policy boundary, not a
   * priority hint, so preemption is disabled for document-order runs.
   */
  setOrdering(ordering: QueueOrdering): void {
    this.ordering = ordering
    this.sort()
    this.preemptOffscreenForVisible()
    if (this.batchDepth === 0) this.drain()
  }

  reprioritize(id: string, visible: boolean, order?: number): void {
    const item = this.pending.find((entry) => entry.id === id)
    if (item) {
      item.visible = visible
      if (order !== undefined) item.order = order
    }
    const active = this.active.get(id)
    if (active) {
      active.item.visible = visible
      if (order !== undefined) active.item.order = order
    }
    this.sort()
    this.preemptOffscreenForVisible()
  }

  remove(id: string): void {
    const before = this.pending.length
    this.pending = this.pending.filter((item) => item.id !== id)
    if (this.pending.length !== before) this.pendingIds.delete(id)
    this.failedIds.delete(id)
    this.active.get(id)?.controller.abort()
  }

  cancelAll(): void {
    this.stopped = true
    this.interactiveStartup = false
    this.pending = []
    this.pendingIds.clear()
    this.failedIds.clear()
    for (const active of this.active.values()) active.controller.abort()
  }

  get size(): number {
    return this.pending.length + this.running().length
  }

  get activeId(): string | undefined {
    return this.running()[0]?.item.id
  }

  get next(): QueueItem<T> | undefined {
    return this.pending[0]
  }

  private sort(): void {
    this.pending.sort((left, right) =>
      this.ordering === 'document'
        ? left.order - right.order
        : Number(right.visible) - Number(left.visible) || left.order - right.order,
    )
  }

  private preemptOffscreenForVisible(): void {
    if (this.ordering === 'document') return
    const visible = this.pending.find((item) => item.visible)
    if (!visible || this.canStart(visible)) return
    const active = this.running().find((entry) => !entry.item.visible)
    if (!active) return
    // Put interrupted offscreen work back in reading order. Bounded capacity
    // admits newly visible work immediately when resources allow it.
    this.callbacks.onPreempt?.(active.item)
    this.pending.push(active.item)
    this.pendingIds.add(active.item.id)
    this.sort()
    active.controller.abort()
  }

  private running(): Array<{ item: QueueItem<T>; controller: AbortController }> {
    return [...this.active.values()].filter((entry) => !entry.controller.signal.aborted)
  }

  private itemCost(item: QueueItem<T>): number {
    return Math.max(0, item.cost ?? 1)
  }

  private canStart(item: QueueItem<T>): boolean {
    const running = this.running()
    if (this.interactiveStartup && running.length >= 1) return false
    if (running.length >= this.maximumConcurrent) return false
    if (running.length === 0) return true
    const activeCost = running.reduce(
      (total, entry) => total + this.itemCost(entry.item),
      0,
    )
    return activeCost + this.itemCost(item) <= this.maximumActiveCost
  }

  private drain(): void {
    if (this.stopped) return
    if (
      this.interactiveStartup &&
      this.running().length === 0 &&
      !this.pending.some((item) => item.visible)
    ) {
      // Nothing else in the initial viewport can produce an interactive
      // result. Restore normal chapter throughput instead of serializing
      // unrelated offscreen work.
      this.interactiveStartup = false
    }
    while (this.pending[0] && this.canStart(this.pending[0])) {
      const item = this.pending.shift()
      if (!item) break
      this.pendingIds.delete(item.id)
      this.start(item)
    }
    if (this.pending.length === 0 && this.running().length === 0) {
      this.callbacks.onIdle?.()
    }
  }

  private start(item: QueueItem<T>): void {
    const controller = new AbortController()
    this.active.set(item.id, { item, controller })
    this.callbacks.onStart?.(item)
    void this.processor(item, controller.signal)
      .then(() => {
        if (!controller.signal.aborted) this.callbacks.onSuccess?.(item)
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          this.failedIds.add(item.id)
          if (this.active.get(item.id)?.controller === controller) {
            this.active.delete(item.id)
          }
          this.callbacks.onFailure?.(item, error)
        }
      })
      .finally(() => {
        if (this.active.get(item.id)?.controller === controller) {
          this.active.delete(item.id)
        }
        if (!this.stopped) this.drain()
      })
  }

  private hasActive(id: string): boolean {
    const active = this.active.get(id)
    return active !== undefined && !active.controller.signal.aborted
  }
}
