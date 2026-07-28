export type QueueItem<T> = {
  id: string
  value: T
  visible: boolean
  order: number
  cost?: number
}

export type QueueCallbacks<T> = {
  onStart?: (item: QueueItem<T>) => void
  onSuccess?: (item: QueueItem<T>) => void
  onFailure?: (item: QueueItem<T>, error: unknown) => void
  onIdle?: () => void
}

export type QueueCapacity = {
  maximumConcurrent?: number
  maximumActiveCost?: number
}

export class VisibleFirstQueue<T> {
  private pending: QueueItem<T>[] = []
  private pendingIds = new Set<string>()
  private failedIds = new Set<string>()
  private active = new Map<string, { item: QueueItem<T>; controller: AbortController }>()
  private stopped = false
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
    void this.drain()
    return true
  }

  retry(item: QueueItem<T>): boolean {
    if (!this.failedIds.delete(item.id)) return false
    return this.enqueue(item)
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
    this.pending.sort(
      (left, right) =>
        Number(right.visible) - Number(left.visible) || left.order - right.order,
    )
  }

  private preemptOffscreenForVisible(): void {
    const visible = this.pending.find((item) => item.visible)
    if (!visible || this.canStart(visible)) return
    const active = this.running().find((entry) => !entry.item.visible)
    if (!active) return
    // Put interrupted offscreen work back in reading order. Bounded capacity
    // admits newly visible work immediately when resources allow it.
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
