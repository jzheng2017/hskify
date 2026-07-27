export type QueueItem<T> = {
  id: string
  value: T
  visible: boolean
  order: number
}

export type QueueCallbacks<T> = {
  onStart?: (item: QueueItem<T>) => void
  onSuccess?: (item: QueueItem<T>) => void
  onFailure?: (item: QueueItem<T>, error: unknown) => void
  onIdle?: () => void
}

export class VisibleFirstQueue<T> {
  private pending: QueueItem<T>[] = []
  private pendingIds = new Set<string>()
  private failedIds = new Set<string>()
  private active: { item: QueueItem<T>; controller: AbortController } | undefined
  private stopped = false

  constructor(
    private readonly processor: (
      item: QueueItem<T>,
      signal: AbortSignal,
    ) => Promise<void>,
    private readonly callbacks: QueueCallbacks<T> = {},
  ) {}

  enqueue(item: QueueItem<T>): boolean {
    if (
      this.failedIds.has(item.id) ||
      this.pendingIds.has(item.id) ||
      (this.active?.item.id === item.id &&
        !this.active.controller.signal.aborted)
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
    if (this.active?.item.id === id) {
      this.active.item.visible = visible
      if (order !== undefined) this.active.item.order = order
    }
    this.sort()
    this.preemptOffscreenForVisible()
  }

  remove(id: string): void {
    const before = this.pending.length
    this.pending = this.pending.filter((item) => item.id !== id)
    if (this.pending.length !== before) this.pendingIds.delete(id)
    this.failedIds.delete(id)
    if (this.active?.item.id === id) this.active.controller.abort()
  }

  cancelAll(): void {
    this.stopped = true
    this.pending = []
    this.pendingIds.clear()
    this.failedIds.clear()
    this.active?.controller.abort()
  }

  get size(): number {
    return this.pending.length + (this.active && !this.active.controller.signal.aborted ? 1 : 0)
  }

  get activeId(): string | undefined {
    return this.active && !this.active.controller.signal.aborted
      ? this.active.item.id
      : undefined
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
    const active = this.active
    if (
      !active ||
      active.controller.signal.aborted ||
      active.item.visible ||
      !this.pending.some((item) => item.visible)
    ) {
      return
    }
    // Only one image is submitted at a time so decoded-image memory stays
    // bounded. If the reader scrolls, put the interrupted offscreen image back
    // in reading order and let the now-visible image start as soon as the
    // daemon observes cancellation at its current batch boundary.
    this.pending.push(active.item)
    this.pendingIds.add(active.item.id)
    this.sort()
    active.controller.abort()
  }

  private async drain(): Promise<void> {
    if (this.active || this.stopped) return
    const item = this.pending.shift()
    if (!item) {
      this.callbacks.onIdle?.()
      return
    }
    this.pendingIds.delete(item.id)
    const controller = new AbortController()
    this.active = { item, controller }
    this.callbacks.onStart?.(item)
    try {
      await this.processor(item, controller.signal)
      if (!controller.signal.aborted) {
        if (this.active?.controller === controller) this.active = undefined
        this.callbacks.onSuccess?.(item)
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        this.failedIds.add(item.id)
        if (this.active?.controller === controller) this.active = undefined
        this.callbacks.onFailure?.(item, error)
      }
    } finally {
      if (this.active?.controller === controller) this.active = undefined
      if (!this.stopped) void this.drain()
    }
  }
}
