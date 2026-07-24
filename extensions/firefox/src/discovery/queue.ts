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
    if (this.pendingIds.has(item.id) || this.active?.item.id === item.id) return false
    this.stopped = false
    this.pending.push(item)
    this.pendingIds.add(item.id)
    this.sort()
    void this.drain()
    return true
  }

  reprioritize(id: string, visible: boolean): void {
    const item = this.pending.find((entry) => entry.id === id)
    if (!item) return
    item.visible = visible
    this.sort()
  }

  remove(id: string): void {
    const before = this.pending.length
    this.pending = this.pending.filter((item) => item.id !== id)
    if (this.pending.length !== before) this.pendingIds.delete(id)
    if (this.active?.item.id === id) this.active.controller.abort()
  }

  cancelAll(): void {
    this.stopped = true
    this.pending = []
    this.pendingIds.clear()
    this.active?.controller.abort()
  }

  get size(): number {
    return this.pending.length + (this.active && !this.active.controller.signal.aborted ? 1 : 0)
  }

  get activeId(): string | undefined {
    return this.active?.item.id
  }

  private sort(): void {
    this.pending.sort(
      (left, right) =>
        Number(right.visible) - Number(left.visible) || left.order - right.order,
    )
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
      if (!controller.signal.aborted) this.callbacks.onSuccess?.(item)
    } catch (error) {
      if (!controller.signal.aborted) this.callbacks.onFailure?.(item, error)
    } finally {
      this.active = undefined
      if (!this.stopped) void this.drain()
    }
  }
}
