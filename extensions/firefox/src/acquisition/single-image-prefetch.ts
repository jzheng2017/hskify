export type ImagePrefetchIdentity = {
  tabId: number
  frameId: number
  pageSessionId: string
  pageUrl: string
  pageIndex: number
  sourceUrl: string
  naturalWidth: number
  naturalHeight: number
}

type PendingPrefetch = {
  generation: number
  identity: ImagePrefetchIdentity
  key: string
  promise: Promise<void>
}

type RetainedPrefetch<T> = {
  identity: ImagePrefetchIdentity
  key: string
  value: T
}

function identityKey(identity: ImagePrefetchIdentity): string {
  return JSON.stringify([
    identity.tabId,
    identity.frameId,
    identity.pageSessionId,
    identity.pageUrl,
    identity.pageIndex,
    identity.sourceUrl,
    identity.naturalWidth,
    identity.naturalHeight,
  ])
}

export class SingleImagePrefetch<T> {
  private generation = 0
  private pending: PendingPrefetch | undefined
  private retained: RetainedPrefetch<T> | undefined
  private runningController: AbortController | undefined
  private tail: Promise<void> = Promise.resolve()

  prefetch(
    identity: ImagePrefetchIdentity,
    acquire: (signal: AbortSignal) => Promise<T>,
  ): Promise<void> {
    const key = identityKey(identity)
    if (this.retained?.key === key) return Promise.resolve()
    if (this.pending?.key === key) return this.pending.promise

    const generation = ++this.generation
    this.retained = undefined
    this.runningController?.abort()

    const previous = this.tail
    const promise = previous
      .catch(() => undefined)
      .then(async () => {
        if (generation !== this.generation) return
        const controller = new AbortController()
        this.runningController = controller
        try {
          const value = await acquire(controller.signal)
          if (generation === this.generation && !controller.signal.aborted) {
            this.retained = { identity, key, value }
          }
        } finally {
          if (generation === this.generation) {
            this.runningController = undefined
            if (this.pending?.generation === generation) this.pending = undefined
          }
        }
      })
    this.pending = { generation, identity, key, promise }
    this.tail = promise
    return promise
  }

  async consume(identity: ImagePrefetchIdentity): Promise<T | undefined> {
    const key = identityKey(identity)
    const pending = this.pending
    if (pending?.key === key) {
      await pending.promise.catch(() => undefined)
    } else if (pending || (this.retained && this.retained.key !== key)) {
      await this.cancel()
      return undefined
    }
    if (
      (this.pending && this.pending.key !== key) ||
      (this.retained && this.retained.key !== key)
    ) {
      await this.cancel()
      return undefined
    }

    if (this.retained?.key !== key) return undefined
    const value = this.retained.value
    this.retained = undefined
    this.generation += 1
    return value
  }

  async cancel(): Promise<void> {
    this.generation += 1
    this.retained = undefined
    this.pending = undefined
    this.runningController?.abort()
    this.runningController = undefined
    await this.tail.catch(() => undefined)
  }

  async cancelIf(
    predicate: (identity: ImagePrefetchIdentity) => boolean,
  ): Promise<void> {
    const identity = this.pending?.identity ?? this.retained?.identity
    if (identity && predicate(identity)) await this.cancel()
  }
}
