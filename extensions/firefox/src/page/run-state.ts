export type ImageRunPhase = 'queued' | 'running' | 'complete' | 'failed'

type ImageRunEntry = {
  phase: ImageRunPhase
  automaticRetries: number
}

export type ChapterRunSnapshot = {
  total: number
  queued: number
  running: number
  completed: number
  failed: number
  resolved: number
  unresolved: number
  allResolved: boolean
}

export class ChapterRunState<Key> {
  private readonly entries = new Map<Key, ImageRunEntry>()

  register(key: Key): boolean {
    if (this.entries.has(key)) return false
    this.entries.set(key, { phase: 'queued', automaticRetries: 0 })
    return true
  }

  start(key: Key): void {
    const entry = this.required(key)
    if (entry.phase !== 'queued') {
      throw new Error(`Cannot start an image while it is ${entry.phase}.`)
    }
    entry.phase = 'running'
  }

  preempt(key: Key): void {
    const entry = this.required(key)
    if (entry.phase !== 'running') {
      throw new Error(`Cannot preempt an image while it is ${entry.phase}.`)
    }
    entry.phase = 'queued'
  }

  automaticRetries(key: Key): number {
    return this.required(key).automaticRetries
  }

  automaticRetryQueued(key: Key): number {
    const entry = this.required(key)
    if (entry.phase !== 'running') {
      throw new Error(`Cannot retry an image automatically while it is ${entry.phase}.`)
    }
    entry.automaticRetries += 1
    entry.phase = 'queued'
    return entry.automaticRetries
  }

  complete(key: Key): void {
    const entry = this.required(key)
    if (entry.phase !== 'running') {
      throw new Error(`Cannot complete an image while it is ${entry.phase}.`)
    }
    entry.phase = 'complete'
  }

  fail(key: Key): void {
    const entry = this.required(key)
    if (entry.phase !== 'running') {
      throw new Error(`Cannot fail an image while it is ${entry.phase}.`)
    }
    entry.phase = 'failed'
  }

  manualRetryQueued(key: Key): boolean {
    const entry = this.entries.get(key)
    if (!entry || entry.phase !== 'failed') return false
    entry.phase = 'queued'
    entry.automaticRetries = 0
    return true
  }

  phase(key: Key): ImageRunPhase | undefined {
    return this.entries.get(key)?.phase
  }

  remove(key: Key): boolean {
    return this.entries.delete(key)
  }

  reset(): void {
    this.entries.clear()
  }

  snapshot(): ChapterRunSnapshot {
    let queued = 0
    let running = 0
    let completed = 0
    let failed = 0
    for (const entry of this.entries.values()) {
      switch (entry.phase) {
        case 'queued':
          queued += 1
          break
        case 'running':
          running += 1
          break
        case 'complete':
          completed += 1
          break
        case 'failed':
          failed += 1
          break
      }
    }
    const total = this.entries.size
    const resolved = completed + failed
    const unresolved = queued + running
    return {
      total,
      queued,
      running,
      completed,
      failed,
      resolved,
      unresolved,
      allResolved: total > 0 && unresolved === 0 && resolved === total,
    }
  }

  private required(key: Key): ImageRunEntry {
    const entry = this.entries.get(key)
    if (!entry) throw new Error('The image is not part of this chapter run.')
    return entry
  }
}
