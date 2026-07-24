import type { StorageArea } from '../../src/messaging/settings'

export class MemoryStorage implements StorageArea {
  readonly values: Record<string, unknown> = {}

  async get(
    keys?: string | string[] | Record<string, unknown> | null,
  ): Promise<Record<string, unknown>> {
    if (keys === null || keys === undefined) return { ...this.values }
    if (typeof keys === 'string') return { [keys]: this.values[keys] }
    if (Array.isArray(keys)) {
      return Object.fromEntries(keys.map((key) => [key, this.values[key]]))
    }
    return Object.fromEntries(
      Object.entries(keys).map(([key, fallback]) => [
        key,
        this.values[key] ?? fallback,
      ]),
    )
  }

  async set(items: Record<string, unknown>): Promise<void> {
    Object.assign(this.values, items)
  }

  async remove(keys: string | string[]): Promise<void> {
    for (const key of typeof keys === 'string' ? [keys] : keys) {
      delete this.values[key]
    }
  }
}
