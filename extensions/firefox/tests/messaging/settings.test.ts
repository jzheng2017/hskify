import { describe, expect, it } from 'vitest'

import {
  DEFAULT_HSK_LEVEL,
  HSK_LEVEL_KEY,
  loadHskLevel,
  saveHskLevel,
} from '../../src/messaging/settings'
import { MemoryStorage } from '../helpers/storage'

describe('popup HSK persistence', () => {
  it('defaults to HSK 5 and remembers every valid cumulative level globally', async () => {
    const storage = new MemoryStorage()
    expect(await loadHskLevel(storage)).toBe(DEFAULT_HSK_LEVEL)
    for (const level of [1, 2, 3, 4, 5, 6] as const) {
      await saveHskLevel(level, storage)
      expect(await loadHskLevel(storage)).toBe(level)
    }
    expect(storage.values[HSK_LEVEL_KEY]).toBe(6)
  })

  it('ignores malformed persisted values', async () => {
    const storage = new MemoryStorage()
    storage.values[HSK_LEVEL_KEY] = 7
    expect(await loadHskLevel(storage)).toBe(5)
  })
})
