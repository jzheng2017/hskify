import { describe, expect, it } from 'vitest'

import {
  DEFAULT_HSK_LEVEL,
  DEFAULT_NAME_TRANSLATION,
  HSK_LEVEL_KEY,
  NAME_TRANSLATION_KEY,
  loadHskLevel,
  loadNameTranslation,
  saveHskLevel,
  saveNameTranslation,
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

  it('keeps original names by default and remembers the name preference', async () => {
    const storage = new MemoryStorage()
    expect(await loadNameTranslation(storage)).toBe(DEFAULT_NAME_TRANSLATION)
    await saveNameTranslation('chinese', storage)
    expect(await loadNameTranslation(storage)).toBe('chinese')
    expect(storage.values[NAME_TRANSLATION_KEY]).toBe('chinese')
    storage.values[NAME_TRANSLATION_KEY] = 'literal-translation'
    expect(await loadNameTranslation(storage)).toBe('keep-original')
  })
})
