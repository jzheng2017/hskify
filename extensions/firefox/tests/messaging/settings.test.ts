import { describe, expect, it } from 'vitest'

import {
  DEFAULT_HSK_LEVEL,
  DEFAULT_LEARNING_MODE,
  DEFAULT_READING_DIRECTION,
  HSK_LEVEL_KEY,
  LEARNING_MODE_KEY,
  READING_DIRECTION_KEY,
  loadHskLevel,
  loadLearningMode,
  loadReadingDirection,
  saveHskLevel,
  saveLearningMode,
  saveReadingDirection,
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

  it('uses natural learning by default and remembers strict HSK mode', async () => {
    const storage = new MemoryStorage()
    expect(await loadLearningMode(storage)).toBe(DEFAULT_LEARNING_MODE)
    await saveLearningMode('strict', storage)
    expect(await loadLearningMode(storage)).toBe('strict')
    expect(storage.values[LEARNING_MODE_KEY]).toBe('strict')
    storage.values[LEARNING_MODE_KEY] = 'automatic'
    expect(await loadLearningMode(storage)).toBe('natural')
  })

  it('uses LTR by default and remembers explicit manga reading order', async () => {
    const storage = new MemoryStorage()
    expect(await loadReadingDirection(storage)).toBe(DEFAULT_READING_DIRECTION)
    await saveReadingDirection('rtl', storage)
    expect(await loadReadingDirection(storage)).toBe('rtl')
    expect(storage.values[READING_DIRECTION_KEY]).toBe('rtl')
    storage.values[READING_DIRECTION_KEY] = 'auto'
    expect(await loadReadingDirection(storage)).toBe('ltr')
  })
})
