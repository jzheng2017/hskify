import type { HskLevel, ReadingDirection } from '../contracts/browser'

export const HSK_LEVEL_KEY = 'hskify.settings.hskLevel'
export const LEARNING_MODE_KEY = 'hskify.settings.learningMode'
export const READING_DIRECTION_KEY = 'hskify.settings.readingDirection'
export const DEFAULT_HSK_LEVEL: HskLevel = 5
export const DEFAULT_LEARNING_MODE: LearningMode = 'natural'
export const DEFAULT_READING_DIRECTION: ReadingDirection = 'ltr'

export type LearningMode = 'natural' | 'strict'

export type StorageArea = {
  get(keys?: string | string[] | Record<string, unknown> | null): Promise<Record<string, unknown>>
  set(items: Record<string, unknown>): Promise<void>
  remove(keys: string | string[]): Promise<void>
}

export function isHskLevel(value: unknown): value is HskLevel {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= 1 &&
    value <= 6
  )
}

export function isLearningMode(value: unknown): value is LearningMode {
  return value === 'natural' || value === 'strict'
}

export function isReadingDirection(value: unknown): value is ReadingDirection {
  return value === 'ltr' || value === 'rtl'
}

export async function loadHskLevel(
  storage: StorageArea = browser.storage.local,
): Promise<HskLevel> {
  const values = await storage.get(HSK_LEVEL_KEY)
  return isHskLevel(values[HSK_LEVEL_KEY]) ? values[HSK_LEVEL_KEY] : DEFAULT_HSK_LEVEL
}

export async function saveHskLevel(
  level: HskLevel,
  storage: StorageArea = browser.storage.local,
): Promise<void> {
  await storage.set({ [HSK_LEVEL_KEY]: level })
}

export async function loadLearningMode(
  storage: StorageArea = browser.storage.local,
): Promise<LearningMode> {
  const values = await storage.get(LEARNING_MODE_KEY)
  return isLearningMode(values[LEARNING_MODE_KEY])
    ? values[LEARNING_MODE_KEY]
    : DEFAULT_LEARNING_MODE
}

export async function saveLearningMode(
  mode: LearningMode,
  storage: StorageArea = browser.storage.local,
): Promise<void> {
  await storage.set({ [LEARNING_MODE_KEY]: mode })
}

export async function loadReadingDirection(
  storage: StorageArea = browser.storage.local,
): Promise<ReadingDirection> {
  const values = await storage.get(READING_DIRECTION_KEY)
  return isReadingDirection(values[READING_DIRECTION_KEY])
    ? values[READING_DIRECTION_KEY]
    : DEFAULT_READING_DIRECTION
}

export async function saveReadingDirection(
  direction: ReadingDirection,
  storage: StorageArea = browser.storage.local,
): Promise<void> {
  await storage.set({ [READING_DIRECTION_KEY]: direction })
}
