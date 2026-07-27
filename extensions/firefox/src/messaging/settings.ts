import type { HskLevel } from '../contracts/browser'

export const HSK_LEVEL_KEY = 'hmt.settings.hskLevel'
export const NAME_TRANSLATION_KEY = 'hmt.settings.nameTranslation'
export const DEFAULT_HSK_LEVEL: HskLevel = 5
export const DEFAULT_NAME_TRANSLATION: NameTranslation = 'keep-original'

export type NameTranslation = 'keep-original' | 'chinese'

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

export function isNameTranslation(value: unknown): value is NameTranslation {
  return value === 'keep-original' || value === 'chinese'
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

export async function loadNameTranslation(
  storage: StorageArea = browser.storage.local,
): Promise<NameTranslation> {
  const values = await storage.get(NAME_TRANSLATION_KEY)
  return isNameTranslation(values[NAME_TRANSLATION_KEY])
    ? values[NAME_TRANSLATION_KEY]
    : DEFAULT_NAME_TRANSLATION
}

export async function saveNameTranslation(
  preference: NameTranslation,
  storage: StorageArea = browser.storage.local,
): Promise<void> {
  await storage.set({ [NAME_TRANSLATION_KEY]: preference })
}
