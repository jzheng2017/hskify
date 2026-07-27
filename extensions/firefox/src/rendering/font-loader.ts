const CATEGORY_FALLBACKS = {
  sans: '"Noto Sans CJK SC", "Microsoft YaHei", sans-serif',
  serif: '"Noto Serif CJK SC", SimSun, serif',
  handwritten: '"KaiTi", "STKaiti", cursive',
  display: '"Microsoft YaHei", sans-serif',
  brush: '"FZKai-Z03", "KaiTi", cursive',
} as const

export type FontCategory = keyof typeof CATEGORY_FALLBACKS
export type FontFetcher = (fontId: string, jobId: string) => Promise<ArrayBuffer>

export class FontLoader {
  private readonly cache = new Map<string, Promise<string>>()

  constructor(
    private readonly fetcher: FontFetcher,
    private readonly fontSet: FontFaceSet = document.fonts,
    private readonly FontFaceType: typeof FontFace | undefined = globalThis.FontFace,
  ) {}

  load(fontId: string, category: FontCategory, jobId: string): Promise<string> {
    const cacheKey = `${category}:${fontId}`
    const cached = this.cache.get(cacheKey)
    if (cached) return cached
    const task = this.loadUncached(fontId, category, jobId)
    this.cache.set(cacheKey, task)
    return task
  }

  private async loadUncached(
    fontId: string,
    category: FontCategory,
    jobId: string,
  ): Promise<string> {
    const fallback = CATEGORY_FALLBACKS[category]
    if (!this.FontFaceType) return fallback
    try {
      const bytes = await this.fetcher(fontId, jobId)
      const family = `HMT-${fontId.replace(/[^\w-]/g, '-')}`
      const face = new this.FontFaceType(family, bytes)
      await face.load()
      this.fontSet.add(face)
      return `"${family}", ${fallback}`
    } catch {
      return fallback
    }
  }
}
