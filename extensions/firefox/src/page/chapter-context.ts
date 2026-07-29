import type { BrowserJobRequest } from '../contracts/browser'

export type ChapterContextRegion = {
  id: string
  readingOrder: number
  sourceEnglish: string
  displayedChinese: string
}

type DialogueContext = NonNullable<BrowserJobRequest['precedingContext']>[number]

/**
 * Stores completed dialogue by immutable document page and region order.
 *
 * Page jobs may be cancelled, retried, or completed after viewport
 * reprioritization. Context must therefore be derived from page order rather
 * than from the order in which asynchronous jobs happen to finish.
 */
export class ChapterContextLedger {
  private readonly pages = new Map<number, ChapterContextRegion[]>()

  constructor(private readonly maximumEntries = 6) {}

  commitPage(pageIndex: number, regions: readonly ChapterContextRegion[]): void {
    const ordered = regions
      .filter(
        (region) =>
          region.sourceEnglish.trim().length > 0 &&
          region.displayedChinese.trim().length > 0,
      )
      .map((region) => ({ ...region }))
      .sort(
        (left, right) =>
          left.readingOrder - right.readingOrder || left.id.localeCompare(right.id),
      )
    this.pages.set(pageIndex, ordered)
  }

  before(pageIndex: number): DialogueContext[] {
    const orderedPages = [...this.pages.entries()]
      .filter(([index]) => index < pageIndex)
      .sort(([left], [right]) => left - right)
    const context = orderedPages.flatMap(([, regions]) =>
      regions.map((region) => ({
        sourceEnglish: region.sourceEnglish,
        chinese: region.displayedChinese,
      })),
    )
    return context.slice(-this.maximumEntries)
  }

  removePage(pageIndex: number): void {
    this.pages.delete(pageIndex)
  }

  clear(): void {
    this.pages.clear()
  }
}
