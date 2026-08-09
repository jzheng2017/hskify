import { MAX_VISIBLE_BLOCK_IDS } from '../contracts/browser'

export type IntersectionObserverFactory = (
  callback: IntersectionObserverCallback,
  options?: IntersectionObserverInit,
) => IntersectionObserver

export type DocumentFocusCallback = (visibleBlockIds: string[]) => void

const browserObserverFactory: IntersectionObserverFactory | undefined =
  typeof globalThis.IntersectionObserver === 'undefined'
    ? undefined
    : (callback, options) => new globalThis.IntersectionObserver(callback, options)

/**
 * Tracks document focus entirely through IntersectionObserver. The coalesced
 * callback never performs a geometry read and therefore adds no scroll-time
 * layout loop.
 */
export class DocumentFocusTracker {
  private readonly observer: IntersectionObserver | undefined
  private readonly itemByElement = new Map<Element, string>()
  private readonly visibleElements = new Set<Element>()
  private readonly orderByItem = new Map<string, number>()
  private timer: ReturnType<typeof setTimeout> | undefined
  private destroyed = false

  constructor(
    orderedItemIds: readonly string[],
    private readonly onChange: DocumentFocusCallback,
    factory: IntersectionObserverFactory | undefined = browserObserverFactory,
  ) {
    orderedItemIds.forEach((itemId, index) => this.orderByItem.set(itemId, index))
    this.observer = factory?.(this.onIntersection, {
      root: null,
      rootMargin: '200px 0px',
      threshold: 0,
    })
  }

  observe(element: Element, itemId: string): void {
    if (this.destroyed || this.itemByElement.has(element)) return
    this.itemByElement.set(element, itemId)
    this.observer?.observe(element)
  }

  unobserve(element: Element): void {
    this.observer?.unobserve(element)
    this.itemByElement.delete(element)
    this.visibleElements.delete(element)
    this.schedule()
  }

  visibleItemIds(): string[] {
    const visible = new Set<string>()
    for (const element of this.visibleElements) {
      const itemId = this.itemByElement.get(element)
      if (itemId) visible.add(itemId)
    }
    return [...visible]
      .sort(
        (left, right) =>
          (this.orderByItem.get(left) ?? Number.MAX_SAFE_INTEGER) -
          (this.orderByItem.get(right) ?? Number.MAX_SAFE_INTEGER),
      )
      .slice(0, MAX_VISIBLE_BLOCK_IDS)
  }

  destroy(): void {
    this.destroyed = true
    if (this.timer !== undefined) clearTimeout(this.timer)
    this.timer = undefined
    this.observer?.disconnect()
    this.itemByElement.clear()
    this.visibleElements.clear()
  }

  private readonly onIntersection: IntersectionObserverCallback = (entries): void => {
    for (const entry of entries) {
      if (!this.itemByElement.has(entry.target)) continue
      if (entry.isIntersecting) this.visibleElements.add(entry.target)
      else this.visibleElements.delete(entry.target)
    }
    this.schedule()
  }

  private schedule(): void {
    if (this.destroyed || this.timer !== undefined) return
    this.timer = setTimeout(() => {
      this.timer = undefined
      if (!this.destroyed) this.onChange(this.visibleItemIds())
    }, 100)
  }
}
