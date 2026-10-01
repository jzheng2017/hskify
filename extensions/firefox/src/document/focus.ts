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
  private resolveReady!: () => void
  readonly ready = new Promise<void>((resolve) => {
    this.resolveReady = resolve
  })

  constructor(
    orderedItemIds: readonly string[],
    private readonly onChange: DocumentFocusCallback,
    factory: IntersectionObserverFactory | undefined = browserObserverFactory,
  ) {
    orderedItemIds.forEach((itemId, index) => this.orderByItem.set(itemId, index))
    this.observer = factory?.(this.onIntersection, {
      root: null,
      rootMargin: '0px',
      threshold: 0,
    })
    if (!this.observer) this.resolveReady()
    document.addEventListener('visibilitychange', this.schedule)
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

  initializeFromRects(): void {
    // A single fallback snapshot, used only if initial observer delivery is unavailable.
    for (const element of this.itemByElement.keys()) {
      const rect = element.getBoundingClientRect(),
        view = element.ownerDocument.defaultView
      if (
        view &&
        rect.width > 0 &&
        rect.height > 0 &&
        rect.bottom > 0 &&
        rect.right > 0 &&
        rect.top < view.innerHeight &&
        rect.left < view.innerWidth
      )
        this.visibleElements.add(element)
    }
    this.resolveReady()
  }

  visibleItemIds(): string[] {
    if (document.hidden) return []
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
    this.resolveReady()
    if (this.timer !== undefined) clearTimeout(this.timer)
    this.timer = undefined
    this.observer?.disconnect()
    document.removeEventListener('visibilitychange', this.schedule)
    this.itemByElement.clear()
    this.visibleElements.clear()
  }

  private readonly onIntersection: IntersectionObserverCallback = (entries): void => {
    for (const entry of entries) {
      if (!this.itemByElement.has(entry.target)) continue
      if (entry.isIntersecting) this.visibleElements.add(entry.target)
      else this.visibleElements.delete(entry.target)
    }
    this.resolveReady()
    this.schedule()
  }

  private readonly schedule = (): void => {
    if (this.destroyed || this.timer !== undefined) return
    this.timer = setTimeout(() => {
      this.timer = undefined
      if (!this.destroyed) this.onChange(this.visibleItemIds())
    }, 100)
  }
}
