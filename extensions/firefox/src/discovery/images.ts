const MIN_NATURAL_WIDTH = 320
const MIN_NATURAL_HEIGHT = 240
const MIN_DISPLAY_WIDTH = 180
const MIN_DISPLAY_HEIGHT = 140
const MIN_DISPLAY_AREA = 36_000
const EXCLUDED_SEMANTIC_WORDS =
  /\b(?:avatar|badge|button|comment|control|cover|emoji|favicon|icon|logo|profile|sprite|thumbnail|userpic)\b/i
const DEFERRED_SOURCE_ATTRIBUTE = /^data-(?:.*(?:src|url|image|original).*)$/i

export type ImageOwner = HTMLImageElement | HTMLPictureElement

export type DiscoveredImage = {
  element: HTMLImageElement
  owner: ImageOwner
  sourceUrl: string
  domIndex: number
  visible: boolean
}

export type DiscoveryDecision =
  | { supported: true; candidate: DiscoveredImage }
  | { supported: false; reason: string }

export type DiscoveryEvent =
  | { type: 'added'; candidate: DiscoveredImage }
  | {
      type: 'updated'
      candidate: DiscoveredImage
      previousSourceUrl: string
      previousDomIndex: number
    }
  | { type: 'removed'; candidate: DiscoveredImage }
  | { type: 'visibility'; candidate: DiscoveredImage }

type IntersectionObserverLike = {
  observe(target: Element): void
  unobserve(target: Element): void
  disconnect(): void
}

export type ObserverFactories = {
  mutation(callback: MutationCallback): Pick<MutationObserver, 'observe' | 'disconnect'>
  intersection(callback: IntersectionObserverCallback): IntersectionObserverLike | undefined
}

const defaultFactories: ObserverFactories = {
  mutation: (callback) => new MutationObserver(callback),
  intersection: (callback) =>
    typeof IntersectionObserver === 'undefined'
      ? undefined
      : new IntersectionObserver(callback, { rootMargin: '20% 0px' }),
}

function imageOwner(image: HTMLImageElement): ImageOwner {
  const parent = image.parentElement
  return parent instanceof HTMLPictureElement ? parent : image
}

function semanticText(image: HTMLImageElement): string {
  const owner = imageOwner(image)
  const ancestor = owner.closest('[class],[id],[role],button,nav')
  return [
    image.alt,
    image.title,
    image.id,
    image.className,
    owner.id,
    owner.className,
    ancestor?.id ?? '',
    ancestor?.className ?? '',
    ancestor?.getAttribute('role') ?? '',
    ancestor?.tagName ?? '',
  ].join(' ')
}

function hasUnsafeTransform(image: HTMLImageElement): boolean {
  const transform = getComputedStyle(image).transform
  return transform !== '' && transform !== 'none' && transform !== 'matrix(1, 0, 0, 1, 0, 0)'
}

function normalizedImageUrl(value: string): string | undefined {
  try {
    const url = new URL(value, document.baseURI)
    return ['http:', 'https:', 'blob:', 'data:'].includes(url.protocol) ? url.href : undefined
  } catch {
    return undefined
  }
}

export function deferredImageSourceUrl(image: HTMLImageElement): string | undefined {
  for (const attribute of image.attributes) {
    if (!DEFERRED_SOURCE_ATTRIBUTE.test(attribute.name)) continue
    const value = attribute.value.trim()
    if (!value) continue
    const normalized = normalizedImageUrl(value)
    if (normalized) return normalized
  }
  return undefined
}

function isRendered(image: HTMLImageElement, rect: DOMRect): boolean {
  const style = getComputedStyle(image)
  return (
    image.isConnected &&
    style.display !== 'none' &&
    style.visibility !== 'hidden' &&
    style.visibility !== 'collapse' &&
    Number(style.opacity || '1') > 0 &&
    rect.width > 0 &&
    rect.height > 0
  )
}

export function isRectVisible(rect: DOMRect, view: Window = window): boolean {
  return (
    rect.bottom > 0 && rect.right > 0 && rect.top < view.innerHeight && rect.left < view.innerWidth
  )
}

export function evaluateImage(image: HTMLImageElement, domIndex: number): DiscoveryDecision {
  if (image.closest('[data-hmt-owned="true"]') || image.hasAttribute('data-hmt-original')) {
    return { supported: false, reason: 'owned-by-extension' }
  }
  const sourceUrl = image.currentSrc || image.src
  if (!sourceUrl) return { supported: false, reason: 'missing-source' }
  if (!image.complete || image.naturalWidth === 0 || image.naturalHeight === 0) {
    return { supported: false, reason: 'not-loaded' }
  }
  if (image.naturalWidth < MIN_NATURAL_WIDTH || image.naturalHeight < MIN_NATURAL_HEIGHT) {
    return { supported: false, reason: 'intrinsic-size' }
  }
  const rect = image.getBoundingClientRect()
  if (!isRendered(image, rect)) return { supported: false, reason: 'hidden' }
  if (
    rect.width < MIN_DISPLAY_WIDTH ||
    rect.height < MIN_DISPLAY_HEIGHT ||
    rect.width * rect.height < MIN_DISPLAY_AREA
  ) {
    return { supported: false, reason: 'display-size' }
  }
  if (
    image.closest('button,[role="button"]') ||
    EXCLUDED_SEMANTIC_WORDS.test(semanticText(image))
  ) {
    return { supported: false, reason: 'page-control' }
  }
  if (hasUnsafeTransform(image)) {
    return { supported: false, reason: 'unsupported-transform' }
  }
  return {
    supported: true,
    candidate: {
      element: image,
      owner: imageOwner(image),
      sourceUrl,
      domIndex,
      visible: isRectVisible(rect),
    },
  }
}

export function isDeferredPageImage(image: HTMLImageElement): boolean {
  const deferredSource = deferredImageSourceUrl(image)
  if (!deferredSource) return false
  const decision = evaluateImage(image, 0)
  if (decision.supported) return false
  if (decision.reason !== 'not-loaded' && decision.reason !== 'intrinsic-size') {
    return false
  }
  const currentSource = normalizedImageUrl(image.currentSrc || image.src)
  if (decision.reason === 'intrinsic-size' && currentSource === deferredSource) {
    return false
  }
  const rect = image.getBoundingClientRect()
  if (!isRendered(image, rect)) return false
  if (
    rect.width < MIN_DISPLAY_WIDTH ||
    rect.height < MIN_DISPLAY_HEIGHT ||
    rect.width * rect.height < MIN_DISPLAY_AREA
  ) {
    return false
  }
  if (
    image.closest('button,[role="button"]') ||
    EXCLUDED_SEMANTIC_WORDS.test(semanticText(image)) ||
    hasUnsafeTransform(image)
  ) {
    return false
  }
  return true
}

export function discoverDeferredImages(root: ParentNode = document): HTMLImageElement[] {
  return [...root.querySelectorAll('img')].filter(isDeferredPageImage)
}

export function discoverImages(root: ParentNode = document): DiscoveredImage[] {
  return [...root.querySelectorAll('img')]
    .map((image, index) => evaluateImage(image, index))
    .filter(
      (decision): decision is Extract<DiscoveryDecision, { supported: true }> => decision.supported,
    )
    .map((decision) => decision.candidate)
}

/**
 * Recognizes the page-image geometry shared by long-strip webtoons and
 * paginated comics without relying on a publisher, URL, class name, or title.
 * A single ordinary article image is intentionally insufficient.
 */
export function looksLikeSequentialArtReader(root: ParentNode = document): boolean {
  const pages = discoverImages(root).filter(({ element }) => {
    const { naturalWidth: width, naturalHeight: height } = element
    return width >= 500 && height >= 700 && width * height >= 500_000
  })
  if (pages.some(({ element }) => element.naturalHeight >= element.naturalWidth * 2.5)) {
    return true
  }
  if (pages.length < 2) return false
  const widths = pages.map(({ element }) => element.naturalWidth).sort((left, right) => left - right)
  const medianWidth = widths[Math.floor(widths.length / 2)]!
  const consistentlySized = pages.filter(({ element }) => {
    const widthRatio = element.naturalWidth / medianWidth
    return widthRatio >= 0.75 && widthRatio <= 1.25
  })
  return consistentlySized.length >= 2
}

export function visibleFirst(candidates: readonly DiscoveredImage[]): DiscoveredImage[] {
  return [...candidates].sort(
    (left, right) => Number(right.visible) - Number(left.visible) || left.domIndex - right.domIndex,
  )
}

export class ImageDiscovery {
  private readonly candidates = new Map<HTMLImageElement, DiscoveredImage>()
  private mutationObserver: Pick<MutationObserver, 'observe' | 'disconnect'> | undefined
  private intersectionObserver: IntersectionObserverLike | undefined

  constructor(
    private readonly onEvent: (event: DiscoveryEvent) => void,
    private readonly root: Document = document,
    private readonly factories: ObserverFactories = defaultFactories,
  ) {}

  start(): void {
    this.scan()
    this.mutationObserver = this.factories.mutation((mutations) => {
      let needsScan = false
      for (const mutation of mutations) {
        if (
          mutation.type === 'childList' ||
          (mutation.type === 'attributes' && mutation.target instanceof HTMLImageElement)
        ) {
          needsScan = true
          break
        }
      }
      if (needsScan) this.scan()
    })
    this.mutationObserver.observe(this.root.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'sizes', 'style', 'class'],
    })
    this.root.addEventListener('load', this.onLoad, true)
    this.intersectionObserver = this.factories.intersection((entries) => {
      for (const entry of entries) {
        if (!(entry.target instanceof HTMLImageElement)) continue
        const candidate = this.candidates.get(entry.target)
        if (!candidate) continue
        // The observer's 20% root margin is useful as an early prefetch signal,
        // but it must not promote a near-offscreen image to viewport priority.
        const visible = isRectVisible(entry.boundingClientRect)
        if (candidate.visible !== visible) {
          candidate.visible = visible
          this.onEvent({ type: 'visibility', candidate })
        }
      }
    })
    for (const image of this.candidates.keys()) {
      this.intersectionObserver?.observe(image)
    }
  }

  stop(): void {
    this.mutationObserver?.disconnect()
    this.intersectionObserver?.disconnect()
    this.root.removeEventListener('load', this.onLoad, true)
    this.candidates.clear()
  }

  current(): DiscoveredImage[] {
    return visibleFirst([...this.candidates.values()])
  }

  deferred(): HTMLImageElement[] {
    return discoverDeferredImages(this.root)
  }

  completionKey(): string {
    const images = [...this.root.querySelectorAll('img')]
    const indexByImage = new Map(images.map((image, index) => [image, index]))
    const candidates = this.current()
      .map(
        (candidate) =>
          `ready:${candidate.domIndex}:${candidate.sourceUrl}:${candidate.element.naturalWidth}x${candidate.element.naturalHeight}`,
      )
      .sort()
    const deferred = this.deferred()
      .map(
        (image) =>
          `deferred:${indexByImage.get(image) ?? -1}:${image.currentSrc || image.src}:${deferredImageSourceUrl(image) ?? ''}`,
      )
      .sort()
    return [...candidates, ...deferred].join('|')
  }

  private readonly onLoad = (event: Event): void => {
    if (event.target instanceof HTMLImageElement) this.scan()
  }

  private scan(): void {
    const images = [...this.root.querySelectorAll('img')]
    const live = new Set(images)
    for (const [element, candidate] of this.candidates) {
      if (!live.has(element) || !element.isConnected) {
        this.candidates.delete(element)
        this.intersectionObserver?.unobserve(element)
        this.onEvent({ type: 'removed', candidate })
      }
    }
    images.forEach((image, domIndex) => {
      const previous = this.candidates.get(image)
      if (
        previous &&
        (image.closest('[data-hmt-owned="true"]') || image.hasAttribute('data-hmt-original'))
      ) {
        const sourceUrl = image.currentSrc || image.src
        if (sourceUrl && sourceUrl !== previous.sourceUrl) {
          const candidate = {
            ...previous,
            sourceUrl,
            domIndex,
          }
          this.candidates.set(image, candidate)
          this.onEvent({
            type: 'updated',
            candidate,
            previousSourceUrl: previous.sourceUrl,
            previousDomIndex: previous.domIndex,
          })
        } else if (previous.domIndex !== domIndex) {
          const previousDomIndex = previous.domIndex
          previous.domIndex = domIndex
          previous.owner = imageOwner(image)
          this.onEvent({
            type: 'updated',
            candidate: previous,
            previousSourceUrl: previous.sourceUrl,
            previousDomIndex,
          })
        }
        return
      }
      const decision = evaluateImage(image, domIndex)
      if (!decision.supported) {
        if (previous) {
          this.candidates.delete(image)
          this.intersectionObserver?.unobserve(image)
          this.onEvent({ type: 'removed', candidate: previous })
        }
        return
      }
      const candidate = decision.candidate
      if (!previous) {
        this.candidates.set(image, candidate)
        this.intersectionObserver?.observe(image)
        this.onEvent({ type: 'added', candidate })
        return
      }
      candidate.visible = previous.visible
      if (previous.sourceUrl !== candidate.sourceUrl) {
        this.candidates.set(image, candidate)
        this.onEvent({
          type: 'updated',
          candidate,
          previousSourceUrl: previous.sourceUrl,
          previousDomIndex: previous.domIndex,
        })
        return
      }
      const previousDomIndex = previous.domIndex
      previous.owner = candidate.owner
      previous.domIndex = candidate.domIndex
      if (previousDomIndex !== previous.domIndex) {
        this.onEvent({
          type: 'updated',
          candidate: previous,
          previousSourceUrl: previous.sourceUrl,
          previousDomIndex,
        })
      }
    })
  }
}
