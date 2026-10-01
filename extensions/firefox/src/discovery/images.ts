const MIN_NATURAL_WIDTH = 320
const MIN_NATURAL_HEIGHT = 240
const MIN_DISPLAY_WIDTH = 180
const MIN_DISPLAY_HEIGHT = 140
const MIN_DISPLAY_AREA = 36_000
const DEFERRED_SOURCE_ATTRIBUTE = /^data-(?:.*(?:src|url|image|original).*)$/i

import type { DiscoveredSurface } from './surfaces'

const EXCLUDED_SURFACE =
  '[data-hskify-owned],button,[role="button"],nav,aside,form,[hidden],[aria-hidden="true"]'

export function eligibleSurfaceElement(
  element: Element,
  cache?: WeakMap<Element, boolean>,
): boolean {
  if (!cache) return !element.closest(EXCLUDED_SURFACE)
  const existing = cache.get(element)
  if (existing !== undefined) return existing
  if (element.matches(EXCLUDED_SURFACE)) {
    cache.set(element, false)
    return false
  }
  const parent = element.parentElement,
    inherited = parent ? cache.get(parent) : true
  if (inherited !== undefined) {
    cache.set(element, inherited)
    return inherited
  }
  const pending: Element[] = []
  let current: Element | null = element
  while (current && !cache.has(current)) {
    pending.push(current)
    if (current.matches(EXCLUDED_SURFACE)) {
      cache.set(current, false)
      break
    }
    current = current.parentElement
  }
  const eligible = !current || cache.get(current) === true
  for (const ancestor of pending) cache.set(ancestor, eligible)
  return eligible
}

export type ImageOwner = HTMLImageElement | HTMLPictureElement

export type DiscoveredImage = DiscoveredSurface & {
  kind: 'image'
  element: HTMLImageElement
  owner: ImageOwner
}

export type DiscoveryDecision =
  | { supported: true; candidate: DiscoveredImage }
  | { supported: false; reason: string }

function imageOwner(image: HTMLImageElement): ImageOwner {
  const parent = image.parentElement
  return parent instanceof HTMLPictureElement ? parent : image
}

function normalizedImageUrl(value: string, ownerDocument: Document = document): string | undefined {
  try {
    const url = new URL(value, ownerDocument.baseURI)
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
    const normalized = normalizedImageUrl(value, image.ownerDocument)
    if (normalized) return normalized
  }
  return undefined
}

function isRendered(image: HTMLImageElement, rect: DOMRect): boolean {
  const style = image.ownerDocument.defaultView?.getComputedStyle(image) ?? getComputedStyle(image)
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
  if (image.closest('[data-hskify-owned="true"]')) {
    return { supported: false, reason: 'owned-by-extension' }
  }
  if (!eligibleSurfaceElement(image)) return { supported: false, reason: 'page-control' }
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
  if (image.closest('button,[role="button"]')) {
    return { supported: false, reason: 'page-control' }
  }
  return {
    supported: true,
    candidate: {
      id: `image:${domIndex}:${sourceUrl}`,
      kind: 'image',
      element: image,
      owner: imageOwner(image),
      sourceUrl,
      sourceWidth: image.naturalWidth,
      sourceHeight: image.naturalHeight,
      domIndex,
      visible: isRectVisible(rect, image.ownerDocument.defaultView ?? window),
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
  if (image.closest('button,[role="button"]')) {
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
  const widths = pages
    .map(({ element }) => element.naturalWidth)
    .sort((left, right) => left - right)
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
