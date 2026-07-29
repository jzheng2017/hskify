import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ImageDiscovery,
  discoverDeferredImages,
  discoverImages,
  evaluateImage,
  looksLikeSequentialArtReader,
  visibleFirst,
  type DiscoveryEvent,
  type ObserverFactories,
} from '../../src/discovery/images'
import { loadedImage } from '../helpers/images'

afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

function viewportRect(top: number, bottom: number): DOMRect {
  return {
    x: 0,
    y: top,
    left: 0,
    top,
    right: 720,
    bottom,
    width: 720,
    height: bottom - top,
    toJSON: () => ({}),
  }
}

describe('conservative image discovery', () => {
  it('recognizes sequential-art geometry without publisher-specific markers', () => {
    const articleHero = loadedImage('https://reader.test/article.jpg', 1200, 900)
    document.body.append(articleHero)
    expect(looksLikeSequentialArtReader()).toBe(false)

    articleHero.remove()
    const longStrip = loadedImage('https://reader.test/strip.webp', 900, 16_000, {
      width: 720,
      height: 12_800,
      right: 720,
      bottom: 12_800,
    })
    document.body.append(longStrip)
    expect(looksLikeSequentialArtReader()).toBe(true)

    longStrip.remove()
    document.body.append(
      loadedImage('https://reader.test/page-1.jpg', 800, 1200),
      loadedImage('https://reader.test/page-2.jpg', 840, 1280),
    )
    expect(looksLikeSequentialArtReader()).toBe(true)
  })

  it('accepts a loaded manga-sized image and preserves picture ownership', () => {
    const picture = document.createElement('picture')
    const image = loadedImage()
    picture.append(image)
    document.body.append(picture)

    const decision = evaluateImage(image, 3)
    expect(decision.supported).toBe(true)
    if (decision.supported) {
      expect(decision.candidate.owner).toBe(picture)
      expect(decision.candidate.domIndex).toBe(3)
      expect(decision.candidate.visible).toBe(true)
    }
  })

  it.each([
    [
      'tiny intrinsic dimensions',
      loadedImage('https://reader.test/tiny.png', 64, 64),
      'intrinsic-size',
    ],
    ['avatar semantics', loadedImage(), 'page-control'],
    ['CSS rotation', loadedImage(), 'unsupported-transform'],
  ])('rejects %s', (_label, image, expectedReason) => {
    if (expectedReason === 'page-control') image.className = 'profile-avatar icon'
    if (expectedReason === 'unsupported-transform') image.style.transform = 'rotate(4deg)'
    document.body.append(image)
    expect(evaluateImage(image, 0)).toEqual({
      supported: false,
      reason: expectedReason,
    })
  })

  it('keeps navigation-link images while rejecting actual buttons', () => {
    const link = document.createElement('a')
    const linked = loadedImage('https://reader.test/linked.png')
    link.append(linked)
    const button = document.createElement('button')
    const controlled = loadedImage('https://reader.test/control.png')
    button.append(controlled)
    document.body.append(link, button)

    expect(evaluateImage(linked, 0).supported).toBe(true)
    expect(evaluateImage(controlled, 1)).toEqual({
      supported: false,
      reason: 'page-control',
    })
  })

  it('selects exactly 20 long query-string webtoon pages among 154 site images', () => {
    const cover = loadedImage('https://reader.test/images/cover.png', 1200, 1800, {
      width: 320,
      height: 480,
      right: 320,
      bottom: 480,
    })
    cover.className = 'manga-cover'
    document.body.append(cover)

    for (let index = 0; index < 20; index += 1) {
      const top = index === 0 ? 0 : 17_000 + index * 100
      const page = loadedImage(
        `https://cdn.test/chapter.webp?chapter=synthetic&page=${index}`,
        900,
        16_000,
        {
          top,
          bottom: top + 12_800,
          width: 720,
          height: 12_800,
          right: 720,
        },
      )
      page.dataset.pageIndex = String(index)
      document.body.append(page)
    }

    for (let index = 0; index < 133; index += 1) {
      const avatar = loadedImage(`https://cdn.test/avatar.webp?user=${index}`, 900, 900, {
        width: 48,
        height: 48,
        right: 48,
        bottom: 48,
      })
      avatar.className = 'comment-avatar'
      document.body.append(avatar)
    }

    const selected = discoverImages()
    expect(document.querySelectorAll('img')).toHaveLength(154)
    expect(selected).toHaveLength(20)
    expect(selected.every((candidate) => candidate.element.hasAttribute('data-page-index'))).toBe(
      true,
    )
    expect(selected.every((candidate) => candidate.sourceUrl.includes('.webp?'))).toBe(true)
    expect(selected[0]?.visible).toBe(true)
    expect(selected.slice(1).every((candidate) => !candidate.visible)).toBe(true)

    const responsive = selected[0]?.element
    if (!responsive) throw new Error('Synthetic responsive page missing.')
    responsive.getBoundingClientRect = () =>
      ({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: 465,
        bottom: 8_267,
        width: 465,
        height: 8_267,
        toJSON: () => ({}),
      }) satisfies DOMRect
    expect(evaluateImage(responsive, 1).supported).toBe(true)
  })

  it('orders visible images before offscreen images and then by DOM order', () => {
    const candidates = [
      { ...discoverCandidate(2), visible: false },
      { ...discoverCandidate(5), visible: true },
      { ...discoverCandidate(1), visible: true },
    ]
    expect(visibleFirst(candidates).map((candidate) => candidate.domIndex)).toEqual([1, 5, 2])
  })

  it('tracks large cross-site lazy reader placeholders until their real source loads', () => {
    const webtoon = loadedImage(
      'https://webtoons-static.pstatic.net/image/bg_transparency.png',
      1,
      1,
      { width: 700, height: 1280, right: 700, bottom: 1280 },
    )
    webtoon.className = '_images'
    webtoon.dataset.url = 'https://webtoon-phinf.pstatic.net/episode/page-001.jpg?type=q90'

    const asura = loadedImage(
      'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==',
      1,
      1,
      { width: 800, height: 1200, right: 800, bottom: 1200 },
    )
    asura.className = 'chapter-image'
    asura.dataset.src = 'https://cdn.asura.example/chapter/page-002.webp'

    const thumbnail = loadedImage('https://reader.test/placeholder.png', 1, 1, {
      width: 92,
      height: 87,
      right: 92,
      bottom: 87,
    })
    thumbnail.className = 'episode-thumbnail'
    thumbnail.dataset.url = 'https://reader.test/thumbnail.jpg'
    document.body.append(webtoon, asura, thumbnail)

    expect(discoverImages()).toEqual([])
    expect(discoverDeferredImages()).toEqual([webtoon, asura])

    const resolved = webtoon.dataset.url!
    webtoon.src = resolved
    Object.defineProperties(webtoon, {
      currentSrc: { configurable: true, value: resolved },
      naturalWidth: { configurable: true, value: 700 },
      naturalHeight: { configurable: true, value: 1280 },
    })
    expect(discoverImages().map((candidate) => candidate.element)).toEqual([webtoon])
    expect(discoverDeferredImages()).toEqual([asura])
  })

  it('does not treat a root-margin prefetch intersection as true viewport visibility', () => {
    const events: DiscoveryEvent[] = []
    let intersectionCallback: IntersectionObserverCallback = () => undefined
    const factories: ObserverFactories = {
      mutation: () => ({ observe: vi.fn(), disconnect: vi.fn() }),
      intersection(callback) {
        intersectionCallback = callback
        return {
          observe: vi.fn(),
          unobserve: vi.fn(),
          disconnect: vi.fn(),
        }
      },
    }
    const visibleImage = loadedImage('https://reader.test/visible.png')
    const nearOffscreenImage = loadedImage('https://reader.test/near-offscreen.png')
    document.body.append(visibleImage, nearOffscreenImage)
    const discovery = new ImageDiscovery((event) => events.push(event), document, factories)
    discovery.start()
    events.splice(0)

    intersectionCallback(
      [
        {
          target: visibleImage,
          isIntersecting: true,
          intersectionRatio: 1,
          boundingClientRect: viewportRect(0, 400),
        } as unknown as IntersectionObserverEntry,
        {
          target: nearOffscreenImage,
          isIntersecting: true,
          intersectionRatio: 0.25,
          boundingClientRect: viewportRect(window.innerHeight + 1, window.innerHeight + 401),
        } as unknown as IntersectionObserverEntry,
      ],
      {} as IntersectionObserver,
    )

    expect(events).toHaveLength(1)
    expect(events[0]?.type).toBe('visibility')
    expect(
      discovery.current().map((candidate) => ({
        element: candidate.element,
        visible: candidate.visible,
      })),
    ).toEqual([
      { element: visibleImage, visible: true },
      { element: nearOffscreenImage, visible: false },
    ])

    intersectionCallback(
      [
        {
          target: nearOffscreenImage,
          isIntersecting: true,
          intersectionRatio: 0.25,
          boundingClientRect: viewportRect(window.innerHeight - 1, window.innerHeight + 399),
        } as unknown as IntersectionObserverEntry,
      ],
      {} as IntersectionObserver,
    )

    expect(discovery.current().every((candidate) => candidate.visible)).toBe(true)
    discovery.stop()
  })

  it('publishes same-source DOM-order changes on rescan', () => {
    const events: DiscoveryEvent[] = []
    let mutationCallback: MutationCallback = () => undefined
    const factories: ObserverFactories = {
      mutation(callback) {
        mutationCallback = callback
        return { observe: vi.fn(), disconnect: vi.fn() }
      },
      intersection: () => undefined,
    }
    const first = loadedImage('https://reader.test/first.png')
    const second = loadedImage('https://reader.test/second.png')
    document.body.append(first, second)
    const discovery = new ImageDiscovery((event) => events.push(event), document, factories)
    discovery.start()
    events.splice(0)

    document.body.prepend(second)
    mutationCallback([{ type: 'childList' } as MutationRecord], {} as MutationObserver)

    const updates = events.filter(
      (event): event is Extract<DiscoveryEvent, { type: 'updated' }> => event.type === 'updated',
    )
    expect(
      updates.map((event) => ({
        element: event.candidate.element,
        previousDomIndex: event.previousDomIndex,
        domIndex: event.candidate.domIndex,
        sameSource: event.previousSourceUrl === event.candidate.sourceUrl,
      })),
    ).toEqual([
      { element: second, previousDomIndex: 1, domIndex: 0, sameSource: true },
      { element: first, previousDomIndex: 0, domIndex: 1, sameSource: true },
    ])
    expect(discovery.current().map((candidate) => candidate.element)).toEqual([second, first])
    discovery.stop()
  })

  it('observes lazy additions, visibility, source replacement, and removal', () => {
    const events: DiscoveryEvent[] = []
    let mutationCallback: MutationCallback = () => undefined
    let intersectionCallback: IntersectionObserverCallback = () => undefined
    const factories: ObserverFactories = {
      mutation(callback) {
        mutationCallback = callback
        return { observe: vi.fn(), disconnect: vi.fn() }
      },
      intersection(callback) {
        intersectionCallback = callback
        return {
          observe: vi.fn(),
          unobserve: vi.fn(),
          disconnect: vi.fn(),
        }
      },
    }
    const discovery = new ImageDiscovery((event) => events.push(event), document, factories)
    discovery.start()
    const image = loadedImage()
    document.body.append(image)
    mutationCallback([{ type: 'childList' } as MutationRecord], {} as MutationObserver)
    expect(events.at(-1)?.type).toBe('added')

    intersectionCallback(
      [
        {
          target: image,
          isIntersecting: false,
          intersectionRatio: 0,
          boundingClientRect: viewportRect(10_000, 10_400),
        } as unknown as IntersectionObserverEntry,
      ],
      {} as IntersectionObserver,
    )
    expect(events.at(-1)?.type).toBe('visibility')
    expect(discovery.current()[0]?.visible).toBe(false)

    Object.defineProperty(image, 'currentSrc', {
      configurable: true,
      value: 'https://reader.test/replaced.png',
    })
    mutationCallback(
      [{ type: 'attributes', target: image } as unknown as MutationRecord],
      {} as MutationObserver,
    )
    expect(events.at(-1)?.type).toBe('updated')

    image.remove()
    mutationCallback([{ type: 'childList' } as MutationRecord], {} as MutationObserver)
    expect(events.at(-1)?.type).toBe('removed')
    discovery.stop()
  })
})

function discoverCandidate(domIndex: number) {
  const image = loadedImage(`https://reader.test/${domIndex}.png`)
  document.body.append(image)
  const result = evaluateImage(image, domIndex)
  if (!result.supported) throw new Error('Test candidate was rejected.')
  return result.candidate
}
