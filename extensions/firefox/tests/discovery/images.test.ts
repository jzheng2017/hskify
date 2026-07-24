import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ImageDiscovery,
  discoverImages,
  evaluateImage,
  visibleFirst,
  type DiscoveryEvent,
  type ObserverFactories,
} from '../../src/discovery/images'
import { loadedImage } from '../helpers/images'

afterEach(() => {
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

describe('conservative image discovery', () => {
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
    ['tiny intrinsic dimensions', loadedImage('https://reader.test/tiny.png', 64, 64), 'intrinsic-size'],
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

  it('orders visible images before offscreen images and then by DOM order', () => {
    const candidates = [
      { ...discoverCandidate(2), visible: false },
      { ...discoverCandidate(5), visible: true },
      { ...discoverCandidate(1), visible: true },
    ]
    expect(visibleFirst(candidates).map((candidate) => candidate.domIndex)).toEqual([1, 5, 2])
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
    mutationCallback(
      [{ type: 'childList' } as MutationRecord],
      {} as MutationObserver,
    )
    expect(events.at(-1)?.type).toBe('added')

    intersectionCallback(
      [
        {
          target: image,
          isIntersecting: false,
          intersectionRatio: 0,
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
    mutationCallback(
      [{ type: 'childList' } as MutationRecord],
      {} as MutationObserver,
    )
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
