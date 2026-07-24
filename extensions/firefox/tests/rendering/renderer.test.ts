import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DiscoveredImage } from '../../src/discovery/images'
import { createFixtureResult } from '../../src/messaging/fixture-service'
import { SelectableRenderer } from '../../src/rendering/renderer'
import { loadedImage, pngHeader } from '../helpers/images'

class TestResizeObserver {
  static instances: TestResizeObserver[] = []
  readonly observe = vi.fn()
  readonly disconnect = vi.fn()
  constructor(readonly callback: ResizeObserverCallback) {
    TestResizeObserver.instances.push(this)
  }
  unobserve = vi.fn()
  trigger(): void {
    this.callback([], this as unknown as ResizeObserver)
  }
}

function candidate(image: HTMLImageElement): DiscoveredImage {
  return {
    element: image,
    owner: image,
    sourceUrl: image.currentSrc,
    domIndex: 0,
    visible: true,
  }
}

function fixturePayload() {
  return {
    result: createFixtureResult({
      jobId: 'fixture-job',
      sourceSha256: 'a'.repeat(64),
      sourceWidth: 1200,
      sourceHeight: 1800,
    }),
    cleanImage: pngHeader(1, 1),
  }
}

function shadowOf(wrapper: HTMLElement): ShadowRoot {
  const host = [...wrapper.children].find(
    (element) => element instanceof HTMLElement && element.shadowRoot,
  ) as HTMLElement | undefined
  if (!host?.shadowRoot) throw new Error('Renderer shadow root not found.')
  return host.shadowRoot
}

beforeEach(() => {
  TestResizeObserver.instances = []
  document.body.replaceChildren()
  if (!URL.createObjectURL) {
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: () => 'blob:fixture',
    })
  }
  if (!URL.revokeObjectURL) {
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: () => undefined,
    })
  }
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fixture')
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
  if (!Range.prototype.getBoundingClientRect) {
    Object.defineProperty(Range.prototype, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({
        left: 20,
        right: 80,
        top: 20,
        bottom: 40,
        width: 60,
        height: 20,
      }),
    })
  }
})

afterEach(() => {
  window.getSelection()?.removeAllRanges()
  document.body.replaceChildren()
  vi.restoreAllMocks()
})

describe('selectable image renderer', () => {
  it('delays wrapping until render, moves the live image, and creates only real Chinese text', async () => {
    const image = loadedImage()
    const clickListener = vi.fn()
    image.addEventListener('custom-live-listener', clickListener)
    document.body.append(image)
    const renderer = new SelectableRenderer(
      {
        fetchFont: async () => new ArrayBuffer(0),
        lookup: async (request) => ({
          selectedText: request.selectedText,
          tokens: [],
        }),
      },
      TestResizeObserver as unknown as typeof ResizeObserver,
    )
    expect(image.parentElement).toBe(document.body)
    const rendered = await renderer.render(candidate(image), fixturePayload())
    expect(rendered.wrapper.contains(image)).toBe(true)
    image.dispatchEvent(new Event('custom-live-listener'))
    expect(clickListener).toHaveBeenCalledTimes(1)

    const shadow = shadowOf(rendered.wrapper)
    const regions = [...shadow.querySelectorAll<HTMLElement>('.hmt-region')]
    expect(regions.map((region) => region.textContent)).toEqual([
      '我们现在要走！',
      '等我！',
    ])
    expect(shadow.textContent).not.toContain('We have to leave now!')
    expect(regions[0]?.lang).toBe('zh-CN')
    expect(regions[0]?.classList.contains('hmt-region')).toBe(true)
    expect(regions[0]?.style.left).toBe('18%')
    expect(image.style.opacity).toBe('0')
  })

  it('supports original, Chinese, and press-to-compare modes', async () => {
    const image = loadedImage()
    document.body.append(image)
    const rendered = await new SelectableRenderer(
      {
        fetchFont: async () => new ArrayBuffer(0),
        lookup: async (request) => ({ selectedText: request.selectedText, tokens: [] }),
      },
      TestResizeObserver as unknown as typeof ResizeObserver,
    ).render(candidate(image), fixturePayload())
    const shadow = shadowOf(rendered.wrapper)
    const buttons = [...shadow.querySelectorAll('button')]
    const original = buttons.find((button) => button.textContent === 'Original')
    const chinese = buttons.find((button) => button.textContent === 'Chinese')
    const compare = buttons.find((button) => button.textContent === 'Hold to compare')
    original?.click()
    expect(rendered.currentMode).toBe('original')
    expect(image.style.opacity).toBe('')
    chinese?.click()
    expect(rendered.currentMode).toBe('chinese')
    expect(image.style.opacity).toBe('0')
    compare?.dispatchEvent(new Event('pointerdown', { bubbles: true, composed: true }))
    expect(image.style.opacity).toBe('')
    compare?.dispatchEvent(new Event('pointerup', { bubbles: true, composed: true }))
    expect(image.style.opacity).toBe('0')
  })

  it('preserves normal link clicks but prevents a click produced by text selection', async () => {
    const link = document.createElement('a')
    link.href = '#next'
    const image = loadedImage()
    link.append(image)
    document.body.append(link)
    const navigated = vi.fn((event: Event) => event.preventDefault())
    link.addEventListener('click', navigated)
    const rendered = await new SelectableRenderer(
      {
        fetchFont: async () => new ArrayBuffer(0),
        lookup: async (request) => ({ selectedText: request.selectedText, tokens: [] }),
      },
      TestResizeObserver as unknown as typeof ResizeObserver,
    ).render(candidate(image), fixturePayload())
    const region = shadowOf(rendered.wrapper).querySelector<HTMLElement>('.hmt-region')
    region?.click()
    expect(navigated).toHaveBeenCalledTimes(1)

    const range = document.createRange()
    range.selectNodeContents(region as HTMLElement)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    region?.click()
    expect(navigated).toHaveBeenCalledTimes(1)
  })

  it('copies exactly selected Chinese and opens the local lookup popover', async () => {
    const image = loadedImage()
    document.body.append(image)
    const lookup = vi.fn(async (request) => ({
      selectedText: request.selectedText,
      tokens: [
        {
          simplified: '离开',
          pinyin: 'lí kāi',
          definitions: ['leave', 'depart'],
          hskLevel: 2 as const,
          properName: false,
        },
      ],
      region: {
        displayedChinese: '我们现在要走！',
        faithfulChinese: '我们得马上离开！',
        sourceEnglish: 'We have to leave now!',
      },
    }))
    const rendered = await new SelectableRenderer(
      { fetchFont: async () => new ArrayBuffer(0), lookup },
      TestResizeObserver as unknown as typeof ResizeObserver,
    ).render(candidate(image), fixturePayload())
    const shadow = shadowOf(rendered.wrapper)
    const region = shadow.querySelector<HTMLElement>('.hmt-region')
    const text = region?.firstChild
    if (!region || !text) throw new Error('Fixture region missing.')
    const range = document.createRange()
    range.setStart(text, 0)
    range.setEnd(text, 2)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)

    const clipboard = { setData: vi.fn() }
    const copy = new Event('copy', { bubbles: true, composed: true, cancelable: true })
    Object.defineProperty(copy, 'clipboardData', { value: clipboard })
    region.dispatchEvent(copy)
    expect(clipboard.setData).toHaveBeenCalledWith('text/plain', '我们')

    region.dispatchEvent(new Event('mouseup', { bubbles: true, composed: true }))
    await vi.waitFor(() => expect(lookup).toHaveBeenCalled())
    const popover = shadow.querySelector<HTMLElement>('.hmt-lookup')
    await vi.waitFor(() => expect(popover?.textContent).toContain('lí kāi'))
    expect(popover?.textContent).toContain('We have to leave now!')
  })

  it('refits on resize and restores the exact original node', async () => {
    const before = document.createElement('span')
    const image = loadedImage()
    const after = document.createElement('span')
    document.body.append(before, image, after)
    const restored = vi.fn()
    const rendered = await new SelectableRenderer(
      {
        fetchFont: async () => new ArrayBuffer(0),
        lookup: async (request) => ({ selectedText: request.selectedText, tokens: [] }),
        onRestore: restored,
      },
      TestResizeObserver as unknown as typeof ResizeObserver,
    ).render(candidate(image), fixturePayload())
    expect(TestResizeObserver.instances[0]?.observe).toHaveBeenCalledWith(image)
    TestResizeObserver.instances[0]?.trigger()
    rendered.destroy()
    expect(document.body.children[1]).toBe(image)
    expect(image.nextSibling).toBe(after)
    expect(image.hasAttribute('data-hmt-original')).toBe(false)
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fixture')
    expect(restored).toHaveBeenCalled()
  })
})
