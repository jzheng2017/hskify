import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DiscoveredImage } from '../../src/discovery/images'
import { createFixtureRegions } from '../../src/messaging/fixture-service'
import { SelectableRenderer, type RenderedImage } from '../../src/rendering/renderer'
import type { TextSpeaker } from '../../src/selection/speech'
import { loadedImage, pngHeader } from '../helpers/images'

class TestResizeObserver {
  static instances: TestResizeObserver[] = []
  readonly observe = vi.fn()
  readonly disconnect = vi.fn()
  readonly unobserve = vi.fn()
  constructor(readonly callback: ResizeObserverCallback) {
    TestResizeObserver.instances.push(this)
  }
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

function fixtureRegions() {
  return createFixtureRegions({
    jobId: 'fixture-job',
    sourceSha256: 'a'.repeat(64),
    sourceWidth: 1200,
    sourceHeight: 1800,
  })
}

function shadowOf(rendered: RenderedImage): ShadowRoot {
  const host = [...rendered.wrapper.children].find(
    (element) => element instanceof HTMLElement && element.shadowRoot,
  ) as HTMLElement | undefined
  if (!host?.shadowRoot) throw new Error('Renderer shadow root not found.')
  return host.shadowRoot
}

function controlsHost(): HTMLElement {
  const host = document.querySelector<HTMLElement>('[data-hmt-mode-controls="true"]')
  if (!host?.shadowRoot) throw new Error('Fixed mode controls were not found.')
  return host
}

async function decodeFixturePatch(image: HTMLImageElement): Promise<void> {
  const second = image.dataset.patchId?.endsWith('-2')
  Object.defineProperties(image, {
    complete: { configurable: true, value: true },
    naturalWidth: { configurable: true, value: second ? 432 : 456 },
    naturalHeight: { configurable: true, value: second ? 468 : 396 },
  })
}

function renderer(
  lookup: ConstructorParameters<typeof SelectableRenderer>[0]['lookup'] = async (
    request,
  ) => ({
    selectedText:
      request.interaction === 'selection' ? request.selectedText : '离开',
    tokens: [],
  }),
  decoder = decodeFixturePatch,
  speaker?: TextSpeaker,
): SelectableRenderer {
  return new SelectableRenderer(
    {
      fetchFont: async () => new ArrayBuffer(1),
      lookup,
    },
    TestResizeObserver as unknown as typeof ResizeObserver,
    decoder,
    speaker,
  )
}

async function renderAll(
  image: HTMLImageElement,
  selectedRenderer = renderer(),
  regions = fixtureRegions(),
): Promise<RenderedImage> {
  const rendered = selectedRenderer.begin(candidate(image), {
    jobId: 'fixture-job',
    sourceWidth: 1200,
    sourceHeight: 1800,
  })
  for (const region of regions) {
    await rendered.installRegion(region, pngHeader())
  }
  return rendered
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
  let blob = 0
  vi.spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:fixture-${++blob}`)
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

describe('progressive selectable image renderer', () => {
  it('keeps the live original visible and inserts Chinese only after patch decode', async () => {
    const image = loadedImage()
    const liveListener = vi.fn()
    image.addEventListener('custom-live-listener', liveListener)
    document.body.append(image)
    let releaseDecode!: () => void
    const gate = new Promise<void>((resolve) => {
      releaseDecode = resolve
    })
    const selectedRenderer = renderer(undefined, async (patch) => {
      await gate
      await decodeFixturePatch(patch)
    })
    const rendered = selectedRenderer.begin(candidate(image), {
      jobId: 'fixture-job',
      sourceWidth: 1200,
      sourceHeight: 1800,
    })
    const shadow = shadowOf(rendered)
    const installing = rendered.installRegion(fixtureRegions()[0]!, pngHeader())

    expect(image.isConnected).toBe(true)
    expect(image.style.opacity).toBe('')
    expect(shadow.querySelector('.hmt-patch')).toBeNull()
    expect(shadow.querySelector('.hmt-region')).toBeNull()
    releaseDecode()
    await installing
    expect(rendered.wrapper.style.position).toBe('absolute')
    expect(shadow.querySelector('.hmt-patch')).not.toBeNull()
    expect(shadow.querySelector('.hmt-region')?.textContent).toBe('我们现在要走！')
    image.dispatchEvent(new Event('custom-live-listener'))
    expect(liveListener).toHaveBeenCalledTimes(1)
  })

  it('renders retained Latin names at the measured text size without changing selectable text', async () => {
    const image = loadedImage()
    document.body.append(image)
    const rendered = renderer().begin(candidate(image), {
      jobId: 'fixture-job',
      sourceWidth: 1200,
      sourceHeight: 1800,
    })
    const region = {
      ...fixtureRegions()[0]!,
      displayedChinese: '帝国称它为 SILVER HARBOR。',
      layout: {
        ...fixtureRegions()[0]!.layout,
        suggestedLines: [],
      },
    }

    await rendered.installRegion(region, pngHeader())

    const translated = shadowOf(rendered).querySelector<HTMLElement>('.hmt-region')
    expect(translated?.textContent).toBe(region.displayedChinese)
    expect(translated?.querySelector('.hmt-latin-run')).toBeNull()
  })

  it('lets document scrolling stay compositor-only without scheduling a refit', async () => {
    const image = loadedImage()
    document.body.append(image)
    const rendered = await renderAll(image)
    const animationFrame = vi.spyOn(window, 'requestAnimationFrame')

    document.dispatchEvent(new Event('scroll'))

    expect(animationFrame).not.toHaveBeenCalled()
    expect(rendered.wrapper.style.position).toBe('absolute')
  })

  it('shares one fixed, synchronized mode control across a long multi-image reader', async () => {
    const first = loadedImage()
    const second = loadedImage('https://reader.test/panel-2.png')
    document.body.append(first, second)
    const selectedRenderer = renderer()
    const firstRendered = await renderAll(first, selectedRenderer)
    const firstViewport = shadowOf(firstRendered).querySelector<HTMLElement>('.hmt-viewport')
    const host = controlsHost()
    const controls = host.shadowRoot
    const button = (name: string) =>
      [...(controls?.querySelectorAll('button') ?? [])].find(
        (item) => item.textContent === name,
      )

    expect(document.querySelectorAll('[data-hmt-mode-controls="true"]')).toHaveLength(1)
    expect(host.style.position).toBe('fixed')
    button('Original')?.click()
    expect(firstRendered.currentMode).toBe('original')
    expect(firstViewport?.hidden).toBe(true)

    const secondRendered = await renderAll(second, selectedRenderer)
    const secondViewport = shadowOf(secondRendered).querySelector<HTMLElement>('.hmt-viewport')
    expect(document.querySelectorAll('[data-hmt-mode-controls="true"]')).toHaveLength(1)
    expect(secondRendered.currentMode).toBe('original')
    expect(secondViewport?.hidden).toBe(true)
    expect(first.style.opacity).toBe('')
    expect(second.style.opacity).toBe('')
    button('Chinese')?.click()
    expect(firstViewport?.hidden).toBe(false)
    expect(secondViewport?.hidden).toBe(false)
    button('Hold to compare')?.dispatchEvent(
      new Event('pointerdown', { bubbles: true, composed: true }),
    )
    expect(firstViewport?.hidden).toBe(true)
    expect(secondViewport?.hidden).toBe(true)
    button('Hold to compare')?.dispatchEvent(
      new Event('pointerup', { bubbles: true, composed: true }),
    )
    expect(firstViewport?.hidden).toBe(false)
    expect(secondViewport?.hidden).toBe(false)
    expect(first.isConnected).toBe(true)
    expect(second.isConnected).toBe(true)

    firstRendered.destroy()
    expect(document.querySelectorAll('[data-hmt-mode-controls="true"]')).toHaveLength(1)
    secondRendered.destroy()
    expect(document.querySelector('[data-hmt-mode-controls="true"]')).toBeNull()
  })

  it('underlines only preserved learning terms without changing selectable text', async () => {
    const image = loadedImage()
    document.body.append(image)
    const region = fixtureRegions()[0]!
    region.displayedChinese = '我们现在要走！'
    region.pinyin = 'wǒ men xiàn zài yào zǒu'
    region.hsk = {
      requestedLevel: 2,
      learningMode: 'natural',
      strictlyValid: false,
      levelCoverage: 0.8,
      aboveLevelTokens: ['现在'],
      teachingTerms: [
        {
          text: '现在',
          startChar: 2,
          endChar: 4,
          pinyin: 'xiàn zài',
          definitions: ['now'],
          requiredLevel: 3,
          reason: 'above-level',
        },
      ],
      repairState: 'accepted',
    }
    const rendered = await renderAll(image, renderer(), [region])
    const shadow = shadowOf(rendered)

    const translated = shadow.querySelector<HTMLElement>(`[data-region-id="${region.id}"]`)
    expect(translated?.textContent).toBe('我们现在要走！')
    expect(translated?.querySelector('.hmt-learning-term')?.textContent).toBe('现在')
    expect(translated?.dataset.hskLearningMode).toBe('natural')
    expect(translated?.dataset.hskTeachingTerms).toBe('1')
  })

  it('keeps selection, dictionary pinyin, and Mandarin speech wired to progressive text', async () => {
    const image = loadedImage()
    document.body.append(image)
    const lookup = vi.fn(async (request) => ({
      selectedText:
        request.interaction === 'selection' ? request.selectedText : '离开',
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
        baseChinese: '我们得马上离开！',
        sourceEnglish: 'We have to leave now!',
      },
    }))
    let speaking = false
    const speaker: TextSpeaker = {
      isAvailable: () => true,
      toggle: vi.fn((_text, onStateChange) => {
        speaking = !speaking
        onStateChange(
          speaking ? 'speaking' : 'idle',
          speaking
            ? {
                name: 'Microsoft Yunxi',
                lang: 'zh-CN',
                localService: true,
              }
            : undefined,
        )
      }),
      stop: vi.fn(() => {
        speaking = false
      }),
    }
    const rendered = await renderAll(image, renderer(lookup, decodeFixturePatch, speaker))
    const shadow = shadowOf(rendered)
    const region = shadow.querySelector<HTMLElement>('.hmt-region')
    if (!region) throw new Error('Fixture region missing.')
    const range = document.createRange()
    range.selectNodeContents(region)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    region.dispatchEvent(new Event('mouseup', { bubbles: true, composed: true }))

    await vi.waitFor(() => expect(lookup).toHaveBeenCalled())
    const popover = shadow.querySelector<HTMLElement>('.hmt-lookup')
    await vi.waitFor(() => expect(popover?.textContent).toContain('lí kāi'))
    expect(popover?.textContent).toContain('We have to leave now!')
    const speak = popover?.querySelector<HTMLButtonElement>('.hmt-speak')
    speak?.click()
    expect(speaker.toggle).toHaveBeenCalledWith('我们现在要走！', expect.any(Function))
    expect(speak?.textContent).toBe('Stop')
    expect(speak?.dataset.hmtVoiceName).toBe('Microsoft Yunxi')
    expect(speak?.dataset.hmtVoiceLang).toBe('zh-CN')
    expect(speak?.dataset.hmtVoiceLocalService).toBe('true')
  })

  it('forwards unselected primary clicks to reader navigation', async () => {
    const link = document.createElement('a')
    link.href = '#next'
    const image = loadedImage()
    link.append(image)
    document.body.append(link)
    const navigated = vi.fn((event: Event) => event.preventDefault())
    const imageClicked = vi.fn()
    link.addEventListener('click', navigated)
    image.addEventListener('click', imageClicked)
    const rendered = await renderAll(image)
    shadowOf(rendered).querySelector<HTMLElement>('.hmt-region')?.click()
    expect(navigated).toHaveBeenCalledTimes(1)
    expect(imageClicked).toHaveBeenCalledTimes(1)
  })

  it('uses measured binary search and leaves no scroll overflow', async () => {
    const image = loadedImage()
    document.body.append(image)
    const rendered = await renderAll(image)
    const region = shadowOf(rendered).querySelector<HTMLElement>('.hmt-region')
    if (!region) throw new Error('Fixture region missing.')
    Object.defineProperties(region, {
      clientWidth: { configurable: true, value: 100 },
      clientHeight: { configurable: true, value: 40 },
      scrollWidth: {
        configurable: true,
        get: () => Number.parseFloat(region.style.fontSize || '0') * 12,
      },
      scrollHeight: {
        configurable: true,
        get: () => Number.parseFloat(region.style.fontSize || '0') * 3,
      },
    })
    rendered.refit()
    expect(region.scrollWidth).toBeLessThanOrEqual(region.clientWidth + 0.5)
    expect(region.scrollHeight).toBeLessThanOrEqual(region.clientHeight + 0.5)
  })

  it('never installs text for a corrupt patch and restores the exact original node', async () => {
    const before = document.createElement('span')
    const image = loadedImage()
    image.srcset =
      'https://reader.test/panel-small.png 480w, https://reader.test/panel.png 1200w'
    image.sizes = '(max-width: 800px) 100vw, 800px'
    image.className = 'webtoon-page preserved-class'
    image.setAttribute('style', 'display: block; width: 100%; height: auto;')
    image.setAttribute('data-reader-page', '7')
    const after = document.createElement('span')
    document.body.append(before, image, after)
    const exactHtml = document.body.innerHTML
    const exactChildren = [...document.body.childNodes]
    const rendered = renderer(undefined, async () => {
      throw new Error('corrupt patch')
    }).begin(candidate(image), {
      jobId: 'fixture-job',
      sourceWidth: 1200,
      sourceHeight: 1800,
    })
    expect(image.parentNode).toBe(document.body)
    expect(document.body.children[1]).toBe(image)
    expect(rendered.wrapper.parentNode).toBe(document.body)
    await expect(
      rendered.installRegion(fixtureRegions()[0]!, pngHeader()),
    ).rejects.toMatchObject({ code: 'PATCH_DECODE_FAILED' })
    expect(shadowOf(rendered).querySelector('.hmt-region')).toBeNull()
    expect(image.style.opacity).toBe('')
    rendered.destroy()
    rendered.destroy()
    expect(document.body.innerHTML).toBe(exactHtml)
    expect([...document.body.childNodes]).toEqual(exactChildren)
    expect(document.body.children[1]).toBe(image)
    expect(image.nextSibling).toBe(after)
    expect(image.getAttribute('src')).toBe('https://reader.test/panel.png')
    expect(image.getAttribute('srcset')).toBe(
      'https://reader.test/panel-small.png 480w, https://reader.test/panel.png 1200w',
    )
    expect(image.getAttribute('sizes')).toBe('(max-width: 800px) 100vw, 800px')
    expect(image.className).toBe('webtoon-page preserved-class')
    expect(image.getAttribute('style')).toBe('display: block; width: 100%; height: auto;')
    expect(image.hasAttribute('data-hmt-original')).toBe(false)
    expect(URL.revokeObjectURL).toHaveBeenCalled()
  })
})
