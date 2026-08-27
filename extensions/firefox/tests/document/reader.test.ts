import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TranslatedText } from '../../src/contracts/browser'
import {
  DocumentReader,
  type DocumentChapter,
  type DocumentStructureItem,
  type IntersectionObserverFactory,
} from '../../src/document'

class TestIntersectionObserver implements IntersectionObserver {
  readonly root = null
  readonly rootMargin = '200px 0px'
  readonly scrollMargin = '0px'
  readonly thresholds = [0]
  readonly observed = new Set<Element>()

  constructor(private readonly callback: IntersectionObserverCallback) {}

  disconnect(): void {
    this.observed.clear()
  }

  observe(target: Element): void {
    this.observed.add(target)
  }

  takeRecords(): IntersectionObserverEntry[] {
    return []
  }

  unobserve(target: Element): void {
    this.observed.delete(target)
  }

  trigger(targets: readonly Element[]): void {
    this.callback(
      targets.map(
        (target) =>
          ({
            target,
            isIntersecting: true,
            intersectionRatio: 1,
            boundingClientRect: target.getBoundingClientRect(),
            intersectionRect: target.getBoundingClientRect(),
            rootBounds: null,
            time: performance.now(),
          }) as IntersectionObserverEntry,
      ),
      this,
    )
  }
}

function fixtureChapter(): { chapter: DocumentChapter; root: HTMLElement } {
  const root = document.createElement('article')
  root.id = 'source-chapter'
  root.className = 'site-prose'
  root.setAttribute('aria-hidden', 'false')
  root.setAttribute('data-site-state', 'ready')
  const structure: DocumentStructureItem[] = []
  const sourceElements = new Map<string, HTMLElement>()
  const kinds = [
    'title',
    'paragraph',
    'blockquote',
    'ordered-list-item',
    'ordered-list-item',
    'caption',
  ] as const
  for (let order = 0; order < kinds.length; order += 1) {
    const kind = kinds[order]!
    const itemId = `block-${order}`
    const text = order === 0 ? 'Chapter One' : `English source block number ${order}.`
    const source = document.createElement(order === 0 ? 'h1' : 'p')
    source.textContent = text
    root.append(source)
    sourceElements.set(itemId, source)
    structure.push({
      type: 'text',
      itemId,
      order,
      kind,
      text,
      ...(kind === 'title' ? { headingLevel: 1 as const } : {}),
    })
    if (order === 4) {
      const liveImage = document.createElement('img')
      liveImage.src = 'https://images.example.test/plate.jpg'
      liveImage.alt = 'Chapter illustration'
      root.append(liveImage)
      structure.push({
        type: 'image',
        itemId: 'image-1',
        order: 100,
        sourceUrl: 'https://images.example.test/plate.jpg',
        alt: 'Chapter illustration',
      })
    }
    if (order === 5) {
      root.append(document.createElement('hr'))
      structure.push({ type: 'separator', itemId: 'separator-1', order: 101 })
    }
  }
  document.body.append(root)
  const blocks = structure
    .filter((item) => item.type === 'text')
    .map((item, itemOrder) => ({
      itemId: item.itemId,
      sourceIndex: 0 as const,
      itemOrder,
      kind:
        item.kind === 'title'
          ? ('heading' as const)
          : item.kind === 'blockquote'
            ? ('dialogue' as const)
            : item.kind === 'caption'
              ? ('caption' as const)
              : ('prose' as const),
      provenance: 'dom' as const,
      text: item.text,
    }))
  return {
    root,
    chapter: {
      sourceRoot: root,
      sourceElements,
      structure,
      snapshot: {
        sourceUrl: 'https://novels.example.test/chapter/1',
        sourceSha256: 'a'.repeat(64),
        title: 'Chapter One',
        characterCount: blocks.reduce((sum, block) => sum + [...block.text].length, 0),
        blocks,
      },
    },
  }
}

function translation(sourceText: string): TranslatedText {
  return {
    sourceText,
    baseChinese: '\u6211\u4eec\u73b0\u5728\u51fa\u53d1\u3002',
    displayedChinese: '\u6211\u4eec\u73b0\u5728\u51fa\u53d1\u3002',
    pinyin: 'w\u01d2 men xi\u00e0n z\u00e0i ch\u016b f\u0101',
    hsk: {
      requestedLevel: 2,
      learningMode: 'natural',
      strictlyValid: false,
      levelCoverage: 0.6,
      aboveLevelTokens: ['\u73b0\u5728'],
      teachingTerms: [
        {
          text: '\u73b0\u5728',
          startChar: 2,
          endChar: 4,
          pinyin: 'xi\u00e0n z\u00e0i',
          definitions: ['now'],
          requiredLevel: 3,
          reason: 'above-level',
        },
      ],
      repairState: 'not-needed',
    },
  }
}

function modeButton(name: string): HTMLButtonElement | undefined {
  const host = document.querySelector<HTMLElement>('[data-hskify-mode-controls="true"]')
  return [...(host?.shadowRoot?.querySelectorAll('button') ?? [])].find(
    (button) => button.textContent === name,
  ) as HTMLButtonElement | undefined
}

describe('document reader', () => {
  beforeEach(() => {
    document.head.replaceChildren()
    document.body.replaceChildren()
  })

  afterEach(() => {
    vi.useRealTimers()
    document.body.replaceChildren()
  })

  it('mounts translated-only placeholders in the connected source without a reader surface', () => {
    const { chapter, root } = fixtureChapter()
    const before = root.innerHTML
    const reader = new DocumentReader(chapter)
    const host = document.querySelector<HTMLElement>('[data-hskify-document-reader="true"]')!

    expect(root.isConnected).toBe(true)
    expect(root.hidden).toBe(false)
    expect(root.hasAttribute('inert')).toBe(false)
    expect(host.dataset.hskifyDocumentReader).toBe('true')
    expect(host.dataset.hskifySourceBlockCount).toBe('6')
    expect(host.dataset.hskifySourceCharacterCount).toBe(String(chapter.snapshot.characterCount))
    expect(root.querySelectorAll('[data-hskify-item-id^="block-"]')).toHaveLength(6)
    expect(chapter.sourceElements.get('block-0')?.textContent).toBe('\u200b')
    expect(root.querySelector('img')?.getAttribute('src')).toBe(
      'https://images.example.test/plate.jpg',
    )
    expect(root.innerHTML).not.toBe(before)

    reader.destroy()
    expect(root.innerHTML).toBe(before)
  })

  it('installs only terminal text once and wires teaching and interaction metadata', () => {
    const { chapter } = fixtureChapter()
    const detach = vi.fn()
    const attach = vi.fn(() => detach)
    const reader = new DocumentReader(chapter, { attachTranslatedText: attach })
    const text = translation('English source block number 1.')

    expect(reader.installBlock('block-1', text)).toBe(true)
    expect(reader.installBlock('block-1', text)).toBe(false)
    const element = chapter.sourceElements.get('block-1')
    expect(element?.textContent).toBe(text.displayedChinese)
    expect(element?.querySelector('.hskify-learning-term')?.textContent).toBe('\u73b0\u5728')
    expect(element?.dataset.hskifyHskLearningMode).toBe('natural')
    expect(element?.dataset.hskifyPinyin).toBe(text.pinyin)
    expect(element?.lang).toBe('zh-CN')
    expect(attach).toHaveBeenCalledTimes(1)
    expect(attach).toHaveBeenCalledWith(element, 'block-1')

    reader.destroy()
    expect(detach).toHaveBeenCalledTimes(1)
  })

  it('withholds failed English blocks from Chinese mode and rejects stale source payloads', () => {
    const { chapter } = fixtureChapter()
    const reader = new DocumentReader(chapter)
    expect(
      reader.preserveBlock('block-2', 'English source block number 2.', 'joined-validation-failed'),
    ).toBe(true)
    expect(reader.preserveBlock('block-2', 'English source block number 2.', 'duplicate')).toBe(
      false,
    )
    const preserved = chapter.sourceElements.get('block-2')
    expect(preserved?.textContent).toBe('\u200b')
    expect(preserved?.textContent).not.toContain('English')
    expect(preserved?.dataset.hskifyState).toBe('preserved')
    expect(() => reader.installBlock('block-3', translation('different source'))).toThrow(
      /Source mismatch/u,
    )
    expect(reader.counts()).toEqual({ translated: 0, preserved: 1, pending: 5 })
    reader.destroy()
  })

  it('uses one shared control and restores exact source attributes in Original and destroy', () => {
    const { chapter, root } = fixtureChapter()
    const originalAttributes = [...root.attributes].map((attribute) => [
      attribute.name,
      attribute.value,
    ])
    const reader = new DocumentReader(chapter)
    const first = chapter.sourceElements.get('block-0')!
    expect(document.querySelectorAll('[data-hskify-mode-controls="true"]')).toHaveLength(1)

    modeButton('Original')?.click()
    expect(first.textContent).toBe('Chapter One')
    expect(first.hasAttribute('data-hskify-item-id')).toBe(false)
    expect([...root.attributes].map((attribute) => [attribute.name, attribute.value])).toEqual(
      originalAttributes,
    )

    modeButton('Chinese')?.click()
    expect(first.textContent).toBe('\u200b')
    expect(first.dataset.hskifyState).toBe('pending')
    expect(root.hidden).toBe(false)
    modeButton('Hold to compare')?.dispatchEvent(
      new Event('pointerdown', { bubbles: true, composed: true }),
    )
    expect(first.textContent).toBe('Chapter One')
    expect(root.getAttribute('aria-hidden')).toBe('false')
    modeButton('Hold to compare')?.dispatchEvent(
      new Event('pointerup', { bubbles: true, composed: true }),
    )
    expect(first.textContent).toBe('\u200b')
    expect(root.getAttribute('aria-hidden')).toBe('false')

    reader.destroy()
    expect([...root.attributes].map((attribute) => [attribute.name, attribute.value])).toEqual(
      originalAttributes,
    )
    expect(document.querySelector('[data-hskify-document-reader]')).toBeNull()
    expect(document.querySelector('[data-hskify-mode-controls="true"]')).toBeNull()
  })

  it('coalesces visible block focus at 100 ms without a scroll geometry handler', () => {
    vi.useFakeTimers()
    const { chapter } = fixtureChapter()
    let observer: TestIntersectionObserver | undefined
    const factory: IntersectionObserverFactory = (callback) => {
      observer = new TestIntersectionObserver(callback)
      return observer
    }
    const focus = vi.fn()
    const geometry = vi.spyOn(Element.prototype, 'getBoundingClientRect')
    const reader = new DocumentReader(
      chapter,
      { onVisibleBlocksChanged: focus },
      { intersectionObserverFactory: factory },
    )
    const first = chapter.sourceElements.get('block-0')!
    const second = chapter.sourceElements.get('block-1')!
    observer?.trigger([second])
    observer?.trigger([first])
    expect(focus).not.toHaveBeenCalled()
    vi.advanceTimersByTime(99)
    expect(focus).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(focus).toHaveBeenCalledTimes(1)
    expect(focus).toHaveBeenCalledWith(['block-0', 'block-1'])

    geometry.mockClear()
    window.dispatchEvent(new Event('scroll'))
    expect(geometry).not.toHaveBeenCalled()
    reader.destroy()
  })

  it('preserves the visible block anchor when switching between source and reader', () => {
    vi.useFakeTimers()
    const { chapter } = fixtureChapter()
    let observer: TestIntersectionObserver | undefined
    const factory: IntersectionObserverFactory = (callback) => {
      observer = new TestIntersectionObserver(callback)
      return observer
    }
    const reader = new DocumentReader(chapter, {}, { intersectionObserverFactory: factory })
    const source = chapter.sourceElements.get('block-1')!
    observer?.trigger([source])
    vi.advanceTimersByTime(100)
    vi.spyOn(source, 'getBoundingClientRect')
      .mockReturnValueOnce({ top: 40 } as DOMRect)
      .mockReturnValueOnce({ top: 115 } as DOMRect)
    const scrollBy = vi.spyOn(window, 'scrollBy').mockImplementation(() => undefined)

    modeButton('Original')?.click()

    expect(scrollBy).toHaveBeenCalledWith(0, 75)
    reader.destroy()
  })

  it('restores and invalidates immediately when the source mutates', async () => {
    const { chapter, root } = fixtureChapter()
    const originalAria = root.getAttribute('aria-hidden')
    const invalidated = vi.fn()
    const reader = new DocumentReader(chapter, { onInvalidated: invalidated })

    root.setAttribute('data-site-state', 'changed')
    chapter.sourceElements.get('block-1')!.textContent = 'The site replaced this paragraph.'

    await vi.waitFor(() => expect(invalidated).toHaveBeenCalledWith('source-mutation'))
    expect(reader.isDestroyed).toBe(true)
    expect(root.hidden).toBe(false)
    expect(root.hasAttribute('inert')).toBe(false)
    expect(root.getAttribute('aria-hidden')).toBe(originalAria)
    expect(root.getAttribute('data-site-state')).toBe('changed')
    expect(document.querySelector('[data-hskify-document-reader]')).toBeNull()
  })

  it('uses the same restoration path when SPA navigation detaches the source root', async () => {
    const { chapter, root } = fixtureChapter()
    const invalidated = vi.fn()
    const reader = new DocumentReader(chapter, { onInvalidated: invalidated })

    root.remove()

    await vi.waitFor(() => expect(invalidated).toHaveBeenCalledWith('source-detached'))
    expect(reader.isDestroyed).toBe(true)
    expect(document.querySelector('[data-hskify-document-reader]')).toBeNull()
    expect(document.querySelector('[data-hskify-mode-controls="true"]')).toBeNull()
  })

  it('leaves the site illustration connected and unchanged', () => {
    const { chapter, root } = fixtureChapter()
    const image = chapter.structure.find((item) => item.type === 'image')
    if (!image || image.type !== 'image') throw new Error('Fixture image missing.')
    const liveImage = root.querySelector('img')
    image.sourceUrl = 'javascript:alert(1)'
    const reader = new DocumentReader(chapter)
    expect(root.querySelector('img')).toBe(liveImage)
    expect(root.querySelector('img')?.src).toBe('https://images.example.test/plate.jpg')
    reader.destroy()
  })
})
