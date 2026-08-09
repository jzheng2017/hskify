import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { JobUpdate } from '../../src/contracts/browser'
import type { DiscoveredImage, DiscoveryEvent } from '../../src/discovery/images'
import type { VisibleFirstQueue } from '../../src/discovery/queue'
import { RuntimeMessageError } from '../../src/messaging/messages'
import { ImageChapterMode } from '../../src/page/controller'
import { SelectableRenderer, type RenderedImage } from '../../src/rendering/renderer'
import { loadedImage } from '../helpers/images'

type ControllerInternals = {
  renderer: SelectableRenderer
  discovery: {
    scan(): void
  }
  rendered: Map<HTMLImageElement, RenderedImage>
  processed: Set<HTMLImageElement>
  scope: 'visible' | 'all' | undefined
  queue: VisibleFirstQueue<unknown>
  queueIds: Map<HTMLImageElement, string>
  chapterSourceOrder: number[]
  canonicalSourceIndex(candidate: DiscoveredImage): number
  establishCanonicalPageOrder(candidates: readonly DiscoveredImage[]): void
  sourceSnapshot(candidate: DiscoveredImage): {
    generation: number
    pageSessionId: string
    navigationUrl: string
    sourceUrl: string
    naturalWidth: number
    naturalHeight: number
  }
  assertCurrent(
    candidate: DiscoveredImage,
    snapshot: {
      generation: number
      pageSessionId: string
      navigationUrl: string
      sourceUrl: string
      naturalWidth: number
      naturalHeight: number
    },
    signal: AbortSignal,
  ): void
  onDiscovery(event: DiscoveryEvent): void
  checkNavigation(): void
}

type ChapterFixture = {
  chapter: HTMLElement
  picture: HTMLPictureElement
  first: HTMLImageElement
  second: HTMLImageElement
}

function candidate(image: HTMLImageElement, domIndex: number): DiscoveredImage {
  return {
    // Discovery identities are element-bound; a DOM reorder must not create a
    // second logical page in the chapter stream.
    id: `image-surface:${image.dataset.page || image.currentSrc || image.src}`,
    kind: 'image',
    element: image,
    owner: image.parentElement instanceof HTMLPictureElement ? image.parentElement : image,
    sourceUrl: image.currentSrc || image.src,
    sourceWidth: image.naturalWidth,
    sourceHeight: image.naturalHeight,
    domIndex,
    visible: true,
  }
}

function fixture(): ChapterFixture {
  const chapter = document.createElement('main')
  chapter.id = 'chapter'
  chapter.setAttribute('aria-label', 'Reader chapter')

  const picture = document.createElement('picture')
  picture.className = 'reader-picture preserved'
  picture.setAttribute('style', 'display: block; margin: 0px;')
  const source = document.createElement('source')
  source.srcset = 'https://reader.test/page-1.avif 1x'
  source.sizes = '(max-width: 800px) 100vw, 800px'
  const first = loadedImage('https://reader.test/page-1.webp')
  first.srcset = 'https://reader.test/page-1-small.webp 480w, https://reader.test/page-1.webp 1200w'
  first.sizes = '(max-width: 800px) 100vw, 800px'
  first.className = 'webtoon-page first-page'
  first.setAttribute('style', 'display: block; width: 100%; height: auto;')
  first.setAttribute('data-page', '1')
  first.setAttribute('fetchpriority', 'high')
  picture.append(source, first)

  const separator = document.createElement('span')
  separator.className = 'chapter-separator'
  separator.textContent = 'between'

  const second = loadedImage('https://reader.test/page-2.webp')
  second.srcset = 'https://reader.test/page-2.webp 1200w'
  second.sizes = '100vw'
  second.className = 'webtoon-page second-page'
  second.setAttribute('style', 'display: block; width: 75%; margin: 0px auto;')
  second.setAttribute('data-page', '2')

  chapter.append(
    document.createTextNode('\n  before\n  '),
    picture,
    document.createTextNode('\n  '),
    separator,
    document.createComment('preserved-reader-boundary'),
    document.createTextNode('\n  '),
    second,
    document.createTextNode('\n  after\n'),
  )
  document.body.append(chapter)
  return { chapter, picture, first, second }
}

function addTrackedOverlay(
  controller: ImageChapterMode,
  image: HTMLImageElement,
  domIndex: number,
  processed: boolean,
): void {
  const internals = controller as unknown as ControllerInternals
  const rendered = internals.renderer.begin(candidate(image, domIndex), {
    jobId: `job-${domIndex}`,
    sourceWidth: image.naturalWidth,
    sourceHeight: image.naturalHeight,
  })
  const host = [...rendered.wrapper.children].find(
    (element) => element instanceof HTMLElement && element.shadowRoot,
  )
  if (!(host instanceof HTMLElement) || !host.shadowRoot) {
    throw new Error('Renderer shadow root was not created.')
  }
  const patch = document.createElement('img')
  patch.className = 'hskify-patch'
  patch.dataset.hskifyPatchId = `patch-${domIndex}`
  const text = document.createElement('span')
  text.className = 'hskify-region'
  text.dataset.hskifyItemId = `region-${domIndex}`
  text.textContent = '完整文本'
  host.shadowRoot.append(patch, text)

  internals.rendered.set(image, rendered)
  if (processed) internals.processed.add(image)
}

function expectExactChapter(
  fixture: ChapterFixture,
  expectedHtml: string,
  expectedChildren: readonly ChildNode[],
): void {
  expect(fixture.chapter.innerHTML).toBe(expectedHtml)
  expect([...fixture.chapter.childNodes]).toEqual(expectedChildren)
  expect([...fixture.chapter.querySelectorAll('img')]).toEqual([fixture.first, fixture.second])
  expect(fixture.first.parentElement).toBe(fixture.picture)
  expect(fixture.picture.nextSibling).toBe(expectedChildren[2])
  expect(fixture.second.previousSibling).toBe(expectedChildren[5])
  expect(fixture.chapter.querySelector('[data-hskify-owned], [data-hskify-original]')).toBeNull()
  expect(fixture.chapter.querySelector('.hskify-wrapper')).toBeNull()
  expect(document.querySelector('[data-hskify-mode-controls="true"]')).toBeNull()
}

function installJobLifecycle(failuresBeforeSuccess: number): {
  submitCount(): number
} {
  let submitted = 0
  const sendMessage = vi.mocked(browser.runtime.sendMessage)
  sendMessage.mockImplementation(async (raw: unknown) => {
    const message = raw as Record<string, unknown>
    const type = String(message.type)
    if (type === 'jobs:recover') {
      return { ok: true, value: [] }
    }
    if (type === 'job:submit-image') {
      submitted += 1
      return {
        ok: true,
        value: {
          jobId: `job-${submitted}`,
          kind: 'image',
          clientImageId: `image-${submitted}`,
          sourceSha256: 'a'.repeat(64),
          sourceUrl: message.imageUrl,
          sourceWidth: message.naturalWidth,
          sourceHeight: message.naturalHeight,
          sourceIndex: message.sourceIndex,
          acknowledgedSequence: 0,
        },
      }
    }
    if (type === 'job:updates') {
      const attempt = Number(String(message.jobId).split('-').at(-1))
      const update =
        attempt <= failuresBeforeSuccess
          ? {
              sequence: 1,
              type: 'failed',
              code: 'TEMPORARY_PIPELINE_FAILURE',
              message: 'Temporary fixture failure',
              retryable: true,
            }
          : {
              sequence: 1,
              type: 'complete',
              translatedCount: 0,
              preservedCount: 0,
              message: 'Complete',
            }
      return {
        ok: true,
        value: {
          jobId: message.jobId,
          nextSequence: 1,
          updates: [update],
        },
      }
    }
    return { ok: true, value: undefined }
  })
  return { submitCount: () => submitted }
}

function installUpdateLifecycle(updates: readonly JobUpdate[]): void {
  const sendMessage = vi.mocked(browser.runtime.sendMessage)
  sendMessage.mockImplementation(async (raw: unknown) => {
    const message = raw as Record<string, unknown>
    switch (String(message.type)) {
      case 'jobs:recover':
        return { ok: true, value: [] }
      case 'job:submit-image':
        return {
          ok: true,
          value: {
            jobId: 'job-preservation',
            kind: 'image',
            clientImageId: 'image-preservation',
            sourceSha256: 'a'.repeat(64),
            sourceUrl: message.imageUrl,
            sourceWidth: message.naturalWidth,
            sourceHeight: message.naturalHeight,
            sourceIndex: message.sourceIndex,
            acknowledgedSequence: 0,
          },
        }
      case 'job:updates':
        return {
          ok: true,
          value: {
            jobId: 'job-preservation',
            nextSequence: updates.at(-1)?.sequence ?? 0,
            updates,
          },
        }
      default:
        return { ok: true, value: undefined }
    }
  })
}

function renderedShadowRoot(controller: ImageChapterMode, image: HTMLImageElement): ShadowRoot {
  const rendered = (controller as unknown as ControllerInternals).rendered.get(image)
  const host = [...(rendered?.wrapper.children ?? [])].find(
    (element) => element instanceof HTMLElement && element.shadowRoot,
  )
  if (!(host instanceof HTMLElement) || !host.shadowRoot) {
    throw new Error('Renderer shadow root was not created.')
  }
  return host.shadowRoot
}

beforeEach(() => {
  document.body.replaceChildren()
  sessionStorage.clear()
  vi.stubGlobal('browser', {
    runtime: {
      sendMessage: vi.fn(async () => ({ ok: true, value: undefined })),
    },
  })
})
afterEach(() => {
  document.documentElement
    .querySelectorAll('[data-hskify-owned]')
    .forEach((element) => element.remove())
  document.body.replaceChildren()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('page controller terminal restoration', () => {
  it('fails a retryable image once without creating hidden duplicate work', async () => {
    const image = loadedImage('https://reader.test/retry-page.webp')
    document.body.append(image)
    const lifecycle = installJobLifecycle(Number.POSITIVE_INFINITY)
    const controller = new ImageChapterMode()

    await controller.start('all', 3, 'natural', 'ltr')
    await vi.waitFor(
      () =>
        expect(controller.snapshot()).toMatchObject({
          state: 'failed',
          current: 0,
          total: 1,
        }),
      { timeout: 3_000 },
    )
    expect(lifecycle.submitCount()).toBe(1)
    controller.destroy()
  })

  it('preserves known artwork without adding a visible or interactive overlay', async () => {
    const image = loadedImage('https://reader.test/credits.webp')
    document.body.append(image)
    installUpdateLifecycle([
      {
        sequence: 1,
        type: 'imageRegionPreserved',
        region: {
          itemId: 'credits',
          itemOrder: 0,
          textPolygon: [
            { x: 0.1, y: 0.1 },
            { x: 0.4, y: 0.1 },
            { x: 0.4, y: 0.2 },
            { x: 0.1, y: 0.2 },
          ],
          sourceText: '',
          confidence: 0.99,
          reason: 'non-story artwork',
        },
      },
      {
        sequence: 2,
        type: 'complete',
        translatedCount: 0,
        preservedCount: 1,
        message: 'Complete',
      },
    ])
    const controller = new ImageChapterMode()

    await controller.start('all', 3, 'natural', 'ltr')
    await vi.waitFor(() => expect(controller.snapshot().state).toBe('complete'))

    expect(renderedShadowRoot(controller, image).querySelector('.hskify-region')).toBeNull()
    controller.destroy()
  })

  it('keeps an explicit source notice for genuinely unreadable story text', async () => {
    const image = loadedImage('https://reader.test/unreadable.webp')
    document.body.append(image)
    installUpdateLifecycle([
      {
        sequence: 1,
        type: 'imageRegionPreserved',
        region: {
          itemId: 'uncertain-dialogue',
          itemOrder: 0,
          textPolygon: [
            { x: 0.1, y: 0.1 },
            { x: 0.4, y: 0.1 },
            { x: 0.4, y: 0.2 },
            { x: 0.1, y: 0.2 },
          ],
          sourceText: 'What did she say?',
          confidence: 0.31,
          reason: 'OCR views disagreed',
        },
      },
      {
        sequence: 2,
        type: 'complete',
        translatedCount: 0,
        preservedCount: 1,
        message: 'Complete',
      },
    ])
    const controller = new ImageChapterMode()

    await controller.start('all', 3, 'natural', 'ltr')
    await vi.waitFor(() => expect(controller.snapshot().state).toBe('complete'))

    const notice = renderedShadowRoot(controller, image).querySelector('.hskify-source-notice')
    expect(notice?.getAttribute('data-hskify-source-text')).toBe('What did she say?')
    controller.destroy()
  })

  it('does not publish complete while a cross-site lazy chapter image is unresolved', async () => {
    const ready = loadedImage('https://reader.test/ready-page.webp')
    const deferred = loadedImage('https://reader.test/transparent-placeholder.png', 1, 1, {
      width: 800,
      height: 1280,
      right: 800,
      bottom: 1280,
    })
    deferred.className = 'chapter-image'
    deferred.dataset.url = 'https://cdn.reader.test/deferred-page.webp'
    document.body.append(ready, deferred)
    const lifecycle = installJobLifecycle(0)
    const controller = new ImageChapterMode()
    const internals = controller as unknown as ControllerInternals

    await controller.start('all', 3, 'natural', 'ltr')
    await vi.waitFor(
      () =>
        expect(controller.snapshot()).toMatchObject({
          state: 'running',
          current: 1,
          total: 2,
        }),
      { timeout: 3_000 },
    )
    expect(lifecycle.submitCount()).toBe(1)

    const resolvedSource = deferred.dataset.url!
    deferred.src = resolvedSource
    Object.defineProperties(deferred, {
      currentSrc: { configurable: true, value: resolvedSource },
      naturalWidth: { configurable: true, value: 800 },
      naturalHeight: { configurable: true, value: 1280 },
    })
    internals.discovery.scan()

    await vi.waitFor(
      () =>
        expect(controller.snapshot()).toMatchObject({
          state: 'complete',
          current: 2,
          total: 2,
        }),
      { timeout: 3_000 },
    )
    expect(lifecycle.submitCount()).toBe(2)
    controller.destroy()
  })

  it('waits instead of failing when the chapter initially contains only lazy placeholders', async () => {
    const deferred = loadedImage('https://reader.test/transparent-placeholder.png', 1, 1, {
      width: 800,
      height: 1280,
      right: 800,
      bottom: 1280,
    })
    deferred.className = 'chapter-image'
    deferred.dataset.lazySourceUrl = 'https://cdn.reader.test/deferred-page.webp'
    document.body.append(deferred)
    const lifecycle = installJobLifecycle(0)
    const controller = new ImageChapterMode()
    const internals = controller as unknown as ControllerInternals

    await expect(controller.start('all', 3, 'natural', 'ltr')).resolves.toMatchObject({
      state: 'running',
      current: 0,
      total: 1,
    })
    expect(lifecycle.submitCount()).toBe(0)

    const resolvedSource = deferred.dataset.lazySourceUrl!
    deferred.src = resolvedSource
    Object.defineProperties(deferred, {
      currentSrc: { configurable: true, value: resolvedSource },
      naturalWidth: { configurable: true, value: 800 },
      naturalHeight: { configurable: true, value: 1280 },
    })
    internals.discovery.scan()

    await vi.waitFor(
      () =>
        expect(controller.snapshot()).toMatchObject({
          state: 'complete',
          current: 1,
          total: 1,
        }),
      { timeout: 3_000 },
    )
    expect(lifecycle.submitCount()).toBe(1)
    controller.destroy()
  })

  it('cancellation restores completed and partial overlays to an exact DOM snapshot', () => {
    const page = fixture()
    const expectedHtml = page.chapter.innerHTML
    const expectedChildren = [...page.chapter.childNodes]
    const controller = new ImageChapterMode()
    addTrackedOverlay(controller, page.first, 0, true)
    addTrackedOverlay(controller, page.second, 1, false)

    expect(document.querySelectorAll('.hskify-wrapper')).toHaveLength(2)
    expect(document.querySelectorAll('[data-hskify-mode-controls="true"]')).toHaveLength(1)
    controller.cancel()
    expectExactChapter(page, expectedHtml, expectedChildren)

    controller.cancel()
    expectExactChapter(page, expectedHtml, expectedChildren)
    controller.destroy()
  })

  it('a source replacement terminates the run and restores every image exactly', () => {
    const page = fixture()
    const expected = page.chapter.cloneNode(true) as HTMLElement
    const expectedChildren = [...page.chapter.childNodes]
    const controller = new ImageChapterMode()
    addTrackedOverlay(controller, page.first, 0, true)
    addTrackedOverlay(controller, page.second, 1, true)
    const internals = controller as unknown as ControllerInternals
    internals.scope = 'all'

    const replacement = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=='
    page.first.src = replacement
    Object.defineProperty(page.first, 'currentSrc', {
      configurable: true,
      value: replacement,
    })
    expected.querySelector<HTMLImageElement>('img[data-page="1"]')!.src = replacement
    internals.onDiscovery({
      type: 'updated',
      candidate: candidate(page.first, 0),
      previousSourceUrl: 'https://reader.test/page-1.webp',
      previousDomIndex: 0,
    })

    expectExactChapter(page, expected.innerHTML, expectedChildren)
    expect(page.first.getAttribute('src')).toBe(replacement)
    expect(internals.scope).toBeUndefined()
    controller.destroy()
  })

  it('updates queued order for a same-source discovery update without ending the run', () => {
    const page = fixture()
    const controller = new ImageChapterMode()
    const internals = controller as unknown as ControllerInternals
    internals.scope = 'all'
    internals.queueIds.set(page.second, 'queued-second')
    const reprioritize = vi.spyOn(internals.queue, 'reprioritize')
    const reordered = candidate(page.second, 0)

    internals.onDiscovery({
      type: 'updated',
      candidate: reordered,
      previousSourceUrl: reordered.sourceUrl,
      previousDomIndex: 1,
    })

    expect(reprioritize).toHaveBeenCalledWith('queued-second', true, 0)
    expect(internals.scope).toBe('all')
    controller.destroy()
  })

  it('freezes canonical page indexes when lazy readers insert or reorder surfaces', () => {
    const page = fixture()
    const late = loadedImage('https://reader.test/page-late.webp')
    late.dataset.page = 'late'
    document.body.append(late)
    const controller = new ImageChapterMode()
    const internals = controller as unknown as ControllerInternals
    const first = candidate(page.first, 0)
    const second = candidate(page.second, 1)
    const insertedBeforeFirst = candidate(late, 0)

    internals.establishCanonicalPageOrder([first, second])
    expect(internals.canonicalSourceIndex(first)).toBe(0)
    expect(internals.canonicalSourceIndex(second)).toBe(1)

    // The same elements report new mutable DOM indexes after a reader
    // prepends a lazy page. Their chapter identities and context positions
    // remain unchanged; the genuinely new surface appends to the stream.
    expect(internals.canonicalSourceIndex(candidate(page.second, 0))).toBe(1)
    expect(internals.canonicalSourceIndex(candidate(page.first, 1))).toBe(0)
    expect(internals.canonicalSourceIndex(insertedBeforeFirst)).toBe(2)
    expect(internals.chapterSourceOrder).toEqual([0, 1, 2])

    // A page removed before submission must not remain in the daemon's
    // expected-page barrier. Re-inserting the same element restores its
    // frozen identity rather than allocating a new position.
    internals.onDiscovery({ type: 'removed', candidate: first })
    expect(internals.chapterSourceOrder).toEqual([1, 2])
    expect(internals.canonicalSourceIndex(first)).toBe(0)
    expect(internals.chapterSourceOrder).toEqual([0, 1, 2])
    controller.destroy()
  })

  it('same-tab navigation and repeated disposal both restore the exact original DOM', () => {
    const page = fixture()
    const expectedHtml = page.chapter.innerHTML
    const expectedChildren = [...page.chapter.childNodes]
    const controller = new ImageChapterMode()
    addTrackedOverlay(controller, page.first, 0, true)
    addTrackedOverlay(controller, page.second, 1, false)
    const internals = controller as unknown as ControllerInternals
    internals.scope = 'all'
    const originalUrl = location.href
    const firstCandidate = candidate(page.first, 0)
    const sourceSnapshot = internals.sourceSnapshot(firstCandidate)

    history.pushState({}, '', `${location.pathname}?same-tab-cleanup=1`)
    expect(() =>
      internals.assertCurrent(firstCandidate, sourceSnapshot, new AbortController().signal),
    ).toThrowError(expect.objectContaining({ name: 'AbortError' }))
    controller.destroy()
    controller.destroy()
    expectExactChapter(page, expectedHtml, expectedChildren)
    history.replaceState({}, '', originalUrl)
  })
})
