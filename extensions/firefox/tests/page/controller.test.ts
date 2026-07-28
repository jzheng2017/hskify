import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DiscoveredImage, DiscoveryEvent } from '../../src/discovery/images'
import type { VisibleFirstQueue } from '../../src/discovery/queue'
import { RuntimeMessageError } from '../../src/messaging/messages'
import {
  AUTOMATIC_IMAGE_RETRY_LIMIT,
  PageTranslationController,
  shouldAutomaticallyRetryImage,
} from '../../src/page/controller'
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
    element: image,
    owner: image.parentElement instanceof HTMLPictureElement ? image.parentElement : image,
    sourceUrl: image.currentSrc || image.src,
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
  controller: PageTranslationController,
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
  patch.className = 'hmt-patch'
  patch.dataset.patchId = `patch-${domIndex}`
  const text = document.createElement('span')
  text.className = 'hmt-region'
  text.dataset.regionId = `region-${domIndex}`
  text.textContent = 'å®Œæ•´æ–‡æœ¬'
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
  expect(fixture.chapter.querySelector('[data-hmt-owned], [data-hmt-original]')).toBeNull()
  expect(fixture.chapter.querySelector('.hmt-wrapper')).toBeNull()
  expect(document.querySelector('[data-hmt-mode-controls="true"]')).toBeNull()
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
    if (type === 'job:submit') {
      submitted += 1
      return {
        ok: true,
        value: {
          jobId: `job-${submitted}`,
          clientImageId: `image-${submitted}`,
          sourceSha256: 'a'.repeat(64),
          sourceUrl: message.imageUrl,
          sourceWidth: message.naturalWidth,
          sourceHeight: message.naturalHeight,
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
          : { sequence: 1, type: 'complete', message: 'Complete' }
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
    .querySelectorAll('[data-hmt-owned]')
    .forEach((element) => element.remove())
  document.body.replaceChildren()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('page controller terminal restoration', () => {
  it('automatically retries only retryable image failures within the limit', () => {
    const retryable = new RuntimeMessageError('PIPELINE_FAILED', 'Temporary local failure', true)
    const permanent = new RuntimeMessageError('UNSUPPORTED_IMAGE', 'Unsupported image', false)

    expect(shouldAutomaticallyRetryImage(retryable, 0)).toBe(true)
    expect(shouldAutomaticallyRetryImage(retryable, AUTOMATIC_IMAGE_RETRY_LIMIT - 1)).toBe(true)
    expect(shouldAutomaticallyRetryImage(retryable, AUTOMATIC_IMAGE_RETRY_LIMIT)).toBe(false)
    expect(shouldAutomaticallyRetryImage(permanent, 0)).toBe(false)
    expect(shouldAutomaticallyRetryImage(new Error('unknown'), 0)).toBe(false)
  })

  it('retries a transient image twice and publishes one stable chapter completion', async () => {
    const image = loadedImage('https://reader.test/retry-page.webp')
    document.body.append(image)
    const lifecycle = installJobLifecycle(2)
    const controller = new PageTranslationController()

    await controller.start('all', 3, 'natural', 'keep-original')
    await vi.waitFor(
      () =>
        expect(controller.snapshot()).toMatchObject({
          state: 'complete',
          current: 1,
          total: 1,
        }),
      { timeout: 3_000 },
    )
    expect(lifecycle.submitCount()).toBe(3)
    await new Promise((resolve) => setTimeout(resolve, AUTOMATIC_IMAGE_RETRY_LIMIT * 25))
    expect(controller.snapshot().state).toBe('complete')
    expect(lifecycle.submitCount()).toBe(3)
    controller.destroy()
  })

  it('exhausts the automatic retry budget before publishing attention state', async () => {
    const image = loadedImage('https://reader.test/retry-page.webp')
    document.body.append(image)
    const lifecycle = installJobLifecycle(Number.POSITIVE_INFINITY)
    const controller = new PageTranslationController()

    await controller.start('all', 3, 'natural', 'keep-original')
    await vi.waitFor(
      () =>
        expect(controller.snapshot()).toMatchObject({
          state: 'failed',
          current: 0,
          total: 1,
        }),
      { timeout: 3_000 },
    )
    expect(lifecycle.submitCount()).toBe(1 + AUTOMATIC_IMAGE_RETRY_LIMIT)
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
    const controller = new PageTranslationController()
    const internals = controller as unknown as ControllerInternals

    await controller.start('all', 3, 'natural', 'keep-original')
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
    const controller = new PageTranslationController()
    const internals = controller as unknown as ControllerInternals

    await expect(controller.start('all', 3, 'natural', 'keep-original')).resolves.toMatchObject({
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
    const controller = new PageTranslationController()
    addTrackedOverlay(controller, page.first, 0, true)
    addTrackedOverlay(controller, page.second, 1, false)

    expect(document.querySelectorAll('.hmt-wrapper')).toHaveLength(2)
    expect(document.querySelectorAll('[data-hmt-mode-controls="true"]')).toHaveLength(1)
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
    const controller = new PageTranslationController()
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
    const controller = new PageTranslationController()
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

  it('same-tab navigation and repeated disposal both restore the exact original DOM', () => {
    const page = fixture()
    const expectedHtml = page.chapter.innerHTML
    const expectedChildren = [...page.chapter.childNodes]
    const controller = new PageTranslationController()
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
    expectExactChapter(page, expectedHtml, expectedChildren)
    expect(internals.scope).toBeUndefined()

    addTrackedOverlay(controller, page.first, 0, true)
    addTrackedOverlay(controller, page.second, 1, false)
    controller.destroy()
    controller.destroy()
    expectExactChapter(page, expectedHtml, expectedChildren)
    history.replaceState({}, '', originalUrl)
  })
})
