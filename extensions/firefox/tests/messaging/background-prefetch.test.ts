import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeBrowser } from 'wxt/testing'

import { BackgroundRouter } from '../../src/messaging/background'
import { FixtureService } from '../support/fixture-service'

const pageUrl = 'https://reader.test/chapter'

function sender() {
  return {
    tab: { id: 7 },
    frameId: 0,
    url: pageUrl,
  } as browser.runtime.MessageSender
}

function source(sourceIndex = 1) {
  return {
    pageSessionId: 'chapter-session',
    sourceIndex,
    imageUrl: `https://cdn.test/${sourceIndex}.webp`,
    pageUrl,
    naturalWidth: 900,
    naturalHeight: 16_000,
  }
}

describe('background acquisition prefetch handoff', () => {
  beforeEach(() => {
    fakeBrowser.reset()
    vi.stubGlobal('browser', fakeBrowser)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('acquires and hashes once without creating a job, then consumes those bytes on submit', async () => {
    const fixture = new FixtureService()
    const acquire = vi.spyOn(fixture, 'sourceImage')
    const createJob = vi.spyOn(fixture, 'createJobId')
    const digest = vi.spyOn(globalThis.crypto.subtle, 'digest')
    const router = new BackgroundRouter({ fixture })

    await router.route({ type: 'image:prefetch', ...source() }, sender())

    expect(acquire).toHaveBeenCalledTimes(1)
    expect(digest).toHaveBeenCalledTimes(1)
    expect(createJob).not.toHaveBeenCalled()

    await router.route(
      {
        type: 'job:submit-image',
        clientRequestId: 'test-request',
        retryItemIds: [],
        ...source(),
        chapterSourceOrder: [1],
        surfaceKind: 'image',
        hskLevel: 5,
        learningMode: 'natural',
        readingDirection: 'ltr',
        visibleRects: [],
      },
      sender(),
    )

    expect(acquire).toHaveBeenCalledTimes(1)
    expect(digest).toHaveBeenCalledTimes(1)
    expect(createJob).toHaveBeenCalledTimes(1)
  })

  it('checks current image bytes using existing acquisition and enforces renderer ownership', async () => {
    const fixture = new FixtureService()
    const router = new BackgroundRouter({ fixture })
    const submitted = (await router.route(
      {
        type: 'job:submit-image',
        clientRequestId: 'revision',
        retryItemIds: [],
        ...source(),
        chapterSourceOrder: [1],
        surfaceKind: 'image',
        hskLevel: 3,
        learningMode: 'natural',
        readingDirection: 'ltr',
        visibleRects: [],
      },
      sender(),
    )) as { jobId: string; sourceSha256: string }
    const request = { type: 'source:image-revision' as const, jobId: submitted.jobId }
    expect(await router.route(request, sender())).toBe(submitted.sourceSha256)
    await expect(
      router.route(request, { ...sender(), tab: { id: 8 } } as browser.runtime.MessageSender),
    ).rejects.toMatchObject({ code: 'RESULT_OWNER_MISMATCH' })
    const bytes = await fixture.sourceImage(900, 16_000)
    const view = new Uint8Array(bytes)
    view[bytes.byteLength - 1] = view[bytes.byteLength - 1]! ^ 1
    vi.spyOn(fixture, 'sourceImage').mockResolvedValue(bytes)
    expect(await router.route(request, sender())).not.toBe(submitted.sourceSha256)
  })

  it('drops retained bytes on cancellation and does not hand them to a later submit', async () => {
    const fixture = new FixtureService()
    const acquire = vi.spyOn(fixture, 'sourceImage')
    const router = new BackgroundRouter({ fixture })

    await router.route({ type: 'image:prefetch', ...source() }, sender())
    await router.route(
      {
        type: 'image:prefetch-cancel',
        pageSessionId: 'chapter-session',
        pageUrl,
      },
      sender(),
    )
    await router.route(
      {
        type: 'job:submit-image',
        clientRequestId: 'test-request',
        retryItemIds: [],
        ...source(),
        chapterSourceOrder: [1],
        surfaceKind: 'image',
        hskLevel: 5,
        learningMode: 'natural',
        readingDirection: 'ltr',
        visibleRects: [],
      },
      sender(),
    )

    expect(acquire).toHaveBeenCalledTimes(2)
  })
})
