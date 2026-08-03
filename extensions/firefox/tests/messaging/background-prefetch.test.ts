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

function source(pageIndex = 1) {
  return {
    pageSessionId: 'chapter-session',
    pageIndex,
    imageUrl: `https://cdn.test/${pageIndex}.webp`,
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
        type: 'job:submit',
        ...source(),
        chapterPageOrder: [0],
        surfaceKind: 'image',
        hskLevel: 5,
        learningMode: 'natural',
        nameTranslation: 'keep-original',
        visibleRects: [],
      },
      sender(),
    )

    expect(acquire).toHaveBeenCalledTimes(1)
    expect(digest).toHaveBeenCalledTimes(1)
    expect(createJob).toHaveBeenCalledTimes(1)
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
        type: 'job:submit',
        ...source(),
        chapterPageOrder: [0],
        surfaceKind: 'image',
        hskLevel: 5,
        learningMode: 'natural',
        nameTranslation: 'keep-original',
        visibleRects: [],
      },
      sender(),
    )

    expect(acquire).toHaveBeenCalledTimes(2)
  })
})
