import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeBrowser } from 'wxt/testing'

import { PendingOriginPermissionStore } from '../../src/acquisition/origin-permissions'
import { BackgroundRouter } from '../../src/messaging/background'
import { FixtureService } from '../../src/messaging/fixture-service'
import type { DeliveredJobResult } from '../../src/messaging/messages'
import { MemoryStorage } from '../helpers/storage'

describe('Gate 1 background fixture adapter', () => {
  let now = 1_000

  beforeEach(() => {
    fakeBrowser.reset()
    vi.stubGlobal('browser', fakeBrowser)
    now = 1_000
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('submits, recovers after background reconstruction, and transfers the result', async () => {
    const sender = {
      tab: { id: 7 },
      frameId: 0,
      url: 'https://reader.test/chapter',
    } as browser.runtime.MessageSender
    const firstBackground = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const submitted = (await firstBackground.route(
      {
        type: 'job:submit',
        pageSessionId: 'fixture-page-session',
        pageIndex: 0,
        imageUrl: 'https://reader.test/panel.svg',
        pageUrl: 'https://reader.test/chapter',
        naturalWidth: 1200,
        naturalHeight: 1800,
        hskLevel: 5,
      },
      sender,
    )) as { jobId: string; sourceSha256: string }
    expect(submitted.jobId).toMatch(/^fixture-/)
    expect(submitted.sourceSha256).toMatch(/^[a-f0-9]{64}$/)
    await expect(
      firstBackground.route(
        { type: 'job:poll', jobId: submitted.jobId },
        {
          ...sender,
          url: 'https://reader.test/different-chapter',
        } as browser.runtime.MessageSender,
      ),
    ).rejects.toMatchObject({ code: 'DOCUMENT_IDENTITY_MISMATCH' })

    now = 1_600
    const restartedBackground = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const recovered = (await restartedBackground.route(
      {
        type: 'jobs:recover',
        pageSessionId: 'fixture-page-session',
        pageUrl: 'https://reader.test/chapter',
        candidates: [
          {
            sourceUrl: 'https://reader.test/panel.svg',
            naturalWidth: 1200,
            naturalHeight: 1800,
          },
        ],
      },
      sender,
    )) as Array<{ jobId: string; status: { state: string } }>
    expect(recovered).toHaveLength(1)
    expect(recovered[0]).toMatchObject({
      jobId: submitted.jobId,
      status: { state: 'running' },
    })

    now = 2_500
    const complete = (await restartedBackground.route(
      { type: 'job:poll', jobId: submitted.jobId },
      sender,
    )) as { state: string }
    expect(complete.state).toBe('complete')
    const delivered = (await restartedBackground.route(
      {
        type: 'job:result',
        jobId: submitted.jobId,
        pageSessionId: 'fixture-page-session',
        sourceUrl: 'https://reader.test/panel.svg',
        sourceSha256: submitted.sourceSha256,
        sourceWidth: 1200,
        sourceHeight: 1800,
      },
      sender,
    )) as DeliveredJobResult
    expect(delivered.result.sourceWidth).toBe(1200)
    expect(delivered.result.regions).toHaveLength(2)
    expect(delivered.result.regions[0]?.vocabulary.requestedHskLevel).toBe(5)
    expect(delivered.cleanImage).toBeInstanceOf(ArrayBuffer)
    expect(delivered.cleanImage.byteLength).toBeGreaterThanOrEqual(24)

    const regionId = delivered.result.regions[0]?.id
    if (!regionId) throw new Error('Fixture region missing.')
    await expect(
      restartedBackground.route(
        {
          type: 'dictionary:lookup',
          request: {
            selectedText: '我',
            jobId: submitted.jobId,
            regionId,
          },
        },
        { ...sender, tab: { id: 99 } } as browser.runtime.MessageSender,
      ),
    ).rejects.toMatchObject({ code: 'RESULT_OWNER_MISMATCH' })
    await expect(
      restartedBackground.route(
        { type: 'font:get', jobId: submitted.jobId, fontId: 'not-from-result' },
        sender,
      ),
    ).rejects.toMatchObject({ code: 'FONT_RESULT_MISMATCH' })
    const font = (await restartedBackground.route(
      { type: 'font:get', jobId: submitted.jobId, fontId: 'fixture-sans' },
      sender,
    )) as { bytes: ArrayBuffer }
    expect(font.bytes).toBeInstanceOf(ArrayBuffer)

    expect(
      await restartedBackground.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'fixture-page-session',
          pageUrl: 'https://reader.test/chapter',
          candidates: [],
        },
        sender,
      ),
    ).toEqual([])
  })

  it('cancellation removes recovery metadata and leaves no partial result', async () => {
    const sender = {
      tab: { id: 9 },
      frameId: 0,
      url: 'https://reader.test/cancel',
    } as browser.runtime.MessageSender
    const router = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const submitted = (await router.route(
      {
        type: 'job:submit',
        pageSessionId: 'cancel-page',
        pageIndex: 0,
        imageUrl: 'https://reader.test/panel.svg',
        pageUrl: 'https://reader.test/cancel',
        naturalWidth: 1200,
        naturalHeight: 1800,
        hskLevel: 3,
      },
      sender,
    )) as { jobId: string }
    await router.route({ type: 'job:cancel', jobId: submitted.jobId }, sender)
    expect(
      await router.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'cancel-page',
          pageUrl: 'https://reader.test/cancel',
          candidates: [],
        },
        sender,
      ),
    ).toEqual([])
    await expect(
      router.route(
        {
          type: 'job:result',
          jobId: submitted.jobId,
          pageSessionId: 'cancel-page',
          sourceUrl: 'https://reader.test/panel.svg',
          sourceSha256: 'a'.repeat(64),
          sourceWidth: 1200,
          sourceHeight: 1800,
        },
        sender,
      ),
    ).rejects.toThrow(/could not be recovered/i)
  })

  it('never recovers by DOM index when source URL/hash/dimensions do not match', async () => {
    const sender = {
      tab: { id: 12 },
      frameId: 0,
      url: 'https://reader.test/recovery',
    } as browser.runtime.MessageSender
    const router = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    await router.route(
      {
        type: 'job:submit',
        pageSessionId: 'recovery-page',
        pageIndex: 0,
        imageUrl: 'https://cdn.test/original.webp?token=one',
        pageUrl: 'https://reader.test/recovery',
        naturalWidth: 900,
        naturalHeight: 16_000,
        hskLevel: 5,
      },
      sender,
    )
    const recovered = await router.route(
      {
        type: 'jobs:recover',
        pageSessionId: 'recovery-page',
        pageUrl: 'https://reader.test/recovery',
        candidates: [
          {
            sourceUrl: 'https://cdn.test/replaced.webp?token=two',
            naturalWidth: 900,
            naturalHeight: 16_000,
          },
        ],
      },
      sender,
    )
    expect(recovered).toEqual([])
    expect(await router.route(
      {
        type: 'jobs:recover',
        pageSessionId: 'recovery-page',
        pageUrl: 'https://reader.test/recovery',
        candidates: [],
      },
      sender,
    )).toEqual([])
  })

  it('cancels only the matching tab/frame/page session on navigation', async () => {
    const sender = {
      tab: { id: 14 },
      frameId: 0,
      url: 'https://reader.test/chapter',
    } as browser.runtime.MessageSender
    const router = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    for (const pageSessionId of ['old-page', 'new-page']) {
      await router.route(
        {
          type: 'job:submit',
          pageSessionId,
          pageIndex: pageSessionId === 'old-page' ? 0 : 1,
          imageUrl: `https://reader.test/${pageSessionId}.png`,
          pageUrl: 'https://reader.test/chapter',
          naturalWidth: 1200,
          naturalHeight: 1800,
          hskLevel: 3,
        },
        sender,
      )
    }
    await router.route(
      { type: 'jobs:cancel-page', pageSessionId: 'old-page' },
      sender,
    )
    const recovered = (await router.route(
      {
        type: 'jobs:recover',
        pageSessionId: 'new-page',
        pageUrl: 'https://reader.test/chapter',
        candidates: [
          {
            sourceUrl: 'https://reader.test/new-page.png',
            naturalWidth: 1200,
            naturalHeight: 1800,
          },
        ],
      },
      sender,
    )) as unknown[]
    expect(recovered).toHaveLength(1)
  })

  it('persists a permission discovered during acquisition for the next popup click', async () => {
    const sender = {
      tab: { id: 23 },
      frameId: 0,
      url: 'https://reader.test/chapter',
    } as browser.runtime.MessageSender
    const pendingPermissions = new PendingOriginPermissionStore(new MemoryStorage())
    vi.spyOn(fakeBrowser.permissions, 'contains').mockResolvedValue(false)
    const fetcher = vi.fn()
    vi.stubGlobal('fetch', fetcher)
    const router = new BackgroundRouter({ pendingPermissions })

    await expect(
      router.route(
        {
          type: 'job:submit',
          pageSessionId: 'permission-page',
          pageIndex: 0,
          imageUrl: 'https://redirect-cdn.test/page.webp',
          pageUrl: 'https://reader.test/chapter',
          naturalWidth: 900,
          naturalHeight: 16_000,
          hskLevel: 5,
        },
        sender,
      ),
    ).rejects.toMatchObject({
      code: 'IMAGE_PERMISSION_REQUIRED',
      originPattern: 'https://redirect-cdn.test/*',
    })
    expect(fetcher).not.toHaveBeenCalled()
    expect(await pendingPermissions.list(23)).toEqual([
      'https://redirect-cdn.test/*',
    ])
    vi.spyOn(fakeBrowser.tabs, 'query').mockResolvedValue([
      {
        id: 23,
        index: 0,
        highlighted: true,
        active: true,
        pinned: false,
        incognito: false,
      },
    ])
    vi.spyOn(fakeBrowser.scripting, 'executeScript').mockResolvedValue([])
    vi.spyOn(fakeBrowser.tabs, 'sendMessage').mockResolvedValue({
      visibleOrigins: [],
      allOrigins: [],
    })
    await expect(
      router.route(
        { type: 'popup:prepare' },
        { id: fakeBrowser.runtime.id } as browser.runtime.MessageSender,
      ),
    ).resolves.toEqual({
      visibleOrigins: ['https://redirect-cdn.test/*'],
      allOrigins: ['https://redirect-cdn.test/*'],
    })
  })
})
