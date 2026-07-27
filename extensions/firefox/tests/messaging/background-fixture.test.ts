import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeBrowser } from 'wxt/testing'

import {
  ActiveJobStore,
  PageArtifactStore,
} from '../../src/messaging/active-jobs'
import { BackgroundRouter } from '../../src/messaging/background'
import { FixtureService } from '../../src/messaging/fixture-service'

const visibleRects = [{ x: 0, y: 0.1, width: 1, height: 0.4 }]

function sender(url = 'https://reader.test/chapter', tabId = 7) {
  return {
    tab: { id: tabId },
    frameId: 0,
    url,
  } as browser.runtime.MessageSender
}

function submitMessage(pageSessionId = 'fixture-page-session') {
  return {
    type: 'job:submit' as const,
    pageSessionId,
    pageIndex: 0,
    imageUrl: 'https://reader.test/panel.svg',
    pageUrl: 'https://reader.test/chapter',
    naturalWidth: 1200,
    naturalHeight: 1800,
    hskLevel: 5 as const,
    visibleRects,
  }
}

describe('progressive background fixture adapter', () => {
  let now = 1_000

  beforeEach(() => {
    fakeBrowser.reset()
    vi.stubGlobal('browser', fakeBrowser)
    now = 1_000
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('persists the installed cursor across MV3 reconstruction and streams region assets', async () => {
    const first = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const submitted = (await first.route(
      submitMessage(),
      sender(),
    )) as { jobId: string; sourceSha256: string; acknowledgedSequence: number }
    expect(submitted).toMatchObject({ acknowledgedSequence: 0 })
    expect(submitted.jobId).toMatch(/^fixture-/)

    await expect(
      first.route(
        { type: 'job:updates', jobId: submitted.jobId, after: 0 },
        sender('https://reader.test/other'),
      ),
    ).rejects.toMatchObject({ code: 'DOCUMENT_IDENTITY_MISMATCH' })

    now = 1_600
    const firstBatch = (await first.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )) as {
      nextSequence: number
      updates: Array<{ type: string; region?: { patch: { blobId: string } } }>
    }
    expect(firstBatch.updates.map((update) => update.type)).toEqual([
      'progress',
      'progress',
      'regionReady',
    ])
    const patchId = firstBatch.updates.at(-1)?.region?.patch.blobId
    if (!patchId) throw new Error('Fixture patch update missing.')
    const patch = (await first.route(
      {
        type: 'job:patch',
        jobId: submitted.jobId,
        patchId,
        mimeType: 'image/png',
      },
      sender(),
    )) as { patchId: string; bytes: ArrayBuffer }
    expect(patch.patchId).toBe(patchId)
    expect(patch.bytes).toBeInstanceOf(ArrayBuffer)
    expect(await new PageArtifactStore().get(submitted.jobId)).toBeUndefined()

    await first.route(
      {
        type: 'job:ack',
        jobId: submitted.jobId,
        sequence: firstBatch.nextSequence,
      },
      sender(),
    )
    expect(await new PageArtifactStore().get(submitted.jobId)).toBeUndefined()

    const restarted = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const recovered = (await restarted.route(
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
      sender(),
    )) as Array<{ jobId: string; acknowledgedSequence: number }>
    expect(recovered).toEqual([
      expect.objectContaining({
        jobId: submitted.jobId,
        acknowledgedSequence: 3,
      }),
    ])

    now = 2_500
    const finalBatch = (await restarted.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 3 },
      sender(),
    )) as { nextSequence: number; updates: Array<{ type: string }> }
    expect(finalBatch.updates.map((update) => update.type)).toEqual([
      'regionReady',
      'regionRefined',
      'complete',
    ])
    await restarted.route(
      {
        type: 'job:ack',
        jobId: submitted.jobId,
        sequence: finalBatch.nextSequence,
        terminalType: 'complete',
      },
      sender(),
    )
    expect(await new ActiveJobStore().get(submitted.jobId)).toBeUndefined()
    expect(await new PageArtifactStore().get(submitted.jobId)).toBeDefined()

    const lookup = (await restarted.route(
      {
        type: 'dictionary:lookup',
        request: {
          selectedText: '离开',
          jobId: submitted.jobId,
          regionId: `${submitted.sourceSha256.slice(0, 8)}-region-0001`,
        },
      },
      sender(),
    )) as { region?: { baseChinese: string } }
    expect(lookup.region?.baseChinese).toBe('我们得马上离开！')
    const font = (await restarted.route(
      { type: 'font:get', jobId: submitted.jobId, fontId: 'fixture-sans' },
      sender(),
    )) as { bytes: ArrayBuffer }
    expect(font.bytes).toBeInstanceOf(ArrayBuffer)

    // Terminal jobs are deleted instead of being mistaken for resumable work.
    expect(
      await restarted.route(
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
        sender(),
      ),
    ).toEqual([])
  })

  it('accepts throttled viewport state only from the owning document', async () => {
    const router = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const submitted = (await router.route(submitMessage(), sender())) as { jobId: string }
    await expect(
      router.route(
        {
          type: 'job:viewport',
          jobId: submitted.jobId,
          visibleRects: [{ x: 0, y: 0.4, width: 1, height: 0.3 }],
          active: true,
        },
        sender(),
      ),
    ).resolves.toBeUndefined()
    await expect(
      router.route(
        {
          type: 'job:viewport',
          jobId: submitted.jobId,
          visibleRects: [],
          active: false,
        },
        sender('https://reader.test/other'),
      ),
    ).rejects.toMatchObject({ code: 'DOCUMENT_IDENTITY_MISMATCH' })
  })

  it('requires acknowledgement before advancing the external update cursor', async () => {
    const router = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const submitted = (await router.route(submitMessage(), sender())) as { jobId: string }
    now = 1_600
    await router.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )
    await expect(
      router.route(
        { type: 'job:updates', jobId: submitted.jobId, after: 3 },
        sender(),
      ),
    ).rejects.toMatchObject({ code: 'UPDATE_CURSOR_MISMATCH' })
  })

  it('cancellation removes recovery and partial artifact metadata', async () => {
    const router = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const submitted = (await router.route(submitMessage('cancel-page'), sender())) as {
      jobId: string
    }
    now = 1_600
    await router.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )
    await router.route({ type: 'job:cancel', jobId: submitted.jobId }, sender())
    expect(
      await router.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'cancel-page',
          pageUrl: 'https://reader.test/chapter',
          candidates: [],
        },
        sender(),
      ),
    ).toEqual([])
    await expect(
      router.route(
        {
          type: 'job:patch',
          jobId: submitted.jobId,
          patchId: 'stale',
          mimeType: 'image/png',
        },
        sender(),
      ),
    ).rejects.toMatchObject({ code: 'ACTIVE_JOB_NOT_FOUND' })
  })

  it('never recovers a different source at the same DOM index', async () => {
    const router = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    await router.route(submitMessage('source-page'), sender())
    expect(
      await router.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'source-page',
          pageUrl: 'https://reader.test/chapter',
          candidates: [
            {
              sourceUrl: 'https://reader.test/replaced.svg',
              naturalWidth: 1200,
              naturalHeight: 1800,
            },
          ],
        },
        sender(),
      ),
    ).toEqual([])
  })
})
