import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeBrowser } from 'wxt/testing'

import { BackgroundRouter } from '../../src/messaging/background'
import { FixtureService } from '../../src/messaging/fixture-service'

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
    const sender = { tab: { id: 7 }, frameId: 0 } as browser.runtime.MessageSender
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
        pageOrigin: 'https://reader.test',
        naturalWidth: 1200,
        naturalHeight: 1800,
        hskLevel: 5,
        fixtureMode: true,
      },
      sender,
    )) as { jobId: string; sourceSha256: string }
    expect(submitted.jobId).toMatch(/^fixture-/)
    expect(submitted.sourceSha256).toMatch(/^[a-f0-9]{64}$/)

    now = 1_600
    const restartedBackground = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const recovered = (await restartedBackground.route(
      { type: 'jobs:recover', pageSessionId: 'fixture-page-session' },
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
      { type: 'job:result', jobId: submitted.jobId },
      sender,
    )) as { result: { sourceWidth: number; regions: unknown[] }; cleanImage: ArrayBuffer }
    expect(delivered.result.sourceWidth).toBe(1200)
    expect(delivered.result.regions).toHaveLength(2)
    expect(delivered.cleanImage).toBeInstanceOf(ArrayBuffer)
    expect(delivered.cleanImage.byteLength).toBeGreaterThan(32)

    expect(
      await restartedBackground.route(
        { type: 'jobs:recover', pageSessionId: 'fixture-page-session' },
        sender,
      ),
    ).toEqual([])
  })

  it('cancellation removes recovery metadata and leaves no partial result', async () => {
    const sender = { tab: { id: 9 }, frameId: 0 } as browser.runtime.MessageSender
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
        pageOrigin: 'https://reader.test',
        naturalWidth: 1200,
        naturalHeight: 1800,
        hskLevel: 3,
        fixtureMode: true,
      },
      sender,
    )) as { jobId: string }
    await router.route({ type: 'job:cancel', jobId: submitted.jobId }, sender)
    expect(
      await router.route(
        { type: 'jobs:recover', pageSessionId: 'cancel-page' },
        sender,
      ),
    ).toEqual([])
    await expect(
      router.route({ type: 'job:result', jobId: submitted.jobId }, sender),
    ).rejects.toThrow(/could not be recovered/i)
  })
})
