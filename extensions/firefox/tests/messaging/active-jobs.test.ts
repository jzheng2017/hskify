import { describe, expect, it } from 'vitest'

import { ActiveJobStore, type ActiveJobRecord } from '../../src/messaging/active-jobs'
import { FixtureService } from '../../src/messaging/fixture-service'
import { MemoryStorage } from '../helpers/storage'

function record(overrides: Partial<ActiveJobRecord> = {}): ActiveJobRecord {
  return {
    tabId: 7,
    frameId: 0,
    pageSessionId: 'page',
    pageUrl: 'https://reader.test/chapter',
    clientImageId: 'page-0-hash',
    jobId: 'fixture-job',
    sourceSha256: 'a'.repeat(64),
    sourceUrl: 'https://cdn.test/page.webp?chapter=1&page=0',
    sourceWidth: 900,
    sourceHeight: 16_000,
    pageIndex: 0,
    hskLevel: 5,
    createdAtUnixMs: 1_000,
    ...overrides,
  }
}

describe('active-job recovery metadata', () => {
  it('persists each job independently and scopes recovery to tab/frame/page', async () => {
    const storage = new MemoryStorage()
    const firstBackground = new ActiveJobStore(storage)
    await firstBackground.put(record())
    await firstBackground.put(record({ jobId: 'other-tab', tabId: 8 }))

    const restartedBackground = new ActiveJobStore(storage)
    expect(await restartedBackground.forPage(7, 0, 'page')).toEqual([record()])
    await restartedBackground.remove('fixture-job')
    expect(await restartedBackground.forPage(7, 0, 'page')).toEqual([])
    expect(await restartedBackground.forTab(8)).toHaveLength(1)
  })

  it('derives fixture progress after popup closure or background suspension', () => {
    const running = new FixtureService(() => 1_600).status(record())
    const reconstructed = new FixtureService(() => 2_300).status(record())
    expect(running.state).toBe('running')
    expect(reconstructed).toMatchObject({
      state: 'complete',
      stage: 'complete',
      overallProgress: 1,
    })
  })

  it('ignores incomplete or malformed recovery metadata', async () => {
    const storage = new MemoryStorage()
    const missingPageUrl = { ...record() } as Partial<ActiveJobRecord>
    delete missingPageUrl.pageUrl
    await storage.set({
      'hmt.activeJob.missing-page-url': missingPageUrl,
      'hmt.activeJob.invalid-tab': record({
        jobId: 'invalid-tab',
        tabId: Number.NaN,
      }),
    })

    expect(await new ActiveJobStore(storage).list()).toEqual([])
  })
})
