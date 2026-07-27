import { describe, expect, it } from 'vitest'

import { BUILD_FINGERPRINT } from '../../src/contracts/browser'
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
    submittedRequest: {
      buildFingerprint: BUILD_FINGERPRINT,
      clientImageId: 'page-0-hash',
      sourceSha256: 'a'.repeat(64),
      sourceMimeType: 'image/webp',
      naturalWidth: 900,
      naturalHeight: 16_000,
      pageSessionId: 'page',
      pageIndex: 0,
      visibleRects: [],
      settings: {
        sourceLanguage: 'en',
        targetLanguage: 'zh-CN',
        hskStandard: '2.0',
        hskLevel: 5,
        readingDirection: 'auto',
        translateSoundEffects: false,
        nameTranslation: 'keep-original',
      },
    },
    uploadedImageBytes: 123_456,
    submittedAtUnixMs: 990,
    acknowledgedSequence: 0,
    deliveredSequence: 0,
    regionIds: [],
    patchIds: [],
    fontIds: [],
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

  it('replays progressive updates from the persisted acknowledgement cursor', () => {
    const running = new FixtureService(() => 1_600).updates(record(), 0)
    const reconstructed = new FixtureService(() => 2_300).updates(
      record({ acknowledgedSequence: 4, deliveredSequence: 4 }),
      4,
    )
    expect(running.updates.map((update) => update.type)).toEqual([
      'progress',
      'progress',
      'regionReady',
    ])
    expect(reconstructed).toMatchObject({
      nextSequence: 6,
      updates: [
        { sequence: 5, type: 'regionRefined' },
        { sequence: 6, type: 'complete' },
      ],
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
