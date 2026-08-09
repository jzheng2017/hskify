import { describe, expect, it } from 'vitest'

import { BUILD_FINGERPRINT } from '../../src/contracts/browser'
import { ActiveJobStore, type ActiveJobRecord } from '../../src/messaging/active-jobs'
import { MemoryStorage } from '../helpers/storage'
import { FixtureService } from '../support/fixture-service'

function record(overrides: Partial<ActiveJobRecord> = {}): ActiveJobRecord {
  const sourceSha256 = 'a'.repeat(64)
  return {
    tabId: 7,
    frameId: 0,
    pageSessionId: 'page',
    pageUrl: 'https://reader.test/chapter',
    jobId: 'fixture-job',
    sourceSha256,
    source: {
      kind: 'image',
      clientImageId: 'page-0-hash',
      sourceUrl: 'https://cdn.test/page.webp?chapter=1&page=0',
      sourceWidth: 900,
      sourceHeight: 16_000,
      sourceIndex: 0,
      request: {
        buildFingerprint: BUILD_FINGERPRINT,
        clientImageId: 'page-0-hash',
        sourceSha256,
        sourceMimeType: 'image/webp',
        naturalWidth: 900,
        naturalHeight: 16_000,
        pageSessionId: 'page',
        sourceIndex: 0,
        chapterSourceOrder: [0],
        surfaceKind: 'image',
        visibleRects: [],
        readingDirection: 'ltr',
        settings: {
          sourceLanguage: 'en',
          targetLanguage: 'zh-CN',
          hskStandard: '2.0',
          hskLevel: 5,
          learningMode: 'natural',
        },
      },
      uploadedBytes: 123_456,
    },
    submittedAtUnixMs: 990,
    acknowledgedSequence: 0,
    deliveredSequence: 0,
    itemIds: [],
    patchIds: [],
    fontIds: [],
    createdAtUnixMs: 1_000,
    ...overrides,
  }
}

function documentRecord(overrides: Partial<ActiveJobRecord> = {}): ActiveJobRecord {
  return {
    ...record(),
    sourceSha256: 'b'.repeat(64),
    source: {
      kind: 'document',
      settings: {
        sourceLanguage: 'en',
        targetLanguage: 'zh-CN',
        hskStandard: '2.0',
        hskLevel: 4,
        learningMode: 'strict',
      },
      blockCount: 2_000,
      uploadedBytes: 1024 * 1024,
    },
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
      'imageRegionReady',
    ])
    expect(reconstructed).toMatchObject({
      nextSequence: 5,
      updates: [
        {
          sequence: 5,
          type: 'complete',
          translatedCount: 2,
          preservedCount: 0,
        },
      ],
    })
  })

  it('persists only compact document recovery identity, never the full chapter request', async () => {
    const storage = new MemoryStorage()
    const store = new ActiveJobStore(storage)
    const active = documentRecord()

    await store.put(active)

    expect(await new ActiveJobStore(storage).get(active.jobId)).toEqual(active)
    const persisted = storage.values[`hskify.activeJob.${active.jobId}`] as {
      source: Record<string, unknown>
    }
    expect(persisted.source).toEqual(active.source)
    expect(persisted.source).not.toHaveProperty('request')
    expect(new TextEncoder().encode(JSON.stringify(persisted)).byteLength).toBeLessThan(2_048)
  })

  it('ignores malformed clean-schema recovery metadata', async () => {
    const storage = new MemoryStorage()
    const missingPageUrl = { ...record() } as Partial<ActiveJobRecord>
    delete missingPageUrl.pageUrl
    await storage.set({
      'hskify.activeJob.missing-page-url': missingPageUrl,
      'hskify.activeJob.invalid-tab': record({ jobId: 'invalid-tab', tabId: Number.NaN }),
      'hskify.activeJob.full-document-request': {
        ...documentRecord({ jobId: 'full-document-request' }),
        source: { kind: 'document', request: {}, uploadedBytes: 1 },
      },
    })

    expect(await new ActiveJobStore(storage).list()).toEqual([])
  })
})
