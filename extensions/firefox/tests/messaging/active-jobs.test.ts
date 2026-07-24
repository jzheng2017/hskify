import { describe, expect, it } from 'vitest'

import { ActiveJobStore, type ActiveJobRecord } from '../../src/messaging/active-jobs'
import { FixtureService } from '../../src/messaging/fixture-service'
import { MemoryStorage } from '../helpers/storage'

function record(overrides: Partial<ActiveJobRecord> = {}): ActiveJobRecord {
  return {
    tabId: 7,
    frameId: 0,
    pageSessionId: 'page',
    clientImageId: 'page-0-hash',
    jobId: 'fixture-job',
    sourceSha256: 'a'.repeat(64),
    pageIndex: 0,
    fixtureMode: true,
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

  it('keeps cloned binary messages intact without persisting image blobs', () => {
    const bytes = new Uint8Array(8 * 1024 * 1024)
    bytes[0] = 21
    bytes[bytes.length - 1] = 42
    const cloned = structuredClone({ cleanImage: bytes.buffer, font: bytes.slice(0, 32).buffer })
    expect(cloned.cleanImage.byteLength).toBe(8 * 1024 * 1024)
    expect(new Uint8Array(cloned.cleanImage).at(-1)).toBe(42)
    expect(cloned.font).toBeInstanceOf(ArrayBuffer)
  })
})
