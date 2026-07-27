import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeBrowser } from 'wxt/testing'

import { BackgroundRouter } from '../../src/messaging/background'
import type { CompanionClient } from '../../src/messaging/companion-client'

describe('first-run background routing', () => {
  beforeEach(() => {
    fakeBrowser.reset()
    vi.stubGlobal('browser', fakeBrowser)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('proxies setup status and model preparation without opening tabs', async () => {
    const companion = {
      getSetupStatus: vi.fn(async () => ({
        state: 'missing-models' as const,
        modelId: 'qwen3.5-4b',
        message: 'Models are missing.',
      })),
      startModelSetup: vi.fn(async () => ({
        state: 'downloading' as const,
        modelId: 'qwen3.5-4b',
        completedBytes: 0,
        totalBytes: 2048,
        message: 'Downloading.',
      })),
    } as unknown as CompanionClient
    const create = vi.spyOn(fakeBrowser.tabs, 'create')
    const router = new BackgroundRouter({ companion })
    const sender = { id: fakeBrowser.runtime.id } as browser.runtime.MessageSender

    await expect(router.route({ type: 'setup:status' }, sender)).resolves.toMatchObject({
      state: 'missing-models',
    })
    await expect(router.route({ type: 'setup:start' }, sender)).resolves.toMatchObject({
      state: 'downloading',
    })
    expect(companion.getSetupStatus).toHaveBeenCalledTimes(1)
    expect(companion.startModelSetup).toHaveBeenCalledTimes(1)
    expect(create).not.toHaveBeenCalled()
  })
})
