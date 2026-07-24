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

  it('proxies setup status/start and opens only the bundled install guide', async () => {
    const companion = {
      getSetupStatus: vi.fn(async () => ({
        state: 'missing-models' as const,
        message: 'Models are missing.',
      })),
      startModelSetup: vi.fn(async () => ({
        state: 'downloading' as const,
        selectedPackId: 'standard-v1',
        completedBytes: 0,
        totalBytes: 2048,
        message: 'Downloading.',
      })),
    } as unknown as CompanionClient
    const create = vi.spyOn(fakeBrowser.tabs, 'create').mockResolvedValue({
      id: 3,
      index: 0,
      highlighted: true,
      active: true,
      pinned: false,
      incognito: false,
    })
    const router = new BackgroundRouter({ companion })
    const sender = { id: fakeBrowser.runtime.id } as browser.runtime.MessageSender

    await expect(router.route({ type: 'setup:status' }, sender)).resolves.toMatchObject({
      state: 'missing-models',
    })
    await expect(router.route({ type: 'setup:start' }, sender)).resolves.toMatchObject({
      state: 'downloading',
    })
    await expect(router.route({ type: 'setup:open-installer' }, sender)).resolves.toBeUndefined()
    expect(companion.getSetupStatus).toHaveBeenCalledTimes(1)
    expect(companion.startModelSetup).toHaveBeenCalledTimes(1)
    expect(create).toHaveBeenCalledWith({
      url: expect.stringMatching(/\/setup\.html$/),
    })
  })
})
