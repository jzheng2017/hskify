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

  it.each(['document', 'image'] as const)(
    'retries a detected %s warmup after resources become installed',
    async (contentKind) => {
      let installed = false
      const companion = {
        getSetupStatus: vi.fn(async () => installed
          ? {
              state: 'warming' as const,
              modelId: 'qwen3.5-4b',
              message: 'Resources are installed and a runtime is warming.',
            }
          : {
              state: 'missing-models' as const,
              modelId: 'qwen3.5-4b',
              message: 'Models are missing.',
            }),
        warmup: vi.fn(async (kind: 'document' | 'image') => ({
          state: 'ready' as const,
          modelId: 'qwen3.5-4b',
          message: `${kind} runtime is ready.`,
        })),
      } as unknown as CompanionClient
      const router = new BackgroundRouter({ companion })
      const sender = { id: fakeBrowser.runtime.id } as browser.runtime.MessageSender

      await expect(router.route({ type: 'engine:warmup', contentKind }, sender)).resolves
        .toMatchObject({ state: 'missing-models' })
      expect(companion.warmup).not.toHaveBeenCalled()

      installed = true
      await expect(router.route({ type: 'engine:warmup', contentKind }, sender)).resolves
        .toMatchObject({ state: 'ready' })
      expect(companion.warmup).toHaveBeenCalledExactlyOnceWith(contentKind)
    },
  )
})
