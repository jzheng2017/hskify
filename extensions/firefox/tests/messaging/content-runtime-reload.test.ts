import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { BackgroundRouter } from '../../src/messaging/background'
import { bootContentRuntime } from '../../src/page/controller'

const idleState = {
  state: 'idle' as const,
  contentKind: 'document' as const,
  current: 0,
  total: 8,
  message: 'Light-novel chapter detected',
}

beforeEach(() => {
  document.body.replaceChildren()
  globalThis.__hskifyContentRuntime = undefined
})

afterEach(() => {
  globalThis.__hskifyContentRuntime?.dispose()
  globalThis.__hskifyContentRuntime = undefined
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('content runtime reload recovery', () => {
  it('reinjects a stale open tab and verifies the new listener before reporting it prepared', async () => {
    let injected = false
    const sendMessage = vi.fn(async (_tabId: number, message: { type: string }) => {
      expect(message).toEqual({ type: 'content:state' })
      if (!injected)
        throw new Error('Could not establish connection. Receiving end does not exist.')
      return idleState
    })
    const executeScript = vi.fn(async () => {
      injected = true
      return []
    })
    vi.stubGlobal('browser', {
      tabs: {
        query: vi.fn(async () => [{ id: 17 }]),
        sendMessage,
      },
      scripting: { executeScript },
      storage: {
        local: {
          get: vi.fn(async () => ({})),
          set: vi.fn(async () => undefined),
          remove: vi.fn(async () => undefined),
        },
      },
    })

    const router = new BackgroundRouter()
    await router.route({ type: 'popup:prepare' }, {})
    await router.route({ type: 'popup:prepare' }, {})

    expect(executeScript).toHaveBeenCalledTimes(1)
    expect(executeScript).toHaveBeenCalledWith({
      target: { tabId: 17, allFrames: false },
      files: ['translator.js'],
    })
    expect(sendMessage).toHaveBeenCalledTimes(3)
  })

  it('replaces a stale page runtime instead of trusting its old boot marker', () => {
    const listeners = new Set<(...args: never[]) => unknown>()
    const addListener = vi.fn((listener: (...args: never[]) => unknown) => {
      listeners.add(listener)
    })
    const removeListener = vi.fn((listener: (...args: never[]) => unknown) => {
      listeners.delete(listener)
    })
    vi.stubGlobal('browser', {
      runtime: {
        id: 'hskify@local.hskify',
        onMessage: { addListener, removeListener },
      },
    })

    bootContentRuntime()
    const first = globalThis.__hskifyContentRuntime
    bootContentRuntime()

    expect(first).toBeDefined()
    expect(globalThis.__hskifyContentRuntime).not.toBe(first)
    expect(addListener).toHaveBeenCalledTimes(2)
    expect(removeListener).toHaveBeenCalledTimes(1)
    expect(listeners).toHaveLength(1)
    expect(document.documentElement.dataset.hskifyInjected).toBe('true')
  })
})
