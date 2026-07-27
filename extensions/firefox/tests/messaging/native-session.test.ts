import { describe, expect, it, vi } from 'vitest'

import {
  BUILD_FINGERPRINT,
} from '../../src/contracts/browser'
import {
  NATIVE_HOST_NAME,
  NativeSessionError,
  NativeSessionManager,
} from '../../src/messaging/native-session'
import { MemoryStorage } from '../helpers/storage'

function ready(token = 'A'.repeat(43), expires = 2_000_000) {
  return {
    type: 'ready',
    buildFingerprint: BUILD_FINGERPRINT,
    engineVersion: '0.1.0',
    port: 43127,
    token,
    sessionExpiresAtUnixMs: expires,
    capabilities: {
      sourceLanguages: ['en'],
      targetLanguages: ['zh-CN'],
      hskLevels: [1, 2, 3, 4, 5, 6],
      modelsReady: true,
    },
  }
}

describe('one-shot native session handshake', () => {
  it('launches with the frozen identity and stores only in session storage', async () => {
    const storage = new MemoryStorage()
    const sendNativeMessage = vi.fn(async () => ready())
    const manager = new NativeSessionManager(
      storage,
      {
        getManifest: () => ({ version: '0.1.0' }),
        getURL: () => 'moz-extension://fixture-installation/',
        sendNativeMessage,
      },
      () => 1_000,
    )
    const session = await manager.getOrLaunch()
    expect(session.port).toBe(43127)
    expect(sendNativeMessage).toHaveBeenCalledWith(NATIVE_HOST_NAME, {
      type: 'start-or-discover-daemon',
      buildFingerprint: BUILD_FINGERPRINT,
      extensionVersion: '0.1.0',
      extensionOrigin: 'moz-extension://fixture-installation',
    })
    expect(Object.keys(storage.values)).toEqual(['hmt.nativeSession'])
  })

  it('recovers a valid session without relying on background globals', async () => {
    const storage = new MemoryStorage()
    storage.values['hmt.nativeSession'] = ready()
    const runtime = {
      getManifest: () => ({ version: '0.1.0' }),
      getURL: () => 'moz-extension://fixture/',
      sendNativeMessage: vi.fn(async () => ready()),
    }
    const firstBackground = new NativeSessionManager(storage, runtime, () => 1_000)
    const secondBackground = new NativeSessionManager(storage, runtime, () => 1_000)
    expect((await firstBackground.getOrLaunch()).token).toBe('A'.repeat(43))
    expect((await secondBackground.getOrLaunch()).token).toBe('A'.repeat(43))
    expect(runtime.sendNativeMessage).not.toHaveBeenCalled()
  })

  it('refreshes expiring sessions and rejects malformed native replies', async () => {
    const storage = new MemoryStorage()
    storage.values['hmt.nativeSession'] = ready('A'.repeat(43), 5_500)
    const manager = new NativeSessionManager(
      storage,
      {
        getManifest: () => ({ version: '0.1.0' }),
        getURL: () => 'moz-extension://fixture/',
        sendNativeMessage: vi.fn(async () => ({ type: 'not-ready' })),
      },
      () => 1_000,
    )
    await expect(manager.getOrLaunch()).rejects.toBeInstanceOf(NativeSessionError)
  })
})
