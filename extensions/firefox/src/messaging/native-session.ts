import {
  PROTOCOL_VERSION,
  parseNativeReadyResponse,
  type NativeHandshakeRequest,
  type NativeReadyResponse,
} from '../contracts/browser'
import type { StorageArea } from './settings'

export const NATIVE_HOST_NAME = 'local.mangalations.hsk_manga'
export const SESSION_STORAGE_KEY = 'hmt.nativeSession'
const EXPIRY_SAFETY_WINDOW_MS = 5_000

export type RuntimeNativeApi = {
  getManifest(): { version: string }
  getURL(path: string): string
  sendNativeMessage(application: string, message: unknown): Promise<unknown>
}

export function extensionOrigin(runtime: RuntimeNativeApi): string {
  const url = new URL(runtime.getURL(''))
  return `${url.protocol}//${url.host}`
}

function isStoredSession(value: unknown): value is NativeReadyResponse {
  try {
    parseNativeReadyResponse(value)
    return true
  } catch {
    return false
  }
}

export class NativeSessionError extends Error {
  readonly code = 'COMPANION_UNAVAILABLE'
  readonly retryable = true

  constructor(message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'NativeSessionError'
  }
}

export class NativeSessionManager {
  constructor(
    private readonly storage: StorageArea = browser.storage.session,
    private readonly runtime: RuntimeNativeApi = browser.runtime,
    private readonly now: () => number = Date.now,
  ) {}

  async getOrLaunchWithState(
    forceRefresh = false,
  ): Promise<{ session: NativeReadyResponse; reused: boolean }> {
    if (!forceRefresh) {
      const values = await this.storage.get(SESSION_STORAGE_KEY)
      const stored = values[SESSION_STORAGE_KEY]
      if (
        isStoredSession(stored) &&
        stored.sessionExpiresAtUnixMs > this.now() + EXPIRY_SAFETY_WINDOW_MS
      ) {
        return { session: stored, reused: true }
      }
    }

    const manifest = this.runtime.getManifest()
    const request: NativeHandshakeRequest = {
      type: 'start-or-discover-daemon',
      protocolVersion: PROTOCOL_VERSION,
      extensionVersion: manifest.version,
      extensionOrigin: extensionOrigin(this.runtime),
    }

    let raw: unknown
    try {
      raw = await this.runtime.sendNativeMessage(NATIVE_HOST_NAME, request)
    } catch (error) {
      throw new NativeSessionError(
        'The local translation engine is not installed or could not be started.',
        { cause: error },
      )
    }

    let ready: NativeReadyResponse
    try {
      ready = parseNativeReadyResponse(raw)
    } catch (error) {
      throw new NativeSessionError('The local translation engine returned an invalid handshake.', {
        cause: error,
      })
    }
    if (ready.sessionExpiresAtUnixMs <= this.now()) {
      throw new NativeSessionError('The local translation engine returned an expired session.')
    }
    await this.storage.set({ [SESSION_STORAGE_KEY]: ready })
    return { session: ready, reused: false }
  }

  async getOrLaunch(forceRefresh = false): Promise<NativeReadyResponse> {
    return (await this.getOrLaunchWithState(forceRefresh)).session
  }

  origin(): string {
    return extensionOrigin(this.runtime)
  }

  async invalidate(): Promise<void> {
    await this.storage.remove(SESSION_STORAGE_KEY)
  }
}
