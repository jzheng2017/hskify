import { describe, expect, it, vi } from 'vitest'

import { BUILD_FINGERPRINT, type BrowserJobRequest } from '../../src/contracts/browser'
import { CompanionClient } from '../../src/messaging/companion-client'
import { NativeSessionManager, SESSION_STORAGE_KEY } from '../../src/messaging/native-session'
import { pngHeader } from '../helpers/images'
import { MemoryStorage } from '../helpers/storage'

function ready(token: string) {
  return {
    type: 'ready',
    buildFingerprint: BUILD_FINGERPRINT,
    engineVersion: '0.2.0',
    port: 43127,
    token,
    sessionExpiresAtUnixMs: Date.now() + 60_000,
    capabilities: {
      sourceLanguages: ['en'],
      targetLanguages: ['zh-CN'],
      hskLevels: [1, 2, 3, 4, 5, 6],
      modelsReady: true,
    },
  }
}

function sessionManager() {
  let calls = 0
  const runtime = {
    getManifest: () => ({ version: '0.1.0' }),
    getURL: () => 'moz-extension://fixture/',
    sendNativeMessage: vi.fn(async () => ready((calls++ === 0 ? 'A' : 'B').repeat(43))),
  }
  return { manager: new NativeSessionManager(new MemoryStorage(), runtime), runtime }
}

function request(): BrowserJobRequest {
  return {
    buildFingerprint: BUILD_FINGERPRINT,
    clientImageId: 'page-0-hash',
    sourceSha256: 'a'.repeat(64),
    sourceMimeType: 'image/png',
    naturalWidth: 1200,
    naturalHeight: 1800,
    pageSessionId: 'page',
    pageIndex: 0,
    visibleRects: [{ x: 0, y: 0, width: 1, height: 0.5 }],
    settings: {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel: 5,
      readingDirection: 'auto',
      translateSoundEffects: false,
    },
  }
}

function emptyUpdates(jobId = 'job', nextSequence = 0): Response {
  return new Response(JSON.stringify({ jobId, nextSequence, updates: [] }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('authenticated unversioned companion client', () => {
  it('re-handshakes and retries exactly once after a 401 without a protocol header', async () => {
    const { manager, runtime } = sessionManager()
    const authorizations: string[] = []
    const fetcher = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      authorizations.push(new Headers(init?.headers).get('Authorization') ?? '')
      if (authorizations.length === 1) return new Response(null, { status: 401 })
      return emptyUpdates()
    })
    const client = new CompanionClient(manager, fetcher)
    expect((await client.getJobUpdates('job', 0)).updates).toEqual([])
    expect(authorizations).toEqual([`Bearer ${'A'.repeat(43)}`, `Bearer ${'B'.repeat(43)}`])
    expect(runtime.sendNativeMessage).toHaveBeenCalledTimes(2)
    const headers = new Headers(fetcher.mock.calls[1]?.[1]?.headers)
    expect(headers.has('X-HSK-Manga-Protocol')).toBe(false)
    expect(headers.get('X-HSK-Manga-Extension-Origin')).toBe('moz-extension://fixture')
  })

  it('health-checks a cached root endpoint and re-handshakes after transport failure', async () => {
    const storage = new MemoryStorage()
    storage.values[SESSION_STORAGE_KEY] = ready('A'.repeat(43))
    const runtime = {
      getManifest: () => ({ version: '0.1.0' }),
      getURL: () => 'moz-extension://fixture/',
      sendNativeMessage: vi.fn(async () => ready('B'.repeat(43))),
    }
    const requests: string[] = []
    const client = new CompanionClient(
      new NativeSessionManager(storage, runtime),
      async (input) => {
        requests.push(String(input))
        if (requests.length === 1) throw new TypeError('stale cached port')
        return emptyUpdates()
      },
    )
    expect((await client.getJobUpdates('job', 0)).jobId).toBe('job')
    expect(requests).toEqual([
      'http://127.0.0.1:43127/health',
      'http://127.0.0.1:43127/jobs/job/updates?after=0&waitMs=20000',
    ])
    expect(runtime.sendNativeMessage).toHaveBeenCalledTimes(1)
  })

  it('uploads original bytes and the exact build-fingerprinted request', async () => {
    const { manager } = sessionManager()
    let body: FormData | undefined
    const client = new CompanionClient(manager, async (input, init) => {
      expect(String(input)).toBe('http://127.0.0.1:43127/jobs')
      body = init?.body as FormData
      return new Response(
        JSON.stringify({
          buildFingerprint: BUILD_FINGERPRINT,
          jobId: 'fixture-job',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    })
    const metadata = request()
    expect(await client.createJob(pngHeader(), metadata)).toBe('fixture-job')
    expect(body?.get('image')).toBeInstanceOf(Blob)
    const requestPart = body?.get('request')
    expect(requestPart).toBeInstanceOf(Blob)
    expect(JSON.parse(await (requestPart as Blob).text())).toEqual(metadata)
  })

  it('uses only progressive viewport, update, patch, and delete root routes', async () => {
    const { manager } = sessionManager()
    const requests: Array<{ url: string; method: string; body?: string }> = []
    const client = new CompanionClient(manager, async (input, init) => {
      const url = String(input)
      const method = init?.method ?? 'GET'
      requests.push({
        url,
        method,
        ...(typeof init?.body === 'string' ? { body: init.body } : {}),
      })
      if (url.includes('/updates?')) {
        return new Response(
          JSON.stringify({
            jobId: 'job',
            nextSequence: 5,
            updates: [
              {
                sequence: 5,
                type: 'progress',
                stage: 'ocr',
                message: 'Reading',
              },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        )
      }
      if (url.includes('/blobs/')) {
        return new Response(Uint8Array.of(137, 80, 78, 71), {
          status: 200,
          headers: { 'Content-Type': 'image/png' },
        })
      }
      return new Response(null, { status: 204 })
    })

    await client.updateViewport('job', {
      visibleRects: [{ x: 0, y: 0.2, width: 1, height: 0.4 }],
      active: true,
    })
    expect((await client.getJobUpdates('job', 4)).nextSequence).toBe(5)
    expect(await client.getPatch('patch/1', 'image/png')).toBeInstanceOf(ArrayBuffer)
    await client.cancelJob('job')

    expect(requests).toEqual([
      {
        url: 'http://127.0.0.1:43127/jobs/job/viewport',
        method: 'PUT',
        body: JSON.stringify({
          visibleRects: [{ x: 0, y: 0.2, width: 1, height: 0.4 }],
          active: true,
        }),
      },
      {
        url: 'http://127.0.0.1:43127/jobs/job/updates?after=4&waitMs=20000',
        method: 'GET',
      },
      {
        url: 'http://127.0.0.1:43127/blobs/patch%2F1',
        method: 'GET',
      },
      {
        url: 'http://127.0.0.1:43127/jobs/job',
        method: 'DELETE',
      },
    ])
    expect(requests.some(({ url }) => url.includes('/result'))).toBe(false)
    expect(requests.some(({ url }) => url.includes('/browser/v1'))).toBe(false)
  })

  it('keeps setup, lookup, and fonts on authenticated root routes', async () => {
    const { manager } = sessionManager()
    const requests: string[] = []
    const client = new CompanionClient(manager, async (input, init) => {
      const url = String(input)
      requests.push(`${init?.method ?? 'GET'} ${url}`)
      if (url.endsWith('/setup')) {
        return new Response(
          JSON.stringify({ state: 'ready', modelId: 'qwen3.5-4b', message: 'Ready' }),
          {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          },
        )
      }
      if (url.endsWith('/lookup')) {
        return new Response(JSON.stringify({ selectedText: '我', tokens: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      return new Response(Uint8Array.of(0, 1, 0, 0), {
        status: 200,
        headers: { 'Content-Type': 'font/ttf' },
      })
    })
    expect((await client.getSetupStatus()).state).toBe('ready')
    expect((await client.lookup({ selectedText: '我' })).selectedText).toBe('我')
    expect([...new Uint8Array(await client.getFont('hmt-sans'))]).toEqual([0, 1, 0, 0])
    expect(requests).toEqual([
      'GET http://127.0.0.1:43127/setup',
      'POST http://127.0.0.1:43127/lookup',
      'GET http://127.0.0.1:43127/fonts/hmt-sans',
    ])
  })
})
