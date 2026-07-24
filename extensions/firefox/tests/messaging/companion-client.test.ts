import { describe, expect, it, vi } from 'vitest'

import { CompanionClient } from '../../src/messaging/companion-client'
import {
  NativeSessionManager,
  SESSION_STORAGE_KEY,
} from '../../src/messaging/native-session'
import type { BrowserJobRequest } from '../../src/contracts/browser'
import { pngHeader } from '../helpers/images'
import { MemoryStorage } from '../helpers/storage'

function ready(token: string) {
  return {
    type: 'ready',
    protocolVersion: 1,
    engineVersion: '0.1.0',
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
    sendNativeMessage: vi.fn(async () =>
      ready((calls++ === 0 ? 'A' : 'B').repeat(43)),
    ),
  }
  return { manager: new NativeSessionManager(new MemoryStorage(), runtime), runtime }
}

describe('authenticated localhost companion client', () => {
  it('re-handshakes and retries exactly once after a 401', async () => {
    const { manager, runtime } = sessionManager()
    const authorizations: string[] = []
    const fetcher = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      authorizations.push(new Headers(init?.headers).get('Authorization') ?? '')
      if (authorizations.length === 1) return new Response(null, { status: 401 })
      return new Response(
        JSON.stringify({
          revision: 1,
          jobId: 'job',
          state: 'running',
          stage: 'queued',
          message: 'Queued',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    })
    const client = new CompanionClient(manager, fetcher)
    expect((await client.getJobStatus('job')).stage).toBe('queued')
    expect(authorizations).toEqual([
      `Bearer ${'A'.repeat(43)}`,
      `Bearer ${'B'.repeat(43)}`,
    ])
    expect(runtime.sendNativeMessage).toHaveBeenCalledTimes(2)
    expect(fetcher).toHaveBeenCalledTimes(2)
    expect(
      new Headers(fetcher.mock.calls[1]?.[1]?.headers).get('X-HSK-Manga-Protocol'),
    ).toBe('1')
  })

  it('health-checks a cached endpoint and re-handshakes after transport failure', async () => {
    const storage = new MemoryStorage()
    storage.values[SESSION_STORAGE_KEY] = ready('A'.repeat(43))
    const runtime = {
      getManifest: () => ({ version: '0.1.0' }),
      getURL: () => 'moz-extension://fixture/',
      sendNativeMessage: vi.fn(async () => ready('B'.repeat(43))),
    }
    const manager = new NativeSessionManager(storage, runtime)
    const requests: string[] = []
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      requests.push(String(input))
      if (requests.length === 1) throw new TypeError('stale cached port')
      return new Response(
        JSON.stringify({
          revision: 1,
          jobId: 'job',
          state: 'running',
          stage: 'queued',
          message: 'Queued',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    })
    const client = new CompanionClient(manager, fetcher)
    expect((await client.getJobStatus('job')).jobId).toBe('job')
    expect(requests[0]).toContain('/health')
    expect(requests[1]).toContain('/jobs/job')
    expect(runtime.sendNativeMessage).toHaveBeenCalledTimes(1)
  })

  it('invalidates and retries once when an established request loses transport', async () => {
    const { manager, runtime } = sessionManager()
    let calls = 0
    const client = new CompanionClient(manager, async () => {
      calls += 1
      if (calls === 1) throw new TypeError('connection refused')
      return new Response(
        JSON.stringify({
          revision: 1,
          jobId: 'job',
          state: 'running',
          stage: 'queued',
          message: 'Queued',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    })
    expect((await client.getJobStatus('job')).stage).toBe('queued')
    expect(runtime.sendNativeMessage).toHaveBeenCalledTimes(2)
    expect(calls).toBe(2)
  })

  it('sends original bytes and frozen metadata as multipart form data', async () => {
    const { manager } = sessionManager()
    let body: FormData | undefined
    const client = new CompanionClient(manager, async (_input, init) => {
      body = init?.body as FormData
      return new Response(JSON.stringify({ protocolVersion: 1, jobId: 'fixture-job' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })
    const request: BrowserJobRequest = {
      protocolVersion: 1,
      clientImageId: 'page-0-hash',
      sourceSha256: 'a'.repeat(64),
      sourceMimeType: 'image/png',
      naturalWidth: 1200,
      naturalHeight: 1800,
      pageSessionId: 'page',
      pageIndex: 0,
      settings: {
        sourceLanguage: 'en',
        targetLanguage: 'zh-CN',
        hskStandard: '2.0',
        hskLevel: 5,
        readingDirection: 'auto',
        translateSoundEffects: false,
      },
    }
    expect(await client.createJob(pngHeader(), request)).toBe('fixture-job')
    expect(body?.get('image')).toBeInstanceOf(Blob)
    const requestPart = body?.get('request')
    expect(requestPart).toBeInstanceOf(Blob)
    expect(JSON.parse(await (requestPart as Blob).text())).toEqual(request)
  })

  it('returns clean-image and font bytes as ArrayBuffers with MIME checks', async () => {
    const { manager } = sessionManager()
    const fiveMegabytes = new Uint8Array(5 * 1024 * 1024)
    fiveMegabytes[0] = 137
    const client = new CompanionClient(manager, async (input) => {
      const isFont = String(input).includes('/fonts/')
      return new Response(isFont ? Uint8Array.of(1, 2, 3) : fiveMegabytes, {
        status: 200,
        headers: {
          'Content-Type': isFont ? 'font/woff2' : 'image/png',
        },
      })
    })
    const clean = await client.getCleanImage('blob', 'image/png')
    const font = await client.getFont('fixture')
    expect(clean).toBeInstanceOf(ArrayBuffer)
    expect(clean.byteLength).toBe(5 * 1024 * 1024)
    expect(new Uint8Array(clean)[0]).toBe(137)
    expect(font).toBeInstanceOf(ArrayBuffer)
    expect([...new Uint8Array(font)]).toEqual([1, 2, 3])
  })

  it('enforces binary caps while streaming before materializing the response', async () => {
    const { manager } = sessionManager()
    let produced = 0
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (produced >= 13) {
          controller.close()
          return
        }
        produced += 1
        controller.enqueue(new Uint8Array(1024 * 1024))
      },
      cancel() {},
    })
    const client = new CompanionClient(
      manager,
      async () =>
        new Response(body, {
          status: 200,
          headers: { 'Content-Type': 'font/woff2' },
        }),
    )
    await expect(client.getFont('oversized')).rejects.toMatchObject({
      code: 'BINARY_RESPONSE_TOO_LARGE',
    })
    expect(produced).toBeLessThan(20)
  })
})
