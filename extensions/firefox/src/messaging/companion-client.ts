import {
  parseBrowserJobCreated,
  parseBrowserSetupStatus,
  parseErrorResponse,
  parseHealthResponse,
  parseJobUpdateBatch,
  parseLookupResult,
  type BrowserJobRequest,
  type JobUpdateBatch,
  type BrowserSetupStatus,
  type LookupRequest,
  type LookupResult,
  type NativeReadyResponse,
  type ViewportUpdate,
} from '../contracts/browser'
import { NativeSessionManager } from './native-session'

const MAX_PATCH_BYTES = 25 * 1024 * 1024
const MAX_FONT_BYTES = 32 * 1024 * 1024
const EXTENSION_ORIGIN_HEADER = 'X-HSK-Manga-Extension-Origin'
export const UPDATE_WAIT_MS = 20_000
export const REQUEST_TIMEOUT_MS = 30_000
export const UPDATE_TIMEOUT_GRACE_MS = 5_000

export class CompanionHttpError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable: boolean,
    readonly status?: number,
    options?: ErrorOptions,
  ) {
    super(message, options)
    this.name = 'CompanionHttpError'
  }
}

async function parseJsonResponse(response: Response): Promise<unknown> {
  const text = await response.text()
  try {
    return JSON.parse(text) as unknown
  } catch (error) {
    throw new CompanionHttpError(
      'INVALID_COMPANION_RESPONSE',
      'The local translation engine returned malformed JSON.',
      true,
      response.status,
      { cause: error },
    )
  }
}

async function responseError(response: Response): Promise<CompanionHttpError> {
  try {
    const payload = parseErrorResponse(await parseJsonResponse(response))
    return new CompanionHttpError(
      payload.code,
      payload.message,
      payload.retryable,
      response.status,
    )
  } catch (error) {
    if (error instanceof CompanionHttpError && error.code !== 'INVALID_COMPANION_RESPONSE') {
      return error
    }
    return new CompanionHttpError(
      `HTTP_${response.status}`,
      `The local translation engine request failed with HTTP ${response.status}.`,
      response.status >= 500,
      response.status,
    )
  }
}

async function boundedArrayBuffer(response: Response, maximum: number): Promise<ArrayBuffer> {
  const header = response.headers.get('content-length')
  if (header !== null) {
    const length = Number(header)
    if (!Number.isSafeInteger(length) || length < 0 || length > maximum) {
      throw new CompanionHttpError(
        'BINARY_RESPONSE_TOO_LARGE',
        'The local translation engine returned an oversized binary response.',
        false,
      )
    }
  }
  if (!response.body) {
    const bytes = await response.arrayBuffer()
    if (bytes.byteLength > maximum) {
      throw new CompanionHttpError(
        'BINARY_RESPONSE_TOO_LARGE',
        'The local translation engine returned an oversized binary response.',
        false,
      )
    }
    return bytes
  }
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let total = 0
  try {
    while (true) {
      const item = await reader.read()
      if (item.done) break
      total += item.value.byteLength
      if (total > maximum) {
        await reader.cancel()
        throw new CompanionHttpError(
          'BINARY_RESPONSE_TOO_LARGE',
          'The local translation engine returned an oversized binary response.',
          false,
        )
      }
      chunks.push(item.value)
    }
  } finally {
    reader.releaseLock()
  }
  const merged = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.byteLength
  }
  return merged.buffer
}

export type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

export class CompanionClient {
  private validatedSessionToken: string | undefined

  constructor(
    private readonly sessions = new NativeSessionManager(),
    private readonly fetcher: FetchLike = (input, init) => fetch(input, init),
  ) {}

  private headers(session: NativeReadyResponse, init: RequestInit): Headers {
    const headers = new Headers(init.headers)
    headers.set('Authorization', `Bearer ${session.token}`)
    headers.set(EXTENSION_ORIGIN_HEADER, this.sessions.origin())
    return headers
  }

  private async fetchSession(
    session: NativeReadyResponse,
    path: string,
    init: RequestInit,
    timeoutMs = REQUEST_TIMEOUT_MS,
  ): Promise<Response> {
    const controller = new AbortController()
    const upstreamSignal = init.signal
    const abortFromUpstream = (): void => controller.abort(upstreamSignal?.reason)
    if (upstreamSignal?.aborted) abortFromUpstream()
    else upstreamSignal?.addEventListener('abort', abortFromUpstream, { once: true })
    const timer = setTimeout(() => controller.abort(new DOMException('Timed out', 'TimeoutError')), timeoutMs)
    try {
      return await this.fetcher(`http://127.0.0.1:${session.port}${path}`, {
        ...init,
        headers: this.headers(session, init),
        redirect: 'error',
        signal: controller.signal,
      })
    } finally {
      clearTimeout(timer)
      upstreamSignal?.removeEventListener('abort', abortFromUpstream)
    }
  }

  private transportError(error: unknown): CompanionHttpError {
    return new CompanionHttpError(
      'COMPANION_TRANSPORT_FAILED',
      'The local translation engine connection failed.',
      true,
      undefined,
      { cause: error },
    )
  }

  private async freshSession(): Promise<NativeReadyResponse> {
    this.validatedSessionToken = undefined
    await this.sessions.invalidate()
    const session = await this.sessions.getOrLaunch(true)
    // A successful one-shot handshake proves that the newly issued endpoint is
    // live. Cached endpoints are checked separately below.
    this.validatedSessionToken = session.token
    return session
  }

  private async liveSession(): Promise<NativeReadyResponse> {
    const lease = await this.sessions.getOrLaunchWithState()
    if (!lease.reused || this.validatedSessionToken === lease.session.token) {
      this.validatedSessionToken = lease.session.token
      return lease.session
    }
    try {
      const response = await this.fetchSession(lease.session, '/health', {}, 5_000)
      if (!response.ok) throw await responseError(response)
      parseHealthResponse(await parseJsonResponse(response))
      this.validatedSessionToken = lease.session.token
      return lease.session
    } catch {
      return this.freshSession()
    }
  }

  private async request(
    path: string,
    init: RequestInit = {},
    timeoutMs = REQUEST_TIMEOUT_MS,
  ): Promise<Response> {
    let session = await this.liveSession()
    let response: Response
    try {
      response = await this.fetchSession(session, path, init, timeoutMs)
    } catch (error) {
      session = await this.freshSession()
      try {
        response = await this.fetchSession(session, path, init, timeoutMs)
      } catch (retryError) {
        await this.sessions.invalidate()
        this.validatedSessionToken = undefined
        throw this.transportError(retryError ?? error)
      }
    }
    if (response.status === 401) {
      session = await this.freshSession()
      try {
        response = await this.fetchSession(session, path, init, timeoutMs)
      } catch (error) {
        await this.sessions.invalidate()
        this.validatedSessionToken = undefined
        throw this.transportError(error)
      }
    }
    if (response.status === 401) {
      await this.sessions.invalidate()
      this.validatedSessionToken = undefined
    }
    if (!response.ok) throw await responseError(response)
    return response
  }

  async createJob(bytes: ArrayBuffer, request: BrowserJobRequest): Promise<string> {
    const form = new FormData()
    form.append('image', new Blob([bytes], { type: request.sourceMimeType }), 'source-image')
    form.append(
      'request',
      new Blob([JSON.stringify(request)], { type: 'application/json' }),
      'request.json',
    )
    const response = await this.request('/jobs', { method: 'POST', body: form })
    return parseBrowserJobCreated(await parseJsonResponse(response)).jobId
  }

  async updateViewport(jobId: string, viewport: ViewportUpdate): Promise<void> {
    await this.request(`/jobs/${encodeURIComponent(jobId)}/viewport`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(viewport),
    })
  }

  async getJobUpdates(
    jobId: string,
    after: number,
    waitMs = UPDATE_WAIT_MS,
  ): Promise<JobUpdateBatch> {
    const query = new URLSearchParams({
      after: String(after),
      waitMs: String(waitMs),
    })
    const response = await this.request(
      `/jobs/${encodeURIComponent(jobId)}/updates?${query.toString()}`,
      {},
      waitMs + UPDATE_TIMEOUT_GRACE_MS,
    )
    return parseJobUpdateBatch(await parseJsonResponse(response), after)
  }

  async cancelJob(jobId: string): Promise<void> {
    await this.request(`/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' })
  }

  async getSetupStatus(): Promise<BrowserSetupStatus> {
    const response = await this.request('/setup')
    return parseBrowserSetupStatus(await parseJsonResponse(response))
  }

  async startModelSetup(): Promise<BrowserSetupStatus> {
    const response = await this.request('/setup/models', { method: 'POST' })
    return parseBrowserSetupStatus(await parseJsonResponse(response))
  }

  async lookup(request: LookupRequest): Promise<LookupResult> {
    const response = await this.request('/lookup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return parseLookupResult(await parseJsonResponse(response))
  }

  async getPatch(blobId: string, expectedMimeType: 'image/png'): Promise<ArrayBuffer> {
    const response = await this.request(`/blobs/${encodeURIComponent(blobId)}`)
    const mimeType = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase()
    if (mimeType !== expectedMimeType) {
      throw new CompanionHttpError(
        'INVALID_PATCH_MIME',
        'The local translation engine returned an unexpected patch image type.',
        false,
      )
    }
    return boundedArrayBuffer(response, MAX_PATCH_BYTES)
  }

  async getFont(fontId: string): Promise<ArrayBuffer> {
    const response = await this.request(`/fonts/${encodeURIComponent(fontId)}`)
    const mimeType = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase()
    if (
      mimeType !== 'font/woff2' &&
      mimeType !== 'font/woff' &&
      mimeType !== 'font/ttf' &&
      mimeType !== 'font/otf' &&
      mimeType !== 'application/font-woff'
    ) {
      throw new CompanionHttpError(
        'INVALID_FONT_MIME',
        'The local translation engine returned an unexpected font type.',
        false,
      )
    }
    return boundedArrayBuffer(response, MAX_FONT_BYTES)
  }
}
