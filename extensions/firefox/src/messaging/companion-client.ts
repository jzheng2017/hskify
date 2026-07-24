import {
  PROTOCOL_VERSION,
  parseBrowserJobCreated,
  parseBrowserJobResult,
  parseBrowserJobStatus,
  parseErrorResponse,
  parseHealthResponse,
  parseLookupResult,
  type BrowserJobRequest,
  type BrowserJobResult,
  type BrowserJobStatus,
  type LookupRequest,
  type LookupResult,
  type NativeReadyResponse,
} from '../contracts/browser'
import { NativeSessionManager } from './native-session'

const MAX_CLEAN_IMAGE_BYTES = 25 * 1024 * 1024
const MAX_FONT_BYTES = 12 * 1024 * 1024

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
    private readonly fetcher: FetchLike = fetch,
  ) {}

  private headers(session: NativeReadyResponse, init: RequestInit): Headers {
    const headers = new Headers(init.headers)
    headers.set('Authorization', `Bearer ${session.token}`)
    headers.set('X-HSK-Manga-Protocol', String(PROTOCOL_VERSION))
    return headers
  }

  private fetchSession(
    session: NativeReadyResponse,
    path: string,
    init: RequestInit,
  ): Promise<Response> {
    return this.fetcher(
      `http://127.0.0.1:${session.port}/browser/v1${path}`,
      {
        ...init,
        headers: this.headers(session, init),
        redirect: 'error',
      },
    )
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
      const response = await this.fetchSession(lease.session, '/health', {})
      if (!response.ok) throw await responseError(response)
      parseHealthResponse(await parseJsonResponse(response))
      this.validatedSessionToken = lease.session.token
      return lease.session
    } catch {
      return this.freshSession()
    }
  }

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    let session = await this.liveSession()
    let response: Response
    try {
      response = await this.fetchSession(session, path, init)
    } catch (error) {
      session = await this.freshSession()
      try {
        response = await this.fetchSession(session, path, init)
      } catch (retryError) {
        await this.sessions.invalidate()
        this.validatedSessionToken = undefined
        throw this.transportError(retryError ?? error)
      }
    }
    if (response.status === 401) {
      session = await this.freshSession()
      try {
        response = await this.fetchSession(session, path, init)
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

  async getJobStatus(jobId: string): Promise<BrowserJobStatus> {
    const response = await this.request(`/jobs/${encodeURIComponent(jobId)}`)
    return parseBrowserJobStatus(await parseJsonResponse(response))
  }

  async getJobResult(jobId: string): Promise<BrowserJobResult> {
    const response = await this.request(`/jobs/${encodeURIComponent(jobId)}/result`)
    return parseBrowserJobResult(await parseJsonResponse(response))
  }

  async cancelJob(jobId: string): Promise<void> {
    await this.request(`/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' })
  }

  async lookup(request: LookupRequest): Promise<LookupResult> {
    const response = await this.request('/lookup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return parseLookupResult(await parseJsonResponse(response))
  }

  async getCleanImage(blobId: string, expectedMimeType: string): Promise<ArrayBuffer> {
    const response = await this.request(`/blobs/${encodeURIComponent(blobId)}`)
    const mimeType = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase()
    if (mimeType !== expectedMimeType) {
      throw new CompanionHttpError(
        'INVALID_CLEAN_IMAGE_MIME',
        'The local translation engine returned an unexpected clean-image type.',
        false,
      )
    }
    return boundedArrayBuffer(response, MAX_CLEAN_IMAGE_BYTES)
  }

  async getFont(fontId: string): Promise<ArrayBuffer> {
    const response = await this.request(`/fonts/${encodeURIComponent(fontId)}`)
    const mimeType = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase()
    if (
      mimeType !== 'font/woff2' &&
      mimeType !== 'font/woff' &&
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
