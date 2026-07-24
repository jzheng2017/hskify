import {
  PROTOCOL_VERSION,
  parseBrowserJobCreated,
  parseBrowserJobResult,
  parseBrowserJobStatus,
  parseErrorResponse,
  parseLookupResult,
  type BrowserJobRequest,
  type BrowserJobResult,
  type BrowserJobStatus,
  type LookupRequest,
  type LookupResult,
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

export type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

export class CompanionClient {
  constructor(
    private readonly sessions = new NativeSessionManager(),
    private readonly fetcher: FetchLike = fetch,
  ) {}

  private async request(
    path: string,
    init: RequestInit = {},
    retryUnauthorized = true,
  ): Promise<Response> {
    const session = await this.sessions.getOrLaunch()
    const headers = new Headers(init.headers)
    headers.set('Authorization', `Bearer ${session.token}`)
    headers.set('X-HSK-Manga-Protocol', String(PROTOCOL_VERSION))
    const response = await this.fetcher(
      `http://127.0.0.1:${session.port}/browser/v1${path}`,
      {
        ...init,
        headers,
        redirect: 'error',
      },
    )
    if (response.status === 401 && retryUnauthorized) {
      await this.sessions.invalidate()
      await this.sessions.getOrLaunch(true)
      return this.request(path, init, false)
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
