import {
  DEFAULT_IMAGE_LIMITS,
  ImageValidationError,
  normalizeMimeType,
  validateImageBytes,
  type ImageLimits,
} from './image-format'

const MAX_REDIRECTS = 3

export type PermissionApi = {
  contains(permissions: browser.permissions.Permissions): Promise<boolean>
  request(permissions: browser.permissions.Permissions): Promise<boolean>
}

export type AcquisitionOptions = {
  pageOrigin: string
  limits?: ImageLimits
}

export type AcquiredImage = {
  bytes: ArrayBuffer
  mimeType: 'image/png' | 'image/jpeg' | 'image/webp' | 'image/gif'
  width: number
  height: number
  finalUrl: string
}

function optionalOriginPattern(url: URL): string {
  return `${url.origin}/*`
}

function safeHttpUrl(value: string, base?: URL): URL {
  let url: URL
  try {
    url = new URL(value, base)
  } catch (error) {
    throw new ImageValidationError('INVALID_IMAGE_URL', 'The image URL is invalid.', false, {
      cause: error,
    })
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new ImageValidationError(
      'UNSUPPORTED_IMAGE_URL',
      'Only HTTP and HTTPS image URLs can be acquired in the background.',
    )
  }
  url.hash = ''
  return url
}

async function ensureOriginPermission(
  url: URL,
  pageOrigin: string,
  permissions: PermissionApi,
): Promise<void> {
  if (url.origin === pageOrigin) return
  const origins = [optionalOriginPattern(url)]
  if (await permissions.contains({ origins })) return
  const granted = await permissions.request({ origins })
  if (!granted) {
    throw new ImageValidationError(
      'IMAGE_PERMISSION_DENIED',
      `Permission to read images from ${url.origin} was denied.`,
      true,
    )
  }
}

async function readBoundedBody(response: Response, maximumBytes: number): Promise<ArrayBuffer> {
  const contentLength = response.headers.get('content-length')
  if (contentLength !== null) {
    const parsed = Number(contentLength)
    if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > maximumBytes) {
      throw new ImageValidationError(
        'IMAGE_TOO_LARGE',
        `The image exceeds the ${maximumBytes} byte limit.`,
      )
    }
  }
  if (!response.body) {
    const bytes = await response.arrayBuffer()
    if (bytes.byteLength > maximumBytes) {
      throw new ImageValidationError(
        'IMAGE_TOO_LARGE',
        `The image exceeds the ${maximumBytes} byte limit.`,
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
      if (total > maximumBytes) {
        await reader.cancel()
        throw new ImageValidationError(
          'IMAGE_TOO_LARGE',
          `The image exceeds the ${maximumBytes} byte limit.`,
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

async function fetchWithRedirectChecks(
  initialUrl: URL,
  pageOrigin: string,
  permissions: PermissionApi,
  fetcher: typeof fetch,
  credentials: RequestCredentials,
): Promise<{ response: Response; finalUrl: URL }> {
  let current = initialUrl
  for (let redirects = 0; redirects <= MAX_REDIRECTS; redirects += 1) {
    await ensureOriginPermission(current, pageOrigin, permissions)
    const response = await fetcher(current, {
      method: 'GET',
      credentials,
      cache: 'no-store',
      redirect: 'manual',
      referrerPolicy: 'no-referrer',
      headers: {
        Accept: 'image/png,image/jpeg,image/webp,image/gif',
      },
    })
    if (response.status < 300 || response.status >= 400) {
      return { response, finalUrl: current }
    }
    if (redirects === MAX_REDIRECTS) {
      throw new ImageValidationError(
        'IMAGE_REDIRECT_LIMIT',
        'The image URL redirected too many times.',
      )
    }
    const location = response.headers.get('location')
    if (!location) {
      throw new ImageValidationError(
        'IMAGE_REDIRECT_UNREADABLE',
        'The image redirect could not be validated safely.',
      )
    }
    const next = safeHttpUrl(location, current)
    if (next.origin !== current.origin && credentials === 'include') {
      throw new ImageValidationError(
        'CROSS_ORIGIN_CREDENTIAL_REDIRECT',
        'An authenticated image redirected to a different origin.',
      )
    }
    current = next
  }
  throw new ImageValidationError('IMAGE_REDIRECT_LIMIT', 'The image URL redirected too many times.')
}

export async function acquireRemoteImage(
  sourceUrl: string,
  options: AcquisitionOptions,
  permissions: PermissionApi = browser.permissions,
  fetcher: typeof fetch = fetch,
): Promise<AcquiredImage> {
  const limits = options.limits ?? DEFAULT_IMAGE_LIMITS
  const initialUrl = safeHttpUrl(sourceUrl)
  let fetched = await fetchWithRedirectChecks(
    initialUrl,
    options.pageOrigin,
    permissions,
    fetcher,
    'omit',
  )
  if (fetched.response.status === 401 || fetched.response.status === 403) {
    fetched = await fetchWithRedirectChecks(
      initialUrl,
      options.pageOrigin,
      permissions,
      fetcher,
      'include',
    )
  }
  if (!fetched.response.ok) {
    throw new ImageValidationError(
      'IMAGE_FETCH_FAILED',
      `The image request failed with HTTP ${fetched.response.status}.`,
      fetched.response.status >= 500,
    )
  }
  const declaredMimeType = normalizeMimeType(fetched.response.headers.get('content-type'))
  const bytes = await readBoundedBody(fetched.response, limits.maximumBytes)
  const validated = validateImageBytes(bytes, declaredMimeType, limits)
  return {
    bytes,
    mimeType: validated.mimeType,
    width: validated.dimensions.width,
    height: validated.dimensions.height,
    finalUrl: fetched.finalUrl.href,
  }
}

export function validateInlineImage(
  bytes: ArrayBuffer,
  declaredMimeType: string | undefined,
  limits: ImageLimits = DEFAULT_IMAGE_LIMITS,
): AcquiredImage {
  const validated = validateImageBytes(bytes, declaredMimeType, limits)
  return {
    bytes,
    mimeType: validated.mimeType,
    width: validated.dimensions.width,
    height: validated.dimensions.height,
    finalUrl: 'inline://content-script',
  }
}
