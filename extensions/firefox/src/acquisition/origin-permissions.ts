import type { StorageArea } from '../messaging/settings'

const PENDING_ORIGIN_PREFIX = 'hmt.pendingImageOrigins.'
const MAX_PENDING_ORIGINS = 128
const EXACT_ORIGIN_PATTERN = /^https?:\/\/(?:\[[0-9a-f:]+\]|[^/:*]+)\/\*$/iu

export class OriginPatternError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'OriginPatternError'
  }
}

function httpUrl(value: string, base?: string): URL {
  let url: URL
  try {
    url = new URL(value, base)
  } catch (error) {
    throw new OriginPatternError('The image URL is invalid.', { cause: error })
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new OriginPatternError('Only HTTP and HTTPS origins can be requested.')
  }
  return url
}

/**
 * Firefox match patterns do not have a port component. `URL.origin` must not be
 * used here because it retains explicit ports such as `:8443`.
 */
export function firefoxOriginPattern(value: string, base?: string): string {
  const url = httpUrl(value, base)
  return `${url.protocol}//${url.hostname}/*`
}

export function requiredCrossOriginPatterns(
  pageUrl: string,
  sourceUrls: readonly string[],
): string[] {
  const page = httpUrl(pageUrl)
  const patterns = new Set<string>()
  for (const value of sourceUrls) {
    let source: URL
    try {
      source = new URL(value, page)
    } catch {
      continue
    }
    if (
      (source.protocol !== 'http:' && source.protocol !== 'https:') ||
      source.origin === page.origin
    ) {
      continue
    }
    patterns.add(firefoxOriginPattern(source.href))
  }
  return [...patterns].sort()
}

function isExactOriginPattern(value: unknown): value is string {
  return typeof value === 'string' && EXACT_ORIGIN_PATTERN.test(value)
}

export class PendingOriginPermissionStore {
  constructor(private readonly storage: StorageArea = browser.storage.session) {}

  private key(tabId: number): string {
    return `${PENDING_ORIGIN_PREFIX}${tabId}`
  }

  async list(tabId: number): Promise<string[]> {
    const key = this.key(tabId)
    const values = await this.storage.get(key)
    const stored = values[key]
    if (!Array.isArray(stored)) return []
    return [...new Set(stored.filter(isExactOriginPattern))]
      .sort()
      .slice(0, MAX_PENDING_ORIGINS)
  }

  async add(tabId: number, origin: string): Promise<void> {
    if (!isExactOriginPattern(origin)) {
      throw new OriginPatternError('The pending image origin pattern is invalid.')
    }
    const origins = [...new Set([...(await this.list(tabId)), origin])]
      .sort()
      .slice(0, MAX_PENDING_ORIGINS)
    await this.storage.set({ [this.key(tabId)]: origins })
  }

  async replace(tabId: number, origins: readonly string[]): Promise<void> {
    const valid = [...new Set(origins.filter(isExactOriginPattern))]
      .sort()
      .slice(0, MAX_PENDING_ORIGINS)
    if (valid.length === 0) {
      await this.removeForTab(tabId)
      return
    }
    await this.storage.set({ [this.key(tabId)]: valid })
  }

  async removeForTab(tabId: number): Promise<void> {
    await this.storage.remove(this.key(tabId))
  }
}
