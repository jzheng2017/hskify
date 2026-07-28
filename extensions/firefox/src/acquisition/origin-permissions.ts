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
