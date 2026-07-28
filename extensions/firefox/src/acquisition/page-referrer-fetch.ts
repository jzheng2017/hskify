const REQUEST_CONTEXT_HEADER = 'X-Hskify-Request-Context'
const HTTP_URLS = ['http://*/*', 'https://*/*']

type BeforeSendHeadersEvent = Pick<
  typeof browser.webRequest.onBeforeSendHeaders,
  'addListener' | 'removeListener'
>

export type PageReferrerFetchDependencies = {
  event?: BeforeSendHeadersEvent
  fetcher?: typeof fetch
  createToken?: () => string
}

function pageOriginReferrer(pageOrigin: string): string {
  const url = new URL(pageOrigin)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new TypeError('The page origin must use HTTP or HTTPS.')
  }
  url.pathname = '/'
  url.search = ''
  url.hash = ''
  return url.href
}

/**
 * Fetches an image as the extension while retaining the source page's
 * cross-origin referrer context.
 *
 * Firefox intentionally omits Origin and Referer from privileged extension
 * fetches. Some reader CDNs reject those otherwise-valid requests. A private
 * one-request marker lets the blocking listener identify only this fetch,
 * remove the marker before it reaches the network, and restore the same
 * origin-only Referer a normal cross-origin page image request would send.
 */
export async function fetchImageWithPageReferrer(
  input: URL,
  init: RequestInit,
  pageOrigin: string,
  dependencies: PageReferrerFetchDependencies = {},
): Promise<Response> {
  const event = dependencies.event ?? browser.webRequest.onBeforeSendHeaders
  const fetcher = dependencies.fetcher ?? fetch
  const token = dependencies.createToken?.() ?? crypto.randomUUID()
  const requestHeaders = new Headers(init.headers)
  requestHeaders.set(REQUEST_CONTEXT_HEADER, token)
  const referrer = pageOriginReferrer(pageOrigin)

  const listener = (
    details: browser.webRequest._OnBeforeSendHeadersDetails,
  ): browser.webRequest.BlockingResponse | void => {
    const headers = details.requestHeaders ?? []
    if (
      details.url !== input.href ||
      !headers.some(
        (header) =>
          header.name.toLowerCase() === REQUEST_CONTEXT_HEADER.toLowerCase() &&
          header.value === token,
      )
    ) {
      return
    }

    const forwarded = headers.filter(
      (header) =>
        header.name.toLowerCase() !== REQUEST_CONTEXT_HEADER.toLowerCase() &&
        header.name.toLowerCase() !== 'referer',
    )
    forwarded.push({ name: 'Referer', value: referrer })
    return { requestHeaders: forwarded }
  }

  event.addListener(
    listener,
    { urls: HTTP_URLS, types: ['xmlhttprequest'] },
    ['blocking', 'requestHeaders'],
  )
  try {
    return await fetcher(input, { ...init, headers: requestHeaders })
  } finally {
    event.removeListener(listener)
  }
}
