import { describe, expect, it, vi } from 'vitest'

import { fetchImageWithPageReferrer } from '../../src/acquisition/page-referrer-fetch'

describe('page-referrer image fetch', () => {
  it('restores only the page origin for the marked extension request', async () => {
    let listener:
      | ((
          details: browser.webRequest._OnBeforeSendHeadersDetails,
        ) => browser.webRequest.BlockingResponse | void)
      | undefined
    const event = {
      addListener: vi.fn((callback) => {
        listener = callback
      }),
      removeListener: vi.fn(),
      hasListener: vi.fn(() => false),
    }
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const headers = [...new Headers(init?.headers)].map(([name, value]) => ({ name, value }))
      const modified = listener?.({
        requestId: 'request-1',
        url: String(input),
        method: 'GET',
        frameId: -1,
        parentFrameId: -1,
        tabId: -1,
        type: 'xmlhttprequest',
        timeStamp: 0,
        thirdParty: true,
        requestHeaders: headers,
      })
      expect(modified?.requestHeaders).toContainEqual({
        name: 'Referer',
        value: 'https://reader.test/',
      })
      expect(
        modified?.requestHeaders?.some(
          (header) => header.name.toLowerCase() === 'x-hskify-request-context',
        ),
      ).toBe(false)
      return new Response()
    })

    await fetchImageWithPageReferrer(
      new URL('https://cdn.test/chapter/1.png'),
      { headers: { Accept: 'image/png' } },
      'https://reader.test',
      {
        event,
        fetcher,
        createToken: () => 'private-token',
      },
    )

    expect(event.addListener).toHaveBeenCalledWith(
      expect.any(Function),
      { urls: ['http://*/*', 'https://*/*'], types: ['xmlhttprequest'] },
      ['blocking', 'requestHeaders'],
    )
    expect(event.removeListener).toHaveBeenCalledWith(listener)
  })

  it('does not modify an unrelated request and always removes its listener', async () => {
    let listener:
      | ((
          details: browser.webRequest._OnBeforeSendHeadersDetails,
        ) => browser.webRequest.BlockingResponse | void)
      | undefined
    const event = {
      addListener: vi.fn((callback) => {
        listener = callback
      }),
      removeListener: vi.fn(),
      hasListener: vi.fn(() => false),
    }
    const failure = new Error('network failed')
    const fetcher = vi.fn(async () => {
      expect(
        listener?.({
          requestId: 'other',
          url: 'https://cdn.test/chapter/1.png',
          method: 'GET',
          frameId: -1,
          parentFrameId: -1,
          tabId: -1,
          type: 'xmlhttprequest',
          timeStamp: 0,
          thirdParty: true,
          requestHeaders: [{ name: 'X-Hskify-Request-Context', value: 'other-token' }],
        }),
      ).toBeUndefined()
      throw failure
    })

    await expect(
      fetchImageWithPageReferrer(
        new URL('https://cdn.test/chapter/1.png'),
        {},
        'https://reader.test',
        {
          event,
          fetcher,
          createToken: () => 'private-token',
        },
      ),
    ).rejects.toBe(failure)
    expect(event.removeListener).toHaveBeenCalledWith(listener)
  })
})
