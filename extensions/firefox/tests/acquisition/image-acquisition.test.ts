import { describe, expect, it, vi } from 'vitest'

import { acquireRemoteImage } from '../../src/acquisition/image-acquisition'
import { sha256Hex } from '../../src/acquisition/hash'
import { pngHeader } from '../helpers/images'

describe('background image acquisition', () => {
  it('requests only the exact redirect origin and validates the final body', async () => {
    const requested: string[][] = []
    const permissions = {
      contains: vi.fn(async () => false),
      request: vi.fn(async ({ origins }: browser.permissions.Permissions) => {
        requested.push(origins ?? [])
        return true
      }),
    }
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('reader.test')) {
        return new Response(null, {
          status: 302,
          headers: { Location: 'https://cdn.test/pages/1.png' },
        })
      }
      return new Response(pngHeader(), {
        status: 200,
        headers: { 'Content-Type': 'image/png' },
      })
    })
    const result = await acquireRemoteImage(
      'https://reader.test/page.png',
      { pageOrigin: 'https://reader.test' },
      permissions,
      fetcher,
    )
    expect(result.finalUrl).toBe('https://cdn.test/pages/1.png')
    expect(result.width).toBe(1200)
    expect(requested).toEqual([['https://cdn.test/*']])
  })

  it('tries credentials only after an unauthenticated 401', async () => {
    const credentials: RequestCredentials[] = []
    const fetcher = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      credentials.push(init?.credentials ?? 'same-origin')
      if (credentials.length === 1) return new Response(null, { status: 401 })
      return new Response(pngHeader(), {
        status: 200,
        headers: { 'Content-Type': 'image/png' },
      })
    })
    const permissions = {
      contains: vi.fn(async () => true),
      request: vi.fn(async () => true),
    }
    await acquireRemoteImage(
      'https://cdn.test/private.png',
      { pageOrigin: 'https://reader.test' },
      permissions,
      fetcher,
    )
    expect(credentials).toEqual(['omit', 'include'])
  })

  it('rejects unsafe redirect chains and oversized content-length', async () => {
    const permissions = {
      contains: vi.fn(async () => true),
      request: vi.fn(async () => true),
    }
    await expect(
      acquireRemoteImage(
        'https://cdn.test/private.png',
        { pageOrigin: 'https://reader.test' },
        permissions,
        async () =>
          new Response(null, {
            status: 302,
            headers: { Location: 'file:///secret.png' },
          }),
      ),
    ).rejects.toThrow(/HTTP and HTTPS/i)
    await expect(
      acquireRemoteImage(
        'https://cdn.test/large.png',
        { pageOrigin: 'https://reader.test' },
        permissions,
        async () =>
          new Response(null, {
            status: 200,
            headers: {
              'Content-Type': 'image/png',
              'Content-Length': String(30 * 1024 * 1024),
            },
          }),
      ),
    ).rejects.toThrow(/exceeds/i)
  })

  it('calculates a deterministic SHA-256', async () => {
    const digest = await sha256Hex(new TextEncoder().encode('abc').buffer)
    expect(digest).toBe(
      'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
    )
  })
})
