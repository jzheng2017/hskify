import { afterEach, describe, expect, it, vi } from 'vitest'

import { waitForVisibleSource } from '../../src/acquisition/rendered-region'

describe('source visibility suspension', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    document.body.replaceChildren()
  })
  it('waits for visibility without polling and responds to cancellation', async () => {
    let hidden = true
    vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden)
    const element = document.createElement('canvas')
    document.body.append(element)
    const abort = new AbortController()
    const pending = waitForVisibleSource(element, abort.signal)
    abort.abort()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
    const visible = waitForVisibleSource(element)
    hidden = false
    document.dispatchEvent(new Event('visibilitychange'))
    await expect(visible).resolves.toBeUndefined()
  })
})
