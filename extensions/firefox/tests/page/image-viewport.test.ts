import { afterEach, describe, expect, it, vi } from 'vitest'

import { sendBackgroundMessage } from '../../src/messaging/messages'
import { ImageFocusReporter } from '../../src/page/image-chapter-mode'

vi.mock('../../src/messaging/messages', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../src/messaging/messages')>()),
  sendBackgroundMessage: vi.fn(async () => undefined),
}))

afterEach(() => vi.restoreAllMocks())

describe('image viewport activity', () => {
  it('sends inactive focus and avoids layout reads while the document is hidden', async () => {
    vi.spyOn(document, 'hidden', 'get').mockReturnValue(true)
    const source = document.createElement('img')
    const bounds = vi.spyOn(source, 'getBoundingClientRect')
    const tracker = new ImageFocusReporter('image-job', source, 700, 900)
    try {
      await vi.waitFor(() =>
        expect(sendBackgroundMessage).toHaveBeenCalledWith({
          type: 'job:focus',
          jobId: 'image-job',
          focus: { kind: 'image', active: false, visibleRects: [] },
        }),
      )
      expect(bounds).not.toHaveBeenCalled()
    } finally {
      await tracker.stop()
    }
  })
})
