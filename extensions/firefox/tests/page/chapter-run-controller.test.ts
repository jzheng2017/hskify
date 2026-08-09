import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { JobUpdateBatch, TranslationSettings } from '../../src/contracts/browser'
import { ChapterRunController } from '../../src/page/chapter-run-controller'

const settings: TranslationSettings = {
  sourceLanguage: 'en',
  targetLanguage: 'zh-CN',
  hskStandard: '2.0',
  hskLevel: 3,
  learningMode: 'natural',
}

function installBackend(): ReturnType<typeof vi.fn> {
  const sendMessage = vi.fn(async (raw: unknown) => {
    const message = raw as Record<string, unknown>
    switch (message.type) {
      case 'jobs:recover':
        return { ok: true, value: [] }
      case 'job:updates': {
        const batch: JobUpdateBatch = {
          jobId: String(message.jobId),
          nextSequence: 2,
          updates: [
            {
              sequence: 1,
              type: 'progress',
              stage: 'translating',
              current: 1,
              total: 2,
              message: 'Translating',
            },
            {
              sequence: 2,
              type: 'complete',
              translatedCount: 2,
              preservedCount: 0,
            },
          ],
        }
        return { ok: true, value: batch }
      }
      default:
        return { ok: true, value: undefined }
    }
  })
  vi.stubGlobal('browser', { runtime: { sendMessage } })
  return sendMessage
}

describe('shared chapter run controller', () => {
  beforeEach(() => {
    document.body.replaceChildren()
    sessionStorage.clear()
  })

  afterEach(() => {
    document.documentElement
      .querySelectorAll('[data-hskify-owned]')
      .forEach((element) => element.remove())
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('owns recovery, replay-safe streaming, terminal acknowledgement, and replacement sessions', async () => {
    const sendMessage = installBackend()
    const restore = vi.fn()
    const run = new ChapterRunController('document', restore)
    const first = await run.start(2, 'Preparing document')

    await expect(
      run.recover(first, [
        {
          kind: 'document',
          sourceSha256: 'a'.repeat(64),
          settings,
        },
      ]),
    ).resolves.toEqual([])

    const install = vi.fn()
    await expect(
      run.stream(first, 'document-job', 0, first.signal, install),
    ).resolves.toMatchObject({
      type: 'complete',
      translatedCount: 2,
    })
    expect(install).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'job:ack',
        jobId: 'document-job',
        sequence: 2,
        terminalType: 'complete',
      }),
    )
    expect(run.finish({ current: 2, total: 2 })).toMatchObject({
      state: 'complete',
      contentKind: 'document',
    })
    run.finish({ current: 2, total: 2 })
    expect(
      sendMessage.mock.calls.filter(
        ([message]) => (message as { type?: string }).type === 'chapter:finish',
      ),
    ).toHaveLength(1)

    const second = await run.start(1, 'Preparing replacement')
    expect(second.pageSessionId).not.toBe(first.pageSessionId)
    expect(restore).toHaveBeenCalledTimes(1)
    expect(sendMessage).toHaveBeenCalledWith({
      type: 'jobs:cancel-page',
      pageSessionId: first.pageSessionId,
    })
    expect(sendMessage).not.toHaveBeenCalledWith({
      type: 'chapter:cancel',
      pageSessionId: first.pageSessionId,
      pageUrl: first.pageUrl,
    })
    run.destroy()
  })

  it.each(['image', 'document'] as const)(
    'uses the same cancellation and restoration path for %s chapters',
    async (contentKind) => {
      const sendMessage = installBackend()
      const restore = vi.fn()
      const run = new ChapterRunController(contentKind, restore)
      const token = await run.start(3, 'Preparing chapter')
      run.update({ current: 1, total: 3, message: 'One item ready' })
      run.registerJob(token, `${contentKind}-job`)

      expect(run.cancel({ current: 1, total: 3 })).toMatchObject({
        state: 'cancelled',
        contentKind,
        current: 1,
        total: 3,
      })
      expect(restore).toHaveBeenCalledTimes(1)
      expect(sendMessage).toHaveBeenCalledWith({
        type: 'chapter:cancel',
        pageSessionId: token.pageSessionId,
        pageUrl: token.pageUrl,
      })
      await vi.waitFor(() =>
        expect(sendMessage).toHaveBeenCalledWith({
          type: 'jobs:cancel-page',
          pageSessionId: token.pageSessionId,
        }),
      )
      expect(sendMessage).not.toHaveBeenCalledWith({
        type: 'job:cancel',
        jobId: `${contentKind}-job`,
      })
      expect(() => run.assertCurrent(token)).toThrowError(
        expect.objectContaining({ name: 'AbortError' }),
      )
      run.cancel({ current: 1, total: 3 })
      expect(
        sendMessage.mock.calls.filter(([message]) => {
          const candidate = message as { type?: string; pageSessionId?: string }
          return (
            candidate.type === 'chapter:cancel' && candidate.pageSessionId === token.pageSessionId
          )
        }),
      ).toHaveLength(1)
      run.destroy()
      expect(
        sendMessage.mock.calls.filter(([message]) => {
          const candidate = message as { type?: string; pageSessionId?: string }
          return (
            candidate.type === 'jobs:cancel-page' &&
            candidate.pageSessionId === token.pageSessionId
          )
        }),
      ).toHaveLength(1)
    },
  )

  it('retains a completed page until destroy and releases it exactly once', async () => {
    const sendMessage = installBackend()
    const run = new ChapterRunController('document', vi.fn())
    const token = await run.start(1, 'Preparing document')

    run.finish({ current: 1, total: 1 })
    await Promise.resolve()
    expect(
      sendMessage.mock.calls.filter(
        ([message]) =>
          (message as { type?: string; pageSessionId?: string }).type === 'jobs:cancel-page',
      ),
    ).toHaveLength(0)

    run.destroy()
    await vi.waitFor(() =>
      expect(
        sendMessage.mock.calls.filter(([message]) => {
          const candidate = message as { type?: string; pageSessionId?: string }
          return (
            candidate.type === 'jobs:cancel-page' &&
            candidate.pageSessionId === token.pageSessionId
          )
        }),
      ).toHaveLength(1),
    )
    run.destroy()
    expect(
      sendMessage.mock.calls.filter(
        ([message]) => (message as { type?: string }).type === 'chapter:finish',
      ),
    ).toHaveLength(1)
    expect(
      sendMessage.mock.calls.filter(
        ([message]) => (message as { type?: string }).type === 'chapter:cancel',
      ),
    ).toHaveLength(0)
    expect(
      sendMessage.mock.calls.filter(([message]) => {
        const candidate = message as { type?: string; pageSessionId?: string }
        return (
          candidate.type === 'jobs:cancel-page' && candidate.pageSessionId === token.pageSessionId
        )
      }),
    ).toHaveLength(1)
  })

  it('keeps a failed image run open for its explicit retry and closes it once on success', async () => {
    const sendMessage = installBackend()
    const run = new ChapterRunController('image', vi.fn())
    const token = await run.start(1, 'Preparing image')

    expect(run.finish({ current: 0, total: 1 }, 'One image needs attention.', true)).toMatchObject({
      state: 'failed',
    })
    expect(() => run.assertCurrent(token)).not.toThrow()
    expect(
      sendMessage.mock.calls.some(
        ([message]) => (message as { type?: string }).type === 'chapter:finish',
      ),
    ).toBe(false)

    run.update({ current: 1, total: 1, message: 'Retry complete' })
    expect(run.finish({ current: 1, total: 1 })).toMatchObject({ state: 'complete' })
    run.finish({ current: 1, total: 1 })
    expect(
      sendMessage.mock.calls.filter(
        ([message]) => (message as { type?: string }).type === 'chapter:finish',
      ),
    ).toHaveLength(1)
    run.destroy()
  })

  it('closes a replaced active session with the page URL captured for that session', async () => {
    const sendMessage = installBackend()
    const run = new ChapterRunController('document', vi.fn())
    const originalUrl = location.href
    try {
      const first = await run.start(1, 'Preparing original')
      history.pushState({}, '', `${location.pathname}?chapter-run-replacement=1`)
      const second = await run.start(1, 'Preparing replacement')

      expect(second.pageUrl).not.toBe(first.pageUrl)
      expect(sendMessage).toHaveBeenCalledWith({
        type: 'chapter:cancel',
        pageSessionId: first.pageSessionId,
        pageUrl: first.pageUrl,
      })
      expect(sendMessage).not.toHaveBeenCalledWith({
        type: 'chapter:cancel',
        pageSessionId: first.pageSessionId,
        pageUrl: second.pageUrl,
      })
    } finally {
      run.destroy()
      history.replaceState({}, '', originalUrl)
    }
  })
})
