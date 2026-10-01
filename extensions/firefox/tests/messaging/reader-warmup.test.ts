import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  detectDocumentChapter: vi.fn(async () => ({
    kind: 'not-document' as const,
    reason: 'insufficient-content',
  })),
  looksLikeSequentialArtReader: vi.fn(() => false),
  discoverPageSurfaces: vi.fn(() => ({ surfaces: [] })),
  sendBackgroundMessage: vi.fn(),
}))

vi.mock('../../src/document/extraction', () => ({
  detectDocumentChapter: mocks.detectDocumentChapter,
}))
vi.mock('../../src/discovery/images', () => ({
  looksLikeSequentialArtReader: mocks.looksLikeSequentialArtReader,
}))
vi.mock('../../src/discovery/surfaces', () => ({
  discoverPageSurfaces: mocks.discoverPageSurfaces,
}))
vi.mock('../../src/messaging/messages', () => ({
  sendBackgroundMessage: mocks.sendBackgroundMessage,
}))

describe('passive reader warmup', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('defineContentScript', (definition: unknown) => definition)
    mocks.detectDocumentChapter.mockClear()
    mocks.looksLikeSequentialArtReader.mockClear()
    mocks.discoverPageSurfaces.mockClear()
    mocks.sendBackgroundMessage.mockClear()
  })

  afterEach(async () => {
    await vi.advanceTimersByTimeAsync(21_000)
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.resetModules()
  })

  it('reclassifies an unsupported page so late prose can warm the language lane', async () => {
    const definition = (await import('../../entrypoints/reader-warmup.content')).default as {
      main(): void
    }
    definition.main()

    await vi.advanceTimersByTimeAsync(6_000)

    expect(mocks.detectDocumentChapter).toHaveBeenCalledTimes(1)
    document.body.append(document.createElement('article'))
    await vi.advanceTimersByTimeAsync(1_100)
    expect(mocks.detectDocumentChapter).toHaveBeenCalledTimes(2)
    expect(mocks.looksLikeSequentialArtReader.mock.calls.length).toBeGreaterThan(1)
    expect(mocks.discoverPageSurfaces.mock.calls.length).toBeGreaterThan(1)
    expect(mocks.sendBackgroundMessage).not.toHaveBeenCalled()
  })
})
