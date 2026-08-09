import { Readability } from '@mozilla/readability'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DocumentJobRequest, JobUpdate, TranslatedText } from '../../src/contracts/browser'
import { looksLikeSequentialArtReader } from '../../src/discovery/images'
import { detectDocumentChapter } from '../../src/document'
import { ChapterController, DocumentChapterMode } from '../../src/page/controller'
import { loadedImage } from '../helpers/images'

const sentence =
  'The lantern light followed Mara through the old archive while she considered the promise that had brought her home. '

function appendNovel(id: string, suffix = ''): HTMLElement {
  const main = document.createElement('main')
  const article = document.createElement('article')
  article.id = id
  const heading = document.createElement('h1')
  heading.textContent = `Chapter ${id}`
  article.append(heading)
  for (let index = 0; index < 8; index += 1) {
    const paragraph = document.createElement('p')
    paragraph.textContent = `${index + 1}. ${sentence.repeat(2)}${suffix}`
    article.append(paragraph)
  }
  main.append(article)
  document.body.append(main)
  document.title = `Chapter ${id}`
  return article
}

function appendOversizeNovelWithMangaImage(): HTMLElement {
  const article = document.createElement('article')
  article.id = 'oversize-story'
  const heading = document.createElement('h1')
  heading.textContent = 'An oversized chapter'
  article.append(heading)
  const largeBlock = 'A complete English sentence with stable words and punctuation. '.repeat(250)
  for (let index = 0; index < 80; index += 1) {
    const paragraph = document.createElement('p')
    paragraph.textContent = `${index + 1}. ${largeBlock}`
    article.append(paragraph)
  }
  const main = document.createElement('main')
  main.append(article)
  document.body.append(main)
  document.body.append(loadedImage('https://reader.test/manga-page.png', 1200, 3600))
  document.title = 'An oversized chapter'
  return article
}

function translation(sourceText: string, request: DocumentJobRequest): TranslatedText {
  return {
    sourceText,
    baseChinese: '\u7ffb\u8bd1\u5b8c\u6210\u3002',
    displayedChinese: '\u7ffb\u8bd1\u5b8c\u6210\u3002',
    pinyin: 'f\u0101n y\u00ec w\u00e1n ch\u00e9ng',
    hsk: {
      requestedLevel: request.settings.hskLevel,
      learningMode: request.settings.learningMode,
      strictlyValid: true,
      levelCoverage: 1,
      aboveLevelTokens: [],
      teachingTerms: [],
      repairState: 'not-needed',
    },
  }
}

function installDocumentBackend(): ReturnType<typeof vi.fn> {
  let request: DocumentJobRequest | undefined
  let jobNumber = 0
  const sendMessage = vi.fn(async (raw: unknown) => {
    const message = raw as Record<string, unknown>
    switch (message.type) {
      case 'jobs:recover':
        return { ok: true, value: [] }
      case 'job:submit-document':
        request = message.request as DocumentJobRequest
        jobNumber += 1
        return {
          ok: true,
          value: {
            kind: 'document',
            jobId: `document-job-${jobNumber}`,
            sourceSha256: request.sourceSha256,
            acknowledgedSequence: 0,
          },
        }
      case 'job:updates': {
        if (!request) throw new Error('Document request was not submitted.')
        const updates: JobUpdate[] = request.blocks.map((block, index) => ({
          sequence: index + 1,
          type: 'documentBlockReady' as const,
          block: {
            itemId: block.itemId,
            sourceIndex: block.sourceIndex,
            itemOrder: block.itemOrder,
            kind: block.kind,
            text: translation(block.text, request!),
          },
        }))
        updates.push({
          sequence: updates.length + 1,
          type: 'complete',
          translatedCount: request.blocks.length,
          preservedCount: 0,
        })
        return {
          ok: true,
          value: {
            jobId: message.jobId,
            nextSequence: updates.length,
            updates,
          },
        }
      }
      default:
        return { ok: true, value: undefined }
    }
  })
  vi.stubGlobal('browser', { runtime: { sendMessage } })
  return sendMessage
}

function currentDocumentMode(controller: ChapterController): DocumentChapterMode | undefined {
  const mode = (controller as unknown as { mode?: unknown }).mode
  return mode instanceof DocumentChapterMode ? mode : undefined
}

describe('chapter controller document mode lifecycle', () => {
  beforeEach(() => {
    document.head.replaceChildren()
    document.body.replaceChildren()
    sessionStorage.clear()
    installDocumentBackend()
  })

  afterEach(() => {
    document.body.replaceChildren()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('reuses document mode on a second Translate action without parsing the hidden source', async () => {
    const source = appendNovel('one')
    const parse = vi.spyOn(Readability.prototype, 'parse')
    const controller = new ChapterController()

    await expect(controller.start('all', 3, 'natural', 'ltr')).resolves.toMatchObject({
      state: 'complete',
      contentKind: 'document',
    })
    const firstMode = currentDocumentMode(controller)
    const parsesAfterFirstRun = parse.mock.calls.length
    expect(firstMode).toBeDefined()
    expect(parsesAfterFirstRun).toBe(1)
    expect(source.hidden).toBe(true)

    await expect(controller.start('all', 3, 'natural', 'ltr')).resolves.toMatchObject({
      state: 'complete',
      contentKind: 'document',
    })
    expect(currentDocumentMode(controller)).toBe(firstMode)
    expect(parse).toHaveBeenCalledTimes(parsesAfterFirstRun)
    expect(source.hidden).toBe(true)
    controller.destroy()
  })

  it('replaces an invalidated descriptor after a live source mutation', async () => {
    const source = appendNovel('mutation')
    const controller = new ChapterController()
    await controller.start('all', 3, 'natural', 'ltr')
    const previous = currentDocumentMode(controller)
    if (!previous) throw new Error('Document mode was not detected.')
    const previousHash = previous.chapter.snapshot.sourceSha256

    source.querySelector('p')?.append(' The sealed letter now contained an additional warning.')
    await vi.waitFor(() => expect(previous.isInvalidated).toBe(true))

    await expect(controller.start('all', 3, 'natural', 'ltr')).resolves.toMatchObject({
      state: 'complete',
      contentKind: 'document',
    })
    const replacement = currentDocumentMode(controller)
    expect(replacement).toBeDefined()
    expect(replacement).not.toBe(previous)
    expect(replacement?.chapter.snapshot.sourceSha256).not.toBe(previousHash)
    controller.destroy()
  })

  it('can be temporarily unsupported after detach and detects a later replacement chapter', async () => {
    const source = appendNovel('detached')
    const controller = new ChapterController()
    await controller.start('all', 3, 'natural', 'ltr')
    const previous = currentDocumentMode(controller)
    if (!previous) throw new Error('Document mode was not detected.')

    source.remove()
    await vi.waitFor(() => expect(previous.isInvalidated).toBe(true))
    await expect(controller.start('all', 3, 'natural', 'ltr')).resolves.toMatchObject({
      state: 'failed',
      contentKind: 'unsupported',
    })

    const replacementSource = appendNovel('replacement', ' The replacement chapter is current.')
    await expect(controller.start('all', 3, 'natural', 'ltr')).resolves.toMatchObject({
      state: 'complete',
      contentKind: 'document',
    })
    expect(currentDocumentMode(controller)).toBeDefined()
    expect(replacementSource.hidden).toBe(true)
    controller.destroy()
  })

  it('does not fall through to image mode when an oversized document is rejected', async () => {
    const source = appendOversizeNovelWithMangaImage()
    expect(looksLikeSequentialArtReader()).toBe(true)
    await expect(detectDocumentChapter(document)).resolves.toEqual({
      kind: 'rejected',
      reason: 'input-too-large',
    })
    const controller = new ChapterController()

    await expect(controller.start('all', 3, 'natural', 'ltr')).resolves.toMatchObject({
      state: 'failed',
      contentKind: 'unsupported',
    })
    expect((controller as unknown as { mode?: unknown }).mode).toBeUndefined()
    expect(source.hidden).toBe(false)
    expect(document.querySelector('[data-hskify-document-reader]')).toBeNull()
    const sendMessage = browser.runtime.sendMessage as ReturnType<typeof vi.fn>
    expect(
      sendMessage.mock.calls.some(
        ([message]) => (message as { type?: string }).type === 'job:submit-image',
      ),
    ).toBe(false)
    controller.destroy()
  })

  it('releases a completed document once when SPA navigation replaces the page', async () => {
    const source = appendNovel('spa-completed')
    const controller = new ChapterController()
    const sendMessage = browser.runtime.sendMessage as ReturnType<typeof vi.fn>
    const originalUrl = location.href

    try {
      await expect(controller.start('all', 3, 'natural', 'ltr')).resolves.toMatchObject({
        state: 'complete',
        contentKind: 'document',
      })
      const chapterStart = sendMessage.mock.calls.find(
        ([message]) => (message as { type?: string }).type === 'chapter:start',
      )?.[0] as { pageSessionId?: string } | undefined
      if (!chapterStart?.pageSessionId) throw new Error('Chapter session was not started.')
      expect(source.hidden).toBe(true)
      expect(
        sendMessage.mock.calls.filter(
          ([message]) => (message as { type?: string }).type === 'jobs:cancel-page',
        ),
      ).toHaveLength(0)

      history.pushState({}, '', `${location.pathname}?completed-spa-replacement=1`)
      ;(controller as unknown as { checkNavigation(): void }).checkNavigation()

      await vi.waitFor(() =>
        expect(
          sendMessage.mock.calls.filter(([message]) => {
            const candidate = message as { type?: string; pageSessionId?: string }
            return (
              candidate.type === 'jobs:cancel-page' &&
              candidate.pageSessionId === chapterStart.pageSessionId
            )
          }),
        ).toHaveLength(1),
      )
      expect(source.hidden).toBe(false)
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

      ;(controller as unknown as { checkNavigation(): void }).checkNavigation()
      controller.destroy()
      expect(
        sendMessage.mock.calls.filter(([message]) => {
          const candidate = message as { type?: string; pageSessionId?: string }
          return (
            candidate.type === 'jobs:cancel-page' &&
            candidate.pageSessionId === chapterStart.pageSessionId
          )
        }),
      ).toHaveLength(1)
    } finally {
      controller.destroy()
      history.replaceState({}, '', originalUrl)
    }
  })
})
