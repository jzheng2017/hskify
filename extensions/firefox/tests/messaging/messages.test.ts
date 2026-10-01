import { describe, expect, it } from 'vitest'

import { BUILD_FINGERPRINT } from '../../src/contracts/browser'
import { parseBackgroundRequest, parseContentRequest } from '../../src/messaging/messages'

describe('strict extension runtime messages', () => {
  it('parses every submitted-image field and rejects page-controlled fixture switches', () => {
    const valid = {
      type: 'job:submit-image',
      clientRequestId: 'test-request',
      retryItemIds: [],
      pageSessionId: 'page-session',
      sourceIndex: 3,
      chapterSourceOrder: [3],
      surfaceKind: 'image',
      imageUrl: 'https://cdn.test/chapter.webp?page=3',
      pageUrl: 'https://reader.test/chapter/1',
      naturalWidth: 900,
      naturalHeight: 16_000,
      sourceMimeType: 'image/webp',
      sourceBytes: Uint8Array.of(1, 2, 3).buffer,
      hskLevel: 4,
      learningMode: 'natural',
      readingDirection: 'ltr',
      visibleRects: [{ x: 0, y: 0.25, width: 1, height: 0.5 }],
    }
    expect(parseBackgroundRequest(valid)).toEqual(valid)
    expect(() => parseBackgroundRequest({ ...valid, fixtureMode: true })).toThrow(
      /fixtureMode is not permitted/i,
    )
  })

  it('parses a complete authoritative DOM document request', () => {
    const request = {
      buildFingerprint: BUILD_FINGERPRINT,
      clientRequestId: 'test-request',
      retryItemIds: [],
      focus: { kind: 'document', active: true, visibleBlockIds: [] },
      pageSessionId: 'document-page',
      sourceSha256: 'a'.repeat(64),
      settings: {
        sourceLanguage: 'en',
        targetLanguage: 'zh-CN',
        hskStandard: '2.0',
        hskLevel: 3,
        learningMode: 'strict',
      },
      blocks: [
        {
          parentBlockId: 'block-0',
          subItemOrder: 0,
          itemId: 'block-0',
          sourceIndex: 0,
          itemOrder: 0,
          kind: 'prose',
          provenance: 'dom',
          text: 'The chapter begins here.',
        },
      ],
    }
    expect(
      parseBackgroundRequest({
        type: 'job:submit-document',
        pageUrl: 'https://reader.test/chapter/1',
        request,
      }),
    ).toMatchObject({ type: 'job:submit-document', request: { blocks: request.blocks } })
    expect(() =>
      parseBackgroundRequest({
        type: 'job:submit-document',
        pageUrl: 'https://reader.test/chapter/1',
        request: { ...request, readingDirection: 'ltr' },
      }),
    ).toThrow(/readingDirection/i)
  })

  it('requires job ownership on updates, patches, hover lookup, and font operations', () => {
    expect(() => parseBackgroundRequest({ type: 'job:updates', after: 0 })).toThrow(/jobId/i)
    expect(() =>
      parseBackgroundRequest({ type: 'job:patch', jobId: 'job', patchId: 'patch' }),
    ).toThrow(/mimeType/i)
    expect(() =>
      parseBackgroundRequest({
        type: 'dictionary:lookup',
        request: { interaction: 'hover', characterOffset: 0 },
      }),
    ).toThrow(/context/i)
    expect(() => parseBackgroundRequest({ type: 'font:get', fontId: 'font' })).toThrow(/jobId/i)
    expect(() => parseBackgroundRequest({ type: 'job:result', jobId: 'job' })).toThrow(
      /not supported/i,
    )
  })

  it('requires exact settings and image direction in recovery identities', () => {
    const settings = {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel: 4,
      learningMode: 'strict',
    }
    const valid = {
      type: 'jobs:recover',
      pageSessionId: 'page-session',
      pageUrl: 'https://reader.test/chapter',
      candidates: [
        { kind: 'document', sourceSha256: 'a'.repeat(64), settings },
        {
          kind: 'image',
          sourceUrl: 'https://reader.test/page.webp',
          naturalWidth: 900,
          naturalHeight: 1600,
          sourceIndex: 0,
          settings,
          readingDirection: 'rtl',
        },
      ],
    }
    expect(parseBackgroundRequest(valid)).toEqual(valid)
    expect(() =>
      parseBackgroundRequest({
        ...valid,
        candidates: [{ kind: 'document', sourceSha256: 'a'.repeat(64) }],
      }),
    ).toThrow(/settings/i)
    expect(() =>
      parseBackgroundRequest({
        ...valid,
        candidates: [{ ...valid.candidates[1], readingDirection: undefined }],
      }),
    ).toThrow(/readingDirection/i)
  })

  it('validates progressive cursors and tagged focus updates', () => {
    expect(parseBackgroundRequest({ type: 'job:updates', jobId: 'job', after: 17 })).toEqual({
      type: 'job:updates',
      jobId: 'job',
      after: 17,
    })
    expect(
      parseBackgroundRequest({
        type: 'job:focus',
        jobId: 'job',
        focus: {
          kind: 'image',
          visibleRects: [{ x: 0.1, y: 0.2, width: 0.4, height: 0.5 }],
          active: true,
        },
      }),
    ).toMatchObject({ type: 'job:focus', focus: { kind: 'image', active: true } })
    expect(() =>
      parseBackgroundRequest({
        type: 'job:focus',
        jobId: 'job',
        focus: {
          kind: 'image',
          visibleRects: [{ x: 0.9, y: 0, width: 0.2, height: 1 }],
          active: true,
        },
      }),
    ).toThrow(/source/i)
    expect(
      parseBackgroundRequest({
        type: 'job:focus',
        jobId: 'document-job',
        focus: { kind: 'document', visibleBlockIds: ['block-1'], active: true },
      }),
    ).toMatchObject({ focus: { kind: 'document', visibleBlockIds: ['block-1'] } })
  })

  it('accepts only bounded source identity for acquisition prefetch lifecycle', () => {
    const source = {
      pageSessionId: 'page-session',
      sourceIndex: 4,
      imageUrl: 'https://cdn.test/4.webp',
      pageUrl: 'https://reader.test/chapter',
      naturalWidth: 900,
      naturalHeight: 16_000,
    }
    expect(parseBackgroundRequest({ type: 'image:prefetch', ...source })).toEqual({
      type: 'image:prefetch',
      ...source,
    })
    expect(
      parseBackgroundRequest({
        type: 'image:prefetch-cancel',
        pageSessionId: source.pageSessionId,
        pageUrl: source.pageUrl,
      }),
    ).toEqual({
      type: 'image:prefetch-cancel',
      pageSessionId: source.pageSessionId,
      pageUrl: source.pageUrl,
    })
    expect(() =>
      parseBackgroundRequest({ type: 'image:prefetch', ...source, daemonJob: true }),
    ).toThrow(/daemonJob is not permitted/i)
  })

  it('rejects malformed content commands rather than trusting the message type', () => {
    expect(
      parseContentRequest({
        type: 'content:start',
        scope: 'all',
        hskLevel: 5,
        learningMode: 'natural',
        readingDirection: 'ltr',
      }),
    ).toEqual({
      type: 'content:start',
      scope: 'all',
      hskLevel: 5,
      learningMode: 'natural',
      readingDirection: 'ltr',
    })
    expect(() =>
      parseContentRequest({
        type: 'content:start',
        scope: 'everything',
        hskLevel: 9,
        learningMode: 'natural',
        readingDirection: 'ltr',
      }),
    ).toThrow(/scope/i)
    expect(() =>
      parseContentRequest({
        type: 'content:start',
        scope: 'all',
        hskLevel: 3,
        learningMode: 'natural',
        nameTranslation: 'keep-original',
        readingDirection: 'ltr',
      }),
    ).toThrow(/nameTranslation/i)
    expect(() => parseContentRequest({ type: 'content:cancel', pageControlled: true })).toThrow(
      /pageControlled is not permitted/i,
    )
  })

  it('accepts only exact setup and modality-specific warm-up messages', () => {
    expect(parseBackgroundRequest({ type: 'setup:status' })).toEqual({ type: 'setup:status' })
    expect(parseBackgroundRequest({ type: 'setup:start' })).toEqual({ type: 'setup:start' })
    expect(parseBackgroundRequest({ type: 'engine:warmup', contentKind: 'document' })).toEqual({
      type: 'engine:warmup',
      contentKind: 'document',
    })
    expect(() => parseBackgroundRequest({ type: 'engine:warmup' })).toThrow(/contentKind/i)
    expect(() => parseBackgroundRequest({ type: 'setup:start', model: 'untrusted-model' })).toThrow(
      /not permitted/i,
    )
  })
})
