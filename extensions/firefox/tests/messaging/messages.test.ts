import { describe, expect, it } from 'vitest'

import {
  parseBackgroundRequest,
  parseContentRequest,
} from '../../src/messaging/messages'

describe('strict extension runtime messages', () => {
  it('parses every submitted-image field and rejects page-controlled fixture switches', () => {
    const valid = {
      type: 'job:submit',
      pageSessionId: 'page-session',
      pageIndex: 3,
      imageUrl: 'https://cdn.test/chapter.webp?page=3',
      pageUrl: 'https://reader.test/chapter/1',
      naturalWidth: 900,
      naturalHeight: 16_000,
      sourceMimeType: 'image/webp',
      sourceBytes: Uint8Array.of(1, 2, 3).buffer,
      hskLevel: 4,
      nameTranslation: 'keep-original',
      visibleRects: [{ x: 0, y: 0.25, width: 1, height: 0.5 }],
      properNameGlossary: [{ sourceEnglish: 'Cheon Yeo Woon', chinese: '天汝云' }],
    }
    expect(parseBackgroundRequest(valid)).toEqual(valid)
    expect(() =>
      parseBackgroundRequest({ ...valid, fixtureMode: true }),
    ).toThrow(/fixtureMode is not permitted/i)
  })

  it('requires job ownership on updates, patches, lookup, and font operations', () => {
    expect(() =>
      parseBackgroundRequest({ type: 'job:updates', after: 0 }),
    ).toThrow(/jobId/i)
    expect(() =>
      parseBackgroundRequest({ type: 'job:patch', jobId: 'job', patchId: 'patch' }),
    ).toThrow(/mimeType/i)
    expect(() =>
      parseBackgroundRequest({
        type: 'dictionary:lookup',
        request: {
          interaction: 'selection',
          selectedText: '我',
        },
      }),
    ).toThrow(/identify the translated job and region/i)
    expect(() =>
      parseBackgroundRequest({ type: 'font:get', fontId: 'font' }),
    ).toThrow(/jobId/i)
    expect(() =>
      parseBackgroundRequest({ type: 'job:result', jobId: 'job' }),
    ).toThrow(/not supported/i)
  })

  it('validates progressive cursors and normalized viewport updates', () => {
    expect(
      parseBackgroundRequest({
        type: 'job:updates',
        jobId: 'job',
        after: 17,
      }),
    ).toEqual({ type: 'job:updates', jobId: 'job', after: 17 })
    expect(
      parseBackgroundRequest({
        type: 'job:viewport',
        jobId: 'job',
        visibleRects: [{ x: 0.1, y: 0.2, width: 0.4, height: 0.5 }],
        active: true,
      }),
    ).toMatchObject({ type: 'job:viewport', active: true })
    expect(() =>
      parseBackgroundRequest({
        type: 'job:viewport',
        jobId: 'job',
        visibleRects: [{ x: 0.9, y: 0, width: 0.2, height: 1 }],
        active: true,
      }),
    ).toThrow(/invalid/i)
  })

  it('accepts only bounded source identity for acquisition prefetch lifecycle', () => {
    const source = {
      pageSessionId: 'page-session',
      pageIndex: 4,
      imageUrl: 'https://cdn.test/4.webp',
      pageUrl: 'https://reader.test/chapter',
      naturalWidth: 900,
      naturalHeight: 16_000,
    }
    expect(
      parseBackgroundRequest({ type: 'image:prefetch', ...source }),
    ).toEqual({ type: 'image:prefetch', ...source })
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
      parseBackgroundRequest({
        type: 'image:prefetch',
        ...source,
        daemonJob: true,
      }),
    ).toThrow(/daemonJob is not permitted/i)
  })

  it('rejects malformed content commands rather than trusting the message type', () => {
    expect(
      parseContentRequest({
        type: 'content:start',
        scope: 'all',
        hskLevel: 5,
        nameTranslation: 'keep-original',
        properNameGlossary: [
          { sourceEnglish: 'Cheon Yeo Woon', chinese: '天汝云' },
        ],
      }),
    ).toMatchObject({
      properNameGlossary: [{ sourceEnglish: 'Cheon Yeo Woon', chinese: '天汝云' }],
    })
    expect(() =>
      parseContentRequest({
        type: 'content:start',
        scope: 'everything',
        hskLevel: 9,
        nameTranslation: 'literal',
      }),
    ).toThrow(/scope/i)
    expect(() =>
      parseContentRequest({
        type: 'content:start',
        scope: 'all',
        hskLevel: 3,
        nameTranslation: 'literal',
      }),
    ).toThrow(/nameTranslation/i)
    expect(() =>
      parseContentRequest({
        type: 'content:cancel',
        pageControlled: true,
      }),
    ).toThrow(/pageControlled is not permitted/i)
  })

  it('accepts only exact setup control messages', () => {
    expect(parseBackgroundRequest({ type: 'setup:status' })).toEqual({
      type: 'setup:status',
    })
    expect(parseBackgroundRequest({ type: 'setup:start' })).toEqual({
      type: 'setup:start',
    })
    expect(() =>
      parseBackgroundRequest({ type: 'setup:start', model: 'untrusted-model' }),
    ).toThrow(/not permitted/i)
  })
})
