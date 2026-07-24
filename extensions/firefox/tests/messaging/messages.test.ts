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
    }
    expect(parseBackgroundRequest(valid)).toEqual(valid)
    expect(() =>
      parseBackgroundRequest({ ...valid, fixtureMode: true }),
    ).toThrow(/fixtureMode is not permitted/i)
  })

  it('requires source ownership on result, lookup, and font operations', () => {
    expect(() =>
      parseBackgroundRequest({ type: 'job:result', jobId: 'job' }),
    ).toThrow(/pageSessionId/i)
    expect(() =>
      parseBackgroundRequest({
        type: 'dictionary:lookup',
        request: { selectedText: '我' },
      }),
    ).toThrow(/identify the translated job and region/i)
    expect(() =>
      parseBackgroundRequest({ type: 'font:get', fontId: 'font' }),
    ).toThrow(/jobId/i)
  })

  it('rejects malformed content commands rather than trusting the message type', () => {
    expect(() =>
      parseContentRequest({
        type: 'content:start',
        scope: 'everything',
        hskLevel: 9,
      }),
    ).toThrow(/scope/i)
    expect(() =>
      parseContentRequest({
        type: 'content:cancel',
        pageControlled: true,
      }),
    ).toThrow(/pageControlled is not permitted/i)
  })
})
