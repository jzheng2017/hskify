import { describe, expect, it } from 'vitest'

import { ChapterLifecycleStore } from '../../src/messaging/chapter-lifecycle'

describe('chapter lifecycle reducer', () => {
  it('keeps ordered source registration and focus progress monotonic', () => {
    const store = new ChapterLifecycleStore()
    store.start('chapter', 'https://reader.test/chapter', 'image')
    store.source('chapter', 'https://reader.test/chapter', 4)
    const state = store.source('chapter', 'https://reader.test/chapter', 1)
    expect(state.phase).toBe('active')
    expect(state.contentKind).toBe('image')
    expect(state.highestSourceIndex).toBe(4)
    expect(state.submittedSources).toBe(2)
    expect(store.focus('chapter', 'https://reader.test/chapter').focusRevision).toBe(1)
    expect(store.finish('chapter', 'https://reader.test/chapter').phase).toBe('finished')
    expect(store.source('chapter', 'https://reader.test/chapter', 5).phase).toBe('finished')
  })

  it('does not permit another document or modality to append to a chapter', () => {
    const store = new ChapterLifecycleStore()
    store.start('chapter', 'https://reader.test/chapter', 'document')
    expect(() => store.source('chapter', 'https://reader.test/other', 0)).toThrow(/not active/i)
    expect(store.start('chapter', 'https://reader.test/chapter', 'image').contentKind).toBe('image')
  })
})
