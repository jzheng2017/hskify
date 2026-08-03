import { describe, expect, it } from 'vitest'

import { ChapterLifecycleStore } from '../../src/messaging/chapter-lifecycle'

describe('chapter lifecycle reducer', () => {
  it('keeps page registration and viewport progress monotonic despite job order', () => {
    const store = new ChapterLifecycleStore()
    store.start('chapter', 'https://reader.test/chapter')
    store.page('chapter', 'https://reader.test/chapter', 4)
    const state = store.page('chapter', 'https://reader.test/chapter', 1)
    expect(state.phase).toBe('active')
    expect(state.highestPageIndex).toBe(4)
    expect(state.submittedPages).toBe(2)
    expect(store.viewport('chapter', 'https://reader.test/chapter').viewportRevision).toBe(1)
    expect(store.seal('chapter', 'https://reader.test/chapter').phase).toBe('sealed')
    expect(store.page('chapter', 'https://reader.test/chapter', 5).phase).toBe('sealed')
  })

  it('does not permit a different document to append to a chapter', () => {
    const store = new ChapterLifecycleStore()
    store.start('chapter', 'https://reader.test/chapter')
    expect(() => store.page('chapter', 'https://reader.test/other', 0)).toThrow(/not active/i)
  })
})
