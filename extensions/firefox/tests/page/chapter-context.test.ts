import { describe, expect, it } from 'vitest'

import { ChapterContextLedger } from '../../src/page/chapter-context'

function region(id: string, readingOrder: number, sourceEnglish: string, displayedChinese: string) {
  return { id, readingOrder, sourceEnglish, displayedChinese }
}

describe('chapter context ledger', () => {
  it('derives context from document and region order, not completion order', () => {
    const ledger = new ChapterContextLedger(6)
    ledger.commitPage(5, [region('later', 0, 'Later page', '后一页')])
    ledger.commitPage(2, [
      region('second', 20, 'Second bubble', '第二句'),
      region('first', 10, 'First bubble', '第一句'),
    ])

    expect(ledger.before(5)).toEqual([
      { sourceEnglish: 'First bubble', chinese: '第一句' },
      { sourceEnglish: 'Second bubble', chinese: '第二句' },
    ])
    expect(ledger.before(6)).toEqual([
      { sourceEnglish: 'First bubble', chinese: '第一句' },
      { sourceEnglish: 'Second bubble', chinese: '第二句' },
      { sourceEnglish: 'Later page', chinese: '后一页' },
    ])
  })

  it('replaces retried page context and excludes the current page', () => {
    const ledger = new ChapterContextLedger(2)
    ledger.commitPage(1, [region('old', 0, 'Old', '旧')])
    ledger.commitPage(1, [
      region('new-a', 0, 'New A', '新甲'),
      region('new-b', 1, 'New B', '新乙'),
    ])
    ledger.commitPage(2, [region('current', 0, 'Current', '当前')])

    expect(ledger.before(2)).toEqual([
      { sourceEnglish: 'New A', chinese: '新甲' },
      { sourceEnglish: 'New B', chinese: '新乙' },
    ])
    ledger.removePage(1)
    expect(ledger.before(3)).toEqual([{ sourceEnglish: 'Current', chinese: '当前' }])
  })
})
