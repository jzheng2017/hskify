import { describe, expect, it } from 'vitest'

import type { TeachingTerm } from '../../src/contracts/browser'
import {
  buildTeachingText,
  buildTeachingTextRange,
  installTranslatedText,
} from '../../src/language/translated-text'

describe('shared translated text markup', () => {
  it('uses Unicode code-point offsets and preserves exactly selectable text', () => {
    const text = '\ud842\udfb7\u6211\u73b0\u5728\u8d70'
    const terms: TeachingTerm[] = [
      {
        text: '\u73b0\u5728',
        startChar: 2,
        endChar: 4,
        pinyin: 'xi\u00e0n z\u00e0i',
        definitions: ['now'],
        requiredLevel: 3,
        reason: 'above-level',
      },
    ]
    const host = document.createElement('span')
    host.append(buildTeachingText(document, text, terms))

    expect(host.textContent).toBe(text)
    expect(host.querySelector('.hskify-learning-term')?.textContent).toBe('\u73b0\u5728')
    expect(
      host.querySelector<HTMLElement>('.hskify-learning-term')?.dataset.hskifyRequiredLevel,
    ).toBe('3')
  })

  it('ignores invalid and overlapping term metadata without losing source characters', () => {
    const text = '\u6211\u4eec\u73b0\u5728\u8d70'
    const terms: TeachingTerm[] = [
      {
        text: '\u9519\u8bef',
        startChar: 0,
        endChar: 2,
        pinyin: 'cu\u00f2 w\u00f9',
        definitions: ['wrong'],
        reason: 'outside-list',
      },
      {
        text: '\u73b0\u5728',
        startChar: 2,
        endChar: 4,
        pinyin: 'xi\u00e0n z\u00e0i',
        definitions: ['now'],
        reason: 'above-level',
      },
      {
        text: '\u5728\u8d70',
        startChar: 3,
        endChar: 5,
        pinyin: 'z\u00e0i z\u01d2u',
        definitions: ['walking'],
        reason: 'above-level',
      },
    ]
    const host = document.createElement('span')
    host.append(buildTeachingText(document, text, terms))

    expect(host.textContent).toBe(text)
    expect(host.querySelectorAll('.hskify-learning-term')).toHaveLength(1)
    expect(host.querySelector('.hskify-learning-term')?.textContent).toBe('\u73b0\u5728')
  })

  it('renders a line range using offsets from the complete translated text', () => {
    const text = '\u6211\u4eec\u73b0\u5728\u8d70\n\u4f60\u4eec\u7b49'
    const terms: TeachingTerm[] = [
      {
        text: '\u73b0\u5728',
        startChar: 2,
        endChar: 4,
        pinyin: 'xi\u00e0n z\u00e0i',
        definitions: ['now'],
        reason: 'above-level',
      },
    ]
    const firstLine = document.createElement('span')
    firstLine.append(buildTeachingTextRange(document, text, terms, 0, 5))
    const secondLine = document.createElement('span')
    secondLine.append(buildTeachingTextRange(document, text, terms, 6, 9))

    expect(firstLine.textContent).toBe('\u6211\u4eec\u73b0\u5728\u8d70')
    expect(firstLine.querySelector('.hskify-learning-term')?.textContent).toBe('\u73b0\u5728')
    expect(secondLine.textContent).toBe('\u4f60\u4eec\u7b49')
    expect(secondLine.querySelector('.hskify-learning-term')).toBeNull()
  })

  it('installs the shared source and displayed identities for every renderer', () => {
    const host = document.createElement('span')
    installTranslatedText(host, {
      termination: 'stop',
      protectedNames: [],
      sourceText: 'The lantern was still burning.',
      baseChinese: '灯还亮着。',
      displayedChinese: '灯还亮着。',
      pinyin: 'dēng hái liàng zhe',
      hsk: {
        requestedLevel: 3,
        learningMode: 'natural',
        strictlyValid: true,
        levelCoverage: 1,
        aboveLevelTokens: [],
        teachingTerms: [],
        repairState: 'not-needed',
      },
    })

    expect(host.dataset.hskifySourceText).toBe('The lantern was still burning.')
    expect(host.dataset.hskifyDisplayedChinese).toBe('灯还亮着。')
  })
})
