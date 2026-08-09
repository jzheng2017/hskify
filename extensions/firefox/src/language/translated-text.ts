import type { TeachingTerm, TranslatedText } from '../contracts/browser'

export type TeachingTextOptions = {
  termClassName?: string
}

function usableTerms(text: string, terms: readonly TeachingTerm[]): TeachingTerm[] {
  const characters = [...text]
  const sorted = [...terms].sort(
    (left, right) => left.startChar - right.startChar || left.endChar - right.endChar,
  )
  const result: TeachingTerm[] = []
  let cursor = 0
  for (const term of sorted) {
    if (
      term.startChar < cursor ||
      term.startChar < 0 ||
      term.endChar <= term.startChar ||
      term.endChar > characters.length ||
      characters.slice(term.startChar, term.endChar).join('') !== term.text
    ) {
      continue
    }
    result.push(term)
    cursor = term.endChar
  }
  return result
}

/**
 * Builds safe selectable text with teaching-term spans. Offsets are Unicode
 * code-point offsets, matching the native validator rather than UTF-16 DOM
 * offsets. Invalid or overlapping metadata is ignored without altering text.
 */
export function buildTeachingText(
  documentRef: Document,
  text: string,
  terms: readonly TeachingTerm[],
  options: TeachingTextOptions = {},
): DocumentFragment {
  return buildTeachingTextRange(documentRef, text, terms, 0, [...text].length, options)
}

/**
 * Builds one code-point range while interpreting teaching offsets against the
 * complete text. This lets image line wrapping and document prose share the
 * same validation and overlap policy without rebasing metadata.
 */
export function buildTeachingTextRange(
  documentRef: Document,
  fullText: string,
  terms: readonly TeachingTerm[],
  startChar: number,
  endChar: number,
  options: TeachingTextOptions = {},
): DocumentFragment {
  const fragment = documentRef.createDocumentFragment()
  const characters = [...fullText]
  if (
    !Number.isInteger(startChar) ||
    !Number.isInteger(endChar) ||
    startChar < 0 ||
    endChar < startChar ||
    endChar > characters.length
  ) {
    throw new RangeError('Teaching text range must be within the full Unicode text.')
  }
  let cursor = startChar
  for (const term of usableTerms(fullText, terms)) {
    const start = Math.max(startChar, term.startChar)
    const end = Math.min(endChar, term.endChar)
    if (start >= end || start < cursor) continue
    if (start > cursor) {
      fragment.append(documentRef.createTextNode(characters.slice(cursor, start).join('')))
    }
    const teaching = documentRef.createElement('span')
    teaching.className = options.termClassName ?? 'hskify-learning-term'
    teaching.dataset.hskifyLearningTerm = term.text
    teaching.dataset.hskifyLearningReason = term.reason
    teaching.dataset.hskifyPinyin = term.pinyin
    if (term.requiredLevel !== undefined) {
      teaching.dataset.hskifyRequiredLevel = String(term.requiredLevel)
    }
    teaching.textContent = characters.slice(start, end).join('')
    fragment.append(teaching)
    cursor = end
  }
  if (cursor < endChar) {
    fragment.append(documentRef.createTextNode(characters.slice(cursor, endChar).join('')))
  }
  return fragment
}

/** Applies the common language metadata without introducing renderer markup. */
export function installTranslatedText(
  element: HTMLElement,
  translated: TranslatedText,
  options: TeachingTextOptions = {},
): void {
  element.replaceChildren(
    buildTeachingText(
      element.ownerDocument,
      translated.displayedChinese,
      translated.hsk.teachingTerms,
      options,
    ),
  )
  installTranslatedTextMetadata(element, translated, options)
}

/** Applies the shared final language metadata without changing existing line markup. */
export function installTranslatedTextMetadata(
  element: HTMLElement,
  translated: TranslatedText,
  options: TeachingTextOptions = {},
): void {
  element.lang = 'zh-CN'
  element.dataset.hskifySourceText = translated.sourceText
  element.dataset.hskifyDisplayedChinese = translated.displayedChinese
  element.dataset.hskifyPinyin = translated.pinyin
  element.dataset.hskifyHskValid = String(translated.hsk.strictlyValid)
  element.dataset.hskifyHskRepairState = translated.hsk.repairState
  element.dataset.hskifyHskLearningMode = translated.hsk.learningMode
  element.dataset.hskifyHskLevelCoverage = String(translated.hsk.levelCoverage)
  element.dataset.hskifyHskTeachingTerms = String(
    element.querySelectorAll(`.${options.termClassName ?? 'hskify-learning-term'}`).length,
  )
  element.setAttribute(
    'aria-label',
    translated.pinyin
      ? `${translated.displayedChinese}; ${translated.pinyin}`
      : translated.displayedChinese,
  )
}
