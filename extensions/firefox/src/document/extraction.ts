import { Readability } from '@mozilla/readability'

import { sha256Hex } from '../acquisition/hash'
import {
  BUILD_FINGERPRINT,
  MAX_DOCUMENT_BLOCKS,
  MAX_DOCUMENT_BLOCK_UTF8_BYTES,
  MAX_DOCUMENT_UTF8_BYTES,
  type DocumentJobRequest,
} from '../contracts/browser'
import {
  DOCUMENT_EXTRACTION_END_MARK,
  DOCUMENT_EXTRACTION_MEASURE,
  DOCUMENT_EXTRACTION_START_MARK,
  markDocumentPerformance,
  measureDocumentPerformance,
} from './metrics'
import {
  canonicalDocumentText,
  toNativeDocumentBlocks,
  type DocumentBlockKind,
  type DocumentChapter,
  type DocumentDetection,
  type DocumentImageItem,
  type DocumentSeparatorItem,
  type DocumentStructureItem,
  type DocumentTextBlock,
} from './types'

const SOURCE_MARKER = 'data-hskify-source-marker'
const MIN_BLOCKS = 5
const MIN_CHARACTERS = 1_000
const MIN_LATIN_LETTER_RATIO = 0.75
const MIN_LATIN_LETTERS = 500
const MAX_PAGE_SESSION_ID_CHARACTERS = 256

const EXCLUDED_SELECTOR = [
  'nav',
  'aside',
  'form',
  'button',
  'input',
  'select',
  'textarea',
  'script',
  'style',
  'template',
  '[role="navigation"]',
  '[role="banner"]',
  '[role="complementary"]',
  '[role="contentinfo"]',
  '[role="form"]',
  '[aria-hidden="true"]',
  '[data-hskify-document-reader]',
].join(',')

const UI_NAME =
  /(?:^|[-_\s])(?:ad|ads|advert|advertisement|author|byline|breadcrumb|comment|comments|cookie|footer|header|login|menu|metadata|modal|nav|newsletter|pagination|promo|recommend|related|share|sidebar|social|toolbar|widget)(?:$|[-_\s])/iu
const ANCESTOR_UI_NAME =
  /(?:^|[-_\s])(?:ad|ads|advert|advertisement|breadcrumb|comment|comments|cookie|login|menu|modal|nav|newsletter|pagination|promo|recommend|related|share|sidebar|social|toolbar|widget)(?:$|[-_\s])/iu

type AnnotatedClone = {
  document: Document
  liveByMarker: Map<string, HTMLElement>
  markerByLive: Map<HTMLElement, string>
}

type LiveTextCandidate = {
  marker: string
  element: HTMLElement
  text: string
}

type PendingText = {
  type: 'text'
  kind: DocumentBlockKind
  text: string
  headingLevel?: 1 | 2 | 3 | 4 | 5 | 6
  liveElement: HTMLElement
  marker: string
}

type PendingImage = Omit<DocumentImageItem, 'itemId' | 'order'>
type PendingSeparator = Omit<DocumentSeparatorItem, 'itemId' | 'order'>
type PendingItem = PendingText | PendingImage | PendingSeparator

export type DocumentExtractionOptions = {
  crypto?: Crypto
}

export function normalizeDocumentText(value: string): string {
  return value
    .replaceAll('\u00a0', ' ')
    .replace(/\r\n?/gu, '\n')
    .split('\n')
    .map((line) => line.replace(/[\t\f\v ]+/gu, ' ').trim())
    .join('\n')
    .replace(/\n{3,}/gu, '\n\n')
    .trim()
}

function utf8Length(value: string): number {
  return new TextEncoder().encode(value).byteLength
}

/** Largest exact `/jobs/document` JSON envelope these extracted blocks can enter. */
export function documentWireRequestUtf8Bytes(blocks: DocumentJobRequest['blocks']): number {
  const request: DocumentJobRequest = {
    buildFingerprint: BUILD_FINGERPRINT,
    pageSessionId: 'x'.repeat(MAX_PAGE_SESSION_ID_CHARACTERS),
    sourceSha256: '0'.repeat(64),
    settings: {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel: 6,
      learningMode: 'natural',
    },
    blocks,
  }
  return utf8Length(JSON.stringify(request))
}

function stableTextId(order: number, kind: DocumentBlockKind, text: string): string {
  // FNV-1a is used only as a compact stable label. The complete source is
  // independently protected by SHA-256 before it crosses the native boundary.
  let hash = 0x811c9dc5
  const bytes = new TextEncoder().encode(`${order}\u001f${kind}\u001f${text}`)
  for (const byte of bytes) {
    hash ^= byte
    hash = Math.imul(hash, 0x01000193)
  }
  return `document-${order}-${kind}-${(hash >>> 0).toString(16).padStart(8, '0')}`
}

function annotateClone(live: Document): AnnotatedClone {
  const cloned = live.cloneNode(true) as Document
  const liveElements: HTMLElement[] = [
    live.documentElement as HTMLElement,
    ...live.documentElement.querySelectorAll<HTMLElement>('*'),
  ]
  const clonedElements: HTMLElement[] = [
    cloned.documentElement as HTMLElement,
    ...cloned.documentElement.querySelectorAll<HTMLElement>('*'),
  ]
  if (liveElements.length !== clonedElements.length) {
    throw new Error('The cloned document does not match the live source tree.')
  }

  const liveByMarker = new Map<string, HTMLElement>()
  const markerByLive = new Map<HTMLElement, string>()
  for (let index = 0; index < clonedElements.length; index += 1) {
    const liveElement = liveElements[index]
    const clonedElement = clonedElements[index]
    if (!liveElement || !clonedElement) continue
    const marker = index.toString(36)
    clonedElement.setAttribute(SOURCE_MARKER, marker)
    liveByMarker.set(marker, liveElement)
    markerByLive.set(liveElement, marker)
  }
  return { document: cloned, liveByMarker, markerByLive }
}

function nearestSourceMarker(element: Element): string | undefined {
  const marked = element.closest(`[${SOURCE_MARKER}]`)
  if (marked) return marked.getAttribute(SOURCE_MARKER) ?? undefined
  return element.querySelector(`[${SOURCE_MARKER}]`)?.getAttribute(SOURCE_MARKER) ?? undefined
}

function excluded(element: Element): boolean {
  if (element.matches(EXCLUDED_SELECTOR) || element.closest(EXCLUDED_SELECTOR)) return true
  for (const candidate of [element.id, element.className]) {
    if (typeof candidate === 'string' && UI_NAME.test(candidate)) return true
  }
  for (let current = element.parentElement; current; current = current.parentElement) {
    for (const candidate of [current.id, current.className]) {
      if (typeof candidate === 'string' && ANCESTOR_UI_NAME.test(candidate)) return true
    }
  }
  return false
}

function blockKind(element: Element): {
  kind: DocumentBlockKind
  headingLevel?: 1 | 2 | 3 | 4 | 5 | 6
} | null {
  const tag = element.tagName.toLowerCase()
  if (/^h[1-6]$/u.test(tag)) {
    return {
      kind: 'heading',
      headingLevel: Number(tag.slice(1)) as 1 | 2 | 3 | 4 | 5 | 6,
    }
  }
  if (tag === 'p') return { kind: 'paragraph' }
  if (tag === 'blockquote') return { kind: 'blockquote' }
  if (tag === 'figcaption') return { kind: 'caption' }
  if (tag === 'li') {
    return {
      kind: element.closest('ol') ? 'ordered-list-item' : 'unordered-list-item',
    }
  }
  if (
    tag === 'div' &&
    !element.querySelector(
      'article, section, div, p, h1, h2, h3, h4, h5, h6, blockquote, ol, ul, figure, figcaption',
    )
  ) {
    return { kind: 'paragraph' }
  }
  return null
}

function normalizedOwnBlockText(element: Element, kind: DocumentBlockKind): string {
  const clone = element.cloneNode(true) as Element
  if (kind.endsWith('list-item')) {
    for (const nested of clone.querySelectorAll('ol, ul')) nested.remove()
  }
  const separator = '\u0000'
  for (const lineBreak of clone.querySelectorAll('br')) {
    lineBreak.replaceWith(clone.ownerDocument.createTextNode(separator))
  }
  if (kind === 'blockquote') {
    for (const block of clone.querySelectorAll('p, div')) {
      block.append(clone.ownerDocument.createTextNode(separator))
    }
  }
  return (clone.textContent ?? '')
    .split(separator)
    .map((part) => part.replace(/[\s\u00a0]+/gu, ' ').trim())
    .join('\n')
    .replace(/\n{3,}/gu, '\n\n')
    .trim()
}

function liveTextCandidates(
  live: Document,
  markerByLive: ReadonlyMap<HTMLElement, string>,
): LiveTextCandidate[] {
  const result: LiveTextCandidate[] = []
  for (const element of live.body.querySelectorAll<HTMLElement>('*')) {
    const semantic = blockKind(element)
    const marker = markerByLive.get(element)
    if (!semantic || !marker || excluded(element)) continue
    const text = normalizedOwnBlockText(element, semantic.kind)
    if (text) result.push({ marker, element, text })
  }
  return result
}

function mappedTextSource(
  parsed: Element,
  text: string,
  liveByMarker: ReadonlyMap<string, HTMLElement>,
  candidatesByText: ReadonlyMap<string, readonly LiveTextCandidate[]>,
  usedMarkers: ReadonlySet<string>,
): { marker: string; element: HTMLElement } | undefined {
  const nearestMarker = nearestSourceMarker(parsed)
  const nearestLive = nearestMarker ? liveByMarker.get(nearestMarker) : undefined
  if (nearestMarker && nearestLive && !usedMarkers.has(nearestMarker) && !excluded(nearestLive)) {
    const semantic = blockKind(nearestLive)
    if (semantic && normalizedOwnBlockText(nearestLive, semantic.kind) === text) {
      return { marker: nearestMarker, element: nearestLive }
    }
  }
  const candidate = candidatesByText
    .get(text)
    ?.find(
      (item) =>
        !usedMarkers.has(item.marker) &&
        item.text === text &&
        (!nearestLive || nearestLive.contains(item.element)),
    )
  return candidate ? { marker: candidate.marker, element: candidate.element } : undefined
}

function safeImageSource(
  liveImage: HTMLImageElement,
  parsedImage: Element,
  baseUrl: string,
): string | undefined {
  const candidates = [
    parsedImage.getAttribute('src'),
    parsedImage.getAttribute('data-src'),
    liveImage.currentSrc,
    liveImage.getAttribute('src'),
    liveImage.getAttribute('data-src'),
  ]
  for (const candidate of candidates) {
    if (!candidate) continue
    let url: URL
    try {
      url = new URL(candidate, baseUrl)
    } catch {
      continue
    }
    if (url.protocol === 'http:' || url.protocol === 'https:' || url.protocol === 'blob:') {
      return url.href
    }
    if (url.protocol === 'data:' && /^data:image\//iu.test(url.href)) return url.href
  }
  return undefined
}

function collectPendingItems(
  content: HTMLElement,
  liveByMarker: ReadonlyMap<string, HTMLElement>,
  candidates: readonly LiveTextCandidate[],
  sourceUrl: string,
): PendingItem[] {
  const items: PendingItem[] = []
  const usedTextMarkers = new Set<string>()
  const candidatesByText = new Map<string, LiveTextCandidate[]>()
  for (const candidate of candidates) {
    const matches = candidatesByText.get(candidate.text)
    if (matches) matches.push(candidate)
    else candidatesByText.set(candidate.text, [candidate])
  }

  const visit = (element: Element): void => {
    if (excluded(element)) return
    const tag = element.tagName.toLowerCase()
    const marker = nearestSourceMarker(element)
    const liveElement = marker ? liveByMarker.get(marker) : undefined

    if (tag === 'img') {
      if (liveElement?.tagName.toLowerCase() === 'img' && !excluded(liveElement)) {
        const liveImage = liveElement as HTMLImageElement
        const source = safeImageSource(liveImage, element, sourceUrl)
        if (source) {
          const image: PendingImage = {
            type: 'image',
            sourceUrl: source,
            alt: normalizeDocumentText(liveImage.alt),
          }
          if (liveImage.naturalWidth > 0) image.width = liveImage.naturalWidth
          if (liveImage.naturalHeight > 0) image.height = liveImage.naturalHeight
          items.push(image)
        }
      }
      return
    }
    if (tag === 'hr') {
      items.push({ type: 'separator' })
      return
    }

    const semantic = blockKind(element)
    if (semantic) {
      const text = normalizedOwnBlockText(element, semantic.kind)
      const source = text
        ? mappedTextSource(element, text, liveByMarker, candidatesByText, usedTextMarkers)
        : undefined
      if (text && source) {
        usedTextMarkers.add(source.marker)
        items.push({
          type: 'text',
          kind: semantic.kind,
          text,
          liveElement: source.element,
          marker: source.marker,
          ...(semantic.headingLevel === undefined ? {} : { headingLevel: semantic.headingLevel }),
        })
      }
      // A selected block owns its inline descendants, but media and nested
      // lists remain distinct structural items.
      for (const child of element.children) {
        const childTag = child.tagName.toLowerCase()
        if (childTag === 'img' || childTag === 'ol' || childTag === 'ul') visit(child)
        else for (const image of child.querySelectorAll(':scope img')) visit(image)
      }
      return
    }

    for (const child of element.children) visit(child)
  }

  visit(content)
  return items
}

function commonSourceRoot(elements: readonly HTMLElement[]): HTMLElement | null {
  let root: HTMLElement | null = elements[0] ?? null
  while (root && !elements.every((element) => root!.contains(element))) {
    root = root.parentElement
  }
  return root
}

function predominantlyEnglish(text: string): boolean {
  let letters = 0
  let latin = 0
  for (const character of text) {
    if (!/\p{Letter}/u.test(character)) continue
    letters += 1
    if (/[A-Za-z]/u.test(character)) latin += 1
  }
  return latin >= MIN_LATIN_LETTERS && letters > 0 && latin / letters >= MIN_LATIN_LETTER_RATIO
}

function titleBlock(
  items: PendingItem[],
  readabilityTitle: string | null | undefined,
  candidates: readonly LiveTextCandidate[],
): void {
  const normalizedTitle = normalizeDocumentText(readabilityTitle ?? '')
  if (!normalizedTitle) return
  const heading = items.find(
    (item): item is PendingText =>
      item.type === 'text' && item.kind === 'heading' && item.text === normalizedTitle,
  )
  if (heading) {
    heading.kind = 'title'
    return
  }
  const firstLevelOne = items.find(
    (item): item is PendingText =>
      item.type === 'text' && item.kind === 'heading' && item.headingLevel === 1,
  )
  if (firstLevelOne) {
    firstLevelOne.kind = 'title'
    return
  }
  const used = new Set(
    items.filter((item): item is PendingText => item.type === 'text').map((item) => item.marker),
  )
  const mappedRoot = commonSourceRoot(
    items
      .filter((item): item is PendingText => item.type === 'text')
      .map((item) => item.liveElement),
  )
  const mappedElements = items
    .filter((item): item is PendingText => item.type === 'text')
    .map((item) => item.liveElement)
  const safelySharesStoryRoot = (item: LiveTextCandidate): boolean => {
    const shared = commonSourceRoot([...mappedElements, item.element])
    return Boolean(
      shared &&
      shared !== item.element.ownerDocument.body &&
      shared !== item.element.ownerDocument.documentElement,
    )
  }
  const candidate =
    candidates.find(
      (item) =>
        !used.has(item.marker) &&
        item.text === normalizedTitle &&
        mappedRoot?.contains(item.element),
    ) ??
    candidates.find(
      (item) =>
        !used.has(item.marker) &&
        item.text === normalizedTitle &&
        /^h[12]$/iu.test(item.element.tagName) &&
        safelySharesStoryRoot(item),
    ) ??
    candidates.find(
      (item) =>
        !used.has(item.marker) &&
        /^h[12]$/iu.test(item.element.tagName) &&
        mappedRoot?.contains(item.element),
    )
  if (candidate) {
    items.unshift({
      type: 'text',
      kind: 'title',
      headingLevel: 1,
      text: candidate.text,
      liveElement: candidate.element,
      marker: candidate.marker,
    })
  }
}

function finalizedStructure(items: readonly PendingItem[]): {
  structure: DocumentStructureItem[]
  blocks: DocumentTextBlock[]
  sourceElements: Map<string, HTMLElement>
} {
  const structure: DocumentStructureItem[] = []
  const blocks: DocumentTextBlock[] = []
  const sourceElements = new Map<string, HTMLElement>()
  let textOrder = 0

  for (let order = 0; order < items.length; order += 1) {
    const item = items[order]!
    if (item.type === 'text') {
      const itemId = stableTextId(textOrder, item.kind, item.text)
      const block: DocumentTextBlock = {
        itemId,
        order: textOrder,
        kind: item.kind,
        text: item.text,
        ...(item.headingLevel === undefined ? {} : { headingLevel: item.headingLevel }),
      }
      blocks.push(block)
      structure.push({ type: 'text', ...block })
      sourceElements.set(itemId, item.liveElement)
      textOrder += 1
      continue
    }
    if (item.type === 'image') {
      structure.push({
        ...item,
        itemId: `document-image-${order}`,
        order,
      })
      continue
    }
    structure.push({ type: 'separator', itemId: `document-separator-${order}`, order })
  }
  return { structure, blocks, sourceElements }
}

async function detectDocumentChapterInternal(
  live: Document,
  options: DocumentExtractionOptions = {},
): Promise<DocumentDetection> {
  let annotated: AnnotatedClone
  try {
    annotated = annotateClone(live)
  } catch {
    return { kind: 'not-document', reason: 'unmapped-content' }
  }

  let article: ReturnType<Readability<HTMLElement>['parse']>
  try {
    article = new Readability<HTMLElement>(annotated.document, {
      charThreshold: 500,
      serializer: (node) => node as HTMLElement,
    }).parse()
  } catch {
    return { kind: 'not-document', reason: 'readability-rejected' }
  }
  if (!article?.content) return { kind: 'not-document', reason: 'readability-rejected' }
  const candidates = liveTextCandidates(live, annotated.markerByLive)
  const pending = collectPendingItems(article.content, annotated.liveByMarker, candidates, live.URL)
  titleBlock(pending, article.title, candidates)
  const finalized = finalizedStructure(pending)

  if (finalized.blocks.length > MAX_DOCUMENT_BLOCKS) {
    return { kind: 'rejected', reason: 'too-many-blocks' }
  }
  if (finalized.blocks.length < MIN_BLOCKS) {
    return { kind: 'not-document', reason: 'too-few-blocks' }
  }
  if (finalized.blocks.some((block) => utf8Length(block.text) > MAX_DOCUMENT_BLOCK_UTF8_BYTES)) {
    return { kind: 'rejected', reason: 'block-too-large' }
  }

  const normalizedChapter = finalized.blocks.map((block) => block.text).join('\n\n')
  const characterCount = [...normalizedChapter].length
  if (characterCount < MIN_CHARACTERS) return { kind: 'not-document', reason: 'too-short' }
  if (!predominantlyEnglish(normalizedChapter)) {
    return { kind: 'not-document', reason: 'not-predominantly-english' }
  }

  const mappedElements = [...finalized.sourceElements.values()]
  if (mappedElements.length !== finalized.blocks.length) {
    return { kind: 'not-document', reason: 'unmapped-content' }
  }
  const sourceRoot = commonSourceRoot(mappedElements)
  if (
    !sourceRoot ||
    !sourceRoot.isConnected ||
    sourceRoot === live.body ||
    sourceRoot === live.documentElement ||
    !live.body.contains(sourceRoot)
  ) {
    return { kind: 'not-document', reason: 'unsafe-content-root' }
  }

  const nativeBlocks = toNativeDocumentBlocks(finalized.blocks)
  const title = finalized.blocks.find((block) => block.kind === 'title')?.text
  if (documentWireRequestUtf8Bytes(nativeBlocks) > MAX_DOCUMENT_UTF8_BYTES) {
    return { kind: 'rejected', reason: 'input-too-large' }
  }
  const canonical = canonicalDocumentText(nativeBlocks)
  const digestBytes = new TextEncoder().encode(canonical)
  const sourceSha256 = await sha256Hex(digestBytes.buffer, options.crypto)
  const snapshot = {
    sourceUrl: live.URL,
    sourceSha256,
    ...(title === undefined ? {} : { title }),
    characterCount,
    blocks: nativeBlocks,
  }
  const chapter: DocumentChapter = {
    snapshot,
    structure: finalized.structure,
    sourceRoot,
    sourceElements: finalized.sourceElements,
  }
  return { kind: 'document', chapter }
}

export async function detectDocumentChapter(
  live: Document,
  options: DocumentExtractionOptions = {},
): Promise<DocumentDetection> {
  const performanceApi = live.defaultView?.performance
  markDocumentPerformance(performanceApi, DOCUMENT_EXTRACTION_START_MARK)
  try {
    return await detectDocumentChapterInternal(live, options)
  } finally {
    markDocumentPerformance(performanceApi, DOCUMENT_EXTRACTION_END_MARK)
    measureDocumentPerformance(
      performanceApi,
      DOCUMENT_EXTRACTION_MEASURE,
      DOCUMENT_EXTRACTION_START_MARK,
      DOCUMENT_EXTRACTION_END_MARK,
    )
  }
}
