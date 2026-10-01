import { Readability } from '@mozilla/readability'
import { franc } from 'franc-min'

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
  type DocumentStructureItem,
  type DocumentTextBlock,
  type SourceTextPart,
} from './types'

const SOURCE_MARKER = 'data-hskify-source-marker'
const EXCLUDED =
  'nav,aside,form,button,input,select,textarea,script,style,template,' +
  '[role="navigation"],[role="banner"],[role="complementary"],[role="contentinfo"],' +
  '[role="form"],[aria-hidden="true"],[hidden],[contenteditable="true"],[data-hskify-owned]'
const UI_NAME =
  /(?:^|[-_\s])(?:ad|ads|advert|advertisement|author|byline|breadcrumb|comment|comments|cookie|login|menu|metadata|modal|nav|newsletter|pagination|promo|recommend|related|share|sidebar|social|toolbar|widget)(?:$|[-_\s])/iu
const BLOCK_TAGS = /^(?:article|section|div|p|h[1-6]|blockquote|li|figcaption)$/iu
const utf8 = new TextEncoder()

export type DocumentExtractionOptions = { crypto?: Crypto; root?: HTMLElement }

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

export function documentWireRequestUtf8Bytes(blocks: DocumentJobRequest['blocks']): number {
  const request: DocumentJobRequest = {
    clientRequestId: 'x'.repeat(128),
    retryItemIds: blocks.map((block) => block.itemId),
    focus: {
      kind: 'document',
      active: true,
      visibleBlockIds: blocks.slice(0, 64).map((block) => block.itemId),
    },
    buildFingerprint: BUILD_FINGERPRINT,
    pageSessionId: 'x'.repeat(256),
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
  return utf8.encode(JSON.stringify(request)).byteLength
}

export function eligibleStoryElement(
  element: Element,
  styles = new WeakMap<Element, boolean>(),
): boolean {
  if (element.closest(EXCLUDED)) return false
  for (let current: Element | null = element; current; current = current.parentElement) {
    if (
      UI_NAME.test(
        `${current.id} ${typeof current.className === 'string' ? current.className : ''}`,
      )
    )
      return false
    let shown = styles.get(current)
    if (shown === undefined) {
      const style = current.ownerDocument.defaultView?.getComputedStyle(current)
      shown =
        !style ||
        (style.display !== 'none' &&
          style.visibility !== 'hidden' &&
          style.visibility !== 'collapse')
      styles.set(current, shown)
    }
    if (!shown) return false
  }
  return true
}

/** Bounded language evidence sampled across the chapter, rather than inferred from its alphabet. */
export function consistentlyEnglish(text: string): boolean {
  if (text.length < 160) return false
  const size = Math.min(1_024, text.length)
  const offsets = new Set([0, Math.floor((text.length - size) / 2), text.length - size])
  for (const offset of offsets) {
    if (franc(text.slice(offset, offset + size), { minLength: 160 }) !== 'eng') return false
  }
  return true
}

function commonRoot(elements: readonly HTMLElement[]): HTMLElement | null {
  let root = elements[0] ?? null
  while (root && !elements.every((element) => root!.contains(element))) root = root.parentElement
  return root
}

function identifyRegion(live: Document): { root: HTMLElement; title?: string } | undefined {
  const styles = new WeakMap<Element, boolean>()
  const clone = live.cloneNode(true) as Document
  const original = [live.documentElement, ...live.documentElement.querySelectorAll('*')]
  const copied = [clone.documentElement, ...clone.documentElement.querySelectorAll('*')]
  if (original.length !== copied.length) return undefined
  const byMarker = new Map<string, HTMLElement>()
  for (let index = 0; index < copied.length; index++) {
    const marker = index.toString(36)
    copied[index]!.setAttribute(SOURCE_MARKER, marker)
    byMarker.set(marker, original[index] as HTMLElement)
  }
  const article = new Readability<HTMLElement>(clone, {
    charThreshold: 500,
    serializer: (node) => node as HTMLElement,
  }).parse()
  if (!article?.content) return undefined
  const elements: HTMLElement[] = []
  for (const marked of article.content.querySelectorAll(`[${SOURCE_MARKER}]`)) {
    const source = byMarker.get(marked.getAttribute(SOURCE_MARKER)!)
    if (
      source &&
      BLOCK_TAGS.test(source.tagName) &&
      eligibleStoryElement(source, styles) &&
      normalizeDocumentText(marked.textContent ?? '')
    )
      elements.push(source)
  }
  let root = commonRoot(elements)
  const title = normalizeDocumentText(article.title ?? '')
  const heading = [...live.querySelectorAll<HTMLElement>('h1,h2')].find(
    (element) =>
      eligibleStoryElement(element, styles) &&
      normalizeDocumentText(element.textContent ?? '') === title,
  )
  if (root && heading && !root.contains(heading)) {
    const shared = commonRoot([root, heading])
    if (shared && shared !== live.body && shared !== live.documentElement) root = shared
  }
  return root ? { root, ...(title ? { title } : {}) } : undefined
}

function kindFor(element: HTMLElement): {
  kind: DocumentBlockKind
  headingLevel?: 1 | 2 | 3 | 4 | 5 | 6
} {
  if (/^h[1-6]$/iu.test(element.tagName))
    return {
      kind: 'heading',
      headingLevel: Number(element.tagName[1]) as 1 | 2 | 3 | 4 | 5 | 6,
    }
  if (element.tagName === 'BLOCKQUOTE') return { kind: 'blockquote' }
  if (element.tagName === 'FIGCAPTION') return { kind: 'caption' }
  if (element.tagName === 'LI')
    return {
      kind: element.closest('ol') ? 'ordered-list-item' : 'unordered-list-item',
    }
  return { kind: element.closest('blockquote') ? 'blockquote' : 'paragraph' }
}

function stableId(order: number, kind: string, text: string): string {
  let hash = 0x811c9dc5
  for (const byte of utf8.encode(`${order}\u001f${kind}\u001f${text}`))
    hash = Math.imul(hash ^ byte, 0x01000193)
  return `document-${order}-${kind}-${(hash >>> 0).toString(16).padStart(8, '0')}`
}

const sourceIdentities = new WeakMap<Node, string>()
function sourceIdentity(node: Node): string {
  const known = sourceIdentities.get(node)
  if (known) return known
  const path: string[] = []
  for (let current: Node | null = node; current?.parentNode; current = current.parentNode) {
    path.push(
      `${current.nodeName}:${Array.prototype.indexOf.call(current.parentNode.childNodes, current)}`,
    )
  }
  const id = stableId(0, 'source', path.reverse().join('/'))
  sourceIdentities.set(node, id)
  return id
}

function sentenceGroupEnds(text: string): number[] {
  const ends: number[] = []
  let start = 0,
    end = 0
  for (const sentence of new Intl.Segmenter('en', { granularity: 'sentence' }).segment(text)) {
    const next = sentence.index + sentence.segment.length
    if (next - start > 600 && end > start) {
      ends.push(end)
      start = end
    }
    while (next - start > 600) {
      let split = text.lastIndexOf(' ', start + 600)
      if (split <= start) split = start + 600
      ends.push(split)
      start = split
    }
    end = next
  }
  if (end > start) ends.push(end)
  return ends
}

/** Visits live source once. Every eligible text node belongs to exactly one slot. */
function extractRegion(root: HTMLElement, title?: string) {
  const styles = new WeakMap<Element, boolean>()
  const structure: DocumentStructureItem[] = []
  const blocks: DocumentTextBlock[] = []
  const sourceElements = new Map<string, HTMLElement>()
  const sourceSlots = new Map<string, readonly SourceTextPart[]>()
  const subOrders = new Map<HTMLElement, number>()
  let owner = root
  let nodes: Array<{ node: Text; offset: number }> = []
  let offset = 0
  let parts: string[] = []
  const flush = () => {
    const raw = parts.join('')
    const semantic = kindFor(owner)
    let start = 0
    for (const end of sentenceGroupEnds(raw)) {
      const text = normalizeDocumentText(raw.slice(start, end))
      if (text && nodes.length) {
        if (
          semantic.kind === 'heading' &&
          (text === title ||
            (semantic.headingLevel === 1 && !blocks.some((block) => block.kind === 'title')))
        )
          semantic.kind = 'title'
        const order = blocks.length
        const parentBlockId = sourceIdentity(owner)
        const subItemOrder = subOrders.get(owner) ?? 0
        subOrders.set(owner, subItemOrder + 1)
        const segments = nodes.flatMap(({ node, offset }) => {
          const from = Math.max(0, start - offset),
            to = Math.min(node.length, end - offset)
          return from < to ? [{ node, start: from, end: to }] : []
        })
        const itemId = `${sourceIdentity(segments[0]!.node)}-${stableId(start, semantic.kind, text)}`
        const block = { itemId, parentBlockId, subItemOrder, order, ...semantic, text }
        blocks.push(block)
        structure.push({ type: 'text', ...block })
        sourceElements.set(itemId, owner)
        sourceSlots.set(itemId, segments)
      }
      start = end
    }
    nodes = []
    parts = []
    offset = 0
  }
  const visit = (node: Node): void => {
    if (node.nodeType === 3) {
      nodes.push({ node: node as Text, offset })
      const data = (node as Text).data
      parts.push(data.replace(/[\r\n\t\f\v]/gu, ' '))
      offset += data.length
      return
    }
    if (node.nodeType !== 1) return
    const element = node as HTMLElement
    if (!eligibleStoryElement(element, styles)) return
    if (element.tagName === 'BR') {
      parts.push('\n')
      offset++
      return
    }
    if (/^(?:IMG|HR|CANVAS|VIDEO)$/u.test(element.tagName)) {
      flush()
      const order = structure.length
      if (element.tagName === 'HR')
        structure.push({
          type: 'separator',
          itemId: `document-separator-${order}`,
          order,
        })
      if (element.tagName === 'IMG') {
        const image = element as HTMLImageElement
        const src = image.currentSrc || image.getAttribute('src') || image.getAttribute('data-src')
        if (src) {
          try {
            const url = new URL(src, root.ownerDocument.URL)
            if (/^(?:https?:|blob:|data:)$/u.test(url.protocol))
              structure.push({
                type: 'image',
                itemId: `document-image-${order}`,
                order,
                sourceUrl: url.href,
                alt: image.alt,
                ...(image.naturalWidth
                  ? { width: image.naturalWidth, height: image.naturalHeight }
                  : {}),
              })
          } catch {
            /* Invalid media never removes its live node. */
          }
        }
      }
      return
    }
    const previous = owner
    const block = BLOCK_TAGS.test(element.tagName)
    if (block) {
      flush()
      owner = element
    }
    for (const child of element.childNodes) visit(child)
    if (block) {
      flush()
      owner = previous
    }
  }
  visit(root)
  flush()
  return { structure, blocks, sourceElements, sourceSlots }
}

async function detectInternal(
  live: Document,
  options: DocumentExtractionOptions,
): Promise<DocumentDetection> {
  let region: ReturnType<typeof identifyRegion>
  try {
    region = options.root ? { root: options.root } : identifyRegion(live)
  } catch {
    return { kind: 'not-document', reason: 'readability-rejected' }
  }
  if (!region) return { kind: 'not-document', reason: 'readability-rejected' }
  const { root } = region
  if (
    !root.isConnected ||
    root === live.body ||
    root === live.documentElement ||
    !live.body.contains(root)
  )
    return { kind: 'not-document', reason: 'unsafe-content-root' }
  const extracted = extractRegion(root, region.title)
  if (extracted.blocks.length > MAX_DOCUMENT_BLOCKS)
    return { kind: 'rejected', reason: 'too-many-blocks' }
  if (extracted.blocks.length < (options.root ? 1 : 5))
    return { kind: 'not-document', reason: 'too-few-blocks' }
  if (
    extracted.blocks.some(
      (block) => utf8.encode(block.text).byteLength > MAX_DOCUMENT_BLOCK_UTF8_BYTES,
    )
  )
    return {
      kind: 'rejected',
      reason: 'block-too-large',
    }
  const normalized = extracted.blocks.map((block) => block.text).join('\n\n')
  const characterCount = [...normalized].length
  if (!options.root && characterCount < 1_000) return { kind: 'not-document', reason: 'too-short' }
  if (!options.root && !consistentlyEnglish(normalized))
    return {
      kind: 'not-document',
      reason: 'not-predominantly-english',
    }
  const nativeBlocks = toNativeDocumentBlocks(extracted.blocks)
  if (documentWireRequestUtf8Bytes(nativeBlocks) > MAX_DOCUMENT_UTF8_BYTES)
    return {
      kind: 'rejected',
      reason: 'input-too-large',
    }
  const revisions = new Map<Text, string>()
  for (const segments of extracted.sourceSlots.values())
    for (const { node } of segments) revisions.set(node, node.data)
  let changed = false
  const observer = new MutationObserver((records) => {
    changed ||= records.length > 0
  })
  observer.observe(root, {
    subtree: true,
    childList: true,
    characterData: true,
    attributes: true,
    attributeFilter: ['hidden', 'aria-hidden', 'contenteditable'],
  })
  let sourceSha256: string
  try {
    sourceSha256 = await sha256Hex(
      utf8.encode(canonicalDocumentText(nativeBlocks)).buffer,
      options.crypto,
    )
    changed ||= observer.takeRecords().length > 0
  } finally {
    observer.disconnect()
  }
  if (
    changed ||
    !root.isConnected ||
    [...revisions].some(([node, text]) => !root.contains(node) || node.data !== text)
  )
    return { kind: 'not-document', reason: 'unmapped-content' }
  const title = extracted.blocks.find((block) => block.kind === 'title')?.text
  const chapter: DocumentChapter = {
    ...extracted,
    sourceRoot: root,
    sourceRevisions: revisions,
    snapshot: {
      sourceUrl: live.URL,
      sourceSha256,
      characterCount,
      blocks: nativeBlocks,
      ...(title ? { title } : {}),
    },
  }
  return { kind: 'document', chapter }
}

export async function detectDocumentChapter(
  live: Document,
  options: DocumentExtractionOptions = {},
): Promise<DocumentDetection> {
  const performance = live.defaultView?.performance
  markDocumentPerformance(performance, DOCUMENT_EXTRACTION_START_MARK)
  try {
    return await detectInternal(live, options)
  } finally {
    markDocumentPerformance(performance, DOCUMENT_EXTRACTION_END_MARK)
    measureDocumentPerformance(
      performance,
      DOCUMENT_EXTRACTION_MEASURE,
      DOCUMENT_EXTRACTION_START_MARK,
      DOCUMENT_EXTRACTION_END_MARK,
    )
  }
}
