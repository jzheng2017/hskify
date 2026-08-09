import type { TranslatedText } from '../contracts/browser'
import { installTranslatedText } from '../language/translated-text'
import { LOOKUP_CSS } from '../selection/popover-style'
import {
  attachComparisonControls,
  type ChapterDisplayMode,
  type ComparisonRenderTarget,
} from './controls'
import {
  DocumentFocusTracker,
  type DocumentFocusCallback,
  type IntersectionObserverFactory,
} from './focus'
import {
  DOCUMENT_EXTRACTION_AND_SKELETON_MEASURE,
  DOCUMENT_EXTRACTION_START_MARK,
  DOCUMENT_SKELETON_MOUNTED_MARK,
  markDocumentPerformance,
  measureDocumentPerformance,
} from './metrics'
import type {
  DocumentChapter,
  DocumentImageItem,
  DocumentStructureItem,
  DocumentTextBlock,
} from './types'

const READER_CSS = `
:host {
  all: initial;
  color-scheme: light dark;
  display: block;
}
*, *::before, *::after { box-sizing: border-box; }
.hskify-reader {
  --hskify-text: #1f2937;
  --hskify-muted: #6b7280;
  --hskify-surface: #fffdf8;
  --hskify-accent: #b45309;
  background: var(--hskify-surface);
  color: var(--hskify-text);
  font: 400 18px/1.82 ui-serif, Georgia, Cambria, "Times New Roman", serif;
  margin: 0 auto;
  max-width: 48rem;
  min-height: 12rem;
  padding: clamp(24px, 5vw, 64px) clamp(20px, 6vw, 72px);
  text-rendering: optimizeLegibility;
}
.hskify-reader h1,
.hskify-reader h2,
.hskify-reader h3,
.hskify-reader h4,
.hskify-reader h5,
.hskify-reader h6 {
  color: var(--hskify-text);
  font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
  line-height: 1.25;
  margin: 1.7em 0 .65em;
}
.hskify-reader h1:first-child { margin-top: 0; }
.hskify-reader h1 { font-size: clamp(1.85rem, 4vw, 2.7rem); }
.hskify-reader h2 { font-size: 1.55rem; }
.hskify-reader h3 { font-size: 1.3rem; }
.hskify-reader p,
.hskify-reader blockquote,
.hskify-reader li,
.hskify-reader figcaption { white-space: pre-wrap; }
.hskify-reader p { margin: 0 0 1.18em; }
.hskify-reader blockquote {
  border-inline-start: 3px solid color-mix(in srgb, var(--hskify-accent) 60%, transparent);
  color: color-mix(in srgb, var(--hskify-text) 82%, var(--hskify-muted));
  margin: 1.35em 0;
  padding: .2em 0 .2em 1.15em;
}
.hskify-reader ol,
.hskify-reader ul { margin: 0 0 1.2em; padding-inline-start: 1.65em; }
.hskify-reader li { margin: .4em 0; }
.hskify-reader figure { margin: 2em auto; }
.hskify-reader img {
  display: block;
  height: auto;
  margin: 1.8em auto;
  max-width: 100%;
}
.hskify-reader figcaption {
  color: var(--hskify-muted);
  font-size: .9em;
  margin: -.8em auto 1.8em;
  text-align: center;
}
.hskify-reader hr {
  border: 0;
  color: var(--hskify-muted);
  margin: 2.4em auto;
  text-align: center;
}
.hskify-reader hr::after { content: "• • •"; letter-spacing: .45em; }
.hskify-placeholder[data-hskify-state="pending"] { color: var(--hskify-muted); }
.hskify-placeholder[data-hskify-state="preserved"] { color: var(--hskify-text); }
.hskify-placeholder[data-hskify-state="translated"] { color: var(--hskify-text); }
.hskify-placeholder:focus-visible { outline: 2px solid #2563eb; outline-offset: 4px; }
.hskify-learning-term {
  text-decoration: underline dotted var(--hskify-accent) 1.5px;
  text-underline-offset: .18em;
}
${LOOKUP_CSS}
@media (prefers-color-scheme: dark) {
  .hskify-reader {
    --hskify-text: #e5e7eb;
    --hskify-muted: #9ca3af;
    --hskify-surface: #15171a;
    --hskify-accent: #f59e0b;
  }
}
`

type AttributeSnapshot = {
  name: string
  value: string
}

type BlockState =
  | { state: 'pending' }
  | { state: 'translated'; text: TranslatedText }
  | { state: 'preserved'; reason?: string }

export type DocumentReaderCallbacks = {
  onVisibleBlocksChanged?: DocumentFocusCallback
  onInvalidated?: (reason: 'source-mutation' | 'source-detached') => void
  /**
   * Connects the existing dictionary, selection, pinyin, and speech owner to
   * a final translated element. The returned cleanup is called exactly once.
   */
  attachTranslatedText?: (element: HTMLElement, itemId: string) => void | (() => void)
}

export type DocumentReaderDependencies = {
  intersectionObserverFactory?: IntersectionObserverFactory
}

export type DocumentReaderCounts = {
  translated: number
  preserved: number
  pending: number
}

function sourceAttributes(root: HTMLElement): AttributeSnapshot[] {
  return [...root.attributes].map((attribute) => ({
    name: attribute.name,
    value: attribute.value,
  }))
}

function restoreAttributes(root: HTMLElement, snapshots: readonly AttributeSnapshot[]): void {
  for (const attribute of [...root.attributes]) root.removeAttribute(attribute.name)
  for (const snapshot of snapshots) {
    root.setAttribute(snapshot.name, snapshot.value)
  }
}

function readerImageUrl(documentRef: Document, value: string): string | undefined {
  let parsed: URL
  try {
    parsed = new URL(value, documentRef.URL)
  } catch {
    return undefined
  }
  if (parsed.protocol === 'http:' || parsed.protocol === 'https:' || parsed.protocol === 'blob:') {
    return parsed.href
  }
  if (parsed.protocol === 'data:' && /^data:image\//iu.test(parsed.href)) return parsed.href
  return undefined
}

function sourceImage(documentRef: Document, item: DocumentImageItem): HTMLImageElement {
  const image = documentRef.createElement('img')
  const sourceUrl = readerImageUrl(documentRef, item.sourceUrl)
  if (sourceUrl) image.src = sourceUrl
  image.alt = item.alt
  image.loading = 'lazy'
  image.decoding = 'async'
  if (item.width !== undefined) image.width = item.width
  if (item.height !== undefined) image.height = item.height
  image.dataset.hskifyItemId = item.itemId
  return image
}

function textElement(documentRef: Document, block: DocumentTextBlock): HTMLElement {
  let element: HTMLElement
  switch (block.kind) {
    case 'title':
      element = documentRef.createElement('h1')
      break
    case 'heading':
      element = documentRef.createElement(`h${block.headingLevel ?? 2}`)
      break
    case 'blockquote':
      element = documentRef.createElement('blockquote')
      break
    case 'ordered-list-item':
    case 'unordered-list-item':
      element = documentRef.createElement('li')
      break
    case 'caption':
      element = documentRef.createElement('figcaption')
      break
    case 'paragraph':
      element = documentRef.createElement('p')
      break
  }
  element.className = 'hskify-placeholder'
  element.dataset.hskifyItemId = block.itemId
  element.dataset.hskifyState = 'pending'
  element.lang = 'en'
  element.textContent = block.text
  return element
}

function blockMap(chapter: DocumentChapter): Map<string, DocumentTextBlock> {
  const result = new Map<string, DocumentTextBlock>()
  for (const item of chapter.structure) {
    if (item.type === 'text') result.set(item.itemId, item)
  }
  return result
}

export class DocumentReader implements ComparisonRenderTarget {
  private mode: ChapterDisplayMode = 'chinese'
  private displayedMode: ChapterDisplayMode = 'chinese'
  private readonly documentRef: Document
  private readonly host: HTMLElement
  readonly shadowRoot: ShadowRoot
  readonly lookupElement: HTMLElement
  private readonly reader: HTMLElement
  private readonly blocks: Map<string, DocumentTextBlock>
  private readonly blockElements = new Map<string, HTMLElement>()
  private readonly states = new Map<string, BlockState>()
  private readonly interactionCleanup = new Map<string, () => void>()
  private readonly originalAttributes: AttributeSnapshot[]
  private readonly focus: DocumentFocusTracker
  private readonly mutationObserver: MutationObserver
  private readonly sourceParent: Node
  private detachControls: (() => void) | undefined
  private destroyed = false

  constructor(
    readonly chapter: DocumentChapter,
    private readonly callbacks: DocumentReaderCallbacks = {},
    dependencies: DocumentReaderDependencies = {},
  ) {
    this.documentRef = chapter.sourceRoot.ownerDocument
    const parent = chapter.sourceRoot.parentNode
    if (!parent || !chapter.sourceRoot.isConnected) {
      throw new Error('The document source root must be connected before mounting the reader.')
    }
    this.sourceParent = parent
    this.originalAttributes = sourceAttributes(chapter.sourceRoot)
    this.blocks = blockMap(chapter)

    this.host = this.documentRef.createElement('div')
    this.host.dataset.hskifyOwned = 'true'
    this.host.dataset.hskifyDocumentReader = 'true'
    this.host.dataset.hskifySourceBlockCount = String(chapter.snapshot.blocks.length)
    this.host.dataset.hskifySourceCharacterCount = String(chapter.snapshot.characterCount)
    this.host.dataset.hskifySourceSha256 = chapter.snapshot.sourceSha256
    this.host.style.setProperty('all', 'initial', 'important')
    this.host.style.setProperty('display', 'block', 'important')
    this.host.style.setProperty('isolation', 'isolate', 'important')
    this.shadowRoot = this.host.attachShadow({ mode: 'open' })
    const style = this.documentRef.createElement('style')
    style.textContent = READER_CSS
    this.reader = this.documentRef.createElement('main')
    this.reader.className = 'hskify-reader'
    this.reader.lang = 'zh-CN'
    this.reader.setAttribute('aria-label', 'Translated light-novel chapter')
    this.lookupElement = this.documentRef.createElement('aside')
    this.lookupElement.className = 'hskify-lookup'
    this.lookupElement.hidden = true
    this.buildSkeleton(chapter.structure)
    this.shadowRoot.append(style, this.reader, this.lookupElement)

    const MutationObserverClass = this.documentRef.defaultView?.MutationObserver
    if (!MutationObserverClass) throw new Error('MutationObserver is unavailable.')
    const orderedIds = chapter.snapshot.blocks.map((block) => block.itemId)
    this.focus = new DocumentFocusTracker(
      orderedIds,
      callbacks.onVisibleBlocksChanged ?? (() => undefined),
      dependencies.intersectionObserverFactory,
    )
    this.mutationObserver = new MutationObserverClass(this.onMutation)

    try {
      parent.insertBefore(this.host, chapter.sourceRoot.nextSibling)
      markDocumentPerformance(
        this.documentRef.defaultView?.performance,
        DOCUMENT_SKELETON_MOUNTED_MARK,
      )
      measureDocumentPerformance(
        this.documentRef.defaultView?.performance,
        DOCUMENT_EXTRACTION_AND_SKELETON_MEASURE,
        DOCUMENT_EXTRACTION_START_MARK,
        DOCUMENT_SKELETON_MOUNTED_MARK,
      )
      this.applySourceHidden()
      for (const itemId of orderedIds) {
        const placeholder = this.blockElements.get(itemId)
        const source = chapter.sourceElements.get(itemId)
        if (placeholder) this.focus.observe(placeholder, itemId)
        if (source) this.focus.observe(source, itemId)
      }
      this.observeSource()
      this.detachControls = attachComparisonControls(this.documentRef, this)
    } catch (error) {
      this.mutationObserver.disconnect()
      this.focus.destroy()
      restoreAttributes(chapter.sourceRoot, this.originalAttributes)
      this.host.remove()
      throw error
    }
  }

  get currentMode(): ChapterDisplayMode {
    return this.mode
  }

  get isDestroyed(): boolean {
    return this.destroyed
  }

  counts(): DocumentReaderCounts {
    let translated = 0
    let preserved = 0
    for (const state of this.states.values()) {
      if (state.state === 'translated') translated += 1
      if (state.state === 'preserved') preserved += 1
    }
    return {
      translated,
      preserved,
      pending: this.states.size - translated - preserved,
    }
  }

  /** Installs a terminal translation once; replay of the same update is a no-op. */
  installBlock(itemId: string, text: TranslatedText): boolean {
    this.ensureActive()
    const block = this.blocks.get(itemId)
    const element = this.blockElements.get(itemId)
    if (!block || !element) throw new Error(`Unknown document block: ${itemId}`)
    if (text.sourceText !== block.text) {
      throw new Error(`Source mismatch for document block: ${itemId}`)
    }
    const state = this.states.get(itemId)
    if (!state || state.state !== 'pending') return false

    installTranslatedText(element, text)
    element.dataset.hskifyState = 'translated'
    element.tabIndex = 0
    this.states.set(itemId, { state: 'translated', text })
    const cleanup = this.callbacks.attachTranslatedText?.(element, itemId)
    if (cleanup) this.interactionCleanup.set(itemId, cleanup)
    return true
  }

  /** Leaves the final source text visible for a block the native job preserved. */
  preserveBlock(itemId: string, sourceText: string, reason?: string): boolean {
    this.ensureActive()
    const block = this.blocks.get(itemId)
    const element = this.blockElements.get(itemId)
    if (!block || !element) throw new Error(`Unknown document block: ${itemId}`)
    if (sourceText !== block.text) {
      throw new Error(`Source mismatch for document block: ${itemId}`)
    }
    const state = this.states.get(itemId)
    if (!state || state.state !== 'pending') return false
    element.dataset.hskifyState = 'preserved'
    if (reason) element.dataset.hskifyPreservedReason = reason
    this.states.set(
      itemId,
      reason === undefined ? { state: 'preserved' } : { state: 'preserved', reason },
    )
    return true
  }

  setMode(mode: ChapterDisplayMode): void {
    this.ensureActive()
    this.mode = mode
    this.display(mode)
  }

  showOriginalForComparison(): void {
    if (!this.destroyed) this.display('original')
  }

  restoreSelectedMode(): void {
    if (!this.destroyed) this.display(this.mode)
  }

  /** Cancellation, fatal failure, and navigation all use the same exact restoration path. */
  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.mutationObserver.disconnect()
    this.focus.destroy()
    for (const cleanup of this.interactionCleanup.values()) cleanup()
    this.interactionCleanup.clear()
    this.detachControls?.()
    this.detachControls = undefined
    restoreAttributes(this.chapter.sourceRoot, this.originalAttributes)
    this.host.remove()
  }

  private buildSkeleton(structure: readonly DocumentStructureItem[]): void {
    let list: HTMLOListElement | HTMLUListElement | undefined
    let listKind: 'ordered-list-item' | 'unordered-list-item' | undefined
    for (const item of structure) {
      if (item.type === 'text') {
        const element = textElement(this.documentRef, item)
        this.blockElements.set(item.itemId, element)
        this.states.set(item.itemId, { state: 'pending' })
        if (item.kind === 'ordered-list-item' || item.kind === 'unordered-list-item') {
          if (!list || listKind !== item.kind) {
            list = this.documentRef.createElement(item.kind === 'ordered-list-item' ? 'ol' : 'ul')
            listKind = item.kind
            this.reader.append(list)
          }
          list.append(element as HTMLLIElement)
          continue
        }
        list = undefined
        listKind = undefined
        if (item.kind === 'caption') {
          const previous = this.reader.lastElementChild
          if (
            previous?.tagName.toLowerCase() === 'figure' &&
            previous.querySelector('img') &&
            !previous.querySelector('figcaption')
          ) {
            previous.append(element)
          } else {
            const figure = this.documentRef.createElement('figure')
            figure.append(element)
            this.reader.append(figure)
          }
        } else {
          this.reader.append(element)
        }
        continue
      }

      list = undefined
      listKind = undefined
      if (item.type === 'image') {
        const figure = this.documentRef.createElement('figure')
        figure.append(sourceImage(this.documentRef, item))
        this.reader.append(figure)
      } else {
        const separator = this.documentRef.createElement('hr')
        separator.dataset.hskifyItemId = item.itemId
        this.reader.append(separator)
      }
    }
  }

  private ensureActive(): void {
    if (this.destroyed) throw new Error('The document reader has been destroyed.')
  }

  private applySourceHidden(): void {
    const root = this.chapter.sourceRoot
    root.setAttribute('hidden', '')
    root.setAttribute('inert', '')
    root.setAttribute('aria-hidden', 'true')
  }

  private observeSource(): void {
    this.mutationObserver.observe(this.chapter.sourceRoot, {
      attributes: true,
      characterData: true,
      childList: true,
      subtree: true,
    })
    this.mutationObserver.observe(this.sourceParent, { childList: true })
  }

  private display(mode: ChapterDisplayMode): void {
    if (this.displayedMode === mode) return
    const anchorId = this.focus.visibleItemIds()[0]
    const beforeElement = anchorId
      ? this.displayedMode === 'chinese'
        ? this.blockElements.get(anchorId)
        : this.chapter.sourceElements.get(anchorId)
      : undefined
    const beforeTop = beforeElement?.getBoundingClientRect().top

    this.mutationObserver.disconnect()
    if (mode === 'original') {
      restoreAttributes(this.chapter.sourceRoot, this.originalAttributes)
      this.host.hidden = true
      this.host.style.setProperty('display', 'none', 'important')
    } else {
      this.host.hidden = false
      this.host.style.setProperty('display', 'block', 'important')
      this.applySourceHidden()
    }
    this.displayedMode = mode
    this.observeSource()

    const afterElement = anchorId
      ? mode === 'chinese'
        ? this.blockElements.get(anchorId)
        : this.chapter.sourceElements.get(anchorId)
      : undefined
    const afterTop = afterElement?.getBoundingClientRect().top
    if (
      beforeTop !== undefined &&
      afterTop !== undefined &&
      Number.isFinite(beforeTop) &&
      Number.isFinite(afterTop) &&
      Math.abs(afterTop - beforeTop) > 0.5
    ) {
      this.documentRef.defaultView?.scrollBy?.(0, afterTop - beforeTop)
    }
  }

  private readonly onMutation: MutationCallback = (records): void => {
    if (this.destroyed) return
    let reason: 'source-mutation' | 'source-detached' | undefined
    for (const record of records) {
      if (record.type === 'childList' && record.target === this.sourceParent) {
        if (
          [...record.removedNodes].some(
            (node) => node === this.chapter.sourceRoot || node.contains?.(this.chapter.sourceRoot),
          )
        ) {
          reason = 'source-detached'
          break
        }
        continue
      }
      if (
        record.target === this.chapter.sourceRoot ||
        this.chapter.sourceRoot.contains(record.target)
      ) {
        reason = 'source-mutation'
        break
      }
    }
    if (!reason) return
    this.destroy()
    this.callbacks.onInvalidated?.(reason)
  }
}

export function mountDocumentReader(
  chapter: DocumentChapter,
  callbacks: DocumentReaderCallbacks = {},
  dependencies: DocumentReaderDependencies = {},
): DocumentReader {
  return new DocumentReader(chapter, callbacks, dependencies)
}
