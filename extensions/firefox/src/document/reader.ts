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
import type { DocumentChapter, DocumentTextBlock } from './types'

const OVERLAY_CSS = `
:host {
  all: initial;
  height: 0;
  inset: 0;
  pointer-events: none;
  position: fixed;
  width: 0;
  z-index: 2147483646;
}
${LOOKUP_CSS}
`

type AttributeSnapshot = {
  name: string
  value: string
}

type SourceElementSnapshot = {
  attributes: AttributeSnapshot[]
  childNodes: Node[]
}

type BlockState =
  | { state: 'pending' }
  | { state: 'translated'; text: TranslatedText }
  | { state: 'preserved'; reason?: string }

export type DocumentReaderCallbacks = {
  onVisibleBlocksChanged?: DocumentFocusCallback
  onInvalidated?: (reason: 'source-mutation' | 'source-detached') => void
  /** Connects dictionary, selection, pinyin, and speech to a final translated block. */
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
  for (const snapshot of snapshots) root.setAttribute(snapshot.name, snapshot.value)
}

function blockMap(chapter: DocumentChapter): Map<string, DocumentTextBlock> {
  const result = new Map<string, DocumentTextBlock>()
  for (const item of chapter.structure) {
    if (item.type === 'text') result.set(item.itemId, item)
  }
  return result
}

function pendingPlaceholder(documentRef: Document): HTMLElement {
  const placeholder = documentRef.createElement('span')
  placeholder.dataset.hskifyPending = 'true'
  placeholder.setAttribute('aria-hidden', 'true')
  // Preserve a line box without exposing source-language text while the final
  // translation is pending or source-preserved.
  placeholder.textContent = '\u200b'
  return placeholder
}

export class DocumentReader implements ComparisonRenderTarget {
  private mode: ChapterDisplayMode = 'chinese'
  private displayedMode: ChapterDisplayMode = 'chinese'
  private readonly documentRef: Document
  private readonly host: HTMLElement
  readonly shadowRoot: ShadowRoot
  readonly lookupElement: HTMLElement
  readonly interactionRoot: HTMLElement
  private readonly blocks: Map<string, DocumentTextBlock>
  private readonly blockElements = new Map<string, HTMLElement>()
  private readonly sourceSnapshots = new Map<string, SourceElementSnapshot>()
  private readonly states = new Map<string, BlockState>()
  private readonly interactionCleanup = new Map<string, () => void>()
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
    this.interactionRoot = chapter.sourceRoot
    this.blocks = blockMap(chapter)

    for (const block of chapter.snapshot.blocks) {
      const element = chapter.sourceElements.get(block.itemId)
      if (!element || !element.isConnected || !chapter.sourceRoot.contains(element)) {
        throw new Error(`Document block is not connected beneath the source root: ${block.itemId}`)
      }
      this.blockElements.set(block.itemId, element)
      this.sourceSnapshots.set(block.itemId, {
        attributes: sourceAttributes(element),
        childNodes: [...element.childNodes],
      })
      this.states.set(block.itemId, { state: 'pending' })
    }

    this.host = this.documentRef.createElement('div')
    this.host.dataset.hskifyOwned = 'true'
    this.host.dataset.hskifyDocumentReader = 'true'
    this.host.dataset.hskifySourceBlockCount = String(chapter.snapshot.blocks.length)
    this.host.dataset.hskifySourceCharacterCount = String(chapter.snapshot.characterCount)
    this.host.dataset.hskifySourceSha256 = chapter.snapshot.sourceSha256
    this.shadowRoot = this.host.attachShadow({ mode: 'open' })
    const style = this.documentRef.createElement('style')
    style.textContent = OVERLAY_CSS
    this.lookupElement = this.documentRef.createElement('aside')
    this.lookupElement.className = 'hskify-lookup'
    this.lookupElement.hidden = true
    this.shadowRoot.append(style, this.lookupElement)

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
      ;(this.documentRef.body ?? this.documentRef.documentElement).append(this.host)
      this.applyChineseView()
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
      for (const itemId of orderedIds) {
        const source = this.blockElements.get(itemId)
        if (source) this.focus.observe(source, itemId)
      }
      this.observeSource()
      this.detachControls = attachComparisonControls(this.documentRef, this)
    } catch (error) {
      this.mutationObserver.disconnect()
      this.focus.destroy()
      this.restoreOriginalView()
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
    if (!block || !this.blockElements.has(itemId)) {
      throw new Error(`Unknown document block: ${itemId}`)
    }
    if (text.sourceText !== block.text) {
      throw new Error(`Source mismatch for document block: ${itemId}`)
    }
    const state = this.states.get(itemId)
    if (!state || state.state !== 'pending') return false

    this.states.set(itemId, { state: 'translated', text })
    if (this.displayedMode === 'chinese') this.mutateSource(() => this.renderChineseBlock(itemId))
    return true
  }

  /** Records a terminal preservation without leaking English into Chinese mode. */
  preserveBlock(itemId: string, sourceText: string, reason?: string): boolean {
    this.ensureActive()
    const block = this.blocks.get(itemId)
    if (!block || !this.blockElements.has(itemId)) {
      throw new Error(`Unknown document block: ${itemId}`)
    }
    if (sourceText !== block.text) {
      throw new Error(`Source mismatch for document block: ${itemId}`)
    }
    const state = this.states.get(itemId)
    if (!state || state.state !== 'pending') return false
    this.states.set(
      itemId,
      reason === undefined ? { state: 'preserved' } : { state: 'preserved', reason },
    )
    if (this.displayedMode === 'chinese') this.mutateSource(() => this.renderChineseBlock(itemId))
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
    this.restoreOriginalView()
    this.detachControls?.()
    this.detachControls = undefined
    this.host.remove()
  }

  private ensureActive(): void {
    if (this.destroyed) throw new Error('The document reader has been destroyed.')
  }

  private clearInteraction(itemId: string): void {
    this.interactionCleanup.get(itemId)?.()
    this.interactionCleanup.delete(itemId)
  }

  private renderChineseBlock(itemId: string): void {
    const element = this.blockElements.get(itemId)
    const snapshot = this.sourceSnapshots.get(itemId)
    const state = this.states.get(itemId)
    if (!element || !snapshot || !state) return

    this.clearInteraction(itemId)
    restoreAttributes(element, snapshot.attributes)
    element.dataset.hskifyItemId = itemId
    element.dataset.hskifyState = state.state

    if (state.state === 'translated') {
      installTranslatedText(element, state.text)
      element.tabIndex = 0
      for (const term of element.querySelectorAll<HTMLElement>('.hskify-learning-term')) {
        term.style.setProperty('text-decoration', 'underline dotted currentColor 1.5px')
        term.style.setProperty('text-underline-offset', '.18em')
      }
      const cleanup = this.callbacks.attachTranslatedText?.(element, itemId)
      if (cleanup) this.interactionCleanup.set(itemId, cleanup)
      return
    }

    element.lang = 'zh-CN'
    if (state.state === 'preserved' && state.reason) {
      element.dataset.hskifyPreservedReason = state.reason
    }
    element.replaceChildren(pendingPlaceholder(this.documentRef))
  }

  private applyChineseView(): void {
    for (const itemId of this.chapter.snapshot.blocks.map((block) => block.itemId)) {
      this.renderChineseBlock(itemId)
    }
  }

  private restoreOriginalView(): void {
    for (const [itemId, snapshot] of this.sourceSnapshots) {
      const element = this.blockElements.get(itemId)
      if (!element) continue
      this.clearInteraction(itemId)
      restoreAttributes(element, snapshot.attributes)
      element.replaceChildren(...snapshot.childNodes)
    }
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

  private mutateSource(update: () => void): void {
    this.mutationObserver.disconnect()
    try {
      update()
    } finally {
      this.observeSource()
    }
  }

  private display(mode: ChapterDisplayMode): void {
    if (this.displayedMode === mode) return
    const anchorId = this.focus.visibleItemIds()[0]
    const anchor = anchorId ? this.blockElements.get(anchorId) : undefined
    const beforeTop = anchor?.getBoundingClientRect().top

    this.mutationObserver.disconnect()
    if (mode === 'original') this.restoreOriginalView()
    else this.applyChineseView()
    this.displayedMode = mode
    this.observeSource()

    const afterTop = anchor?.getBoundingClientRect().top
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
    if (reason === 'source-mutation') this.adoptExternalMutations(records)
    this.destroy()
    this.callbacks.onInvalidated?.(reason)
  }

  private adoptExternalMutations(records: readonly MutationRecord[]): void {
    for (const record of records) {
      const element = [...this.blockElements.values()].find(
        (candidate) => candidate === record.target || candidate.contains(record.target),
      )
      if (!element) continue
      const itemId = [...this.blockElements].find(([, candidate]) => candidate === element)?.[0]
      const snapshot = itemId ? this.sourceSnapshots.get(itemId) : undefined
      if (!snapshot) continue
      if (record.type === 'childList' || record.type === 'characterData') {
        snapshot.childNodes = [...element.childNodes]
        continue
      }
      if (record.type === 'attributes' && record.attributeName) {
        const current = element.getAttribute(record.attributeName)
        snapshot.attributes = snapshot.attributes.filter(
          (attribute) => attribute.name !== record.attributeName,
        )
        if (current !== null) {
          snapshot.attributes.push({ name: record.attributeName, value: current })
        }
      }
    }
  }
}

export function mountDocumentReader(
  chapter: DocumentChapter,
  callbacks: DocumentReaderCallbacks = {},
  dependencies: DocumentReaderDependencies = {},
): DocumentReader {
  return new DocumentReader(chapter, callbacks, dependencies)
}
