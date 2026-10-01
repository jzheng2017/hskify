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
import type { DocumentChapter } from './types'

type BlockState =
  | { state: 'pending' }
  | { state: 'translated'; text: TranslatedText }
  | { state: 'preserved'; reason?: string }
type SourceNode = { original: string; expected: string; itemId: string }
type Slot = { element: HTMLElement; nodes: readonly Text[] }

export type DocumentReaderCallbacks = {
  onVisibleBlocksChanged?: DocumentFocusCallback
  onInvalidated?: (reason: 'source-mutation' | 'source-detached') => void
  onRetry?: (itemId: string) => void
  attachTranslatedText?: (element: HTMLElement, itemId: string) => void | (() => void)
}
export type DocumentReaderDependencies = {
  intersectionObserverFactory?: IntersectionObserverFactory
}
export type DocumentReaderCounts = { translated: number; preserved: number; pending: number }

/** Owns only inserted text slots and writes to retained Text nodes, never site elements. */
export class DocumentReader implements ComparisonRenderTarget {
  private mode: ChapterDisplayMode = 'chinese'
  private displayedMode: ChapterDisplayMode = 'chinese'
  private readonly documentRef: Document
  private readonly host: HTMLElement
  readonly shadowRoot: ShadowRoot
  readonly lookupElement: HTMLElement
  readonly interactionRoot: HTMLElement
  private readonly sources = new Map<Text, SourceNode>()
  private readonly sourceAncestors = new WeakSet<Node>()
  private readonly slots = new Map<string, Slot>()
  private readonly sourceText: Map<string, string>
  private readonly states = new Map<string, BlockState>()
  private readonly interactionCleanup = new Map<string, () => void>()
  private readonly focus: DocumentFocusTracker
  private readonly mutationObserver: MutationObserver
  private detachControls: (() => void) | undefined
  private destroyed = false

  constructor(
    readonly chapter: DocumentChapter,
    private readonly callbacks: DocumentReaderCallbacks = {},
    dependencies: DocumentReaderDependencies = {},
  ) {
    this.documentRef = chapter.sourceRoot.ownerDocument
    if (!chapter.sourceRoot.isConnected)
      throw new Error('The document source root must be connected.')
    this.interactionRoot = chapter.sourceRoot
    this.sourceText = new Map(chapter.snapshot.blocks.map((block) => [block.itemId, block.text]))
    const ranges = new Map<Text, number>()
    for (const block of chapter.snapshot.blocks) {
      const parts = chapter.sourceSlots.get(block.itemId)
      const nodes = parts?.map((part) => part.node)
      if (
        !nodes?.length ||
        nodes.some(
          (node) =>
            !node.isConnected ||
            !chapter.sourceRoot.contains(node) ||
            chapter.sourceRevisions.get(node) !== node.data,
        )
      ) {
        throw new Error(`Document block is not connected beneath the source root: ${block.itemId}`)
      }
      for (const part of parts!) {
        if (
          part.start < (ranges.get(part.node) ?? 0) ||
          part.start < 0 ||
          part.end > part.node.length ||
          part.end <= part.start
        )
          throw new Error('Document source slots must not overlap.')
        ranges.set(part.node, part.end)
      }
      for (const node of nodes) {
        if (!this.sources.has(node))
          this.sources.set(node, { original: node.data, expected: node.data, itemId: block.itemId })
        for (let ancestor: Node | null = node; ancestor; ancestor = ancestor.parentNode)
          this.sourceAncestors.add(ancestor)
      }
      const element = this.documentRef.createElement('span')
      element.dataset.hskifyOwned = 'true'
      element.dataset.hskifyItemId = block.itemId
      this.slots.set(block.itemId, { element, nodes })
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
    style.textContent = `:host { all: initial; position: fixed; inset: 0; width: 0; height: 0; pointer-events: none; z-index: 2147483646; }\n${LOOKUP_CSS}`
    this.lookupElement = this.documentRef.createElement('aside')
    this.lookupElement.className = 'hskify-lookup'
    this.lookupElement.hidden = true
    this.shadowRoot.append(style, this.lookupElement)
    this.focus = new DocumentFocusTracker(
      chapter.snapshot.blocks.map((block) => block.itemId),
      callbacks.onVisibleBlocksChanged ?? (() => undefined),
      dependencies.intersectionObserverFactory,
    )
    const Observer = this.documentRef.defaultView?.MutationObserver
    if (!Observer) throw new Error('MutationObserver is unavailable.')
    this.mutationObserver = new Observer(this.onMutation)
    try {
      const restoreAnchor = this.readingAnchor()
      this.documentRef.body.append(this.host)
      this.applyChineseView()
      restoreAnchor()
      for (const [id, slot] of this.slots) this.focus.observe(slot.element, id)
      this.observeSource()
      this.detachControls = attachComparisonControls(this.documentRef, this)
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
    } catch (error) {
      this.destroy()
      throw error
    }
  }

  get currentMode(): ChapterDisplayMode {
    return this.mode
  }
  get isDestroyed(): boolean {
    return this.destroyed
  }

  visibleItemIds(): string[] {
    return this.focus.visibleItemIds()
  }
  async waitForInitialFocus(signal: AbortSignal): Promise<void> {
    if (this.documentRef.hidden) return
    const cancelled = () => new DOMException('Reader cancelled.', 'AbortError')
    if (signal.aborted) throw cancelled()
    let abort!: () => void
    const aborted = new Promise<never>((_, reject) => {
      abort = () => reject(cancelled())
      signal.addEventListener('abort', abort, { once: true })
    })
    const fallback = setTimeout(() => this.focus.initializeFromRects(), 100)
    try {
      await Promise.race([this.focus.ready, aborted])
    } finally {
      clearTimeout(fallback)
      signal.removeEventListener('abort', abort)
    }
  }

  counts(): DocumentReaderCounts {
    let translated = 0,
      preserved = 0
    for (const state of this.states.values()) {
      if (state.state === 'translated') translated++
      if (state.state === 'preserved') preserved++
    }
    return { translated, preserved, pending: this.states.size - translated - preserved }
  }

  translatedBlocks(): Map<string, TranslatedText> {
    const result = new Map<string, TranslatedText>()
    for (const [id, state] of this.states)
      if (state.state === 'translated') result.set(id, state.text)
    return result
  }

  installBlock(itemId: string, text: TranslatedText): boolean {
    this.checkSource(itemId, text.sourceText)
    if (this.states.get(itemId)?.state !== 'pending') return false
    this.states.set(itemId, { state: 'translated', text })
    if (this.displayedMode === 'chinese') this.write(() => this.renderSlot(itemId))
    return true
  }

  preserveBlock(itemId: string, sourceText: string, reason?: string): boolean {
    this.checkSource(itemId, sourceText)
    if (this.states.get(itemId)?.state !== 'pending') return false
    this.states.set(itemId, { state: 'preserved', ...(reason ? { reason } : {}) })
    if (this.displayedMode === 'chinese') this.write(() => this.renderSlot(itemId))
    return true
  }

  retryBlock(itemId: string): void {
    this.checkSource(itemId, this.sourceText.get(itemId)!)
    if (this.states.get(itemId)?.state !== 'preserved') return
    this.states.set(itemId, { state: 'pending' })
    this.write(() => this.renderSlot(itemId))
    this.callbacks.onRetry?.(itemId)
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

  destroy(): void {
    if (this.destroyed) return
    // External edits queued immediately before cancellation still own their source.
    this.adoptEdits(this.mutationObserver.takeRecords())
    this.destroyed = true
    this.mutationObserver.disconnect()
    this.focus.destroy()
    this.restoreOriginalView()
    this.detachControls?.()
    this.host.remove()
  }

  private ensureActive(): void {
    this.onMutation(this.mutationObserver.takeRecords(), this.mutationObserver)
    if (this.destroyed) throw new Error('The document reader has been destroyed.')
    if (!this.chapter.sourceRoot.isConnected) {
      this.invalidate('source-detached')
      throw new Error('The document source has been detached.')
    }
  }

  private checkSource(itemId: string, text: string): void {
    this.ensureActive()
    const slot = this.slots.get(itemId)
    if (!slot) throw new Error(`Unknown document block: ${itemId}`)
    if (this.sourceText.get(itemId) !== text)
      throw new Error(`Source mismatch for document block: ${itemId}`)
    if (
      slot.nodes.some((node) => !node.isConnected || node.data !== this.sources.get(node)!.expected)
    ) {
      this.invalidate('source-mutation')
      throw new Error('The document source revision changed.')
    }
  }

  private clearInteraction(id: string): void {
    this.interactionCleanup.get(id)?.()
    this.interactionCleanup.delete(id)
  }

  private renderSlot(id: string): void {
    const slot = this.slots.get(id)!
    const state = this.states.get(id)!
    this.clearInteraction(id)
    const element = slot.element
    const first = slot.nodes[0]!
    if (!element.isConnected) first.parentNode!.insertBefore(element, first)
    for (const node of slot.nodes) {
      node.data = ''
      this.sources.get(node)!.expected = ''
    }
    element.dataset.hskifyState = state.state
    element.removeAttribute('style')
    if (state.state === 'translated') {
      installTranslatedText(element, state.text)
      element.tabIndex = 0
      const cleanup = this.callbacks.attachTranslatedText?.(element, id)
      if (cleanup) this.interactionCleanup.set(id, cleanup)
    } else if (state.state === 'preserved') {
      element.replaceChildren()
      element.removeAttribute('aria-hidden')
      element.setAttribute('role', 'status')
      element.append('Translation failed. ')
      const retry = this.documentRef.createElement('button')
      retry.type = 'button'
      retry.textContent = 'Retry'
      retry.addEventListener('click', () => this.retryBlock(id))
      element.append(retry)
      element.title = state.reason ?? 'Use Original or hold Compare to read the source.'
    } else {
      // Invisible original-sized text retains line wrapping instead of collapsing the chapter.
      element.replaceChildren(this.documentRef.createTextNode(this.sourceText.get(id)!))
      element.style.visibility = 'hidden'
      element.setAttribute('aria-hidden', 'true')
    }
  }

  private applyChineseView(): void {
    for (const id of this.slots.keys()) this.renderSlot(id)
  }

  private restoreOriginalView(): void {
    for (const [id, slot] of this.slots) {
      this.clearInteraction(id)
      slot.element.remove()
    }
    for (const [node, source] of this.sources) {
      // Never reinsert a source node removed by the site, or overwrite a site edit.
      if (this.chapter.sourceRoot.contains(node) && node.data === source.expected)
        node.data = source.original
      source.expected = node.data
    }
  }

  private observeSource(): void {
    this.mutationObserver.observe(this.documentRef.documentElement, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ['hidden', 'aria-hidden', 'contenteditable'],
    })
  }

  private write(update: () => void): void {
    this.ensureActive() // takeRecords must precede disconnect.
    const restoreAnchor = this.readingAnchor()
    this.mutationObserver.disconnect()
    try {
      update()
      restoreAnchor()
    } finally {
      if (!this.destroyed) this.observeSource()
    }
  }

  private display(mode: ChapterDisplayMode): void {
    if (this.displayedMode === mode) return
    this.write(() => {
      if (mode === 'original') this.restoreOriginalView()
      else this.applyChineseView()
      this.displayedMode = mode
    })
  }

  private readingAnchor(): () => void {
    const win = this.documentRef.defaultView
    if (!win) return () => undefined
    const focused = this.focus.visibleItemIds()[0]
    let anchor = focused ? this.chapter.sourceElements.get(focused) : undefined
    if (!anchor) {
      const hit = this.documentRef.elementFromPoint?.(
        Math.max(1, win.innerWidth / 2),
        Math.max(1, win.innerHeight / 3),
      )
      if (hit)
        anchor = [...this.chapter.sourceElements.values()].find(
          (element) => element === hit || element.contains(hit),
        )
    }
    if (!anchor) return () => undefined
    const top = anchor.getBoundingClientRect().top
    const scrollers: HTMLElement[] = []
    for (
      let parent = anchor.parentElement;
      parent && parent !== this.documentRef.body;
      parent = parent.parentElement
    ) {
      if (
        parent.scrollHeight > parent.clientHeight &&
        /auto|scroll/u.test(win.getComputedStyle(parent).overflowY)
      )
        scrollers.push(parent)
    }
    return () => {
      if (!anchor?.isConnected) return
      for (const scroller of scrollers) {
        const delta = anchor.getBoundingClientRect().top - top
        if (Math.abs(delta) > 0.5) scroller.scrollTop += delta
      }
      const delta = anchor.getBoundingClientRect().top - top
      if (Math.abs(delta) > 0.5) win.scrollBy(0, delta)
    }
  }

  private owned(node: Node): boolean {
    const element = node.nodeType === 1 ? (node as Element) : node.parentElement
    return Boolean(element?.closest('[data-hskify-owned]'))
  }

  private adoptEdits(records: readonly MutationRecord[]): boolean {
    let changed = false
    for (const record of records) {
      if (this.owned(record.target)) continue
      if (record.type === 'characterData') {
        const source = this.sources.get(record.target as Text)
        if (source && (record.target as Text).data !== source.expected) {
          source.original = (record.target as Text).data
          source.expected = source.original
          changed = true
        }
      } else if (record.type === 'childList' && this.chapter.sourceRoot.contains(record.target)) {
        for (const node of record.removedNodes) {
          if (this.sourceAncestors.has(node)) changed = true
        }
        for (const node of record.addedNodes) {
          if (!this.owned(node) && node.textContent?.trim()) changed = true
        }
      } else if (record.type === 'attributes' && this.chapter.sourceRoot.contains(record.target))
        changed = true
    }
    return changed
  }

  private invalidate(reason: 'source-mutation' | 'source-detached'): void {
    this.destroy()
    this.callbacks.onInvalidated?.(reason)
  }

  private readonly onMutation: MutationCallback = (records) => {
    if (this.destroyed) return
    if (!this.chapter.sourceRoot.isConnected) {
      this.invalidate('source-detached')
      return
    }
    if (this.adoptEdits(records)) this.invalidate('source-mutation')
  }
}

export function mountDocumentReader(
  chapter: DocumentChapter,
  callbacks: DocumentReaderCallbacks = {},
  dependencies: DocumentReaderDependencies = {},
): DocumentReader {
  return new DocumentReader(chapter, callbacks, dependencies)
}
