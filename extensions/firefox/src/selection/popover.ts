import type { LookupRequest, LookupResult } from '../contracts/browser'
import {
  MandarinSpeaker,
  type SpeechState,
  type SpeechStateListener,
  type TextSpeaker,
} from './speech'

export type LookupCallback = (request: LookupRequest) => Promise<LookupResult>
export type PrimaryClickForwarder = (event: MouseEvent) => void

type SelectionRegion = {
  element: HTMLElement
  jobId: string
  regionId: string
}

export type HoverTextHit = {
  characterOffset: number
  range: Range
}

export type HoverHitTester = (
  element: HTMLElement,
  clientX: number,
  clientY: number,
) => HoverTextHit | null

const HOVER_LOOKUP_DELAY_MS = 120
const HOVER_DISMISS_DELAY_MS = 140

function nodeElement(node: Node | null): Element | null {
  if (!node) return null
  return node.nodeType === 1 ? (node as Element) : node.parentElement
}

function eventNode(value: EventTarget | null): Node | null {
  return value && typeof value === 'object' && 'nodeType' in value
    ? (value as Node)
    : null
}

function textNodes(element: HTMLElement): Text[] {
  const nodes: Text[] = []
  const documentRef = element.ownerDocument
  const showText = documentRef.defaultView?.NodeFilter.SHOW_TEXT ?? 4
  const walker = documentRef.createTreeWalker(element, showText)
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeType === 3 && (node as Text).data.length > 0) nodes.push(node as Text)
  }
  return nodes
}

export function textRange(
  element: HTMLElement,
  characterOffset: number,
  characterLength = 1,
): Range | null {
  if (characterOffset < 0 || characterLength < 1) return null
  const documentRef = element.ownerDocument
  const range = documentRef.createRange()
  let globalOffset = 0
  let start: { node: Text; offset: number } | undefined
  let end: { node: Text; offset: number } | undefined
  const targetEnd = characterOffset + characterLength

  for (const node of textNodes(element)) {
    let localUtf16 = 0
    for (const character of node.data) {
      const nextGlobal = globalOffset + 1
      const nextUtf16 = localUtf16 + character.length
      if (globalOffset === characterOffset) {
        start = { node, offset: localUtf16 }
      }
      if (nextGlobal === targetEnd) {
        end = { node, offset: nextUtf16 }
      }
      globalOffset = nextGlobal
      localUtf16 = nextUtf16
    }
  }
  if (!start || !end) return null
  range.setStart(start.node, start.offset)
  range.setEnd(end.node, end.offset)
  return range
}

/**
 * Resolves the range returned by a hover lookup without assuming that the
 * dictionary always returns the single character under the pointer. The
 * daemon may return the longest phrase beginning at the pointer, or a suffix
 * when the pointer is inside a longer phrase.
 */
export function hoverResultRange(
  element: HTMLElement,
  characterOffset: number,
  selectedText: string,
): Range | null {
  const requested = [...selectedText]
  if (requested.length === 0) return null
  const direct = textRange(element, characterOffset, requested.length)
  if (direct && (direct.cloneContents().textContent ?? '') === selectedText) {
    return direct
  }
  // A malformed or stale dictionary response must not move the popover to a
  // different occurrence of the same word. Keep the explanation anchored at
  // the hovered character and show only the character that was actually hit.
  return textRange(element, characterOffset)
}

export const characterRangeAtPoint: HoverHitTester = (
  element,
  clientX,
  clientY,
) => {
  const characterCount = [...(element.textContent ?? '')].length
  for (let characterOffset = 0; characterOffset < characterCount; characterOffset += 1) {
    const range = textRange(element, characterOffset)
    if (!range) continue
    const rects = [...range.getClientRects()]
    if (
      rects.some(
        (rect) =>
          clientX >= rect.left - 1 &&
          clientX <= rect.right + 1 &&
          clientY >= rect.top - 1 &&
          clientY <= rect.bottom + 1,
      )
    ) {
      return { characterOffset, range }
    }
  }
  return null
}

export class ExplanationController {
  private readonly regions = new Map<HTMLElement, SelectionRegion>()
  private requestRevision = 0
  private destroyed = false
  private speechOwner: SpeechStateListener | undefined
  private popoverPointerActive = false
  private activeInteraction: 'selection' | 'hover' | undefined
  private activeHoverKey: string | undefined
  private pendingHoverKey: string | undefined
  private hoverLookupTimer: ReturnType<typeof setTimeout> | undefined
  private hoverDismissTimer: ReturnType<typeof setTimeout> | undefined

  constructor(
    private readonly root: ShadowRoot,
    private readonly popover: HTMLElement,
    private readonly lookup: LookupCallback,
    private readonly forwardPrimaryClick?: PrimaryClickForwarder,
    private readonly speaker: TextSpeaker = new MandarinSpeaker(),
    private readonly hitTest: HoverHitTester = characterRangeAtPoint,
  ) {
    root.addEventListener('mouseup', this.onSelectionComplete)
    root.addEventListener('keyup', this.onKeyUp)
    root.addEventListener('copy', this.onCopy)
    root.addEventListener('click', this.onClick, true)
    root.addEventListener('pointermove', this.onPointerMove)
    root.addEventListener('pointerout', this.onPointerOut)
    popover.addEventListener('pointerenter', this.onPopoverPointerEnter)
    popover.addEventListener('pointerleave', this.onPopoverPointerLeave)
    root.host.ownerDocument.defaultView?.addEventListener('scroll', this.onViewportChange, true)
    root.host.ownerDocument.defaultView?.addEventListener('resize', this.onViewportChange)
    root.host.ownerDocument.addEventListener(
      'pointerdown',
      this.onDocumentPointerDown,
      true,
    )
    root.host.ownerDocument.addEventListener('pointerup', this.onDocumentPointerUp, true)
    root.host.ownerDocument.addEventListener(
      'selectionchange',
      this.onDocumentSelectionChange,
    )
  }

  register(element: HTMLElement, jobId: string, regionId: string): void {
    this.regions.set(element, { element, jobId, regionId })
    element.addEventListener('keydown', this.onRegionKeyDown)
  }

  unregister(element: HTMLElement): void {
    const region = this.regions.get(element)
    if (!region) return
    region.element.removeEventListener('keydown', this.onRegionKeyDown)
    this.regions.delete(element)
    this.dismiss()
  }

  destroy(): void {
    this.destroyed = true
    this.dismiss()
    this.root.removeEventListener('mouseup', this.onSelectionComplete)
    this.root.removeEventListener('keyup', this.onKeyUp)
    this.root.removeEventListener('copy', this.onCopy)
    this.root.removeEventListener('click', this.onClick, true)
    this.root.removeEventListener('pointermove', this.onPointerMove)
    this.root.removeEventListener('pointerout', this.onPointerOut)
    this.popover.removeEventListener('pointerenter', this.onPopoverPointerEnter)
    this.popover.removeEventListener('pointerleave', this.onPopoverPointerLeave)
    this.root.host.ownerDocument.defaultView?.removeEventListener(
      'scroll',
      this.onViewportChange,
      true,
    )
    this.root.host.ownerDocument.defaultView?.removeEventListener(
      'resize',
      this.onViewportChange,
    )
    this.root.host.ownerDocument.removeEventListener(
      'pointerdown',
      this.onDocumentPointerDown,
      true,
    )
    this.root.host.ownerDocument.removeEventListener(
      'pointerup',
      this.onDocumentPointerUp,
      true,
    )
    this.root.host.ownerDocument.removeEventListener(
      'selectionchange',
      this.onDocumentSelectionChange,
    )
    for (const region of this.regions.values()) {
      region.element.removeEventListener('keydown', this.onRegionKeyDown)
    }
    this.regions.clear()
  }

  dismiss(): void {
    this.clearHoverTimers()
    this.requestRevision += 1
    this.popover.hidden = true
    this.popover.replaceChildren()
    if (this.speechOwner) this.speaker.stop(this.speechOwner)
    this.speechOwner = undefined
    this.activeInteraction = undefined
    this.activeHoverKey = undefined
    this.pendingHoverKey = undefined
  }

  private clearHoverTimers(): void {
    if (this.hoverLookupTimer !== undefined) clearTimeout(this.hoverLookupTimer)
    if (this.hoverDismissTimer !== undefined) clearTimeout(this.hoverDismissTimer)
    this.hoverLookupTimer = undefined
    this.hoverDismissTimer = undefined
  }

  private cancelHoverDismiss(): void {
    if (this.hoverDismissTimer !== undefined) clearTimeout(this.hoverDismissTimer)
    this.hoverDismissTimer = undefined
  }

  private scheduleHoverDismiss(): void {
    this.cancelHoverDismiss()
    this.hoverDismissTimer = setTimeout(() => this.dismiss(), HOVER_DISMISS_DELAY_MS)
  }

  private selectedRegion(): {
    region: SelectionRegion
    selection: Selection
    range: Range
  } | null {
    const documentRef = [...this.regions.keys()][0]?.ownerDocument
    const selection = documentRef?.defaultView?.getSelection() ?? documentRef?.getSelection()
    if (!selection || selection.isCollapsed || selection.rangeCount !== 1) return null
    const range = selection.getRangeAt(0)
    for (const region of this.regions.values()) {
      if (
        region.element.contains(range.startContainer) &&
        region.element.contains(range.endContainer)
      ) {
        return { region, selection, range }
      }
    }
    return null
  }

  private readonly onDocumentPointerDown = (event: Event): void => {
    if (event.composedPath().includes(this.popover)) {
      this.popoverPointerActive = true
      this.cancelHoverDismiss()
      return
    }
    this.popoverPointerActive = false
    this.dismiss()
  }

  private readonly onDocumentPointerUp = (): void => {
    queueMicrotask(() => {
      this.popoverPointerActive = false
    })
  }

  private readonly onViewportChange = (): void => {
    this.dismiss()
  }

  private readonly onDocumentSelectionChange = (): void => {
    queueMicrotask(() => {
      if (
        this.destroyed ||
        this.activeInteraction !== 'selection' ||
        this.popoverPointerActive ||
        this.selectionTouches(this.popover)
      ) {
        return
      }
      if (!this.selectedRegion()) this.dismiss()
    })
  }

  private readonly onClick = (event: Event): void => {
    const target = nodeElement(eventNode(event.target))
    if (!target) return
    const clickedRegion = [...this.regions.values()].find((region) =>
      region.element.contains(target),
    )
    if (!clickedRegion) return
    const selected = this.selectedRegion()
    if (selected && clickedRegion === selected.region) {
      event.preventDefault()
      event.stopPropagation()
      return
    }
    if (event.type === 'click' && (event as MouseEvent).button === 0) {
      this.forwardPrimaryClick?.(event as MouseEvent)
    }
  }

  private readonly onSelectionComplete = (event: Event): void => {
    const target = nodeElement(eventNode(event.target))
    if (
      !target ||
      ![...this.regions.values()].some((region) => region.element.contains(target))
    ) {
      return
    }
    queueMicrotask(() => {
      if (this.selectedRegion()) void this.showSelection()
    })
  }

  private readonly onKeyUp = (event: Event): void => {
    const target = nodeElement(eventNode(event.target))
    if (
      !target ||
      ![...this.regions.values()].some((region) => region.element.contains(target))
    ) {
      return
    }
    queueMicrotask(() => void this.showSelection())
  }

  private readonly onRegionKeyDown = (event: KeyboardEvent): void => {
    const target = event.currentTarget
    if (
      !target ||
      event.key.toLowerCase() !== 'a' ||
      (!event.ctrlKey && !event.metaKey)
    ) {
      return
    }
    event.preventDefault()
    const elementTarget = target as HTMLElement
    const documentRef = elementTarget.ownerDocument
    const range = documentRef.createRange()
    range.selectNodeContents(elementTarget)
    const selection = documentRef.defaultView?.getSelection() ?? documentRef.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)
    void this.showSelection()
  }

  private readonly onCopy = (event: Event): void => {
    const clipboardData = (event as ClipboardEvent).clipboardData
    if (!clipboardData) return
    const selected = this.selectedRegion()
    if (!selected) return
    clipboardData.setData('text/plain', this.selectedText(selected))
    event.preventDefault()
  }

  private readonly onPopoverPointerEnter = (): void => {
    this.cancelHoverDismiss()
  }

  private readonly onPopoverPointerLeave = (): void => {
    if (this.activeInteraction === 'hover') this.scheduleHoverDismiss()
  }

  private readonly onPointerOut = (event: Event): void => {
    if (this.activeInteraction !== 'hover' && !this.pendingHoverKey) return
    const related = nodeElement(
      eventNode((event as PointerEvent).relatedTarget),
    )
    if (related && (this.popover.contains(related) || [...this.regions.keys()].some(
      (element) => element.contains(related),
    ))) {
      return
    }
    this.scheduleHoverDismiss()
  }

  private readonly onPointerMove = (event: Event): void => {
    if (this.destroyed || event.type !== 'pointermove') return
    const pointerEvent = event as PointerEvent
    if (
      pointerEvent.pointerType !== undefined &&
      pointerEvent.pointerType !== 'mouse' &&
      pointerEvent.pointerType !== 'pen'
    ) {
      return
    }
    // A previous selection can remain in the browser after its popover was
    // dismissed. It must not disable direct hover lookup; only an actively
    // displayed selection explanation temporarily owns the pointer.
    if (this.activeInteraction === 'selection' && this.selectedRegion()) return
    const target = nodeElement(eventNode(event.target))
    const region = target
      ? [...this.regions.values()].find((candidate) =>
          candidate.element.contains(target),
        )
      : undefined
    if (!region) return
    this.cancelHoverDismiss()
    const hit = this.hitTest(region.element, pointerEvent.clientX, pointerEvent.clientY)
    if (!hit) {
      this.scheduleHoverDismiss()
      return
    }
    const key = `${region.jobId}\0${region.regionId}\0${hit.characterOffset}`
    if (key === this.activeHoverKey || key === this.pendingHoverKey) return
    if (this.hoverLookupTimer !== undefined) clearTimeout(this.hoverLookupTimer)
    this.pendingHoverKey = key
    this.hoverLookupTimer = setTimeout(() => {
      this.hoverLookupTimer = undefined
      if (this.pendingHoverKey !== key || this.destroyed) return
      void this.showHover(region, hit, key)
    }, HOVER_LOOKUP_DELAY_MS)
  }

  private async showSelection(): Promise<void> {
    const selected = this.selectedRegion()
    if (!selected || this.destroyed) {
      if (!this.destroyed) this.dismiss()
      return
    }
    const selectedText = this.selectedText(selected).trim()
    if (!selectedText || [...selectedText].length > 256) {
      this.dismiss()
      return
    }
    await this.showLookup(
      selected.region,
      {
        interaction: 'selection',
        selectedText,
        jobId: selected.region.jobId,
        regionId: selected.region.regionId,
      },
      selectedText,
      selected.range,
      'selection',
    )
  }

  private async showHover(
    region: SelectionRegion,
    hit: HoverTextHit,
    key: string,
  ): Promise<void> {
    this.pendingHoverKey = undefined
    this.activeHoverKey = key
    const character = hit.range.cloneContents().textContent ?? ''
    await this.showLookup(
      region,
      {
        interaction: 'hover',
        characterOffset: hit.characterOffset,
        jobId: region.jobId,
        regionId: region.regionId,
      },
      character,
      hit.range,
      'hover',
      hit.characterOffset,
    )
  }

  private async showLookup(
    region: SelectionRegion,
    request: LookupRequest,
    displayText: string,
    range: Range,
    interaction: 'selection' | 'hover',
    characterOffset?: number,
  ): Promise<void> {
    const revision = ++this.requestRevision
    this.activeInteraction = interaction
    this.speaker.stop()
    this.speechOwner = undefined
    this.popover.hidden = false
    this.popover.replaceChildren()
    const heading = this.createSpeechHeading(displayText)
    const documentRef = region.element.ownerDocument
    const loading = documentRef.createElement('span')
    loading.textContent = 'Looking up…'
    this.popover.append(heading, loading)
    this.positionPopover(region.element, range)
    try {
      const result = await this.lookup(request)
      if (revision !== this.requestRevision || this.destroyed) return
      if (result.tokens.length === 0 && !result.region) {
        this.dismiss()
        return
      }
      this.renderResult(result, result.selectedText)
      const resolvedRange =
        interaction === 'hover' && characterOffset !== undefined
          ? hoverResultRange(region.element, characterOffset, result.selectedText) ?? range
          : range
      this.positionPopover(region.element, resolvedRange)
    } catch {
      if (revision !== this.requestRevision || this.destroyed) return
      const heading = this.popover.querySelector<HTMLElement>('.hmt-lookup-heading')
      this.popover.replaceChildren()
      if (heading) this.popover.append(heading)
      const message = documentRef.createElement('span')
      message.textContent = 'Dictionary lookup unavailable.'
      this.popover.append(message)
      this.positionPopover(region.element, range)
    }
  }

  private selectionTouches(element: HTMLElement): boolean {
    const documentRef = element.ownerDocument
    const selection = documentRef.defaultView?.getSelection() ?? documentRef.getSelection()
    if (!selection || selection.rangeCount === 0) return false
    const anchor = nodeElement(selection.anchorNode)
    const focus = nodeElement(selection.focusNode)
    return Boolean(
      (anchor && element.contains(anchor)) || (focus && element.contains(focus)),
    )
  }

  private positionPopover(region: HTMLElement, range: Range): void {
    const gap = 8
    const edge = 4
    const hostRect = this.root.host.getBoundingClientRect()
    const regionRect = region.getBoundingClientRect()
    const selectedRect =
      typeof range.getBoundingClientRect === 'function'
        ? range.getBoundingClientRect()
        : regionRect
    const hasSelectedRect = selectedRect.width > 0 && selectedRect.height > 0
    const obstruction = hasSelectedRect
      ? {
          left: Math.min(regionRect.left, selectedRect.left),
          top: Math.min(regionRect.top, selectedRect.top),
          right: Math.max(regionRect.right, selectedRect.right),
          bottom: Math.max(regionRect.bottom, selectedRect.bottom),
        }
      : regionRect
    const popoverRect = this.popover.getBoundingClientRect()
    const popoverWidth = popoverRect.width || this.popover.offsetWidth
    const popoverHeight = popoverRect.height || this.popover.offsetHeight
    const view = region.ownerDocument.defaultView
    const viewportWidth = view?.innerWidth ?? hostRect.right
    const viewportHeight = view?.innerHeight ?? hostRect.bottom
    const viewportRight = Math.min(hostRect.right, viewportWidth - edge)
    const maximumLeft = Math.max(edge, viewportRight - hostRect.left - popoverWidth)
    const left = Math.min(
      maximumLeft,
      Math.max(edge, obstruction.left - hostRect.left),
    )
    const below = obstruction.bottom - hostRect.top + gap
    const belowSpace = Math.max(0, viewportHeight - edge - obstruction.bottom - gap)
    const aboveSpace = Math.max(0, obstruction.top - edge - gap)
    const placeBelow = popoverHeight === 0 || belowSpace >= popoverHeight || belowSpace >= aboveSpace
    const availableHeight = Math.max(1, placeBelow ? belowSpace : aboveSpace)
    const renderedHeight = Math.min(popoverHeight || availableHeight, availableHeight)
    const top = placeBelow
      ? below
      : obstruction.top - hostRect.top - gap - renderedHeight

    this.popover.style.left = `${left}px`
    this.popover.style.maxHeight = `${availableHeight}px`
    // Treat both the translated region and the full selected range as the
    // obstruction. If the panel cannot fit at its natural height, it scrolls
    // in the larger outside space instead of covering selected lettering.
    this.popover.style.top = `${Math.max(edge - hostRect.top, top)}px`
  }

  private selectedText(selected: {
    selection: Selection
    range: Range
  }): string {
    // `textContent` joins the renderer's visual line spans without adding
    // layout whitespace, preserving the exact Chinese source text.
    return selected.range.cloneContents().textContent ?? selected.selection.toString()
  }

  private createSpeechHeading(spokenText: string): HTMLElement {
    const documentRef = this.popover.ownerDocument
    const heading = documentRef.createElement('div')
    heading.className = 'hmt-lookup-heading'
    const selectedText = documentRef.createElement('strong')
    selectedText.textContent = spokenText
    const speak = documentRef.createElement('button')
    speak.type = 'button'
    speak.className = 'hmt-speak'
    const available = this.speaker.isAvailable()
    speak.textContent = available ? 'Listen' : 'Voice unavailable'
    speak.disabled = !available
    speak.setAttribute('aria-pressed', 'false')
    speak.setAttribute('aria-live', 'polite')
    speak.setAttribute(
      'aria-label',
      available ? 'Listen to Mandarin pronunciation' : 'Mandarin voice unavailable',
    )
    speak.title = speak.disabled
      ? 'Mandarin speech is not available in this Firefox profile.'
      : 'Play Mandarin pronunciation using the best available local voice.'
    const updateSpeechState: SpeechStateListener = (state, voice): void => {
      if (!speak.isConnected) return
      if (voice) {
        speak.dataset.hmtVoiceName = voice.name
        speak.dataset.hmtVoiceLang = voice.lang
        speak.dataset.hmtVoiceLocalService = String(voice.localService)
      } else if (state === 'unavailable' || state === 'error') {
        delete speak.dataset.hmtVoiceName
        delete speak.dataset.hmtVoiceLang
        delete speak.dataset.hmtVoiceLocalService
      }
      const active = state === 'loading' || state === 'speaking'
      const runtimeAvailable = this.speaker.isAvailable()
      speak.disabled = state === 'unavailable' && !runtimeAvailable
      speak.textContent =
        state === 'loading'
          ? 'Loading…'
          : state === 'speaking'
            ? 'Stop'
            : state === 'unavailable'
              ? runtimeAvailable
                ? 'Retry voice'
                : 'Voice unavailable'
              : state === 'error'
                ? 'Try again'
                : 'Listen'
      speak.setAttribute('aria-pressed', String(active))
      speak.setAttribute('aria-busy', String(state === 'loading'))
      speak.title =
        state === 'unavailable'
          ? 'Install or enable a local Simplified Chinese voice, then restart Firefox.'
          : state === 'error'
            ? 'Mandarin pronunciation could not play. Click to try again.'
            : 'Play Mandarin pronunciation using the best available local voice.'
      speak.setAttribute(
        'aria-label',
        state === 'loading'
          ? 'Loading Mandarin voice; activate to cancel'
          : state === 'speaking'
            ? 'Stop Mandarin pronunciation'
            : state === 'unavailable'
              ? runtimeAvailable
                ? 'Retry voice for Mandarin pronunciation'
                : 'Mandarin voice unavailable'
              : state === 'error'
                ? 'Try again: Mandarin pronunciation'
                : 'Listen to Mandarin pronunciation',
      )
    }
    this.speechOwner = updateSpeechState
    speak.addEventListener('click', () => {
      this.speaker.toggle(spokenText, updateSpeechState)
    })
    heading.append(selectedText, speak)
    return heading
  }

  private renderResult(result: LookupResult, selectedText: string): void {
    if (this.speechOwner) this.speaker.stop(this.speechOwner)
    const heading = this.createSpeechHeading(selectedText)
    this.popover.replaceChildren()
    this.popover.append(heading)
    const documentRef = this.popover.ownerDocument
    for (const token of result.tokens) {
      const entry = documentRef.createElement('div')
      entry.className = 'hmt-lookup-entry'
      const word = documentRef.createElement('b')
      word.textContent = token.simplified
      const detail = documentRef.createElement('span')
      const hsk = token.properName
        ? 'Proper name · outside HSK list'
        : token.hskLevel
          ? `HSK ${token.hskLevel}`
          : 'Outside HSK list'
      detail.textContent = `${token.pinyin} · ${hsk}`
      const definitions = documentRef.createElement('span')
      definitions.textContent = token.definitions.join('; ')
      entry.append(word, detail, definitions)
      this.popover.append(entry)
    }
    if (result.region) {
      const context = documentRef.createElement('div')
      context.className = 'hmt-lookup-context'
      const base = documentRef.createElement('span')
      base.textContent =
        result.tokens.length === 0
          ? 'Original text kept; no reliable translation was available.'
          : result.region.baseChinese
      const source = documentRef.createElement('span')
      source.textContent = result.region.sourceEnglish
      context.append(base, source)
      this.popover.append(context)
    }
  }
}
