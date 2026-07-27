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

function nodeElement(node: Node | null): Element | null {
  if (!node) return null
  return node instanceof Element ? node : node.parentElement
}

export class SelectionController {
  private readonly regions = new Map<HTMLElement, SelectionRegion>()
  private requestRevision = 0
  private destroyed = false
  private speechOwner: SpeechStateListener | undefined
  private popoverPointerActive = false

  constructor(
    private readonly root: ShadowRoot,
    private readonly popover: HTMLElement,
    private readonly lookup: LookupCallback,
    private readonly forwardPrimaryClick?: PrimaryClickForwarder,
    private readonly speaker: TextSpeaker = new MandarinSpeaker(),
  ) {
    root.addEventListener('mouseup', this.onSelectionComplete)
    root.addEventListener('keyup', this.onKeyUp)
    root.addEventListener('copy', this.onCopy)
    root.addEventListener('click', this.onClick, true)
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
    this.requestRevision += 1
    this.popover.hidden = true
    this.popover.replaceChildren()
    if (this.speechOwner) this.speaker.stop(this.speechOwner)
    this.speechOwner = undefined
  }

  private selectedRegion(): {
    region: SelectionRegion
    selection: Selection
    range: Range
  } | null {
    const selection = window.getSelection()
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
        this.popoverPointerActive ||
        this.selectionTouches(this.popover)
      ) {
        return
      }
      if (!this.selectedRegion()) this.dismiss()
    })
  }

  private readonly onClick = (event: Event): void => {
    const target = nodeElement(event.target instanceof Node ? event.target : null)
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
    if (event instanceof MouseEvent && event.button === 0) {
      this.forwardPrimaryClick?.(event)
    }
  }

  private readonly onSelectionComplete = (event: Event): void => {
    const target = nodeElement(event.target instanceof Node ? event.target : null)
    if (
      !target ||
      ![...this.regions.values()].some((region) => region.element.contains(target))
    ) {
      return
    }
    queueMicrotask(() => void this.showSelection())
  }

  private readonly onKeyUp = (event: Event): void => {
    const target = nodeElement(event.target instanceof Node ? event.target : null)
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
      !(target instanceof HTMLElement) ||
      event.key.toLowerCase() !== 'a' ||
      (!event.ctrlKey && !event.metaKey)
    ) {
      return
    }
    event.preventDefault()
    const range = document.createRange()
    range.selectNodeContents(target)
    const selection = window.getSelection()
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
    const revision = ++this.requestRevision
    this.speaker.stop()
    this.speechOwner = undefined
    this.popover.hidden = false
    this.popover.replaceChildren()
    const heading = this.createSpeechHeading(selectedText)
    const loading = document.createElement('span')
    loading.textContent = 'Looking up…'
    this.popover.append(heading, loading)
    this.positionPopover(selected.region.element, selected.range)
    try {
      const result = await this.lookup({
        selectedText,
        jobId: selected.region.jobId,
        regionId: selected.region.regionId,
      })
      if (revision !== this.requestRevision || this.destroyed) return
      this.renderResult(result, selectedText)
      this.positionPopover(selected.region.element, selected.range)
    } catch {
      if (revision !== this.requestRevision || this.destroyed) return
      const heading = this.popover.querySelector<HTMLElement>('.hmt-lookup-heading')
      this.popover.replaceChildren()
      if (heading) this.popover.append(heading)
      const message = document.createElement('span')
      message.textContent = 'Dictionary lookup unavailable.'
      this.popover.append(message)
      this.positionPopover(selected.region.element, selected.range)
    }
  }

  private selectionTouches(element: HTMLElement): boolean {
    const selection = window.getSelection()
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
    const selectedRect = range.getBoundingClientRect()
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
    const viewportRight = Math.min(hostRect.right, window.innerWidth - edge)
    const maximumLeft = Math.max(edge, viewportRight - hostRect.left - popoverWidth)
    const left = Math.min(
      maximumLeft,
      Math.max(edge, obstruction.left - hostRect.left),
    )
    const below = obstruction.bottom - hostRect.top + gap
    const belowSpace = Math.max(0, window.innerHeight - edge - obstruction.bottom - gap)
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
    const heading = document.createElement('div')
    heading.className = 'hmt-lookup-heading'
    const selectedText = document.createElement('strong')
    selectedText.textContent = spokenText
    const speak = document.createElement('button')
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
    const heading =
      this.popover.querySelector<HTMLElement>('.hmt-lookup-heading') ??
      this.createSpeechHeading(selectedText)
    this.popover.replaceChildren()
    this.popover.append(heading)
    for (const token of result.tokens) {
      const entry = document.createElement('div')
      entry.className = 'hmt-lookup-entry'
      const word = document.createElement('b')
      word.textContent = token.simplified
      const detail = document.createElement('span')
      const hsk = token.properName
        ? 'Proper name · outside HSK list'
        : token.hskLevel
          ? `HSK ${token.hskLevel}`
          : 'Outside HSK list'
      detail.textContent = `${token.pinyin} · ${hsk}`
      const definitions = document.createElement('span')
      definitions.textContent = token.definitions.join('; ')
      entry.append(word, detail, definitions)
      this.popover.append(entry)
    }
    if (result.region) {
      const context = document.createElement('div')
      context.className = 'hmt-lookup-context'
      const base = document.createElement('span')
      base.textContent = result.region.baseChinese
      const source = document.createElement('span')
      source.textContent = result.region.sourceEnglish
      context.append(base, source)
      this.popover.append(context)
    }
  }
}
