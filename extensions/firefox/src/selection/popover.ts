import type { LookupRequest, LookupResult } from '../contracts/browser'

export type LookupCallback = (request: LookupRequest) => Promise<LookupResult>

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

  constructor(
    private readonly root: ShadowRoot,
    private readonly popover: HTMLElement,
    private readonly lookup: LookupCallback,
  ) {
    root.addEventListener('mouseup', this.onSelectionComplete)
    root.addEventListener('keyup', this.onKeyUp)
    root.addEventListener('copy', this.onCopy)
    root.addEventListener('pointerdown', this.onPointerDown)
    root.addEventListener('click', this.onClick, true)
  }

  register(element: HTMLElement, jobId: string, regionId: string): void {
    this.regions.set(element, { element, jobId, regionId })
    element.addEventListener('keydown', this.onRegionKeyDown)
  }

  destroy(): void {
    this.destroyed = true
    this.requestRevision += 1
    this.root.removeEventListener('mouseup', this.onSelectionComplete)
    this.root.removeEventListener('keyup', this.onKeyUp)
    this.root.removeEventListener('copy', this.onCopy)
    this.root.removeEventListener('pointerdown', this.onPointerDown)
    this.root.removeEventListener('click', this.onClick, true)
    for (const region of this.regions.values()) {
      region.element.removeEventListener('keydown', this.onRegionKeyDown)
    }
    this.regions.clear()
  }

  private selectedRegion(): {
    region: SelectionRegion
    selection: Selection
    range: Range
  } | null {
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null
    const anchor = nodeElement(selection.anchorNode)
    const focus = nodeElement(selection.focusNode)
    for (const region of this.regions.values()) {
      if (
        (anchor && region.element.contains(anchor)) ||
        (focus && region.element.contains(focus))
      ) {
        return { region, selection, range: selection.getRangeAt(0) }
      }
    }
    return null
  }

  private readonly onPointerDown = (event: Event): void => {
    const target = event.target
    if (target instanceof Node && !this.popover.contains(target)) {
      this.popover.hidden = true
    }
  }

  private readonly onClick = (event: Event): void => {
    const target = nodeElement(event.target instanceof Node ? event.target : null)
    if (!target) return
    const clickedRegion = [...this.regions.values()].find((region) =>
      region.element.contains(target),
    )
    const selected = this.selectedRegion()
    if (!clickedRegion || !selected || clickedRegion !== selected.region) return
    event.preventDefault()
    event.stopPropagation()
  }

  private readonly onSelectionComplete = (): void => {
    queueMicrotask(() => void this.showSelection())
  }

  private readonly onKeyUp = (): void => {
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
    clipboardData.setData('text/plain', selected.selection.toString())
    event.preventDefault()
  }

  private async showSelection(): Promise<void> {
    const selected = this.selectedRegion()
    if (!selected || this.destroyed) return
    const selectedText = selected.selection.toString().trim()
    if (!selectedText || [...selectedText].length > 256) return
    const revision = ++this.requestRevision
    this.popover.hidden = false
    this.popover.replaceChildren()
    const loading = document.createElement('span')
    loading.textContent = 'Looking up…'
    this.popover.append(loading)
    const rangeRect = selected.range.getBoundingClientRect()
    const hostRect = this.root.host.getBoundingClientRect()
    this.popover.style.left = `${Math.max(4, rangeRect.left - hostRect.left)}px`
    this.popover.style.top = `${Math.max(4, rangeRect.bottom - hostRect.top + 6)}px`
    try {
      const result = await this.lookup({
        selectedText,
        jobId: selected.region.jobId,
        regionId: selected.region.regionId,
      })
      if (revision !== this.requestRevision || this.destroyed) return
      this.renderResult(result)
    } catch {
      if (revision !== this.requestRevision || this.destroyed) return
      this.popover.replaceChildren()
      const message = document.createElement('span')
      message.textContent = 'Dictionary lookup unavailable.'
      this.popover.append(message)
    }
  }

  private renderResult(result: LookupResult): void {
    this.popover.replaceChildren()
    const heading = document.createElement('strong')
    heading.textContent = result.selectedText
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
      const faithful = document.createElement('span')
      faithful.textContent = result.region.faithfulChinese
      const source = document.createElement('span')
      source.textContent = result.region.sourceEnglish
      context.append(faithful, source)
      this.popover.append(context)
    }
  }
}
