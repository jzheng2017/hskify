import { eligibleStoryElement } from './extraction'

/** Explicit source selection for short or ambiguous chapters. Esc always restores the page. */
export function selectStoryRegion(
  live: Document = document,
  signal?: AbortSignal,
): Promise<HTMLElement | undefined> {
  return new Promise((resolve) => {
    const panel = live.createElement('div'),
      outline = live.createElement('div')
    panel.dataset.hskifyOwned = outline.dataset.hskifyOwned = 'true'
    panel.style.cssText =
      'position:fixed;bottom:24px;left:24px;z-index:2147483647;background:#fff;color:#111;padding:16px;border:1px solid #777;border-radius:8px;font:16px sans-serif;box-shadow:0 4px 20px #0004'
    outline.style.cssText =
      'position:fixed;pointer-events:none;border:3px solid #2673d9;z-index:2147483646;box-sizing:border-box'
    const label = live.createElement('span'),
      confirm = live.createElement('button'),
      cancel = live.createElement('button')
    label.textContent = 'Point at the story. ↑ selects its parent. '
    confirm.textContent = 'Translate this region'
    confirm.disabled = true
    cancel.textContent = 'Cancel'
    panel.append(label, confirm, cancel)
    live.body.append(panel, outline)
    let selected: HTMLElement | undefined
    const update = (element?: HTMLElement): void => {
      selected =
        element &&
        element !== live.body &&
        element !== live.documentElement &&
        eligibleStoryElement(element)
          ? element
          : undefined
      confirm.disabled = !selected
      outline.hidden = !selected
      if (selected) {
        const rect = selected.getBoundingClientRect()
        Object.assign(outline.style, {
          left: rect.left + 'px',
          top: rect.top + 'px',
          width: rect.width + 'px',
          height: rect.height + 'px',
        })
        label.textContent =
          'Selected ' + selected.tagName.toLowerCase() + '. ↑ selects its parent. '
      }
    }
    const finish = (element?: HTMLElement): void => {
      live.removeEventListener('pointermove', move, true)
      live.removeEventListener('keydown', key, true)
      panel.remove()
      outline.remove()
      resolve(element)
      signal?.removeEventListener('abort', abort)
    }
    const abort = (): void => finish()
    const move = (event: PointerEvent): void => {
      const target = event.target instanceof HTMLElement ? event.target : undefined
      if (!target || target.closest('[data-hskify-owned]')) return
      update(target.closest<HTMLElement>('article,main,section,[role=main]') ?? target)
    }
    const key = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        finish()
      }
      if (event.key === 'ArrowUp' && selected) {
        event.preventDefault()
        update(selected.parentElement ?? undefined)
      }
      if (event.key === 'Enter' && selected) {
        event.preventDefault()
        finish(selected)
      }
    }
    confirm.addEventListener('click', () => {
      if (selected) finish(selected)
    })
    cancel.addEventListener('click', () => finish())
    live.addEventListener('pointermove', move, true)
    live.addEventListener('keydown', key, true)
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) finish()
  })
}
