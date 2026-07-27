import { afterEach, describe, expect, it, vi } from 'vitest'

import { ImageStatusBadge, PageHud } from '../../src/progress/hud'
import { loadedImage } from '../helpers/images'

afterEach(() => {
  document.documentElement.querySelectorAll('[data-hmt-owned]').forEach((item) => item.remove())
  document.body.replaceChildren()
})

describe('page and image progress UI', () => {
  it('shows measurable progress and cancellation without inventing percentages', () => {
    const cancel = vi.fn()
    const hud = new PageHud(cancel)
    hud.update({
      current: 1,
      total: 4,
      status: {
        sequence: 2,
        type: 'progress',
        stage: 'ocr',
        overallProgress: 0.3,
        current: 2,
        total: 7,
        message: 'Running OCR batch 2 of 7',
      },
    })
    const shadow = hud.host.shadowRoot
    expect(shadow?.textContent).toContain('Image 2 of 4')
    expect(shadow?.textContent).toContain('Translating this chapter')
    expect(shadow?.textContent).not.toContain('OCR')
    expect(shadow?.textContent).not.toContain('2 of 7')
    expect(shadow?.querySelector('progress')?.getAttribute('value')).toBe('0.3')
    shadow?.querySelector('button')?.dispatchEvent(new Event('click'))
    expect(cancel).toHaveBeenCalled()

    hud.update({
      current: 1,
      total: 4,
      status: {
        sequence: 3,
        type: 'progress',
        stage: 'inpainting',
        message: 'Removing lettering',
      },
    })
    expect(shadow?.querySelector('progress')?.hasAttribute('value')).toBe(false)
  })

  it('reports complete, failure, and cancellation states explicitly', () => {
    const hud = new PageHud(() => undefined)
    hud.complete(3, 3)
    expect(hud.snapshot()).toMatchObject({ state: 'complete', current: 3, total: 3 })
    hud.fail('This image couldn’t be translated. Try again.', 1, 3)
    expect(hud.snapshot()).toMatchObject({
      state: 'failed',
      message: 'This image couldn’t be translated. Try again.',
    })
    hud.cancelled(1, 3)
    expect(hud.snapshot().message).toContain('unfinished was left unchanged')
  })

  it('keeps retry as the one clear image-level failure action', () => {
    const image = loadedImage()
    document.body.append(image)
    const retry = vi.fn()
    const badge = new ImageStatusBadge(image, retry)
    badge.failure('Image permission denied')
    const button = document.documentElement
      .querySelector<HTMLElement>('[data-hmt-owned]')
      ?.shadowRoot?.querySelector('button')
    expect(button?.hidden).toBe(false)
    button?.dispatchEvent(new Event('click'))
    expect(retry).toHaveBeenCalled()
    badge.destroy()
  })
})
