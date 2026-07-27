import { describe, expect, it, vi } from 'vitest'

import { SelectionController } from '../../src/selection/popover'

function rect(
  left: number,
  top: number,
  width: number,
  height: number,
): DOMRect {
  return {
    x: left,
    y: top,
    left,
    top,
    right: left + width,
    bottom: top + height,
    width,
    height,
    toJSON: () => ({}),
  }
}

function fixture() {
  const host = document.createElement('span')
  document.body.append(host)
  const root = host.attachShadow({ mode: 'open' })
  const region = document.createElement('span')
  region.textContent = '恩里克，谢尔盖耶维奇，英雄党前小偷。'
  const outside = document.createElement('span')
  outside.textContent = 'Untranslated page text'
  const popover = document.createElement('span')
  popover.hidden = true
  root.append(region, outside, popover)
  vi.spyOn(host, 'getBoundingClientRect').mockReturnValue(rect(100, 0, 720, 1000))
  vi.spyOn(region, 'getBoundingClientRect').mockReturnValue(rect(220, 100, 520, 300))
  vi.spyOn(popover, 'getBoundingClientRect').mockReturnValue(
    rect(0, 0, 320, 220),
  )
  const lookup = vi.fn().mockResolvedValue({
    selectedText: '恩里克',
    tokens: [],
  })
  const controller = new SelectionController(root, popover, lookup)
  controller.register(region, 'job-1', 'region-1')
  const range = document.createRange()
  range.setStart(region.firstChild!, 0)
  range.setEnd(region.firstChild!, 3)
  Object.defineProperty(range, 'getBoundingClientRect', {
    configurable: true,
    value: () => rect(240, 150, 120, 24),
  })
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
  return { controller, host, lookup, outside, popover, range, region, selection }
}

describe('selection popover', () => {
  it('positions the explanation after the whole translated region', async () => {
    const item = fixture()
    item.region.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }))

    await vi.waitFor(() => expect(item.lookup).toHaveBeenCalledTimes(1))
    expect(item.popover.hidden).toBe(false)
    expect(item.popover.style.left).toBe('120px')
    expect(item.popover.style.top).toBe('408px')
    item.controller.destroy()
    item.host.remove()
  })

  it('positions outside the union of the selected range and translated region', async () => {
    const item = fixture()
    Object.defineProperty(item.range, 'getBoundingClientRect', {
      configurable: true,
      value: () => rect(200, 90, 560, 340),
    })
    item.region.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }))

    await vi.waitFor(() => expect(item.lookup).toHaveBeenCalledTimes(1))
    expect(item.popover.style.left).toBe('100px')
    expect(item.popover.style.top).toBe('438px')
    item.controller.destroy()
    item.host.remove()
  })

  it('dismisses when the selection is collapsed', async () => {
    const item = fixture()
    item.region.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }))
    await vi.waitFor(() => expect(item.popover.hidden).toBe(false))

    item.selection.removeAllRanges()
    document.dispatchEvent(new Event('selectionchange'))
    await vi.waitFor(() => expect(item.popover.hidden).toBe(true))
    item.controller.destroy()
    item.host.remove()
  })

  it('dismisses and clears the explanation for a cross-boundary selection', async () => {
    const item = fixture()
    item.region.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }))
    await vi.waitFor(() => expect(item.popover.hidden).toBe(false))

    const range = document.createRange()
    range.setStart(item.region.firstChild!, 0)
    range.setEnd(item.outside.firstChild!, 5)
    item.selection.removeAllRanges()
    item.selection.addRange(range)
    document.dispatchEvent(new Event('selectionchange'))

    await vi.waitFor(() => expect(item.popover.hidden).toBe(true))
    expect(item.popover.childNodes).toHaveLength(0)
    expect(item.lookup).toHaveBeenCalledTimes(1)
    item.controller.destroy()
    item.host.remove()
  })

  it('dismisses on a pointer press outside the translation shadow tree', async () => {
    const item = fixture()
    item.region.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }))
    await vi.waitFor(() => expect(item.popover.hidden).toBe(false))

    document.body.dispatchEvent(
      new Event('pointerdown', { bubbles: true, composed: true }),
    )
    expect(item.popover.hidden).toBe(true)
    item.controller.destroy()
    item.host.remove()
  })

  it('dismisses on scroll so the explanation never trails the selected text', async () => {
    const item = fixture()
    item.region.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }))
    await vi.waitFor(() => expect(item.popover.hidden).toBe(false))

    window.dispatchEvent(new Event('scroll'))
    expect(item.popover.hidden).toBe(true)
    item.controller.destroy()
    item.host.remove()
  })
})
