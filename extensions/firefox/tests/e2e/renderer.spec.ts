import { expect, test, type Page } from '@playwright/test'

async function waitForHarness(page: Page): Promise<void> {
  await expect.poll(() => page.evaluate(() => window.hmtHarness?.ready)).toBe(true)
}

async function hoverCharacter(
  page: Page,
  regionIndex: number,
  characterOffset: number,
): Promise<void> {
  const point = await page
    .locator('.hmt-region')
    .nth(regionIndex)
    .evaluate((region, offset) => {
      const walker = document.createTreeWalker(region, NodeFilter.SHOW_TEXT)
      let remaining = offset
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        if (!(node instanceof Text)) continue
        const characters = [...node.data]
        if (remaining >= characters.length) {
          remaining -= characters.length
          continue
        }
        const start = characters.slice(0, remaining).join('').length
        const end = start + characters[remaining]!.length
        const range = document.createRange()
        range.setStart(node, start)
        range.setEnd(node, end)
        const rect = range.getBoundingClientRect()
        return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }
      }
      throw new Error('Character offset is outside the translated region.')
    }, characterOffset)
  await page.mouse.move(point.x, point.y)
}

test('explains the longest expression beginning at the hovered character', async ({ page }) => {
  await page.goto('/?hover=1')
  await waitForHarness(page)
  const heading = page.locator('.hmt-lookup-heading strong')

  await hoverCharacter(page, 0, 0)
  await expect(heading).toHaveText('\u7814\u7a76\u751f')
  expect(await page.evaluate(() => window.getSelection()?.isCollapsed ?? true)).toBe(true)

  await hoverCharacter(page, 0, 2)
  await expect(heading).toHaveText('\u751f')
})

test('renders exact selectable Chinese and dictionary context', async ({ page }) => {
  await page.goto('/')
  await waitForHarness(page)
  const regions = page.locator('.hmt-region')
  await expect(regions).toHaveCount(2)
  await expect(regions.first()).toHaveText('我们现在要走！')

  const selected = await regions.first().evaluate((region) => {
    const range = document.createRange()
    range.selectNodeContents(region)
    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)
    region.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, composed: true }))
    return selection?.toString()
  })
  expect(selected).toBe('我们现在要走！')
  await expect(page.locator('.hmt-lookup')).toContainText('lí kāi')
  await expect(page.locator('.hmt-lookup')).toContainText('We have to leave now!')

  await regions.first().focus()
  await page.keyboard.press('Control+A')
  expect(await page.evaluate(() => window.getSelection()?.toString())).toBe('我们现在要走！')

  const speak = page.locator('.hmt-speak')
  await expect(speak).toHaveAttribute('aria-label', 'Listen to Mandarin pronunciation')
  await speak.click()
  await expect(speak).toHaveAttribute('aria-pressed', 'true')
  await expect(speak).toHaveText(/Loading…|Stop/)

  const original = page.locator('.hmt-controls button', { hasText: 'Original' })
  await original.focus()
  await page.keyboard.press('Enter')
  await expect(page.locator('.hmt-lookup')).toBeHidden()
  await expect(speak).toHaveCount(0)
})

test('keeps reader navigation for clicks and suppresses only selection clicks', async ({
  page,
}) => {
  await page.goto('/')
  await waitForHarness(page)
  const region = page.locator('.hmt-region').first()
  await region.click()
  await expect.poll(() => page.evaluate(() => window.hmtHarness.navigationCount())).toBe(1)
  await expect.poll(() => page.evaluate(() => window.hmtHarness.directImageClickCount())).toBe(1)

  await region.evaluate((element) => {
    const range = document.createRange()
    range.selectNodeContents(element)
    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, composed: true }))
  })
  expect(await page.evaluate(() => window.hmtHarness.navigationCount())).toBe(1)
})

test('refits normalized geometry within two CSS pixels after resize', async ({ page }) => {
  await page.goto('/')
  await waitForHarness(page)
  await page.evaluate(() => window.hmtHarness.setWidth(420))
  await page.waitForTimeout(50)
  const delta = await page.evaluate(() => {
    const image = document.querySelector<HTMLImageElement>('#source')
    const region = document
      .querySelector<HTMLElement>('[aria-label="HSK manga translation controls"]')
      ?.shadowRoot?.querySelector<HTMLElement>('.hmt-region')
    if (!image || !region) return Number.POSITIVE_INFINITY
    const imageRect = image.getBoundingClientRect()
    const regionRect = region.getBoundingClientRect()
    const expectedLeft = imageRect.left + imageRect.width * 0.18
    return Math.abs(regionRect.left - expectedLeft)
  })
  expect(delta).toBeLessThanOrEqual(2)
})

test('keeps translated regions aligned in the same frame as document scrolling', async ({
  page,
}) => {
  await page.goto('/')
  await waitForHarness(page)
  const drift = await page.evaluate(() => {
    const image = document.querySelector<HTMLImageElement>('#source')
    const region = document
      .querySelector<HTMLElement>('[aria-label="HSK manga translation controls"]')
      ?.shadowRoot?.querySelector<HTMLElement>('.hmt-region')
    if (!image || !region) return Number.POSITIVE_INFINITY
    const beforeImage = image.getBoundingClientRect()
    const beforeRegion = region.getBoundingClientRect()
    const beforeOffset = beforeRegion.top - beforeImage.top
    window.scrollTo(0, 300)
    const afterImage = image.getBoundingClientRect()
    const afterRegion = region.getBoundingClientRect()
    return Math.abs(afterRegion.top - afterImage.top - beforeOffset)
  })

  expect(drift).toBeLessThanOrEqual(1)
})

test('measures mixed-script chapter text into an irregular bubble without clipping', async ({
  page,
}) => {
  await page.goto('/?stress=1')
  await waitForHarness(page)

  const measure = async () =>
    page
      .locator('.hmt-region')
      .first()
      .evaluate((region) => {
        const content = region.querySelector<HTMLElement>('.hmt-region-text')
        if (!(region instanceof HTMLElement) || !content) {
          return { fits: false, fontSize: 0, text: '' }
        }
        const outer = region.getBoundingClientRect()
        const inner = content.getBoundingClientRect()
        return {
          fits:
            region.scrollWidth <= region.clientWidth + 1 &&
            region.scrollHeight <= region.clientHeight + 1 &&
            inner.left >= outer.left - 1 &&
            inner.right <= outer.right + 1 &&
            inner.top >= outer.top - 1 &&
            inner.bottom <= outer.bottom + 1,
          fontSize: Number.parseFloat(getComputedStyle(region).fontSize),
          text: content.textContent ?? '',
        }
      })

  let metrics = await measure()
  expect(metrics.fits).toBe(true)
  expect(metrics.fontSize).toBeGreaterThan(0)
  expect(metrics.text).toContain('Enrique')
  expect(metrics.text).toContain('四十七号政变')

  await page.evaluate(() => window.hmtHarness.setWidth(420))
  await page.waitForTimeout(50)
  metrics = await measure()
  expect(metrics.fits).toBe(true)
  expect(metrics.fontSize).toBeGreaterThan(0)
})

test('maps contain and cover object-fit content boxes', async ({ page }) => {
  for (const fit of ['contain', 'cover'] as const) {
    await page.goto(`/?fit=${fit}`)
    await waitForHarness(page)
    await expect.poll(() => page.evaluate(() => window.hmtHarness?.errorCode)).toBeUndefined()
    const delta = await page.evaluate((objectFit) => {
      const image = document.querySelector<HTMLImageElement>('#source')
      const region = document
        .querySelector<HTMLElement>('[aria-label="HSK manga translation controls"]')
        ?.shadowRoot?.querySelector<HTMLElement>('.hmt-region')
      if (!image || !region) return Number.POSITIVE_INFINITY
      const imageRect = image.getBoundingClientRect()
      const sourceWidth = 700
      const sourceHeight = 1280
      const containScale = Math.min(imageRect.width / sourceWidth, imageRect.height / sourceHeight)
      const coverScale = Math.max(imageRect.width / sourceWidth, imageRect.height / sourceHeight)
      const scale = objectFit === 'contain' ? containScale : coverScale
      const drawnWidth = sourceWidth * scale
      const objectPositionLeft = (imageRect.width - drawnWidth) * 0.7
      const expected = imageRect.left + objectPositionLeft + drawnWidth * 0.18
      return Math.abs(region.getBoundingClientRect().left - expected)
    }, fit)
    expect(delta).toBeLessThanOrEqual(2)
  }
})

test('supports original, Chinese, press compare, vertical text, and safe rejection', async ({
  page,
}) => {
  await page.goto('/?vertical=1')
  await waitForHarness(page)
  const image = page.locator('#source')
  await expect(image).toHaveCSS('opacity', '1')
  await page.getByRole('button', { name: 'Original' }).click()
  await expect(image).toHaveCSS('opacity', '1')
  await expect(page.locator('.hmt-viewport')).toBeHidden()
  await page.getByRole('button', { name: 'Chinese' }).click()
  await expect(image).toHaveCSS('opacity', '1')
  await expect(page.locator('.hmt-viewport')).toBeVisible()
  const compare = page.getByRole('button', { name: 'Hold to compare' })
  await compare.dispatchEvent('pointerdown')
  await expect(image).toHaveCSS('opacity', '1')
  await expect(page.locator('.hmt-viewport')).toBeHidden()
  await compare.dispatchEvent('pointerup')
  await expect(image).toHaveCSS('opacity', '1')
  await expect(page.locator('.hmt-viewport')).toBeVisible()
  await expect(page.locator('.hmt-region').first()).toHaveCSS('writing-mode', 'vertical-rl')

  await page.goto('/?rotated=1')
  await waitForHarness(page)
  await expect
    .poll(() => page.evaluate(() => window.hmtHarness.errorCode))
    .toBe('UNSUPPORTED_IMAGE_TRANSFORM')
  await expect(page.locator('#source')).toHaveCSS('opacity', '1')
})

test('browser decodes a content-addressed real long WebP at production dimensions', async ({
  page,
}) => {
  await page.goto('/')
  await waitForHarness(page)
  const fixture = await page.evaluate(() => window.hmtHarness.longFixture())
  expect(fixture).toMatchObject({ width: 900, height: 16_000 })
  expect(fixture.url).toContain('/__real-reader/asura-mercenary-98-page-6')
})

declare global {
  interface Window {
    hmtHarness: {
      ready: boolean
      errorCode?: string
      navigationCount(): number
      directImageClickCount(): number
      longFixture(): { width: number; height: number; url: string }
      setWidth(width: number): void
      destroy(): void
    }
  }
}
