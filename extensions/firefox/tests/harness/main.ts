import type { DiscoveredImage } from '../../src/discovery/images'
import fixturePanelUrl from '../../../../fixtures/images/synthetic-panel-a.png?url'
import longWebtoonUrl from '../../../../fixtures/images/synthetic-webtoon-long.webp?url'
import {
  createFixtureRegions,
} from '../../src/messaging/fixture-service'
import {
  SelectableRenderer,
  type RenderedImage,
} from '../../src/rendering/renderer'

const source = document.querySelector<HTMLImageElement>('#source')
const frame = document.querySelector<HTMLElement>('#frame')
const link = document.querySelector<HTMLAnchorElement>('#reader-link')
const navigationOutput = document.querySelector<HTMLOutputElement>('#navigation-count')
if (!source || !frame || !link || !navigationOutput) throw new Error('Harness DOM is incomplete.')

source.src = fixturePanelUrl
await source.decode()
const patchImage = await (await fetch(fixturePanelUrl)).arrayBuffer()
const longWebtoonProbe = new Image()
longWebtoonProbe.src = `${longWebtoonUrl}?chapter=synthetic&page=0`
await longWebtoonProbe.decode()

let navigationCount = 0
let directImageClickCount = 0
source.addEventListener('click', () => {
  directImageClickCount += 1
})
link.addEventListener('click', (event) => {
  event.preventDefault()
  navigationCount += 1
  navigationOutput.value = String(navigationCount)
  navigationOutput.textContent = String(navigationCount)
})

const query = new URL(location.href).searchParams
if (query.get('fit') === 'contain' || query.get('fit') === 'cover') {
  source.style.height = '520px'
  source.style.objectFit = query.get('fit') ?? 'fill'
  source.style.objectPosition = '70% 50%'
}
if (query.get('rotated') === '1') source.style.transform = 'rotate(3deg)'

const regions = createFixtureRegions({
  jobId: 'playwright-fixture',
  sourceSha256: 'b'.repeat(64),
  sourceWidth: 1200,
  sourceHeight: 1800,
})
if (query.get('vertical') === '1' && regions[0]) {
  regions[0].style.writingMode = 'vertical-rl'
}
if (query.get('stress') === '1' && regions[0]) {
  regions[0].displayedChinese =
    '帝国发生了一件从来没有发生过的事，Enrique把这件事叫作“四十七号政变”。'
  regions[0].baseChinese = regions[0].displayedChinese
  regions[0].layout.suggestedLines = [
    '帝国发生了一件从来没有发生过的事，',
    'Enrique把这件事叫作“四十七号政变”。',
  ]
  regions[0].layout.fontSizeToImageWidth = 0.075
  regions[0].layout.safePolygon = [
    { x: 0.2, y: 0.11 },
    { x: 0.45, y: 0.11 },
    { x: 0.49, y: 0.18 },
    { x: 0.45, y: 0.26 },
    { x: 0.2, y: 0.26 },
    { x: 0.17, y: 0.18 },
  ]
}
if (query.get('hover') === '1' && regions[0]) {
  regions[0].displayedChinese = '\u7814\u7a76\u751f\u79bb\u5f00'
  regions[0].baseChinese = regions[0].displayedChinese
  regions[0].layout.suggestedLines = ['\u7814\u7a76\u751f\u79bb\u5f00']
}
for (const region of regions) {
  region.patch.rect = { x: 0, y: 0, width: 1, height: 1 }
}
const candidate: DiscoveredImage = {
  element: source,
  owner: source,
  sourceUrl: source.currentSrc || source.src,
  domIndex: 0,
  visible: true,
}

let rendered: RenderedImage | undefined
let errorCode: string | undefined
try {
  rendered = await new SelectableRenderer({
    fetchFont: async () => {
      throw new Error('The harness intentionally exercises font fallback.')
    },
    lookup: async (request) => {
      const displayedText =
        regions.find((region) => region.id === request.regionId)?.displayedChinese ?? ''
      const suffix =
        request.interaction === 'hover'
          ? [...displayedText].slice(request.characterOffset).join('')
          : ''
      const selectedText =
        request.interaction === 'selection'
          ? request.selectedText
          : [
              '\u7814\u7a76\u751f',
              '\u7814\u7a76',
              '\u79bb\u5f00',
              '\u751f',
              '\u5f00',
            ].find((word) => suffix.startsWith(word)) ?? [...suffix][0] ?? ''
      return {
        selectedText,
        tokens: selectedText
          ? [
              {
                simplified: selectedText,
                pinyin:
                  request.interaction === 'selection' ? 'lí kāi' : 'fixture',
                definitions:
                  request.interaction === 'selection'
                    ? ['leave', 'depart']
                    : ['fixture dictionary entry'],
                hskLevel: 2 as const,
                properName: false,
              },
            ]
          : [],
        region: {
          displayedChinese: displayedText,
          baseChinese: displayedText,
          sourceEnglish: 'We have to leave now!',
        },
      }
    },
  }).begin(candidate, {
    jobId: 'playwright-fixture',
    sourceWidth: 1200,
    sourceHeight: 1800,
  })
  for (const region of regions) {
    await rendered.installRegion(region, patchImage)
  }
} catch (error) {
  errorCode =
    error instanceof Error && 'code' in error ? String(error.code) : 'UNKNOWN_RENDERER_ERROR'
}

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

window.hmtHarness = {
  ready: true,
  ...(errorCode ? { errorCode } : {}),
  navigationCount: () => navigationCount,
  directImageClickCount: () => directImageClickCount,
  longFixture: () => ({
    width: longWebtoonProbe.naturalWidth,
    height: longWebtoonProbe.naturalHeight,
    url: longWebtoonProbe.currentSrc || longWebtoonProbe.src,
  }),
  setWidth(width) {
    frame.style.width = `${width}px`
  },
  destroy() {
    rendered?.destroy()
  },
}
