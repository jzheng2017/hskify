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
    lookup: async (request) => ({
      selectedText: request.selectedText,
      tokens: [
        {
          simplified: request.selectedText,
          pinyin: 'lí kāi',
          definitions: ['leave', 'depart'],
          hskLevel: 2,
          properName: false,
        },
      ],
      region: {
        displayedChinese: '我们现在要走！',
        baseChinese: '我们得马上离开！',
        sourceEnglish: 'We have to leave now!',
      },
    }),
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
