import type { DiscoveredImage } from '../../src/discovery/images'
import {
  createFixtureResult,
  fixtureSourceBytes,
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

const sourceSvg = `
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1800" viewBox="0 0 1200 1800">
  <rect width="1200" height="1800" fill="#f8fafc"/>
  <rect x="35" y="35" width="1130" height="1730" fill="#cbd5e1" stroke="#111827" stroke-width="18"/>
  <ellipse cx="390" cy="320" rx="250" ry="180" fill="#fff" stroke="#111827" stroke-width="12"/>
  <text x="390" y="300" text-anchor="middle" font-family="Arial" font-size="56" font-weight="700">WE HAVE TO</text>
  <text x="390" y="370" text-anchor="middle" font-family="Arial" font-size="56" font-weight="700">LEAVE NOW!</text>
  <ellipse cx="850" cy="1250" rx="220" ry="180" fill="#fff" stroke="#111827" stroke-width="12"/>
  <text x="850" y="1260" text-anchor="middle" font-family="Arial" font-size="52" font-weight="700">WAIT FOR ME!</text>
</svg>`
source.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(sourceSvg)}`
await source.decode()

let navigationCount = 0
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

const result = createFixtureResult({
  jobId: 'playwright-fixture',
  sourceSha256: 'b'.repeat(64),
  sourceWidth: 1200,
  sourceHeight: 1800,
})
if (query.get('vertical') === '1' && result.regions[0]) {
  result.regions[0].style.writingMode = 'vertical-rl'
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
        faithfulChinese: '我们得马上离开！',
        sourceEnglish: 'We have to leave now!',
      },
    }),
  }).render(candidate, {
    result,
    cleanImage: fixtureSourceBytes(),
  })
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
      setWidth(width: number): void
      destroy(): void
    }
  }
}

window.hmtHarness = {
  ready: true,
  ...(errorCode ? { errorCode } : {}),
  navigationCount: () => navigationCount,
  setWidth(width) {
    frame.style.width = `${width}px`
  },
  destroy() {
    rendered?.destroy()
  },
}
