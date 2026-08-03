import {
  parseJobUpdate,
  parseLookupResult,
  type BrowserRegion,
  type JobUpdate,
  type JobUpdateBatch,
  type LookupRequest,
  type LookupResult,
  type RegionReadyJobUpdate,
} from '../../src/contracts/browser'
import type { ActiveJobRecord } from '../../src/messaging/active-jobs'

export function fixtureFontBytes(): ArrayBuffer {
  // Deliberately tiny WOFF-shaped data exercises ArrayBuffer transfer and the
  // renderer's measured system-font fallback without shipping a font asset.
  return Uint8Array.of(0x77, 0x4f, 0x46, 0x46, 0, 0, 0, 0).buffer
}

export type FixtureRegionInput = {
  jobId: string
  sourceSha256: string
  sourceWidth: number
  sourceHeight: number
  hskLevel?: 1 | 2 | 3 | 4 | 5 | 6
}

export function createFixtureRegions(input: FixtureRegionInput): BrowserRegion[] {
  const requestedLevel = input.hskLevel ?? 2
  return [
    (parseJobUpdate({
      sequence: 1,
      type: 'regionReady',
      region: {
        id: `${input.sourceSha256.slice(0, 8)}-region-0001`,
        textPolygon: [
          { x: 0.19, y: 0.12 },
          { x: 0.46, y: 0.12 },
          { x: 0.46, y: 0.25 },
          { x: 0.19, y: 0.25 },
        ],
        bubblePolygon: [
          { x: 0.16, y: 0.09 },
          { x: 0.49, y: 0.09 },
          { x: 0.51, y: 0.27 },
          { x: 0.16, y: 0.28 },
        ],
        patch: {
          blobId: `fixture-patch-${input.jobId}-1`,
          mimeType: 'image/png',
          rect: { x: 0.15, y: 0.08, width: 0.38, height: 0.22 },
        },
        sourceEnglish: 'We have to leave now!',
        baseChinese: '我们得马上离开！',
        displayedChinese: '我们现在要走！',
        pinyin: 'wǒ men xiàn zài yào zǒu',
        ocrConfidence: 0.97,
        readingOrder: 0,
        hsk: {
          requestedLevel,
          learningMode: 'natural',
          strictlyValid: true,
          levelCoverage: 1,
          aboveLevelTokens: [],
          teachingTerms: [],
          repairState: 'not-needed',
        },
        style: {
          fontId: 'fixture-sans',
          category: 'sans',
          foreground: '#151515',
          weight: 700,
          italicDegrees: 0,
          outlineColor: '#ffffff',
          outlineWidthRatio: 0.035,
          shadowColor: '#00000033',
          shadowXRatio: 0.01,
          shadowYRatio: 0.015,
          alignment: 'center',
          writingMode: 'horizontal-tb',
          lineHeight: 1.12,
          letterSpacingEm: 0,
        },
        layout: {
          suggestedLines: ['我们现在', '要走！'],
          fontSizeToImageWidth: 0.034,
          safePolygon: [
            { x: 0.18, y: 0.11 },
            { x: 0.48, y: 0.11 },
            { x: 0.48, y: 0.26 },
            { x: 0.18, y: 0.26 },
          ],
        },
      },
    }) as RegionReadyJobUpdate).region,
    (parseJobUpdate({
      sequence: 1,
      type: 'regionReady',
      region: {
        id: `${input.sourceSha256.slice(0, 8)}-region-0002`,
        textPolygon: [
          { x: 0.58, y: 0.66 },
          { x: 0.85, y: 0.62 },
          { x: 0.88, y: 0.78 },
          { x: 0.61, y: 0.82 },
        ],
        patch: {
          blobId: `fixture-patch-${input.jobId}-2`,
          mimeType: 'image/png',
          rect: { x: 0.55, y: 0.59, width: 0.36, height: 0.26 },
        },
        sourceEnglish: 'Wait for me!',
        baseChinese: '等等我！',
        displayedChinese: '等我！',
        pinyin: 'děng wǒ',
        ocrConfidence: 0.93,
        readingOrder: 1,
        hsk: {
          requestedLevel,
          learningMode: 'natural',
          strictlyValid: true,
          levelCoverage: 1,
          aboveLevelTokens: [],
          teachingTerms: [],
          repairState: 'not-needed',
        },
        style: {
          fontId: 'fixture-display',
          category: 'display',
          foreground: '#172a52',
          weight: 800,
          italicDegrees: -4,
          outlineColor: '#ffffff',
          outlineWidthRatio: 0.025,
          shadowXRatio: 0,
          shadowYRatio: 0,
          alignment: 'center',
          writingMode: 'horizontal-tb',
          lineHeight: 1,
          letterSpacingEm: 0.02,
        },
        layout: {
          suggestedLines: ['等我！'],
          fontSizeToImageWidth: 0.04,
          safePolygon: [
            { x: 0.59, y: 0.65 },
            { x: 0.84, y: 0.63 },
            { x: 0.86, y: 0.77 },
            { x: 0.61, y: 0.79 },
          ],
        },
      },
    }) as RegionReadyJobUpdate).region,
  ]
}

function fixtureTimeline(record: ActiveJobRecord): Array<{ at: number; update: JobUpdate }> {
  const regions = createFixtureRegions({
    jobId: record.jobId,
    sourceSha256: record.sourceSha256,
    sourceWidth: record.sourceWidth,
    sourceHeight: record.sourceHeight,
    hskLevel: record.hskLevel,
  })
  const first = regions[0]
  const second = regions[1]
  if (!first || !second) throw new Error('Fixture regions are incomplete.')
  return [
    {
      at: 0,
      update: parseJobUpdate({
        sequence: 1,
        type: 'progress',
        stage: 'queued',
        overallProgress: 0,
        message: 'Queued',
      }),
    },
    {
      at: 250,
      update: parseJobUpdate({
        sequence: 2,
        type: 'progress',
        stage: 'ocr',
        overallProgress: 0.3,
        message: 'Reading text',
      }),
    },
    {
      at: 500,
      update: { sequence: 3, type: 'regionReady', region: first },
    },
    {
      at: 750,
      update: { sequence: 4, type: 'regionReady', region: second },
    },
    {
      at: 1_000,
      update: parseJobUpdate({
        sequence: 5,
        type: 'complete',
        message: 'Complete',
      }),
    },
  ]
}

export class FixtureService {
  constructor(private readonly now: () => number = Date.now) {}

  sourceImage(width: number, height: number): Promise<ArrayBuffer> {
    return createFixturePng(width, height)
  }

  createJobId(pageSessionId: string, pageIndex: number, sourceSha256: string): string {
    return `fixture-${pageSessionId.slice(0, 12)}-${pageIndex}-${sourceSha256.slice(0, 12)}`
  }

  updates(record: ActiveJobRecord, after: number): JobUpdateBatch {
    const elapsed = Math.max(0, this.now() - record.createdAtUnixMs)
    const available = fixtureTimeline(record)
      .filter((entry) => entry.at <= elapsed && entry.update.sequence > after)
      .map((entry) => entry.update)
    return {
      jobId: record.jobId,
      nextSequence: available.at(-1)?.sequence ?? after,
      updates: available,
    }
  }

  viewport(_record: ActiveJobRecord): void {
    // The deterministic fixture has no scheduler, but accepting viewport
    // updates exercises the same background route as the real companion.
  }

  async patch(record: ActiveJobRecord, patchId: string): Promise<ArrayBuffer> {
    const region = createFixtureRegions({
      jobId: record.jobId,
      sourceSha256: record.sourceSha256,
      sourceWidth: record.sourceWidth,
      sourceHeight: record.sourceHeight,
      hskLevel: record.hskLevel,
    }).find((candidate) => candidate.patch.blobId === patchId)
    if (!region) throw new Error('Fixture patch does not belong to this job.')
    return createFixturePng(
      Math.max(1, Math.round(region.patch.rect.width * record.sourceWidth)),
      Math.max(1, Math.round(region.patch.rect.height * record.sourceHeight)),
    )
  }

  font(): ArrayBuffer {
    return fixtureFontBytes()
  }

  lookup(request: LookupRequest): LookupResult {
    const selectedText =
      request.interaction === 'selection' ? request.selectedText : '离开'
    const isLeave = selectedText.includes('离开')
    return parseLookupResult({
      selectedText,
      tokens: [
        isLeave
          ? {
              simplified: '离开',
              pinyin: 'lí kāi',
              definitions: ['leave', 'depart'],
              hskLevel: 2,
              properName: false,
            }
          : {
              simplified: selectedText,
              pinyin: selectedText === '等' ? 'děng' : 'fixture',
              definitions: ['fixture dictionary entry'],
              hskLevel: 1,
              properName: false,
            },
      ],
      ...(request.jobId
        ? {
            region: {
              displayedChinese: '我们现在就走！',
              baseChinese: '我们得马上离开！',
              sourceEnglish: 'We have to leave now!',
            },
          }
        : {}),
    })
  }
}

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff
  for (const byte of bytes) {
    crc ^= byte
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1))
    }
  }
  return (crc ^ 0xffffffff) >>> 0
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const typeBytes = new TextEncoder().encode(type)
  const chunk = new Uint8Array(12 + data.byteLength)
  const view = new DataView(chunk.buffer)
  view.setUint32(0, data.byteLength)
  chunk.set(typeBytes, 4)
  chunk.set(data, 8)
  view.setUint32(8 + data.byteLength, crc32(chunk.subarray(4, 8 + data.byteLength)))
  return chunk
}

export async function createFixturePng(width: number, height: number): Promise<ArrayBuffer> {
  const raw = new Uint8Array((width + 1) * height)
  const compressed = new Uint8Array(
    await new Response(
      new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate')),
    ).arrayBuffer(),
  )
  const header = new Uint8Array(13)
  const headerView = new DataView(header.buffer)
  headerView.setUint32(0, width)
  headerView.setUint32(4, height)
  header.set([8, 0, 0, 0, 0], 8)
  const chunks = [
    Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a),
    pngChunk('IHDR', header),
    pngChunk('IDAT', compressed),
    pngChunk('IEND', new Uint8Array()),
  ]
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0)
  const png = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    png.set(chunk, offset)
    offset += chunk.byteLength
  }
  return png.buffer
}
