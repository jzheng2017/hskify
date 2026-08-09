import {
  parseJobUpdate,
  parseLookupResult,
  type DocumentJobRequest,
  type ImageRegion,
  type ImageRegionReadyJobUpdate,
  type JobFocus,
  type JobUpdate,
  type JobUpdateBatch,
  type LookupRequest,
  type LookupResult,
} from '../../src/contracts/browser'
import type { ActiveJobRecord } from '../../src/messaging/active-jobs'

export function fixtureFontBytes(): ArrayBuffer {
  return Uint8Array.of(0x77, 0x4f, 0x46, 0x46, 0, 0, 0, 0).buffer
}

export type FixtureRegionInput = {
  jobId: string
  sourceSha256: string
  sourceWidth: number
  sourceHeight: number
  hskLevel?: 1 | 2 | 3 | 4 | 5 | 6
}

export function createFixtureRegions(input: FixtureRegionInput): ImageRegion[] {
  const requestedLevel = input.hskLevel ?? 2
  const commonHsk = {
    requestedLevel,
    learningMode: 'natural' as const,
    strictlyValid: true,
    levelCoverage: 1,
    aboveLevelTokens: [],
    teachingTerms: [],
    repairState: 'not-needed' as const,
  }
  const first = parseJobUpdate({
    sequence: 1,
    type: 'imageRegionReady',
    region: {
      itemId: `${input.sourceSha256.slice(0, 8)}-region-0001`,
      itemOrder: 0,
      kind: 'dialogue',
      provenance: 'ocr',
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
      text: {
        sourceText: 'We have to leave now!',
        baseChinese: '\u6211\u4eec\u5f97\u9a6c\u4e0a\u79bb\u5f00\uff01',
        displayedChinese: '\u6211\u4eec\u73b0\u5728\u8981\u8d70\uff01',
        pinyin: 'w\u01d2 men xi\u00e0n z\u00e0i y\u00e0o z\u01d2u',
        hsk: commonHsk,
      },
      confidence: 0.97,
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
        suggestedLines: ['\u6211\u4eec\u73b0\u5728', '\u8981\u8d70\uff01'],
        fontSizeToImageWidth: 0.034,
        safePolygon: [
          { x: 0.18, y: 0.11 },
          { x: 0.48, y: 0.11 },
          { x: 0.48, y: 0.26 },
          { x: 0.18, y: 0.26 },
        ],
      },
    },
  }) as ImageRegionReadyJobUpdate
  const second = parseJobUpdate({
    sequence: 2,
    type: 'imageRegionReady',
    region: {
      itemId: `${input.sourceSha256.slice(0, 8)}-region-0002`,
      itemOrder: 1,
      kind: 'dialogue',
      provenance: 'ocr',
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
      text: {
        sourceText: 'Wait for me!',
        baseChinese: '\u7b49\u7b49\u6211\uff01',
        displayedChinese: '\u7b49\u6211\uff01',
        pinyin: 'd\u011bng w\u01d2',
        hsk: commonHsk,
      },
      confidence: 0.93,
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
        suggestedLines: ['\u7b49\u6211\uff01'],
        fontSizeToImageWidth: 0.04,
        safePolygon: [
          { x: 0.59, y: 0.65 },
          { x: 0.84, y: 0.63 },
          { x: 0.86, y: 0.77 },
          { x: 0.61, y: 0.79 },
        ],
      },
    },
  }) as ImageRegionReadyJobUpdate
  return [first.region, second.region]
}

function fixtureTimeline(
  record: ActiveJobRecord,
  documentRequest?: DocumentJobRequest,
): Array<{ at: number; update: JobUpdate }> {
  if (record.source.kind === 'document') {
    const source = record.source
    if (
      !documentRequest ||
      documentRequest.sourceSha256 !== record.sourceSha256 ||
      documentRequest.blocks.length !== source.blockCount
    ) {
      throw new Error('Fixture document request is not registered for this active job.')
    }
    const ready = documentRequest.blocks.map((block, index) =>
      parseJobUpdate({
        sequence: index + 3,
        type: 'documentBlockReady',
        block: {
          itemId: block.itemId,
          sourceIndex: block.sourceIndex,
          itemOrder: block.itemOrder,
          kind: block.kind,
          text: {
            sourceText: block.text,
            baseChinese: `\u7ffb\u8bd1 ${index + 1}`,
            displayedChinese: `\u7ffb\u8bd1 ${index + 1}`,
            pinyin: `f\u0101n y\u00ec ${index + 1}`,
            hsk: {
              requestedLevel: source.settings.hskLevel,
              learningMode: source.settings.learningMode,
              strictlyValid: true,
              levelCoverage: 1,
              aboveLevelTokens: [],
              teachingTerms: [],
              repairState: 'not-needed',
            },
          },
        },
      }),
    )
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
          stage: 'registering',
          overallProgress: 0.1,
          message: 'Registered document blocks',
        }),
      },
      ...ready.map((update, index) => ({ at: 500 + index * 50, update })),
      {
        at: Math.max(1_000, 550 + ready.length * 50),
        update: parseJobUpdate({
          sequence: ready.length + 3,
          type: 'complete',
          translatedCount: ready.length,
          preservedCount: 0,
          message: 'Complete',
        }),
      },
    ]
  }

  const source = record.source
  const regions = createFixtureRegions({
    jobId: record.jobId,
    sourceSha256: record.sourceSha256,
    sourceWidth: source.sourceWidth,
    sourceHeight: source.sourceHeight,
    hskLevel: source.request.settings.hskLevel,
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
    { at: 500, update: { sequence: 3, type: 'imageRegionReady', region: first } },
    { at: 750, update: { sequence: 4, type: 'imageRegionReady', region: second } },
    {
      at: 1_000,
      update: parseJobUpdate({
        sequence: 5,
        type: 'complete',
        translatedCount: 2,
        preservedCount: 0,
        message: 'Complete',
      }),
    },
  ]
}

export class FixtureService {
  private readonly documentRequests = new Map<string, DocumentJobRequest>()

  constructor(private readonly now: () => number = Date.now) {}

  sourceImage(width: number, height: number): Promise<ArrayBuffer> {
    return createFixturePng(width, height)
  }

  createJobId(pageSessionId: string, sourceIndex: number, sourceSha256: string): string {
    return `fixture-${pageSessionId.slice(0, 12)}-${sourceIndex}-${sourceSha256.slice(0, 12)}`
  }

  registerDocument(jobId: string, request: DocumentJobRequest): void {
    this.documentRequests.set(jobId, request)
  }

  releaseJob(jobId: string): void {
    this.documentRequests.delete(jobId)
  }

  updates(record: ActiveJobRecord, after: number): JobUpdateBatch {
    const elapsed = Math.max(0, this.now() - record.createdAtUnixMs)
    const available = fixtureTimeline(record, this.documentRequests.get(record.jobId))
      .filter((entry) => entry.at <= elapsed && entry.update.sequence > after)
      .map((entry) => entry.update)
    return {
      jobId: record.jobId,
      nextSequence: available.at(-1)?.sequence ?? after,
      updates: available,
    }
  }

  focus(_record: ActiveJobRecord, _focus: JobFocus): void {}

  async patch(record: ActiveJobRecord, patchId: string): Promise<ArrayBuffer> {
    if (record.source.kind !== 'image') throw new Error('Document jobs do not own patches.')
    const source = record.source
    const region = createFixtureRegions({
      jobId: record.jobId,
      sourceSha256: record.sourceSha256,
      sourceWidth: source.sourceWidth,
      sourceHeight: source.sourceHeight,
      hskLevel: source.request.settings.hskLevel,
    }).find((candidate) => candidate.patch.blobId === patchId)
    if (!region) throw new Error('Fixture patch does not belong to this job.')
    return createFixturePng(
      Math.max(1, Math.round(region.patch.rect.width * source.sourceWidth)),
      Math.max(1, Math.round(region.patch.rect.height * source.sourceHeight)),
    )
  }

  font(): ArrayBuffer {
    return fixtureFontBytes()
  }

  lookup(request: LookupRequest): LookupResult {
    const selectedText = request.interaction === 'selection' ? request.selectedText : '\u79bb\u5f00'
    const isLeave = selectedText.includes('\u79bb\u5f00')
    return parseLookupResult({
      selectedText,
      tokens: [
        isLeave
          ? {
              simplified: '\u79bb\u5f00',
              pinyin: 'l\u00ed k\u0101i',
              definitions: ['leave', 'depart'],
              hskLevel: 2,
              properName: false,
            }
          : {
              simplified: selectedText,
              pinyin: selectedText === '\u7b49' ? 'd\u011bng' : 'fixture',
              definitions: ['fixture dictionary entry'],
              hskLevel: 1,
              properName: false,
            },
      ],
      ...(request.jobId
        ? {
            item: {
              displayedChinese: '\u6211\u4eec\u73b0\u5728\u5c31\u8d70\uff01',
              baseChinese: '\u6211\u4eec\u5f97\u9a6c\u4e0a\u79bb\u5f00\uff01',
              sourceText: 'We have to leave now!',
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
