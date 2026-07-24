import {
  parseBrowserJobResult,
  parseLookupResult,
  type BrowserJobResult,
  type BrowserJobStatus,
  type LookupRequest,
  type LookupResult,
} from '../contracts/browser'
import type { ActiveJobRecord } from './active-jobs'

// A valid one-pixel PNG is intentionally embedded in the fixture adapter. The
// renderer stretches it over its image-space layer; production paths always
// transfer the clean blob returned by the companion.
const FIXTURE_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='

function decodeBase64(value: string): ArrayBuffer {
  const decoded = atob(value)
  const bytes = new Uint8Array(decoded.length)
  for (let index = 0; index < decoded.length; index += 1) {
    bytes[index] = decoded.charCodeAt(index)
  }
  return bytes.buffer
}

export function fixtureSourceBytes(): ArrayBuffer {
  return decodeBase64(FIXTURE_PNG_BASE64)
}

export function fixtureFontBytes(): ArrayBuffer {
  // Deliberately tiny WOFF-shaped data exercises ArrayBuffer transfer and the
  // renderer's measured system-font fallback without shipping a font asset.
  return Uint8Array.of(0x77, 0x4f, 0x46, 0x46, 0, 0, 0, 0).buffer
}

export type FixtureResultInput = {
  jobId: string
  sourceSha256: string
  sourceWidth: number
  sourceHeight: number
}

export function createFixtureResult(input: FixtureResultInput): BrowserJobResult {
  return parseBrowserJobResult({
    protocolVersion: 1,
    jobId: input.jobId,
    sourceSha256: input.sourceSha256,
    sourceWidth: input.sourceWidth,
    sourceHeight: input.sourceHeight,
    cleanImageBlobId: `fixture-clean-${input.jobId}`,
    cleanImageMimeType: 'image/png',
    regions: [
      {
        id: `${input.sourceSha256.slice(0, 8)}-region-0001`,
        kind: 'dialogue',
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
        rotationDegrees: 0,
        sourceEnglish: 'We have to leave now!',
        faithfulChinese: '我们得马上离开！',
        displayedChinese: '我们现在要走！',
        pinyin: 'wǒ men xiàn zài yào zǒu',
        ocrConfidence: 0.97,
        readingOrder: 0,
        vocabulary: {
          requestedHskLevel: 2,
          strictlyValid: true,
          exceptions: [],
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
      {
        id: `${input.sourceSha256.slice(0, 8)}-region-0002`,
        kind: 'caption',
        textPolygon: [
          { x: 0.58, y: 0.66 },
          { x: 0.85, y: 0.62 },
          { x: 0.88, y: 0.78 },
          { x: 0.61, y: 0.82 },
        ],
        rotationDegrees: -8,
        sourceEnglish: 'Wait for me!',
        faithfulChinese: '等等我！',
        displayedChinese: '等我！',
        pinyin: 'děng wǒ',
        ocrConfidence: 0.93,
        readingOrder: 1,
        vocabulary: {
          requestedHskLevel: 2,
          strictlyValid: true,
          exceptions: [],
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
        },
      },
    ],
    warnings: [],
    cache: {
      detectionHit: false,
      ocrHit: false,
      inpaintHit: false,
      translationHit: false,
    },
  })
}

type FixtureStep = {
  at: number
  stage: BrowserJobStatus['stage']
  overallProgress: number
  message: string
}

const FIXTURE_STEPS: FixtureStep[] = [
  { at: 0, stage: 'queued', overallProgress: 0, message: 'Queued' },
  {
    at: 250,
    stage: 'detecting',
    overallProgress: 0.12,
    message: 'Finding text regions',
  },
  { at: 500, stage: 'ocr', overallProgress: 0.3, message: 'Reading text' },
  {
    at: 750,
    stage: 'inpainting',
    overallProgress: 0.48,
    message: 'Removing source lettering',
  },
  {
    at: 1_000,
    stage: 'hsk-validating',
    overallProgress: 0.88,
    message: 'Checking HSK vocabulary',
  },
  {
    at: 1_250,
    stage: 'complete',
    overallProgress: 1,
    message: 'Complete',
  },
]

export class FixtureService {
  constructor(private readonly now: () => number = Date.now) {}

  createJobId(pageSessionId: string, pageIndex: number, sourceSha256: string): string {
    return `fixture-${pageSessionId.slice(0, 12)}-${pageIndex}-${sourceSha256.slice(0, 12)}`
  }

  status(record: ActiveJobRecord): BrowserJobStatus {
    const elapsed = Math.max(0, this.now() - record.createdAtUnixMs)
    let stepIndex = 0
    for (let index = 0; index < FIXTURE_STEPS.length; index += 1) {
      if (elapsed >= (FIXTURE_STEPS[index]?.at ?? Number.POSITIVE_INFINITY)) {
        stepIndex = index
      }
    }
    const step = FIXTURE_STEPS[stepIndex] ?? FIXTURE_STEPS[0]
    if (!step) throw new Error('Fixture progress sequence is empty.')
    if (step.stage === 'complete') {
      return {
        revision: stepIndex + 1,
        jobId: record.jobId,
        state: 'complete',
        stage: 'complete',
        stageProgress: 1,
        overallProgress: 1,
        message: step.message,
      }
    }
    return {
      revision: stepIndex + 1,
      jobId: record.jobId,
      state: 'running',
      stage: step.stage,
      overallProgress: step.overallProgress,
      message: step.message,
    }
  }

  result(record: ActiveJobRecord, width: number, height: number): BrowserJobResult {
    return createFixtureResult({
      jobId: record.jobId,
      sourceSha256: record.sourceSha256,
      sourceWidth: width,
      sourceHeight: height,
    })
  }

  cleanImage(): ArrayBuffer {
    return fixtureSourceBytes()
  }

  lookup(request: LookupRequest): LookupResult {
    const selectedText = request.selectedText
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
              displayedChinese: '我们现在要走！',
              faithfulChinese: '我们得马上离开！',
              sourceEnglish: 'We have to leave now!',
            },
          }
        : {}),
    })
  }
}
