import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  BUILD_FINGERPRINT,
  MAX_DOCUMENT_BLOCKS,
  MAX_DOCUMENT_BLOCK_UTF8_BYTES,
  MAX_DOCUMENT_UTF8_BYTES,
  MAX_VISIBLE_BLOCK_IDS,
  parseBrowserSetupStatus,
  parseDocumentJobRequest,
  parseErrorResponse,
  parseHealthResponse,
  parseImageJobRequest,
  parseJobCreated,
  parseJobFocus,
  parseJobUpdate,
  parseJobUpdateBatch,
  parseLookupRequest,
  parseLookupResult,
  parseNativeHandshakeRequest,
  parseNativeReadyResponse,
} from '../../src/contracts/browser'
import { canonicalDocumentText } from '../../src/document'
import { createFixtureRegions } from '../support/fixture-service'

function sharedFixture(name: string): unknown {
  const path = resolve(process.cwd(), '../../fixtures/contracts', name)
  return JSON.parse(readFileSync(path, 'utf8')) as unknown
}

type RepeatedDocumentRequestFixture = {
  fixtureType: 'repeatedDocumentRequest'
  pageSessionId: string
  blockCount: number
  textSeed: string
  textRepeat: number
}

function repeatedDocumentRequestFixture(name: string) {
  const descriptor = sharedFixture(name) as RepeatedDocumentRequestFixture
  if (descriptor.fixtureType !== 'repeatedDocumentRequest') {
    throw new Error(`${name} is not a repeated document request fixture.`)
  }
  const text = descriptor.textSeed.repeat(descriptor.textRepeat)
  const blocks = Array.from({ length: descriptor.blockCount }, (_, itemOrder) => ({
    itemId: `block-${itemOrder}`,
    sourceIndex: 0 as const,
    itemOrder,
    kind: 'prose' as const,
    provenance: 'dom' as const,
    text,
  }))
  const sourceSha256 = createHash('sha256')
    .update(canonicalDocumentText(blocks), 'utf8')
    .digest('hex')
  return {
    buildFingerprint: BUILD_FINGERPRINT,
    pageSessionId: descriptor.pageSessionId,
    sourceSha256,
    settings: {
      sourceLanguage: 'en' as const,
      targetLanguage: 'zh-CN' as const,
      hskStandard: '2.0' as const,
      hskLevel: 3 as const,
      learningMode: 'natural' as const,
    },
    blocks,
  }
}

type RepeatedDocumentFocusFixture = {
  fixtureType: 'repeatedDocumentFocus'
  visibleBlockCount: number
  itemIdPrefix: string
  active: boolean
}

function repeatedDocumentFocusFixture(name: string) {
  const descriptor = sharedFixture(name) as RepeatedDocumentFocusFixture
  if (descriptor.fixtureType !== 'repeatedDocumentFocus') {
    throw new Error(`${name} is not a repeated document focus fixture.`)
  }
  return {
    kind: 'document' as const,
    visibleBlockIds: Array.from(
      { length: descriptor.visibleBlockCount },
      (_, index) => `${descriptor.itemIdPrefix}${index}`,
    ),
    active: descriptor.active,
  }
}

function imageRequest() {
  return {
    buildFingerprint: BUILD_FINGERPRINT,
    clientImageId: 'page-0-hash',
    sourceSha256: 'a'.repeat(64),
    sourceMimeType: 'image/png',
    naturalWidth: 1200,
    naturalHeight: 1800,
    pageSessionId: 'page',
    sourceIndex: 0,
    chapterSourceOrder: [0],
    surfaceKind: 'image',
    visibleRects: [{ x: 0, y: 0.2, width: 1, height: 0.4 }],
    readingDirection: 'ltr',
    settings: {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel: 5,
      learningMode: 'natural',
    },
  } as const
}

function documentRequest() {
  return {
    buildFingerprint: BUILD_FINGERPRINT,
    pageSessionId: 'chapter-document',
    sourceSha256: 'b'.repeat(64),
    settings: {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel: 3,
      learningMode: 'strict',
    },
    blocks: [
      {
        itemId: 'heading-0',
        sourceIndex: 0,
        itemOrder: 0,
        kind: 'heading',
        provenance: 'dom',
        text: 'Chapter One',
      },
      {
        itemId: 'prose-1',
        sourceIndex: 0,
        itemOrder: 1,
        kind: 'prose',
        provenance: 'dom',
        text: 'The rain stopped before dawn.',
      },
    ],
  } as const
}

function ready() {
  return {
    type: 'ready',
    buildFingerprint: BUILD_FINGERPRINT,
    engineVersion: '0.2.0',
    port: 43127,
    token: 'A'.repeat(43),
    sessionExpiresAtUnixMs: 2_000_000,
    capabilities: {
      sourceLanguages: ['en'],
      targetLanguages: ['zh-CN'],
      hskLevels: [1, 2, 3, 4, 5, 6],
      modelsReady: true,
    },
  }
}

describe('unversioned browser contract', () => {
  it('parses every shared image and document fixture without adaptation', () => {
    expect(parseImageJobRequest(sharedFixture('job-request.valid.json')).sourceIndex).toBe(0)
    expect(
      parseDocumentJobRequest(sharedFixture('document-job-request.valid.json')).blocks,
    ).toHaveLength(2)
    expect(parseJobCreated(sharedFixture('job-created.valid.json')).jobId).toBe('fixture-job-0001')
    expect(parseJobFocus(sharedFixture('focus-image.valid.json'))).toMatchObject({
      kind: 'image',
      active: true,
    })
    expect(parseJobFocus(sharedFixture('focus-document.valid.json'))).toMatchObject({
      kind: 'document',
      visibleBlockIds: ['block-0', 'block-1'],
    })
    expect(
      parseJobUpdateBatch(sharedFixture('job-updates.success.json')).updates.map(
        ({ type }) => type,
      ),
    ).toEqual(['progress', 'imageRegionReady', 'complete'])
    expect(
      parseJobUpdateBatch(sharedFixture('document-updates.success.json')).updates.map(
        ({ type }) => type,
      ),
    ).toEqual(['documentBlockReady', 'documentBlockPreserved', 'complete'])
    expect(
      parseJobUpdateBatch(sharedFixture('job-updates.failure.json')).updates.at(-1)?.type,
    ).toBe('failed')
    expect(
      parseJobUpdateBatch(sharedFixture('job-updates.cancelled.json')).updates.at(-1)?.type,
    ).toBe('cancelled')
    expect(parseNativeHandshakeRequest(sharedFixture('native-request.valid.json')).type).toBe(
      'start-or-discover-daemon',
    )
    expect(parseNativeReadyResponse(sharedFixture('native-ready.valid.json')).type).toBe('ready')
    expect(parseHealthResponse(sharedFixture('health.ready.json')).resourceIdentities).toHaveLength(
      13,
    )
    expect(parseBrowserSetupStatus(sharedFixture('setup.ready.json'))).toMatchObject({
      state: 'ready',
      modelId: 'qwen3.5-4b',
    })
    expect(parseLookupResult(sharedFixture('lookup.valid.json')).item?.sourceText).toBe(
      'We have to leave now!',
    )
    expect(parseErrorResponse(sharedFixture('error.valid.json')).code).toBe('FIXTURE_ERROR')
  })

  it('rejects every shared document size and visible-focus bound fixture', () => {
    const totalBytes = repeatedDocumentRequestFixture(
      'invalid/document-job-request.total-bytes.descriptor.json',
    )
    expect(
      new TextEncoder().encode(JSON.stringify(totalBytes)).byteLength,
    ).toBeGreaterThan(MAX_DOCUMENT_UTF8_BYTES)
    expect(() => parseDocumentJobRequest(totalBytes)).toThrow(/1048576/u)

    const blockCount = repeatedDocumentRequestFixture(
      'invalid/document-job-request.block-count.descriptor.json',
    )
    expect(blockCount.blocks).toHaveLength(MAX_DOCUMENT_BLOCKS + 1)
    expect(() => parseDocumentJobRequest(blockCount)).toThrow(/at most 2000/u)

    const blockBytes = repeatedDocumentRequestFixture(
      'invalid/document-job-request.block-bytes.descriptor.json',
    )
    expect(
      new TextEncoder().encode(blockBytes.blocks[0]!.text).byteLength,
    ).toBeGreaterThan(MAX_DOCUMENT_BLOCK_UTF8_BYTES)
    expect(() => parseDocumentJobRequest(blockBytes)).toThrow(/UTF-8 bytes/u)

    const focus = repeatedDocumentFocusFixture(
      'invalid/focus-document.visible-block-count.descriptor.json',
    )
    expect(focus.visibleBlockIds).toHaveLength(MAX_VISIBLE_BLOCK_IDS + 1)
    expect(() => parseJobFocus(focus)).toThrow(/at most 64/u)
  })

  it('rejects every shared tagged focus and update modality mismatch fixture', () => {
    for (const name of [
      'invalid/focus-document.image-fields.json',
      'invalid/focus-image.document-fields.json',
    ]) {
      expect(() => parseJobFocus(sharedFixture(name))).toThrow()
    }
    for (const name of [
      'invalid/job-updates.document-tag-image-field.json',
      'invalid/job-updates.image-tag-document-field.json',
    ]) {
      expect(() => parseJobUpdateBatch(sharedFixture(name))).toThrow()
    }
  })

  it('keeps image-only fields out of document requests and document blocks authoritative', () => {
    expect(parseImageJobRequest(imageRequest())).toMatchObject({
      readingDirection: 'ltr',
      sourceIndex: 0,
    })
    expect(parseDocumentJobRequest(documentRequest()).blocks[1]).toMatchObject({
      provenance: 'dom',
      kind: 'prose',
    })
    expect(() => parseImageJobRequest({ ...imageRequest(), blocks: [] })).toThrow(/blocks/)
    expect(() =>
      parseDocumentJobRequest({ ...documentRequest(), readingDirection: 'ltr' }),
    ).toThrow(/readingDirection/)
    expect(() =>
      parseDocumentJobRequest({
        ...documentRequest(),
        blocks: [{ ...documentRequest().blocks[0], provenance: 'ocr' }],
      }),
    ).toThrow(/provenance/)
  })

  it('enforces document ordering, identity, and input limits', () => {
    const request = documentRequest()
    expect(() =>
      parseDocumentJobRequest({ ...request, blocks: [request.blocks[1], request.blocks[0]] }),
    ).toThrow(/ordered/)
    expect(() =>
      parseDocumentJobRequest({
        ...request,
        blocks: [request.blocks[0], { ...request.blocks[1], itemId: request.blocks[0].itemId }],
      }),
    ).toThrow(/unique/)
    expect(() =>
      parseDocumentJobRequest({
        ...request,
        blocks: [
          { ...request.blocks[0], text: '\u00e9'.repeat(MAX_DOCUMENT_BLOCK_UTF8_BYTES / 2 + 1) },
        ],
      }),
    ).toThrow(/UTF-8 bytes/)
    expect(() =>
      parseDocumentJobRequest({
        ...request,
        blocks: [{ ...request.blocks[0], text: ' Chapter One ' }],
      }),
    ).toThrow(/normalized/)
    expect(() =>
      parseDocumentJobRequest({
        ...request,
        blocks: [{ ...request.blocks[0], sourceIndex: 0x1_0000_0000 }],
      }),
    ).toThrow(/4294967295/)
    const block = request.blocks[0]
    expect(() =>
      parseDocumentJobRequest({
        ...request,
        blocks: Array.from({ length: MAX_DOCUMENT_BLOCKS + 1 }, (_, itemOrder) => ({
          ...block,
          itemId: `block-${itemOrder}`,
          itemOrder,
        })),
      }),
    ).toThrow(/at most 2000/)
  })

  it('parses tagged focus and rejects modality fields and bounds', () => {
    expect(parseJobFocus({ kind: 'image', visibleRects: [], active: false })).toEqual({
      kind: 'image',
      visibleRects: [],
      active: false,
    })
    expect(parseJobFocus({ kind: 'document', visibleBlockIds: ['block-1'], active: true })).toEqual(
      { kind: 'document', visibleBlockIds: ['block-1'], active: true },
    )
    expect(() => parseJobFocus({ kind: 'image', visibleBlockIds: [], active: true })).toThrow(
      /visibleBlockIds/,
    )
    expect(() => parseJobFocus({ kind: 'document', visibleRects: [], active: true })).toThrow(
      /visibleRects/,
    )
    expect(() =>
      parseJobFocus({
        kind: 'image',
        visibleRects: [{ x: 0.8, y: 0, width: 0.3, height: 1 }],
        active: true,
      }),
    ).toThrow(/source/)
    expect(() =>
      parseJobFocus({
        kind: 'document',
        visibleBlockIds: Array.from(
          { length: MAX_VISIBLE_BLOCK_IDS + 1 },
          (_, index) => `block-${index}`,
        ),
        active: true,
      }),
    ).toThrow(/at most 64/)
    expect(() =>
      parseJobFocus({ kind: 'document', visibleBlockIds: ['same', 'same'], active: true }),
    ).toThrow(/unique/)
  })

  it('parses final-only image payloads and terminal counts', () => {
    const region = createFixtureRegions({
      jobId: 'fixture-job',
      sourceSha256: 'a'.repeat(64),
      sourceWidth: 1200,
      sourceHeight: 1800,
    })[0]
    if (!region) throw new Error('Fixture region is missing.')
    region.style.colorBands = [
      { position: 0.25, foreground: '#111111' },
      { position: 0.75, foreground: '#2580df', outlineColor: '#ffffff' },
    ]
    const batch = parseJobUpdateBatch({
      jobId: 'fixture-job',
      nextSequence: 3,
      updates: [
        {
          sequence: 1,
          type: 'progress',
          stage: 'ocr',
          overallProgress: 0.3,
          message: 'Reading text',
        },
        { sequence: 2, type: 'imageRegionReady', region },
        {
          sequence: 3,
          type: 'complete',
          translatedCount: 1,
          preservedCount: 0,
          message: 'Complete',
        },
      ],
    })
    expect(batch.updates.map(({ type }) => type)).toEqual([
      'progress',
      'imageRegionReady',
      'complete',
    ])
    const installed = batch.updates.find((update) => update.type === 'imageRegionReady')
    expect(
      installed?.type === 'imageRegionReady' && installed.region.style.colorBands,
    ).toHaveLength(2)
    expect(() =>
      parseJobUpdate({
        sequence: 1,
        type: 'imageRegionReady',
        region: {
          ...region,
          text: { ...region.text, hsk: { ...region.text.hsk, repairState: 'rejected' } },
        },
      }),
    ).toThrow(/rejected repair/)
    expect(() =>
      parseJobUpdate({
        sequence: 1,
        type: 'imageRegionReady',
        region: { ...region, text: { ...region.text, pinyin: '' } },
      }),
    ).toThrow(/pinyin/)
    expect(() => parseJobUpdate({ sequence: 1, type: 'complete' })).toThrow(/translatedCount/)
  })

  it('parses document ready/preserved payloads and rejects cross-modality fields', () => {
    const readyUpdate = (sharedFixture('document-updates.success.json') as { updates: unknown[] })
      .updates[0]
    expect(parseJobUpdate(readyUpdate)).toMatchObject({
      type: 'documentBlockReady',
      block: { itemId: 'block-0', sourceIndex: 0 },
    })
    expect(() => parseJobUpdate({ ...(readyUpdate as object), region: {} })).toThrow(/region/)
    expect(() =>
      parseJobUpdate({
        sequence: 1,
        type: 'documentBlockPreserved',
        block: {
          itemId: 'block-0',
          sourceIndex: 0,
          itemOrder: 0,
          kind: 'prose',
          provenance: 'dom',
          sourceText: 'Original',
          reason: 'preserved',
        },
      }),
    ).toThrow(/provenance/)
  })

  it('accepts build-matched health, setup, lookup, and errors', () => {
    expect(parseNativeReadyResponse(ready()).buildFingerprint).toBe(BUILD_FINGERPRINT)
    expect(
      parseBrowserSetupStatus({
        state: 'warming',
        modelId: 'qwen3.5-4b',
        message: 'Hskify is getting ready.',
      }).state,
    ).toBe('warming')
    expect(
      parseLookupRequest({
        interaction: 'hover',
        characterOffset: 2,
        jobId: 'job-1',
        itemId: 'item-1',
      }).interaction,
    ).toBe('hover')
    expect(
      parseLookupRequest({ interaction: 'selection', selectedText: '\u7814\u7a76\u751f' })
        .interaction,
    ).toBe('selection')
    expect(
      parseLookupResult({
        selectedText: '\u79bb\u5f00',
        tokens: [
          {
            simplified: '\u79bb\u5f00',
            pinyin: 'lí kāi',
            definitions: ['leave'],
            hskLevel: 2,
            properName: false,
          },
        ],
        item: {
          displayedChinese: '\u6211\u4eec\u73b0\u5728\u8981\u8d70\uff01',
          baseChinese: '\u6211\u4eec\u5f97\u9a6c\u4e0a\u79bb\u5f00\uff01',
          sourceText: 'We have to leave now!',
        },
      }).item?.sourceText,
    ).toBe('We have to leave now!')
    expect(
      parseErrorResponse({ code: 'IMAGE_TOO_LARGE', message: 'Too large', retryable: false }).code,
    ).toBe('IMAGE_TOO_LARGE')
  })

  it('rejects removed protocol fields, build mismatches, invalid hashes, and replay gaps', () => {
    expect(() => parseImageJobRequest({ ...imageRequest(), protocolVersion: 1 })).toThrow(
      /protocolVersion/,
    )
    expect(() => parseJobCreated({ buildFingerprint: 'other-build', jobId: 'job' })).toThrow(
      /buildFingerprint/,
    )
    expect(() => parseImageJobRequest({ ...imageRequest(), sourceSha256: 'A'.repeat(64) })).toThrow(
      /lowercase SHA-256/,
    )
    expect(() => parseImageJobRequest({ ...imageRequest(), chapterSourceOrder: [1] })).toThrow(
      /sourceIndex/,
    )
    expect(() => parseLookupRequest({ interaction: 'hover', characterOffset: 0 })).toThrow(/jobId/)
    expect(() =>
      parseLookupRequest({ interaction: 'selection', selectedText: 'word', itemId: 'item-only' }),
    ).toThrow(/paired/)
    expect(() =>
      parseJobUpdateBatch(
        {
          jobId: 'job',
          nextSequence: 4,
          updates: [{ sequence: 3, type: 'complete', translatedCount: 1, preservedCount: 0 }],
        },
        3,
      ),
    ).toThrow(/sequence/)
    expect(() =>
      parseJobUpdateBatch(
        {
          jobId: 'job',
          nextSequence: 3,
          updates: [{ sequence: 3, type: 'complete', translatedCount: 1, preservedCount: 0 }],
        },
        1,
      ),
    ).toThrow(/contiguous/)
  })

  it('accepts the native 1,024-update response bound and rejects one more', () => {
    const updates = Array.from({ length: 1_024 }, (_, index) => ({
      sequence: index + 1,
      type: 'progress' as const,
      stage: 'registering' as const,
      message: 'Registered',
    }))
    expect(parseJobUpdateBatch({
      jobId: 'maximal-document-replay',
      nextSequence: 1_024,
      updates,
    }).updates).toHaveLength(1_024)
    expect(() => parseJobUpdateBatch({
      jobId: 'oversized-document-replay',
      nextSequence: 1_025,
      updates: [
        ...updates,
        { sequence: 1_025, type: 'progress', stage: 'registering', message: 'Registered' },
      ],
    })).toThrow(/at most 1024/u)
  })
})
