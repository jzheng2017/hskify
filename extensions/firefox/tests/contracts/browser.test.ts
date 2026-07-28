import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  BUILD_FINGERPRINT,
  parseBrowserJobCreated,
  parseBrowserJobRequest,
  parseBrowserSetupStatus,
  parseErrorResponse,
  parseHealthResponse,
  parseJobUpdateBatch,
  parseLookupRequest,
  parseLookupResult,
  parseNativeHandshakeRequest,
  parseNativeReadyResponse,
  parseViewportUpdate,
} from '../../src/contracts/browser'
import { createFixtureRegions } from '../../src/messaging/fixture-service'

function sharedFixture(name: string): unknown {
  const path = resolve(process.cwd(), '../../fixtures/contracts', name)
  return JSON.parse(readFileSync(path, 'utf8')) as unknown
}

function pinnedResourceIdentities(): unknown[] {
  return (sharedFixture('health.ready.json') as { resourceIdentities: unknown[] })
    .resourceIdentities
}

function jobRequest() {
  return {
    buildFingerprint: BUILD_FINGERPRINT,
    clientImageId: 'page-0-hash',
    sourceSha256: 'a'.repeat(64),
    sourceMimeType: 'image/png',
    naturalWidth: 1200,
    naturalHeight: 1800,
    pageSessionId: 'page',
    pageIndex: 0,
    visibleRects: [{ x: 0, y: 0.2, width: 1, height: 0.4 }],
    properNameGlossary: [{ sourceEnglish: 'Cheon Yeo Woon', chinese: '天汝云' }],
    settings: {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel: 5,
      readingDirection: 'auto',
      translateSoundEffects: false,
      nameTranslation: 'keep-original',
    },
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

describe('unversioned progressive browser contract', () => {
  it('parses the repository-wide companion contract fixtures without adaptation', () => {
    expect(parseBrowserJobRequest(sharedFixture('job-request.valid.json')).buildFingerprint).toBe(
      BUILD_FINGERPRINT,
    )
    expect(parseBrowserJobCreated(sharedFixture('job-created.valid.json')).jobId).toBe(
      'fixture-job-0001',
    )
    expect(parseViewportUpdate(sharedFixture('viewport.valid.json')).active).toBe(true)
    expect(
      parseJobUpdateBatch(sharedFixture('job-updates.success.json')).updates.map(
        (update) => update.type,
      ),
    ).toEqual(['progress', 'regionReady', 'regionRefined', 'complete'])
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
      10,
    )
    expect(parseBrowserSetupStatus(sharedFixture('setup.ready.json'))).toMatchObject({
      state: 'ready',
      modelId: 'qwen3.5-4b',
    })
    expect(parseLookupResult(sharedFixture('lookup.valid.json')).tokens).toHaveLength(1)
    expect(parseErrorResponse(sharedFixture('error.valid.json')).code).toBe('FIXTURE_ERROR')
  })

  it('accepts job creation with visible source rectangles and a hard build match', () => {
    expect(parseBrowserJobRequest(jobRequest())).toMatchObject({
      buildFingerprint: BUILD_FINGERPRINT,
      visibleRects: [{ y: 0.2, height: 0.4 }],
      properNameGlossary: [{ sourceEnglish: 'Cheon Yeo Woon', chinese: '天汝云' }],
    })
    expect(
      parseBrowserJobCreated({
        buildFingerprint: BUILD_FINGERPRINT,
        jobId: 'fixture-job',
      }),
    ).toEqual({ buildFingerprint: BUILD_FINGERPRINT, jobId: 'fixture-job' })
    expect(() =>
      parseBrowserJobRequest({
        ...jobRequest(),
        settings: { ...jobRequest().settings, nameTranslation: 'literal' },
      }),
    ).toThrow(/nameTranslation/i)
  })

  it('parses monotonic progressive region, refinement, and terminal updates', () => {
    const region = createFixtureRegions({
      jobId: 'fixture-job',
      sourceSha256: 'a'.repeat(64),
      sourceWidth: 1200,
      sourceHeight: 1800,
    })[0]
    if (region) {
      region.style.colorBands = [
        { position: 0.25, foreground: '#111111' },
        { position: 0.75, foreground: '#2580df', outlineColor: '#ffffff' },
      ]
    }
    const batch = parseJobUpdateBatch({
      jobId: 'fixture-job',
      nextSequence: 4,
      updates: [
        {
          sequence: 1,
          type: 'progress',
          stage: 'ocr',
          overallProgress: 0.3,
          message: 'Reading text',
        },
        { sequence: 2, type: 'regionReady', region },
        {
          sequence: 3,
          type: 'regionRefined',
          regionId: region?.id,
          displayedChinese: '我们现在就走！',
          pinyin: 'wǒ men xiàn zài jiù zǒu',
          hsk: {
            requestedLevel: 2,
            strictlyValid: true,
            aboveLevelTokens: [],
            repairState: 'accepted',
          },
        },
        { sequence: 4, type: 'complete', message: 'Complete' },
      ],
    })
    expect(batch.updates.map((update) => update.type)).toEqual([
      'progress',
      'regionReady',
      'regionRefined',
      'complete',
    ])
    expect(batch.nextSequence).toBe(4)
    const readyRegion = batch.updates.find((update) => update.type === 'regionReady')
    expect(readyRegion?.type === 'regionReady' && readyRegion.region.style.colorBands).toHaveLength(
      2,
    )
  })

  it('accepts build-matched native health, setup, lookup, and errors', () => {
    expect(
      parseNativeHandshakeRequest({
        type: 'start-or-discover-daemon',
        buildFingerprint: BUILD_FINGERPRINT,
        extensionVersion: '0.1.0',
        extensionOrigin: 'moz-extension://fixture',
      }).type,
    ).toBe('start-or-discover-daemon')
    expect(parseNativeReadyResponse(ready()).buildFingerprint).toBe(BUILD_FINGERPRINT)
    expect(
      parseHealthResponse({
        buildFingerprint: BUILD_FINGERPRINT,
        engineVersion: '0.2.0',
        status: 'ready',
        setupState: 'ready',
        resourceIdentities: pinnedResourceIdentities(),
      }).status,
    ).toBe('ready')
    expect(
      parseBrowserSetupStatus({
        state: 'ready',
        modelId: 'qwen3.5-4b',
        message: 'Ready',
      }).state,
    ).toBe('ready')
    expect(
      parseLookupRequest({
        interaction: 'hover',
        characterOffset: 2,
        jobId: 'job-1',
        regionId: 'region-1',
      }).interaction,
    ).toBe('hover')
    expect(
      parseLookupRequest({
        interaction: 'selection',
        selectedText: '研究生',
      }).interaction,
    ).toBe('selection')
    expect(
      parseLookupResult({
        selectedText: '离开',
        tokens: [
          {
            simplified: '离开',
            pinyin: 'lí kāi',
            definitions: ['leave'],
            hskLevel: 2,
            properName: false,
          },
        ],
        region: {
          displayedChinese: '我们现在要走！',
          baseChinese: '我们得马上离开！',
          sourceEnglish: 'We have to leave now!',
        },
      }).region?.baseChinese,
    ).toBe('我们得马上离开！')
    expect(
      parseErrorResponse({ code: 'IMAGE_TOO_LARGE', message: 'Too large', retryable: false }).code,
    ).toBe('IMAGE_TOO_LARGE')
  })

  it('rejects removed protocol fields, build mismatches, invalid geometry, and replayed updates', () => {
    expect(() => parseBrowserJobRequest({ ...jobRequest(), protocolVersion: 1 })).toThrow(
      /protocolVersion/,
    )
    expect(() => parseBrowserJobCreated({ buildFingerprint: 'other-build', jobId: 'job' })).toThrow(
      /buildFingerprint/,
    )
    expect(() =>
      parseLookupRequest({
        interaction: 'hover',
        characterOffset: 0,
      }),
    ).toThrow(/jobId/)
    expect(() =>
      parseBrowserJobRequest({
        ...jobRequest(),
        visibleRects: [{ x: 0.8, y: 0, width: 0.3, height: 1 }],
      }),
    ).toThrow(/image width/)
    expect(() =>
      parseJobUpdateBatch(
        {
          jobId: 'job',
          nextSequence: 4,
          updates: [{ sequence: 3, type: 'complete' }],
        },
        3,
      ),
    ).toThrow(/sequence/)

    const region = createFixtureRegions({
      jobId: 'fixture-job',
      sourceSha256: 'a'.repeat(64),
      sourceWidth: 1200,
      sourceHeight: 1800,
    })[0]
    const disconnected = structuredClone(region)
    if (!disconnected) throw new Error('fixture region is required')
    disconnected.patch.rect = { x: 0.85, y: 0.85, width: 0.1, height: 0.1 }
    expect(() =>
      parseJobUpdateBatch({
        jobId: 'job',
        nextSequence: 1,
        updates: [{ sequence: 1, type: 'regionReady', region: disconnected }],
      }),
    ).toThrow(/must overlap/)
  })

  it('requires exact, lowercase, sorted resident resource identities', () => {
    const health = {
      buildFingerprint: BUILD_FINGERPRINT,
      engineVersion: '0.2.0',
      status: 'ready',
      setupState: 'ready',
      resourceIdentities: pinnedResourceIdentities(),
    }
    expect(parseHealthResponse(health).resourceIdentities.map(({ id }) => id)).toEqual([
      'comic-text-bubble-detector-config',
      'comic-text-bubble-detector-preprocessor-config',
      'comic-text-bubble-detector-weights',
      'lama-manga-inpainter-weights',
      'manga-text-segmentation-weights',
      'pp-ocr-v5-english-recognizer-config',
      'pp-ocr-v5-english-recognizer-model',
      'speech-bubble-segmentation-config',
      'speech-bubble-segmentation-weights',
      'translation-model',
    ])
    expect(() =>
      parseHealthResponse({
        ...health,
        resourceIdentities: [...pinnedResourceIdentities()].reverse(),
      }),
    ).toThrow(/sorted/)
    expect(() =>
      parseHealthResponse({
        ...health,
        resourceIdentities: pinnedResourceIdentities().map((identity, index) =>
          index === 0
            ? {
                ...(identity as Record<string, unknown>),
                repositoryRevision: '16E8A622F91FABC6B5B65C96D32D1183F8843546',
              }
            : identity,
        ),
      }),
    ).toThrow(/repositoryRevision/)
  })
})
