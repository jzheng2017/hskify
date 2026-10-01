import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeBrowser } from 'wxt/testing'

import {
  BUILD_FINGERPRINT,
  type DocumentBlockReadyJobUpdate,
  type DocumentJobRequest,
  type JobUpdateBatch,
} from '../../src/contracts/browser'
import {
  DocumentReader,
  type DocumentChapter,
  type DocumentStructureItem,
} from '../../src/document'
import {
  ActiveJobStore,
  PageArtifactStore,
  type ActiveJobRecord,
  type PageArtifactRecord,
} from '../../src/messaging/active-jobs'
import { BackgroundRouter } from '../../src/messaging/background'
import type { CompanionClient } from '../../src/messaging/companion-client'
import { FixtureService } from '../support/fixture-service'

const pageUrl = 'https://reader.test/chapter'
const visibleRects = [{ x: 0, y: 0.1, width: 1, height: 0.4 }]
const imageSettings = {
  sourceLanguage: 'en' as const,
  targetLanguage: 'zh-CN' as const,
  hskStandard: '2.0' as const,
  hskLevel: 5 as const,
  learningMode: 'natural' as const,
}
const documentSettings = {
  sourceLanguage: 'en' as const,
  targetLanguage: 'zh-CN' as const,
  hskStandard: '2.0' as const,
  hskLevel: 3 as const,
  learningMode: 'natural' as const,
}

function sender(url = pageUrl, tabId = 7) {
  return { tab: { id: tabId }, frameId: 0, url } as browser.runtime.MessageSender
}

function submitImageMessage(pageSessionId = 'fixture-page-session') {
  return {
    type: 'job:submit-image' as const,
    clientRequestId: 'fixture-image-request',
    retryItemIds: [],
    pageSessionId,
    sourceIndex: 0,
    chapterSourceOrder: [0],
    surfaceKind: 'image' as const,
    imageUrl: 'https://reader.test/panel.svg',
    pageUrl,
    naturalWidth: 1200,
    naturalHeight: 1800,
    hskLevel: 5 as const,
    learningMode: 'natural' as const,
    readingDirection: 'ltr' as const,
    visibleRects,
  }
}

function submitDocumentMessage(pageSessionId = 'fixture-document-session') {
  return {
    type: 'job:submit-document' as const,
    pageUrl,
    request: {
      buildFingerprint: BUILD_FINGERPRINT,
      clientRequestId: 'fixture-document',
      retryItemIds: [],
      pageSessionId,
      focus: { kind: 'document' as const, active: true, visibleBlockIds: [] },
      sourceSha256: 'b'.repeat(64),
      settings: documentSettings,
      blocks: [
        {
          parentBlockId: 'block-0',
          subItemOrder: 0,
          itemId: 'block-0',
          sourceIndex: 0,
          itemOrder: 0,
          kind: 'heading' as const,
          provenance: 'dom' as const,
          text: 'Chapter One',
        },
        {
          parentBlockId: 'block-1',
          subItemOrder: 0,
          itemId: 'block-1',
          sourceIndex: 0,
          itemOrder: 1,
          kind: 'prose' as const,
          provenance: 'dom' as const,
          text: 'The rain stopped before dawn.',
        },
      ],
    },
  }
}

function imageCandidate(sourceUrl = 'https://reader.test/panel.svg') {
  return {
    kind: 'image' as const,
    sourceUrl,
    naturalWidth: 1200,
    naturalHeight: 1800,
    sourceIndex: 0,
    settings: imageSettings,
    readingDirection: 'ltr' as const,
  }
}

function documentCandidate(sourceSha256 = 'b'.repeat(64)) {
  return { kind: 'document' as const, sourceSha256, settings: documentSettings }
}

function activeDocumentRecord(
  jobId: string,
  pageSessionId: string,
  itemIds: string[] = [],
): ActiveJobRecord {
  return {
    tabId: 7,
    frameId: 0,
    pageSessionId,
    pageUrl,
    jobId,
    sourceSha256: 'd'.repeat(64),
    source: {
      kind: 'document',
      settings: documentSettings,
      blockCount: 1,
      uploadedBytes: 128,
    },
    submittedAtUnixMs: 1_000,
    acknowledgedSequence: 0,
    deliveredSequence: 1,
    itemIds,
    patchIds: [],
    fontIds: [],
    createdAtUnixMs: 1_000,
  }
}

function documentArtifact(jobId: string, pageSessionId: string): PageArtifactRecord {
  return {
    tabId: 7,
    frameId: 0,
    pageSessionId,
    pageUrl,
    jobId,
    sourceSha256: 'e'.repeat(64),
    source: { kind: 'document' },
    itemIds: [`${jobId}-block`],
    patchIds: [],
    fontIds: [],
    createdAtUnixMs: 1_000,
  }
}

function documentReader(request: DocumentJobRequest, attachTranslatedText: () => () => void) {
  const sourceRoot = document.createElement('article')
  const sourceElements = new Map<string, HTMLElement>()
  const snapshotBlocks = request.blocks.map((block) => ({ ...block, sourceIndex: 0 as const }))
  const structure: DocumentStructureItem[] = request.blocks.map((block, order) => {
    const element = document.createElement(block.kind === 'heading' ? 'h1' : 'p')
    element.textContent = block.text
    sourceRoot.append(element)
    sourceElements.set(block.itemId, element)
    return block.kind === 'heading'
      ? {
          type: 'text',
          parentBlockId: block.itemId,
          subItemOrder: 0,
          itemId: block.itemId,
          order,
          kind: 'title',
          headingLevel: 1,
          text: block.text,
        }
      : {
          type: 'text',
          parentBlockId: block.itemId,
          subItemOrder: 0,
          itemId: block.itemId,
          order,
          kind: 'paragraph',
          text: block.text,
        }
  })
  document.body.append(sourceRoot)
  const chapter: DocumentChapter = {
    sourceRoot,
    sourceElements,
    sourceRevisions: new Map(
      [...sourceElements.values()].map((element) => [
        element.firstChild as Text,
        (element.firstChild as Text).data,
      ]),
    ),
    sourceSlots: new Map(
      [...sourceElements].map(([id, element]) => [
        id,
        [{ node: element.firstChild as Text, start: 0, end: (element.firstChild as Text).length }],
      ]),
    ),
    structure,
    snapshot: {
      sourceUrl: pageUrl,
      sourceSha256: request.sourceSha256,
      title: 'Chapter One',
      characterCount: request.blocks.reduce((total, block) => total + [...block.text].length, 0),
      blocks: snapshotBlocks,
    },
  }
  return new DocumentReader(chapter, { attachTranslatedText })
}

describe('progressive background fixture adapter', () => {
  let now = 1_000

  beforeEach(() => {
    fakeBrowser.reset()
    vi.stubGlobal('browser', fakeBrowser)
    document.body.replaceChildren()
    now = 1_000
  })

  afterEach(() => {
    document.body.replaceChildren()
    vi.unstubAllGlobals()
  })

  it('persists image acknowledgement across MV3 reconstruction and streams image assets', async () => {
    const first = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    const submitted = (await first.route(submitImageMessage(), sender())) as {
      jobId: string
      sourceSha256: string
      acknowledgedSequence: number
    }
    expect(submitted).toMatchObject({ kind: 'image', sourceIndex: 0, acknowledgedSequence: 0 })
    expect(submitted.jobId).toMatch(/^fixture-/)

    await expect(
      first.route(
        { type: 'job:updates', jobId: submitted.jobId, after: 0 },
        sender('https://reader.test/other'),
      ),
    ).rejects.toMatchObject({ code: 'DOCUMENT_IDENTITY_MISMATCH' })

    now = 1_600
    const firstBatch = (await first.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )) as {
      nextSequence: number
      updates: Array<{ type: string; region?: { patch: { blobId: string } } }>
    }
    expect(firstBatch.updates.map(({ type }) => type)).toEqual([
      'progress',
      'progress',
      'imageRegionReady',
    ])
    const patchId = firstBatch.updates.at(-1)?.region?.patch.blobId
    if (!patchId) throw new Error('Fixture patch update missing.')
    const patch = (await first.route(
      { type: 'job:patch', jobId: submitted.jobId, patchId, mimeType: 'image/png' },
      sender(),
    )) as { patchId: string; bytes: ArrayBuffer }
    expect(patch).toMatchObject({ patchId })
    expect(patch.bytes).toBeInstanceOf(ArrayBuffer)
    expect(await new PageArtifactStore().get(submitted.jobId)).toBeUndefined()

    await first.route(
      { type: 'job:ack', jobId: submitted.jobId, sequence: firstBatch.nextSequence },
      sender(),
    )

    const restarted = new BackgroundRouter({
      fixture: new FixtureService(() => now),
      now: () => now,
    })
    const recovered = (await restarted.route(
      {
        type: 'jobs:recover',
        pageSessionId: 'fixture-page-session',
        pageUrl,
        candidates: [imageCandidate()],
      },
      sender(),
    )) as Array<{ jobId: string; acknowledgedSequence: number }>
    expect(recovered).toEqual([
      expect.objectContaining({ kind: 'image', jobId: submitted.jobId, acknowledgedSequence: 0 }),
    ])

    now = 2_500
    const finalBatch = (await restarted.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )) as { nextSequence: number; updates: Array<{ type: string }> }
    expect(finalBatch.updates.map(({ type }) => type)).toEqual([
      'progress',
      'progress',
      'imageRegionReady',
      'imageRegionReady',
      'complete',
    ])
    await restarted.route(
      {
        type: 'job:ack',
        jobId: submitted.jobId,
        sequence: finalBatch.nextSequence,
        terminalType: 'complete',
      },
      sender(),
    )
    expect(await new ActiveJobStore().get(submitted.jobId)).toBeUndefined()
    expect(await new PageArtifactStore().get(submitted.jobId)).toBeDefined()

    const itemId = `${submitted.sourceSha256.slice(0, 8)}-region-0001`
    const lookup = (await restarted.route(
      {
        type: 'dictionary:lookup',
        request: {
          interaction: 'selection',
          selectedText: '\u79bb\u5f00',
          context: {
            displayedChinese: '我们现在就走！',
            baseChinese: '我们得马上离开！',
            sourceText: 'We have to leave now!',
            properNames: [],
          },
          itemId,
        },
      },
      sender(),
    )) as { item?: { baseChinese: string } }
    expect(lookup.item?.baseChinese).toBe('\u6211\u4eec\u5f97\u9a6c\u4e0a\u79bb\u5f00\uff01')
    const font = (await restarted.route(
      { type: 'font:get', jobId: submitted.jobId, fontId: 'fixture-sans' },
      sender(),
    )) as { bytes: ArrayBuffer }
    expect(font.bytes).toBeInstanceOf(ArrayBuffer)
    expect(
      await restarted.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'fixture-page-session',
          pageUrl,
          candidates: [imageCandidate()],
        },
        sender(),
      ),
    ).toEqual([])
  })

  it('streams and recovers document blocks with the exact document hash', async () => {
    const fixture = new FixtureService(() => now)
    const first = new BackgroundRouter({ fixture, now: () => now })
    await first.route(
      {
        type: 'chapter:start',
        pageSessionId: 'fixture-document-session',
        pageUrl,
        contentKind: 'document',
      },
      sender(),
    )
    const submitted = (await first.route(submitDocumentMessage(), sender())) as {
      kind: string
      jobId: string
      sourceSha256: string
    }
    expect(submitted).toMatchObject({ kind: 'document', sourceSha256: 'b'.repeat(64) })

    now = 1_600
    const firstBatch = (await first.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )) as { nextSequence: number; updates: Array<{ type: string }> }
    expect(firstBatch.updates.map(({ type }) => type)).toEqual([
      'progress',
      'progress',
      'documentBlockReady',
      'documentBlockReady',
    ])
    await first.route(
      { type: 'job:ack', jobId: submitted.jobId, sequence: firstBatch.nextSequence },
      sender(),
    )
    await expect(
      first.route(
        {
          type: 'job:focus',
          jobId: submitted.jobId,
          focus: { kind: 'document', visibleBlockIds: ['block-1'], active: true },
        },
        sender(),
      ),
    ).resolves.toBeUndefined()
    await expect(
      first.route(
        {
          type: 'job:focus',
          jobId: submitted.jobId,
          focus: { kind: 'image', visibleRects: [], active: true },
        },
        sender(),
      ),
    ).rejects.toMatchObject({ code: 'JOB_MODALITY_MISMATCH' })
    await expect(
      first.route(
        {
          type: 'job:patch',
          jobId: submitted.jobId,
          patchId: 'document-patch',
          mimeType: 'image/png',
        },
        sender(),
      ),
    ).rejects.toMatchObject({ code: 'PATCH_JOB_MISMATCH' })

    const restarted = new BackgroundRouter({
      fixture,
      now: () => now,
    })
    expect(
      await restarted.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'fixture-document-session',
          pageUrl,
          candidates: [documentCandidate(submitted.sourceSha256)],
        },
        sender(),
      ),
    ).toEqual([
      expect.objectContaining({
        kind: 'document',
        jobId: submitted.jobId,
        acknowledgedSequence: 0,
      }),
    ])

    now = 2_500
    const finalBatch = (await restarted.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )) as {
      nextSequence: number
      updates: Array<{ type: string; translatedCount?: number; preservedCount?: number }>
    }
    expect(finalBatch.updates.filter((update) => update.type === 'complete')).toEqual([
      expect.objectContaining({ type: 'complete', translatedCount: 2, preservedCount: 0 }),
    ])
    await restarted.route(
      {
        type: 'job:ack',
        jobId: submitted.jobId,
        sequence: finalBatch.nextSequence,
        terminalType: 'complete',
      },
      sender(),
    )
    expect(await new PageArtifactStore().get(submitted.jobId)).toMatchObject({
      source: { kind: 'document' },
      itemIds: ['block-0', 'block-1'],
    })
  })

  it('replays unacknowledged document blocks after MV3 reconstruction and installs each once', async () => {
    const fixture = new FixtureService(() => now)
    const first = new BackgroundRouter({ fixture, now: () => now })
    const message = submitDocumentMessage('unacknowledged-document-session')
    const attachTranslatedText = vi.fn(() => vi.fn())
    const reader = documentReader(message.request, attachTranslatedText)
    await first.route(
      {
        type: 'chapter:start',
        pageSessionId: message.request.pageSessionId,
        pageUrl,
        contentKind: 'document',
      },
      sender(),
    )
    const submitted = (await first.route(message, sender())) as {
      jobId: string
      sourceSha256: string
    }

    now = 1_600
    const delivered = (await first.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )) as JobUpdateBatch
    const deliveredBlocks = delivered.updates.filter(
      (update): update is DocumentBlockReadyJobUpdate => update.type === 'documentBlockReady',
    )
    expect(deliveredBlocks).toHaveLength(2)
    expect(
      deliveredBlocks.map(({ block }) => reader.installBlock(block.itemId, block.text)),
    ).toEqual([true, true])
    expect(reader.counts()).toEqual({ translated: 2, preserved: 0, pending: 0 })
    expect(attachTranslatedText).toHaveBeenCalledTimes(2)

    const restarted = new BackgroundRouter({ fixture, now: () => now })
    expect(
      await restarted.route(
        {
          type: 'jobs:recover',
          pageSessionId: message.request.pageSessionId,
          pageUrl,
          candidates: [documentCandidate(submitted.sourceSha256)],
        },
        sender(),
      ),
    ).toEqual([
      expect.objectContaining({
        kind: 'document',
        jobId: submitted.jobId,
        acknowledgedSequence: 0,
      }),
    ])
    const replayed = (await restarted.route(
      { type: 'job:updates', jobId: submitted.jobId, after: 0 },
      sender(),
    )) as JobUpdateBatch
    expect(replayed.updates.map(({ sequence }) => sequence)).toEqual(
      delivered.updates.map(({ sequence }) => sequence),
    )
    const replayedBlocks = replayed.updates.filter(
      (update): update is DocumentBlockReadyJobUpdate => update.type === 'documentBlockReady',
    )
    expect(
      replayedBlocks.map(({ block }) => reader.installBlock(block.itemId, block.text)),
    ).toEqual([false, false])
    expect(reader.counts()).toEqual({ translated: 2, preserved: 0, pending: 0 })
    expect(attachTranslatedText).toHaveBeenCalledTimes(2)

    await restarted.route(
      { type: 'job:ack', jobId: submitted.jobId, sequence: replayed.nextSequence },
      sender(),
    )
    reader.destroy()
  })

  it('refuses stale document output when the current document hash differs', async () => {
    const router = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    const submitted = (await router.route(submitDocumentMessage('hash-page'), sender())) as {
      jobId: string
    }
    expect(
      await router.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'hash-page',
          pageUrl,
          candidates: [documentCandidate('c'.repeat(64))],
        },
        sender(),
      ),
    ).toEqual([])
    expect(await new ActiveJobStore().get(submitted.jobId)).toBeUndefined()
  })

  it('refuses document recovery when translation settings changed', async () => {
    const router = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    const submitted = (await router.route(submitDocumentMessage('settings-page'), sender())) as {
      jobId: string
      sourceSha256: string
    }
    expect(
      await router.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'settings-page',
          pageUrl,
          candidates: [
            {
              ...documentCandidate(submitted.sourceSha256),
              settings: { ...documentSettings, learningMode: 'strict' as const },
            },
          ],
        },
        sender(),
      ),
    ).toEqual([])
    expect(await new ActiveJobStore().get(submitted.jobId)).toBeUndefined()
  })

  it('accepts tagged image focus only from the owning document', async () => {
    const router = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    await router.route(
      {
        type: 'chapter:start',
        pageSessionId: 'fixture-page-session',
        pageUrl,
        contentKind: 'image',
      },
      sender(),
    )
    const submitted = (await router.route(submitImageMessage(), sender())) as { jobId: string }
    await expect(
      router.route(
        {
          type: 'job:focus',
          jobId: submitted.jobId,
          focus: {
            kind: 'image',
            visibleRects: [{ x: 0, y: 0.4, width: 1, height: 0.3 }],
            active: true,
          },
        },
        sender(),
      ),
    ).resolves.toBeUndefined()
    await expect(
      router.route(
        {
          type: 'job:focus',
          jobId: submitted.jobId,
          focus: { kind: 'image', visibleRects: [], active: false },
        },
        sender('https://reader.test/other'),
      ),
    ).rejects.toMatchObject({ code: 'DOCUMENT_IDENTITY_MISMATCH' })
  })

  it('requires acknowledgement before advancing the external update cursor', async () => {
    const router = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    const submitted = (await router.route(submitImageMessage(), sender())) as { jobId: string }
    now = 1_600
    await router.route({ type: 'job:updates', jobId: submitted.jobId, after: 0 }, sender())
    await expect(
      router.route({ type: 'job:updates', jobId: submitted.jobId, after: 3 }, sender()),
    ).rejects.toMatchObject({ code: 'UPDATE_CURSOR_MISMATCH' })
  })

  it('cancellation removes recovery and partial artifact metadata', async () => {
    const router = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    const submitted = (await router.route(submitImageMessage('cancel-page'), sender())) as {
      jobId: string
    }
    now = 1_600
    await router.route({ type: 'job:updates', jobId: submitted.jobId, after: 0 }, sender())
    await router.route({ type: 'job:cancel', jobId: submitted.jobId }, sender())
    expect(
      await router.route(
        { type: 'jobs:recover', pageSessionId: 'cancel-page', pageUrl, candidates: [] },
        sender(),
      ),
    ).toEqual([])
    await expect(
      router.route(
        { type: 'job:patch', jobId: submitted.jobId, patchId: 'stale', mimeType: 'image/png' },
        sender(),
      ),
    ).rejects.toMatchObject({ code: 'ACTIVE_JOB_NOT_FOUND' })
  })

  it.each(['failed', 'cancelled', 'complete'] as const)(
    'immediately releases a terminal %s job when no page artifact is retained',
    async (terminalType) => {
      const jobs = new ActiveJobStore()
      const artifacts = new PageArtifactStore()
      const jobId = `no-artifact-${terminalType}`
      await jobs.put(activeDocumentRecord(jobId, `terminal-${terminalType}`))
      const cancelJob = vi.fn(async (releasedJobId: string) => {
        expect(await jobs.get(releasedJobId)).toBeDefined()
      })
      const companion = {
        cancelJob,
        closeChapter: vi.fn(async () => undefined),
      } as unknown as CompanionClient
      const router = new BackgroundRouter({ jobs, artifacts, companion })

      await router.route({ type: 'job:ack', jobId, sequence: 1, terminalType }, sender())

      expect(cancelJob).toHaveBeenCalledOnce()
      expect(cancelJob).toHaveBeenCalledWith(jobId)
      expect(await jobs.get(jobId)).toBeUndefined()
      expect(await artifacts.get(jobId)).toBeUndefined()
    },
  )

  it('releases every completed companion artifact before tab cleanup removes it', async () => {
    const artifacts = new PageArtifactStore()
    const pageSessionId = 'completed-tab-session'
    const jobIds = ['completed-document-one', 'completed-document-two']
    for (const jobId of jobIds) await artifacts.put(documentArtifact(jobId, pageSessionId))
    const cancelJob = vi.fn(async (jobId: string) => {
      expect(await artifacts.get(jobId)).toBeDefined()
    })
    const closeChapter = vi.fn(async () => undefined)
    const companion = { cancelJob, closeChapter } as unknown as CompanionClient
    const router = new BackgroundRouter({ artifacts, companion })
    await router.route(
      {
        type: 'chapter:start',
        pageSessionId,
        pageUrl,
        contentKind: 'document',
      },
      sender(),
    )
    await router.route({ type: 'chapter:finish', pageSessionId, pageUrl }, sender())

    await router.cancelJobsForTab(7)

    expect(cancelJob.mock.calls.map(([jobId]) => jobId).sort()).toEqual([...jobIds].sort())
    expect(closeChapter).toHaveBeenCalledOnce()
    expect(closeChapter).toHaveBeenCalledWith(pageSessionId)
    await expect(artifacts.forTab(7)).resolves.toEqual([])

    await router.cancelJobsForTab(7)
    expect(cancelJob).toHaveBeenCalledTimes(2)
    expect(closeChapter).toHaveBeenCalledOnce()
  })

  it('releases fixture artifacts exactly once during tab cleanup', async () => {
    const artifacts = new PageArtifactStore()
    const artifact = documentArtifact('completed-fixture-document', 'fixture-tab-session')
    await artifacts.put(artifact)
    const fixture = new FixtureService(() => now)
    const releaseJob = vi.spyOn(fixture, 'releaseJob')
    const router = new BackgroundRouter({ artifacts, fixture })

    await router.cancelJobsForTab(7)
    await router.cancelJobsForTab(7)

    expect(releaseJob).toHaveBeenCalledOnce()
    expect(releaseJob).toHaveBeenCalledWith(artifact.jobId)
    await expect(artifacts.get(artifact.jobId)).resolves.toBeUndefined()
  })

  it('clears active and artifact storage even when backend release fails', async () => {
    const jobs = new ActiveJobStore()
    const artifacts = new PageArtifactStore()
    const pageSessionId = 'release-failure-session'
    const active = activeDocumentRecord('release-failure-active', pageSessionId, ['active-block'])
    const artifact = documentArtifact('release-failure-artifact', pageSessionId)
    await jobs.put(active)
    await artifacts.put(artifact)
    const companion = {
      cancelJob: vi.fn(async () => {
        throw new Error('Native release failed.')
      }),
      closeChapter: vi.fn(async () => undefined),
    } as unknown as CompanionClient
    const router = new BackgroundRouter({ jobs, artifacts, companion })

    await router.cancelJobsForTab(7)

    await expect(jobs.get(active.jobId)).resolves.toBeUndefined()
    await expect(artifacts.get(artifact.jobId)).resolves.toBeUndefined()
  })

  it('closes chapter context once for direct page cleanup and not again after terminal close', async () => {
    const closeChapter = vi.fn(async () => undefined)
    const companion = {
      cancelJob: vi.fn(async () => undefined),
      closeChapter,
    } as unknown as CompanionClient
    const router = new BackgroundRouter({ companion })

    await router.route(
      {
        type: 'chapter:start',
        pageSessionId: 'direct-cleanup-session',
        pageUrl,
        contentKind: 'document',
      },
      sender(),
    )
    await router.route(
      { type: 'jobs:cancel-page', pageSessionId: 'direct-cleanup-session' },
      sender(),
    )
    await router.route(
      { type: 'jobs:cancel-page', pageSessionId: 'direct-cleanup-session' },
      sender(),
    )
    expect(closeChapter).toHaveBeenCalledTimes(1)
    expect(closeChapter).toHaveBeenCalledWith('direct-cleanup-session')

    await router.route(
      {
        type: 'chapter:start',
        pageSessionId: 'finished-cleanup-session',
        pageUrl,
        contentKind: 'document',
      },
      sender(),
    )
    await router.route(
      { type: 'chapter:finish', pageSessionId: 'finished-cleanup-session', pageUrl },
      sender(),
    )
    await router.route(
      { type: 'jobs:cancel-page', pageSessionId: 'finished-cleanup-session' },
      sender(),
    )
    expect(closeChapter).toHaveBeenCalledTimes(2)
    expect(closeChapter).toHaveBeenLastCalledWith('finished-cleanup-session')
  })

  it('never recovers a different image source at the same source index', async () => {
    const router = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    await router.route(submitImageMessage('source-page'), sender())
    expect(
      await router.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'source-page',
          pageUrl,
          candidates: [imageCandidate('https://reader.test/replaced.svg')],
        },
        sender(),
      ),
    ).toEqual([])
  })

  it('refuses image recovery when translation settings changed', async () => {
    const router = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    const submitted = (await router.route(submitImageMessage('image-settings-page'), sender())) as {
      jobId: string
    }
    expect(
      await router.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'image-settings-page',
          pageUrl,
          candidates: [
            {
              ...imageCandidate(),
              settings: { ...imageSettings, hskLevel: 4 as const },
            },
          ],
        },
        sender(),
      ),
    ).toEqual([])
    expect(await new ActiveJobStore().get(submitted.jobId)).toBeUndefined()
  })

  it('refuses image recovery when reading direction changed', async () => {
    const router = new BackgroundRouter({ fixture: new FixtureService(() => now), now: () => now })
    const submitted = (await router.route(submitImageMessage('direction-page'), sender())) as {
      jobId: string
    }
    expect(
      await router.route(
        {
          type: 'jobs:recover',
          pageSessionId: 'direction-page',
          pageUrl,
          candidates: [{ ...imageCandidate(), readingDirection: 'rtl' as const }],
        },
        sender(),
      ),
    ).toEqual([])
    expect(await new ActiveJobStore().get(submitted.jobId)).toBeUndefined()
  })
})
