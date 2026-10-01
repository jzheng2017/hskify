import { sha256Hex } from '../acquisition/hash'
import {
  acquireRemoteImage,
  validateInlineImage,
  type AcquiredImage,
} from '../acquisition/image-acquisition'
import {
  DEFAULT_IMAGE_LIMITS,
  ImageValidationError,
  validateImageBytes,
} from '../acquisition/image-format'
import {
  SingleImagePrefetch,
  type ImagePrefetchIdentity,
} from '../acquisition/single-image-prefetch'
import {
  BUILD_FINGERPRINT,
  type BrowserSetupStatus,
  type DocumentJobRequest,
  type ImageJobRequest,
  type JobFocus,
  type JobUpdateBatch,
  type LookupRequest,
  type LookupResult,
  type ReadingDirection,
  type TranslationSettings,
} from '../contracts/browser'
import {
  ActiveJobStore,
  PageArtifactStore,
  type ActiveJobRecord,
  type PageArtifactRecord,
} from './active-jobs'
import { ChapterLifecycleStore } from './chapter-lifecycle'
import { CompanionClient, CompanionHttpError } from './companion-client'
import {
  parseBackgroundRequest,
  parsePageState,
  type BackgroundRequest,
  type FontPayload,
  type ImageRecoveryCandidate,
  type MessageError,
  type MessageResponse,
  type PageState,
  type PatchPayload,
  type PopupState,
  type RecoveredJob,
  type SubmittedDocumentJob,
  type SubmittedImageJob,
} from './messages'
import { NativeSessionError } from './native-session'
import {
  loadHskLevel,
  loadLearningMode,
  loadReadingDirection,
  saveHskLevel,
  saveLearningMode,
  saveReadingDirection,
  type LearningMode,
} from './settings'

type Sender = browser.runtime.MessageSender

type FixtureBackend = {
  sourceImage(width: number, height: number): Promise<ArrayBuffer>
  createJobId(pageSessionId: string, sourceIndex: number, sourceSha256: string): string
  registerDocument(jobId: string, request: DocumentJobRequest): void
  releaseJob(jobId: string): void
  updates(record: ActiveJobRecord, after: number): JobUpdateBatch
  focus(record: ActiveJobRecord, focus: JobFocus): void
  patch(record: ActiveJobRecord, patchId: string): Promise<ArrayBuffer>
  font(): ArrayBuffer
  lookup(request: LookupRequest): LookupResult
}

type BackgroundDependencies = {
  jobs: ActiveJobStore
  artifacts: PageArtifactStore
  companion: CompanionClient
  fixture: FixtureBackend
  prefetches: SingleImagePrefetch<PrefetchedAcquisition>
  now: () => number
}

type ImageAcquisitionMessage = Extract<
  BackgroundRequest,
  { type: 'image:prefetch' | 'job:submit-image' }
>

type PrefetchedAcquisition = { acquired: AcquiredImage; sourceSha256: string }

class BackgroundOperationError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable = false,
    options?: ErrorOptions,
  ) {
    super(message, options)
    this.name = 'BackgroundOperationError'
  }
}

function messageError(error: unknown): MessageError {
  if (
    error instanceof BackgroundOperationError ||
    error instanceof ImageValidationError ||
    error instanceof CompanionHttpError ||
    error instanceof NativeSessionError
  )
    return { code: error.code, message: error.message, retryable: error.retryable }
  return error instanceof Error
    ? { code: 'EXTENSION_OPERATION_FAILED', message: error.message, retryable: true }
    : {
        code: 'EXTENSION_OPERATION_FAILED',
        message: 'The extension operation failed.',
        retryable: true,
      }
}

function senderLocation(sender: Sender): { tabId: number; frameId: number } {
  const tabId = sender.tab?.id
  if (tabId === undefined)
    throw new BackgroundOperationError(
      'MISSING_TAB_CONTEXT',
      'This action must be started from a webpage tab.',
    )
  return { tabId, frameId: sender.frameId ?? 0 }
}

function sameTranslationSettings(left: TranslationSettings, right: TranslationSettings): boolean {
  return (
    left.sourceLanguage === right.sourceLanguage &&
    left.targetLanguage === right.targetLanguage &&
    left.hskStandard === right.hskStandard &&
    left.hskLevel === right.hskLevel &&
    left.learningMode === right.learningMode
  )
}

function activeTabId(tabs: browser.tabs.Tab[]): number {
  const tabId = tabs[0]?.id
  if (tabId === undefined)
    throw new BackgroundOperationError('NO_ACTIVE_TAB', 'No active webpage tab is available.', true)
  return tabId
}

function normalizedDocumentUrl(value: string): string {
  const url = new URL(value)
  url.hash = ''
  return url.href
}

function assertSenderDocument(sender: Sender, pageUrl: string): void {
  if (!sender.url)
    throw new BackgroundOperationError(
      'MISSING_DOCUMENT_IDENTITY',
      'The webpage document identity is missing.',
    )
  let actual: string
  let expected: string
  try {
    actual = normalizedDocumentUrl(sender.url)
    expected = normalizedDocumentUrl(pageUrl)
  } catch {
    throw new BackgroundOperationError('INVALID_DOCUMENT_IDENTITY', 'The webpage URL is invalid.')
  }
  if (actual !== expected)
    throw new BackgroundOperationError(
      'DOCUMENT_IDENTITY_MISMATCH',
      'The page navigated before the extension request was handled.',
      true,
    )
}

function normalizeSourceUrl(value: string, pageUrl: string): string {
  try {
    const url = new URL(value, pageUrl)
    url.hash = ''
    return url.href
  } catch {
    throw new BackgroundOperationError('INVALID_IMAGE_URL', 'The image URL is invalid.')
  }
}

function sameOwner(
  record: Pick<ActiveJobRecord | PageArtifactRecord, 'tabId' | 'frameId'>,
  sender: Sender,
): boolean {
  const { tabId, frameId } = senderLocation(sender)
  return record.tabId === tabId && record.frameId === frameId
}

function unique(items: readonly string[]): string[] {
  return [...new Set(items)]
}

export class BackgroundRouter {
  private readonly jobs: ActiveJobStore
  private readonly artifacts: PageArtifactStore
  private readonly companion: CompanionClient
  private readonly fixture: FixtureBackend | undefined
  private readonly prefetches: SingleImagePrefetch<PrefetchedAcquisition>
  private readonly chapters = new ChapterLifecycleStore()
  private readonly contentPreparations = new Map<number, Promise<void>>()
  private readonly now: () => number

  constructor(dependencies?: Partial<BackgroundDependencies>) {
    this.jobs = dependencies?.jobs ?? new ActiveJobStore()
    this.artifacts = dependencies?.artifacts ?? new PageArtifactStore()
    this.companion = dependencies?.companion ?? new CompanionClient()
    this.fixture = dependencies?.fixture
    this.prefetches = dependencies?.prefetches ?? new SingleImagePrefetch()
    this.now = dependencies?.now ?? Date.now
  }

  private async activeTab(): Promise<number> {
    return activeTabId(await browser.tabs.query({ active: true, currentWindow: true }))
  }

  private async injectContent(tabId: number): Promise<void> {
    try {
      await browser.scripting.executeScript({
        target: { tabId, allFrames: false },
        files: ['translator.js'],
      })
    } catch (error) {
      throw new BackgroundOperationError(
        'PAGE_INJECTION_FAILED',
        'This Firefox page does not allow chapter translation.',
        false,
        { cause: error },
      )
    }
  }

  private async prepareContentRuntime(tabId: number): Promise<void> {
    if (await this.contentState(tabId)) return
    await this.injectContent(tabId)
    if (await this.contentState(tabId)) return
    throw new BackgroundOperationError(
      'PAGE_INJECTION_FAILED',
      'Hskify could not connect to this page after loading its chapter reader.',
      true,
    )
  }

  private async ensureContent(tabId: number): Promise<void> {
    const existing = this.contentPreparations.get(tabId)
    if (existing) return existing
    const preparation = this.prepareContentRuntime(tabId).finally(() => {
      if (this.contentPreparations.get(tabId) === preparation) {
        this.contentPreparations.delete(tabId)
      }
    })
    this.contentPreparations.set(tabId, preparation)
    return preparation
  }

  private async prepareContent(): Promise<void> {
    await this.ensureContent(await this.activeTab())
  }

  private async warmup(contentKind: 'image' | 'document'): Promise<BrowserSetupStatus> {
    const status = await this.companion.getSetupStatus()
    if (
      status.state === 'missing-models' ||
      status.state === 'downloading' ||
      status.state === 'verifying' ||
      status.state === 'failed'
    )
      return status
    return this.companion.warmup(contentKind)
  }

  private async contentState(tabId: number): Promise<PageState | undefined> {
    try {
      return parsePageState(await browser.tabs.sendMessage(tabId, { type: 'content:state' }))
    } catch {
      return undefined
    }
  }

  private async startContent(
    scope: 'visible' | 'all',
    hskLevel: 1 | 2 | 3 | 4 | 5 | 6,
    learningMode: LearningMode,
    readingDirection: ReadingDirection,
  ): Promise<PageState> {
    const tabId = await this.activeTab()
    await Promise.all([
      saveHskLevel(hskLevel),
      saveLearningMode(learningMode),
      saveReadingDirection(readingDirection),
    ])
    await this.ensureContent(tabId)
    return parsePageState(
      await browser.tabs.sendMessage(tabId, {
        type: 'content:start',
        scope,
        hskLevel,
        learningMode,
        readingDirection,
      }),
    )
  }

  private async cancelTab(tabId: number): Promise<PageState> {
    await this.prefetches.cancelIf((identity) => identity.tabId === tabId)
    try {
      return parsePageState(await browser.tabs.sendMessage(tabId, { type: 'content:cancel' }))
    } catch {
      const records = await this.jobs.forTab(tabId)
      const artifacts = await this.artifacts.forTab(tabId)
      await this.cancelJobsForTab(tabId)
      return {
        state: 'cancelled',
        contentKind: records[0]?.source.kind ?? artifacts[0]?.source.kind ?? 'unsupported',
        current: 0,
        total: records.length + artifacts.length,
        message: 'Cancelled',
      }
    }
  }

  private async popupState(): Promise<PopupState> {
    const tabId = await this.activeTab()
    const [hskLevel, learningMode, readingDirection] = await Promise.all([
      loadHskLevel(),
      loadLearningMode(),
      loadReadingDirection(),
    ])
    const content = await this.contentState(tabId)
    if (content) return { ...content, hskLevel, learningMode, readingDirection }
    const active = await this.jobs.forTab(tabId)
    return {
      state: active.length > 0 ? 'running' : 'idle',
      contentKind: active[0]?.source.kind ?? 'unsupported',
      current: 0,
      total: active.length,
      message: active.length > 0 ? 'Translation continues in this tab.' : 'Ready',
      hskLevel,
      learningMode,
      readingDirection,
    }
  }

  private async acquire(
    message: ImageAcquisitionMessage,
    signal?: AbortSignal,
  ): Promise<AcquiredImage> {
    if (this.fixture)
      return validateInlineImage(
        await this.fixture.sourceImage(message.naturalWidth, message.naturalHeight),
        'image/png',
      )
    if ('sourceBytes' in message && message.sourceBytes)
      return validateInlineImage(message.sourceBytes, message.sourceMimeType)
    return acquireRemoteImage(message.imageUrl, {
      pageOrigin: new URL(message.pageUrl).origin,
      ...(signal ? { signal } : {}),
    })
  }

  private prefetchIdentity(
    message: ImageAcquisitionMessage,
    sender: Sender,
  ): ImagePrefetchIdentity {
    const { tabId, frameId } = senderLocation(sender)
    return {
      tabId,
      frameId,
      pageSessionId: message.pageSessionId,
      pageUrl: normalizedDocumentUrl(message.pageUrl),
      sourceIndex: message.sourceIndex,
      sourceUrl: normalizeSourceUrl(message.imageUrl, message.pageUrl),
      naturalWidth: message.naturalWidth,
      naturalHeight: message.naturalHeight,
    }
  }

  private async acquireAndHash(
    message: ImageAcquisitionMessage,
    signal?: AbortSignal,
  ): Promise<PrefetchedAcquisition> {
    const acquired = await this.acquire(message, signal)
    if (acquired.width !== message.naturalWidth || acquired.height !== message.naturalHeight) {
      throw new BackgroundOperationError(
        'SOURCE_DIMENSIONS_CHANGED',
        'The decoded image dimensions changed while translation was starting.',
        true,
      )
    }
    const sourceSha256 = await sha256Hex(acquired.bytes)
    if (signal?.aborted) throw new DOMException('The image prefetch was cancelled.', 'AbortError')
    return { acquired, sourceSha256 }
  }

  private async prefetch(
    message: Extract<BackgroundRequest, { type: 'image:prefetch' }>,
    sender: Sender,
  ): Promise<void> {
    assertSenderDocument(sender, message.pageUrl)
    await this.prefetches.prefetch(this.prefetchIdentity(message, sender), (signal) =>
      this.acquireAndHash(message, signal),
    )
  }

  private async cancelPrefetch(
    message: Extract<BackgroundRequest, { type: 'image:prefetch-cancel' }>,
    sender: Sender,
  ): Promise<void> {
    const { tabId, frameId } = senderLocation(sender)
    assertSenderDocument(sender, message.pageUrl)
    const pageUrl = normalizedDocumentUrl(message.pageUrl)
    await this.prefetches.cancelIf(
      (identity) =>
        identity.tabId === tabId &&
        identity.frameId === frameId &&
        identity.pageSessionId === message.pageSessionId &&
        identity.pageUrl === pageUrl,
    )
  }

  private async submitImage(
    message: Extract<BackgroundRequest, { type: 'job:submit-image' }>,
    sender: Sender,
  ): Promise<SubmittedImageJob> {
    const { tabId, frameId } = senderLocation(sender)
    assertSenderDocument(sender, message.pageUrl)
    const sourceUrl = normalizeSourceUrl(message.imageUrl, message.pageUrl)
    const prefetched = await this.prefetches.consume(this.prefetchIdentity(message, sender))
    const { acquired, sourceSha256 } = prefetched ?? (await this.acquireAndHash(message))
    const clientImageId = `${message.pageSessionId}-${message.sourceIndex}-${sourceSha256.slice(0, 16)}`
    const request: ImageJobRequest = {
      clientRequestId: message.clientRequestId,
      retryItemIds: message.retryItemIds,
      buildFingerprint: BUILD_FINGERPRINT,
      clientImageId,
      sourceSha256,
      sourceMimeType: acquired.mimeType,
      naturalWidth: acquired.width,
      naturalHeight: acquired.height,
      pageSessionId: message.pageSessionId,
      sourceIndex: message.sourceIndex,
      chapterSourceOrder: message.chapterSourceOrder,
      surfaceKind: message.surfaceKind,
      visibleRects: message.visibleRects,
      readingDirection: message.readingDirection,
      settings: {
        sourceLanguage: 'en',
        targetLanguage: 'zh-CN',
        hskStandard: '2.0',
        hskLevel: message.hskLevel,
        learningMode: message.learningMode,
      },
    }
    const submittedAtUnixMs = this.now()
    const jobId = this.fixture
      ? this.fixture.createJobId(message.pageSessionId, message.sourceIndex, sourceSha256)
      : await this.companion.createImageJob(acquired.bytes, request)
    const record: ActiveJobRecord = {
      tabId,
      frameId,
      pageSessionId: message.pageSessionId,
      pageUrl: normalizedDocumentUrl(message.pageUrl),
      jobId,
      sourceSha256,
      source: {
        kind: 'image',
        clientImageId,
        sourceUrl,
        sourceWidth: acquired.width,
        sourceHeight: acquired.height,
        sourceIndex: message.sourceIndex,
        request,
        uploadedBytes: acquired.bytes.byteLength,
      },
      submittedAtUnixMs,
      acknowledgedSequence: 0,
      deliveredSequence: 0,
      itemIds: [],
      patchIds: [],
      fontIds: [],
      createdAtUnixMs: this.now(),
    }
    await this.jobs.put(record)
    return {
      kind: 'image',
      jobId,
      clientImageId,
      sourceSha256,
      sourceUrl,
      sourceWidth: acquired.width,
      sourceHeight: acquired.height,
      sourceIndex: message.sourceIndex,
      acknowledgedSequence: 0,
    }
  }

  private async submitDocument(
    message: Extract<BackgroundRequest, { type: 'job:submit-document' }>,
    sender: Sender,
  ): Promise<SubmittedDocumentJob> {
    const { tabId, frameId } = senderLocation(sender)
    assertSenderDocument(sender, message.pageUrl)
    const request = message.request
    const uploadedBytes = new TextEncoder().encode(JSON.stringify(request)).byteLength
    const submittedAtUnixMs = this.now()
    const jobId = this.fixture
      ? this.fixture.createJobId(request.pageSessionId, 0, request.sourceSha256)
      : await this.companion.createDocumentJob(request)
    await this.jobs.put({
      tabId,
      frameId,
      pageSessionId: request.pageSessionId,
      pageUrl: normalizedDocumentUrl(message.pageUrl),
      jobId,
      sourceSha256: request.sourceSha256,
      source: {
        kind: 'document',
        settings: request.settings,
        blockCount: request.blocks.length,
        uploadedBytes,
      },
      submittedAtUnixMs,
      acknowledgedSequence: 0,
      deliveredSequence: 0,
      itemIds: [],
      patchIds: [],
      fontIds: [],
      createdAtUnixMs: this.now(),
    })
    this.fixture?.registerDocument(jobId, request)
    return { kind: 'document', jobId, sourceSha256: request.sourceSha256, acknowledgedSequence: 0 }
  }

  private async ownedActive(jobId: string, sender: Sender): Promise<ActiveJobRecord> {
    const record = await this.jobs.get(jobId)
    if (!record)
      throw new BackgroundOperationError(
        'ACTIVE_JOB_NOT_FOUND',
        'The active translation job could not be recovered.',
        true,
      )
    if (!sameOwner(record, sender))
      throw new BackgroundOperationError(
        'JOB_OWNER_MISMATCH',
        'This document does not own the requested translation job.',
      )
    assertSenderDocument(sender, record.pageUrl)
    return record
  }

  private artifactFrom(record: ActiveJobRecord): PageArtifactRecord {
    const source: PageArtifactRecord['source'] =
      record.source.kind === 'document'
        ? { kind: 'document' }
        : {
            kind: 'image',
            sourceUrl: record.source.sourceUrl,
            sourceWidth: record.source.sourceWidth,
            sourceHeight: record.source.sourceHeight,
            sourceIndex: record.source.sourceIndex,
          }
    return {
      tabId: record.tabId,
      frameId: record.frameId,
      pageSessionId: record.pageSessionId,
      pageUrl: record.pageUrl,
      jobId: record.jobId,
      sourceSha256: record.sourceSha256,
      source,
      itemIds: record.itemIds,
      patchIds: record.patchIds,
      fontIds: record.fontIds,
      createdAtUnixMs: record.createdAtUnixMs,
    }
  }

  private async recordDeliveredUpdates(
    record: ActiveJobRecord,
    batch: JobUpdateBatch,
  ): Promise<void> {
    if (batch.jobId !== record.jobId)
      throw new BackgroundOperationError(
        'UPDATE_IDENTITY_MISMATCH',
        'The local translation updates did not match the active job.',
      )
    const itemIds: string[] = []
    const patchIds: string[] = []
    const fontIds: string[] = []
    for (const update of batch.updates) {
      if (update.type === 'imageRegionReady') {
        itemIds.push(update.region.itemId)
        patchIds.push(update.region.patch.blobId)
        fontIds.push(update.region.style.fontId)
      } else if (update.type === 'imageRegionPreserved') itemIds.push(update.region.itemId)
      else if (update.type === 'documentBlockReady' || update.type === 'documentBlockPreserved')
        itemIds.push(update.block.itemId)
    }
    await this.jobs.put({
      ...record,
      deliveredSequence: Math.max(record.deliveredSequence, batch.nextSequence),
      itemIds: unique([...record.itemIds, ...itemIds]),
      patchIds: unique([...record.patchIds, ...patchIds]),
      fontIds: unique([...record.fontIds, ...fontIds]),
    })
  }

  private async updates(
    message: Extract<BackgroundRequest, { type: 'job:updates' }>,
    sender: Sender,
  ): Promise<JobUpdateBatch> {
    const record = await this.ownedActive(message.jobId, sender)
    if (message.after !== record.acknowledgedSequence)
      throw new BackgroundOperationError(
        'UPDATE_CURSOR_MISMATCH',
        'The update cursor does not match the last installed page update.',
        true,
      )
    const batch = this.fixture
      ? this.fixture.updates(record, message.after)
      : await this.companion.getJobUpdates(record.jobId, message.after)
    await this.recordDeliveredUpdates(record, batch)
    return batch
  }

  private async acknowledge(
    message: Extract<BackgroundRequest, { type: 'job:ack' }>,
    sender: Sender,
  ): Promise<void> {
    const record = await this.ownedActive(message.jobId, sender)
    if (
      message.sequence < record.acknowledgedSequence ||
      message.sequence > record.deliveredSequence
    )
      throw new BackgroundOperationError(
        'UPDATE_ACK_OUT_OF_RANGE',
        'The page tried to acknowledge updates it has not received.',
      )
    if (message.terminalType && message.sequence !== record.deliveredSequence)
      throw new BackgroundOperationError(
        'TERMINAL_ACK_OUT_OF_RANGE',
        'A terminal update must acknowledge the complete delivered batch.',
      )
    if (message.terminalType) {
      const retainForPage = message.terminalType === 'complete' && record.itemIds.length > 0
      if (retainForPage) {
        await this.artifacts.put(this.artifactFrom(record))
      } else {
        await this.releaseBackendJob(record.jobId)
      }
      await this.jobs.remove(record.jobId)
      if (retainForPage) this.fixture?.releaseJob(record.jobId)
      return
    }
    await this.jobs.put({ ...record, acknowledgedSequence: message.sequence })
  }

  private async focus(
    message: Extract<BackgroundRequest, { type: 'job:focus' }>,
    sender: Sender,
  ): Promise<void> {
    const record = await this.ownedActive(message.jobId, sender)
    if (record.source.kind !== message.focus.kind)
      throw new BackgroundOperationError(
        'JOB_MODALITY_MISMATCH',
        'The focus update modality does not match its job.',
      )
    this.chapters.focus(record.pageSessionId, record.pageUrl)
    if (this.fixture) this.fixture.focus(record, message.focus)
    else await this.companion.updateFocus(record.jobId, message.focus)
  }

  private async patch(
    message: Extract<BackgroundRequest, { type: 'job:patch' }>,
    sender: Sender,
  ): Promise<PatchPayload> {
    const record = await this.ownedActive(message.jobId, sender)
    if (record.source.kind !== 'image' || !record.patchIds.includes(message.patchId))
      throw new BackgroundOperationError(
        'PATCH_JOB_MISMATCH',
        'The requested patch does not belong to this image job.',
      )
    const bytes = this.fixture
      ? await this.fixture.patch(record, message.patchId)
      : await this.companion.getPatch(message.patchId, message.mimeType)
    validateImageBytes(bytes, 'image/png', DEFAULT_IMAGE_LIMITS)
    return { patchId: message.patchId, mimeType: 'image/png', bytes }
  }

  private async removeRecord(record: ActiveJobRecord): Promise<void> {
    await this.jobs.remove(record.jobId)
    await this.artifacts.remove(record.jobId)
  }

  private async cancelRecord(record: ActiveJobRecord): Promise<void> {
    try {
      await this.releaseBackendJob(record.jobId)
    } finally {
      await this.removeRecord(record)
    }
  }

  private async releaseBackendJob(jobId: string): Promise<void> {
    if (this.fixture) {
      this.fixture.releaseJob(jobId)
      return
    }
    await this.companion.cancelJob(jobId)
  }

  private async releaseArtifact(record: PageArtifactRecord): Promise<void> {
    try {
      await this.releaseBackendJob(record.jobId)
    } finally {
      await this.artifacts.remove(record.jobId)
    }
  }

  private async cancelJob(jobId: string, sender: Sender): Promise<void> {
    const record = await this.jobs.get(jobId)
    if (!record) return
    if (!sameOwner(record, sender))
      throw new BackgroundOperationError(
        'JOB_OWNER_MISMATCH',
        'This document does not own the requested translation job.',
      )
    assertSenderDocument(sender, record.pageUrl)
    await this.cancelRecord(record)
  }

  private candidateForRecord(
    record: ActiveJobRecord,
    candidates: Extract<BackgroundRequest, { type: 'jobs:recover' }>['candidates'],
  ): (typeof candidates)[number] | undefined {
    if (record.source.kind === 'document') {
      const source = record.source
      return candidates.find(
        (candidate) =>
          candidate.kind === 'document' &&
          candidate.sourceSha256 === record.sourceSha256 &&
          sameTranslationSettings(candidate.settings, source.settings),
      )
    }
    const source = record.source
    return candidates.find(
      (candidate) =>
        candidate.kind === 'image' &&
        normalizeSourceUrl(candidate.sourceUrl, record.pageUrl) === source.sourceUrl &&
        candidate.naturalWidth === source.sourceWidth &&
        candidate.naturalHeight === source.sourceHeight &&
        candidate.sourceIndex === source.sourceIndex &&
        candidate.readingDirection === source.request.readingDirection &&
        sameTranslationSettings(candidate.settings, source.request.settings),
    )
  }

  private async verifyImageRecovery(
    record: ActiveJobRecord & { source: Extract<ActiveJobRecord['source'], { kind: 'image' }> },
    candidate: ImageRecoveryCandidate,
  ): Promise<boolean> {
    if (candidate.sourceSha256) return candidate.sourceSha256 === record.sourceSha256
    if (this.fixture)
      return (
        (await sha256Hex(
          await this.fixture.sourceImage(record.source.sourceWidth, record.source.sourceHeight),
        )) === record.sourceSha256
      )
    const acquired = await acquireRemoteImage(record.source.sourceUrl, {
      pageOrigin: new URL(record.pageUrl).origin,
    })
    return (
      acquired.width === record.source.sourceWidth &&
      acquired.height === record.source.sourceHeight &&
      (await sha256Hex(acquired.bytes)) === record.sourceSha256
    )
  }

  private async recover(
    message: Extract<BackgroundRequest, { type: 'jobs:recover' }>,
    sender: Sender,
  ): Promise<RecoveredJob[]> {
    const { tabId, frameId } = senderLocation(sender)
    assertSenderDocument(sender, message.pageUrl)
    const recovered: RecoveredJob[] = []
    for (const record of await this.jobs.forPage(tabId, frameId, message.pageSessionId)) {
      const candidate = this.candidateForRecord(record, message.candidates)
      try {
        if (
          !candidate ||
          (record.source.kind === 'image' &&
            (candidate.kind !== 'image' ||
              !(await this.verifyImageRecovery(
                record as ActiveJobRecord & {
                  source: Extract<ActiveJobRecord['source'], { kind: 'image' }>
                },
                candidate,
              ))))
        ) {
          await this.cancelRecord(record)
          continue
        }
        // A new renderer has no installed items. Replay from the beginning,
        // independently of the previous renderer's delivery cursor.
        await this.jobs.put({ ...record, acknowledgedSequence: 0, deliveredSequence: 0 })
        if (record.source.kind === 'document') {
          recovered.push({
            kind: 'document',
            jobId: record.jobId,
            sourceSha256: record.sourceSha256,
            acknowledgedSequence: 0,
          })
        } else {
          recovered.push({
            kind: 'image',
            jobId: record.jobId,
            clientImageId: record.source.clientImageId,
            sourceSha256: record.sourceSha256,
            sourceUrl: record.source.sourceUrl,
            sourceWidth: record.source.sourceWidth,
            sourceHeight: record.source.sourceHeight,
            sourceIndex: record.source.sourceIndex,
            acknowledgedSequence: 0,
          })
        }
      } catch {
        await this.cancelRecord(record).catch(() => this.removeRecord(record))
      }
    }
    return recovered
  }

  private async ownedArtifact(jobId: string, sender: Sender): Promise<PageArtifactRecord> {
    const active = await this.jobs.get(jobId)
    if (active) {
      if (!sameOwner(active, sender))
        throw new BackgroundOperationError(
          'RESULT_OWNER_MISMATCH',
          'This document does not own the requested translation artifact.',
        )
      assertSenderDocument(sender, active.pageUrl)
      return this.artifactFrom(active)
    }
    const artifact = await this.artifacts.get(jobId)
    if (!artifact || !sameOwner(artifact, sender))
      throw new BackgroundOperationError(
        'RESULT_OWNER_MISMATCH',
        'This document does not own the requested translation artifact.',
      )
    assertSenderDocument(sender, artifact.pageUrl)
    return artifact
  }

  private async lookup(
    message: Extract<BackgroundRequest, { type: 'dictionary:lookup' }>,
    sender: Sender,
  ): Promise<LookupResult> {
    senderLocation(sender)
    return this.fixture
      ? this.fixture.lookup(message.request)
      : this.companion.lookup(message.request)
  }

  private async font(
    message: Extract<BackgroundRequest, { type: 'font:get' }>,
    sender: Sender,
  ): Promise<FontPayload> {
    const artifact = await this.ownedArtifact(message.jobId, sender)
    if (artifact.source.kind !== 'image' || !artifact.fontIds.includes(message.fontId))
      throw new BackgroundOperationError(
        'FONT_RESULT_MISMATCH',
        'The requested font does not belong to this image job.',
      )
    return {
      fontId: message.fontId,
      bytes: this.fixture ? this.fixture.font() : await this.companion.getFont(message.fontId),
    }
  }

  private async cancelPage(pageSessionId: string, sender: Sender): Promise<void> {
    const { tabId, frameId } = senderLocation(sender)
    const chapterExists = this.chapters.state(pageSessionId) !== undefined
    await this.prefetches.cancelIf(
      (identity) =>
        identity.tabId === tabId &&
        identity.frameId === frameId &&
        identity.pageSessionId === pageSessionId,
    )
    const records = await this.jobs.forPage(tabId, frameId, pageSessionId)
    const artifacts = await this.artifacts.forPage(tabId, frameId, pageSessionId)
    const activeJobIds = new Set(records.map((record) => record.jobId))
    await Promise.allSettled([
      ...records.map((record) => this.cancelRecord(record)),
      ...artifacts
        .filter((artifact) => !activeJobIds.has(artifact.jobId))
        .map((artifact) => this.releaseArtifact(artifact)),
    ])
    if (chapterExists && !this.fixture)
      await this.companion.closeChapter(pageSessionId).catch(() => undefined)
    this.chapters.remove(pageSessionId)
  }

  private chapterStart(
    message: Extract<BackgroundRequest, { type: 'chapter:start' }>,
    sender: Sender,
  ): void {
    assertSenderDocument(sender, message.pageUrl)
    this.chapters.start(
      message.pageSessionId,
      normalizedDocumentUrl(message.pageUrl),
      message.contentKind,
    )
  }

  private chapterSource(
    message: Extract<BackgroundRequest, { type: 'chapter:source' }>,
    sender: Sender,
  ): void {
    assertSenderDocument(sender, message.pageUrl)
    this.chapters.source(
      message.pageSessionId,
      normalizedDocumentUrl(message.pageUrl),
      message.sourceIndex,
    )
  }

  private async chapterFinish(
    message: Extract<BackgroundRequest, { type: 'chapter:finish' | 'chapter:cancel' }>,
    sender: Sender,
  ): Promise<void> {
    assertSenderDocument(sender, message.pageUrl)
    const pageUrl = normalizedDocumentUrl(message.pageUrl)
    const chapter = this.chapters.state(message.pageSessionId)
    if (chapter && chapter.pageUrl !== pageUrl)
      throw new BackgroundOperationError(
        'DOCUMENT_IDENTITY_MISMATCH',
        'The chapter session does not belong to this document.',
      )
    if (chapter) {
      if (message.type === 'chapter:finish') this.chapters.finish(message.pageSessionId, pageUrl)
      else this.chapters.cancel(message.pageSessionId, pageUrl)
    }
    if (!this.fixture)
      await this.companion.closeChapter(message.pageSessionId).catch(() => undefined)
    this.chapters.remove(message.pageSessionId)
  }

  async route(message: BackgroundRequest, sender: Sender): Promise<unknown> {
    switch (message.type) {
      case 'popup:prepare':
        return this.prepareContent()
      case 'popup:start':
        return this.startContent(
          message.scope,
          message.hskLevel,
          message.learningMode,
          message.readingDirection,
        )
      case 'popup:cancel':
        return this.cancelTab(await this.activeTab())
      case 'popup:state':
        return this.popupState()
      case 'setup:status':
        return this.companion.getSetupStatus()
      case 'setup:start':
        return this.companion.startModelSetup()
      case 'engine:warmup':
        return this.warmup(message.contentKind)
      case 'chapter:start':
        return this.chapterStart(message, sender)
      case 'chapter:source':
        return this.chapterSource(message, sender)
      case 'chapter:finish':
      case 'chapter:cancel':
        return this.chapterFinish(message, sender)
      case 'image:prefetch':
        return this.prefetch(message, sender)
      case 'image:prefetch-cancel':
        return this.cancelPrefetch(message, sender)
      case 'job:submit-image':
        return this.submitImage(message, sender)
      case 'job:submit-document':
        return this.submitDocument(message, sender)
      case 'job:updates':
        return this.updates(message, sender)
      case 'job:ack':
        return this.acknowledge(message, sender)
      case 'job:focus':
        return this.focus(message, sender)
      case 'job:patch':
        return this.patch(message, sender)
      case 'job:cancel':
        return this.cancelJob(message.jobId, sender)
      case 'jobs:recover':
        return this.recover(message, sender)
      case 'jobs:cancel-page':
        return this.cancelPage(message.pageSessionId, sender)
      case 'dictionary:lookup':
        return this.lookup(message, sender)
      case 'source:image-revision': {
        const artifact = await this.ownedArtifact(message.jobId, sender)
        if (artifact.source.kind !== 'image')
          throw new BackgroundOperationError(
            'SOURCE_MODALITY_MISMATCH',
            'This source is not an image.',
          )
        const source = artifact.source
        const result = await this.acquireAndHash({
          type: 'image:prefetch',
          pageSessionId: artifact.pageSessionId,
          sourceIndex: source.sourceIndex,
          imageUrl: source.sourceUrl,
          pageUrl: artifact.pageUrl,
          naturalWidth: source.sourceWidth,
          naturalHeight: source.sourceHeight,
        })
        return result.sourceSha256
      }
      case 'source:capture-region': {
        const { tabId } = senderLocation(sender)
        assertSenderDocument(sender, message.pageUrl)
        const tab = await browser.tabs.get(tabId)
        if (!tab.active)
          throw new BackgroundOperationError(
            'CAPTURE_REQUIRES_VISIBLE_TAB',
            'The selected source must be in the active tab.',
          )
        const captured = await browser.tabs.captureVisibleTab(tab.windowId!, {
          format: 'png',
          rect: message.rect,
          scale: 1,
        })
        const current = await browser.tabs.get(tabId)
        if (!current.active || current.url !== tab.url)
          throw new BackgroundOperationError(
            'CAPTURE_SOURCE_CHANGED',
            'The visible tab changed during source capture.',
          )
        return captured
      }
      case 'font:get':
        return this.font(message, sender)
    }
  }

  async cancelJobsForTab(tabId: number): Promise<void> {
    await this.prefetches.cancelIf((identity) => identity.tabId === tabId)
    const records = await this.jobs.forTab(tabId)
    const artifacts = await this.artifacts.forTab(tabId)
    const activeJobIds = new Set(records.map((record) => record.jobId))
    await Promise.allSettled([
      ...records.map((record) => this.cancelRecord(record)),
      ...artifacts
        .filter((artifact) => !activeJobIds.has(artifact.jobId))
        .map((artifact) => this.releaseArtifact(artifact)),
    ])
    const pageSessionIds = new Set([
      ...records.map((record) => record.pageSessionId),
      ...artifacts.map((artifact) => artifact.pageSessionId),
    ])
    const openPageSessionIds = [...pageSessionIds].filter(
      (pageSessionId) => this.chapters.state(pageSessionId) !== undefined,
    )
    if (!this.fixture) {
      await Promise.allSettled(
        openPageSessionIds.map((pageSessionId) => this.companion.closeChapter(pageSessionId)),
      )
    }
    for (const pageSessionId of pageSessionIds) this.chapters.remove(pageSessionId)
  }
}

declare global {
  var __hskifyBackgroundRegistered: boolean | undefined
}

const BACKGROUND_MESSAGE_TYPES = new Set<BackgroundRequest['type']>([
  'popup:prepare',
  'popup:start',
  'popup:cancel',
  'popup:state',
  'setup:status',
  'setup:start',
  'engine:warmup',
  'chapter:start',
  'chapter:source',
  'chapter:finish',
  'chapter:cancel',
  'image:prefetch',
  'image:prefetch-cancel',
  'job:submit-image',
  'job:submit-document',
  'job:updates',
  'job:ack',
  'job:focus',
  'job:patch',
  'job:cancel',
  'jobs:recover',
  'jobs:cancel-page',
  'dictionary:lookup',
  'source:capture-region',
  'source:image-revision',
  'font:get',
])

function looksLikeBackgroundRequest(value: unknown): boolean {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    BACKGROUND_MESSAGE_TYPES.has(
      String((value as Record<string, unknown>).type) as BackgroundRequest['type'],
    )
  )
}

export function registerBackgroundHandlers(): void {
  if (globalThis.__hskifyBackgroundRegistered) return
  globalThis.__hskifyBackgroundRegistered = true
  const router = new BackgroundRouter()
  browser.runtime.onMessage.addListener(
    async (raw: unknown, sender): Promise<MessageResponse<unknown> | undefined> => {
      if (!looksLikeBackgroundRequest(raw)) return undefined
      if (sender.id !== browser.runtime.id)
        return {
          ok: false,
          error: {
            code: 'INVALID_MESSAGE_SENDER',
            message: 'The runtime message did not come from this extension.',
            retryable: false,
          },
        }
      let message: BackgroundRequest
      try {
        message = parseBackgroundRequest(raw)
      } catch (error) {
        return {
          ok: false,
          error: {
            code: 'INVALID_RUNTIME_MESSAGE',
            message: error instanceof Error ? error.message : 'The runtime message was invalid.',
            retryable: false,
          },
        }
      }
      try {
        return { ok: true, value: await router.route(message, sender) }
      } catch (error) {
        return { ok: false, error: messageError(error) }
      }
    },
  )
  browser.tabs.onRemoved.addListener((tabId) => {
    void router.cancelJobsForTab(tabId)
  })
}
