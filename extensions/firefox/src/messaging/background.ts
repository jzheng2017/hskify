import {
  BUILD_FINGERPRINT,
  type BrowserJobRequest,
  type BrowserSetupStatus,
  type JobUpdateBatch,
  type LookupRequest,
  type LookupResult,
  type ViewportUpdate,
} from '../contracts/browser'
import {
  acquireRemoteImage,
  ImagePermissionRequiredError,
  validateInlineImage,
  type AcquiredImage,
} from '../acquisition/image-acquisition'
import { PendingOriginPermissionStore } from '../acquisition/origin-permissions'
import {
  DEFAULT_IMAGE_LIMITS,
  ImageValidationError,
  validateImageBytes,
} from '../acquisition/image-format'
import { sha256Hex } from '../acquisition/hash'
import {
  SingleImagePrefetch,
  type ImagePrefetchIdentity,
} from '../acquisition/single-image-prefetch'
import {
  ActiveJobStore,
  PageArtifactStore,
  type ActiveJobRecord,
  type PageArtifactRecord,
} from './active-jobs'
import { CompanionClient, CompanionHttpError } from './companion-client'
import { NativeSessionError } from './native-session'
import {
  parseBackgroundRequest,
  parsePageState,
  parsePermissionPlan,
  type BackgroundRequest,
  type FontPayload,
  type MessageError,
  type MessageResponse,
  type PageState,
  type PatchPayload,
  type PermissionPlan,
  type PopupState,
  type RecoveredJob,
  type RecoveryCandidate,
  type SubmittedJob,
} from './messages'
import { loadHskLevel, saveHskLevel } from './settings'

type Sender = browser.runtime.MessageSender

type FixtureBackend = {
  sourceImage(width: number, height: number): Promise<ArrayBuffer>
  createJobId(pageSessionId: string, pageIndex: number, sourceSha256: string): string
  updates(record: ActiveJobRecord, after: number): JobUpdateBatch
  viewport(record: ActiveJobRecord, viewport: ViewportUpdate): void
  patch(record: ActiveJobRecord, patchId: string): Promise<ArrayBuffer>
  font(): ArrayBuffer
  lookup(request: LookupRequest): LookupResult
}

type BackgroundDependencies = {
  jobs: ActiveJobStore
  artifacts: PageArtifactStore
  companion: CompanionClient
  fixture: FixtureBackend
  pendingPermissions: PendingOriginPermissionStore
  prefetches: SingleImagePrefetch<PrefetchedAcquisition>
  now: () => number
}

type ImageAcquisitionMessage = Extract<
  BackgroundRequest,
  { type: 'image:prefetch' | 'job:submit' }
>

type PrefetchedAcquisition = {
  acquired: AcquiredImage
  sourceSha256: string
}

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
  ) {
    return {
      code: error.code,
      message: error.message,
      retryable: error.retryable,
    }
  }
  if (error instanceof Error) {
    return {
      code: 'EXTENSION_OPERATION_FAILED',
      message: error.message,
      retryable: true,
    }
  }
  return {
    code: 'EXTENSION_OPERATION_FAILED',
    message: 'The extension operation failed.',
    retryable: true,
  }
}

function senderLocation(sender: Sender): { tabId: number; frameId: number } {
  const tabId = sender.tab?.id
  if (tabId === undefined) {
    throw new BackgroundOperationError(
      'MISSING_TAB_CONTEXT',
      'This action must be started from a webpage tab.',
    )
  }
  return { tabId, frameId: sender.frameId ?? 0 }
}

function activeTabId(tabs: browser.tabs.Tab[]): number {
  const tabId = tabs[0]?.id
  if (tabId === undefined) {
    throw new BackgroundOperationError('NO_ACTIVE_TAB', 'No active webpage tab is available.', true)
  }
  return tabId
}

function normalizedDocumentUrl(value: string): string {
  const url = new URL(value)
  url.hash = ''
  return url.href
}

function assertSenderDocument(sender: Sender, pageUrl: string): void {
  if (!sender.url) {
    throw new BackgroundOperationError(
      'MISSING_DOCUMENT_IDENTITY',
      'The webpage document identity is missing.',
    )
  }
  let actual: string
  let expected: string
  try {
    actual = normalizedDocumentUrl(sender.url)
    expected = normalizedDocumentUrl(pageUrl)
  } catch {
    throw new BackgroundOperationError('INVALID_DOCUMENT_IDENTITY', 'The webpage URL is invalid.')
  }
  if (actual !== expected) {
    throw new BackgroundOperationError(
      'DOCUMENT_IDENTITY_MISMATCH',
      'The page navigated before the extension request was handled.',
      true,
    )
  }
}

function normalizeSourceUrl(value: string, pageUrl: string): string {
  let url: URL
  try {
    url = new URL(value, pageUrl)
  } catch {
    throw new BackgroundOperationError('INVALID_IMAGE_URL', 'The image URL is invalid.')
  }
  url.hash = ''
  return url.href
}

function sameOwner(
  record: Pick<ActiveJobRecord, 'tabId' | 'frameId'>,
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
  private readonly pendingPermissions: PendingOriginPermissionStore
  private readonly prefetches: SingleImagePrefetch<PrefetchedAcquisition>
  private readonly now: () => number

  constructor(dependencies?: Partial<BackgroundDependencies>) {
    this.jobs = dependencies?.jobs ?? new ActiveJobStore()
    this.artifacts = dependencies?.artifacts ?? new PageArtifactStore()
    this.companion = dependencies?.companion ?? new CompanionClient()
    this.fixture = dependencies?.fixture
    this.pendingPermissions =
      dependencies?.pendingPermissions ?? new PendingOriginPermissionStore()
    this.prefetches = dependencies?.prefetches ?? new SingleImagePrefetch()
    this.now = dependencies?.now ?? Date.now
  }

  private async activeTab(): Promise<number> {
    return activeTabId(await browser.tabs.query({ active: true, currentWindow: true }))
  }

  private async ensureContent(tabId: number): Promise<void> {
    try {
      await browser.scripting.executeScript({
        target: { tabId, allFrames: false },
        files: ['translator.js'],
      })
    } catch (error) {
      throw new BackgroundOperationError(
        'PAGE_INJECTION_FAILED',
        'This Firefox page does not allow manga translation.',
        false,
        { cause: error },
      )
    }
  }

  private async prepareContent(): Promise<PermissionPlan> {
    const tabId = await this.activeTab()
    await this.ensureContent(tabId)
    const raw = await browser.tabs.sendMessage(tabId, { type: 'content:prepare' })
    const plan = parsePermissionPlan(raw)
    const pending = await this.pendingPermissions.list(tabId)
    const unresolved: string[] = []
    for (const origin of pending) {
      if (!(await browser.permissions.contains({ origins: [origin] }))) {
        unresolved.push(origin)
      }
    }
    await this.pendingPermissions.replace(tabId, unresolved)
    return {
      visibleOrigins: unique([...plan.visibleOrigins, ...unresolved]).sort(),
      allOrigins: unique([...plan.allOrigins, ...unresolved]).sort(),
    }
  }

  private async contentState(tabId: number): Promise<PageState | undefined> {
    try {
      return parsePageState(
        await browser.tabs.sendMessage(tabId, {
          type: 'content:state',
        }),
      )
    } catch {
      return undefined
    }
  }

  private async startContent(
    scope: 'visible' | 'all',
    hskLevel: 1 | 2 | 3 | 4 | 5 | 6,
  ): Promise<PageState> {
    const tabId = await this.activeTab()
    await saveHskLevel(hskLevel)
    await this.ensureContent(tabId)
    return parsePageState(
      await browser.tabs.sendMessage(tabId, {
        type: 'content:start',
        scope,
        hskLevel,
      }),
    )
  }

  private async cancelTab(tabId: number): Promise<PageState> {
    await this.prefetches.cancelIf((identity) => identity.tabId === tabId)
    try {
      return parsePageState(
        await browser.tabs.sendMessage(tabId, {
          type: 'content:cancel',
        }),
      )
    } catch {
      const records = await this.jobs.forTab(tabId)
      await Promise.allSettled(records.map((record) => this.cancelRecord(record)))
      await this.artifacts.removeForTab(tabId)
      return {
        state: 'cancelled',
        current: 0,
        total: records.length,
        message: 'Cancelled',
      }
    }
  }

  private async popupState(): Promise<PopupState> {
    const tabId = await this.activeTab()
    const level = await loadHskLevel()
    const content = await this.contentState(tabId)
    if (content) return { ...content, hskLevel: level }
    const active = await this.jobs.forTab(tabId)
    return {
      state: active.length > 0 ? 'running' : 'idle',
      current: 0,
      total: active.length,
      message: active.length > 0 ? 'Translation continues in this tab.' : 'Ready',
      hskLevel: level,
    }
  }

  private setupStatus(): Promise<BrowserSetupStatus> {
    return this.companion.getSetupStatus()
  }

  private startSetup(): Promise<BrowserSetupStatus> {
    return this.companion.startModelSetup()
  }

  private async openInstaller(): Promise<void> {
    await browser.tabs.create({ url: browser.runtime.getURL('/setup.html') })
  }

  private async acquire(
    message: ImageAcquisitionMessage,
    signal?: AbortSignal,
  ): Promise<AcquiredImage> {
    if (this.fixture) {
      return validateInlineImage(
        await this.fixture.sourceImage(message.naturalWidth, message.naturalHeight),
        'image/png',
      )
    }
    if ('sourceBytes' in message && message.sourceBytes) {
      return validateInlineImage(message.sourceBytes, message.sourceMimeType)
    }
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
      pageIndex: message.pageIndex,
      sourceUrl: normalizeSourceUrl(message.imageUrl, message.pageUrl),
      naturalWidth: message.naturalWidth,
      naturalHeight: message.naturalHeight,
    }
  }

  private async acquireAndHash(
    message: ImageAcquisitionMessage,
    tabId: number,
    signal?: AbortSignal,
  ): Promise<PrefetchedAcquisition> {
    let acquired: AcquiredImage
    try {
      acquired = await this.acquire(message, signal)
    } catch (error) {
      if (error instanceof ImagePermissionRequiredError) {
        await this.pendingPermissions.add(tabId, error.originPattern)
      }
      throw error
    }
    if (
      acquired.width !== message.naturalWidth ||
      acquired.height !== message.naturalHeight
    ) {
      throw new BackgroundOperationError(
        'SOURCE_DIMENSIONS_CHANGED',
        'The decoded image dimensions changed while translation was starting.',
        true,
      )
    }
    const sourceSha256 = await sha256Hex(acquired.bytes)
    if (signal?.aborted) {
      throw new DOMException('The image prefetch was cancelled.', 'AbortError')
    }
    return { acquired, sourceSha256 }
  }

  private async prefetch(
    message: Extract<BackgroundRequest, { type: 'image:prefetch' }>,
    sender: Sender,
  ): Promise<void> {
    const { tabId } = senderLocation(sender)
    assertSenderDocument(sender, message.pageUrl)
    const identity = this.prefetchIdentity(message, sender)
    await this.prefetches.prefetch(identity, (signal) =>
      this.acquireAndHash(message, tabId, signal),
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

  private async submit(
    message: Extract<BackgroundRequest, { type: 'job:submit' }>,
    sender: Sender,
  ): Promise<SubmittedJob> {
    const { tabId, frameId } = senderLocation(sender)
    assertSenderDocument(sender, message.pageUrl)
    const expectedSourceUrl = normalizeSourceUrl(message.imageUrl, message.pageUrl)
    const identity = this.prefetchIdentity(message, sender)
    const prefetched = await this.prefetches.consume(identity)
    const { acquired, sourceSha256 } =
      prefetched ?? (await this.acquireAndHash(message, tabId))
    const clientImageId = `${message.pageSessionId}-${message.pageIndex}-${sourceSha256.slice(0, 16)}`
    const request: BrowserJobRequest = {
      buildFingerprint: BUILD_FINGERPRINT,
      clientImageId,
      sourceSha256,
      sourceMimeType: acquired.mimeType,
      naturalWidth: acquired.width,
      naturalHeight: acquired.height,
      pageSessionId: message.pageSessionId,
      pageIndex: message.pageIndex,
      visibleRects: message.visibleRects,
      settings: {
        sourceLanguage: 'en',
        targetLanguage: 'zh-CN',
        hskStandard: '2.0',
        hskLevel: message.hskLevel,
        readingDirection: 'auto',
        translateSoundEffects: false,
      },
      ...(message.precedingContext?.length
        ? { precedingContext: message.precedingContext.slice(-6) }
        : {}),
      ...(message.properNameGlossary?.length
        ? { properNameGlossary: message.properNameGlossary }
        : {}),
    }
    const submittedAtUnixMs = this.now()
    const jobId = this.fixture
      ? this.fixture.createJobId(message.pageSessionId, message.pageIndex, sourceSha256)
      : await this.companion.createJob(acquired.bytes, request)
    const record: ActiveJobRecord = {
      tabId,
      frameId,
      pageSessionId: message.pageSessionId,
      pageUrl: normalizedDocumentUrl(message.pageUrl),
      clientImageId,
      jobId,
      sourceSha256,
      sourceUrl: expectedSourceUrl,
      sourceWidth: acquired.width,
      sourceHeight: acquired.height,
      pageIndex: message.pageIndex,
      hskLevel: message.hskLevel,
      submittedRequest: request,
      uploadedImageBytes: acquired.bytes.byteLength,
      submittedAtUnixMs,
      acknowledgedSequence: 0,
      deliveredSequence: 0,
      regionIds: [],
      patchIds: [],
      fontIds: [],
      createdAtUnixMs: this.now(),
    }
    await this.jobs.put(record)
    return {
      jobId,
      clientImageId,
      sourceSha256,
      sourceUrl: record.sourceUrl,
      sourceWidth: record.sourceWidth,
      sourceHeight: record.sourceHeight,
      acknowledgedSequence: 0,
    }
  }

  private async ownedActive(jobId: string, sender: Sender): Promise<ActiveJobRecord> {
    const record = await this.jobs.get(jobId)
    if (!record) {
      throw new BackgroundOperationError(
        'ACTIVE_JOB_NOT_FOUND',
        'The active translation job could not be recovered.',
        true,
      )
    }
    if (!sameOwner(record, sender)) {
      throw new BackgroundOperationError(
        'JOB_OWNER_MISMATCH',
        'This document does not own the requested translation job.',
      )
    }
    assertSenderDocument(sender, record.pageUrl)
    return record
  }

  private artifactFrom(record: ActiveJobRecord): PageArtifactRecord {
    return {
      tabId: record.tabId,
      frameId: record.frameId,
      pageSessionId: record.pageSessionId,
      pageUrl: record.pageUrl,
      jobId: record.jobId,
      sourceSha256: record.sourceSha256,
      sourceUrl: record.sourceUrl,
      sourceWidth: record.sourceWidth,
      sourceHeight: record.sourceHeight,
      regionIds: record.regionIds,
      patchIds: record.patchIds,
      fontIds: record.fontIds,
      createdAtUnixMs: record.createdAtUnixMs,
    }
  }

  private async recordDeliveredUpdates(
    record: ActiveJobRecord,
    batch: JobUpdateBatch,
  ): Promise<void> {
    if (batch.jobId !== record.jobId) {
      throw new BackgroundOperationError(
        'UPDATE_IDENTITY_MISMATCH',
        'The local translation updates did not match the active job.',
      )
    }
    const regions = batch.updates
      .filter((update) => update.type === 'regionReady')
      .map((update) => update.region)
    const next: ActiveJobRecord = {
      ...record,
      deliveredSequence: Math.max(record.deliveredSequence, batch.nextSequence),
      regionIds: unique([...record.regionIds, ...regions.map((region) => region.id)]),
      patchIds: unique([
        ...record.patchIds,
        ...regions.map((region) => region.patch.blobId),
      ]),
      fontIds: unique([
        ...record.fontIds,
        ...regions.map((region) => region.style.fontId),
      ]),
    }
    await this.jobs.put(next)
  }

  private async updates(
    message: Extract<BackgroundRequest, { type: 'job:updates' }>,
    sender: Sender,
  ): Promise<JobUpdateBatch> {
    const record = await this.ownedActive(message.jobId, sender)
    if (message.after !== record.acknowledgedSequence) {
      throw new BackgroundOperationError(
        'UPDATE_CURSOR_MISMATCH',
        'The update cursor does not match the last installed page update.',
        true,
      )
    }
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
    ) {
      throw new BackgroundOperationError(
        'UPDATE_ACK_OUT_OF_RANGE',
        'The page tried to acknowledge updates it has not received.',
      )
    }
    if (message.terminalType && message.sequence !== record.deliveredSequence) {
      throw new BackgroundOperationError(
        'TERMINAL_ACK_OUT_OF_RANGE',
        'A terminal update must acknowledge the complete delivered batch.',
      )
    }
    if (message.terminalType) {
      if (record.regionIds.length > 0) {
        await this.artifacts.put(this.artifactFrom(record))
      }
      await this.jobs.remove(record.jobId)
      return
    }
    await this.jobs.put({ ...record, acknowledgedSequence: message.sequence })
  }

  private async viewport(
    message: Extract<BackgroundRequest, { type: 'job:viewport' }>,
    sender: Sender,
  ): Promise<void> {
    const record = await this.ownedActive(message.jobId, sender)
    const update: ViewportUpdate = {
      visibleRects: message.visibleRects,
      active: message.active,
    }
    if (this.fixture) {
      this.fixture.viewport(record, update)
      return
    }
    await this.companion.updateViewport(record.jobId, update)
  }

  private async patch(
    message: Extract<BackgroundRequest, { type: 'job:patch' }>,
    sender: Sender,
  ): Promise<PatchPayload> {
    const record = await this.ownedActive(message.jobId, sender)
    if (!record.patchIds.includes(message.patchId)) {
      throw new BackgroundOperationError(
        'PATCH_JOB_MISMATCH',
        'The requested patch does not belong to this translation job.',
      )
    }
    const bytes = this.fixture
      ? await this.fixture.patch(record, message.patchId)
      : await this.companion.getPatch(message.patchId, message.mimeType)
    validateImageBytes(bytes, 'image/png', DEFAULT_IMAGE_LIMITS)
    return {
      patchId: message.patchId,
      mimeType: 'image/png',
      bytes,
    }
  }

  private async removeRecord(record: ActiveJobRecord): Promise<void> {
    await this.jobs.remove(record.jobId)
    await this.artifacts.remove(record.jobId)
  }

  private async cancelRecord(record: ActiveJobRecord): Promise<void> {
    if (!this.fixture) {
      try {
        await this.companion.cancelJob(record.jobId)
      } finally {
        await this.removeRecord(record)
      }
      return
    }
    await this.removeRecord(record)
  }

  private async cancelJob(jobId: string, sender: Sender): Promise<void> {
    const record = await this.jobs.get(jobId)
    if (!record) return
    if (!sameOwner(record, sender)) {
      throw new BackgroundOperationError(
        'JOB_OWNER_MISMATCH',
        'This document does not own the requested translation job.',
      )
    }
    assertSenderDocument(sender, record.pageUrl)
    await this.cancelRecord(record)
  }

  private candidateForRecord(
    record: ActiveJobRecord,
    candidates: readonly RecoveryCandidate[],
  ): RecoveryCandidate | undefined {
    return candidates.find(
      (candidate) =>
        normalizeSourceUrl(candidate.sourceUrl, record.pageUrl) === record.sourceUrl &&
        candidate.naturalWidth === record.sourceWidth &&
        candidate.naturalHeight === record.sourceHeight,
    )
  }

  private async verifyRecoverySource(
    record: ActiveJobRecord,
    candidate: RecoveryCandidate,
  ): Promise<boolean> {
    if (candidate.sourceSha256) return candidate.sourceSha256 === record.sourceSha256
    if (this.fixture) {
      return (
        (await sha256Hex(
          await this.fixture.sourceImage(record.sourceWidth, record.sourceHeight),
        )) === record.sourceSha256
      )
    }
    const acquired = await acquireRemoteImage(record.sourceUrl, {
      pageOrigin: new URL(record.pageUrl).origin,
    })
    return (
      acquired.width === record.sourceWidth &&
      acquired.height === record.sourceHeight &&
      (await sha256Hex(acquired.bytes)) === record.sourceSha256
    )
  }

  private async recover(
    message: Extract<BackgroundRequest, { type: 'jobs:recover' }>,
    sender: Sender,
  ): Promise<RecoveredJob[]> {
    const { tabId, frameId } = senderLocation(sender)
    assertSenderDocument(sender, message.pageUrl)
    const records = await this.jobs.forPage(tabId, frameId, message.pageSessionId)
    const recovered: RecoveredJob[] = []
    for (const record of records) {
      const candidate = this.candidateForRecord(record, message.candidates)
      try {
        if (!candidate || !(await this.verifyRecoverySource(record, candidate))) {
          await this.cancelRecord(record)
          continue
        }
        recovered.push({
          jobId: record.jobId,
          clientImageId: record.clientImageId,
          sourceSha256: record.sourceSha256,
          sourceUrl: record.sourceUrl,
          sourceWidth: record.sourceWidth,
          sourceHeight: record.sourceHeight,
          pageIndex: record.pageIndex,
          acknowledgedSequence: record.acknowledgedSequence,
        })
      } catch {
        await this.cancelRecord(record).catch(() => this.removeRecord(record))
      }
    }
    return recovered
  }

  private async ownedArtifact(jobId: string, sender: Sender): Promise<PageArtifactRecord> {
    const active = await this.jobs.get(jobId)
    if (active) {
      if (!sameOwner(active, sender)) {
        throw new BackgroundOperationError(
          'RESULT_OWNER_MISMATCH',
          'This document does not own the requested translation artifact.',
        )
      }
      assertSenderDocument(sender, active.pageUrl)
      return this.artifactFrom(active)
    }
    const artifact = await this.artifacts.get(jobId)
    if (!artifact || !sameOwner(artifact, sender)) {
      throw new BackgroundOperationError(
        'RESULT_OWNER_MISMATCH',
        'This document does not own the requested translation artifact.',
      )
    }
    assertSenderDocument(sender, artifact.pageUrl)
    return artifact
  }

  private async lookup(
    message: Extract<BackgroundRequest, { type: 'dictionary:lookup' }>,
    sender: Sender,
  ): Promise<LookupResult> {
    const jobId = message.request.jobId
    const regionId = message.request.regionId
    if (!jobId || !regionId) {
      throw new BackgroundOperationError(
        'LOOKUP_IDENTITY_MISSING',
        'Dictionary lookups must identify a translated region.',
      )
    }
    const artifact = await this.ownedArtifact(jobId, sender)
    if (!artifact.regionIds.includes(regionId)) {
      throw new BackgroundOperationError(
        'LOOKUP_REGION_MISMATCH',
        'The requested dictionary region does not belong to this job.',
      )
    }
    return this.fixture
      ? this.fixture.lookup(message.request)
      : this.companion.lookup(message.request)
  }

  private async font(
    message: Extract<BackgroundRequest, { type: 'font:get' }>,
    sender: Sender,
  ): Promise<FontPayload> {
    const artifact = await this.ownedArtifact(message.jobId, sender)
    if (!artifact.fontIds.includes(message.fontId)) {
      throw new BackgroundOperationError(
        'FONT_RESULT_MISMATCH',
        'The requested font does not belong to this translation job.',
      )
    }
    return {
      fontId: message.fontId,
      bytes: this.fixture ? this.fixture.font() : await this.companion.getFont(message.fontId),
    }
  }

  private async cancelPage(pageSessionId: string, sender: Sender): Promise<void> {
    const { tabId, frameId } = senderLocation(sender)
    await this.prefetches.cancelIf(
      (identity) =>
        identity.tabId === tabId &&
        identity.frameId === frameId &&
        identity.pageSessionId === pageSessionId,
    )
    const records = await this.jobs.forPage(tabId, frameId, pageSessionId)
    await Promise.allSettled(records.map((record) => this.cancelRecord(record)))
    await this.artifacts.removeForPage(tabId, frameId, pageSessionId)
  }

  async route(message: BackgroundRequest, sender: Sender): Promise<unknown> {
    switch (message.type) {
      case 'popup:prepare':
        return this.prepareContent()
      case 'popup:start':
        return this.startContent(message.scope, message.hskLevel)
      case 'popup:cancel':
        return this.cancelTab(await this.activeTab())
      case 'popup:state':
        return this.popupState()
      case 'setup:status':
        return this.setupStatus()
      case 'setup:start':
        return this.startSetup()
      case 'setup:open-installer':
        return this.openInstaller()
      case 'image:prefetch':
        return this.prefetch(message, sender)
      case 'image:prefetch-cancel':
        return this.cancelPrefetch(message, sender)
      case 'job:submit':
        return this.submit(message, sender)
      case 'job:updates':
        return this.updates(message, sender)
      case 'job:ack':
        return this.acknowledge(message, sender)
      case 'job:viewport':
        return this.viewport(message, sender)
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
      case 'font:get':
        return this.font(message, sender)
    }
  }

  async cancelJobsForTab(tabId: number): Promise<void> {
    await this.prefetches.cancelIf((identity) => identity.tabId === tabId)
    const records = await this.jobs.forTab(tabId)
    await Promise.allSettled(records.map((record) => this.cancelRecord(record)))
    await this.artifacts.removeForTab(tabId)
    await this.pendingPermissions.removeForTab(tabId)
  }
}

declare global {
  var __hmtBackgroundRegistered: boolean | undefined
}

const BACKGROUND_MESSAGE_TYPES = new Set([
  'popup:prepare',
  'popup:start',
  'popup:cancel',
  'popup:state',
  'setup:status',
  'setup:start',
  'setup:open-installer',
  'image:prefetch',
  'image:prefetch-cancel',
  'job:submit',
  'job:updates',
  'job:ack',
  'job:viewport',
  'job:patch',
  'job:cancel',
  'jobs:recover',
  'jobs:cancel-page',
  'dictionary:lookup',
  'font:get',
])

function looksLikeBackgroundRequest(value: unknown): boolean {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    BACKGROUND_MESSAGE_TYPES.has(String((value as Record<string, unknown>).type))
  )
}

export function registerBackgroundHandlers(): void {
  if (globalThis.__hmtBackgroundRegistered) return
  globalThis.__hmtBackgroundRegistered = true
  const router = new BackgroundRouter()
  browser.runtime.onMessage.addListener(
    async (raw: unknown, sender): Promise<MessageResponse<unknown> | undefined> => {
      if (!looksLikeBackgroundRequest(raw)) return undefined
      if (sender.id !== browser.runtime.id) {
        return {
          ok: false,
          error: {
            code: 'INVALID_MESSAGE_SENDER',
            message: 'The runtime message did not come from this extension.',
            retryable: false,
          },
        }
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
  browser.tabs.onUpdated.addListener((tabId, changeInfo) => {
    if (changeInfo.url) void router.cancelJobsForTab(tabId)
  })
}
