import {
  PROTOCOL_VERSION,
  type BrowserJobRequest,
  type BrowserJobStatus,
} from '../contracts/browser'
import {
  validateInlineImage,
  acquireRemoteImage,
  type AcquiredImage,
} from '../acquisition/image-acquisition'
import { ImageValidationError } from '../acquisition/image-format'
import { sha256Hex } from '../acquisition/hash'
import { ActiveJobStore, type ActiveJobRecord } from './active-jobs'
import { CompanionClient, CompanionHttpError } from './companion-client'
import {
  FixtureService,
  fixtureFontBytes,
  fixtureSourceBytes,
} from './fixture-service'
import {
  type BackgroundRequest,
  type DeliveredJobResult,
  type FontPayload,
  type MessageError,
  type MessageResponse,
  type PageState,
  type PopupState,
  type RecoveredJob,
  type SubmittedJob,
} from './messages'
import { loadHskLevel, saveHskLevel } from './settings'

type Sender = browser.runtime.MessageSender

type BackgroundDependencies = {
  jobs: ActiveJobStore
  companion: CompanionClient
  fixture: FixtureService
  now: () => number
}

type DimensionsByJob = {
  width: number
  height: number
}

const DIMENSIONS_PREFIX = 'hmt.activeJobDimensions.'

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
    error instanceof CompanionHttpError
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
    throw new BackgroundOperationError(
      'NO_ACTIVE_TAB',
      'No active webpage tab is available.',
      true,
    )
  }
  return tabId
}

function dimensionsKey(jobId: string): string {
  return `${DIMENSIONS_PREFIX}${jobId}`
}

function isDimensions(value: unknown): value is DimensionsByJob {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    typeof (value as DimensionsByJob).width === 'number' &&
    typeof (value as DimensionsByJob).height === 'number'
  )
}

export class BackgroundRouter {
  private readonly jobs: ActiveJobStore
  private readonly companion: CompanionClient
  private readonly fixture: FixtureService
  private readonly now: () => number

  constructor(dependencies?: Partial<BackgroundDependencies>) {
    this.jobs = dependencies?.jobs ?? new ActiveJobStore()
    this.companion = dependencies?.companion ?? new CompanionClient()
    this.fixture = dependencies?.fixture ?? new FixtureService()
    this.now = dependencies?.now ?? Date.now
  }

  private async activeTab(): Promise<number> {
    return activeTabId(await browser.tabs.query({ active: true, currentWindow: true }))
  }

  private async contentState(tabId: number): Promise<PageState | undefined> {
    try {
      return (await browser.tabs.sendMessage(tabId, {
        type: 'content:state',
      })) as PageState
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
    return (await browser.tabs.sendMessage(tabId, {
      type: 'content:start',
      scope,
      hskLevel,
    })) as PageState
  }

  private async cancelTab(tabId: number): Promise<PageState> {
    try {
      return (await browser.tabs.sendMessage(tabId, {
        type: 'content:cancel',
      })) as PageState
    } catch {
      const records = await this.jobs.forTab(tabId)
      await Promise.allSettled(records.map((record) => this.cancelRecord(record)))
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

  private async acquire(message: Extract<BackgroundRequest, { type: 'job:submit' }>) {
    if (message.fixtureMode) {
      return {
        bytes: fixtureSourceBytes(),
        mimeType: 'image/png',
        width: message.naturalWidth,
        height: message.naturalHeight,
        finalUrl: 'fixture://synthetic-page',
      } satisfies AcquiredImage
    }
    if (message.sourceBytes) {
      return validateInlineImage(message.sourceBytes, message.sourceMimeType)
    }
    return acquireRemoteImage(message.imageUrl, { pageOrigin: message.pageOrigin })
  }

  private async submit(
    message: Extract<BackgroundRequest, { type: 'job:submit' }>,
    sender: Sender,
  ): Promise<SubmittedJob> {
    const { tabId, frameId } = senderLocation(sender)
    const acquired = await this.acquire(message)
    const sourceSha256 = await sha256Hex(acquired.bytes)
    const clientImageId = `${message.pageSessionId}-${message.pageIndex}-${sourceSha256.slice(0, 16)}`
    const request: BrowserJobRequest = {
      protocolVersion: PROTOCOL_VERSION,
      clientImageId,
      sourceSha256,
      sourceMimeType: acquired.mimeType,
      naturalWidth: acquired.width,
      naturalHeight: acquired.height,
      pageSessionId: message.pageSessionId,
      pageIndex: message.pageIndex,
      settings: {
        sourceLanguage: 'en',
        targetLanguage: 'zh-CN',
        hskStandard: '2.0',
        hskLevel: message.hskLevel,
        readingDirection: 'auto',
        translateSoundEffects: false,
      },
      ...(message.precedingContext?.length
        ? { precedingContext: message.precedingContext.slice(-12) }
        : {}),
    }
    const jobId = message.fixtureMode
      ? this.fixture.createJobId(message.pageSessionId, message.pageIndex, sourceSha256)
      : await this.companion.createJob(acquired.bytes, request)
    const record: ActiveJobRecord = {
      tabId,
      frameId,
      pageSessionId: message.pageSessionId,
      clientImageId,
      jobId,
      sourceSha256,
      pageIndex: message.pageIndex,
      fixtureMode: message.fixtureMode,
      createdAtUnixMs: this.now(),
    }
    await Promise.all([
      this.jobs.put(record),
      browser.storage.local.set({
        [dimensionsKey(jobId)]: { width: acquired.width, height: acquired.height },
      }),
    ])
    return { jobId, clientImageId, sourceSha256 }
  }

  private async status(record: ActiveJobRecord): Promise<BrowserJobStatus> {
    return record.fixtureMode
      ? this.fixture.status(record)
      : this.companion.getJobStatus(record.jobId)
  }

  private async poll(jobId: string): Promise<BrowserJobStatus> {
    const record = await this.jobs.get(jobId)
    if (!record) {
      throw new BackgroundOperationError(
        'ACTIVE_JOB_NOT_FOUND',
        'The active translation job could not be recovered.',
        true,
      )
    }
    const status = await this.status(record)
    if (status.state === 'failed' || status.state === 'cancelled') {
      await this.removeRecord(record)
    }
    return status
  }

  private async getDimensions(jobId: string): Promise<DimensionsByJob> {
    const key = dimensionsKey(jobId)
    const values = await browser.storage.local.get(key)
    const dimensions = values[key]
    if (!isDimensions(dimensions)) {
      throw new BackgroundOperationError(
        'JOB_METADATA_MISSING',
        'The image dimensions for this job could not be recovered.',
        true,
      )
    }
    return dimensions
  }

  private async result(jobId: string): Promise<DeliveredJobResult> {
    const record = await this.jobs.get(jobId)
    if (!record) {
      throw new BackgroundOperationError(
        'ACTIVE_JOB_NOT_FOUND',
        'The completed translation job could not be recovered.',
        true,
      )
    }
    const dimensions = await this.getDimensions(jobId)
    const result = record.fixtureMode
      ? this.fixture.result(record, dimensions.width, dimensions.height)
      : await this.companion.getJobResult(jobId)
    if (result.jobId !== record.jobId || result.sourceSha256 !== record.sourceSha256) {
      throw new BackgroundOperationError(
        'RESULT_IDENTITY_MISMATCH',
        'The local translation result did not match the submitted image.',
      )
    }
    const cleanImage = record.fixtureMode
      ? this.fixture.cleanImage()
      : await this.companion.getCleanImage(
          result.cleanImageBlobId,
          result.cleanImageMimeType,
        )
    await this.removeRecord(record)
    return { result, cleanImage }
  }

  private async removeRecord(record: ActiveJobRecord): Promise<void> {
    await Promise.all([
      this.jobs.remove(record.jobId),
      browser.storage.local.remove(dimensionsKey(record.jobId)),
    ])
  }

  private async cancelRecord(record: ActiveJobRecord): Promise<void> {
    if (!record.fixtureMode) {
      try {
        await this.companion.cancelJob(record.jobId)
      } finally {
        await this.removeRecord(record)
      }
      return
    }
    await this.removeRecord(record)
  }

  private async cancelJob(jobId: string): Promise<void> {
    const record = await this.jobs.get(jobId)
    if (record) await this.cancelRecord(record)
  }

  private async recover(
    pageSessionId: string,
    sender: Sender,
  ): Promise<RecoveredJob[]> {
    const { tabId, frameId } = senderLocation(sender)
    const records = await this.jobs.forPage(tabId, frameId, pageSessionId)
    const recovered: RecoveredJob[] = []
    for (const record of records) {
      try {
        recovered.push({
          jobId: record.jobId,
          clientImageId: record.clientImageId,
          sourceSha256: record.sourceSha256,
          pageIndex: record.pageIndex,
          fixtureMode: record.fixtureMode,
          status: await this.status(record),
        })
      } catch {
        await this.removeRecord(record)
      }
    }
    return recovered
  }

  private async font(
    message: Extract<BackgroundRequest, { type: 'font:get' }>,
  ): Promise<FontPayload> {
    return {
      fontId: message.fontId,
      bytes: message.fixtureMode
        ? fixtureFontBytes()
        : await this.companion.getFont(message.fontId),
    }
  }

  async route(message: BackgroundRequest, sender: Sender): Promise<unknown> {
    switch (message.type) {
      case 'popup:start':
        return this.startContent(message.scope, message.hskLevel)
      case 'popup:cancel':
        return this.cancelTab(await this.activeTab())
      case 'popup:state':
        return this.popupState()
      case 'job:submit':
        return this.submit(message, sender)
      case 'job:poll':
        return this.poll(message.jobId)
      case 'job:result':
        return this.result(message.jobId)
      case 'job:cancel':
        return this.cancelJob(message.jobId)
      case 'jobs:recover':
        return this.recover(message.pageSessionId, sender)
      case 'dictionary:lookup':
        return message.fixtureMode
          ? this.fixture.lookup(message.request)
          : this.companion.lookup(message.request)
      case 'font:get':
        return this.font(message)
    }
  }

  async cancelJobsForTab(tabId: number): Promise<void> {
    const records = await this.jobs.forTab(tabId)
    await Promise.allSettled(records.map((record) => this.cancelRecord(record)))
  }
}

function isBackgroundRequest(value: unknown): value is BackgroundRequest {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const type = (value as Record<string, unknown>).type
  return (
    typeof type === 'string' &&
    [
      'popup:start',
      'popup:cancel',
      'popup:state',
      'job:submit',
      'job:poll',
      'job:result',
      'job:cancel',
      'jobs:recover',
      'dictionary:lookup',
      'font:get',
    ].includes(type)
  )
}

declare global {
  var __hmtBackgroundRegistered: boolean | undefined
}

export function registerBackgroundHandlers(): void {
  if (globalThis.__hmtBackgroundRegistered) return
  globalThis.__hmtBackgroundRegistered = true
  const router = new BackgroundRouter()
  browser.runtime.onMessage.addListener(
    async (message: unknown, sender): Promise<MessageResponse<unknown> | undefined> => {
      if (!isBackgroundRequest(message)) return undefined
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
