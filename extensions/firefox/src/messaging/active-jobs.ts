import {
  parseBrowserJobRequest,
  type BrowserJobRequest,
  type HskLevel,
} from '../contracts/browser'
import type { StorageArea } from './settings'

const ACTIVE_JOB_PREFIX = 'hmt.activeJob.'
const PAGE_ARTIFACT_PREFIX = 'hmt.pageArtifact.'

export type ActiveJobRecord = {
  tabId: number
  frameId: number
  pageSessionId: string
  pageUrl: string
  clientImageId: string
  jobId: string
  sourceSha256: string
  sourceUrl: string
  sourceWidth: number
  sourceHeight: number
  pageIndex: number
  hskLevel: HskLevel
  submittedRequest: BrowserJobRequest
  uploadedImageBytes: number
  submittedAtUnixMs: number
  acknowledgedSequence: number
  deliveredSequence: number
  regionIds: string[]
  patchIds: string[]
  fontIds: string[]
  createdAtUnixMs: number
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.length <= 10_000 &&
    value.every(
      (item) =>
        typeof item === 'string' &&
        item.length > 0 &&
        item.length <= 512,
    )
  )
}

function isActiveJobRecord(value: unknown): value is ActiveJobRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  let submittedRequest: BrowserJobRequest
  try {
    submittedRequest = parseBrowserJobRequest(record.submittedRequest)
  } catch {
    return false
  }
  return (
    typeof record.tabId === 'number' &&
    Number.isSafeInteger(record.tabId) &&
    typeof record.frameId === 'number' &&
    Number.isSafeInteger(record.frameId) &&
    typeof record.pageSessionId === 'string' &&
    record.pageSessionId.length > 0 &&
    typeof record.pageUrl === 'string' &&
    record.pageUrl.length > 0 &&
    typeof record.clientImageId === 'string' &&
    record.clientImageId.length > 0 &&
    typeof record.jobId === 'string' &&
    record.jobId.length > 0 &&
    typeof record.sourceSha256 === 'string' &&
    /^[a-f0-9]{64}$/u.test(record.sourceSha256) &&
    typeof record.sourceUrl === 'string' &&
    record.sourceUrl.length > 0 &&
    typeof record.sourceWidth === 'number' &&
    Number.isSafeInteger(record.sourceWidth) &&
    record.sourceWidth > 0 &&
    typeof record.sourceHeight === 'number' &&
    Number.isSafeInteger(record.sourceHeight) &&
    record.sourceHeight > 0 &&
    typeof record.pageIndex === 'number' &&
    Number.isSafeInteger(record.pageIndex) &&
    record.pageIndex >= 0 &&
    submittedRequest.clientImageId === record.clientImageId &&
    submittedRequest.sourceSha256 === record.sourceSha256 &&
    submittedRequest.naturalWidth === record.sourceWidth &&
    submittedRequest.naturalHeight === record.sourceHeight &&
    submittedRequest.pageSessionId === record.pageSessionId &&
    submittedRequest.pageIndex === record.pageIndex &&
    submittedRequest.settings.hskLevel === record.hskLevel &&
    (record.hskLevel === 1 ||
      record.hskLevel === 2 ||
      record.hskLevel === 3 ||
      record.hskLevel === 4 ||
      record.hskLevel === 5 ||
      record.hskLevel === 6) &&
    typeof record.uploadedImageBytes === 'number' &&
    Number.isSafeInteger(record.uploadedImageBytes) &&
    record.uploadedImageBytes > 0 &&
    typeof record.submittedAtUnixMs === 'number' &&
    Number.isSafeInteger(record.submittedAtUnixMs) &&
    record.submittedAtUnixMs >= 0 &&
    typeof record.acknowledgedSequence === 'number' &&
    Number.isSafeInteger(record.acknowledgedSequence) &&
    record.acknowledgedSequence >= 0 &&
    typeof record.deliveredSequence === 'number' &&
    Number.isSafeInteger(record.deliveredSequence) &&
    record.deliveredSequence >= record.acknowledgedSequence &&
    isStringArray(record.regionIds) &&
    isStringArray(record.patchIds) &&
    isStringArray(record.fontIds) &&
    typeof record.createdAtUnixMs === 'number' &&
    Number.isSafeInteger(record.createdAtUnixMs) &&
    record.createdAtUnixMs >= record.submittedAtUnixMs
  )
}

export class ActiveJobStore {
  constructor(private readonly storage: StorageArea = browser.storage.local) {}

  async put(record: ActiveJobRecord): Promise<void> {
    await this.storage.set({ [`${ACTIVE_JOB_PREFIX}${record.jobId}`]: record })
  }

  async get(jobId: string): Promise<ActiveJobRecord | undefined> {
    const key = `${ACTIVE_JOB_PREFIX}${jobId}`
    const result = await this.storage.get(key)
    return isActiveJobRecord(result[key]) ? result[key] : undefined
  }

  async list(): Promise<ActiveJobRecord[]> {
    const values = await this.storage.get(null)
    return Object.entries(values)
      .filter(([key]) => key.startsWith(ACTIVE_JOB_PREFIX))
      .map(([, value]) => value)
      .filter(isActiveJobRecord)
      .sort((left, right) => left.createdAtUnixMs - right.createdAtUnixMs)
  }

  async forPage(
    tabId: number,
    frameId: number,
    pageSessionId: string,
  ): Promise<ActiveJobRecord[]> {
    return (await this.list()).filter(
      (record) =>
        record.tabId === tabId &&
        record.frameId === frameId &&
        record.pageSessionId === pageSessionId,
    )
  }

  async forTab(tabId: number): Promise<ActiveJobRecord[]> {
    return (await this.list()).filter((record) => record.tabId === tabId)
  }

  async remove(jobId: string): Promise<void> {
    await this.storage.remove(`${ACTIVE_JOB_PREFIX}${jobId}`)
  }
}

export type PageArtifactRecord = {
  tabId: number
  frameId: number
  pageSessionId: string
  pageUrl: string
  jobId: string
  sourceSha256: string
  sourceUrl: string
  sourceWidth: number
  sourceHeight: number
  regionIds: string[]
  patchIds: string[]
  fontIds: string[]
  createdAtUnixMs: number
}

function isPageArtifactRecord(value: unknown): value is PageArtifactRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return (
    typeof record.tabId === 'number' &&
    Number.isSafeInteger(record.tabId) &&
    typeof record.frameId === 'number' &&
    Number.isSafeInteger(record.frameId) &&
    typeof record.pageSessionId === 'string' &&
    record.pageSessionId.length > 0 &&
    typeof record.pageUrl === 'string' &&
    record.pageUrl.length > 0 &&
    typeof record.jobId === 'string' &&
    record.jobId.length > 0 &&
    typeof record.sourceSha256 === 'string' &&
    /^[a-f0-9]{64}$/u.test(record.sourceSha256) &&
    typeof record.sourceUrl === 'string' &&
    record.sourceUrl.length > 0 &&
    typeof record.sourceWidth === 'number' &&
    Number.isSafeInteger(record.sourceWidth) &&
    record.sourceWidth > 0 &&
    typeof record.sourceHeight === 'number' &&
    Number.isSafeInteger(record.sourceHeight) &&
    record.sourceHeight > 0 &&
    isStringArray(record.regionIds) &&
    isStringArray(record.patchIds) &&
    isStringArray(record.fontIds) &&
    typeof record.createdAtUnixMs === 'number' &&
    Number.isSafeInteger(record.createdAtUnixMs)
  )
}

export class PageArtifactStore {
  constructor(private readonly storage: StorageArea = browser.storage.session) {}

  async put(record: PageArtifactRecord): Promise<void> {
    await this.storage.set({ [`${PAGE_ARTIFACT_PREFIX}${record.jobId}`]: record })
  }

  async get(jobId: string): Promise<PageArtifactRecord | undefined> {
    const key = `${PAGE_ARTIFACT_PREFIX}${jobId}`
    const values = await this.storage.get(key)
    return isPageArtifactRecord(values[key]) ? values[key] : undefined
  }

  async forTab(tabId: number): Promise<PageArtifactRecord[]> {
    const values = await this.storage.get(null)
    return Object.entries(values)
      .filter(([key]) => key.startsWith(PAGE_ARTIFACT_PREFIX))
      .map(([, value]) => value)
      .filter(isPageArtifactRecord)
      .filter((record) => record.tabId === tabId)
  }

  async remove(jobId: string): Promise<void> {
    await this.storage.remove(`${PAGE_ARTIFACT_PREFIX}${jobId}`)
  }

  async removeForPage(
    tabId: number,
    frameId: number,
    pageSessionId: string,
  ): Promise<void> {
    const records = (await this.forTab(tabId)).filter(
      (record) =>
        record.frameId === frameId && record.pageSessionId === pageSessionId,
    )
    await Promise.all(records.map((record) => this.remove(record.jobId)))
  }

  async removeForTab(tabId: number): Promise<void> {
    await Promise.all((await this.forTab(tabId)).map((record) => this.remove(record.jobId)))
  }
}
