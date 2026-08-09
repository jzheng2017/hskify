import {
  MAX_DOCUMENT_BLOCKS,
  MAX_DOCUMENT_UTF8_BYTES,
  parseImageJobRequest,
  parseTranslationSettings,
  type ImageJobRequest,
  type TranslationSettings,
} from '../contracts/browser'
import type { StorageArea } from './settings'

const ACTIVE_JOB_PREFIX = 'hskify.activeJob.'
const PAGE_ARTIFACT_PREFIX = 'hskify.pageArtifact.'

export type ImageJobSource = {
  kind: 'image'
  clientImageId: string
  sourceUrl: string
  sourceWidth: number
  sourceHeight: number
  sourceIndex: number
  request: ImageJobRequest
  uploadedBytes: number
}

export type DocumentJobSource = {
  kind: 'document'
  settings: TranslationSettings
  blockCount: number
  uploadedBytes: number
}

export type JobSource = ImageJobSource | DocumentJobSource

export type ActiveJobRecord = {
  tabId: number
  frameId: number
  pageSessionId: string
  pageUrl: string
  jobId: string
  sourceSha256: string
  source: JobSource
  submittedAtUnixMs: number
  acknowledgedSequence: number
  deliveredSequence: number
  itemIds: string[]
  patchIds: string[]
  fontIds: string[]
  createdAtUnixMs: number
}

export type PageArtifactRecord = {
  tabId: number
  frameId: number
  pageSessionId: string
  pageUrl: string
  jobId: string
  sourceSha256: string
  source: Readonly<
    | {
        kind: 'image'
        sourceUrl: string
        sourceWidth: number
        sourceHeight: number
        sourceIndex: number
      }
    | { kind: 'document' }
  >
  itemIds: string[]
  patchIds: string[]
  fontIds: string[]
  createdAtUnixMs: number
}

function isInteger(value: unknown, minimum = 0): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= minimum
}

function isString(value: unknown, maximum = 32_768): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= maximum
}

function isSha256(value: unknown): value is string {
  return typeof value === 'string' && /^[a-f0-9]{64}$/u.test(value)
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.length <= 10_000 && value.every((item) => isString(item, 512))
  )
}

function parseSource(value: unknown): JobSource | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const source = value as Record<string, unknown>
  if (source.kind === 'document') {
    if (
      Object.keys(source).some(
        (key) => !['kind', 'settings', 'blockCount', 'uploadedBytes'].includes(key),
      )
    )
      return undefined
    try {
      const settings = parseTranslationSettings(source.settings, 'source.settings')
      return isInteger(source.blockCount, 1) &&
        source.blockCount <= MAX_DOCUMENT_BLOCKS &&
        isInteger(source.uploadedBytes, 1) &&
        source.uploadedBytes <= MAX_DOCUMENT_UTF8_BYTES
        ? {
            kind: 'document',
            settings,
            blockCount: source.blockCount,
            uploadedBytes: source.uploadedBytes,
          }
        : undefined
    } catch {
      return undefined
    }
  }
  if (source.kind !== 'image') return undefined
  if (
    Object.keys(source).some(
      (key) =>
        ![
          'kind',
          'clientImageId',
          'sourceUrl',
          'sourceWidth',
          'sourceHeight',
          'sourceIndex',
          'request',
          'uploadedBytes',
        ].includes(key),
    )
  )
    return undefined
  try {
    const request = parseImageJobRequest(source.request)
    if (
      !isString(source.clientImageId, 512) ||
      !isString(source.sourceUrl) ||
      !isInteger(source.sourceWidth, 1) ||
      !isInteger(source.sourceHeight, 1) ||
      !isInteger(source.sourceIndex) ||
      !isInteger(source.uploadedBytes, 1) ||
      request.clientImageId !== source.clientImageId ||
      request.naturalWidth !== source.sourceWidth ||
      request.naturalHeight !== source.sourceHeight ||
      request.sourceIndex !== source.sourceIndex
    )
      return undefined
    return {
      kind: 'image',
      clientImageId: source.clientImageId,
      sourceUrl: source.sourceUrl,
      sourceWidth: source.sourceWidth,
      sourceHeight: source.sourceHeight,
      sourceIndex: source.sourceIndex,
      request,
      uploadedBytes: source.uploadedBytes,
    }
  } catch {
    return undefined
  }
}

function isActiveJobRecord(value: unknown): value is ActiveJobRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  if (
    Object.keys(record).some(
      (key) =>
        ![
          'tabId',
          'frameId',
          'pageSessionId',
          'pageUrl',
          'jobId',
          'sourceSha256',
          'source',
          'submittedAtUnixMs',
          'acknowledgedSequence',
          'deliveredSequence',
          'itemIds',
          'patchIds',
          'fontIds',
          'createdAtUnixMs',
        ].includes(key),
    )
  )
    return false
  const source = parseSource(record.source)
  return (
    !!source &&
    isInteger(record.tabId) &&
    isInteger(record.frameId) &&
    isString(record.pageSessionId, 256) &&
    isString(record.pageUrl) &&
    isString(record.jobId, 512) &&
    isSha256(record.sourceSha256) &&
    (source.kind === 'document' ||
      (source.request.sourceSha256 === record.sourceSha256 &&
        source.request.pageSessionId === record.pageSessionId)) &&
    isInteger(record.submittedAtUnixMs) &&
    isInteger(record.acknowledgedSequence) &&
    isInteger(record.deliveredSequence) &&
    record.deliveredSequence >= record.acknowledgedSequence &&
    isStringArray(record.itemIds) &&
    isStringArray(record.patchIds) &&
    isStringArray(record.fontIds) &&
    isInteger(record.createdAtUnixMs) &&
    record.createdAtUnixMs >= record.submittedAtUnixMs
  )
}

function isArtifactSource(value: unknown): value is PageArtifactRecord['source'] {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const source = value as Record<string, unknown>
  if (source.kind === 'document') return Object.keys(source).length === 1
  return (
    source.kind === 'image' &&
    Object.keys(source).every((key) =>
      ['kind', 'sourceUrl', 'sourceWidth', 'sourceHeight', 'sourceIndex'].includes(key),
    ) &&
    isString(source.sourceUrl) &&
    isInteger(source.sourceWidth, 1) &&
    isInteger(source.sourceHeight, 1) &&
    isInteger(source.sourceIndex)
  )
}

function isPageArtifactRecord(value: unknown): value is PageArtifactRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return (
    Object.keys(record).every((key) =>
      [
        'tabId',
        'frameId',
        'pageSessionId',
        'pageUrl',
        'jobId',
        'sourceSha256',
        'source',
        'itemIds',
        'patchIds',
        'fontIds',
        'createdAtUnixMs',
      ].includes(key),
    ) &&
    isInteger(record.tabId) &&
    isInteger(record.frameId) &&
    isString(record.pageSessionId, 256) &&
    isString(record.pageUrl) &&
    isString(record.jobId, 512) &&
    isSha256(record.sourceSha256) &&
    isArtifactSource(record.source) &&
    isStringArray(record.itemIds) &&
    isStringArray(record.patchIds) &&
    isStringArray(record.fontIds) &&
    isInteger(record.createdAtUnixMs)
  )
}

export class ActiveJobStore {
  constructor(private readonly storage: StorageArea = browser.storage.local) {}

  async put(record: ActiveJobRecord): Promise<void> {
    await this.storage.set({ [`${ACTIVE_JOB_PREFIX}${record.jobId}`]: record })
  }

  async get(jobId: string): Promise<ActiveJobRecord | undefined> {
    const key = `${ACTIVE_JOB_PREFIX}${jobId}`
    const values = await this.storage.get(key)
    return isActiveJobRecord(values[key]) ? values[key] : undefined
  }

  async list(): Promise<ActiveJobRecord[]> {
    const values = await this.storage.get(null)
    return Object.entries(values)
      .filter(([key]) => key.startsWith(ACTIVE_JOB_PREFIX))
      .map(([, value]) => value)
      .filter(isActiveJobRecord)
      .sort((left, right) => left.createdAtUnixMs - right.createdAtUnixMs)
  }

  async forPage(tabId: number, frameId: number, pageSessionId: string): Promise<ActiveJobRecord[]> {
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

  async forPage(
    tabId: number,
    frameId: number,
    pageSessionId: string,
  ): Promise<PageArtifactRecord[]> {
    return (await this.forTab(tabId)).filter(
      (record) => record.frameId === frameId && record.pageSessionId === pageSessionId,
    )
  }

  async removeForPage(tabId: number, frameId: number, pageSessionId: string): Promise<void> {
    await Promise.all(
      (await this.forPage(tabId, frameId, pageSessionId)).map((record) =>
        this.remove(record.jobId),
      ),
    )
  }

  async removeForTab(tabId: number): Promise<void> {
    await Promise.all((await this.forTab(tabId)).map((record) => this.remove(record.jobId)))
  }
}
