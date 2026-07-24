import type { BrowserJobStatus } from '../contracts/browser'
import type { StorageArea } from './settings'

const ACTIVE_JOB_PREFIX = 'hmt.activeJob.'

export type ActiveJobRecord = {
  tabId: number
  frameId: number
  pageSessionId: string
  clientImageId: string
  jobId: string
  sourceSha256: string
  pageIndex: number
  fixtureMode: boolean
  createdAtUnixMs: number
}

function isActiveJobRecord(value: unknown): value is ActiveJobRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return (
    typeof record.tabId === 'number' &&
    typeof record.frameId === 'number' &&
    typeof record.pageSessionId === 'string' &&
    typeof record.clientImageId === 'string' &&
    typeof record.jobId === 'string' &&
    typeof record.sourceSha256 === 'string' &&
    typeof record.pageIndex === 'number' &&
    typeof record.fixtureMode === 'boolean' &&
    typeof record.createdAtUnixMs === 'number'
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

export type JobStatusResolver = (record: ActiveJobRecord) => Promise<BrowserJobStatus>
