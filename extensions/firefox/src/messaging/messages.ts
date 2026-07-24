import {
  parseBrowserJobResult,
  parseBrowserJobStatus,
  parseBrowserSetupStatus,
  parseLookupRequest,
  parseLookupResult,
  type BrowserJobRequest,
  type BrowserJobResult,
  type BrowserJobStatus,
  type BrowserSetupStatus,
  type HskLevel,
  type LookupRequest,
  type LookupResult,
} from '../contracts/browser'

const MAX_RUNTIME_BINARY_BYTES = 25 * 1024 * 1024
const MAX_RUNTIME_FONT_BYTES = 12 * 1024 * 1024
const MAX_RECOVERY_CANDIDATES = 512

export type TranslationScope = 'visible' | 'all'

export type PermissionPlan = {
  visibleOrigins: string[]
  allOrigins: string[]
}

export type PopupPrepareMessage = { type: 'popup:prepare' }
export type PopupStartMessage = {
  type: 'popup:start'
  scope: TranslationScope
  hskLevel: HskLevel
}
export type PopupCancelMessage = { type: 'popup:cancel' }
export type PopupStateMessage = { type: 'popup:state' }
export type SetupStatusMessage = { type: 'setup:status' }
export type SetupStartMessage = { type: 'setup:start' }
export type SetupOpenInstallerMessage = { type: 'setup:open-installer' }

export type ContentPrepareMessage = { type: 'content:prepare' }
export type ContentStartMessage = {
  type: 'content:start'
  scope: TranslationScope
  hskLevel: HskLevel
}
export type ContentCancelMessage = { type: 'content:cancel' }
export type ContentStateMessage = { type: 'content:state' }

export type SubmitImageMessage = {
  type: 'job:submit'
  pageSessionId: string
  pageIndex: number
  imageUrl: string
  pageUrl: string
  naturalWidth: number
  naturalHeight: number
  sourceMimeType?: string
  sourceBytes?: ArrayBuffer
  hskLevel: HskLevel
  precedingContext?: BrowserJobRequest['precedingContext']
}

export type PollJobMessage = { type: 'job:poll'; jobId: string }

export type JobSourceIdentity = {
  pageSessionId: string
  sourceUrl: string
  sourceSha256: string
  sourceWidth: number
  sourceHeight: number
}

export type GetJobResultMessage = JobSourceIdentity & {
  type: 'job:result'
  jobId: string
}

export type CancelJobMessage = { type: 'job:cancel'; jobId: string }

export type RecoveryCandidate = {
  sourceUrl: string
  naturalWidth: number
  naturalHeight: number
  sourceSha256?: string
}

export type RecoverJobsMessage = {
  type: 'jobs:recover'
  pageSessionId: string
  pageUrl: string
  candidates: RecoveryCandidate[]
}

export type CancelPageJobsMessage = {
  type: 'jobs:cancel-page'
  pageSessionId: string
}

export type LookupMessage = {
  type: 'dictionary:lookup'
  request: LookupRequest
}

export type FontMessage = {
  type: 'font:get'
  jobId: string
  fontId: string
}

export type BackgroundRequest =
  | PopupPrepareMessage
  | PopupStartMessage
  | PopupCancelMessage
  | PopupStateMessage
  | SetupStatusMessage
  | SetupStartMessage
  | SetupOpenInstallerMessage
  | SubmitImageMessage
  | PollJobMessage
  | GetJobResultMessage
  | CancelJobMessage
  | RecoverJobsMessage
  | CancelPageJobsMessage
  | LookupMessage
  | FontMessage

export type ContentRequest =
  | ContentPrepareMessage
  | ContentStartMessage
  | ContentCancelMessage
  | ContentStateMessage

export type PageState = {
  state: 'idle' | 'running' | 'complete' | 'cancelled' | 'failed'
  current: number
  total: number
  stage?: string
  message: string
}

export type SubmittedJob = {
  jobId: string
  clientImageId: string
  sourceSha256: string
  sourceUrl: string
  sourceWidth: number
  sourceHeight: number
}

export type DeliveredJobResult = {
  result: BrowserJobResult
  cleanImage: ArrayBuffer
}

export type FontPayload = {
  fontId: string
  bytes: ArrayBuffer
}

export type RecoveredJob = {
  jobId: string
  clientImageId: string
  sourceSha256: string
  sourceUrl: string
  sourceWidth: number
  sourceHeight: number
  pageIndex: number
  status: BrowserJobStatus
}

export type PopupState = PageState & { hskLevel: HskLevel }

export type MessageError = {
  code: string
  message: string
  retryable: boolean
}

export type MessageResponse<T> =
  | { ok: true; value: T }
  | { ok: false; error: MessageError }

export type MessageResultMap = {
  'popup:prepare': PermissionPlan
  'popup:start': PageState
  'popup:cancel': PageState
  'popup:state': PopupState
  'setup:status': BrowserSetupStatus
  'setup:start': BrowserSetupStatus
  'setup:open-installer': undefined
  'job:submit': SubmittedJob
  'job:poll': BrowserJobStatus
  'job:result': DeliveredJobResult
  'job:cancel': undefined
  'jobs:recover': RecoveredJob[]
  'jobs:cancel-page': undefined
  'dictionary:lookup': LookupResult
  'font:get': FontPayload
}

export type RequestOfType<T extends BackgroundRequest['type']> = Extract<
  BackgroundRequest,
  { type: T }
>

export class RuntimeMessageError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable: boolean,
  ) {
    super(message)
    this.name = 'RuntimeMessageError'
  }
}

class RuntimeMessageValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'RuntimeMessageValidationError'
  }
}

type UnknownRecord = Record<string, unknown>

function record(value: unknown, path = '$'): UnknownRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new RuntimeMessageValidationError(`${path} must be an object.`)
  }
  return value as UnknownRecord
}

function exact(item: UnknownRecord, allowed: readonly string[], path = '$'): void {
  const expected = new Set(allowed)
  const unexpected = Object.keys(item).find((key) => !expected.has(key))
  if (unexpected) {
    throw new RuntimeMessageValidationError(`${path}.${unexpected} is not permitted.`)
  }
}

function string(
  value: unknown,
  path: string,
  maximum = 4_096,
): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > maximum) {
    throw new RuntimeMessageValidationError(
      `${path} must be a non-empty string no longer than ${maximum} characters.`,
    )
  }
  return value
}

function integer(value: unknown, path: string, minimum = 0, maximum = 1_000_000): number {
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new RuntimeMessageValidationError(
      `${path} must be an integer from ${minimum} through ${maximum}.`,
    )
  }
  return value
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') {
    throw new RuntimeMessageValidationError(`${path} must be a boolean.`)
  }
  return value
}

function hskLevel(value: unknown, path: string): HskLevel {
  if (
    value !== 1 &&
    value !== 2 &&
    value !== 3 &&
    value !== 4 &&
    value !== 5 &&
    value !== 6
  ) {
    throw new RuntimeMessageValidationError(`${path} must be an HSK level from 1 through 6.`)
  }
  return value
}

function translationScope(value: unknown, path: string): TranslationScope {
  if (value !== 'visible' && value !== 'all') {
    throw new RuntimeMessageValidationError(`${path} must be "visible" or "all".`)
  }
  return value
}

function sha256(value: unknown, path: string): string {
  const parsed = string(value, path, 64)
  if (!/^[a-f0-9]{64}$/u.test(parsed)) {
    throw new RuntimeMessageValidationError(`${path} must be a lowercase SHA-256 digest.`)
  }
  return parsed
}

function arrayBuffer(value: unknown, path: string, maximum: number): ArrayBuffer {
  if (!(value instanceof ArrayBuffer) || value.byteLength === 0 || value.byteLength > maximum) {
    throw new RuntimeMessageValidationError(
      `${path} must be an ArrayBuffer between 1 and ${maximum} bytes.`,
    )
  }
  return value
}

function stringArray(value: unknown, path: string): string[] {
  if (!Array.isArray(value) || value.length > MAX_RECOVERY_CANDIDATES) {
    throw new RuntimeMessageValidationError(`${path} must be a bounded string array.`)
  }
  return value.map((entry, index) => string(entry, `${path}[${index}]`, 512))
}

export function parsePageState(value: unknown): PageState {
  return pageState(value, false) as PageState
}

function pageState(value: unknown, includeHsk = false): PopupState | PageState {
  const item = record(value)
  exact(
    item,
    includeHsk
      ? ['state', 'current', 'total', 'stage', 'message', 'hskLevel']
      : ['state', 'current', 'total', 'stage', 'message'],
  )
  if (
    item.state !== 'idle' &&
    item.state !== 'running' &&
    item.state !== 'complete' &&
    item.state !== 'cancelled' &&
    item.state !== 'failed'
  ) {
    throw new RuntimeMessageValidationError('$.state is invalid.')
  }
  const parsed = {
    state: item.state,
    current: integer(item.current, '$.current'),
    total: integer(item.total, '$.total'),
    ...(item.stage === undefined ? {} : { stage: string(item.stage, '$.stage', 128) }),
    message: string(item.message, '$.message', 2_048),
  } satisfies PageState
  return includeHsk
    ? { ...parsed, hskLevel: hskLevel(item.hskLevel, '$.hskLevel') }
    : parsed
}

function parsePrecedingContext(value: unknown): BrowserJobRequest['precedingContext'] {
  if (!Array.isArray(value) || value.length > 12) {
    throw new RuntimeMessageValidationError('$.precedingContext must contain at most 12 entries.')
  }
  return value.map((entry, index) => {
    const item = record(entry, `$.precedingContext[${index}]`)
    exact(item, ['sourceEnglish', 'chinese'], `$.precedingContext[${index}]`)
    return {
      sourceEnglish: string(
        item.sourceEnglish,
        `$.precedingContext[${index}].sourceEnglish`,
        4_096,
      ),
      chinese: string(item.chinese, `$.precedingContext[${index}].chinese`, 4_096),
    }
  })
}

function parseRecoveryCandidate(value: unknown, index: number): RecoveryCandidate {
  const path = `$.candidates[${index}]`
  const item = record(value, path)
  exact(item, ['sourceUrl', 'naturalWidth', 'naturalHeight', 'sourceSha256'], path)
  return {
    sourceUrl: string(item.sourceUrl, `${path}.sourceUrl`, 8_192),
    naturalWidth: integer(item.naturalWidth, `${path}.naturalWidth`, 1, 32_768),
    naturalHeight: integer(item.naturalHeight, `${path}.naturalHeight`, 1, 32_768),
    ...(item.sourceSha256 === undefined
      ? {}
      : { sourceSha256: sha256(item.sourceSha256, `${path}.sourceSha256`) }),
  }
}

export function parseBackgroundRequest(value: unknown): BackgroundRequest {
  const item = record(value)
  const type = string(item.type, '$.type', 64)
  switch (type) {
    case 'popup:prepare':
    case 'popup:cancel':
    case 'popup:state':
    case 'setup:status':
    case 'setup:start':
    case 'setup:open-installer':
      exact(item, ['type'])
      return { type }
    case 'popup:start':
      exact(item, ['type', 'scope', 'hskLevel'])
      return {
        type,
        scope: translationScope(item.scope, '$.scope'),
        hskLevel: hskLevel(item.hskLevel, '$.hskLevel'),
      }
    case 'job:submit': {
      exact(item, [
        'type',
        'pageSessionId',
        'pageIndex',
        'imageUrl',
        'pageUrl',
        'naturalWidth',
        'naturalHeight',
        'sourceMimeType',
        'sourceBytes',
        'hskLevel',
        'precedingContext',
      ])
      const sourceBytes =
        item.sourceBytes === undefined
          ? undefined
          : arrayBuffer(item.sourceBytes, '$.sourceBytes', MAX_RUNTIME_BINARY_BYTES)
      const sourceMimeType =
        item.sourceMimeType === undefined
          ? undefined
          : string(item.sourceMimeType, '$.sourceMimeType', 128)
      const precedingContext =
        item.precedingContext === undefined
          ? undefined
          : parsePrecedingContext(item.precedingContext)
      return {
        type,
        pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256),
        pageIndex: integer(item.pageIndex, '$.pageIndex', 0, 100_000),
        imageUrl: string(item.imageUrl, '$.imageUrl', 8_192),
        pageUrl: string(item.pageUrl, '$.pageUrl', 8_192),
        naturalWidth: integer(item.naturalWidth, '$.naturalWidth', 1, 32_768),
        naturalHeight: integer(item.naturalHeight, '$.naturalHeight', 1, 32_768),
        ...(sourceMimeType === undefined ? {} : { sourceMimeType }),
        ...(sourceBytes === undefined ? {} : { sourceBytes }),
        hskLevel: hskLevel(item.hskLevel, '$.hskLevel'),
        ...(precedingContext === undefined ? {} : { precedingContext }),
      }
    }
    case 'job:poll':
    case 'job:cancel':
      exact(item, ['type', 'jobId'])
      return { type, jobId: string(item.jobId, '$.jobId', 512) }
    case 'job:result':
      exact(item, [
        'type',
        'jobId',
        'pageSessionId',
        'sourceUrl',
        'sourceSha256',
        'sourceWidth',
        'sourceHeight',
      ])
      return {
        type,
        jobId: string(item.jobId, '$.jobId', 512),
        pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256),
        sourceUrl: string(item.sourceUrl, '$.sourceUrl', 8_192),
        sourceSha256: sha256(item.sourceSha256, '$.sourceSha256'),
        sourceWidth: integer(item.sourceWidth, '$.sourceWidth', 1, 32_768),
        sourceHeight: integer(item.sourceHeight, '$.sourceHeight', 1, 32_768),
      }
    case 'jobs:recover': {
      exact(item, ['type', 'pageSessionId', 'pageUrl', 'candidates'])
      if (!Array.isArray(item.candidates) || item.candidates.length > MAX_RECOVERY_CANDIDATES) {
        throw new RuntimeMessageValidationError('$.candidates must be a bounded array.')
      }
      return {
        type,
        pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256),
        pageUrl: string(item.pageUrl, '$.pageUrl', 8_192),
        candidates: item.candidates.map(parseRecoveryCandidate),
      }
    }
    case 'jobs:cancel-page':
      exact(item, ['type', 'pageSessionId'])
      return {
        type,
        pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256),
      }
    case 'dictionary:lookup': {
      exact(item, ['type', 'request'])
      const request = parseLookupRequest(item.request)
      if (!request.jobId || !request.regionId) {
        throw new RuntimeMessageValidationError(
          '$.request must identify the translated job and region.',
        )
      }
      return { type, request }
    }
    case 'font:get':
      exact(item, ['type', 'jobId', 'fontId'])
      return {
        type,
        jobId: string(item.jobId, '$.jobId', 512),
        fontId: string(item.fontId, '$.fontId', 512),
      }
    default:
      throw new RuntimeMessageValidationError(`$.type "${type}" is not supported.`)
  }
}

export function parseContentRequest(value: unknown): ContentRequest {
  const item = record(value)
  const type = string(item.type, '$.type', 64)
  switch (type) {
    case 'content:prepare':
    case 'content:cancel':
    case 'content:state':
      exact(item, ['type'])
      return { type }
    case 'content:start':
      exact(item, ['type', 'scope', 'hskLevel'])
      return {
        type,
        scope: translationScope(item.scope, '$.scope'),
        hskLevel: hskLevel(item.hskLevel, '$.hskLevel'),
      }
    default:
      throw new RuntimeMessageValidationError(`$.type "${type}" is not supported.`)
  }
}

export function parsePermissionPlan(value: unknown): PermissionPlan {
  const item = record(value)
  exact(item, ['visibleOrigins', 'allOrigins'])
  const validatePattern = (pattern: string): string => {
    if (!/^https?:\/\/(?:\[[0-9a-f:]+\]|[^/:*]+)\/\*$/iu.test(pattern)) {
      throw new RuntimeMessageValidationError('Permission plans must contain exact, portless origins.')
    }
    return pattern
  }
  return {
    visibleOrigins: stringArray(item.visibleOrigins, '$.visibleOrigins').map(validatePattern),
    allOrigins: stringArray(item.allOrigins, '$.allOrigins').map(validatePattern),
  }
}

function submittedJob(value: unknown): SubmittedJob {
  const item = record(value)
  exact(item, [
    'jobId',
    'clientImageId',
    'sourceSha256',
    'sourceUrl',
    'sourceWidth',
    'sourceHeight',
  ])
  return {
    jobId: string(item.jobId, '$.jobId', 512),
    clientImageId: string(item.clientImageId, '$.clientImageId', 512),
    sourceSha256: sha256(item.sourceSha256, '$.sourceSha256'),
    sourceUrl: string(item.sourceUrl, '$.sourceUrl', 8_192),
    sourceWidth: integer(item.sourceWidth, '$.sourceWidth', 1, 32_768),
    sourceHeight: integer(item.sourceHeight, '$.sourceHeight', 1, 32_768),
  }
}

function recoveredJobs(value: unknown): RecoveredJob[] {
  if (!Array.isArray(value) || value.length > MAX_RECOVERY_CANDIDATES) {
    throw new RuntimeMessageValidationError('$ must be a bounded recovered-job array.')
  }
  return value.map((entry, index) => {
    const item = record(entry, `$[${index}]`)
    exact(item, [
      'jobId',
      'clientImageId',
      'sourceSha256',
      'sourceUrl',
      'sourceWidth',
      'sourceHeight',
      'pageIndex',
      'status',
    ])
    return {
      jobId: string(item.jobId, `$[${index}].jobId`, 512),
      clientImageId: string(item.clientImageId, `$[${index}].clientImageId`, 512),
      sourceSha256: sha256(item.sourceSha256, `$[${index}].sourceSha256`),
      sourceUrl: string(item.sourceUrl, `$[${index}].sourceUrl`, 8_192),
      sourceWidth: integer(item.sourceWidth, `$[${index}].sourceWidth`, 1, 32_768),
      sourceHeight: integer(item.sourceHeight, `$[${index}].sourceHeight`, 1, 32_768),
      pageIndex: integer(item.pageIndex, `$[${index}].pageIndex`, 0, 100_000),
      status: parseBrowserJobStatus(item.status),
    }
  })
}

function parseResult<T extends BackgroundRequest['type']>(
  type: T,
  value: unknown,
): MessageResultMap[T] {
  let parsed: unknown
  switch (type) {
    case 'popup:prepare':
      parsed = parsePermissionPlan(value)
      break
    case 'popup:start':
    case 'popup:cancel':
      parsed = pageState(value)
      break
    case 'popup:state':
      parsed = pageState(value, true)
      break
    case 'setup:status':
    case 'setup:start':
      parsed = parseBrowserSetupStatus(value)
      break
    case 'job:submit':
      parsed = submittedJob(value)
      break
    case 'job:poll':
      parsed = parseBrowserJobStatus(value)
      break
    case 'job:result': {
      const item = record(value)
      exact(item, ['result', 'cleanImage'])
      parsed = {
        result: parseBrowserJobResult(item.result),
        cleanImage: arrayBuffer(item.cleanImage, '$.cleanImage', MAX_RUNTIME_BINARY_BYTES),
      } satisfies DeliveredJobResult
      break
    }
    case 'job:cancel':
    case 'jobs:cancel-page':
    case 'setup:open-installer':
      if (value !== undefined) {
        throw new RuntimeMessageValidationError('$ must be undefined.')
      }
      parsed = undefined
      break
    case 'jobs:recover':
      parsed = recoveredJobs(value)
      break
    case 'dictionary:lookup':
      parsed = parseLookupResult(value)
      break
    case 'font:get': {
      const item = record(value)
      exact(item, ['fontId', 'bytes'])
      parsed = {
        fontId: string(item.fontId, '$.fontId', 512),
        bytes: arrayBuffer(item.bytes, '$.bytes', MAX_RUNTIME_FONT_BYTES),
      } satisfies FontPayload
      break
    }
  }
  return parsed as MessageResultMap[T]
}

export async function sendBackgroundMessage<T extends BackgroundRequest['type']>(
  message: RequestOfType<T>,
): Promise<MessageResultMap[T]> {
  const raw = await browser.runtime.sendMessage(message)
  const response = record(raw)
  exact(response, response.ok === true ? ['ok', 'value'] : ['ok', 'error'])
  if (response.ok === false) {
    const error = record(response.error, '$.error')
    exact(error, ['code', 'message', 'retryable'], '$.error')
    throw new RuntimeMessageError(
      string(error.code, '$.error.code', 256),
      string(error.message, '$.error.message', 2_048),
      boolean(error.retryable, '$.error.retryable'),
    )
  }
  if (response.ok !== true) {
    throw new RuntimeMessageError(
      'INVALID_BACKGROUND_RESPONSE',
      'The extension background returned an invalid response.',
      true,
    )
  }
  try {
    return parseResult(message.type, response.value)
  } catch (error) {
    throw new RuntimeMessageError(
      'INVALID_BACKGROUND_RESPONSE',
      error instanceof Error ? error.message : 'The background response was invalid.',
      true,
    )
  }
}
