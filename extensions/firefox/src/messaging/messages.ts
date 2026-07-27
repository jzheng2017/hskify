import {
  MAX_PRECEDING_CONTEXT,
  MAX_PROPER_NAME_GLOSSARY,
  parseBrowserSetupStatus,
  parseJobUpdateBatch,
  parseLookupRequest,
  parseLookupResult,
  parseViewportUpdate,
  type BrowserJobRequest,
  type BrowserSetupStatus,
  type HskLevel,
  type JobUpdateBatch,
  type LookupRequest,
  type LookupResult,
  type NameTranslation,
  type NormalizedRect,
} from '../contracts/browser'

const MAX_RUNTIME_BINARY_BYTES = 25 * 1024 * 1024
const MAX_RUNTIME_FONT_BYTES = 32 * 1024 * 1024
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
  nameTranslation: NameTranslation
}
export type PopupCancelMessage = { type: 'popup:cancel' }
export type PopupStateMessage = { type: 'popup:state' }
export type SetupStatusMessage = { type: 'setup:status' }
export type SetupStartMessage = { type: 'setup:start' }

export type ContentPrepareMessage = { type: 'content:prepare' }
export type ContentStartMessage = {
  type: 'content:start'
  scope: TranslationScope
  hskLevel: HskLevel
  nameTranslation: NameTranslation
  properNameGlossary?: BrowserJobRequest['properNameGlossary']
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
  nameTranslation: NameTranslation
  visibleRects: NormalizedRect[]
  precedingContext?: BrowserJobRequest['precedingContext']
  properNameGlossary?: BrowserJobRequest['properNameGlossary']
}

export type PrefetchImageMessage = {
  type: 'image:prefetch'
  pageSessionId: string
  pageIndex: number
  imageUrl: string
  pageUrl: string
  naturalWidth: number
  naturalHeight: number
}

export type CancelImagePrefetchMessage = {
  type: 'image:prefetch-cancel'
  pageSessionId: string
  pageUrl: string
}

export type JobUpdatesMessage = {
  type: 'job:updates'
  jobId: string
  after: number
}

export type JobAckMessage = {
  type: 'job:ack'
  jobId: string
  sequence: number
  terminalType?: 'complete' | 'failed' | 'cancelled'
}

export type JobViewportMessage = {
  type: 'job:viewport'
  jobId: string
  visibleRects: NormalizedRect[]
  active: boolean
}

export type JobPatchMessage = {
  type: 'job:patch'
  jobId: string
  patchId: string
  mimeType: 'image/png'
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
  | PrefetchImageMessage
  | CancelImagePrefetchMessage
  | SubmitImageMessage
  | JobUpdatesMessage
  | JobAckMessage
  | JobViewportMessage
  | JobPatchMessage
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
  acknowledgedSequence: number
}

export type PatchPayload = {
  patchId: string
  mimeType: 'image/png'
  bytes: ArrayBuffer
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
  acknowledgedSequence: number
  terminalType?: 'complete' | 'failed' | 'cancelled'
}

export type PopupState = PageState & {
  hskLevel: HskLevel
  nameTranslation: NameTranslation
}

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
  'image:prefetch': undefined
  'image:prefetch-cancel': undefined
  'job:submit': SubmittedJob
  'job:updates': JobUpdateBatch
  'job:ack': undefined
  'job:viewport': undefined
  'job:patch': PatchPayload
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

function string(value: unknown, path: string, maximum = 4_096): string {
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

function nameTranslation(value: unknown, path: string): NameTranslation {
  if (value !== 'keep-original' && value !== 'chinese') {
    throw new RuntimeMessageValidationError(
      `${path} must be "keep-original" or "chinese".`,
    )
  }
  return value
}

function terminalType(
  value: unknown,
  path: string,
): 'complete' | 'failed' | 'cancelled' {
  if (value !== 'complete' && value !== 'failed' && value !== 'cancelled') {
    throw new RuntimeMessageValidationError(`${path} must be a terminal update type.`)
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

function normalizedRect(value: unknown, path: string): NormalizedRect {
  try {
    return parseViewportUpdate({ visibleRects: [value], active: true }).visibleRects[0]!
  } catch (error) {
    throw new RuntimeMessageValidationError(
      `${path} is invalid: ${error instanceof Error ? error.message : 'invalid rectangle'}`,
    )
  }
}

function normalizedRects(value: unknown, path: string): NormalizedRect[] {
  if (!Array.isArray(value) || value.length > 64) {
    throw new RuntimeMessageValidationError(`${path} must be a bounded rectangle array.`)
  }
  return value.map((item, index) => normalizedRect(item, `${path}[${index}]`))
}

export function parsePageState(value: unknown): PageState {
  return pageState(value, false) as PageState
}

function pageState(value: unknown, includeHsk = false): PopupState | PageState {
  const item = record(value)
  exact(
    item,
    includeHsk
      ? ['state', 'current', 'total', 'stage', 'message', 'hskLevel', 'nameTranslation']
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
    ? {
        ...parsed,
        hskLevel: hskLevel(item.hskLevel, '$.hskLevel'),
        nameTranslation: nameTranslation(item.nameTranslation, '$.nameTranslation'),
      }
    : parsed
}

function parsePrecedingContext(value: unknown): BrowserJobRequest['precedingContext'] {
  if (!Array.isArray(value) || value.length > MAX_PRECEDING_CONTEXT) {
    throw new RuntimeMessageValidationError(
      `$.precedingContext must contain at most ${MAX_PRECEDING_CONTEXT} entries.`,
    )
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

function parseProperNameGlossary(
  value: unknown,
): BrowserJobRequest['properNameGlossary'] {
  if (!Array.isArray(value) || value.length > MAX_PROPER_NAME_GLOSSARY) {
    throw new RuntimeMessageValidationError(
      `$.properNameGlossary must contain at most ${MAX_PROPER_NAME_GLOSSARY} entries.`,
    )
  }
  const seen = new Set<string>()
  return value.map((entry, index) => {
    const item = record(entry, `$.properNameGlossary[${index}]`)
    exact(item, ['sourceEnglish', 'chinese'], `$.properNameGlossary[${index}]`)
    const sourceEnglish = string(
      item.sourceEnglish,
      `$.properNameGlossary[${index}].sourceEnglish`,
      256,
    )
    const normalized = sourceEnglish.trim().toLocaleLowerCase('en-US')
    if (seen.has(normalized)) {
      throw new RuntimeMessageValidationError(
        `$.properNameGlossary[${index}].sourceEnglish must be unique ignoring ASCII case.`,
      )
    }
    seen.add(normalized)
    return {
      sourceEnglish,
      chinese: string(item.chinese, `$.properNameGlossary[${index}].chinese`, 128),
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
      exact(item, ['type'])
      return { type }
    case 'popup:start':
      exact(item, ['type', 'scope', 'hskLevel', 'nameTranslation'])
      return {
        type,
        scope: translationScope(item.scope, '$.scope'),
        hskLevel: hskLevel(item.hskLevel, '$.hskLevel'),
        nameTranslation: nameTranslation(item.nameTranslation, '$.nameTranslation'),
      }
    case 'image:prefetch':
      exact(item, [
        'type',
        'pageSessionId',
        'pageIndex',
        'imageUrl',
        'pageUrl',
        'naturalWidth',
        'naturalHeight',
      ])
      return {
        type,
        pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256),
        pageIndex: integer(item.pageIndex, '$.pageIndex', 0, 100_000),
        imageUrl: string(item.imageUrl, '$.imageUrl', 8_192),
        pageUrl: string(item.pageUrl, '$.pageUrl', 8_192),
        naturalWidth: integer(item.naturalWidth, '$.naturalWidth', 1, 32_768),
        naturalHeight: integer(item.naturalHeight, '$.naturalHeight', 1, 32_768),
      }
    case 'image:prefetch-cancel':
      exact(item, ['type', 'pageSessionId', 'pageUrl'])
      return {
        type,
        pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256),
        pageUrl: string(item.pageUrl, '$.pageUrl', 8_192),
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
        'nameTranslation',
        'visibleRects',
        'precedingContext',
        'properNameGlossary',
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
      const properNameGlossary =
        item.properNameGlossary === undefined
          ? undefined
          : parseProperNameGlossary(item.properNameGlossary)
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
        nameTranslation: nameTranslation(item.nameTranslation, '$.nameTranslation'),
        visibleRects: normalizedRects(item.visibleRects, '$.visibleRects'),
        ...(precedingContext === undefined ? {} : { precedingContext }),
        ...(properNameGlossary === undefined ? {} : { properNameGlossary }),
      }
    }
    case 'job:updates':
      exact(item, ['type', 'jobId', 'after'])
      return {
        type,
        jobId: string(item.jobId, '$.jobId', 512),
        after: integer(item.after, '$.after', 0, Number.MAX_SAFE_INTEGER),
      }
    case 'job:ack': {
      exact(item, ['type', 'jobId', 'sequence', 'terminalType'])
      const parsedTerminal =
        item.terminalType === undefined
          ? undefined
          : terminalType(item.terminalType, '$.terminalType')
      return {
        type,
        jobId: string(item.jobId, '$.jobId', 512),
        sequence: integer(item.sequence, '$.sequence', 0, Number.MAX_SAFE_INTEGER),
        ...(parsedTerminal === undefined ? {} : { terminalType: parsedTerminal }),
      }
    }
    case 'job:viewport':
      exact(item, ['type', 'jobId', 'visibleRects', 'active'])
      return {
        type,
        jobId: string(item.jobId, '$.jobId', 512),
        visibleRects: normalizedRects(item.visibleRects, '$.visibleRects'),
        active: boolean(item.active, '$.active'),
      }
    case 'job:patch':
      exact(item, ['type', 'jobId', 'patchId', 'mimeType'])
      if (item.mimeType !== 'image/png') {
        throw new RuntimeMessageValidationError('$.mimeType must be "image/png".')
      }
      return {
        type,
        jobId: string(item.jobId, '$.jobId', 512),
        patchId: string(item.patchId, '$.patchId', 512),
        mimeType: 'image/png',
      }
    case 'job:cancel':
      exact(item, ['type', 'jobId'])
      return { type, jobId: string(item.jobId, '$.jobId', 512) }
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
      exact(item, [
        'type',
        'scope',
        'hskLevel',
        'nameTranslation',
        'properNameGlossary',
      ])
      const properNameGlossary =
        item.properNameGlossary === undefined
          ? undefined
          : parseProperNameGlossary(item.properNameGlossary)
      return {
        type,
        scope: translationScope(item.scope, '$.scope'),
        hskLevel: hskLevel(item.hskLevel, '$.hskLevel'),
        nameTranslation: nameTranslation(item.nameTranslation, '$.nameTranslation'),
        ...(properNameGlossary === undefined ? {} : { properNameGlossary }),
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
    'acknowledgedSequence',
  ])
  return {
    jobId: string(item.jobId, '$.jobId', 512),
    clientImageId: string(item.clientImageId, '$.clientImageId', 512),
    sourceSha256: sha256(item.sourceSha256, '$.sourceSha256'),
    sourceUrl: string(item.sourceUrl, '$.sourceUrl', 8_192),
    sourceWidth: integer(item.sourceWidth, '$.sourceWidth', 1, 32_768),
    sourceHeight: integer(item.sourceHeight, '$.sourceHeight', 1, 32_768),
    acknowledgedSequence: integer(
      item.acknowledgedSequence,
      '$.acknowledgedSequence',
      0,
      Number.MAX_SAFE_INTEGER,
    ),
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
      'acknowledgedSequence',
      'terminalType',
    ])
    const parsedTerminal =
      item.terminalType === undefined
        ? undefined
        : terminalType(item.terminalType, `$[${index}].terminalType`)
    return {
      jobId: string(item.jobId, `$[${index}].jobId`, 512),
      clientImageId: string(item.clientImageId, `$[${index}].clientImageId`, 512),
      sourceSha256: sha256(item.sourceSha256, `$[${index}].sourceSha256`),
      sourceUrl: string(item.sourceUrl, `$[${index}].sourceUrl`, 8_192),
      sourceWidth: integer(item.sourceWidth, `$[${index}].sourceWidth`, 1, 32_768),
      sourceHeight: integer(item.sourceHeight, `$[${index}].sourceHeight`, 1, 32_768),
      pageIndex: integer(item.pageIndex, `$[${index}].pageIndex`, 0, 100_000),
      acknowledgedSequence: integer(
        item.acknowledgedSequence,
        `$[${index}].acknowledgedSequence`,
        0,
        Number.MAX_SAFE_INTEGER,
      ),
      ...(parsedTerminal === undefined ? {} : { terminalType: parsedTerminal }),
    }
  })
}

function parseResult<T extends BackgroundRequest['type']>(
  request: Extract<BackgroundRequest, { type: T }>,
  value: unknown,
): MessageResultMap[T] {
  let parsed: unknown
  switch (request.type) {
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
    case 'job:updates':
      parsed = parseJobUpdateBatch(
        value,
        (request as Extract<BackgroundRequest, { type: 'job:updates' }>).after,
      )
      break
    case 'job:patch': {
      const item = record(value)
      exact(item, ['patchId', 'mimeType', 'bytes'])
      if (item.mimeType !== 'image/png') {
        throw new RuntimeMessageValidationError('$.mimeType must be "image/png".')
      }
      parsed = {
        patchId: string(item.patchId, '$.patchId', 512),
        mimeType: 'image/png',
        bytes: arrayBuffer(item.bytes, '$.bytes', MAX_RUNTIME_BINARY_BYTES),
      } satisfies PatchPayload
      break
    }
    case 'job:ack':
    case 'job:viewport':
    case 'job:cancel':
    case 'jobs:cancel-page':
    case 'image:prefetch':
    case 'image:prefetch-cancel':
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
  message: Extract<BackgroundRequest, { type: T }>,
): Promise<MessageResultMap[T]> {
  const response = (await browser.runtime.sendMessage(message)) as unknown
  const envelope = record(response)
  if (envelope.ok === true) {
    exact(envelope, ['ok', 'value'])
    return parseResult(message, envelope.value)
  }
  if (envelope.ok === false) {
    exact(envelope, ['ok', 'error'])
    const error = record(envelope.error, '$.error')
    exact(error, ['code', 'message', 'retryable'], '$.error')
    throw new RuntimeMessageError(
      string(error.code, '$.error.code', 256),
      string(error.message, '$.error.message', 2_048),
      boolean(error.retryable, '$.error.retryable'),
    )
  }
  throw new RuntimeMessageValidationError('The background response envelope is invalid.')
}
