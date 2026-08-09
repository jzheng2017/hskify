import {
  parseBrowserSetupStatus,
  parseDocumentJobRequest,
  parseJobFocus,
  parseJobUpdateBatch,
  parseLookupRequest,
  parseLookupResult,
  parseTranslationSettings,
  type BrowserSetupStatus,
  type ChapterKind,
  type DocumentJobRequest,
  type HskLevel,
  type JobFocus,
  type JobUpdateBatch,
  type LearningMode,
  type LookupRequest,
  type LookupResult,
  type NormalizedRect,
  type PageSurfaceKind,
  type ReadingDirection,
  type TranslationSettings,
} from '../contracts/browser'
import { DEFAULT_IMAGE_LIMITS } from '../acquisition/image-format'

const MAX_RUNTIME_BINARY_BYTES = 25 * 1024 * 1024
const MAX_RUNTIME_FONT_BYTES = 32 * 1024 * 1024
const MAX_RECOVERY_CANDIDATES = 512

export type TranslationScope = 'visible' | 'all'

export type PopupPrepareMessage = { type: 'popup:prepare' }
export type PopupStartMessage = {
  type: 'popup:start'
  scope: TranslationScope
  hskLevel: HskLevel
  learningMode: LearningMode
  readingDirection: ReadingDirection
}
export type PopupCancelMessage = { type: 'popup:cancel' }
export type PopupStateMessage = { type: 'popup:state' }
export type SetupStatusMessage = { type: 'setup:status' }
export type SetupStartMessage = { type: 'setup:start' }
export type EngineWarmupMessage = {
  type: 'engine:warmup'
  contentKind: Exclude<ChapterKind, 'unsupported'>
}

export type ChapterStartMessage = {
  type: 'chapter:start'
  pageSessionId: string
  pageUrl: string
  contentKind: Exclude<ChapterKind, 'unsupported'>
}
export type ChapterSourceMessage = {
  type: 'chapter:source'
  pageSessionId: string
  pageUrl: string
  sourceIndex: number
}
export type ChapterFinishMessage = {
  type: 'chapter:finish'
  pageSessionId: string
  pageUrl: string
}
export type ChapterCancelMessage = {
  type: 'chapter:cancel'
  pageSessionId: string
  pageUrl: string
}

export type ContentStartMessage = {
  type: 'content:start'
  scope: TranslationScope
  hskLevel: HskLevel
  learningMode: LearningMode
  readingDirection: ReadingDirection
}
export type ContentCancelMessage = { type: 'content:cancel' }
export type ContentStateMessage = { type: 'content:state' }

export type SubmitImageMessage = {
  type: 'job:submit-image'
  pageSessionId: string
  sourceIndex: number
  chapterSourceOrder: number[]
  surfaceKind: PageSurfaceKind
  imageUrl: string
  pageUrl: string
  naturalWidth: number
  naturalHeight: number
  sourceMimeType?: string
  sourceBytes?: ArrayBuffer
  hskLevel: HskLevel
  learningMode: LearningMode
  readingDirection: ReadingDirection
  visibleRects: NormalizedRect[]
}

export type SubmitDocumentMessage = {
  type: 'job:submit-document'
  pageUrl: string
  request: DocumentJobRequest
}

export type PrefetchImageMessage = {
  type: 'image:prefetch'
  pageSessionId: string
  sourceIndex: number
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

export type JobUpdatesMessage = { type: 'job:updates'; jobId: string; after: number }
export type JobAckMessage = {
  type: 'job:ack'
  jobId: string
  sequence: number
  terminalType?: 'complete' | 'failed' | 'cancelled'
}
export type JobFocusMessage = { type: 'job:focus'; jobId: string; focus: JobFocus }
export type JobPatchMessage = {
  type: 'job:patch'
  jobId: string
  patchId: string
  mimeType: 'image/png'
}
export type CancelJobMessage = { type: 'job:cancel'; jobId: string }

export type ImageRecoveryCandidate = {
  kind: 'image'
  sourceUrl: string
  naturalWidth: number
  naturalHeight: number
  sourceIndex: number
  sourceSha256?: string
  settings: TranslationSettings
  readingDirection: ReadingDirection
}
export type DocumentRecoveryCandidate = {
  kind: 'document'
  sourceSha256: string
  settings: TranslationSettings
}
export type RecoveryCandidate = ImageRecoveryCandidate | DocumentRecoveryCandidate
export type RecoverJobsMessage = {
  type: 'jobs:recover'
  pageSessionId: string
  pageUrl: string
  candidates: RecoveryCandidate[]
}
export type CancelPageJobsMessage = { type: 'jobs:cancel-page'; pageSessionId: string }
export type LookupMessage = { type: 'dictionary:lookup'; request: LookupRequest }
export type FontMessage = { type: 'font:get'; jobId: string; fontId: string }

export type BackgroundRequest =
  | PopupPrepareMessage | PopupStartMessage | PopupCancelMessage | PopupStateMessage
  | SetupStatusMessage | SetupStartMessage | EngineWarmupMessage
  | ChapterStartMessage | ChapterSourceMessage | ChapterFinishMessage | ChapterCancelMessage
  | PrefetchImageMessage | CancelImagePrefetchMessage
  | SubmitImageMessage | SubmitDocumentMessage | JobUpdatesMessage | JobAckMessage
  | JobFocusMessage | JobPatchMessage | CancelJobMessage | RecoverJobsMessage
  | CancelPageJobsMessage | LookupMessage | FontMessage

export type ContentRequest = ContentStartMessage | ContentCancelMessage | ContentStateMessage

export type PageState = {
  state: 'idle' | 'running' | 'complete' | 'cancelled' | 'failed'
  contentKind: ChapterKind
  current: number
  total: number
  stage?: string
  message: string
}

export type SubmittedImageJob = {
  kind: 'image'
  jobId: string
  clientImageId: string
  sourceSha256: string
  sourceUrl: string
  sourceWidth: number
  sourceHeight: number
  sourceIndex: number
  acknowledgedSequence: number
}
export type SubmittedDocumentJob = {
  kind: 'document'
  jobId: string
  sourceSha256: string
  acknowledgedSequence: number
}
export type SubmittedJob = SubmittedImageJob | SubmittedDocumentJob

export type PatchPayload = { patchId: string; mimeType: 'image/png'; bytes: ArrayBuffer }
export type FontPayload = { fontId: string; bytes: ArrayBuffer }

export type RecoveredImageJob = SubmittedImageJob & {
  terminalType?: 'complete' | 'failed' | 'cancelled'
}
export type RecoveredDocumentJob = SubmittedDocumentJob & {
  terminalType?: 'complete' | 'failed' | 'cancelled'
}
export type RecoveredJob = RecoveredImageJob | RecoveredDocumentJob

export type PopupState = PageState & {
  hskLevel: HskLevel
  learningMode: LearningMode
  readingDirection: ReadingDirection
}

export type MessageError = { code: string; message: string; retryable: boolean }
export type MessageResponse<T> = { ok: true; value: T } | { ok: false; error: MessageError }

export type MessageResultMap = {
  'popup:prepare': undefined
  'popup:start': PageState
  'popup:cancel': PageState
  'popup:state': PopupState
  'setup:status': BrowserSetupStatus
  'setup:start': BrowserSetupStatus
  'engine:warmup': BrowserSetupStatus
  'chapter:start': undefined
  'chapter:source': undefined
  'chapter:finish': undefined
  'chapter:cancel': undefined
  'image:prefetch': undefined
  'image:prefetch-cancel': undefined
  'job:submit-image': SubmittedImageJob
  'job:submit-document': SubmittedDocumentJob
  'job:updates': JobUpdateBatch
  'job:ack': undefined
  'job:focus': undefined
  'job:patch': PatchPayload
  'job:cancel': undefined
  'jobs:recover': RecoveredJob[]
  'jobs:cancel-page': undefined
  'dictionary:lookup': LookupResult
  'font:get': FontPayload
}

export type RequestOfType<T extends BackgroundRequest['type']> = Extract<BackgroundRequest, { type: T }>

export class RuntimeMessageError extends Error {
  constructor(readonly code: string, message: string, readonly retryable: boolean) {
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

function fail(path: string, message: string): never {
  throw new RuntimeMessageValidationError(`${path} ${message}.`)
}
function record(value: unknown, path = '$'): UnknownRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) fail(path, 'must be an object')
  return value as UnknownRecord
}
function exact(item: UnknownRecord, allowed: readonly string[], path = '$'): void {
  const expected = new Set(allowed)
  const extra = Object.keys(item).find((key) => !expected.has(key))
  if (extra) fail(`${path}.${extra}`, 'is not permitted')
}
function string(value: unknown, path: string, maximum = 4_096): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > maximum) fail(path, `must be a non-empty string no longer than ${maximum} characters`)
  return value
}
function integer(value: unknown, path: string, minimum = 0, maximum = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum || value > maximum) fail(path, `must be an integer from ${minimum} to ${maximum}`)
  return value
}
function bool(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') fail(path, 'must be a boolean')
  return value
}
function hskLevel(value: unknown, path: string): HskLevel {
  if (![1, 2, 3, 4, 5, 6].includes(value as number)) fail(path, 'must be an HSK level')
  return value as HskLevel
}
function learningMode(value: unknown, path: string): LearningMode {
  if (value !== 'natural' && value !== 'strict') fail(path, 'must be natural or strict')
  return value
}
function readingDirection(value: unknown, path: string): ReadingDirection {
  if (value !== 'ltr' && value !== 'rtl') fail(path, 'must be ltr or rtl')
  return value
}
function scope(value: unknown, path: string): TranslationScope {
  if (value !== 'visible' && value !== 'all') fail(path, 'must be visible or all')
  return value
}
function chapterKind(value: unknown, path: string, allowUnsupported = true): ChapterKind {
  if (value !== 'image' && value !== 'document' && (allowUnsupported ? value !== 'unsupported' : true)) fail(path, 'must be a supported chapter kind')
  return value as ChapterKind
}
function terminal(value: unknown, path: string): 'complete' | 'failed' | 'cancelled' | undefined {
  if (value === undefined) return undefined
  if (value !== 'complete' && value !== 'failed' && value !== 'cancelled') fail(path, 'must be a terminal update type')
  return value
}
function sha256(value: unknown, path: string): string {
  const parsed = string(value, path, 64)
  if (!/^[a-f0-9]{64}$/u.test(parsed)) fail(path, 'must be a lowercase SHA-256')
  return parsed
}
function arrayBuffer(value: unknown, path: string, maximum: number): ArrayBuffer {
  if (!(value instanceof ArrayBuffer) || value.byteLength > maximum) fail(path, `must be an ArrayBuffer no larger than ${maximum} bytes`)
  return value
}
function normalizedRect(value: unknown, path: string): NormalizedRect {
  const item = record(value, path)
  exact(item, ['x', 'y', 'width', 'height'], path)
  for (const field of ['x', 'y', 'width', 'height'] as const) if (typeof item[field] !== 'number' || !Number.isFinite(item[field]) || item[field] < 0 || item[field] > 1) fail(`${path}.${field}`, 'must be from 0 to 1')
  const parsed = item as { x: number; y: number; width: number; height: number }
  if (parsed.width <= 0 || parsed.height <= 0 || parsed.x + parsed.width > 1 + Number.EPSILON || parsed.y + parsed.height > 1 + Number.EPSILON) fail(path, 'must be a positive rectangle inside its source')
  return parsed
}
function normalizedRects(value: unknown, path: string): NormalizedRect[] {
  if (!Array.isArray(value) || value.length > 64) fail(path, 'must contain at most 64 rectangles')
  return value.map((entry, index) => normalizedRect(entry, `${path}[${index}]`))
}
function integerArray(value: unknown, path: string, maximum = 100_000): number[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > maximum) fail(path, `must contain from 1 to ${maximum} integers`)
  return value.map((entry, index) => integer(entry, `${path}[${index}]`))
}

export function parsePageState(value: unknown): PageState {
  return pageState(value, false) as PageState
}

function pageState(value: unknown, includeSettings: boolean): PageState | PopupState {
  const item = record(value)
  exact(item, includeSettings ? ['state', 'contentKind', 'current', 'total', 'stage', 'message', 'hskLevel', 'learningMode', 'readingDirection'] : ['state', 'contentKind', 'current', 'total', 'stage', 'message'])
  const state = item.state
  if (!['idle', 'running', 'complete', 'cancelled', 'failed'].includes(state as string)) fail('$.state', 'is invalid')
  const current = integer(item.current, '$.current')
  const total = integer(item.total, '$.total')
  if (current > total) fail('$.current', 'must not exceed total')
  const base: PageState = {
    state: state as PageState['state'],
    contentKind: chapterKind(item.contentKind, '$.contentKind'), current, total,
    ...(item.stage === undefined ? {} : { stage: string(item.stage, '$.stage', 256) }),
    message: string(item.message, '$.message', 2_048),
  }
  return includeSettings ? { ...base, hskLevel: hskLevel(item.hskLevel, '$.hskLevel'), learningMode: learningMode(item.learningMode, '$.learningMode'), readingDirection: readingDirection(item.readingDirection, '$.readingDirection') } : base
}

function parseImageSubmit(item: UnknownRecord): SubmitImageMessage {
  exact(item, ['type', 'pageSessionId', 'sourceIndex', 'chapterSourceOrder', 'surfaceKind', 'imageUrl', 'pageUrl', 'naturalWidth', 'naturalHeight', 'sourceMimeType', 'sourceBytes', 'hskLevel', 'learningMode', 'readingDirection', 'visibleRects'])
  const sourceIndex = integer(item.sourceIndex, '$.sourceIndex', 0, 100_000)
  const chapterSourceOrder = integerArray(item.chapterSourceOrder, '$.chapterSourceOrder')
  if (!chapterSourceOrder.includes(sourceIndex) || chapterSourceOrder.some((entry, index) => index > 0 && chapterSourceOrder[index - 1]! >= entry)) fail('$.chapterSourceOrder', 'must be strictly increasing and include sourceIndex')
  const surfaceKind = item.surfaceKind
  if (!['image', 'background', 'canvas', 'webgl', 'frame'].includes(surfaceKind as string)) fail('$.surfaceKind', 'is invalid')
  return {
    type: 'job:submit-image', pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256), sourceIndex, chapterSourceOrder, surfaceKind: surfaceKind as PageSurfaceKind,
    imageUrl: string(item.imageUrl, '$.imageUrl', 32_768), pageUrl: string(item.pageUrl, '$.pageUrl', 32_768), naturalWidth: integer(item.naturalWidth, '$.naturalWidth', 1, DEFAULT_IMAGE_LIMITS.maximumWidth), naturalHeight: integer(item.naturalHeight, '$.naturalHeight', 1, DEFAULT_IMAGE_LIMITS.maximumHeight),
    ...(item.sourceMimeType === undefined ? {} : { sourceMimeType: string(item.sourceMimeType, '$.sourceMimeType', 128) }),
    ...(item.sourceBytes === undefined ? {} : { sourceBytes: arrayBuffer(item.sourceBytes, '$.sourceBytes', DEFAULT_IMAGE_LIMITS.maximumBytes) }),
    hskLevel: hskLevel(item.hskLevel, '$.hskLevel'), learningMode: learningMode(item.learningMode, '$.learningMode'), readingDirection: readingDirection(item.readingDirection, '$.readingDirection'), visibleRects: normalizedRects(item.visibleRects, '$.visibleRects'),
  }
}

export function parseBackgroundRequest(value: unknown): BackgroundRequest {
  const item = record(value)
  const type = string(item.type, '$.type', 64) as BackgroundRequest['type']
  switch (type) {
    case 'popup:prepare': case 'popup:cancel': case 'popup:state': case 'setup:status': case 'setup:start':
      exact(item, ['type']); return { type }
    case 'engine:warmup':
      exact(item, ['type', 'contentKind']); return { type, contentKind: chapterKind(item.contentKind, '$.contentKind', false) as 'image' | 'document' }
    case 'popup:start':
      exact(item, ['type', 'scope', 'hskLevel', 'learningMode', 'readingDirection']); return { type, scope: scope(item.scope, '$.scope'), hskLevel: hskLevel(item.hskLevel, '$.hskLevel'), learningMode: learningMode(item.learningMode, '$.learningMode'), readingDirection: readingDirection(item.readingDirection, '$.readingDirection') }
    case 'chapter:start':
      exact(item, ['type', 'pageSessionId', 'pageUrl', 'contentKind']); return { type, pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256), pageUrl: string(item.pageUrl, '$.pageUrl', 32_768), contentKind: chapterKind(item.contentKind, '$.contentKind', false) as 'image' | 'document' }
    case 'chapter:source':
      exact(item, ['type', 'pageSessionId', 'pageUrl', 'sourceIndex']); return { type, pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256), pageUrl: string(item.pageUrl, '$.pageUrl', 32_768), sourceIndex: integer(item.sourceIndex, '$.sourceIndex', 0, 100_000) }
    case 'chapter:finish': case 'chapter:cancel':
      exact(item, ['type', 'pageSessionId', 'pageUrl']); return { type, pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256), pageUrl: string(item.pageUrl, '$.pageUrl', 32_768) }
    case 'image:prefetch':
      exact(item, ['type', 'pageSessionId', 'sourceIndex', 'imageUrl', 'pageUrl', 'naturalWidth', 'naturalHeight']); return { type, pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256), sourceIndex: integer(item.sourceIndex, '$.sourceIndex', 0, 100_000), imageUrl: string(item.imageUrl, '$.imageUrl', 32_768), pageUrl: string(item.pageUrl, '$.pageUrl', 32_768), naturalWidth: integer(item.naturalWidth, '$.naturalWidth', 1, DEFAULT_IMAGE_LIMITS.maximumWidth), naturalHeight: integer(item.naturalHeight, '$.naturalHeight', 1, DEFAULT_IMAGE_LIMITS.maximumHeight) }
    case 'image:prefetch-cancel':
      exact(item, ['type', 'pageSessionId', 'pageUrl']); return { type, pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256), pageUrl: string(item.pageUrl, '$.pageUrl', 32_768) }
    case 'job:submit-image': return parseImageSubmit(item)
    case 'job:submit-document': {
      exact(item, ['type', 'pageUrl', 'request'])
      return { type, pageUrl: string(item.pageUrl, '$.pageUrl', 32_768), request: parseDocumentJobRequest(item.request) }
    }
    case 'job:updates':
      exact(item, ['type', 'jobId', 'after']); return { type, jobId: string(item.jobId, '$.jobId', 512), after: integer(item.after, '$.after') }
    case 'job:ack': {
      exact(item, ['type', 'jobId', 'sequence', 'terminalType']); const parsed = terminal(item.terminalType, '$.terminalType'); return { type, jobId: string(item.jobId, '$.jobId', 512), sequence: integer(item.sequence, '$.sequence'), ...(parsed === undefined ? {} : { terminalType: parsed }) }
    }
    case 'job:focus':
      exact(item, ['type', 'jobId', 'focus']); return { type, jobId: string(item.jobId, '$.jobId', 512), focus: parseJobFocus(item.focus) }
    case 'job:patch':
      exact(item, ['type', 'jobId', 'patchId', 'mimeType']); if (item.mimeType !== 'image/png') fail('$.mimeType', 'must be image/png'); return { type, jobId: string(item.jobId, '$.jobId', 512), patchId: string(item.patchId, '$.patchId', 512), mimeType: 'image/png' }
    case 'job:cancel':
      exact(item, ['type', 'jobId']); return { type, jobId: string(item.jobId, '$.jobId', 512) }
    case 'jobs:cancel-page':
      exact(item, ['type', 'pageSessionId']); return { type, pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256) }
    case 'jobs:recover': {
      exact(item, ['type', 'pageSessionId', 'pageUrl', 'candidates'])
      if (!Array.isArray(item.candidates) || item.candidates.length > MAX_RECOVERY_CANDIDATES) fail('$.candidates', `must contain at most ${MAX_RECOVERY_CANDIDATES} candidates`)
      const candidates = item.candidates.map((candidate, index): RecoveryCandidate => {
        const path = `$.candidates[${index}]`; const parsed = record(candidate, path); const kind = chapterKind(parsed.kind, `${path}.kind`, false)
        if (kind === 'document') {
          exact(parsed, ['kind', 'sourceSha256', 'settings'], path)
          return {
            kind,
            sourceSha256: sha256(parsed.sourceSha256, `${path}.sourceSha256`),
            settings: parseTranslationSettings(parsed.settings, `${path}.settings`),
          }
        }
        exact(parsed, ['kind', 'sourceUrl', 'naturalWidth', 'naturalHeight', 'sourceIndex', 'sourceSha256', 'settings', 'readingDirection'], path)
        return {
          kind: 'image',
          sourceUrl: string(parsed.sourceUrl, `${path}.sourceUrl`, 32_768),
          naturalWidth: integer(parsed.naturalWidth, `${path}.naturalWidth`, 1, DEFAULT_IMAGE_LIMITS.maximumWidth),
          naturalHeight: integer(parsed.naturalHeight, `${path}.naturalHeight`, 1, DEFAULT_IMAGE_LIMITS.maximumHeight),
          sourceIndex: integer(parsed.sourceIndex, `${path}.sourceIndex`, 0, 100_000),
          ...(parsed.sourceSha256 === undefined ? {} : { sourceSha256: sha256(parsed.sourceSha256, `${path}.sourceSha256`) }),
          settings: parseTranslationSettings(parsed.settings, `${path}.settings`),
          readingDirection: readingDirection(parsed.readingDirection, `${path}.readingDirection`),
        }
      })
      return { type, pageSessionId: string(item.pageSessionId, '$.pageSessionId', 256), pageUrl: string(item.pageUrl, '$.pageUrl', 32_768), candidates }
    }
    case 'dictionary:lookup':
      exact(item, ['type', 'request']); return { type, request: parseLookupRequest(item.request) }
    case 'font:get':
      exact(item, ['type', 'jobId', 'fontId']); return { type, jobId: string(item.jobId, '$.jobId', 512), fontId: string(item.fontId, '$.fontId', 512) }
    default: fail('$.type', 'is not supported')
  }
}

export function parseContentRequest(value: unknown): ContentRequest {
  const item = record(value)
  const type = string(item.type, '$.type', 64)
  if (type === 'content:cancel' || type === 'content:state') { exact(item, ['type']); return { type } }
  if (type === 'content:start') {
    exact(item, ['type', 'scope', 'hskLevel', 'learningMode', 'readingDirection'])
    return { type, scope: scope(item.scope, '$.scope'), hskLevel: hskLevel(item.hskLevel, '$.hskLevel'), learningMode: learningMode(item.learningMode, '$.learningMode'), readingDirection: readingDirection(item.readingDirection, '$.readingDirection') }
  }
  return fail('$.type', 'is not a content request')
}

function parseSubmittedJob(value: unknown): SubmittedJob {
  const item = record(value)
  const kind = chapterKind(item.kind, '$.kind', false)
  if (kind === 'document') {
    exact(item, ['kind', 'jobId', 'sourceSha256', 'acknowledgedSequence'])
    return { kind, jobId: string(item.jobId, '$.jobId', 512), sourceSha256: sha256(item.sourceSha256, '$.sourceSha256'), acknowledgedSequence: integer(item.acknowledgedSequence, '$.acknowledgedSequence') }
  }
  exact(item, ['kind', 'jobId', 'clientImageId', 'sourceSha256', 'sourceUrl', 'sourceWidth', 'sourceHeight', 'sourceIndex', 'acknowledgedSequence'])
  return { kind: 'image', jobId: string(item.jobId, '$.jobId', 512), clientImageId: string(item.clientImageId, '$.clientImageId', 512), sourceSha256: sha256(item.sourceSha256, '$.sourceSha256'), sourceUrl: string(item.sourceUrl, '$.sourceUrl', 32_768), sourceWidth: integer(item.sourceWidth, '$.sourceWidth', 1), sourceHeight: integer(item.sourceHeight, '$.sourceHeight', 1), sourceIndex: integer(item.sourceIndex, '$.sourceIndex'), acknowledgedSequence: integer(item.acknowledgedSequence, '$.acknowledgedSequence') }
}

function parseRecoveredJobs(value: unknown): RecoveredJob[] {
  if (!Array.isArray(value) || value.length > MAX_RECOVERY_CANDIDATES) fail('$', 'must be a bounded array')
  return value.map((entry, index) => {
    const item = record(entry, `$[${index}]`)
    const parsedTerminal = terminal(item.terminalType, `$[${index}].terminalType`)
    const withoutTerminal = { ...item }; delete withoutTerminal.terminalType
    const parsed = parseSubmittedJob(withoutTerminal)
    return { ...parsed, ...(parsedTerminal === undefined ? {} : { terminalType: parsedTerminal }) } as RecoveredJob
  })
}

function parseResult<T extends BackgroundRequest['type']>(request: Extract<BackgroundRequest, { type: T }>, value: unknown): MessageResultMap[T] {
  let parsed: unknown
  switch (request.type) {
    case 'popup:start': case 'popup:cancel': parsed = parsePageState(value); break
    case 'popup:state': parsed = pageState(value, true); break
    case 'setup:status': case 'setup:start': case 'engine:warmup': parsed = parseBrowserSetupStatus(value); break
    case 'job:submit-image': case 'job:submit-document': parsed = parseSubmittedJob(value); break
    case 'job:updates': parsed = parseJobUpdateBatch(
      value,
      (request as Extract<BackgroundRequest, { type: 'job:updates' }>).after,
    ); break
    case 'jobs:recover': parsed = parseRecoveredJobs(value); break
    case 'dictionary:lookup': parsed = parseLookupResult(value); break
    case 'job:patch': {
      const item = record(value); exact(item, ['patchId', 'mimeType', 'bytes']); if (item.mimeType !== 'image/png') fail('$.mimeType', 'must be image/png')
      parsed = { patchId: string(item.patchId, '$.patchId', 512), mimeType: 'image/png', bytes: arrayBuffer(item.bytes, '$.bytes', MAX_RUNTIME_BINARY_BYTES) } satisfies PatchPayload
      break
    }
    case 'font:get': {
      const item = record(value); exact(item, ['fontId', 'bytes']); parsed = { fontId: string(item.fontId, '$.fontId', 512), bytes: arrayBuffer(item.bytes, '$.bytes', MAX_RUNTIME_FONT_BYTES) } satisfies FontPayload
      break
    }
    default:
      if (value !== undefined) fail('$', 'must be undefined')
      parsed = undefined
  }
  return parsed as MessageResultMap[T]
}

export async function sendBackgroundMessage<T extends BackgroundRequest['type']>(message: Extract<BackgroundRequest, { type: T }>): Promise<MessageResultMap[T]> {
  const response = await browser.runtime.sendMessage(message) as unknown
  const envelope = record(response)
  if (envelope.ok === true) { exact(envelope, ['ok', 'value']); return parseResult(message, envelope.value) }
  if (envelope.ok === false) {
    exact(envelope, ['ok', 'error']); const error = record(envelope.error, '$.error'); exact(error, ['code', 'message', 'retryable'], '$.error')
    throw new RuntimeMessageError(string(error.code, '$.error.code', 256), string(error.message, '$.error.message', 2_048), bool(error.retryable, '$.error.retryable'))
  }
  return fail('$', 'has an invalid response envelope')
}
