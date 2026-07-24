export const PROTOCOL_VERSION = 1 as const
export const HSK_STANDARD = '2.0' as const
export const SOURCE_LANGUAGE = 'en' as const
export const TARGET_LANGUAGE = 'zh-CN' as const
export const MAX_PRECEDING_CONTEXT = 12

export type HskLevel = 1 | 2 | 3 | 4 | 5 | 6
export type Point = { x: number; y: number }

export type NativeHandshakeRequest = {
  type: 'start-or-discover-daemon'
  protocolVersion: 1
  extensionVersion: string
  extensionOrigin: string
}

export type NativeReadyResponse = {
  type: 'ready'
  protocolVersion: 1
  engineVersion: string
  port: number
  token: string
  sessionExpiresAtUnixMs: number
  capabilities: {
    sourceLanguages: ['en']
    targetLanguages: ['zh-CN']
    hskLevels: [1, 2, 3, 4, 5, 6]
    modelsReady: boolean
  }
}

export type HealthResponse = {
  protocolVersion: 1
  engineVersion: string
  status: 'ready'
  setupState: BrowserSetupStatus['state']
}

export type BrowserJobRequest = {
  protocolVersion: 1
  clientImageId: string
  sourceSha256: string
  sourceMimeType: string
  naturalWidth: number
  naturalHeight: number
  pageSessionId: string
  pageIndex: number
  settings: {
    sourceLanguage: 'en'
    targetLanguage: 'zh-CN'
    hskStandard: '2.0'
    hskLevel: HskLevel
    readingDirection: 'auto' | 'ltr' | 'rtl'
    translateSoundEffects: false
  }
  precedingContext?: Array<{
    sourceEnglish: string
    chinese: string
  }>
}

export type BrowserJobCreated = {
  protocolVersion: 1
  jobId: string
}

export type RetranslateRequest = {
  protocolVersion: 1
  settings: {
    hskStandard: '2.0'
    hskLevel: HskLevel
  }
  precedingContext?: BrowserJobRequest['precedingContext']
}

export type BrowserJobResult = {
  protocolVersion: 1
  jobId: string
  sourceSha256: string
  sourceWidth: number
  sourceHeight: number
  cleanImageBlobId: string
  cleanImageMimeType: 'image/png' | 'image/webp'
  regions: BrowserRegion[]
  warnings: BrowserWarning[]
  cache: {
    detectionHit: boolean
    ocrHit: boolean
    inpaintHit: boolean
    translationHit: boolean
  }
}

export type BrowserRegion = {
  id: string
  kind: 'dialogue' | 'caption' | 'thought' | 'sfx'
  textPolygon: Point[]
  bubblePolygon?: Point[]
  rotationDegrees: number
  sourceEnglish: string
  faithfulChinese: string
  displayedChinese: string
  pinyin: string
  ocrConfidence: number
  readingOrder: number
  vocabulary: {
    requestedHskLevel: HskLevel
    strictlyValid: boolean
    exceptions: Array<{
      text: string
      reason: 'person-name' | 'place-name' | 'title' | 'unavoidable-proper-noun'
    }>
  }
  style: {
    fontId: string
    category: 'sans' | 'serif' | 'handwritten' | 'display' | 'brush'
    foreground: string
    weight: number
    italicDegrees: number
    outlineColor?: string
    outlineWidthRatio: number
    shadowColor?: string
    shadowXRatio: number
    shadowYRatio: number
    alignment: 'left' | 'center' | 'right'
    writingMode: 'horizontal-tb' | 'vertical-rl'
    lineHeight: number
    letterSpacingEm: number
  }
  layout: {
    suggestedLines: string[]
    fontSizeToImageWidth: number
    safePolygon?: Point[]
  }
}

export type BrowserWarning = {
  code:
    | 'LOW_OCR_CONFIDENCE'
    | 'HSK_EXCEPTION'
    | 'HSK_REWRITE_FAILED'
    | 'TEXT_FIT_DEGRADED'
    | 'STYLE_LOW_CONFIDENCE'
    | 'SFX_SKIPPED'
  regionId?: string
  message: string
}

export type BrowserJobState = 'running' | 'complete' | 'failed' | 'cancelled'
export type BrowserJobStage =
  | 'queued'
  | 'decoding'
  | 'detecting'
  | 'ocr'
  | 'inpainting'
  | 'translating'
  | 'hsk-rewriting'
  | 'hsk-validating'
  | 'styling'
  | 'packaging'
  | 'complete'
  | 'failed'
  | 'cancelled'

export type BrowserJobStatus = {
  revision: number
  jobId: string
  state: BrowserJobState
  stage: BrowserJobStage
  stageProgress?: number
  overallProgress?: number
  current?: number
  total?: number
  message: string
  errorCode?: string
}

export type BrowserSetupStatus = {
  state: 'missing-models' | 'downloading' | 'verifying' | 'ready' | 'failed'
  selectedPackId?: string
  currentFile?: string
  completedBytes?: number
  totalBytes?: number
  requiredDiskBytes?: number
  message: string
  errorCode?: string
}

export type LookupRequest = {
  selectedText: string
  jobId?: string
  regionId?: string
}

export type LookupResult = {
  selectedText: string
  tokens: Array<{
    simplified: string
    pinyin: string
    definitions: string[]
    hskLevel?: HskLevel
    properName: boolean
  }>
  region?: {
    displayedChinese: string
    faithfulChinese: string
    sourceEnglish: string
  }
}

export type ErrorResponse = {
  protocolVersion: 1
  code: string
  message: string
  retryable: boolean
}

export class ContractValidationError extends Error {
  constructor(
    readonly path: string,
    message: string,
  ) {
    super(`${path}: ${message}`)
    this.name = 'ContractValidationError'
  }
}

type UnknownRecord = Record<string, unknown>

const jobStages: readonly BrowserJobStage[] = [
  'queued',
  'decoding',
  'detecting',
  'ocr',
  'inpainting',
  'translating',
  'hsk-rewriting',
  'hsk-validating',
  'styling',
  'packaging',
  'complete',
  'failed',
  'cancelled',
]

const warningCodes: readonly BrowserWarning['code'][] = [
  'LOW_OCR_CONFIDENCE',
  'HSK_EXCEPTION',
  'HSK_REWRITE_FAILED',
  'TEXT_FIT_DEGRADED',
  'STYLE_LOW_CONFIDENCE',
  'SFX_SKIPPED',
]

function fail(path: string, message: string): never {
  throw new ContractValidationError(path, message)
}

function record(value: unknown, path: string): UnknownRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    fail(path, 'must be an object')
  }
  return value as UnknownRecord
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) fail(path, 'must be an array')
  return value
}

function string(value: unknown, path: string, allowEmpty = false): string {
  if (typeof value !== 'string' || (!allowEmpty && value.trim() === '')) {
    fail(path, allowEmpty ? 'must be a string' : 'must be a non-empty string')
  }
  return value
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') fail(path, 'must be a boolean')
  return value
}

function finite(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail(path, 'must be a finite number')
  }
  return value
}

function integer(value: unknown, path: string, minimum = 0): number {
  const parsed = finite(value, path)
  if (!Number.isSafeInteger(parsed) || parsed < minimum) {
    fail(path, `must be an integer greater than or equal to ${minimum}`)
  }
  return parsed
}

function unit(value: unknown, path: string): number {
  const parsed = finite(value, path)
  if (parsed < 0 || parsed > 1) fail(path, 'must be from 0 to 1')
  return parsed
}

function optional<T>(
  value: unknown,
  path: string,
  parser: (value: unknown, path: string) => T,
): T | undefined {
  return value === undefined ? undefined : parser(value, path)
}

function oneOf<const T extends readonly (string | number | boolean)[]>(
  value: unknown,
  path: string,
  values: T,
): T[number] {
  if (!values.includes(value as never)) {
    fail(path, `must be one of ${values.join(', ')}`)
  }
  return value as T[number]
}

function protocol(value: unknown, path = 'protocolVersion'): 1 {
  if (value !== PROTOCOL_VERSION) fail(path, `must equal ${PROTOCOL_VERSION}`)
  return PROTOCOL_VERSION
}

function hskLevel(value: unknown, path: string): HskLevel {
  return oneOf(value, path, [1, 2, 3, 4, 5, 6] as const)
}

function sha256(value: unknown, path: string): string {
  const parsed = string(value, path)
  if (!/^[a-fA-F0-9]{64}$/.test(parsed)) {
    fail(path, 'must be a 64-character hexadecimal SHA-256')
  }
  return parsed
}

function cssColor(value: unknown, path: string): string {
  const parsed = string(value, path)
  if (!/^#(?:[\da-fA-F]{3}|[\da-fA-F]{4}|[\da-fA-F]{6}|[\da-fA-F]{8})$/.test(parsed)) {
    fail(path, 'must be a hexadecimal CSS color')
  }
  return parsed
}

function point(value: unknown, path: string): Point {
  const item = record(value, path)
  return {
    x: unit(item.x, `${path}.x`),
    y: unit(item.y, `${path}.y`),
  }
}

function polygon(value: unknown, path: string): Point[] {
  const items = array(value, path)
  if (items.length < 3) fail(path, 'must contain at least three points')
  return items.map((item, index) => point(item, `${path}[${index}]`))
}

function stringArray(value: unknown, path: string, allowEmptyItems = false): string[] {
  return array(value, path).map((item, index) =>
    string(item, `${path}[${index}]`, allowEmptyItems),
  )
}

function parseRegion(value: unknown, path: string): BrowserRegion {
  const item = record(value, path)
  const kind = oneOf(item.kind, `${path}.kind`, [
    'dialogue',
    'caption',
    'thought',
    'sfx',
  ] as const)
  const sourceEnglish = string(item.sourceEnglish, `${path}.sourceEnglish`, kind === 'sfx')
  const faithfulChinese = string(item.faithfulChinese, `${path}.faithfulChinese`, kind === 'sfx')
  const displayedChinese = string(item.displayedChinese, `${path}.displayedChinese`, kind === 'sfx')
  const vocabulary = record(item.vocabulary, `${path}.vocabulary`)
  const exceptions = array(vocabulary.exceptions, `${path}.vocabulary.exceptions`).map(
    (exception, index) => {
      const parsed = record(exception, `${path}.vocabulary.exceptions[${index}]`)
      return {
        text: string(parsed.text, `${path}.vocabulary.exceptions[${index}].text`),
        reason: oneOf(parsed.reason, `${path}.vocabulary.exceptions[${index}].reason`, [
          'person-name',
          'place-name',
          'title',
          'unavoidable-proper-noun',
        ] as const),
      }
    },
  )
  const strictlyValid = boolean(vocabulary.strictlyValid, `${path}.vocabulary.strictlyValid`)
  if (strictlyValid && exceptions.length > 0) {
    fail(`${path}.vocabulary.strictlyValid`, 'cannot be true when exceptions are present')
  }

  const style = record(item.style, `${path}.style`)
  const weight = integer(style.weight, `${path}.style.weight`, 1)
  if (weight > 1000) fail(`${path}.style.weight`, 'must be at most 1000')
  const outlineWidthRatio = finite(
    style.outlineWidthRatio,
    `${path}.style.outlineWidthRatio`,
  )
  if (outlineWidthRatio < 0) {
    fail(`${path}.style.outlineWidthRatio`, 'must not be negative')
  }
  const lineHeight = finite(style.lineHeight, `${path}.style.lineHeight`)
  if (lineHeight <= 0) fail(`${path}.style.lineHeight`, 'must be positive')

  const layout = record(item.layout, `${path}.layout`)
  const fontSizeToImageWidth = finite(
    layout.fontSizeToImageWidth,
    `${path}.layout.fontSizeToImageWidth`,
  )
  if (fontSizeToImageWidth <= 0) {
    fail(`${path}.layout.fontSizeToImageWidth`, 'must be positive')
  }
  const bubblePolygon = optional(item.bubblePolygon, `${path}.bubblePolygon`, polygon)
  const outlineColor = optional(style.outlineColor, `${path}.style.outlineColor`, cssColor)
  const shadowColor = optional(style.shadowColor, `${path}.style.shadowColor`, cssColor)
  const safePolygon = optional(layout.safePolygon, `${path}.layout.safePolygon`, polygon)

  return {
    id: string(item.id, `${path}.id`),
    kind,
    textPolygon: polygon(item.textPolygon, `${path}.textPolygon`),
    ...(bubblePolygon === undefined ? {} : { bubblePolygon }),
    rotationDegrees: finite(item.rotationDegrees, `${path}.rotationDegrees`),
    sourceEnglish,
    faithfulChinese,
    displayedChinese,
    pinyin: string(item.pinyin, `${path}.pinyin`, kind === 'sfx'),
    ocrConfidence: unit(item.ocrConfidence, `${path}.ocrConfidence`),
    readingOrder: integer(item.readingOrder, `${path}.readingOrder`),
    vocabulary: {
      requestedHskLevel: hskLevel(
        vocabulary.requestedHskLevel,
        `${path}.vocabulary.requestedHskLevel`,
      ),
      strictlyValid,
      exceptions,
    },
    style: {
      fontId: string(style.fontId, `${path}.style.fontId`),
      category: oneOf(style.category, `${path}.style.category`, [
        'sans',
        'serif',
        'handwritten',
        'display',
        'brush',
      ] as const),
      foreground: cssColor(style.foreground, `${path}.style.foreground`),
      weight,
      italicDegrees: finite(style.italicDegrees, `${path}.style.italicDegrees`),
      ...(outlineColor === undefined ? {} : { outlineColor }),
      outlineWidthRatio,
      ...(shadowColor === undefined ? {} : { shadowColor }),
      shadowXRatio: finite(style.shadowXRatio, `${path}.style.shadowXRatio`),
      shadowYRatio: finite(style.shadowYRatio, `${path}.style.shadowYRatio`),
      alignment: oneOf(style.alignment, `${path}.style.alignment`, [
        'left',
        'center',
        'right',
      ] as const),
      writingMode: oneOf(style.writingMode, `${path}.style.writingMode`, [
        'horizontal-tb',
        'vertical-rl',
      ] as const),
      lineHeight,
      letterSpacingEm: finite(style.letterSpacingEm, `${path}.style.letterSpacingEm`),
    },
    layout: {
      suggestedLines: stringArray(
        layout.suggestedLines,
        `${path}.layout.suggestedLines`,
        kind === 'sfx',
      ),
      fontSizeToImageWidth,
      ...(safePolygon === undefined ? {} : { safePolygon }),
    },
  }
}

export function parseNativeHandshakeRequest(value: unknown): NativeHandshakeRequest {
  const item = record(value, '$')
  const extensionOrigin = string(item.extensionOrigin, 'extensionOrigin')
  if (!extensionOrigin.startsWith('moz-extension://') || extensionOrigin.endsWith('/')) {
    fail('extensionOrigin', 'must be a non-empty moz-extension origin without a trailing slash')
  }
  return {
    type: oneOf(item.type, 'type', ['start-or-discover-daemon'] as const),
    protocolVersion: protocol(item.protocolVersion),
    extensionVersion: string(item.extensionVersion, 'extensionVersion'),
    extensionOrigin,
  }
}

export function parseNativeReadyResponse(value: unknown): NativeReadyResponse {
  const item = record(value, '$')
  const capabilities = record(item.capabilities, 'capabilities')
  const sourceLanguages = stringArray(
    capabilities.sourceLanguages,
    'capabilities.sourceLanguages',
  )
  const targetLanguages = stringArray(
    capabilities.targetLanguages,
    'capabilities.targetLanguages',
  )
  const hskLevels = array(capabilities.hskLevels, 'capabilities.hskLevels').map((level, index) =>
    hskLevel(level, `capabilities.hskLevels[${index}]`),
  )
  if (
    sourceLanguages.length !== 1 ||
    sourceLanguages[0] !== SOURCE_LANGUAGE ||
    targetLanguages.length !== 1 ||
    targetLanguages[0] !== TARGET_LANGUAGE ||
    hskLevels.join(',') !== '1,2,3,4,5,6'
  ) {
    fail('capabilities', 'must advertise exactly the protocol v1 capabilities')
  }
  const port = integer(item.port, 'port', 1)
  if (port > 65535) fail('port', 'must be at most 65535')
  const token = string(item.token, 'token')
  if (!/^[\w-]{43,}$/.test(token)) fail('token', 'must be a base64url session token')
  return {
    type: oneOf(item.type, 'type', ['ready'] as const),
    protocolVersion: protocol(item.protocolVersion),
    engineVersion: string(item.engineVersion, 'engineVersion'),
    port,
    token,
    sessionExpiresAtUnixMs: integer(
      item.sessionExpiresAtUnixMs,
      'sessionExpiresAtUnixMs',
      1,
    ),
    capabilities: {
      sourceLanguages: ['en'],
      targetLanguages: ['zh-CN'],
      hskLevels: [1, 2, 3, 4, 5, 6],
      modelsReady: boolean(capabilities.modelsReady, 'capabilities.modelsReady'),
    },
  }
}

export function parseHealthResponse(value: unknown): HealthResponse {
  const item = record(value, '$')
  return {
    protocolVersion: protocol(item.protocolVersion),
    engineVersion: string(item.engineVersion, 'engineVersion'),
    status: oneOf(item.status, 'status', ['ready'] as const),
    setupState: oneOf(item.setupState, 'setupState', [
      'missing-models',
      'downloading',
      'verifying',
      'ready',
      'failed',
    ] as const),
  }
}

export function parseBrowserJobRequest(value: unknown): BrowserJobRequest {
  const item = record(value, '$')
  const settings = record(item.settings, 'settings')
  const precedingContext =
    item.precedingContext === undefined
      ? undefined
      : array(item.precedingContext, 'precedingContext').map((entry, index) => {
          const parsed = record(entry, `precedingContext[${index}]`)
          return {
            sourceEnglish: string(
              parsed.sourceEnglish,
              `precedingContext[${index}].sourceEnglish`,
            ),
            chinese: string(parsed.chinese, `precedingContext[${index}].chinese`),
          }
        })
  if (precedingContext && precedingContext.length > MAX_PRECEDING_CONTEXT) {
    fail('precedingContext', `must contain at most ${MAX_PRECEDING_CONTEXT} entries`)
  }
  const naturalWidth = integer(item.naturalWidth, 'naturalWidth', 1)
  const naturalHeight = integer(item.naturalHeight, 'naturalHeight', 1)
  const sourceMimeType = oneOf(item.sourceMimeType, 'sourceMimeType', [
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/gif',
  ] as const)

  return {
    protocolVersion: protocol(item.protocolVersion),
    clientImageId: string(item.clientImageId, 'clientImageId'),
    sourceSha256: sha256(item.sourceSha256, 'sourceSha256'),
    sourceMimeType,
    naturalWidth,
    naturalHeight,
    pageSessionId: string(item.pageSessionId, 'pageSessionId'),
    pageIndex: integer(item.pageIndex, 'pageIndex'),
    settings: {
      sourceLanguage: oneOf(settings.sourceLanguage, 'settings.sourceLanguage', ['en'] as const),
      targetLanguage: oneOf(settings.targetLanguage, 'settings.targetLanguage', [
        'zh-CN',
      ] as const),
      hskStandard: oneOf(settings.hskStandard, 'settings.hskStandard', ['2.0'] as const),
      hskLevel: hskLevel(settings.hskLevel, 'settings.hskLevel'),
      readingDirection: oneOf(settings.readingDirection, 'settings.readingDirection', [
        'auto',
        'ltr',
        'rtl',
      ] as const),
      translateSoundEffects: oneOf(
        settings.translateSoundEffects,
        'settings.translateSoundEffects',
        [false] as const,
      ),
    },
    ...(precedingContext === undefined ? {} : { precedingContext }),
  }
}

export function parseBrowserJobCreated(value: unknown): BrowserJobCreated {
  const item = record(value, '$')
  return {
    protocolVersion: protocol(item.protocolVersion),
    jobId: string(item.jobId, 'jobId'),
  }
}

export function parseRetranslateRequest(value: unknown): RetranslateRequest {
  const item = record(value, '$')
  const settings = record(item.settings, 'settings')
  const precedingContext =
    item.precedingContext === undefined
      ? undefined
      : array(item.precedingContext, 'precedingContext').map((entry, index) => {
          const parsed = record(entry, `precedingContext[${index}]`)
          return {
            sourceEnglish: string(
              parsed.sourceEnglish,
              `precedingContext[${index}].sourceEnglish`,
            ),
            chinese: string(parsed.chinese, `precedingContext[${index}].chinese`),
          }
        })
  if (precedingContext && precedingContext.length > MAX_PRECEDING_CONTEXT) {
    fail('precedingContext', `must contain at most ${MAX_PRECEDING_CONTEXT} entries`)
  }
  return {
    protocolVersion: protocol(item.protocolVersion),
    settings: {
      hskStandard: oneOf(settings.hskStandard, 'settings.hskStandard', ['2.0'] as const),
      hskLevel: hskLevel(settings.hskLevel, 'settings.hskLevel'),
    },
    ...(precedingContext === undefined ? {} : { precedingContext }),
  }
}

export function parseBrowserJobResult(value: unknown): BrowserJobResult {
  const item = record(value, '$')
  const regions = array(item.regions, 'regions').map((region, index) =>
    parseRegion(region, `regions[${index}]`),
  )
  const regionIds = new Set<string>()
  for (const [index, region] of regions.entries()) {
    if (regionIds.has(region.id)) fail(`regions[${index}].id`, 'region IDs must be unique')
    regionIds.add(region.id)
  }
  const warnings = array(item.warnings, 'warnings').map((warning, index) => {
    const parsed = record(warning, `warnings[${index}]`)
    const regionId = optional(parsed.regionId, `warnings[${index}].regionId`, string)
    if (regionId && !regionIds.has(regionId)) {
      fail(`warnings[${index}].regionId`, 'must reference a region in this result')
    }
    return {
      code: oneOf(parsed.code, `warnings[${index}].code`, warningCodes),
      ...(regionId === undefined ? {} : { regionId }),
      message: string(parsed.message, `warnings[${index}].message`),
    }
  })
  const cache = record(item.cache, 'cache')
  return {
    protocolVersion: protocol(item.protocolVersion),
    jobId: string(item.jobId, 'jobId'),
    sourceSha256: sha256(item.sourceSha256, 'sourceSha256'),
    sourceWidth: integer(item.sourceWidth, 'sourceWidth', 1),
    sourceHeight: integer(item.sourceHeight, 'sourceHeight', 1),
    cleanImageBlobId: string(item.cleanImageBlobId, 'cleanImageBlobId'),
    cleanImageMimeType: oneOf(item.cleanImageMimeType, 'cleanImageMimeType', [
      'image/png',
      'image/webp',
    ] as const),
    regions,
    warnings,
    cache: {
      detectionHit: boolean(cache.detectionHit, 'cache.detectionHit'),
      ocrHit: boolean(cache.ocrHit, 'cache.ocrHit'),
      inpaintHit: boolean(cache.inpaintHit, 'cache.inpaintHit'),
      translationHit: boolean(cache.translationHit, 'cache.translationHit'),
    },
  }
}

export function parseBrowserJobStatus(value: unknown): BrowserJobStatus {
  const item = record(value, '$')
  const state = oneOf(item.state, 'state', [
    'running',
    'complete',
    'failed',
    'cancelled',
  ] as const)
  const stage = oneOf(item.stage, 'stage', jobStages)
  const expectedTerminalStage = state === 'running' ? undefined : state
  if (
    (expectedTerminalStage && stage !== expectedTerminalStage) ||
    (!expectedTerminalStage && ['complete', 'failed', 'cancelled'].includes(stage))
  ) {
    fail('stage', 'must agree with the job state')
  }
  const current = optional(item.current, 'current', integer)
  const total = optional(item.total, 'total', (candidate, path) => integer(candidate, path, 1))
  if ((current === undefined) !== (total === undefined)) {
    fail('current', 'current and total must be present together')
  }
  if (current !== undefined && total !== undefined && current > total) {
    fail('current', 'must not exceed total')
  }
  const errorCode = optional(item.errorCode, 'errorCode', string)
  if (state === 'failed' && !errorCode) fail('errorCode', 'failed jobs require an error code')
  const stageProgress = optional(item.stageProgress, 'stageProgress', unit)
  const overallProgress = optional(item.overallProgress, 'overallProgress', unit)
  return {
    revision: integer(item.revision, 'revision', 1),
    jobId: string(item.jobId, 'jobId'),
    state,
    stage,
    ...(stageProgress === undefined ? {} : { stageProgress }),
    ...(overallProgress === undefined ? {} : { overallProgress }),
    ...(current === undefined ? {} : { current }),
    ...(total === undefined ? {} : { total }),
    message: string(item.message, 'message'),
    ...(errorCode === undefined ? {} : { errorCode }),
  }
}

export function parseBrowserJobStatusSequence(value: unknown): BrowserJobStatus[] {
  const statuses = array(value, '$').map(parseBrowserJobStatus)
  let revision = 0
  let overallProgress = 0
  for (const [index, status] of statuses.entries()) {
    if (status.revision <= revision) {
      fail(`[${index}].revision`, 'must increase monotonically')
    }
    if (
      status.overallProgress !== undefined &&
      status.overallProgress + Number.EPSILON < overallProgress
    ) {
      fail(`[${index}].overallProgress`, 'must not decrease')
    }
    revision = status.revision
    overallProgress = status.overallProgress ?? overallProgress
  }
  return statuses
}

export function parseBrowserSetupStatus(value: unknown): BrowserSetupStatus {
  const item = record(value, '$')
  const state = oneOf(item.state, 'state', [
    'missing-models',
    'downloading',
    'verifying',
    'ready',
    'failed',
  ] as const)
  const completedBytes = optional(item.completedBytes, 'completedBytes', integer)
  const totalBytes = optional(item.totalBytes, 'totalBytes', integer)
  if ((completedBytes === undefined) !== (totalBytes === undefined)) {
    fail('completedBytes', 'completed and total bytes must be present together')
  }
  if (
    completedBytes !== undefined &&
    totalBytes !== undefined &&
    completedBytes > totalBytes
  ) {
    fail('completedBytes', 'must not exceed total bytes')
  }
  const errorCode = optional(item.errorCode, 'errorCode', string)
  if (state === 'failed' && !errorCode) fail('errorCode', 'failed setup requires an error code')
  const selectedPackId = optional(item.selectedPackId, 'selectedPackId', string)
  const currentFile = optional(item.currentFile, 'currentFile', string)
  const requiredDiskBytes = optional(item.requiredDiskBytes, 'requiredDiskBytes', integer)
  return {
    state,
    ...(selectedPackId === undefined ? {} : { selectedPackId }),
    ...(currentFile === undefined ? {} : { currentFile }),
    ...(completedBytes === undefined ? {} : { completedBytes }),
    ...(totalBytes === undefined ? {} : { totalBytes }),
    ...(requiredDiskBytes === undefined ? {} : { requiredDiskBytes }),
    message: string(item.message, 'message'),
    ...(errorCode === undefined ? {} : { errorCode }),
  }
}

export function parseLookupRequest(value: unknown): LookupRequest {
  const item = record(value, '$')
  const selectedText = string(item.selectedText, 'selectedText')
  if ([...selectedText].length > 256) fail('selectedText', 'must contain at most 256 characters')
  const jobId = optional(item.jobId, 'jobId', string)
  const regionId = optional(item.regionId, 'regionId', string)
  if ((jobId === undefined) !== (regionId === undefined)) {
    fail('regionId', 'jobId and regionId must be present together')
  }
  return {
    selectedText,
    ...(jobId === undefined ? {} : { jobId }),
    ...(regionId === undefined ? {} : { regionId }),
  }
}

export function parseLookupResult(value: unknown): LookupResult {
  const item = record(value, '$')
  const tokens = array(item.tokens, 'tokens').map((token, index) => {
    const parsed = record(token, `tokens[${index}]`)
    const properName = boolean(parsed.properName, `tokens[${index}].properName`)
    const parsedHskLevel = optional(parsed.hskLevel, `tokens[${index}].hskLevel`, hskLevel)
    return {
      simplified: string(parsed.simplified, `tokens[${index}].simplified`),
      pinyin: string(parsed.pinyin, `tokens[${index}].pinyin`, properName),
      definitions: stringArray(parsed.definitions, `tokens[${index}].definitions`),
      ...(parsedHskLevel === undefined ? {} : { hskLevel: parsedHskLevel }),
      properName,
    }
  })
  const region =
    item.region === undefined
      ? undefined
      : (() => {
          const parsed = record(item.region, 'region')
          return {
            displayedChinese: string(parsed.displayedChinese, 'region.displayedChinese'),
            faithfulChinese: string(parsed.faithfulChinese, 'region.faithfulChinese'),
            sourceEnglish: string(parsed.sourceEnglish, 'region.sourceEnglish'),
          }
        })()
  return {
    selectedText: string(item.selectedText, 'selectedText'),
    tokens,
    ...(region === undefined ? {} : { region }),
  }
}

export function parseErrorResponse(value: unknown): ErrorResponse {
  const item = record(value, '$')
  return {
    protocolVersion: protocol(item.protocolVersion),
    code: string(item.code, 'code'),
    message: string(item.message, 'message'),
    retryable: boolean(item.retryable, 'retryable'),
  }
}
