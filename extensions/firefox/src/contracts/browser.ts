export const BUILD_FINGERPRINT = 'hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10' as const
export const HSK_STANDARD = '2.0' as const
export const SOURCE_LANGUAGE = 'en' as const
export const TARGET_LANGUAGE = 'zh-CN' as const
export const MAX_CHAPTER_SOURCE_ORDER = 100_000
export const MAX_DOCUMENT_UTF8_BYTES = 1024 * 1024
export const MAX_DOCUMENT_BLOCKS = 2_000
export const MAX_DOCUMENT_BLOCK_UTF8_BYTES = 16 * 1024
export const MAX_VISIBLE_BLOCK_IDS = 64
const MAX_U32 = 0xffff_ffff

export type HskLevel = 1 | 2 | 3 | 4 | 5 | 6
export type LearningMode = 'natural' | 'strict'
export type ReadingDirection = 'ltr' | 'rtl'
export type ChapterKind = 'image' | 'document' | 'unsupported'
export type SourceSpanKind = 'prose' | 'heading' | 'dialogue' | 'caption' | 'thought' | 'sfx'
export type SourceProvenance = 'dom' | 'ocr'
export type Point = { x: number; y: number }

export type NormalizedRect = {
  x: number
  y: number
  width: number
  height: number
}

export type ResourceIdentity = {
  id: string
  repository: string
  repositoryRevision: string
  filename: string
  bytes: number
  sha256: string
}

export type NativeHandshakeRequest = {
  type: 'start-or-discover-daemon'
  buildFingerprint: typeof BUILD_FINGERPRINT
  extensionVersion: string
  extensionOrigin: string
}

export type NativeReadyResponse = {
  type: 'ready'
  buildFingerprint: typeof BUILD_FINGERPRINT
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
  buildFingerprint: typeof BUILD_FINGERPRINT
  engineVersion: string
  status: 'ready'
  setupState: BrowserSetupStatus['state']
  resourceIdentities: ResourceIdentity[]
}

export type TranslationSettings = {
  sourceLanguage: 'en'
  targetLanguage: 'zh-CN'
  hskStandard: '2.0'
  hskLevel: HskLevel
  learningMode: LearningMode
}

export type PageSurfaceKind = 'image' | 'background' | 'canvas' | 'webgl' | 'frame'

export type ImageJobRequest = {
  buildFingerprint: typeof BUILD_FINGERPRINT
  clientRequestId: string
  retryItemIds: string[]
  clientImageId: string
  sourceSha256: string
  sourceMimeType: string
  naturalWidth: number
  naturalHeight: number
  pageSessionId: string
  sourceIndex: number
  chapterSourceOrder: number[]
  surfaceKind: PageSurfaceKind
  visibleRects: NormalizedRect[]
  readingDirection: ReadingDirection
  settings: TranslationSettings
}

export type DocumentSourceBlock = {
  parentBlockId: string
  subItemOrder: number
  itemId: string
  sourceIndex: number
  itemOrder: number
  kind: SourceSpanKind
  provenance: 'dom'
  text: string
}

export type DocumentJobRequest = {
  buildFingerprint: typeof BUILD_FINGERPRINT
  clientRequestId: string
  retryItemIds: string[]
  focus: Extract<JobFocus, { kind: 'document' }>
  pageSessionId: string
  sourceSha256: string
  settings: TranslationSettings
  blocks: DocumentSourceBlock[]
}

export type JobCreated = {
  buildFingerprint: typeof BUILD_FINGERPRINT
  jobId: string
}

export type JobFocus =
  | { kind: 'image'; visibleRects: NormalizedRect[]; active: boolean }
  | { kind: 'document'; visibleBlockIds: string[]; active: boolean }

export type TeachingTerm = {
  text: string
  startChar: number
  endChar: number
  pinyin: string
  definitions: string[]
  requiredLevel?: HskLevel
  reason: 'above-level' | 'outside-list'
}

export type HskState = {
  requestedLevel: HskLevel
  learningMode: LearningMode
  strictlyValid: boolean
  levelCoverage: number
  aboveLevelTokens: string[]
  teachingTerms: TeachingTerm[]
  repairState: 'not-needed' | 'accepted' | 'rejected'
}

export type ProperNameReason = 'person-name' | 'place-name' | 'title' | 'unavoidable-proper-noun'
export type ProtectedName = { sourceText: string; chineseText: string; reason: ProperNameReason }

export type TranslatedText = {
  termination: 'stop'
  protectedNames: ProtectedName[]
  sourceText: string
  baseChinese: string
  displayedChinese: string
  pinyin: string
  hsk: HskState
}

export type RegionStyle = {
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
  colorBands?: Array<{
    position: number
    foreground: string
    outlineColor?: string
  }>
}

export type RegionLayout = {
  suggestedLines: string[]
  fontSizeToImageWidth: number
  safePolygon: Point[]
}

export type RegionConfidenceEvidence = {
  ocrConsensus: number
  geometryCoverage: number
  contextConsistency: number
  cleanupScore: number
}

export type ImageRegion = {
  itemId: string
  itemOrder: number
  kind: SourceSpanKind
  provenance: 'ocr'
  textPolygon: Point[]
  bubblePolygon?: Point[]
  patch: { blobId: string; mimeType: 'image/png'; rect: NormalizedRect }
  text: TranslatedText
  confidence: number
  contextGroup?: string
  confidenceEvidence?: RegionConfidenceEvidence
  style: RegionStyle
  layout: RegionLayout
}

export type PreservedImageRegion = {
  disposition: 'excluded' | 'failed'
  itemId: string
  itemOrder: number
  textPolygon: Point[]
  sourceText: string
  confidence: number
  reason: string
}

export type ReadyDocumentBlock = {
  parentBlockId: string
  subItemOrder: number
  itemId: string
  sourceIndex: number
  itemOrder: number
  kind: SourceSpanKind
  text: TranslatedText
}

export type PreservedDocumentBlock = {
  parentBlockId: string
  subItemOrder: number
  itemId: string
  sourceIndex: number
  itemOrder: number
  kind: SourceSpanKind
  sourceText: string
  reason: string
}

export type JobStage =
  | 'queued'
  | 'warming'
  | 'registering'
  | 'decoding'
  | 'detecting'
  | 'ocr'
  | 'inpainting'
  | 'translating'
  | 'hsk-validating'
  | 'styling'
  | 'packaging'

export type ProgressJobUpdate = {
  sequence: number
  type: 'progress'
  stage: JobStage
  stageProgress?: number
  overallProgress?: number
  current?: number
  total?: number
  message: string
}

export type ImageRegionReadyJobUpdate = {
  sequence: number
  type: 'imageRegionReady'
  region: ImageRegion
}

export type ImageRegionPreservedJobUpdate = {
  sequence: number
  type: 'imageRegionPreserved'
  region: PreservedImageRegion
}

export type DocumentBlockReadyJobUpdate = {
  sequence: number
  type: 'documentBlockReady'
  block: ReadyDocumentBlock
}

export type DocumentBlockPreservedJobUpdate = {
  sequence: number
  type: 'documentBlockPreserved'
  block: PreservedDocumentBlock
}

export type CompleteJobUpdate = {
  sequence: number
  type: 'complete'
  translatedCount: number
  preservedCount: number
  message?: string
}

export type FailedJobUpdate = {
  sequence: number
  type: 'failed'
  code: string
  message: string
  retryable: boolean
}

export type CancelledJobUpdate = {
  sequence: number
  type: 'cancelled'
  message?: string
}

export type JobUpdate =
  | ProgressJobUpdate
  | ImageRegionReadyJobUpdate
  | ImageRegionPreservedJobUpdate
  | DocumentBlockReadyJobUpdate
  | DocumentBlockPreservedJobUpdate
  | CompleteJobUpdate
  | FailedJobUpdate
  | CancelledJobUpdate

export type JobUpdateBatch = {
  jobId: string
  nextSequence: number
  updates: JobUpdate[]
}

export type BrowserSetupStatus = {
  state: 'missing-models' | 'downloading' | 'verifying' | 'warming' | 'ready' | 'failed'
  modelId: string
  currentFile?: string
  completedBytes?: number
  totalBytes?: number
  requiredDiskBytes?: number
  message: string
  errorCode?: string
}

export type LookupContext = {
  displayedChinese: string
  baseChinese: string
  sourceText: string
  properNames: Array<{ text: string; reason: ProperNameReason }>
}
export type LookupRequest =
  | { interaction: 'selection'; selectedText: string; itemId?: string; context?: LookupContext }
  | { interaction: 'hover'; characterOffset: number; itemId: string; context: LookupContext }

export type LookupResult = {
  selectedText: string
  tokens: Array<{
    simplified: string
    pinyin: string
    definitions: string[]
    hskLevel?: HskLevel
    properName: boolean
  }>
  item?: { displayedChinese: string; baseChinese: string; sourceText: string }
}

export type ErrorResponse = { code: string; message: string; retryable: boolean }

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

const utf8 = new TextEncoder()
const jobStages: readonly JobStage[] = [
  'queued',
  'warming',
  'registering',
  'decoding',
  'detecting',
  'ocr',
  'inpainting',
  'translating',
  'hsk-validating',
  'styling',
  'packaging',
]
const spanKinds: readonly SourceSpanKind[] = [
  'prose',
  'heading',
  'dialogue',
  'caption',
  'thought',
  'sfx',
]

function fail(path: string, message: string): never {
  throw new ContractValidationError(path, message)
}

function record(value: unknown, path: string): UnknownRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value))
    fail(path, 'must be an object')
  return value as UnknownRecord
}

function exact(item: UnknownRecord, allowed: readonly string[], path: string): void {
  const expected = new Set(allowed)
  const unexpected = Object.keys(item).find((key) => !expected.has(key))
  if (unexpected) fail(`${path}.${unexpected}`, 'is not permitted')
}

function array(value: unknown, path: string, maximum = 10_000): unknown[] {
  if (!Array.isArray(value)) fail(path, 'must be an array')
  if (value.length > maximum) fail(path, `must contain at most ${maximum} items`)
  return value
}

function string(value: unknown, path: string, allowEmpty = false, maximum = 8_192): string {
  if (
    typeof value !== 'string' ||
    (!allowEmpty && value.trim() === '') ||
    (typeof value === 'string' && [...value].length > maximum)
  ) {
    fail(
      path,
      allowEmpty
        ? `must be a string no longer than ${maximum} characters`
        : `must be a non-empty string no longer than ${maximum} characters`,
    )
  }
  return value
}

function utf8String(value: unknown, path: string, maximumBytes: number): string {
  const parsed = string(value, path, false, maximumBytes)
  if (utf8.encode(parsed).byteLength > maximumBytes)
    fail(path, `must be at most ${maximumBytes} UTF-8 bytes`)
  return parsed
}

function bool(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') fail(path, 'must be a boolean')
  return value
}

function finite(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(path, 'must be a finite number')
  return value
}

function integer(
  value: unknown,
  path: string,
  minimum = 0,
  maximum = Number.MAX_SAFE_INTEGER,
): number {
  const parsed = finite(value, path)
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    fail(path, `must be an integer from ${minimum} to ${maximum}`)
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
  if (!values.includes(value as never)) fail(path, `must be one of ${values.join(', ')}`)
  return value as T[number]
}

function buildFingerprint(value: unknown, path = 'buildFingerprint'): typeof BUILD_FINGERPRINT {
  if (value !== BUILD_FINGERPRINT) fail(path, `must equal ${BUILD_FINGERPRINT}`)
  return BUILD_FINGERPRINT
}

function hskLevel(value: unknown, path: string): HskLevel {
  return oneOf(value, path, [1, 2, 3, 4, 5, 6] as const)
}

function sha256(value: unknown, path: string): string {
  const parsed = string(value, path, false, 64)
  if (!/^[a-f0-9]{64}$/u.test(parsed)) fail(path, 'must be a lowercase SHA-256')
  return parsed
}

function stringArray(
  value: unknown,
  path: string,
  maximum = 10_000,
  itemMaximum = 4_096,
): string[] {
  return array(value, path, maximum).map((item, index) =>
    string(item, `${path}[${index}]`, false, itemMaximum),
  )
}

function point(value: unknown, path: string): Point {
  const item = record(value, path)
  exact(item, ['x', 'y'], path)
  return { x: unit(item.x, `${path}.x`), y: unit(item.y, `${path}.y`) }
}

function polygon(value: unknown, path: string): Point[] {
  const items = array(value, path, 2_048)
  if (items.length < 3) fail(path, 'must contain at least three points')
  return items.map((item, index) => point(item, `${path}[${index}]`))
}

function normalizedRect(value: unknown, path: string): NormalizedRect {
  const item = record(value, path)
  exact(item, ['x', 'y', 'width', 'height'], path)
  const parsed = {
    x: unit(item.x, `${path}.x`),
    y: unit(item.y, `${path}.y`),
    width: unit(item.width, `${path}.width`),
    height: unit(item.height, `${path}.height`),
  }
  if (parsed.width <= 0 || parsed.height <= 0) fail(path, 'must have positive size')
  if (parsed.x + parsed.width > 1 + Number.EPSILON || parsed.y + parsed.height > 1 + Number.EPSILON)
    fail(path, 'must stay inside its source')
  return parsed
}

function visibleRects(value: unknown, path: string): NormalizedRect[] {
  return array(value, path, 64).map((item, index) => normalizedRect(item, `${path}[${index}]`))
}

function sourceOrder(value: unknown, path: string, sourceIndex: number): number[] {
  const values = array(value, path, MAX_CHAPTER_SOURCE_ORDER).map((item, index) =>
    integer(item, `${path}[${index}]`, 0, MAX_U32),
  )
  if (values.length === 0 || !values.includes(sourceIndex)) fail(path, 'must include sourceIndex')
  if (new Set(values).size !== values.length)
    fail(path, 'must contain unique source identities in DOM order')
  return values
}

export function parseTranslationSettings(value: unknown, path = 'settings'): TranslationSettings {
  const item = record(value, path)
  exact(item, ['sourceLanguage', 'targetLanguage', 'hskStandard', 'hskLevel', 'learningMode'], path)
  return {
    sourceLanguage: oneOf(item.sourceLanguage, `${path}.sourceLanguage`, ['en'] as const),
    targetLanguage: oneOf(item.targetLanguage, `${path}.targetLanguage`, ['zh-CN'] as const),
    hskStandard: oneOf(item.hskStandard, `${path}.hskStandard`, ['2.0'] as const),
    hskLevel: hskLevel(item.hskLevel, `${path}.hskLevel`),
    learningMode: oneOf(item.learningMode, `${path}.learningMode`, ['natural', 'strict'] as const),
  }
}

function parseTeachingTerms(value: unknown, path: string): TeachingTerm[] {
  const terms = array(value, path, 512).map((candidate, index) => {
    const childPath = `${path}[${index}]`
    const item = record(candidate, childPath)
    exact(
      item,
      ['text', 'startChar', 'endChar', 'pinyin', 'definitions', 'requiredLevel', 'reason'],
      childPath,
    )
    const startChar = integer(item.startChar, `${childPath}.startChar`)
    const endChar = integer(item.endChar, `${childPath}.endChar`, 1)
    if (endChar <= startChar) fail(`${childPath}.endChar`, 'must exceed startChar')
    const requiredLevel = optional(item.requiredLevel, `${childPath}.requiredLevel`, hskLevel)
    const definitions = stringArray(item.definitions, `${childPath}.definitions`, 32, 2_048)
    if (definitions.length === 0) fail(`${childPath}.definitions`, 'must not be empty')
    return {
      text: string(item.text, `${childPath}.text`, false, 256),
      startChar,
      endChar,
      pinyin: string(item.pinyin, `${childPath}.pinyin`, false, 512),
      definitions,
      ...(requiredLevel === undefined ? {} : { requiredLevel }),
      reason: oneOf(item.reason, `${childPath}.reason`, ['above-level', 'outside-list'] as const),
    }
  })
  for (let index = 1; index < terms.length; index += 1) {
    if (terms[index]!.startChar < terms[index - 1]!.endChar) {
      fail(`${path}[${index}].startChar`, 'teaching terms must be ordered and non-overlapping')
    }
  }
  return terms
}

function parseHsk(value: unknown, path: string): HskState {
  const item = record(value, path)
  exact(
    item,
    [
      'requestedLevel',
      'learningMode',
      'strictlyValid',
      'levelCoverage',
      'aboveLevelTokens',
      'teachingTerms',
      'repairState',
    ],
    path,
  )
  const aboveLevelTokens = stringArray(item.aboveLevelTokens, `${path}.aboveLevelTokens`, 512, 256)
  if (new Set(aboveLevelTokens).size !== aboveLevelTokens.length) {
    fail(`${path}.aboveLevelTokens`, 'must not contain duplicates')
  }
  const teachingTerms = parseTeachingTerms(item.teachingTerms, `${path}.teachingTerms`)
  const strictlyValid = bool(item.strictlyValid, `${path}.strictlyValid`)
  if (strictlyValid && (aboveLevelTokens.length > 0 || teachingTerms.length > 0))
    fail(`${path}.strictlyValid`, 'cannot accompany unresolved terms')
  return {
    requestedLevel: hskLevel(item.requestedLevel, `${path}.requestedLevel`),
    learningMode: oneOf(item.learningMode, `${path}.learningMode`, ['natural', 'strict'] as const),
    strictlyValid,
    levelCoverage: unit(item.levelCoverage, `${path}.levelCoverage`),
    aboveLevelTokens,
    teachingTerms,
    repairState: oneOf(item.repairState, `${path}.repairState`, [
      'not-needed',
      'accepted',
      'rejected',
    ] as const),
  }
}

function parseTranslatedText(value: unknown, path: string): TranslatedText {
  const item = record(value, path)
  exact(
    item,
    [
      'sourceText',
      'baseChinese',
      'displayedChinese',
      'pinyin',
      'hsk',
      'termination',
      'protectedNames',
    ],
    path,
  )
  const protectedNames = array(item.protectedNames, `${path}.protectedNames`, 32).map(
    (value, index) => {
      const namePath = `${path}.protectedNames[${index}]`
      const name = record(value, namePath)
      exact(name, ['sourceText', 'chineseText', 'reason'], namePath)
      return {
        sourceText: string(name.sourceText, `${namePath}.sourceText`, false, 128),
        chineseText: string(name.chineseText, `${namePath}.chineseText`, false, 128),
        reason: oneOf(name.reason, `${namePath}.reason`, [
          'person-name',
          'place-name',
          'title',
          'unavoidable-proper-noun',
        ] as const),
      }
    },
  )
  const parsed = {
    termination: oneOf(item.termination, `${path}.termination`, ['stop'] as const),
    protectedNames,
    sourceText: string(item.sourceText, `${path}.sourceText`, false, MAX_DOCUMENT_BLOCK_UTF8_BYTES),
    baseChinese: string(
      item.baseChinese,
      `${path}.baseChinese`,
      false,
      MAX_DOCUMENT_BLOCK_UTF8_BYTES,
    ),
    displayedChinese: string(
      item.displayedChinese,
      `${path}.displayedChinese`,
      false,
      MAX_DOCUMENT_BLOCK_UTF8_BYTES,
    ),
    pinyin: string(item.pinyin, `${path}.pinyin`, false, MAX_DOCUMENT_BLOCK_UTF8_BYTES * 2),
    hsk: parseHsk(item.hsk, `${path}.hsk`),
  }
  if (
    protectedNames.some(
      (name) =>
        !parsed.sourceText.includes(name.sourceText) ||
        !parsed.displayedChinese.includes(name.chineseText) ||
        /[a-z]/iu.test(name.chineseText),
    )
  )
    fail(`${path}.protectedNames`, 'names must match exact source and Chinese text')
  if (parsed.hsk.repairState === 'rejected')
    fail(`${path}.hsk.repairState`, 'terminal translated text cannot contain a rejected repair')
  return parsed
}

function cssColor(value: unknown, path: string): string {
  const parsed = string(value, path, false, 9)
  if (!/^#(?:[\da-f]{3}|[\da-f]{4}|[\da-f]{6}|[\da-f]{8})$/iu.test(parsed))
    fail(path, 'must be a hexadecimal CSS color')
  return parsed
}

function parseStyle(value: unknown, path: string): RegionStyle {
  const item = record(value, path)
  exact(
    item,
    [
      'fontId',
      'category',
      'foreground',
      'weight',
      'italicDegrees',
      'outlineColor',
      'outlineWidthRatio',
      'shadowColor',
      'shadowXRatio',
      'shadowYRatio',
      'alignment',
      'writingMode',
      'lineHeight',
      'letterSpacingEm',
      'colorBands',
    ],
    path,
  )
  const outlineColor = optional(item.outlineColor, `${path}.outlineColor`, cssColor)
  const shadowColor = optional(item.shadowColor, `${path}.shadowColor`, cssColor)
  const bands =
    item.colorBands === undefined
      ? undefined
      : array(item.colorBands, `${path}.colorBands`, 512).map((candidate, index) => {
          const bandPath = `${path}.colorBands[${index}]`
          const band = record(candidate, bandPath)
          exact(band, ['position', 'foreground', 'outlineColor'], bandPath)
          const bandOutline = optional(band.outlineColor, `${bandPath}.outlineColor`, cssColor)
          return {
            position: unit(band.position, `${bandPath}.position`),
            foreground: cssColor(band.foreground, `${bandPath}.foreground`),
            ...(bandOutline === undefined ? {} : { outlineColor: bandOutline }),
          }
        })
  const weight = integer(item.weight, `${path}.weight`, 1, 1_000)
  const outlineWidthRatio = finite(item.outlineWidthRatio, `${path}.outlineWidthRatio`)
  const lineHeight = finite(item.lineHeight, `${path}.lineHeight`)
  if (outlineWidthRatio < 0 || lineHeight <= 0) fail(path, 'contains invalid style metrics')
  return {
    fontId: string(item.fontId, `${path}.fontId`, false, 512),
    category: oneOf(item.category, `${path}.category`, [
      'sans',
      'serif',
      'handwritten',
      'display',
      'brush',
    ] as const),
    foreground: cssColor(item.foreground, `${path}.foreground`),
    weight,
    italicDegrees: finite(item.italicDegrees, `${path}.italicDegrees`),
    ...(outlineColor === undefined ? {} : { outlineColor }),
    outlineWidthRatio,
    ...(shadowColor === undefined ? {} : { shadowColor }),
    shadowXRatio: finite(item.shadowXRatio, `${path}.shadowXRatio`),
    shadowYRatio: finite(item.shadowYRatio, `${path}.shadowYRatio`),
    alignment: oneOf(item.alignment, `${path}.alignment`, ['left', 'center', 'right'] as const),
    writingMode: oneOf(item.writingMode, `${path}.writingMode`, [
      'horizontal-tb',
      'vertical-rl',
    ] as const),
    lineHeight,
    letterSpacingEm: finite(item.letterSpacingEm, `${path}.letterSpacingEm`),
    ...(bands === undefined ? {} : { colorBands: bands }),
  }
}

function parseLayout(value: unknown, path: string): RegionLayout {
  const item = record(value, path)
  exact(item, ['suggestedLines', 'fontSizeToImageWidth', 'safePolygon'], path)
  const size = unit(item.fontSizeToImageWidth, `${path}.fontSizeToImageWidth`)
  if (size <= 0) fail(`${path}.fontSizeToImageWidth`, 'must be positive')
  return {
    suggestedLines: stringArray(item.suggestedLines, `${path}.suggestedLines`, 512, 4_096),
    fontSizeToImageWidth: size,
    safePolygon: polygon(item.safePolygon, `${path}.safePolygon`),
  }
}

function parseDocumentBlock(value: unknown, path: string): DocumentSourceBlock {
  const item = record(value, path)
  exact(
    item,
    [
      'itemId',
      'parentBlockId',
      'subItemOrder',
      'sourceIndex',
      'itemOrder',
      'kind',
      'provenance',
      'text',
    ],
    path,
  )
  const text = utf8String(item.text, `${path}.text`, MAX_DOCUMENT_BLOCK_UTF8_BYTES)
  if (text.includes('\r') || text.includes('\0') || text.trim() !== text) {
    fail(`${path}.text`, 'must be normalized, trimmed text using LF line breaks')
  }
  return {
    parentBlockId: string(item.parentBlockId, `${path}.parentBlockId`, false, 256),
    subItemOrder: integer(item.subItemOrder, `${path}.subItemOrder`, 0, MAX_U32),
    itemId: string(item.itemId, `${path}.itemId`, false, 256),
    sourceIndex: integer(item.sourceIndex, `${path}.sourceIndex`, 0, MAX_U32),
    itemOrder: integer(item.itemOrder, `${path}.itemOrder`, 0, MAX_U32),
    kind: oneOf(item.kind, `${path}.kind`, spanKinds),
    provenance: oneOf(item.provenance, `${path}.provenance`, ['dom'] as const),
    text,
  }
}

export function parseNativeHandshakeRequest(value: unknown): NativeHandshakeRequest {
  const item = record(value, '$')
  exact(item, ['type', 'buildFingerprint', 'extensionVersion', 'extensionOrigin'], '$')
  const extensionOrigin = string(item.extensionOrigin, 'extensionOrigin')
  if (!extensionOrigin.startsWith('moz-extension://') || extensionOrigin.endsWith('/'))
    fail('extensionOrigin', 'must be a Firefox extension origin without a trailing slash')
  return {
    type: oneOf(item.type, 'type', ['start-or-discover-daemon'] as const),
    buildFingerprint: buildFingerprint(item.buildFingerprint),
    extensionVersion: string(item.extensionVersion, 'extensionVersion', false, 128),
    extensionOrigin,
  }
}

export function parseNativeReadyResponse(value: unknown): NativeReadyResponse {
  const item = record(value, '$')
  exact(
    item,
    [
      'type',
      'buildFingerprint',
      'engineVersion',
      'port',
      'token',
      'sessionExpiresAtUnixMs',
      'capabilities',
    ],
    '$',
  )
  const capabilities = record(item.capabilities, 'capabilities')
  exact(
    capabilities,
    ['sourceLanguages', 'targetLanguages', 'hskLevels', 'modelsReady'],
    'capabilities',
  )
  const sources = stringArray(capabilities.sourceLanguages, 'capabilities.sourceLanguages', 1)
  const targets = stringArray(capabilities.targetLanguages, 'capabilities.targetLanguages', 1)
  const levels = array(capabilities.hskLevels, 'capabilities.hskLevels', 6).map((level, index) =>
    hskLevel(level, `capabilities.hskLevels[${index}]`),
  )
  if (sources.join() !== 'en' || targets.join() !== 'zh-CN' || levels.join() !== '1,2,3,4,5,6')
    fail('capabilities', 'must advertise the required capabilities exactly')
  const port = integer(item.port, 'port', 1, 65_535)
  const token = string(item.token, 'token')
  if (!/^[\w-]{43,}$/u.test(token)) fail('token', 'must be a base64url session token')
  return {
    type: oneOf(item.type, 'type', ['ready'] as const),
    buildFingerprint: buildFingerprint(item.buildFingerprint),
    engineVersion: string(item.engineVersion, 'engineVersion', false, 128),
    port,
    token,
    sessionExpiresAtUnixMs: integer(item.sessionExpiresAtUnixMs, 'sessionExpiresAtUnixMs', 1),
    capabilities: {
      sourceLanguages: ['en'],
      targetLanguages: ['zh-CN'],
      hskLevels: [1, 2, 3, 4, 5, 6],
      modelsReady: bool(capabilities.modelsReady, 'capabilities.modelsReady'),
    },
  }
}

function parseResourceIdentity(value: unknown, path: string): ResourceIdentity {
  const item = record(value, path)
  exact(item, ['id', 'repository', 'repositoryRevision', 'filename', 'bytes', 'sha256'], path)
  const id = string(item.id, `${path}.id`, false, 128)
  const repository = string(item.repository, `${path}.repository`, false, 256)
  const repositoryRevision = string(
    item.repositoryRevision,
    `${path}.repositoryRevision`,
    false,
    40,
  )
  const filename = string(item.filename, `${path}.filename`, false, 255)
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/u.test(id)) fail(`${path}.id`, 'must be lowercase kebab-case')
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(repository)) {
    fail(`${path}.repository`, 'must contain exactly one owner/name repository')
  }
  if (!/^[a-f0-9]{40}$/u.test(repositoryRevision)) {
    fail(`${path}.repositoryRevision`, 'must be a lowercase 40-character hexadecimal revision')
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(filename) || filename === '.' || filename === '..') {
    fail(`${path}.filename`, 'must be a safe ASCII filename')
  }
  return {
    id,
    repository,
    repositoryRevision,
    filename,
    bytes: integer(item.bytes, `${path}.bytes`, 1),
    sha256: sha256(item.sha256, `${path}.sha256`),
  }
}

export function parseHealthResponse(value: unknown): HealthResponse {
  const item = record(value, '$')
  exact(
    item,
    ['buildFingerprint', 'engineVersion', 'status', 'setupState', 'resourceIdentities'],
    '$',
  )
  const resourceIdentities = array(item.resourceIdentities, 'resourceIdentities', 256).map(
    (entry, index) => parseResourceIdentity(entry, `resourceIdentities[${index}]`),
  )
  if (resourceIdentities.length === 0) fail('resourceIdentities', 'must not be empty')
  for (let index = 1; index < resourceIdentities.length; index += 1) {
    if (resourceIdentities[index - 1]!.id >= resourceIdentities[index]!.id) {
      fail(`resourceIdentities[${index}].id`, 'must be unique and sorted')
    }
  }
  return {
    buildFingerprint: buildFingerprint(item.buildFingerprint),
    engineVersion: string(item.engineVersion, 'engineVersion', false, 128),
    status: oneOf(item.status, 'status', ['ready'] as const),
    setupState: oneOf(item.setupState, 'setupState', [
      'missing-models',
      'downloading',
      'verifying',
      'warming',
      'ready',
      'failed',
    ] as const),
    resourceIdentities,
  }
}

export function parseImageJobRequest(value: unknown): ImageJobRequest {
  const item = record(value, '$')
  exact(
    item,
    [
      'buildFingerprint',
      'clientRequestId',
      'retryItemIds',
      'clientImageId',
      'sourceSha256',
      'sourceMimeType',
      'naturalWidth',
      'naturalHeight',
      'pageSessionId',
      'sourceIndex',
      'chapterSourceOrder',
      'surfaceKind',
      'visibleRects',
      'readingDirection',
      'settings',
    ],
    '$',
  )
  const sourceIndex = integer(item.sourceIndex, 'sourceIndex', 0, MAX_U32)
  return {
    clientRequestId: string(item.clientRequestId, 'clientRequestId', false, 128),
    retryItemIds: retryIds(item.retryItemIds),
    buildFingerprint: buildFingerprint(item.buildFingerprint),
    clientImageId: string(item.clientImageId, 'clientImageId', false, 512),
    sourceSha256: sha256(item.sourceSha256, 'sourceSha256'),
    sourceMimeType: oneOf(item.sourceMimeType, 'sourceMimeType', [
      'image/png',
      'image/jpeg',
      'image/webp',
      'image/gif',
    ] as const),
    naturalWidth: integer(item.naturalWidth, 'naturalWidth', 1),
    naturalHeight: integer(item.naturalHeight, 'naturalHeight', 1),
    pageSessionId: string(item.pageSessionId, 'pageSessionId', false, 256),
    sourceIndex,
    chapterSourceOrder: sourceOrder(item.chapterSourceOrder, 'chapterSourceOrder', sourceIndex),
    surfaceKind: oneOf(item.surfaceKind, 'surfaceKind', [
      'image',
      'background',
      'canvas',
      'webgl',
      'frame',
    ] as const),
    visibleRects: visibleRects(item.visibleRects, 'visibleRects'),
    readingDirection: oneOf(item.readingDirection, 'readingDirection', ['ltr', 'rtl'] as const),
    settings: parseTranslationSettings(item.settings, 'settings'),
  }
}

export function parseDocumentJobRequest(value: unknown): DocumentJobRequest {
  const item = record(value, '$')
  exact(
    item,
    [
      'buildFingerprint',
      'clientRequestId',
      'retryItemIds',
      'pageSessionId',
      'sourceSha256',
      'settings',
      'blocks',
      'focus',
    ],
    '$',
  )
  const blocks = array(item.blocks, 'blocks', MAX_DOCUMENT_BLOCKS).map((block, index) =>
    parseDocumentBlock(block, `blocks[${index}]`),
  )
  if (blocks.length === 0) fail('blocks', 'must not be empty')
  const identities = new Set<string>()
  const parentOrders = new Map<string, number>()
  for (let index = 0; index < blocks.length; index += 1) {
    const block = blocks[index]!
    if (block.subItemOrder !== (parentOrders.get(block.parentBlockId) ?? 0))
      fail(
        `blocks[${index}].subItemOrder`,
        'must start at zero and increase consecutively within its parent',
      )
    parentOrders.set(block.parentBlockId, block.subItemOrder + 1)
    if (identities.has(block.itemId)) fail(`blocks[${index}].itemId`, 'must be unique')
    identities.add(block.itemId)
    if (index > 0) {
      const previous = blocks[index - 1]!
      if (
        previous.sourceIndex > block.sourceIndex ||
        (previous.sourceIndex === block.sourceIndex && previous.itemOrder >= block.itemOrder)
      )
        fail(`blocks[${index}]`, 'must be ordered by sourceIndex and itemOrder')
    }
  }
  const focus = parseJobFocus(item.focus)
  if (focus.kind !== 'document' || focus.visibleBlockIds.some((id) => !identities.has(id)))
    fail('focus', 'must identify submitted document blocks')
  const retryItemIds = retryIds(item.retryItemIds)
  if (retryItemIds.some((id) => !identities.has(id)))
    fail('retryItemIds', 'must identify submitted source blocks')
  const parsed: DocumentJobRequest = {
    focus,
    clientRequestId: string(item.clientRequestId, 'clientRequestId', false, 128),
    retryItemIds,
    buildFingerprint: buildFingerprint(item.buildFingerprint),
    pageSessionId: string(item.pageSessionId, 'pageSessionId', false, 256),
    sourceSha256: sha256(item.sourceSha256, 'sourceSha256'),
    settings: parseTranslationSettings(item.settings, 'settings'),
    blocks,
  }
  if (utf8.encode(JSON.stringify(parsed)).byteLength > MAX_DOCUMENT_UTF8_BYTES)
    fail('$', `must be at most ${MAX_DOCUMENT_UTF8_BYTES} UTF-8 bytes`)
  return parsed
}

function retryIds(value: unknown): string[] {
  const ids = array(value, 'retryItemIds', MAX_DOCUMENT_BLOCKS).map((id, index) =>
    string(id, `retryItemIds[${index}]`, false, 256),
  )
  if (new Set(ids).size !== ids.length) fail('retryItemIds', 'must be unique')
  return ids
}

export function parseJobCreated(value: unknown): JobCreated {
  const item = record(value, '$')
  exact(item, ['buildFingerprint', 'jobId'], '$')
  return {
    buildFingerprint: buildFingerprint(item.buildFingerprint),
    jobId: string(item.jobId, 'jobId', false, 512),
  }
}

export function parseJobFocus(value: unknown): JobFocus {
  const item = record(value, '$')
  const kind = oneOf(item.kind, 'kind', ['image', 'document'] as const)
  if (kind === 'image') {
    exact(item, ['kind', 'visibleRects', 'active'], '$')
    return {
      kind,
      visibleRects: visibleRects(item.visibleRects, 'visibleRects'),
      active: bool(item.active, 'active'),
    }
  }
  exact(item, ['kind', 'visibleBlockIds', 'active'], '$')
  const visibleBlockIds = stringArray(
    item.visibleBlockIds,
    'visibleBlockIds',
    MAX_VISIBLE_BLOCK_IDS,
    256,
  )
  if (new Set(visibleBlockIds).size !== visibleBlockIds.length)
    fail('visibleBlockIds', 'must be unique')
  return { kind, visibleBlockIds, active: bool(item.active, 'active') }
}

function parseImageRegion(value: unknown, path: string): ImageRegion {
  const item = record(value, path)
  exact(
    item,
    [
      'itemId',
      'itemOrder',
      'kind',
      'provenance',
      'textPolygon',
      'bubblePolygon',
      'patch',
      'text',
      'confidence',
      'contextGroup',
      'confidenceEvidence',
      'style',
      'layout',
    ],
    path,
  )
  const patch = record(item.patch, `${path}.patch`)
  exact(patch, ['blobId', 'mimeType', 'rect'], `${path}.patch`)
  const bubblePolygon = optional(item.bubblePolygon, `${path}.bubblePolygon`, polygon)
  const contextGroup = optional(item.contextGroup, `${path}.contextGroup`, string)
  const evidence =
    item.confidenceEvidence === undefined
      ? undefined
      : (() => {
          const parsed = record(item.confidenceEvidence, `${path}.confidenceEvidence`)
          exact(
            parsed,
            ['ocrConsensus', 'geometryCoverage', 'contextConsistency', 'cleanupScore'],
            `${path}.confidenceEvidence`,
          )
          return {
            ocrConsensus: unit(parsed.ocrConsensus, `${path}.confidenceEvidence.ocrConsensus`),
            geometryCoverage: unit(
              parsed.geometryCoverage,
              `${path}.confidenceEvidence.geometryCoverage`,
            ),
            contextConsistency: unit(
              parsed.contextConsistency,
              `${path}.confidenceEvidence.contextConsistency`,
            ),
            cleanupScore: unit(parsed.cleanupScore, `${path}.confidenceEvidence.cleanupScore`),
          }
        })()
  return {
    itemId: string(item.itemId, `${path}.itemId`, false, 512),
    itemOrder: integer(item.itemOrder, `${path}.itemOrder`),
    kind: oneOf(item.kind, `${path}.kind`, spanKinds),
    provenance: oneOf(item.provenance, `${path}.provenance`, ['ocr'] as const),
    textPolygon: polygon(item.textPolygon, `${path}.textPolygon`),
    ...(bubblePolygon === undefined ? {} : { bubblePolygon }),
    patch: {
      blobId: string(patch.blobId, `${path}.patch.blobId`, false, 512),
      mimeType: oneOf(patch.mimeType, `${path}.patch.mimeType`, ['image/png'] as const),
      rect: normalizedRect(patch.rect, `${path}.patch.rect`),
    },
    text: parseTranslatedText(item.text, `${path}.text`),
    confidence: unit(item.confidence, `${path}.confidence`),
    ...(contextGroup === undefined ? {} : { contextGroup }),
    ...(evidence === undefined ? {} : { confidenceEvidence: evidence }),
    style: parseStyle(item.style, `${path}.style`),
    layout: parseLayout(item.layout, `${path}.layout`),
  }
}

function parsePreservedImageRegion(value: unknown, path: string): PreservedImageRegion {
  const item = record(value, path)
  exact(
    item,
    ['disposition', 'itemId', 'itemOrder', 'textPolygon', 'sourceText', 'confidence', 'reason'],
    path,
  )
  return {
    disposition: oneOf(item.disposition, `${path}.disposition`, ['excluded', 'failed'] as const),
    itemId: string(item.itemId, `${path}.itemId`, false, 512),
    itemOrder: integer(item.itemOrder, `${path}.itemOrder`),
    textPolygon: polygon(item.textPolygon, `${path}.textPolygon`),
    sourceText: string(item.sourceText, `${path}.sourceText`, true, MAX_DOCUMENT_BLOCK_UTF8_BYTES),
    confidence: unit(item.confidence, `${path}.confidence`),
    reason: string(item.reason, `${path}.reason`, false, 2_048),
  }
}

function parseReadyDocumentBlock(value: unknown, path: string): ReadyDocumentBlock {
  const item = record(value, path)
  exact(
    item,
    ['itemId', 'parentBlockId', 'subItemOrder', 'sourceIndex', 'itemOrder', 'kind', 'text'],
    path,
  )
  return {
    parentBlockId: string(item.parentBlockId, `${path}.parentBlockId`, false, 256),
    subItemOrder: integer(item.subItemOrder, `${path}.subItemOrder`),
    itemId: string(item.itemId, `${path}.itemId`, false, 512),
    sourceIndex: integer(item.sourceIndex, `${path}.sourceIndex`),
    itemOrder: integer(item.itemOrder, `${path}.itemOrder`),
    kind: oneOf(item.kind, `${path}.kind`, spanKinds),
    text: parseTranslatedText(item.text, `${path}.text`),
  }
}

function parsePreservedDocumentBlock(value: unknown, path: string): PreservedDocumentBlock {
  const item = record(value, path)
  exact(
    item,
    [
      'itemId',
      'parentBlockId',
      'subItemOrder',
      'sourceIndex',
      'itemOrder',
      'kind',
      'sourceText',
      'reason',
    ],
    path,
  )
  return {
    parentBlockId: string(item.parentBlockId, `${path}.parentBlockId`, false, 256),
    subItemOrder: integer(item.subItemOrder, `${path}.subItemOrder`),
    itemId: string(item.itemId, `${path}.itemId`, false, 512),
    sourceIndex: integer(item.sourceIndex, `${path}.sourceIndex`),
    itemOrder: integer(item.itemOrder, `${path}.itemOrder`),
    kind: oneOf(item.kind, `${path}.kind`, spanKinds),
    sourceText: utf8String(item.sourceText, `${path}.sourceText`, MAX_DOCUMENT_BLOCK_UTF8_BYTES),
    reason: string(item.reason, `${path}.reason`, false, 2_048),
  }
}

export function parseJobUpdate(value: unknown, path = '$'): JobUpdate {
  const item = record(value, path)
  const sequence = integer(item.sequence, `${path}.sequence`, 1)
  const type = oneOf(item.type, `${path}.type`, [
    'progress',
    'imageRegionReady',
    'imageRegionPreserved',
    'documentBlockReady',
    'documentBlockPreserved',
    'complete',
    'failed',
    'cancelled',
  ] as const)
  if (type === 'progress') {
    exact(
      item,
      [
        'sequence',
        'type',
        'stage',
        'stageProgress',
        'overallProgress',
        'current',
        'total',
        'message',
      ],
      path,
    )
    const current = optional(item.current, `${path}.current`, integer)
    const total = optional(item.total, `${path}.total`, (candidate, childPath) =>
      integer(candidate, childPath, 1),
    )
    if (
      (current === undefined) !== (total === undefined) ||
      (current !== undefined && total !== undefined && current > total)
    )
      fail(`${path}.current`, 'must be paired with and not exceed total')
    const stageProgress = optional(item.stageProgress, `${path}.stageProgress`, unit)
    const overallProgress = optional(item.overallProgress, `${path}.overallProgress`, unit)
    return {
      sequence,
      type,
      stage: oneOf(item.stage, `${path}.stage`, jobStages),
      ...(stageProgress === undefined ? {} : { stageProgress }),
      ...(overallProgress === undefined ? {} : { overallProgress }),
      ...(current === undefined ? {} : { current }),
      ...(total === undefined ? {} : { total }),
      message: string(item.message, `${path}.message`, false, 2_048),
    }
  }
  if (type === 'imageRegionReady') {
    exact(item, ['sequence', 'type', 'region'], path)
    return { sequence, type, region: parseImageRegion(item.region, `${path}.region`) }
  }
  if (type === 'imageRegionPreserved') {
    exact(item, ['sequence', 'type', 'region'], path)
    return { sequence, type, region: parsePreservedImageRegion(item.region, `${path}.region`) }
  }
  if (type === 'documentBlockReady') {
    exact(item, ['sequence', 'type', 'block'], path)
    return { sequence, type, block: parseReadyDocumentBlock(item.block, `${path}.block`) }
  }
  if (type === 'documentBlockPreserved') {
    exact(item, ['sequence', 'type', 'block'], path)
    return { sequence, type, block: parsePreservedDocumentBlock(item.block, `${path}.block`) }
  }
  if (type === 'complete') {
    exact(item, ['sequence', 'type', 'translatedCount', 'preservedCount', 'message'], path)
    const message = optional(item.message, `${path}.message`, (candidate, childPath) =>
      string(candidate, childPath, false, 2_048),
    )
    return {
      sequence,
      type,
      translatedCount: integer(item.translatedCount, `${path}.translatedCount`),
      preservedCount: integer(item.preservedCount, `${path}.preservedCount`),
      ...(message === undefined ? {} : { message }),
    }
  }
  if (type === 'failed') {
    exact(item, ['sequence', 'type', 'code', 'message', 'retryable'], path)
    return {
      sequence,
      type,
      code: string(item.code, `${path}.code`, false, 256),
      message: string(item.message, `${path}.message`, false, 2_048),
      retryable: bool(item.retryable, `${path}.retryable`),
    }
  }
  exact(item, ['sequence', 'type', 'message'], path)
  const message = optional(item.message, `${path}.message`, (candidate, childPath) =>
    string(candidate, childPath, false, 2_048),
  )
  return { sequence, type, ...(message === undefined ? {} : { message }) }
}

export function parseJobUpdateBatch(value: unknown, after = 0): JobUpdateBatch {
  const item = record(value, '$')
  exact(item, ['jobId', 'nextSequence', 'updates'], '$')
  const nextSequence = integer(item.nextSequence, 'nextSequence')
  if (nextSequence < after) fail('nextSequence', 'must not move backwards')
  const updates = array(item.updates, 'updates', 1_024).map((entry, index) =>
    parseJobUpdate(entry, `updates[${index}]`),
  )
  let previous = after
  for (const [index, update] of updates.entries()) {
    if (update.sequence !== previous + 1 || update.sequence > nextSequence)
      fail(`updates[${index}].sequence`, 'must be contiguous after the requested cursor')
    if (['complete', 'failed', 'cancelled'].includes(update.type) && index !== updates.length - 1)
      fail(`updates[${index}].type`, 'terminal updates must be last')
    previous = update.sequence
  }
  if (
    (updates.length > 0 && previous !== nextSequence) ||
    (updates.length === 0 && nextSequence !== after)
  )
    fail('nextSequence', 'must equal the last returned sequence')
  return { jobId: string(item.jobId, 'jobId', false, 512), nextSequence, updates }
}

export function parseBrowserSetupStatus(value: unknown): BrowserSetupStatus {
  const item = record(value, '$')
  exact(
    item,
    [
      'state',
      'modelId',
      'currentFile',
      'completedBytes',
      'totalBytes',
      'requiredDiskBytes',
      'message',
      'errorCode',
    ],
    '$',
  )
  const state = oneOf(item.state, 'state', [
    'missing-models',
    'downloading',
    'verifying',
    'warming',
    'ready',
    'failed',
  ] as const)
  const currentFile = optional(item.currentFile, 'currentFile', string)
  const completedBytes = optional(item.completedBytes, 'completedBytes', integer)
  const totalBytes = optional(item.totalBytes, 'totalBytes', integer)
  const requiredDiskBytes = optional(item.requiredDiskBytes, 'requiredDiskBytes', integer)
  const errorCode = optional(item.errorCode, 'errorCode', string)
  if (
    (completedBytes === undefined) !== (totalBytes === undefined) ||
    (completedBytes !== undefined && totalBytes !== undefined && completedBytes > totalBytes)
  )
    fail('completedBytes', 'must be paired with and not exceed totalBytes')
  if (state === 'failed' && !errorCode) fail('errorCode', 'is required for failed setup')
  return {
    state,
    modelId: string(item.modelId, 'modelId', false, 128),
    ...(currentFile === undefined ? {} : { currentFile }),
    ...(completedBytes === undefined ? {} : { completedBytes }),
    ...(totalBytes === undefined ? {} : { totalBytes }),
    ...(requiredDiskBytes === undefined ? {} : { requiredDiskBytes }),
    message: string(item.message, 'message', false, 2_048),
    ...(errorCode === undefined ? {} : { errorCode }),
  }
}

export function parseLookupRequest(value: unknown): LookupRequest {
  const item = record(value, '$')
  const interaction = oneOf(item.interaction, 'interaction', ['selection', 'hover'] as const)
  const itemId = optional(item.itemId, 'itemId', string)
  const context =
    item.context === undefined
      ? undefined
      : (() => {
          const ctx = record(item.context, 'context')
          exact(ctx, ['displayedChinese', 'baseChinese', 'sourceText', 'properNames'], 'context')
          const properNames = array(ctx.properNames, 'context.properNames', 32).map(
            (value, index) => {
              const name = record(value, `context.properNames[${index}]`)
              exact(name, ['text', 'reason'], 'context.properNames')
              return {
                text: string(name.text, 'context.properNames.text', false, 128),
                reason: oneOf(name.reason, 'context.properNames.reason', [
                  'person-name',
                  'place-name',
                  'title',
                  'unavoidable-proper-noun',
                ] as const),
              }
            },
          )
          return {
            displayedChinese: string(
              ctx.displayedChinese,
              'context.displayedChinese',
              true,
              MAX_DOCUMENT_BLOCK_UTF8_BYTES,
            ),
            baseChinese: string(
              ctx.baseChinese,
              'context.baseChinese',
              true,
              MAX_DOCUMENT_BLOCK_UTF8_BYTES,
            ),
            sourceText: string(
              ctx.sourceText,
              'context.sourceText',
              true,
              MAX_DOCUMENT_BLOCK_UTF8_BYTES,
            ),
            properNames,
          }
        })()
  if (interaction === 'selection') {
    exact(item, ['interaction', 'selectedText', 'itemId', 'context'], '$')
    if ((itemId === undefined) !== (context === undefined))
      fail('context', 'itemId and context must be paired')
    return {
      interaction,
      selectedText: string(item.selectedText, 'selectedText', false, 256),
      ...(itemId ? { itemId } : {}),
      ...(context ? { context } : {}),
    }
  }
  exact(item, ['interaction', 'characterOffset', 'itemId', 'context'], '$')
  if (!itemId || !context) fail('context', 'hover lookup requires item context')
  const characterOffset = integer(item.characterOffset, 'characterOffset')
  if (characterOffset >= [...context.displayedChinese].length)
    fail('characterOffset', 'must identify a character in the item context')
  return { interaction, characterOffset, itemId, context }
}

export function parseLookupResult(value: unknown): LookupResult {
  const item = record(value, '$')
  exact(item, ['selectedText', 'tokens', 'item'], '$')
  const tokens = array(item.tokens, 'tokens', 512).map((candidate, index) => {
    const path = `tokens[${index}]`
    const token = record(candidate, path)
    exact(token, ['simplified', 'pinyin', 'definitions', 'hskLevel', 'properName'], path)
    const level = optional(token.hskLevel, `${path}.hskLevel`, hskLevel)
    return {
      simplified: string(token.simplified, `${path}.simplified`),
      pinyin: string(token.pinyin, `${path}.pinyin`, true),
      definitions: stringArray(token.definitions, `${path}.definitions`, 32, 2_048),
      ...(level === undefined ? {} : { hskLevel: level }),
      properName: bool(token.properName, `${path}.properName`),
    }
  })
  const context =
    item.item === undefined
      ? undefined
      : (() => {
          const parsed = record(item.item, 'item')
          exact(parsed, ['displayedChinese', 'baseChinese', 'sourceText'], 'item')
          return {
            displayedChinese: string(
              parsed.displayedChinese,
              'item.displayedChinese',
              true,
              MAX_DOCUMENT_BLOCK_UTF8_BYTES,
            ),
            baseChinese: string(
              parsed.baseChinese,
              'item.baseChinese',
              true,
              MAX_DOCUMENT_BLOCK_UTF8_BYTES,
            ),
            sourceText: string(
              parsed.sourceText,
              'item.sourceText',
              true,
              MAX_DOCUMENT_BLOCK_UTF8_BYTES,
            ),
          }
        })()
  return {
    selectedText: string(item.selectedText, 'selectedText'),
    tokens,
    ...(context === undefined ? {} : { item: context }),
  }
}

export function parseErrorResponse(value: unknown): ErrorResponse {
  const item = record(value, '$')
  exact(item, ['code', 'message', 'retryable'], '$')
  return {
    code: string(item.code, 'code', false, 256),
    message: string(item.message, 'message', false, 2_048),
    retryable: bool(item.retryable, 'retryable'),
  }
}
