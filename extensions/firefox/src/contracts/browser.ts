export const BUILD_FINGERPRINT =
  'hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-07-27-r6' as const
export const HSK_STANDARD = '2.0' as const
export const SOURCE_LANGUAGE = 'en' as const
export const TARGET_LANGUAGE = 'zh-CN' as const
export const MAX_PRECEDING_CONTEXT = 6
export const MAX_PROPER_NAME_GLOSSARY = 64

export type HskLevel = 1 | 2 | 3 | 4 | 5 | 6
export type NameTranslation = 'keep-original' | 'chinese'
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

export type BrowserJobRequest = {
  buildFingerprint: typeof BUILD_FINGERPRINT
  clientImageId: string
  sourceSha256: string
  sourceMimeType: string
  naturalWidth: number
  naturalHeight: number
  pageSessionId: string
  pageIndex: number
  visibleRects: NormalizedRect[]
  settings: {
    sourceLanguage: 'en'
    targetLanguage: 'zh-CN'
    hskStandard: '2.0'
    hskLevel: HskLevel
    readingDirection: 'auto' | 'ltr' | 'rtl'
    translateSoundEffects: false
    nameTranslation: NameTranslation
  }
  precedingContext?: Array<{
    sourceEnglish: string
    chinese: string
  }>
  properNameGlossary?: Array<{
    sourceEnglish: string
    chinese: string
  }>
}

export type BrowserJobCreated = {
  buildFingerprint: typeof BUILD_FINGERPRINT
  jobId: string
}

export type ViewportUpdate = {
  visibleRects: NormalizedRect[]
  active: boolean
}

export type RegionHsk = {
  requestedLevel: HskLevel
  strictlyValid: boolean
  aboveLevelTokens: string[]
  repairState: 'not-needed' | 'pending' | 'accepted' | 'rejected'
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
  safePolygon?: Point[]
}

export type BrowserRegion = {
  id: string
  textPolygon: Point[]
  bubblePolygon?: Point[]
  patch: {
    blobId: string
    mimeType: 'image/png'
    rect: NormalizedRect
  }
  sourceEnglish: string
  baseChinese: string
  displayedChinese: string
  pinyin: string
  ocrConfidence: number
  readingOrder: number
  style: RegionStyle
  layout: RegionLayout
  hsk: RegionHsk
}

export type BrowserJobStage =
  | 'queued'
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
  stage: BrowserJobStage
  stageProgress?: number
  overallProgress?: number
  current?: number
  total?: number
  message: string
}

export type RegionReadyJobUpdate = {
  sequence: number
  type: 'regionReady'
  region: BrowserRegion
}

export type RegionRefinedJobUpdate = {
  sequence: number
  type: 'regionRefined'
  regionId: string
  displayedChinese: string
  pinyin: string
  hsk: RegionHsk
}

export type CompleteJobUpdate = {
  sequence: number
  type: 'complete'
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
  | RegionReadyJobUpdate
  | RegionRefinedJobUpdate
  | CompleteJobUpdate
  | FailedJobUpdate
  | CancelledJobUpdate

export type JobUpdateBatch = {
  jobId: string
  nextSequence: number
  updates: JobUpdate[]
}

export type BrowserSetupStatus = {
  state: 'missing-models' | 'downloading' | 'verifying' | 'ready' | 'failed'
  modelId: string
  currentFile?: string
  completedBytes?: number
  totalBytes?: number
  requiredDiskBytes?: number
  message: string
  errorCode?: string
}

export type LookupRequest =
  | {
      interaction: 'selection'
      selectedText: string
      jobId?: string
      regionId?: string
    }
  | {
      interaction: 'hover'
      characterOffset: number
      jobId: string
      regionId: string
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
    baseChinese: string
    sourceEnglish: string
  }
}

export type ErrorResponse = {
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
  'hsk-validating',
  'styling',
  'packaging',
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
  if (typeof value !== 'string' || (!allowEmpty && value.trim() === '') || value.length > maximum) {
    fail(
      path,
      allowEmpty
        ? `must be a string no longer than ${maximum} characters`
        : `must be a non-empty string no longer than ${maximum} characters`,
    )
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
  if (!values.includes(value as never)) fail(path, `must be one of ${values.join(', ')}`)
  return value as T[number]
}

function buildFingerprint(value: unknown, path = 'buildFingerprint'): typeof BUILD_FINGERPRINT {
  if (value !== BUILD_FINGERPRINT) {
    fail(path, `must equal the running extension build ${BUILD_FINGERPRINT}`)
  }
  return BUILD_FINGERPRINT
}

function hskLevel(value: unknown, path: string): HskLevel {
  return oneOf(value, path, [1, 2, 3, 4, 5, 6] as const)
}

function sha256(value: unknown, path: string): string {
  const parsed = string(value, path, false, 64)
  if (!/^[a-f0-9]{64}$/u.test(parsed)) {
    fail(path, 'must be a lowercase 64-character hexadecimal SHA-256')
  }
  return parsed
}

function cssColor(value: unknown, path: string): string {
  const parsed = string(value, path, false, 9)
  if (!/^#(?:[\da-f]{3}|[\da-f]{4}|[\da-f]{6}|[\da-f]{8})$/iu.test(parsed)) {
    fail(path, 'must be a hexadecimal CSS color')
  }
  return parsed
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
  if (parsed.width <= 0 || parsed.height <= 0) {
    fail(path, 'must have positive width and height')
  }
  if (parsed.x + parsed.width > 1 + Number.EPSILON) {
    fail(path, 'must not extend past the image width')
  }
  if (parsed.y + parsed.height > 1 + Number.EPSILON) {
    fail(path, 'must not extend past the image height')
  }
  return parsed
}

function visibleRects(value: unknown, path: string): NormalizedRect[] {
  return array(value, path, 64).map((item, index) => normalizedRect(item, `${path}[${index}]`))
}

function resourceIdentity(value: unknown, path: string): ResourceIdentity {
  const item = record(value, path)
  exact(item, ['id', 'repository', 'repositoryRevision', 'filename', 'bytes', 'sha256'], path)
  const id = string(item.id, `${path}.id`, false, 128)
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/u.test(id)) {
    fail(`${path}.id`, 'must be a lowercase kebab-case identifier')
  }
  const repository = string(item.repository, `${path}.repository`, false, 256)
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(repository)) {
    fail(`${path}.repository`, 'must contain exactly one owner/name repository')
  }
  const repositoryRevision = string(
    item.repositoryRevision,
    `${path}.repositoryRevision`,
    false,
    40,
  )
  if (!/^[0-9a-f]{40}$/u.test(repositoryRevision)) {
    fail(`${path}.repositoryRevision`, 'must be a lowercase 40-character hexadecimal revision')
  }
  const filename = string(item.filename, `${path}.filename`, false, 255)
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(filename)) {
    fail(`${path}.filename`, 'must be a safe ASCII filename')
  }
  const sha256 = string(item.sha256, `${path}.sha256`, false, 64)
  if (!/^[0-9a-f]{64}$/u.test(sha256)) {
    fail(`${path}.sha256`, 'must be a lowercase 64-character hexadecimal SHA-256')
  }
  return {
    id,
    repository,
    repositoryRevision,
    filename,
    bytes: integer(item.bytes, `${path}.bytes`, 1),
    sha256,
  }
}

function resourceIdentities(value: unknown, path: string): ResourceIdentity[] {
  const identities = array(value, path, 256).map((item, index) =>
    resourceIdentity(item, `${path}[${index}]`),
  )
  if (identities.length === 0) fail(path, 'must not be empty')
  for (let index = 1; index < identities.length; index += 1) {
    if (identities[index - 1]!.id >= identities[index]!.id) {
      fail(`${path}[${index}].id`, 'must be unique and sorted in ascending ordinal order')
    }
  }
  return identities
}

function stringArray(
  value: unknown,
  path: string,
  allowEmptyItems = false,
  maximumItems = 10_000,
): string[] {
  return array(value, path, maximumItems).map((item, index) =>
    string(item, `${path}[${index}]`, allowEmptyItems, 4_096),
  )
}

function parseHsk(value: unknown, path: string): RegionHsk {
  const item = record(value, path)
  exact(item, ['requestedLevel', 'strictlyValid', 'aboveLevelTokens', 'repairState'], path)
  const aboveLevelTokens = stringArray(
    item.aboveLevelTokens,
    `${path}.aboveLevelTokens`,
    false,
    512,
  )
  const strictlyValid = boolean(item.strictlyValid, `${path}.strictlyValid`)
  if (strictlyValid && aboveLevelTokens.length > 0) {
    fail(`${path}.strictlyValid`, 'cannot be true when above-level tokens are present')
  }
  return {
    requestedLevel: hskLevel(item.requestedLevel, `${path}.requestedLevel`),
    strictlyValid,
    aboveLevelTokens,
    repairState: oneOf(item.repairState, `${path}.repairState`, [
      'not-needed',
      'pending',
      'accepted',
      'rejected',
    ] as const),
  }
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
  const weight = integer(item.weight, `${path}.weight`, 1)
  if (weight > 1_000) fail(`${path}.weight`, 'must be at most 1000')
  const outlineWidthRatio = finite(item.outlineWidthRatio, `${path}.outlineWidthRatio`)
  if (outlineWidthRatio < 0) fail(`${path}.outlineWidthRatio`, 'must not be negative')
  const lineHeight = finite(item.lineHeight, `${path}.lineHeight`)
  if (lineHeight <= 0) fail(`${path}.lineHeight`, 'must be positive')
  const outlineColor = optional(item.outlineColor, `${path}.outlineColor`, cssColor)
  const shadowColor = optional(item.shadowColor, `${path}.shadowColor`, cssColor)
  const colorBands =
    item.colorBands === undefined
      ? []
      : array(item.colorBands, `${path}.colorBands`, 512).map((value, index) => {
          const bandPath = `${path}.colorBands[${index}]`
          const band = record(value, bandPath)
          exact(band, ['position', 'foreground', 'outlineColor'], bandPath)
          const position = finite(band.position, `${bandPath}.position`)
          if (position < 0 || position > 1) {
            fail(`${bandPath}.position`, 'must be from 0 through 1')
          }
          const outlineColor = optional(
            band.outlineColor,
            `${bandPath}.outlineColor`,
            cssColor,
          )
          return {
            position,
            foreground: cssColor(band.foreground, `${bandPath}.foreground`),
            ...(outlineColor === undefined ? {} : { outlineColor }),
          }
        })
  if (
    colorBands.some(
      (band, index) => index > 0 && band.position <= (colorBands[index - 1]?.position ?? 1),
    )
  ) {
    fail(`${path}.colorBands`, 'positions must be strictly increasing')
  }
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
    ...(colorBands.length === 0 ? {} : { colorBands }),
  }
}

function parseLayout(value: unknown, path: string): RegionLayout {
  const item = record(value, path)
  exact(item, ['suggestedLines', 'fontSizeToImageWidth', 'safePolygon'], path)
  const fontSizeToImageWidth = finite(item.fontSizeToImageWidth, `${path}.fontSizeToImageWidth`)
  if (fontSizeToImageWidth <= 0) {
    fail(`${path}.fontSizeToImageWidth`, 'must be positive')
  }
  const safePolygon = optional(item.safePolygon, `${path}.safePolygon`, polygon)
  return {
    suggestedLines: stringArray(item.suggestedLines, `${path}.suggestedLines`, true, 256),
    fontSizeToImageWidth,
    ...(safePolygon === undefined ? {} : { safePolygon }),
  }
}

function parseRegion(value: unknown, path: string): BrowserRegion {
  const item = record(value, path)
  exact(
    item,
    [
      'id',
      'textPolygon',
      'bubblePolygon',
      'patch',
      'sourceEnglish',
      'baseChinese',
      'displayedChinese',
      'pinyin',
      'ocrConfidence',
      'readingOrder',
      'style',
      'layout',
      'hsk',
    ],
    path,
  )
  const patch = record(item.patch, `${path}.patch`)
  exact(patch, ['blobId', 'mimeType', 'rect'], `${path}.patch`)
  const bubblePolygon = optional(item.bubblePolygon, `${path}.bubblePolygon`, polygon)
  return {
    id: string(item.id, `${path}.id`, false, 512),
    textPolygon: polygon(item.textPolygon, `${path}.textPolygon`),
    ...(bubblePolygon === undefined ? {} : { bubblePolygon }),
    patch: {
      blobId: string(patch.blobId, `${path}.patch.blobId`, false, 512),
      mimeType: oneOf(patch.mimeType, `${path}.patch.mimeType`, ['image/png'] as const),
      rect: normalizedRect(patch.rect, `${path}.patch.rect`),
    },
    sourceEnglish: string(item.sourceEnglish, `${path}.sourceEnglish`, false, 4_096),
    baseChinese: string(item.baseChinese, `${path}.baseChinese`, false, 4_096),
    displayedChinese: string(item.displayedChinese, `${path}.displayedChinese`, false, 4_096),
    pinyin: string(item.pinyin, `${path}.pinyin`, false, 8_192),
    ocrConfidence: unit(item.ocrConfidence, `${path}.ocrConfidence`),
    readingOrder: integer(item.readingOrder, `${path}.readingOrder`),
    style: parseStyle(item.style, `${path}.style`),
    layout: parseLayout(item.layout, `${path}.layout`),
    hsk: parseHsk(item.hsk, `${path}.hsk`),
  }
}

export function parseNativeHandshakeRequest(value: unknown): NativeHandshakeRequest {
  const item = record(value, '$')
  exact(item, ['type', 'buildFingerprint', 'extensionVersion', 'extensionOrigin'], '$')
  const extensionOrigin = string(item.extensionOrigin, 'extensionOrigin')
  if (!extensionOrigin.startsWith('moz-extension://') || extensionOrigin.endsWith('/')) {
    fail('extensionOrigin', 'must be a moz-extension origin without a trailing slash')
  }
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
  const sourceLanguages = stringArray(capabilities.sourceLanguages, 'capabilities.sourceLanguages')
  const targetLanguages = stringArray(capabilities.targetLanguages, 'capabilities.targetLanguages')
  const hskLevels = array(capabilities.hskLevels, 'capabilities.hskLevels', 6).map((level, index) =>
    hskLevel(level, `capabilities.hskLevels[${index}]`),
  )
  if (
    sourceLanguages.length !== 1 ||
    sourceLanguages[0] !== SOURCE_LANGUAGE ||
    targetLanguages.length !== 1 ||
    targetLanguages[0] !== TARGET_LANGUAGE ||
    hskLevels.join(',') !== '1,2,3,4,5,6'
  ) {
    fail('capabilities', 'must advertise exactly the required translation capabilities')
  }
  const port = integer(item.port, 'port', 1)
  if (port > 65_535) fail('port', 'must be at most 65535')
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
      modelsReady: boolean(capabilities.modelsReady, 'capabilities.modelsReady'),
    },
  }
}

export function parseHealthResponse(value: unknown): HealthResponse {
  const item = record(value, '$')
  exact(
    item,
    ['buildFingerprint', 'engineVersion', 'status', 'setupState', 'resourceIdentities'],
    '$',
  )
  return {
    buildFingerprint: buildFingerprint(item.buildFingerprint),
    engineVersion: string(item.engineVersion, 'engineVersion', false, 128),
    status: oneOf(item.status, 'status', ['ready'] as const),
    setupState: oneOf(item.setupState, 'setupState', [
      'missing-models',
      'downloading',
      'verifying',
      'ready',
      'failed',
    ] as const),
    resourceIdentities: resourceIdentities(item.resourceIdentities, 'resourceIdentities'),
  }
}

export function parseBrowserJobRequest(value: unknown): BrowserJobRequest {
  const item = record(value, '$')
  exact(
    item,
    [
      'buildFingerprint',
      'clientImageId',
      'sourceSha256',
      'sourceMimeType',
      'naturalWidth',
      'naturalHeight',
      'pageSessionId',
      'pageIndex',
      'visibleRects',
      'settings',
      'precedingContext',
      'properNameGlossary',
    ],
    '$',
  )
  const settings = record(item.settings, 'settings')
  exact(
    settings,
    [
      'sourceLanguage',
      'targetLanguage',
      'hskStandard',
      'hskLevel',
      'readingDirection',
      'translateSoundEffects',
      'nameTranslation',
    ],
    'settings',
  )
  const precedingContext =
    item.precedingContext === undefined
      ? undefined
      : array(item.precedingContext, 'precedingContext', MAX_PRECEDING_CONTEXT).map(
          (entry, index) => {
            const parsed = record(entry, `precedingContext[${index}]`)
            exact(parsed, ['sourceEnglish', 'chinese'], `precedingContext[${index}]`)
            return {
              sourceEnglish: string(
                parsed.sourceEnglish,
                `precedingContext[${index}].sourceEnglish`,
                false,
                4_096,
              ),
              chinese: string(parsed.chinese, `precedingContext[${index}].chinese`, false, 4_096),
            }
          },
        )
  const properNameGlossary =
    item.properNameGlossary === undefined
      ? undefined
      : array(item.properNameGlossary, 'properNameGlossary', MAX_PROPER_NAME_GLOSSARY).map(
          (entry, index) => {
            const parsed = record(entry, `properNameGlossary[${index}]`)
            exact(parsed, ['sourceEnglish', 'chinese'], `properNameGlossary[${index}]`)
            return {
              sourceEnglish: string(
                parsed.sourceEnglish,
                `properNameGlossary[${index}].sourceEnglish`,
                false,
                256,
              ),
              chinese: string(parsed.chinese, `properNameGlossary[${index}].chinese`, false, 128),
            }
          },
        )
  if (properNameGlossary) {
    const seen = new Set<string>()
    properNameGlossary.forEach((entry, index) => {
      const normalized = entry.sourceEnglish.trim().toLocaleLowerCase('en-US')
      if (seen.has(normalized)) {
        fail(`properNameGlossary[${index}].sourceEnglish`, 'must be unique ignoring ASCII case')
      }
      seen.add(normalized)
    })
  }
  return {
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
    pageIndex: integer(item.pageIndex, 'pageIndex'),
    visibleRects: visibleRects(item.visibleRects, 'visibleRects'),
    settings: {
      sourceLanguage: oneOf(settings.sourceLanguage, 'settings.sourceLanguage', ['en'] as const),
      targetLanguage: oneOf(settings.targetLanguage, 'settings.targetLanguage', ['zh-CN'] as const),
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
      nameTranslation: oneOf(settings.nameTranslation, 'settings.nameTranslation', [
        'keep-original',
        'chinese',
      ] as const),
    },
    ...(precedingContext === undefined ? {} : { precedingContext }),
    ...(properNameGlossary === undefined ? {} : { properNameGlossary }),
  }
}

export function parseBrowserJobCreated(value: unknown): BrowserJobCreated {
  const item = record(value, '$')
  exact(item, ['buildFingerprint', 'jobId'], '$')
  return {
    buildFingerprint: buildFingerprint(item.buildFingerprint),
    jobId: string(item.jobId, 'jobId', false, 512),
  }
}

export function parseViewportUpdate(value: unknown): ViewportUpdate {
  const item = record(value, '$')
  exact(item, ['visibleRects', 'active'], '$')
  return {
    visibleRects: visibleRects(item.visibleRects, 'visibleRects'),
    active: boolean(item.active, 'active'),
  }
}

export function parseJobUpdate(value: unknown, path = '$'): JobUpdate {
  const item = record(value, path)
  const sequence = integer(item.sequence, `${path}.sequence`, 1)
  const type = oneOf(item.type, `${path}.type`, [
    'progress',
    'regionReady',
    'regionRefined',
    'complete',
    'failed',
    'cancelled',
  ] as const)
  switch (type) {
    case 'progress': {
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
      const total = optional(item.total, `${path}.total`, (candidate, itemPath) =>
        integer(candidate, itemPath, 1),
      )
      if ((current === undefined) !== (total === undefined)) {
        fail(`${path}.current`, 'current and total must be present together')
      }
      if (current !== undefined && total !== undefined && current > total) {
        fail(`${path}.current`, 'must not exceed total')
      }
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
    case 'regionReady':
      exact(item, ['sequence', 'type', 'region'], path)
      return {
        sequence,
        type,
        region: parseRegion(item.region, `${path}.region`),
      }
    case 'regionRefined':
      exact(item, ['sequence', 'type', 'regionId', 'displayedChinese', 'pinyin', 'hsk'], path)
      return {
        sequence,
        type,
        regionId: string(item.regionId, `${path}.regionId`, false, 512),
        displayedChinese: string(item.displayedChinese, `${path}.displayedChinese`, false, 4_096),
        pinyin: string(item.pinyin, `${path}.pinyin`, false, 8_192),
        hsk: parseHsk(item.hsk, `${path}.hsk`),
      }
    case 'complete': {
      exact(item, ['sequence', 'type', 'message'], path)
      const message = optional(item.message, `${path}.message`, (candidate, itemPath) =>
        string(candidate, itemPath, false, 2_048),
      )
      return { sequence, type, ...(message === undefined ? {} : { message }) }
    }
    case 'failed':
      exact(item, ['sequence', 'type', 'code', 'message', 'retryable'], path)
      return {
        sequence,
        type,
        code: string(item.code, `${path}.code`, false, 256),
        message: string(item.message, `${path}.message`, false, 2_048),
        retryable: boolean(item.retryable, `${path}.retryable`),
      }
    case 'cancelled': {
      exact(item, ['sequence', 'type', 'message'], path)
      const message = optional(item.message, `${path}.message`, (candidate, itemPath) =>
        string(candidate, itemPath, false, 2_048),
      )
      return { sequence, type, ...(message === undefined ? {} : { message }) }
    }
  }
}

export function parseJobUpdateBatch(value: unknown, after = 0): JobUpdateBatch {
  const item = record(value, '$')
  exact(item, ['jobId', 'nextSequence', 'updates'], '$')
  const nextSequence = integer(item.nextSequence, 'nextSequence')
  if (nextSequence < after) fail('nextSequence', 'must not move backwards')
  const updates = array(item.updates, 'updates', 2_048).map((update, index) =>
    parseJobUpdate(update, `updates[${index}]`),
  )
  let previous = after
  for (const [index, update] of updates.entries()) {
    if (update.sequence <= previous) {
      fail(`updates[${index}].sequence`, 'must increase beyond the requested sequence')
    }
    if (update.sequence > nextSequence) {
      fail(`updates[${index}].sequence`, 'must not exceed nextSequence')
    }
    if (
      (update.type === 'complete' || update.type === 'failed' || update.type === 'cancelled') &&
      index !== updates.length - 1
    ) {
      fail(`updates[${index}].type`, 'terminal updates must be last')
    }
    previous = update.sequence
  }
  if (updates.length > 0 && previous !== nextSequence) {
    fail('nextSequence', 'must equal the final update sequence')
  }
  if (updates.length === 0 && nextSequence !== after) {
    fail('nextSequence', 'must equal the requested sequence when no updates are returned')
  }
  return {
    jobId: string(item.jobId, 'jobId', false, 512),
    nextSequence,
    updates,
  }
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
    'ready',
    'failed',
  ] as const)
  const completedBytes = optional(item.completedBytes, 'completedBytes', integer)
  const totalBytes = optional(item.totalBytes, 'totalBytes', integer)
  if ((completedBytes === undefined) !== (totalBytes === undefined)) {
    fail('completedBytes', 'completed and total bytes must be present together')
  }
  if (completedBytes !== undefined && totalBytes !== undefined && completedBytes > totalBytes) {
    fail('completedBytes', 'must not exceed total bytes')
  }
  const errorCode = optional(item.errorCode, 'errorCode', string)
  if (state === 'failed' && !errorCode) fail('errorCode', 'failed setup requires an error code')
  const modelId = string(item.modelId, 'modelId', false, 128)
  if (modelId !== 'qwen3.5-4b') fail('modelId', 'must identify the mandatory qwen3.5-4b model')
  const currentFile = optional(item.currentFile, 'currentFile', string)
  const requiredDiskBytes = optional(item.requiredDiskBytes, 'requiredDiskBytes', integer)
  return {
    state,
    modelId,
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
  const interaction = oneOf(
    item.interaction,
    'interaction',
    ['selection', 'hover'] as const,
  )
  const jobId = optional(item.jobId, 'jobId', string)
  const regionId = optional(item.regionId, 'regionId', string)
  if ((jobId === undefined) !== (regionId === undefined)) {
    fail('regionId', 'jobId and regionId must be present together')
  }
  if (interaction === 'selection') {
    exact(item, ['interaction', 'selectedText', 'jobId', 'regionId'], '$')
    const selectedText = string(item.selectedText, 'selectedText', false, 256)
    if ([...selectedText].length > 256) {
      fail('selectedText', 'must contain at most 256 characters')
    }
    return {
      interaction,
      selectedText,
      ...(jobId === undefined ? {} : { jobId }),
      ...(regionId === undefined ? {} : { regionId }),
    }
  }
  exact(item, ['interaction', 'characterOffset', 'jobId', 'regionId'], '$')
  if (jobId === undefined || regionId === undefined) {
    fail('jobId', 'hover lookup requires a translated job and region')
  }
  return {
    interaction,
    characterOffset: integer(item.characterOffset, 'characterOffset', 0),
    jobId,
    regionId,
  }
}

export function parseLookupResult(value: unknown): LookupResult {
  const item = record(value, '$')
  exact(item, ['selectedText', 'tokens', 'region'], '$')
  const tokens = array(item.tokens, 'tokens', 512).map((token, index) => {
    const parsed = record(token, `tokens[${index}]`)
    exact(
      parsed,
      ['simplified', 'pinyin', 'definitions', 'hskLevel', 'properName'],
      `tokens[${index}]`,
    )
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
          exact(parsed, ['displayedChinese', 'baseChinese', 'sourceEnglish'], 'region')
          return {
            displayedChinese: string(parsed.displayedChinese, 'region.displayedChinese', true),
            baseChinese: string(parsed.baseChinese, 'region.baseChinese', true),
            sourceEnglish: string(parsed.sourceEnglish, 'region.sourceEnglish', true),
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
  exact(item, ['code', 'message', 'retryable'], '$')
  return {
    code: string(item.code, 'code', false, 256),
    message: string(item.message, 'message', false, 2_048),
    retryable: boolean(item.retryable, 'retryable'),
  }
}
