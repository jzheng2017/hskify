import { createHash } from 'node:crypto'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { extname, isAbsolute, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))

export const DEFAULT_MANIFEST_PATH = resolve(
  REPOSITORY_ROOT,
  'fixtures/real-reader-corpus/manifest.json',
)
export const DEFAULT_CORPUS_ROOT = resolve(REPOSITORY_ROOT, 'local-corpus/real-reader-v2')

export const REAL_READER_SCHEMA_VERSION = 2
export const REAL_READER_CORPUS_ID = 'real-reader-v2'

export const CORE_CHAPTER_IDS = [
  'webtoon-batman-wayne-family-adventures-1',
  'webtoon-lore-olympus-1',
  'webtoon-cursed-princess-club-1',
  'webtoon-school-bus-graveyard-1',
  'webtoon-omniscient-reader-2',
  'webtoon-how-to-be-a-mind-reaver-363',
  'tapas-free2play-1',
  'manga-plus-spy-family-1001834',
  'globalcomix-bloodshot-1',
  'asura-return-of-the-unrivaled-spear-knight-208',
]

export const STRESS_CHAPTER_IDS = [
  'asura-only-i-have-an-ex-grade-summon-33',
  'asura-absolute-regression-110',
  'asura-the-demon-god-40',
]

const SUPPORTED_PROVIDERS = new Set([
  'webtoon',
  'tapas',
  'asura',
  'manga-plus',
  'globalcomix',
])
const READER_KINDS = new Set([
  'continuous-image',
  'paged-image',
  'iframe-image',
  'background',
  'canvas',
  'webgl',
])
export const REQUIRED_READER_KINDS = [...READER_KINDS]
const REGION_ROLES = new Set([
  'dialogue',
  'narration',
  'system',
  'sfx',
  'artwork',
  'furniture',
])
const ENTITY_TYPES = new Set([
  'person',
  'place',
  'organization',
  'coined',
  'relationship',
  'occupation',
  'rank',
  'title',
])

const SHA256 = /^[a-f0-9]{64}$/
const ID = /^[a-z0-9]+(?:-[a-z0-9]+)*$/
const MIME_BY_EXTENSION = new Map([
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.png', 'image/png'],
  ['.webp', 'image/webp'],
])

function assertion(id, passed, expected, actual, message) {
  return {
    id,
    passed,
    expected,
    actual,
    ...(message ? { message } : {}),
  }
}

function objectRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function safeObjectPath(corpusRoot, objectPath) {
  if (
    typeof objectPath !== 'string' ||
    objectPath.length === 0 ||
    isAbsolute(objectPath) ||
    objectPath.includes('\\')
  ) {
    return undefined
  }
  const resolved = resolve(corpusRoot, objectPath)
  const fromRoot = relative(corpusRoot, resolved)
  if (!fromRoot || fromRoot.startsWith('..') || isAbsolute(fromRoot)) return undefined
  return resolved
}

function safeAnnotationPath(manifestPath, annotationPath) {
  if (
    typeof annotationPath !== 'string' ||
    annotationPath.length === 0 ||
    isAbsolute(annotationPath) ||
    annotationPath.includes('\\')
  ) {
    return undefined
  }
  const annotationRoot = resolve(manifestPath, '..')
  const resolved = resolve(annotationRoot, annotationPath)
  const fromRoot = relative(annotationRoot, resolved)
  if (!fromRoot || fromRoot.startsWith('..') || isAbsolute(fromRoot)) return undefined
  return resolved
}

function readUint24LE(bytes, offset) {
  return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16)
}

function pngDimensions(bytes) {
  if (
    bytes.length < 24 ||
    bytes.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a' ||
    bytes.subarray(12, 16).toString('ascii') !== 'IHDR'
  ) {
    return undefined
  }
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) }
}

function jpegDimensions(bytes) {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return undefined
  let offset = 2
  while (offset + 3 < bytes.length) {
    while (offset < bytes.length && bytes[offset] !== 0xff) offset += 1
    while (offset < bytes.length && bytes[offset] === 0xff) offset += 1
    if (offset >= bytes.length) break
    const marker = bytes[offset]
    offset += 1
    if (marker === 0xd8 || marker === 0xd9 || marker === 0x01) continue
    if (offset + 1 >= bytes.length) break
    const length = bytes.readUInt16BE(offset)
    if (length < 2 || offset + length > bytes.length) break
    if (
      (marker >= 0xc0 && marker <= 0xc3) ||
      (marker >= 0xc5 && marker <= 0xc7) ||
      (marker >= 0xc9 && marker <= 0xcb) ||
      (marker >= 0xcd && marker <= 0xcf)
    ) {
      if (length < 7) break
      return {
        width: bytes.readUInt16BE(offset + 5),
        height: bytes.readUInt16BE(offset + 3),
      }
    }
    offset += length
  }
  return undefined
}

function webpDimensions(bytes) {
  if (
    bytes.length < 30 ||
    bytes.subarray(0, 4).toString('ascii') !== 'RIFF' ||
    bytes.subarray(8, 12).toString('ascii') !== 'WEBP'
  ) {
    return undefined
  }
  let offset = 12
  while (offset + 8 <= bytes.length) {
    const type = bytes.subarray(offset, offset + 4).toString('ascii')
    const length = bytes.readUInt32LE(offset + 4)
    const payload = offset + 8
    if (payload + length > bytes.length) break
    if (type === 'VP8X' && length >= 10) {
      return {
        width: readUint24LE(bytes, payload + 4) + 1,
        height: readUint24LE(bytes, payload + 7) + 1,
      }
    }
    if (type === 'VP8L' && length >= 5 && bytes[payload] === 0x2f) {
      const b1 = bytes[payload + 1]
      const b2 = bytes[payload + 2]
      const b3 = bytes[payload + 3]
      const b4 = bytes[payload + 4]
      return {
        width: 1 + b1 + ((b2 & 0x3f) << 8),
        height: 1 + (b2 >> 6) + (b3 << 2) + ((b4 & 0x0f) << 10),
      }
    }
    if (
      type === 'VP8 ' &&
      length >= 10 &&
      bytes[payload + 3] === 0x9d &&
      bytes[payload + 4] === 0x01 &&
      bytes[payload + 5] === 0x2a
    ) {
      return {
        width: bytes.readUInt16LE(payload + 6) & 0x3fff,
        height: bytes.readUInt16LE(payload + 8) & 0x3fff,
      }
    }
    offset = payload + length + (length % 2)
  }
  return undefined
}

export function imageDimensions(bytes, mimeType) {
  if (mimeType === 'image/png') return pngDimensions(bytes)
  if (mimeType === 'image/jpeg') return jpegDimensions(bytes)
  if (mimeType === 'image/webp') return webpDimensions(bytes)
  return undefined
}

function normalizedPoint(value) {
  return (
    objectRecord(value) &&
    Number.isFinite(value.x) &&
    Number.isFinite(value.y) &&
    value.x >= 0 &&
    value.x <= 1 &&
    value.y >= 0 &&
    value.y <= 1
  )
}

function normalizedPolygon(value, { minimum = 3 } = {}) {
  return (
    Array.isArray(value) &&
    value.length >= minimum &&
    value.length <= 256 &&
    value.every(normalizedPoint)
  )
}

function normalizedRect(value) {
  return (
    objectRecord(value) &&
    [value.x, value.y, value.width, value.height].every(Number.isFinite) &&
    value.x >= 0 &&
    value.y >= 0 &&
    value.width > 0 &&
    value.height > 0 &&
    value.x + value.width <= 1 &&
    value.y + value.height <= 1
  )
}

function canonicalAnnotationPath(chapterId, pageOrder, path) {
  const paddedOrder = String(pageOrder + 1).padStart(4, '0')
  return path === `annotations/${chapterId}/${paddedOrder}.json`
}

function canonicalObjectPath(object) {
  if (!objectRecord(object) || typeof object.path !== 'string') return false
  const extension = extname(object.path).toLowerCase()
  return (
    typeof object.sha256 === 'string' &&
    SHA256.test(object.sha256) &&
    object.path === `objects/${object.sha256}${extension}` &&
    MIME_BY_EXTENSION.has(extension)
  )
}

function validStringList(value, { allowEmpty = true } = {}) {
  return (
    Array.isArray(value) &&
    (allowEmpty || value.length > 0) &&
    value.every((entry) => typeof entry === 'string' && entry.trim().length > 0)
  )
}

function validReviewedAlternatives(value) {
  if (!objectRecord(value)) return false
  const modes = ['natural', 'strict']
  return modes.every(
    (mode) =>
      Array.isArray(value[mode]) &&
      value[mode].length > 0 &&
      value[mode].every(
        (entry) =>
          objectRecord(entry) &&
          typeof entry.chinese === 'string' &&
          entry.chinese.trim().length > 0 &&
          (/[\u3400-\u9fff]/u.test(entry.chinese) || entry.preserveOriginal === true) &&
          validStringList(entry.teachingTerms ?? [], { allowEmpty: true }),
      ),
  )
}

function validStyleRuns(value, maximumLength = Number.POSITIVE_INFINITY) {
  if (!Array.isArray(value) || value.length === 0) return false
  return value.every(
    (run) =>
      objectRecord(run) &&
      Number.isSafeInteger(run.start) &&
      Number.isSafeInteger(run.end) &&
      run.start >= 0 &&
      run.end > run.start &&
      run.end <= maximumLength &&
      typeof run.fontCategory === 'string' &&
      run.fontCategory.trim().length > 0 &&
      (run.orientation === undefined ||
        ['horizontal', 'vertical', 'rotated'].includes(run.orientation)) &&
      (run.color === undefined ||
        (typeof run.color === 'string' && /^#[0-9a-f]{6}$/iu.test(run.color)))
  )
}

function validEntitySpans(value, maximumLength = Number.POSITIVE_INFINITY) {
  if (!Array.isArray(value)) return false
  return value.every(
    (entity) =>
      objectRecord(entity) &&
      Number.isSafeInteger(entity.start) &&
      Number.isSafeInteger(entity.end) &&
      entity.start >= 0 &&
      entity.end > entity.start &&
      entity.end <= maximumLength &&
      ENTITY_TYPES.has(entity.type) &&
      (entity.source === undefined ||
        (typeof entity.source === 'string' && entity.source.trim().length > 0)),
  )
}

function validRegionAnnotation(region) {
  if (!objectRecord(region)) return false
  if (typeof region.sourceEnglish !== 'string') return false
  const sourceLength = [...region.sourceEnglish].length
  if (
    typeof region.id !== 'string' ||
    !ID.test(region.id) ||
    !REGION_ROLES.has(region.role) ||
    !normalizedPolygon(region.polygon) ||
    !Number.isSafeInteger(region.readingOrder) ||
    region.readingOrder < 0 ||
    typeof region.sourceEnglish !== 'string' ||
    region.sourceEnglish.trim().length === 0 ||
    !validEntitySpans(region.entities ?? [], sourceLength) ||
    !validStyleRuns(region.styleRuns ?? [], sourceLength) ||
    !validReviewedAlternatives(region.reviewedTranslations) ||
    !Object.hasOwn(region, 'cleanupAllowance')
  ) {
    return false
  }
  if (
    region.continuationGroup !== undefined &&
    region.continuationGroup !== null &&
    (typeof region.continuationGroup !== 'string' || !ID.test(region.continuationGroup))
  ) {
    return false
  }
  if (typeof region.protectedArtwork !== 'boolean') return false
  if (
    region.cleanupAllowance !== null &&
    region.cleanupAllowance !== undefined &&
    !normalizedPolygon(region.cleanupAllowance)
  ) {
    return false
  }
  return true
}

function validPageAnnotation(annotation, chapterId, pageOrder, sourceSha256) {
  if (!objectRecord(annotation)) return false
  if (
    annotation.schemaVersion !== REAL_READER_SCHEMA_VERSION ||
    annotation.chapterId !== chapterId ||
    annotation.pageOrder !== pageOrder ||
    annotation.sourceSha256 !== sourceSha256 ||
    !Array.isArray(annotation.regions) ||
    !Array.isArray(annotation.exclusions) ||
    annotation.regions.length + annotation.exclusions.length === 0
  ) {
    return false
  }
  const regionIds = new Set()
  const validRegions = annotation.regions.every((region) => {
    if (regionIds.has(region?.id)) return false
    if (typeof region?.id === 'string') regionIds.add(region.id)
    return validRegionAnnotation(region)
  })
  const validExclusions = annotation.exclusions.every(
    (exclusion) =>
      objectRecord(exclusion) &&
      typeof exclusion.id === 'string' &&
      ID.test(exclusion.id) &&
      !regionIds.has(exclusion.id) &&
      normalizedPolygon(exclusion.polygon) &&
      typeof exclusion.sourceEnglish === 'string' &&
      exclusion.sourceEnglish.trim().length > 0 &&
      typeof exclusion.reason === 'string' &&
      exclusion.reason.trim().length > 0,
  )
  if (!validRegions || !validExclusions) return false
  const readingOrders = annotation.regions.map((region) => region.readingOrder)
  return new Set(readingOrders).size === readingOrders.length
}

export function validateManifest(manifest) {
  const checks = [
    assertion(
      'manifest.schema-version',
      objectRecord(manifest) && manifest.schemaVersion === REAL_READER_SCHEMA_VERSION,
      REAL_READER_SCHEMA_VERSION,
      objectRecord(manifest) ? manifest.schemaVersion : typeof manifest,
      'The release corpus must use the complete v2 chapter contract.',
    ),
  ]
  if (!objectRecord(manifest)) return checks

  checks.push(
    assertion(
      'manifest.corpus-id',
      manifest.corpusId === REAL_READER_CORPUS_ID,
      REAL_READER_CORPUS_ID,
      manifest.corpusId,
    ),
    assertion(
      'manifest.offline-only',
      manifest.execution?.networkPolicy === 'forbidden',
      'forbidden',
      manifest.execution?.networkPolicy,
    ),
    assertion(
      'manifest.default-corpus-root',
      manifest.execution?.defaultCorpusRoot === 'local-corpus/real-reader-v2',
      'local-corpus/real-reader-v2',
      manifest.execution?.defaultCorpusRoot,
    ),
    assertion(
      'manifest.capture-complete',
      manifest.completeness?.state === 'complete',
      'complete',
      manifest.completeness?.state,
      'A release run requires every page object and every annotation to be captured locally.',
    ),
  )

  const chapters = Array.isArray(manifest.chapters) ? manifest.chapters : []
  checks.push(assertion('manifest.chapter-count', chapters.length > 0, '> 0', chapters.length))
  const totalPages = chapters.reduce(
    (sum, chapter) => sum + (Array.isArray(chapter?.pages) ? chapter.pages.length : 0),
    0,
  )
  checks.push(
    assertion(
      'manifest.completeness-totals',
      objectRecord(manifest.completeness) &&
        manifest.completeness.chapterCount === chapters.length &&
        manifest.completeness.pageCount === totalPages &&
        manifest.completeness.annotationCount === totalPages,
      { chapterCount: chapters.length, pageCount: totalPages, annotationCount: totalPages },
      manifest.completeness,
      'Completeness totals are checked against the manifest graph before local bytes are opened.',
    ),
  )
  const chapterIds = new Set()
  for (const [index, chapter] of chapters.entries()) {
    const prefix = `chapter.${index + 1}`
    const validRecord = objectRecord(chapter)
    checks.push(assertion(`${prefix}.record`, validRecord, 'object', typeof chapter))
    if (!validRecord) continue
    const chapterIdValid =
      typeof chapter.id === 'string' && ID.test(chapter.id) && !chapterIds.has(chapter.id)
    checks.push(assertion(`${prefix}.id`, chapterIdValid, 'unique lowercase kebab-case identifier', chapter.id))
    if (typeof chapter.id === 'string') chapterIds.add(chapter.id)
    checks.push(
      assertion(
        `${prefix}.provider`,
        SUPPORTED_PROVIDERS.has(chapter.provenance?.provider),
        [...SUPPORTED_PROVIDERS].sort(),
        chapter.provenance?.provider,
      ),
      assertion(
        `${prefix}.provenance`,
        typeof chapter.provenance?.chapterUrl === 'string' &&
          /^https:\/\//u.test(chapter.provenance.chapterUrl) &&
          typeof chapter.provenance?.capturedAtUtc === 'string' &&
          chapter.provenance.capturedAtUtc.trim().length > 0,
        'HTTPS discovery URL and capture timestamp',
        chapter.provenance,
      ),
      assertion(
        `${prefix}.reader-kind`,
        READER_KINDS.has(chapter.reader?.kind),
        [...READER_KINDS].sort(),
        chapter.reader?.kind,
      ),
    )
    const pages = Array.isArray(chapter.pages) ? chapter.pages : []
    checks.push(
      assertion(
        `${prefix}.page-count`,
        Number.isSafeInteger(chapter.pageCount) && chapter.pageCount > 0 && pages.length === chapter.pageCount,
        'positive pageCount equal to pages.length',
        { pageCount: chapter.pageCount, pages: pages.length },
      ),
    )
    const pageOrders = pages.map((page) => page?.order)
    checks.push(
      assertion(
        `${prefix}.page-order`,
        pageOrders.every((order, pageIndex) => order === pageIndex),
        'contiguous zero-based canonical page order',
        pageOrders,
      ),
    )
    for (const [pageIndex, page] of pages.entries()) {
      const pagePrefix = `${prefix}.page.${pageIndex + 1}`
      const object = page?.object
      const extension = typeof object?.path === 'string' ? extname(object.path).toLowerCase() : ''
      const annotation = page?.annotation
      checks.push(
        assertion(`${pagePrefix}.record`, objectRecord(page), 'object', typeof page),
        assertion(`${pagePrefix}.order`, page?.order === pageIndex, pageIndex, page?.order),
        assertion(`${pagePrefix}.object-path`, canonicalObjectPath(object), 'objects/<sha256>.<extension>', object?.path),
        assertion(`${pagePrefix}.mime-type`, MIME_BY_EXTENSION.get(extension) === object?.mimeType, MIME_BY_EXTENSION.get(extension), object?.mimeType),
        assertion(`${pagePrefix}.byte-length`, Number.isSafeInteger(object?.bytes) && object.bytes > 0, 'positive integer', object?.bytes),
        assertion(
          `${pagePrefix}.dimensions`,
          Number.isSafeInteger(object?.width) && object.width > 0 && Number.isSafeInteger(object?.height) && object.height > 0,
          'positive integer width and height',
          { width: object?.width, height: object?.height },
        ),
        assertion(
          `${pagePrefix}.annotation-reference`,
          objectRecord(annotation) &&
            canonicalAnnotationPath(chapter.id, pageIndex, annotation.path) &&
            typeof annotation.sha256 === 'string' &&
            SHA256.test(annotation.sha256) &&
            Number.isSafeInteger(annotation.bytes) &&
            annotation.bytes > 0,
          'content-addressed annotation path, hash, and byte length',
          annotation,
        ),
        assertion(
          `${pagePrefix}.expectations`,
          page.expectations === undefined ||
            (objectRecord(page.expectations) &&
              (page.expectations.hskDifferential === undefined ||
                typeof page.expectations.hskDifferential === 'boolean')),
          'optional page expectations with boolean hskDifferential marker',
          page.expectations,
        ),
      )
    }
    const pageCoverage = chapter.coverage
    checks.push(
      assertion(
        `${prefix}.annotation-coverage`,
        objectRecord(pageCoverage) &&
          pageCoverage.annotatedPageCount === chapter.pageCount &&
          Number.isSafeInteger(pageCoverage.storyTargetCount) &&
          pageCoverage.storyTargetCount >= 0 &&
          Number.isSafeInteger(pageCoverage.exclusionCount) &&
          pageCoverage.exclusionCount >= 0 &&
          pageCoverage.storyTargetCount + pageCoverage.exclusionCount > 0,
        'every page annotated with at least one target or exclusion and reviewed totals',
        pageCoverage,
      ),
    )
  }

  const requirements = manifest.coverageRequirements
  checks.push(
    assertion(
      'coverage.requirements',
      objectRecord(requirements) &&
        Array.isArray(requirements.coreChapterIds) &&
        Array.isArray(requirements.stressChapterIds) &&
        Array.isArray(requirements.readerKinds) &&
        requirements.coreChapterIds.length === CORE_CHAPTER_IDS.length &&
        requirements.stressChapterIds.length === STRESS_CHAPTER_IDS.length &&
        requirements.readerKinds.length === REQUIRED_READER_KINDS.length &&
        CORE_CHAPTER_IDS.every((id) => requirements.coreChapterIds.includes(id)) &&
        STRESS_CHAPTER_IDS.every((id) => requirements.stressChapterIds.includes(id)) &&
        REQUIRED_READER_KINDS.every((kind) => requirements.readerKinds.includes(kind)),
      {
        coreChapterIds: CORE_CHAPTER_IDS,
        stressChapterIds: STRESS_CHAPTER_IDS,
        readerKinds: REQUIRED_READER_KINDS,
      },
      requirements,
    ),
    assertion(
      'coverage.chapter-set',
      CORE_CHAPTER_IDS.every((id) => chapterIds.has(id)) && STRESS_CHAPTER_IDS.every((id) => chapterIds.has(id)),
      [...CORE_CHAPTER_IDS, ...STRESS_CHAPTER_IDS],
      [...chapterIds].sort(),
      'The release corpus must contain the complete core and stress chapter set, not sampled pages.',
    ),
  )

  const selections = objectRecord(manifest.selections) ? manifest.selections : {}
  for (const [name, selectedIds] of Object.entries(selections)) {
    checks.push(
      assertion(
        `selection.${name}`,
        ID.test(name) &&
          Array.isArray(selectedIds) &&
          selectedIds.length > 0 &&
          new Set(selectedIds).size === selectedIds.length &&
          selectedIds.every((id) => chapterIds.has(id)),
        'ordered unique known chapter IDs',
        selectedIds,
      ),
    )
  }
  const requiredSelections = {
    core: CORE_CHAPTER_IDS,
    stress: STRESS_CHAPTER_IDS,
    all: [...CORE_CHAPTER_IDS, ...STRESS_CHAPTER_IDS],
  }
  for (const [name, expectedIds] of Object.entries(requiredSelections)) {
    const selectedIds = selections[name]
    checks.push(
      assertion(
        `selection.${name}.complete`,
        Array.isArray(selectedIds) &&
          selectedIds.length === expectedIds.length &&
          selectedIds.every((id, index) => id === expectedIds[index]),
        expectedIds,
        selectedIds,
        'Core, stress, and all selections are canonical chapter order.',
      ),
    )
  }
  return checks
}

export function selectedCases(manifest, selection = 'all') {
  const chapters = Array.isArray(manifest.chapters) ? manifest.chapters : []
  if (selection === 'all') {
    return chapters.flatMap((chapter) =>
      (chapter.pages ?? []).map((page) => pageCase(chapter, page)),
    )
  }
  const ids = manifest.selections?.[selection]
  if (!Array.isArray(ids)) throw new Error(`Unknown corpus selection: ${selection}`)
  const byId = new Map(chapters.map((chapter) => [chapter.id, chapter]))
  return ids.flatMap((id) => {
    const chapter = byId.get(id)
    return chapter ? (chapter.pages ?? []).map((page) => pageCase(chapter, page)) : []
  })
}

function pageCase(chapter, page) {
  return {
    id: `${chapter.id}-page-${String(page.order + 1).padStart(4, '0')}`,
    chapterId: chapter.id,
    object: page.object,
    annotation: page.annotation,
    provenance: {
      ...chapter.provenance,
      pageIndex: page.order + 1,
    },
    qualityFocus: chapter.qualityFocus ?? [],
    expectations: page.expectations ?? {},
    reader: chapter.reader,
  }
}

export function auditCorpus({
  manifestPath = DEFAULT_MANIFEST_PATH,
  corpusRoot = DEFAULT_CORPUS_ROOT,
  selection = 'all',
  manifestOnly = false,
} = {}) {
  const absoluteManifest = resolve(manifestPath)
  const absoluteCorpus = resolve(corpusRoot)
  const manifest = JSON.parse(readFileSync(absoluteManifest, 'utf8'))
  const assertions = validateManifest(manifest)
  const manifestPassed = assertions.every((item) => item.passed)
  const selectionKnown =
    selection === 'all' ||
    (typeof selection === 'string' &&
      selection.length > 0 &&
      Array.isArray(manifest.selections?.[selection]))
  if (!selectionKnown) {
    assertions.push(
      assertion(
        'selection.requested',
        false,
        'all, core, stress, or a declared selection',
        selection,
      ),
    )
  }
  const cases = manifestPassed && selectionKnown ? selectedCases(manifest, selection) : []
  const assets = []
  const annotations = []
  const recomputedCoverage = new Map()

  if (!manifestOnly && manifestPassed) {
    for (const item of cases) {
      const path = safeObjectPath(absoluteCorpus, item.object.path)
      if (!path) {
        assertions.push(
          assertion(`asset.${item.id}.path`, false, 'path contained by corpus root', item.object.path),
        )
        assets.push({ id: item.id, state: 'invalid-path', path: item.object.path })
        continue
      }
      if (!existsSync(path)) {
        assertions.push(assertion(`asset.${item.id}.present`, false, true, false))
        assets.push({ id: item.id, state: 'missing', path })
      } else {
        const bytes = readFileSync(path)
        const byteLength = statSync(path).size
        const sha256 = createHash('sha256').update(bytes).digest('hex')
        const dimensions = imageDimensions(bytes, item.object.mimeType)
        const byteMatch = byteLength === item.object.bytes
        const hashMatch = sha256 === item.object.sha256
        const dimensionMatch =
          dimensions?.width === item.object.width && dimensions?.height === item.object.height
        assertions.push(
          assertion(`asset.${item.id}.present`, true, true, true),
          assertion(`asset.${item.id}.bytes`, byteMatch, item.object.bytes, byteLength),
          assertion(`asset.${item.id}.sha256`, hashMatch, item.object.sha256, sha256),
          assertion(
            `asset.${item.id}.dimensions`,
            dimensionMatch,
            { width: item.object.width, height: item.object.height },
            dimensions,
          ),
        )
        assets.push({
          id: item.id,
          state: byteMatch && hashMatch && dimensionMatch ? 'verified' : 'mismatch',
          path,
          sha256,
          bytes: byteLength,
          dimensions,
          qualityFocus: item.qualityFocus,
        })
      }

      const annotationPath = safeAnnotationPath(absoluteManifest, item.annotation?.path)
      if (!annotationPath) {
        assertions.push(
          assertion(
            `annotation.${item.id}.path`,
            false,
            'path contained by manifest root',
            item.annotation?.path,
          ),
        )
        annotations.push({ id: item.id, state: 'invalid-path', path: item.annotation?.path })
        continue
      }
      if (!existsSync(annotationPath)) {
        assertions.push(assertion(`annotation.${item.id}.present`, false, true, false))
        annotations.push({ id: item.id, state: 'missing', path: annotationPath })
        continue
      }
      const annotationBytes = readFileSync(annotationPath)
      const annotationHash = createHash('sha256').update(annotationBytes).digest('hex')
      const annotationLength = statSync(annotationPath).size
      let annotationValue
      try {
        annotationValue = JSON.parse(annotationBytes.toString('utf8'))
      } catch {
        annotationValue = undefined
      }
      const hashMatch = annotationHash === item.annotation.sha256
      const byteMatch = annotationLength === item.annotation.bytes
      const shapeMatch = validPageAnnotation(
        annotationValue,
        item.chapterId,
        item.provenance.pageIndex - 1,
        item.object.sha256,
      )
      assertions.push(
        assertion(`annotation.${item.id}.present`, true, true, true),
        assertion(`annotation.${item.id}.bytes`, byteMatch, item.annotation.bytes, annotationLength),
        assertion(`annotation.${item.id}.sha256`, hashMatch, item.annotation.sha256, annotationHash),
        assertion(
          `annotation.${item.id}.shape`,
          shapeMatch,
          'page annotation schema 2 with exhaustive regions/exclusions',
          shapeMatch,
        ),
      )
      if (shapeMatch) {
        const current = recomputedCoverage.get(item.chapterId) ?? {
          storyTargetCount: 0,
          exclusionCount: 0,
        }
        current.storyTargetCount += annotationValue.regions.length
        current.exclusionCount += annotationValue.exclusions.length
        recomputedCoverage.set(item.chapterId, current)
      }
      annotations.push({
        id: item.id,
        state: byteMatch && hashMatch && shapeMatch ? 'verified' : 'mismatch',
        path: annotationPath,
        sha256: annotationHash,
        bytes: annotationLength,
      })
    }
    const selectedChapterIds = new Set(cases.map((item) => item.chapterId))
    for (const chapterId of selectedChapterIds) {
      const chapter = (manifest.chapters ?? []).find((item) => item.id === chapterId)
      const actual = recomputedCoverage.get(chapterId) ?? {
        storyTargetCount: 0,
        exclusionCount: 0,
      }
      assertions.push(
        assertion(
          `coverage.${chapterId}.recomputed`,
          actual.storyTargetCount === chapter?.coverage?.storyTargetCount &&
            actual.exclusionCount === chapter?.coverage?.exclusionCount,
          {
            storyTargetCount: chapter?.coverage?.storyTargetCount,
            exclusionCount: chapter?.coverage?.exclusionCount,
          },
          actual,
          'Chapter coverage totals must be recomputed from every verified annotation, not trusted as declarations.',
        ),
      )
    }
  }

  const failures = assertions.filter((item) => !item.passed)
  const missing = assets.filter((item) => item.state === 'missing')
  const missingAnnotations = annotations.filter((item) => item.state === 'missing')
  const captureRequired = manifest?.completeness?.state !== 'complete'
  const captureRequiredPaths = captureRequired
    ? [
        resolve(absoluteCorpus, 'objects'),
        ...((manifest.selections?.[selection] ?? manifest.chapters ?? [])
          .map((item) => (typeof item === 'string' ? item : item?.id))
          .filter((id) => typeof id === 'string' && id.length > 0)
          .map((id) => resolve(absoluteManifest, '..', 'annotations', id))),
      ]
    : []
  return {
    schemaVersion: REAL_READER_SCHEMA_VERSION,
    corpusId: manifest.corpusId,
    selection,
    offline: true,
    manifestPath: absoluteManifest,
    corpusRoot: absoluteCorpus,
    caseCount: cases.length,
    chapterCount: Array.isArray(manifest.chapters) ? manifest.chapters.length : 0,
    verifiedCount: assets.filter((item) => item.state === 'verified').length,
    missingCount: missing.length,
    verifiedAnnotationCount: annotations.filter((item) => item.state === 'verified').length,
    missingAnnotationCount: missingAnnotations.length,
    captureRequired,
    status: failures.length === 0 ? 'passed' : 'failed',
    assertions,
    assets,
    annotations,
    failures,
    remediation: captureRequired
      ? {
          message:
            'The real-reader-v2 corpus is not complete. Restore every content-addressed page object and tracked annotation; the runner will not download live URLs.',
          requiredPaths: [
            ...captureRequiredPaths,
            ...missing.map((item) => item.path),
            ...missingAnnotations.map((item) => item.path),
          ],
        }
      : missing.length > 0 || missingAnnotations.length > 0
        ? {
            message:
              'Local real-reader-v2 objects or annotations are missing. Restore the exact content-addressed files; the runner will not download live URLs.',
            requiredPaths: [
              ...missing.map((item) => item.path),
              ...missingAnnotations.map((item) => item.path),
            ],
          }
        : undefined,
  }
}

function parseArguments(argv) {
  const options = {
    command: 'verify',
    manifestPath: DEFAULT_MANIFEST_PATH,
    corpusRoot: DEFAULT_CORPUS_ROOT,
    selection: 'all',
    json: false,
  }
  const args = [...argv]
  if (args[0] && !args[0].startsWith('--')) options.command = args.shift()
  while (args.length > 0) {
    const argument = args.shift()
    if (argument === '--json') options.json = true
    else if (argument === '--manifest') options.manifestPath = resolve(args.shift() ?? '')
    else if (argument === '--corpus') options.corpusRoot = resolve(args.shift() ?? '')
    else if (argument === '--selection') options.selection = args.shift() ?? ''
    else throw new Error(`Unknown argument: ${argument}`)
  }
  if (!['verify', 'manifest'].includes(options.command)) {
    throw new Error(`Unknown command: ${options.command}`)
  }
  return options
}

function printResult(result, json) {
  if (json) {
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
    return
  }
  process.stdout.write(
    `${result.status.toUpperCase()}: ${result.verifiedCount}/${result.caseCount} real-reader-v2 page objects and ${result.verifiedAnnotationCount ?? 0}/${result.caseCount} annotations verified (${result.selection}).\n`,
  )
  for (const failure of result.failures) {
    process.stdout.write(
      `- ${failure.id}: expected ${JSON.stringify(failure.expected)}, got ${JSON.stringify(failure.actual)}\n`,
    )
  }
  if (result.remediation) {
    process.stdout.write(`${result.remediation.message}\n`)
    for (const path of result.remediation.requiredPaths) process.stdout.write(`  ${path}\n`)
  }
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : undefined
if (invokedPath === fileURLToPath(import.meta.url)) {
  try {
    const options = parseArguments(process.argv.slice(2))
    const result = auditCorpus({
      manifestPath: options.manifestPath,
      corpusRoot: options.corpusRoot,
      selection: options.selection,
      manifestOnly: options.command === 'manifest',
    })
    printResult(result, options.json)
    if (result.status !== 'passed') {
      process.exitCode = result.captureRequired || result.missingCount > 0 ? 2 : 1
    }
  } catch (error) {
    const result = {
      schemaVersion: REAL_READER_SCHEMA_VERSION,
      status: 'error',
      offline: true,
      message: error instanceof Error ? error.message : String(error),
    }
    process.stderr.write(`${JSON.stringify(result, null, 2)}\n`)
    process.exitCode = 1
  }
}
