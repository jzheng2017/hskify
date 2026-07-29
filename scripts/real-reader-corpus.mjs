import { createHash } from 'node:crypto'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { extname, isAbsolute, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))

export const DEFAULT_MANIFEST_PATH = resolve(
  REPOSITORY_ROOT,
  'fixtures/real-reader-corpus/manifest.json',
)
export const DEFAULT_CORPUS_ROOT = resolve(REPOSITORY_ROOT, 'local-corpus/real-reader-v1')

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

export function validateManifest(manifest) {
  const checks = []
  checks.push(
    assertion(
      'manifest.schema-version',
      objectRecord(manifest) && manifest.schemaVersion === 1,
      1,
      objectRecord(manifest) ? manifest.schemaVersion : typeof manifest,
    ),
  )
  if (!objectRecord(manifest)) return checks
  checks.push(
    assertion(
      'manifest.corpus-id',
      typeof manifest.corpusId === 'string' && ID.test(manifest.corpusId),
      'lowercase kebab-case identifier',
      manifest.corpusId,
    ),
  )
  checks.push(
    assertion(
      'manifest.offline-only',
      manifest.execution?.networkPolicy === 'forbidden',
      'forbidden',
      manifest.execution?.networkPolicy,
    ),
  )
  const cases = Array.isArray(manifest.cases) ? manifest.cases : []
  checks.push(assertion('manifest.case-count', cases.length > 0, '> 0', cases.length))

  const ids = new Set()
  for (const [index, item] of cases.entries()) {
    const prefix = `case.${index + 1}`
    const validRecord = objectRecord(item)
    checks.push(assertion(`${prefix}.record`, validRecord, 'object', typeof item))
    if (!validRecord) continue
    checks.push(
      assertion(
        `${prefix}.id`,
        typeof item.id === 'string' && ID.test(item.id) && !ids.has(item.id),
        'unique lowercase kebab-case identifier',
        item.id,
      ),
    )
    if (typeof item.id === 'string') ids.add(item.id)
    checks.push(
      assertion(
        `${prefix}.chapter-id`,
        typeof item.chapterId === 'string' && ID.test(item.chapterId),
        'lowercase kebab-case chapter identifier',
        item.chapterId,
      ),
    )
    const object = item.object
    const sha = object?.sha256
    const extension = typeof object?.path === 'string' ? extname(object.path).toLowerCase() : ''
    checks.push(
      assertion(`${prefix}.sha256`, typeof sha === 'string' && SHA256.test(sha), 'SHA-256', sha),
    )
    checks.push(
      assertion(
        `${prefix}.object-path`,
        typeof sha === 'string' &&
          typeof object?.path === 'string' &&
          object.path === `objects/${sha}${extension}` &&
          MIME_BY_EXTENSION.has(extension),
        `objects/<sha256>.<supported-extension>`,
        object?.path,
      ),
    )
    checks.push(
      assertion(
        `${prefix}.mime-type`,
        MIME_BY_EXTENSION.get(extension) === object?.mimeType,
        MIME_BY_EXTENSION.get(extension),
        object?.mimeType,
      ),
    )
    checks.push(
      assertion(
        `${prefix}.byte-length`,
        Number.isSafeInteger(object?.bytes) && object.bytes > 0,
        'positive integer',
        object?.bytes,
      ),
    )
    checks.push(
      assertion(
        `${prefix}.dimensions`,
        Number.isSafeInteger(object?.width) &&
          object.width > 0 &&
          Number.isSafeInteger(object?.height) &&
          object.height > 0,
        'positive integer width and height',
        { width: object?.width, height: object?.height },
      ),
    )
    checks.push(
      assertion(
        `${prefix}.provenance`,
        ['asura', 'webtoon'].includes(item.provenance?.provider) &&
          typeof item.provenance?.chapterUrl === 'string' &&
          /^https:\/\//u.test(item.provenance.chapterUrl) &&
          Number.isSafeInteger(item.provenance?.pageIndex) &&
          item.provenance.pageIndex > 0 &&
          typeof item.provenance?.capturedAtUtc === 'string',
        'provider, HTTPS chapter provenance, page index, and capture time',
        item.provenance,
      ),
    )
    checks.push(
      assertion(
        `${prefix}.quality-focus`,
        Array.isArray(item.qualityFocus) &&
          item.qualityFocus.length > 0 &&
          item.qualityFocus.every((tag) => typeof tag === 'string' && ID.test(tag)),
        'one or more kebab-case quality tags',
        item.qualityFocus,
      ),
    )
    if (item.expectations?.exactRegionCount !== undefined) {
      checks.push(
        assertion(
          `${prefix}.exact-region-count`,
          Number.isSafeInteger(item.expectations.exactRegionCount) &&
            item.expectations.exactRegionCount >= 0 &&
            item.expectations.minimumRegionCount === undefined,
          'a non-negative integer without minimumRegionCount',
          item.expectations,
        ),
      )
    }
    const expectations = item.expectations
    if (expectations !== undefined) {
      const stringLists = [
        'excludedSourceTexts',
        'preserveNamesWhenDetected',
        'requiredSourceFragments',
      ]
      checks.push(
        assertion(
          `${prefix}.expectations`,
          objectRecord(expectations) &&
            (expectations.minimumRegionCount === undefined ||
              (Number.isSafeInteger(expectations.minimumRegionCount) &&
                expectations.minimumRegionCount > 0)) &&
            (expectations.maximumFirstRegionReadyMs === undefined ||
              (Number.isSafeInteger(expectations.maximumFirstRegionReadyMs) &&
                expectations.maximumFirstRegionReadyMs > 0)) &&
            (expectations.initialVisibleRects === undefined ||
              (Array.isArray(expectations.initialVisibleRects) &&
                expectations.initialVisibleRects.length > 0 &&
                expectations.initialVisibleRects.length <= 64 &&
                expectations.initialVisibleRects.every(
                  (rect) =>
                    objectRecord(rect) &&
                    [rect.x, rect.y, rect.width, rect.height].every(Number.isFinite) &&
                    rect.x >= 0 &&
                    rect.y >= 0 &&
                    rect.width > 0 &&
                    rect.height > 0 &&
                    rect.x + rect.width <= 1 &&
                    rect.y + rect.height <= 1,
                ))) &&
            (expectations.protectedArtworkRects === undefined ||
              (Array.isArray(expectations.protectedArtworkRects) &&
                expectations.protectedArtworkRects.length > 0 &&
                expectations.protectedArtworkRects.length <= 64 &&
                expectations.protectedArtworkRects.every(
                  (rect) =>
                    objectRecord(rect) &&
                    [rect.x, rect.y, rect.width, rect.height].every(Number.isFinite) &&
                    rect.x >= 0 &&
                    rect.y >= 0 &&
                    rect.width > 0 &&
                    rect.height > 0 &&
                    rect.x + rect.width <= 1 &&
                    rect.y + rect.height <= 1,
                ))) &&
            stringLists.every(
              (field) =>
                expectations[field] === undefined ||
                (Array.isArray(expectations[field]) &&
                  expectations[field].length > 0 &&
                  expectations[field].every(
                    (value) => typeof value === 'string' && value.trim().length > 0,
                  )),
            ),
          'supported semantic expectations and normalized viewport probes',
          expectations,
        ),
      )
    }
  }

  const requirements = manifest.coverageRequirements
  if (objectRecord(requirements)) {
    const providerCount = new Set(cases.map((item) => item?.provenance?.provider).filter(Boolean))
      .size
    const chapterCount = new Set(cases.map((item) => item?.chapterId).filter(Boolean)).size
    const qualityFocus = new Set(cases.flatMap((item) => item?.qualityFocus ?? []))
    checks.push(
      assertion(
        'coverage.providers',
        providerCount >= requirements.minimumProviders,
        `>= ${requirements.minimumProviders}`,
        providerCount,
      ),
      assertion(
        'coverage.chapters',
        chapterCount >= requirements.minimumChapters,
        `>= ${requirements.minimumChapters}`,
        chapterCount,
      ),
      assertion(
        'coverage.cases',
        cases.length >= requirements.minimumCases,
        `>= ${requirements.minimumCases}`,
        cases.length,
      ),
      assertion(
        'coverage.quality-focus',
        Array.isArray(requirements.requiredQualityFocus) &&
          requirements.requiredQualityFocus.every((tag) => qualityFocus.has(tag)),
        requirements.requiredQualityFocus,
        [...qualityFocus].sort(),
      ),
    )
  }

  const selections = objectRecord(manifest.selections) ? manifest.selections : {}
  for (const [name, selectedIds] of Object.entries(selections)) {
    checks.push(
      assertion(
        `selection.${name}`,
        ID.test(name) &&
          Array.isArray(selectedIds) &&
          selectedIds.length > 0 &&
          new Set(selectedIds).size === selectedIds.length &&
          selectedIds.every((id) => ids.has(id)),
        'ordered unique known case IDs',
        selectedIds,
      ),
    )
  }
  return checks
}

export function selectedCases(manifest, selection = 'all') {
  if (selection === 'all') return manifest.cases ?? []
  const ids = manifest.selections?.[selection]
  if (!Array.isArray(ids)) throw new Error(`Unknown corpus selection: ${selection}`)
  const byId = new Map((manifest.cases ?? []).map((item) => [item.id, item]))
  return ids.map((id) => byId.get(id)).filter(Boolean)
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
  const cases = manifestPassed ? selectedCases(manifest, selection) : []
  const assets = []

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
        continue
      }
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
  }

  const failures = assertions.filter((item) => !item.passed)
  const missing = assets.filter((item) => item.state === 'missing')
  return {
    schemaVersion: 1,
    corpusId: manifest.corpusId,
    selection,
    offline: true,
    manifestPath: absoluteManifest,
    corpusRoot: absoluteCorpus,
    caseCount: cases.length,
    verifiedCount: assets.filter((item) => item.state === 'verified').length,
    missingCount: missing.length,
    status: failures.length === 0 ? 'passed' : 'failed',
    assertions,
    assets,
    failures,
    remediation:
      missing.length > 0
        ? {
            message:
              'Local real-reader corpus objects are missing. Restore the exact content-addressed files; the runner will not download them.',
            requiredPaths: missing.map((item) => item.path),
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
    `${result.status.toUpperCase()}: ${result.verifiedCount}/${result.caseCount} real-reader assets verified (${result.selection}).\n`,
  )
  for (const failure of result.failures) {
    process.stdout.write(`- ${failure.id}: expected ${JSON.stringify(failure.expected)}, got ${JSON.stringify(failure.actual)}\n`)
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
    if (result.status !== 'passed') process.exitCode = result.missingCount > 0 ? 2 : 1
  } catch (error) {
    const result = {
      schemaVersion: 1,
      status: 'error',
      offline: true,
      message: error instanceof Error ? error.message : String(error),
    }
    process.stderr.write(`${JSON.stringify(result, null, 2)}\n`)
    process.exitCode = 1
  }
}
