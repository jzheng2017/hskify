import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { DEFAULT_CORPUS_ROOT, DEFAULT_MANIFEST_PATH } from './real-reader-corpus.mjs'
import { runBrowserRegression } from './run-real-reader-browser-regression.mjs'

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))

function check(id, passed, expected, actual, detail) {
  return { id, passed, expected, actual, ...(detail ? { detail } : {}) }
}

function polygonBounds(points) {
  if (!Array.isArray(points) || points.length < 3) return undefined
  const xs = points.map((point) => point?.x)
  const ys = points.map((point) => point?.y)
  if (![...xs, ...ys].every((value) => Number.isFinite(value))) return undefined
  return {
    x0: Math.min(...xs),
    y0: Math.min(...ys),
    x1: Math.max(...xs),
    y1: Math.max(...ys),
  }
}

function validRect(rect) {
  return (
    rect &&
    [rect.x, rect.y, rect.width, rect.height].every(Number.isFinite) &&
    rect.x >= 0 &&
    rect.y >= 0 &&
    rect.width > 0 &&
    rect.height > 0 &&
    rect.x + rect.width <= 1.000001 &&
    rect.y + rect.height <= 1.000001
  )
}

function rectsOverlap(left, right) {
  if (!validRect(left) || !validRect(right)) return false
  return (
    Math.min(left.x + left.width, right.x + right.width) > Math.max(left.x, right.x) &&
    Math.min(left.y + left.height, right.y + right.height) > Math.max(left.y, right.y)
  )
}

function rectOverlapsPolygon(rect, polygon) {
  if (!validRect(rect)) return false
  const bounds = polygonBounds(polygon)
  if (!bounds) return false
  const overlapWidth = Math.min(rect.x + rect.width, bounds.x1) - Math.max(rect.x, bounds.x0)
  const overlapHeight = Math.min(rect.y + rect.height, bounds.y1) - Math.max(rect.y, bounds.y0)
  return overlapWidth > 0 && overlapHeight > 0
}

function finalRegions(updates) {
  const regions = new Map()
  for (const update of updates) {
    if (update.type === 'regionReady' && update.region?.id) {
      regions.set(update.region.id, structuredClone(update.region))
    }
  }
  return [...regions.values()]
}

function preservedArtwork(updates) {
  const regions = new Map()
  for (const update of updates) {
    if (update.type === 'artworkPreserved' && update.region?.id) {
      regions.set(update.region.id, structuredClone(update.region))
    }
  }
  return [...regions.values()]
}

function normalizedOcrText(value) {
  return String(value ?? '')
    .toLocaleUpperCase()
    .replaceAll(/[^A-Z0-9]+/gu, '')
}

function orderedCharacterCoverage(expected, actual) {
  const left = normalizedOcrText(expected)
  const right = normalizedOcrText(actual)
  if (!left || !right) return 0
  if (right.includes(left)) return 1
  if (right.length > left.length * 2) return 0
  let previous = new Uint16Array(right.length + 1)
  for (let leftIndex = 0; leftIndex < left.length; leftIndex += 1) {
    const current = new Uint16Array(right.length + 1)
    for (let rightIndex = 0; rightIndex < right.length; rightIndex += 1) {
      current[rightIndex + 1] =
        left[leftIndex] === right[rightIndex]
          ? previous[rightIndex] + 1
          : Math.max(previous[rightIndex + 1], current[rightIndex])
    }
    previous = current
  }
  return previous[right.length] / left.length
}

export function assertSemanticExpectations(item, regions, preserved = []) {
  const assertions = []
  const combinedSource = regions.map((region) => region.sourceEnglish ?? '').join('\n')
  for (const fragment of item.expectations?.requiredSourceFragments ?? []) {
    assertions.push(
      check(
        `semantic.${item.id}.required-source.${fragment}`,
        combinedSource.toLocaleLowerCase().includes(fragment.toLocaleLowerCase()),
        `sourceEnglish contains ${fragment}`,
        combinedSource,
        'Required OCR fragments make missed-text and partial-translation regressions terminal failures.',
      ),
    )
  }
  for (const sourceText of item.expectations?.excludedSourceTexts ?? []) {
    const matches = regions.filter(
      (region) => region.sourceEnglish?.trim().toLocaleUpperCase() === sourceText.toUpperCase(),
    )
    assertions.push(
      check(
        `semantic.${item.id}.excluded-source.${sourceText}`,
        matches.length === 0,
        0,
        matches.length,
        'translateSoundEffects=false must keep excluded SFX out of the translation regions.',
      ),
    )
  }
  const preservedTexts = preserved.map((region) => region.sourceEnglish ?? '').filter(Boolean)
  const translatedTexts = regions.map((region) => region.sourceEnglish ?? '').filter(Boolean)
  const preservedSource = preservedTexts.join('\n')
  for (const fragment of item.expectations?.preservedArtworkSourceFragments ?? []) {
    const preservedCoverage = Math.max(
      0,
      ...preservedTexts.map((source) => orderedCharacterCoverage(fragment, source)),
    )
    const translatedCoverage = Math.max(
      0,
      ...translatedTexts.map((source) => orderedCharacterCoverage(fragment, source)),
    )
    const preservedMatches = preservedCoverage >= 0.68
    const translatedMatches = translatedCoverage >= 0.8
    assertions.push(
      check(
        `semantic.${item.id}.preserved-artwork.${fragment}`,
        preservedMatches && !translatedMatches,
        'preserved source artwork without a translated overlay',
        {
          preservedMatches,
          translatedMatches,
          preservedCoverage,
          translatedCoverage,
          preservedSource,
          combinedSource,
        },
        'Illustrated technique lettering must remain source artwork instead of receiving a cleanup patch and standard-font overlay.',
      ),
    )
  }
  for (const name of item.expectations?.preserveNamesWhenDetected ?? []) {
    const detected = regions.flatMap((region) => {
      const source = region.sourceEnglish ?? ''
      const start = source.toLocaleLowerCase().indexOf(name.toLocaleLowerCase())
      return start < 0
        ? []
        : [{ region, exactSourceSpelling: source.slice(start, start + name.length) }]
    })
    const preserved = detected.filter(({ region, exactSourceSpelling }) =>
      region.displayedChinese?.includes(exactSourceSpelling),
    )
    assertions.push({
      ...check(
        `semantic.${item.id}.preserve-name.${name}`,
        detected.length === 0 || preserved.length === detected.length,
        detected.length,
        preserved.length,
        'When OCR detects an annotated name, keep-original must preserve it in every corresponding Chinese region.',
      ),
      skipped: detected.length === 0,
    })
  }
  return assertions
}

export function assertCompletedJob(item, hskLevel, terminal, updates, patchRecords) {
  const regions = finalRegions(updates)
  const preserved = preservedArtwork(updates)
  const exactRegionCount = item.expectations?.exactRegionCount
  const expectedRegionCount =
    exactRegionCount === undefined
      ? `>= ${item.expectations?.minimumRegionCount ?? 1}`
      : exactRegionCount
  const regionCountPassed =
    exactRegionCount === undefined
      ? regions.length >= (item.expectations?.minimumRegionCount ?? 1)
      : regions.length === exactRegionCount
  const assertions = [
    check(
      `job.${item.id}.hsk-${hskLevel}.terminal`,
      terminal?.type === 'complete',
      'complete',
      terminal?.type,
    ),
    check(
      `job.${item.id}.hsk-${hskLevel}.regions`,
      regionCountPassed,
      expectedRegionCount,
      regions.length,
    ),
  ]
  const patchById = new Map(patchRecords.map((record) => [record.blobId, record]))
  for (const region of regions) {
    const prefix = `job.${item.id}.hsk-${hskLevel}.region.${region.id}`
    assertions.push(
      check(
        `${prefix}.source`,
        Boolean(region.sourceEnglish?.trim()),
        'non-empty',
        region.sourceEnglish,
      ),
      check(
        `${prefix}.translation`,
        Boolean(region.displayedChinese?.trim()),
        'non-empty',
        region.displayedChinese,
      ),
      check(
        `${prefix}.repair-terminal`,
        Boolean(region.hsk?.repairState) && region.hsk.repairState !== 'pending',
        'accepted or exhausted',
        region.hsk?.repairState,
      ),
      check(
        `${prefix}.requested-level`,
        region.hsk?.requestedLevel === hskLevel,
        hskLevel,
        region.hsk?.requestedLevel,
      ),
      check(
        `${prefix}.patch-rect`,
        validRect(region.patch?.rect),
        'normalized non-empty rect',
        region.patch?.rect,
      ),
      check(
        `${prefix}.patch-source-overlap`,
        rectOverlapsPolygon(region.patch?.rect, region.textPolygon),
        true,
        rectOverlapsPolygon(region.patch?.rect, region.textPolygon),
      ),
    )
    const patch = patchById.get(region.patch?.blobId)
    assertions.push(
      check(`${prefix}.patch-present`, Boolean(patch), true, Boolean(patch)),
      check(
        `${prefix}.patch-mime`,
        region.patch?.mimeType === 'image/png',
        'image/png',
        region.patch?.mimeType,
      ),
      check(`${prefix}.patch-png`, patch?.validPng === true, true, patch?.validPng),
    )
  }
  for (const [index, protectedRect] of (item.expectations?.protectedArtworkRects ?? []).entries()) {
    const overlapping = regions
      .filter((region) => rectsOverlap(region.patch?.rect, protectedRect))
      .map((region) => ({
        id: region.id,
        sourceEnglish: region.sourceEnglish,
        patchRect: region.patch?.rect,
      }))
    assertions.push(
      check(
        `semantic.${item.id}.protected-artwork-rect.${index + 1}`,
        overlapping.length === 0,
        'no cleanup patch or translated overlay intersects the annotated source artwork',
        overlapping,
        'Locally annotated illustrated lettering must remain pixel-identical even when OCR can read only a fragment of its stylized glyphs.',
      ),
    )
  }
  assertions.push(...assertSemanticExpectations(item, regions, preserved))
  return { regions, preserved, assertions }
}

function regionComparisonKey(region) {
  return `${region.readingOrder ?? ''}\u0000${region.sourceEnglish?.trim() ?? ''}`
}

export function assertHskDifferential(lowRun, highRun) {
  const highByKey = new Map(highRun.regions.map((region) => [regionComparisonKey(region), region]))
  const shared = lowRun.regions
    .map((region) => [region, highByKey.get(regionComparisonKey(region))])
    .filter((pair) => pair[1])
  const changed = shared.filter(
    ([low, high]) => low.displayedChinese?.trim() !== high.displayedChinese?.trim(),
  )
  return [
    check('differential.hsk-2-vs-5.shared-regions', shared.length > 0, '> 0', shared.length),
    check(
      'differential.hsk-2-vs-5.changed-output',
      changed.length > 0,
      '> 0 shared translations changed',
      changed.length,
    ),
    check(
      'differential.hsk-2.low-level-validator',
      lowRun.regions.length > 0 &&
        lowRun.regions.every(
          (region) =>
            region.hsk?.requestedLevel === 2 &&
            region.hsk?.repairState &&
            region.hsk.repairState !== 'pending',
        ),
      'every HSK2 region used requestedLevel=2 and reached terminal repair state',
      lowRun.regions.map((region) => ({
        id: region.id,
        requestedLevel: region.hsk?.requestedLevel,
        repairState: region.hsk?.repairState,
      })),
    ),
  ]
}

function parseArguments(argv) {
  const timestamp = new Date().toISOString().replaceAll(':', '').replaceAll('.', '-')
  const options = {
    manifestPath: DEFAULT_MANIFEST_PATH,
    corpusRoot: DEFAULT_CORPUS_ROOT,
    configPath: process.env.HSKIFY_REAL_READER_BROWSER_CONFIG
      ? resolve(process.env.HSKIFY_REAL_READER_BROWSER_CONFIG)
      : undefined,
    selection: 'core',
    caseId: undefined,
    outputDirectory: resolve(REPOSITORY_ROOT, `runs/real-reader-${timestamp}`),
    timeoutMinutes: 20,
    headed: false,
  }
  const args = [...argv]
  while (args.length > 0) {
    const argument = args.shift()
    if (argument === '--manifest') options.manifestPath = resolve(args.shift() ?? '')
    else if (argument === '--corpus') options.corpusRoot = resolve(args.shift() ?? '')
    else if (argument === '--config' || argument === '--browser-config')
      options.configPath = resolve(args.shift() ?? '')
    else if (argument === '--selection') options.selection = args.shift() ?? ''
    else if (argument === '--case') options.caseId = args.shift() ?? ''
    else if (argument === '--output') options.outputDirectory = resolve(args.shift() ?? '')
    else if (argument === '--timeout-minutes') options.timeoutMinutes = Number(args.shift())
    else if (argument === '--headed') options.headed = true
    else throw new Error(`Unknown argument: ${argument}`)
  }
  if (!Number.isFinite(options.timeoutMinutes) || options.timeoutMinutes <= 0) {
    throw new Error('--timeout-minutes must be positive.')
  }
  return options
}

/**
 * The old daemon-direct driver was intentionally removed.  The only executable
 * regression entry point is runRegression below, which delegates to the
 * packaged Firefox local-reader harness.  Keeping this boundary explicit
 * prevents a daemon-only green result from being mistaken for browser quality.
 */
/**
 * Run the release regression through the same packaged Firefox path as users.
 *
 * The browser runner intentionally fails closed when the local v2 corpus,
 * packaged extension, or pinned model identities are unavailable.  In
 * particular, this wrapper never falls back to posting jobs directly to the
 * daemon, because that would test a different product surface.
 */
export async function runRegression(options) {
  return runBrowserRegression({
    manifestPath: options.manifestPath,
    corpusRoot: options.corpusRoot,
    selection: options.selection,
    caseId: options.caseId,
    configPath: options.configPath,
    outputDirectory: options.outputDirectory,
    timeoutMs: options.timeoutMinutes * 60_000,
    headed: options.headed === true,
  })
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : undefined
if (invokedPath === fileURLToPath(import.meta.url)) {
  try {
    const options = parseArguments(process.argv.slice(2))
    const summary = await runRegression(options)
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`)
    if (summary.status !== 'passed') process.exitCode = summary.captureRequired ? 2 : 1
  } catch (error) {
    process.stderr.write(
      `${JSON.stringify(
        {
          schemaVersion: 2,
          status: 'error',
          offline: true,
          message: error instanceof Error ? error.message : String(error),
        },
        null,
        2,
      )}\n`,
    )
    process.exitCode = 1
  }
}
