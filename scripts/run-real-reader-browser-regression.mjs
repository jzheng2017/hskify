/*
 * End-to-end real-reader-v2 release harness.
 *
 * This driver deliberately goes through a packaged Firefox extension.  It
 * never creates daemon jobs itself: page discovery, image capture, upload,
 * publication, patch installation, and the terminal state are exercised by
 * the same browser path users run.  The local reader server is only a
 * deterministic replica of the public reader surface; it serves the
 * content-addressed objects from the checked-out v2 corpus and does not
 * contact the provenance URLs.
 */

import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs'
import { createServer } from 'node:http'
import { extname, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  chapterDomEvidence,
  extensionMessage,
  installDomObserver,
  launchPackagedFirefox,
  prepareContentRuntime,
  routeEvidence,
  startJobMonitor,
  stopJobMonitor,
  timedContentStart,
  waitForPageState,
  writeJsonSync,
} from './benchmark/browser-harness.mjs'
import {
  auditCorpus,
  DEFAULT_CORPUS_ROOT,
  DEFAULT_MANIFEST_PATH,
  selectedCases,
} from './real-reader-corpus.mjs'

const REPOSITORY_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const DEFAULT_OUTPUT = resolve(REPOSITORY_ROOT, '.cache/real-reader-browser-regression')
const DEFAULT_TIMEOUT_MS = 5 * 60_000

function jsonError(message, extra = {}) {
  return {
    schemaVersion: 2,
    status: 'failed',
    transport: 'packaged-firefox-local-reader',
    ...extra,
    message,
  }
}

function pageId(chapterId, order) {
  return `${chapterId}-page-${String(order + 1).padStart(4, '0')}`
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function mimeType(path) {
  switch (extname(path).toLowerCase()) {
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.png':
      return 'image/png'
    case '.webp':
      return 'image/webp'
    case '.html':
      return 'text/html; charset=utf-8'
    case '.js':
      return 'text/javascript; charset=utf-8'
    default:
      return 'application/octet-stream'
  }
}

function validContainedPath(root, candidate) {
  const absolute = resolve(root, candidate)
  const fromRoot = relative(root, absolute)
  if (!fromRoot || fromRoot.startsWith('..') || fromRoot.includes(`..${sep}`)) return undefined
  return absolute
}

function polygonBounds(points) {
  if (!Array.isArray(points) || points.length < 3) return undefined
  const xs = points.map((point) => Number(Array.isArray(point) ? point[0] : point?.x))
  const ys = points.map((point) => Number(Array.isArray(point) ? point[1] : point?.y))
  if (![...xs, ...ys].every(Number.isFinite)) return undefined
  return {
    x: Math.min(...xs),
    y: Math.min(...ys),
    width: Math.max(...xs) - Math.min(...xs),
    height: Math.max(...ys) - Math.min(...ys),
  }
}

function overlapOverSmaller(left, right) {
  if (
    !left ||
    !right ||
    left.width <= 0 ||
    left.height <= 0 ||
    right.width <= 0 ||
    right.height <= 0
  )
    return 0
  const overlapWidth = Math.max(
    0,
    Math.min(left.x + left.width, right.x + right.width) - Math.max(left.x, right.x),
  )
  const overlapHeight = Math.max(
    0,
    Math.min(left.y + left.height, right.y + right.height) - Math.max(left.y, right.y),
  )
  const overlap = overlapWidth * overlapHeight
  const smaller = Math.min(left.width * left.height, right.width * right.height)
  return smaller > 0 ? overlap / smaller : 0
}

function normalizedEnglish(value) {
  return String(value ?? '')
    .toLocaleLowerCase()
    .replaceAll(/[^a-z0-9]+/gu, '')
}

function levenshtein(left, right) {
  const previous = new Uint16Array(right.length + 1)
  for (let index = 0; index <= right.length; index += 1) previous[index] = index
  for (let leftIndex = 0; leftIndex < left.length; leftIndex += 1) {
    const current = new Uint16Array(right.length + 1)
    current[0] = leftIndex + 1
    for (let rightIndex = 0; rightIndex < right.length; rightIndex += 1) {
      current[rightIndex + 1] =
        left[leftIndex] === right[rightIndex]
          ? previous[rightIndex]
          : 1 + Math.min(previous[rightIndex], current[rightIndex], previous[rightIndex + 1])
    }
    previous.set(current)
  }
  return previous[right.length]
}

export function annotationCoverage(chapter, manifestPath, route) {
  const expectedByPage = chapter.pages.map((page) => ({
    page: page.order + 1,
    annotation: JSON.parse(readFileSync(resolve(manifestPath, '..', page.annotation.path), 'utf8')),
  }))
  const observedByPage = new Map(
    (route?.jobs ?? []).map((job) => [
      job.pageIndex + 1,
      job.updates.filter((update) => update.type === 'regionReady').map((update) => update.region),
    ]),
  )
  const observed = (page) => observedByPage.get(page) ?? []
  const match = (polygon, regions) =>
    regions.some(
      (region) =>
        overlapOverSmaller(polygonBounds(polygon), polygonBounds(region.textPolygon)) >= 0.5,
    )
  const targets = expectedByPage.flatMap(({ page, annotation }) =>
    annotation.regions.map((region) => ({ page, region })),
  )
  const exclusions = expectedByPage.flatMap(({ page, annotation }) =>
    annotation.exclusions.map((region) => ({ page, region })),
  )
  const missingTargets = targets.filter(
    ({ page, region }) => !match(region.polygon, observed(page)),
  )
  const modifiedExclusions = exclusions.filter(({ page, region }) =>
    match(region.polygon, observed(page)),
  )
  const errors = []
  for (const { page, region } of targets) {
    const candidate = observed(page)
      .map((observedRegion) => ({
        observedRegion,
        overlap: overlapOverSmaller(
          polygonBounds(region.polygon),
          polygonBounds(observedRegion.textPolygon),
        ),
      }))
      .sort((left, right) => right.overlap - left.overlap)[0]
    if (!candidate || candidate.overlap < 0.5) continue
    const expectedText = normalizedEnglish(region.sourceEnglish)
    const actualText = normalizedEnglish(candidate.observedRegion.sourceEnglish)
    if (expectedText.length > 0)
      errors.push({
        page,
        id: region.id,
        cer: levenshtein(expectedText, actualText) / expectedText.length,
      })
  }
  const sortedErrors = [...errors].sort((left, right) => left.cer - right.cer)
  const p95Index =
    sortedErrors.length > 0
      ? Math.min(sortedErrors.length - 1, Math.ceil(sortedErrors.length * 0.95) - 1)
      : -1
  return {
    expectedTargetCount: targets.length,
    matchedTargetCount: targets.length - missingTargets.length,
    missingTargets: missingTargets.map(({ page, region }) => ({ page, id: region.id })),
    exclusionCount: exclusions.length,
    modifiedExclusions: modifiedExclusions.map(({ page, region }) => ({ page, id: region.id })),
    ocrCer:
      errors.length > 0 ? errors.reduce((sum, error) => sum + error.cer, 0) / errors.length : 1,
    p95RegionCer: p95Index >= 0 ? sortedErrors[p95Index].cer : 1,
    highErrorRegions: errors.filter((error) => error.cer > 0.1),
  }
}

/**
 * Check semantic evidence that cannot be reduced to OCR geometry alone. The
 * page annotations provide reviewed entity spans and continuation groups;
 * the browser route must preserve those typed decisions without relying on
 * capitalization, title wordlists, or completion order.
 */
export function semanticConsistency(chapter, manifestPath, route) {
  const observedByPage = new Map(
    (route?.jobs ?? []).map((job) => [
      job.pageIndex + 1,
      job.updates.filter((update) => update.type === 'regionReady').map((update) => update.region),
    ]),
  )
  const findObserved = (page, polygon) =>
    (observedByPage.get(page) ?? [])
      .map((region) => ({
        region,
        overlap: overlapOverSmaller(polygonBounds(polygon), polygonBounds(region.textPolygon)),
      }))
      .sort((left, right) => right.overlap - left.overlap)[0]
  const missingEntities = []
  const nameViolations = []
  const translatedDescriptionViolations = []
  const continuationViolations = []
  const expectedContinuationGroups = new Map()
  for (const page of chapter.pages) {
    const annotation = JSON.parse(readFileSync(resolve(manifestPath, '..', page.annotation.path), 'utf8'))
    for (const expected of annotation.regions) {
      const observed = findObserved(page.order + 1, expected.polygon)
      if (!observed || observed.overlap < 0.5) continue
      const actual = observed.region
      const actualEntities = Array.isArray(actual.entities) ? actual.entities : []
      for (const entity of expected.entities ?? []) {
        const expectedSource =
          entity.source ?? [...expected.sourceEnglish].slice(entity.start, entity.end).join('')
        const match = actualEntities.find(
          (candidate) =>
            candidate.startChar === entity.start &&
            candidate.endChar === entity.end &&
            candidate.entityType === entity.type &&
            String(candidate.source ?? '').toLocaleLowerCase() === expectedSource.toLocaleLowerCase(),
        )
        if (!match) {
          missingEntities.push({ page: page.order + 1, id: expected.id, source: expectedSource, type: entity.type })
          continue
        }
        const opaque = ['person', 'place', 'organization', 'coined'].includes(entity.type)
        if (opaque && match.translated !== expectedSource)
          nameViolations.push({ page: page.order + 1, id: expected.id, source: expectedSource, translated: match.translated })
        if (!opaque && match.translated === expectedSource)
          translatedDescriptionViolations.push({ page: page.order + 1, id: expected.id, source: expectedSource, type: entity.type })
      }
      if (expected.continuationGroup) {
        const values = expectedContinuationGroups.get(expected.continuationGroup) ?? []
        values.push({ id: expected.id, actual: actual.contextGroup })
        expectedContinuationGroups.set(expected.continuationGroup, values)
      }
    }
  }
  for (const [group, members] of expectedContinuationGroups) {
    const actualGroups = new Set(members.map((member) => member.actual).filter(Boolean))
    if (actualGroups.size !== 1 || members.some((member) => !member.actual))
      continuationViolations.push({ group, members })
  }
  return {
    missingEntities,
    nameViolations,
    translatedDescriptionViolations,
    continuationViolations,
  }
}

export function publicationConsistency(dom, route) {
  const publishedRegions = (route?.jobs ?? []).flatMap((job) =>
    job.updates
      .filter((update) =>
        ['regionReady', 'artworkPreserved', 'unreadable'].includes(update.type),
      )
      .map((update) => ({ type: update.type, region: update.region })),
  )
  const published = new Map(
    publishedRegions.map((entry) => [entry.region.id, entry]),
  )
  const rendered = new Map((dom?.regions ?? []).map((region) => [region.regionId, region]))
  const missing = []
  const mismatched = []
  for (const [id, entry] of published) {
    const region = entry.region
    const actual = rendered.get(id)
    if (!actual) {
      missing.push(id)
      continue
    }
    if (entry.type === 'regionReady') {
      if (
        actual.sourcePreserving ||
        actual.text !== region.displayedChinese ||
        actual.pinyin !== region.pinyin
      )
        mismatched.push(id)
    } else {
      const expectedText = region.translatedChinese || region.sourceEnglish
      if (!actual.sourcePreserving || actual.text !== expectedText || (region.pinyin ?? '') !== actual.pinyin)
        mismatched.push(id)
    }
  }
  const duplicatePublishedIds = publishedRegions
    .map((entry) => entry.region.id)
    .filter((id, index, ids) => ids.indexOf(id) !== index)
  const untranslatedEnglish = publishedRegions
    .filter((entry) => entry.type === 'regionReady')
    .map((entry) => entry.region)
    .filter((region) => {
      const displayed = String(region.displayedChinese ?? '')
      const chars = [...displayed]
      const entitySpans = Array.isArray(region.entities) ? region.entities : []
      const protectedSpans = entitySpans
        .filter(
          (entity) =>
            ['person', 'place', 'organization', 'coined'].includes(entity.entityType) &&
            entity.translated === entity.source,
        )
        .flatMap((entity) => {
          const sourceChars = [...String(entity.source ?? '')]
          if (sourceChars.length === 0) return []
          const spans = []
          for (let index = 0; index <= chars.length - sourceChars.length; index += 1) {
            if (sourceChars.every((character, offset) => chars[index + offset] === character))
              spans.push([index, index + sourceChars.length])
          }
          return spans
        })
      let start = -1
      for (let index = 0; index <= chars.length; index += 1) {
        const alphabetic = index < chars.length && /[A-Za-z]/u.test(chars[index])
        if (alphabetic && start < 0) start = index
        if (!alphabetic && start >= 0) {
          const end = index
          const covered = protectedSpans.some(([spanStart, spanEnd]) => spanStart <= start && spanEnd >= end)
          if (!covered) return true
          start = -1
        }
      }
      return false
    })
    .map((region) => region.id)
  const weakEvidence = publishedRegions
    .filter((entry) => entry.type === 'regionReady')
    .map((entry) => entry.region)
    .filter((region) => {
      const evidence = region.confidenceEvidence
      return (
        !evidence ||
        evidence.ocrConsensus < 0.55 ||
        evidence.geometryCoverage < 0.9 ||
        evidence.cleanupScore < 0.5
      )
    })
    .map((region) => region.id)
  return {
    publishedCount: published.size,
    renderedCount: rendered.size,
    missing,
    mismatched,
    duplicatePublishedIds: [...new Set(duplicatePublishedIds)],
    untranslatedEnglish,
    weakEvidence,
  }
}

export function routeJobConsistency(records, route) {
  const expected = records.map((record) => ({
    jobId: record.jobId,
    pageIndex: record.pageIndex,
    sourceSha256: record.sourceSha256,
  }))
  const actual = (route?.jobs ?? []).map((job) => ({
    jobId: job.jobId,
    pageIndex: job.pageIndex,
    sourceSha256: job.sourceSha256,
  }))
  const exact = expected.length === actual.length && expected.every((item, index) => {
    const candidate = actual[index]
    return (
      candidate?.jobId === item.jobId &&
      candidate?.pageIndex === item.pageIndex &&
      candidate?.sourceSha256 === item.sourceSha256
    )
  })
  return { exact, expected, actual }
}

export function pageMarkup(page, chapter, sourceUrl, { frame = false } = {}) {
  const kind = chapter.reader?.kind ?? 'continuous-image'
  const width = page.object.width
  const height = page.object.height
  const alt = escapeHtml(`${chapter.id} page ${page.order + 1}`)
  const style = `width:${width}px;max-width:100%;height:auto;display:block;margin:0 auto;`
  if (kind === 'canvas' || kind === 'webgl') {
    const webgl = kind === 'webgl' ? 'true' : 'false'
    return `<canvas data-reader-surface="${kind}" data-page="${page.order + 1}" width="${width}" height="${height}" style="${style}"></canvas><script>window.__hskifyReaderCanvasPromises=window.__hskifyReaderCanvasPromises||[];window.__hskifyReaderCanvasPromises.push((async()=>{const c=document.currentScript.previousElementSibling;const image=new Image();image.src=${JSON.stringify(sourceUrl)};await image.decode();const useWebgl=${webgl};let gl=useWebgl&&(c.getContext('webgl2')||c.getContext('webgl'));if(gl){const vertex=gl.createShader(gl.VERTEX_SHADER);gl.shaderSource(vertex,'attribute vec2 p;attribute vec2 t;varying vec2 v;void main(){gl_Position=vec4(p,0.,1.);v=t;}');gl.compileShader(vertex);const fragment=gl.createShader(gl.FRAGMENT_SHADER);gl.shaderSource(fragment,'precision mediump float;uniform sampler2D s;varying vec2 v;void main(){gl_FragColor=texture2D(s,v);}');gl.compileShader(fragment);const program=gl.createProgram();gl.attachShader(program,vertex);gl.attachShader(program,fragment);gl.linkProgram(program);gl.useProgram(program);const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,0,1,1,-1,1,1,-1,1,0,0,1,1,1,0]),gl.STATIC_DRAW);const position=gl.getAttribLocation(program,'p');const texcoord=gl.getAttribLocation(program,'t');gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,16,0);gl.enableVertexAttribArray(texcoord);gl.vertexAttribPointer(texcoord,2,gl.FLOAT,false,16,8);const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);gl.drawArrays(gl.TRIANGLE_STRIP,0,4)}else{const ctx=c.getContext('2d');ctx.drawImage(image,0,0,c.width,c.height)}window.parent.__hskifyReaderSurfaceReady=(window.parent.__hskifyReaderSurfaceReady||0)+1})().catch(error=>{document.documentElement.dataset.readerError=String(error)}))</script>`
  }
  if (kind === 'background') {
    return `<div data-reader-surface="background" data-page="${page.order + 1}" style="${style}height:${height}px;background-image:url(${JSON.stringify(sourceUrl)});background-repeat:no-repeat;background-size:100% 100%;"></div>`
  }
  return `<img data-page="${page.order + 1}" data-reader-surface="${kind}" src="${escapeHtml(sourceUrl)}" width="${width}" height="${height}" alt="${alt}" loading="eager" decoding="async" style="${style}" />`
}

export function chapterMarkup(chapter, pages, basePath) {
  const kind = chapter.reader?.kind ?? 'continuous-image'
  const surfaces = pages
    .sort((left, right) => left.order - right.order)
    .map((page) => {
      const id = pageId(chapter.id, page.order)
      const sourceUrl = `${basePath}/object/${encodeURIComponent(id)}`
      if (kind === 'iframe-image') {
        return `<iframe title="${escapeHtml(id)}" data-reader-surface="frame" src="${basePath}/frame/${encodeURIComponent(chapter.id)}/${page.order}" style="width:${page.object.width}px;max-width:100%;height:${page.object.height}px;border:0;display:block;margin:0 auto;" loading="eager"></iframe>`
      }
      return pageMarkup(page, chapter, sourceUrl)
    })
    .join('\n')
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escapeHtml(chapter.id)}</title><style>html,body{margin:0;background:#111;color:#eee}#chapter{width:min(100%,1280px);margin:0 auto}#chapter>img,#chapter>canvas,#chapter>div,#chapter>iframe{box-sizing:border-box}</style></head><body><script>window.__hskifyReaderCanvasPromises=[];window.__hskifyReaderSurfaceReady=0;window.__hskifyReaderExpectedSurfaces=${pages.length}</script><main id="chapter" data-reader-kind="${escapeHtml(kind)}" aria-label="${escapeHtml(chapter.id)}">${surfaces}</main><script>const frameReady=[...document.querySelectorAll('iframe')].map(frame=>new Promise(resolve=>{if(frame.contentDocument?.readyState==='complete')resolve();else frame.addEventListener('load',resolve,{once:true})}));window.__hskifyReaderReady=Promise.all([...document.images].map(image=>image.decode().catch(()=>undefined)).concat(window.__hskifyReaderCanvasPromises,frameReady)).then(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))))</script></body></html>`
}

function frameMarkup(chapter, page, sourceUrl, basePath) {
  return `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;background:#111}img{display:block;width:${page.object.width}px;height:${page.object.height}px;max-width:100%;object-fit:contain}</style></head><body>${pageMarkup(page, { ...chapter, reader: { kind: 'paged-image' } }, sourceUrl, { frame: true })}</body></html>`
}

export function createReaderServer({ manifest, corpusRoot, chapters }) {
  const byChapter = new Map(chapters.map((chapter) => [chapter.id, chapter]))
  const byPage = new Map()
  for (const chapter of chapters) {
    for (const page of chapter.pages ?? [])
      byPage.set(pageId(chapter.id, page.order), { chapter, page })
  }
  const server = createServer((request, response) => {
    try {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1')
      const path = decodeURIComponent(url.pathname)
      if (path === '/manifest.json') {
        const body = Buffer.from(JSON.stringify(manifest))
        response.writeHead(200, {
          'content-type': 'application/json',
          'content-length': body.length,
        })
        response.end(body)
        return
      }
      const chapterMatch = path.match(/^\/chapter\/([^/]+)$/u)
      if (chapterMatch) {
        const chapter = byChapter.get(chapterMatch[1])
        if (!chapter) return response.writeHead(404).end('Unknown chapter')
        const body = Buffer.from(chapterMarkup(chapter, [...(chapter.pages ?? [])], ''))
        response.writeHead(200, {
          'content-type': 'text/html; charset=utf-8',
          'content-length': body.length,
        })
        response.end(body)
        return
      }
      const frameMatch = path.match(/^\/frame\/([^/]+)\/(\d+)$/u)
      if (frameMatch) {
        const chapter = byChapter.get(frameMatch[1])
        const page = chapter?.pages?.find((candidate) => candidate.order === Number(frameMatch[2]))
        if (!chapter || !page) return response.writeHead(404).end('Unknown frame')
        const id = pageId(chapter.id, page.order)
        const body = Buffer.from(
          frameMarkup(chapter, page, `/object/${encodeURIComponent(id)}`, ''),
        )
        response.writeHead(200, {
          'content-type': 'text/html; charset=utf-8',
          'content-length': body.length,
        })
        response.end(body)
        return
      }
      const objectMatch = path.match(/^\/object\/([^/]+)$/u)
      if (objectMatch) {
        const entry = byPage.get(objectMatch[1])
        if (!entry) return response.writeHead(404).end('Unknown object')
        const objectPath = validContainedPath(corpusRoot, entry.page.object.path)
        if (!objectPath || !existsSync(objectPath) || !statSync(objectPath).isFile()) {
          return response.writeHead(503).end('Missing local real-reader object')
        }
        response.writeHead(200, {
          'content-type': entry.page.object.mimeType,
          'cache-control': 'no-store',
          'content-length': statSync(objectPath).size,
          'cross-origin-resource-policy': 'same-origin',
        })
        response.end(readFileSync(objectPath))
        return
      }
      response.writeHead(404).end('Not found')
    } catch (error) {
      if (!response.headersSent) response.writeHead(500)
      response.end(error instanceof Error ? error.message : String(error))
    }
  })
  return new Promise((resolvePromise, rejectPromise) => {
    server.once('error', rejectPromise)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string')
        return rejectPromise(new Error('Reader server did not bind.'))
      resolvePromise({ server, port: address.port })
    })
  })
}

export function requiredBrowserConfig(config) {
  const fields = [
    'extensionPackagePath',
    'firefoxExecutable',
    'playwrightModule',
    'extensionVersion',
    'profileDirectory',
    'stateDirectory',
  ]
  const missing = fields.filter(
    (field) => typeof config?.[field] !== 'string' || config[field].trim().length === 0,
  )
  if (missing.length > 0) return `Packaged Firefox config is missing: ${missing.join(', ')}.`
  for (const field of ['extensionPackagePath', 'firefoxExecutable']) {
    if (!existsSync(resolve(config[field])))
      return `Packaged Firefox ${field} does not exist: ${config[field]}.`
  }
  if (!existsSync(resolve(config.playwrightModule, 'package.json')))
    return `Packaged Playwright module is unavailable: ${config.playwrightModule}.`
  if (
    !Array.isArray(config.expectedResourceIdentities) ||
    config.expectedResourceIdentities.length === 0
  )
    return 'Packaged Firefox config must pin expected resource identities.'
  return undefined
}

async function waitForPackagedSetup(extensionPage, timeoutMs) {
  let status = await extensionMessage(extensionPage, { type: 'setup:status' })
  if (status.state === 'ready') return status
  await extensionMessage(extensionPage, { type: 'setup:start' })
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    status = await extensionMessage(extensionPage, { type: 'setup:status' })
    if (status.state === 'ready') return status
    if (status.state === 'failed') {
      throw new Error(
        `Packaged model setup failed: ${status.message ?? status.errorCode ?? 'unknown error'}.`,
      )
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250))
  }
  throw new Error(`Packaged model setup did not become ready within ${timeoutMs} ms.`)
}

function parseArguments(argv) {
  const options = {
    manifestPath: DEFAULT_MANIFEST_PATH,
    corpusRoot: DEFAULT_CORPUS_ROOT,
    selection: 'core',
    caseId: undefined,
    configPath: process.env.HSKIFY_REAL_READER_BROWSER_CONFIG
      ? resolve(process.env.HSKIFY_REAL_READER_BROWSER_CONFIG)
      : undefined,
    outputDirectory: DEFAULT_OUTPUT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    headed: false,
  }
  const args = [...argv]
  while (args.length > 0) {
    const argument = args.shift()
    if (argument === '--manifest') options.manifestPath = resolve(args.shift() ?? '')
    else if (argument === '--corpus') options.corpusRoot = resolve(args.shift() ?? '')
    else if (argument === '--selection') options.selection = args.shift() ?? ''
    else if (argument === '--case') options.caseId = args.shift() ?? ''
    else if (argument === '--config' || argument === '--browser-config')
      options.configPath = resolve(args.shift() ?? '')
    else if (argument === '--output') options.outputDirectory = resolve(args.shift() ?? '')
    else if (argument === '--timeout-minutes') options.timeoutMs = Number(args.shift()) * 60_000
    else if (argument === '--headed') options.headed = true
    else throw new Error(`Unknown argument: ${argument}`)
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs <= 0)
    throw new Error('--timeout-minutes must be positive.')
  return options
}

export async function runBrowserRegression(options) {
  const integrity = auditCorpus({
    manifestPath: options.manifestPath,
    corpusRoot: options.corpusRoot,
    selection: options.selection,
  })
  if (integrity.status !== 'passed')
    return jsonError('Local real-reader-v2 corpus is not release-ready.', {
      stage: 'corpus-integrity',
      captureRequired: integrity.captureRequired === true,
      integrity,
    })
  const config = options.configPath
    ? JSON.parse(readFileSync(options.configPath, 'utf8'))
    : undefined
  const configError = requiredBrowserConfig(config)
  if (configError)
    return jsonError(configError, {
      stage: 'packaged-firefox-prerequisites',
      captureRequired: false,
      integrity,
    })
  const manifest = JSON.parse(readFileSync(options.manifestPath, 'utf8'))
  process.env.HSK_MANGA_STATE_DIR = resolve(config.stateDirectory)
  const allCases = selectedCases(manifest, options.selection)
  const selected = options.caseId ? allCases.filter((item) => item.id === options.caseId) : allCases
  if (options.caseId && selected.length !== 1)
    return jsonError(`Case ${options.caseId} is not present in selection ${options.selection}.`, {
      stage: 'selection',
      captureRequired: false,
      integrity,
    })
  const chapterIds = [...new Set(selected.map((item) => item.chapterId))]
  const chapters = chapterIds
    .map((id) => manifest.chapters.find((chapter) => chapter.id === id))
    .filter(Boolean)
  mkdirSync(options.outputDirectory, { recursive: true })
  const reader = await createReaderServer({ manifest, corpusRoot: options.corpusRoot, chapters })
  let launched
  const chapterRuns = []
  const failures = []
  try {
    launched = await launchPackagedFirefox({ ...config, headed: options.headed })
    const setup = await waitForPackagedSetup(
      launched.extensionPage,
      Math.min(options.timeoutMs, 5 * 60_000),
    )
    for (const chapter of chapters) {
      const hskLevels = chapter.pages.some((page) => page.expectations?.hskDifferential === true)
        ? [2, 5]
        : [3]
      for (const hskLevel of hskLevels) {
        const chapterPage = await launched.context.newPage()
        const pageUrl = `http://127.0.0.1:${reader.port}/chapter/${encodeURIComponent(chapter.id)}?hsk=${hskLevel}`
        const startedAt = Date.now()
        try {
          await chapterPage.goto(pageUrl, { waitUntil: 'domcontentloaded' })
          await chapterPage.waitForFunction(
            () => globalThis.__hskifyReaderReady instanceof Promise,
            undefined,
            { timeout: 30_000 },
          )
          await chapterPage.evaluate(() => globalThis.__hskifyReaderReady)
          await installDomObserver(chapterPage, `real-reader-v2-${chapter.id}-hsk-${hskLevel}`)
          await prepareContentRuntime(launched.extensionPage, pageUrl)
          await extensionMessage(launched.extensionPage, { type: 'popup:prepare' })
          await startJobMonitor(
            launched.extensionPage,
            pageUrl,
            `real-reader-v2-${chapter.id}-hsk-${hskLevel}`,
          )
          const action = await timedContentStart(
            launched.extensionPage,
            hskLevel,
            pageUrl,
            'keep-original',
          )
          const state = await waitForPageState(
            launched.extensionPage,
            chapterPage,
            ['complete', 'failed', 'cancelled'],
            options.timeoutMs,
          )
          const monitor = await stopJobMonitor(launched.extensionPage)
          const dom = await chapterDomEvidence(chapterPage)
          const records = monitor.observations
          const expectedPages = chapter.pages.length
          const orderedPages = records.map((record) => record.pageIndex)
          const jobsComplete =
            records.length === expectedPages &&
            orderedPages.every((value, index) => value === index) &&
            records.every((record) => record.terminalType === 'complete')
          const regionCount = dom.regionCount
          const textCommitEvents = dom.events.filter(
            (event) => event.type === 'selectableTextDomCommitted',
          )
          const uniqueTextCommitIds = new Set(
            textCommitEvents.map((event) => event.regionId).filter(Boolean),
          )
          const duplicateTextCommitCount = Math.max(
            0,
            textCommitEvents.length - uniqueTextCommitIds.size,
          )
          const expectedTargets = chapter.pages.reduce((sum, page) => {
            const annotation = JSON.parse(
              readFileSync(resolve(options.manifestPath, '..', page.annotation.path), 'utf8'),
            )
            return sum + annotation.regions.length
          }, 0)
          const firstTextEvent = dom.events.find(
            (event) => event.type === 'selectableTextDomCommitted',
          )
          const firstFinalVisibleTextMs = firstTextEvent
            ? firstTextEvent.epochMs - action.issuedAtEpochMs
            : undefined
          const warmRun = chapterRuns.length > 0
          const run = {
            chapterId: chapter.id,
            hskLevel,
            readerKind: chapter.reader?.kind,
            pageUrl,
            pageCount: expectedPages,
            expectedStoryTargetCount: expectedTargets,
            jobCount: records.length,
            orderedPages,
            terminalState: state.state,
            regionCount,
            duplicateTextCommitCount,
            firstActionLatencyMs: action.responseAtEpochMs - action.issuedAtEpochMs,
            firstFinalVisibleTextMs,
            warmRun,
            durationMs: Date.now() - startedAt,
            setupState: setup.state,
            monitor,
            dom,
            assertions: [
              {
                id: `${chapter.id}.hsk-${hskLevel}.terminal`,
                passed: state.state === 'complete',
                expected: 'complete',
                actual: state.state,
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.ordered-complete-pages`,
                passed: jobsComplete,
                expected: `pageIndex 0..${expectedPages - 1}, all complete`,
                actual: {
                  records: records.length,
                  orderedPages,
                  terminalTypes: records.map((record) => record.terminalType),
                },
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.wrapper-count`,
                passed: dom.wrappedImageCount === expectedPages,
                expected: expectedPages,
                actual: dom.wrappedImageCount,
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.story-target-coverage`,
                passed: regionCount >= expectedTargets,
                expected: `>= ${expectedTargets} final regions`,
                actual: regionCount,
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.single-final-publication`,
                passed: dom.regions.every((region) => region.repairState !== 'pending'),
                expected: 'no pending region',
                actual: dom.regions.filter((region) => region.repairState === 'pending').length,
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.patch-before-text`,
                passed: regionCount === 0 || dom.patchBeforeText,
                expected: 'decoded patch DOM commit precedes selectable text commit',
                actual: dom.patchBeforeText,
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.single-text-commit`,
                passed: duplicateTextCommitCount === 0,
                expected: 'exactly one final text commit per region',
                actual: { textCommitEvents: textCommitEvents.length, duplicateTextCommitCount },
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.readable-fit`,
                passed:
                  dom.degradedFitCount === 0 && dom.regions.every((region) => !region.overflows),
                expected: 'all final glyphs readable without degraded fit or overflow',
                actual: {
                  degradedFitCount: dom.degradedFitCount,
                  overflowRegions: dom.regions
                    .filter((region) => region.overflows)
                    .map((region) => region.regionId),
                },
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.first-final-visible-text`,
                passed:
                  Number.isFinite(firstFinalVisibleTextMs) &&
                  firstFinalVisibleTextMs <= (warmRun ? 2_000 : 8_000),
                expected: warmRun
                  ? '<= 2,000 ms for warm packaged run'
                  : '<= 8,000 ms for cold packaged run',
                actual: firstFinalVisibleTextMs,
              },
              {
                id: `${chapter.id}.hsk-${hskLevel}.chapter-duration`,
                passed: Date.now() - startedAt < 5 * 60_000,
                expected: '< 5 minutes',
                actual: Date.now() - startedAt,
              },
            ],
          }
          if (config.expectedResourceIdentities) {
            run.route = await routeEvidence(
              launched.extensionPage,
              records,
              true,
              config.expectedResourceIdentities,
            )
            run.assertions.push({
              id: `${chapter.id}.hsk-${hskLevel}.resource-identities`,
              passed: run.route.resourceIdentityEvidence.gates.every(
                (gate) => gate.status === 'pass',
              ),
              expected: 'exact packaged resource identity set',
              actual: run.route.resourceIdentityEvidence.gates,
            })
            run.coverage = annotationCoverage(chapter, options.manifestPath, run.route)
            run.semantic = semanticConsistency(chapter, options.manifestPath, run.route)
            run.publication = publicationConsistency(dom, run.route)
            run.routeJobs = routeJobConsistency(records, run.route)
            run.assertions.push({
              id: `${chapter.id}.hsk-${hskLevel}.terminal-job-reconciliation`,
              passed: run.routeJobs.exact,
              expected: 'replayed terminal jobs exactly match browser observations in page order',
              actual: run.routeJobs,
            })
            run.assertions.push({
              id: `${chapter.id}.hsk-${hskLevel}.publication-consistency`,
              passed:
                run.publication.publishedCount === run.publication.renderedCount &&
                run.publication.missing.length === 0 &&
                run.publication.mismatched.length === 0 &&
                run.publication.duplicatePublishedIds.length === 0 &&
                run.publication.untranslatedEnglish.length === 0 &&
                run.publication.weakEvidence.length === 0,
              expected: 'terminal route text/pinyin exactly equals browser DOM with verified evidence and no extras',
              actual: run.publication,
            })
            run.assertions.push({
              id: `${chapter.id}.hsk-${hskLevel}.annotated-target-recall`,
              passed: run.coverage.matchedTargetCount === run.coverage.expectedTargetCount,
              expected: '100% annotated story-target recall',
              actual: run.coverage,
            })
            run.assertions.push({
              id: `${chapter.id}.hsk-${hskLevel}.annotated-exclusions-untouched`,
              passed: run.coverage.modifiedExclusions.length === 0,
              expected: 'zero overlays on annotated exclusions',
              actual: run.coverage.modifiedExclusions,
            })
            run.assertions.push({
              id: `${chapter.id}.hsk-${hskLevel}.ocr-cer`,
              passed:
                run.coverage.expectedTargetCount === 0 ||
                (run.coverage.ocrCer <= 0.02 &&
                  run.coverage.p95RegionCer <= 0.05 &&
                  run.coverage.highErrorRegions.length === 0),
              expected: 'global CER <= 2%, p95 region CER <= 5%, no region > 10%',
              actual: {
                ocrCer: run.coverage.ocrCer,
                p95RegionCer: run.coverage.p95RegionCer,
                highErrorRegions: run.coverage.highErrorRegions,
              },
            })
            run.assertions.push({
              id: `${chapter.id}.hsk-${hskLevel}.semantic-evidence`,
              passed:
                run.semantic.missingEntities.length === 0 &&
                run.semantic.nameViolations.length === 0 &&
                run.semantic.translatedDescriptionViolations.length === 0 &&
                run.semantic.continuationViolations.length === 0,
              expected: 'typed entities and continuation groups match reviewed chapter evidence',
              actual: run.semantic,
            })
          }
          run.status = run.assertions.every((assertion) => assertion.passed) ? 'passed' : 'failed'
          chapterRuns.push(run)
          if (run.status !== 'passed')
            failures.push(...run.assertions.filter((assertion) => !assertion.passed))
        } catch (error) {
          const failure = {
            id: `${chapter.id}.hsk-${hskLevel}.browser-run`,
            passed: false,
            expected: 'terminal packaged-Firefox run',
            actual: error instanceof Error ? error.message : String(error),
          }
          failures.push(failure)
          chapterRuns.push({
            chapterId: chapter.id,
            hskLevel,
            pageUrl,
            status: 'failed',
            assertions: [failure],
            durationMs: Date.now() - startedAt,
          })
        } finally {
          await chapterPage.close().catch(() => undefined)
        }
      }
    }
  } catch (error) {
    failures.push({
      id: 'packaged-firefox.launch',
      passed: false,
      expected: 'packaged Firefox launch',
      actual: error instanceof Error ? error.message : String(error),
    })
  } finally {
    launched?.extensionPage?.close()
    await launched?.context?.close().catch(() => undefined)
    await new Promise((resolvePromise) => reader.server.close(resolvePromise))
  }
  const differentialChapterIds = new Set(
    chapters
      .filter((chapter) =>
        chapter.pages.some((page) => page.expectations?.hskDifferential === true),
      )
      .map((chapter) => chapter.id),
  )
  for (const chapterId of differentialChapterIds) {
    const low = chapterRuns.find((run) => run.chapterId === chapterId && run.hskLevel === 2)
    const high = chapterRuns.find((run) => run.chapterId === chapterId && run.hskLevel === 5)
    const lowByRegion = new Map(
      (low?.dom?.regions ?? []).map((region) => [
        `${region.page}\u0000${region.regionId}`,
        region.text,
      ]),
    )
    const shared = (high?.dom?.regions ?? []).filter((region) =>
      lowByRegion.has(`${region.page}\u0000${region.regionId}`),
    )
    const changed = shared.filter(
      (region) => lowByRegion.get(`${region.page}\u0000${region.regionId}`) !== region.text,
    )
    const differentialAssertion = {
      id: `differential.${chapterId}.hsk-2-vs-5`,
      passed: Boolean(low && high && shared.length > 0 && changed.length > 0),
      expected: 'shared final regions with at least one level-specific translation',
      actual: {
        lowRun: Boolean(low),
        highRun: Boolean(high),
        sharedRegions: shared.length,
        changedRegions: changed.length,
      },
    }
    if (!differentialAssertion.passed) failures.push(differentialAssertion)
  }
  const summary = {
    schemaVersion: 2,
    status: failures.length === 0 ? 'passed' : 'failed',
    transport: 'packaged-firefox-local-reader',
    offline: true,
    selection: options.selection,
    corpusId: integrity.corpusId,
    chapterCount: chapters.length,
    outputDirectory: options.outputDirectory,
    chapterRuns,
    failures,
  }
  writeJsonSync(resolve(options.outputDirectory, 'summary.json'), summary)
  return summary
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : undefined
if (invokedPath === fileURLToPath(import.meta.url)) {
  try {
    const options = parseArguments(process.argv.slice(2))
    const summary = await runBrowserRegression(options)
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`)
    if (summary.status !== 'passed') process.exitCode = summary.captureRequired ? 2 : 1
  } catch (error) {
    process.stderr.write(
      `${JSON.stringify(jsonError(error instanceof Error ? error.message : String(error), { stage: 'runner' }), null, 2)}\n`,
    )
    process.exitCode = 1
  }
}
