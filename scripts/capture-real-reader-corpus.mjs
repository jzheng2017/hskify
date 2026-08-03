import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { imageDimensions } from './real-reader-corpus.mjs'

const repositoryRoot = resolve(fileURLToPath(new URL('..', import.meta.url)))
const inputPath = resolve(process.argv[2] ?? resolve(repositoryRoot, 'temp/real-reader-capture-records.json'))
const outputPath = resolve(
  process.argv[3] ?? resolve(repositoryRoot, 'temp/real-reader-capture-result.json'),
)
const corpusRoot = resolve(repositoryRoot, 'local-corpus/real-reader-v2')
const objectRoot = resolve(corpusRoot, 'objects')

const mimeExtensions = new Map([
  ['image/jpeg', '.jpg'],
  ['image/png', '.png'],
  ['image/webp', '.webp'],
])

function normalizedMime(response, sourceUrl) {
  const header = response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase()
  if (mimeExtensions.has(header)) return header
  const lower = sourceUrl.toLowerCase()
  if (lower.includes('.png')) return 'image/png'
  if (lower.includes('.webp')) return 'image/webp'
  return 'image/jpeg'
}

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

async function capturePage(record, page, order) {
  const sourceUrl = page.url
  if (!sourceUrl || sourceUrl.startsWith('blob:')) {
    throw new Error(`browser-only source was not converted to a public response URL: ${sourceUrl}`)
  }
  const response = await fetch(sourceUrl, {
    headers: {
      accept: 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
      referer: record.actualUrl || record.url,
      'user-agent': 'Hskify local corpus capture',
    },
  })
  if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`)
  const bytes = Buffer.from(await response.arrayBuffer())
  const mimeType = normalizedMime(response, sourceUrl)
  const dimensions = imageDimensions(bytes, mimeType)
  if (!dimensions?.width || !dimensions?.height) {
    throw new Error(`unsupported or invalid ${mimeType} image response (${bytes.length} bytes)`)
  }
  const sha256 = digest(bytes)
  const extension = mimeExtensions.get(mimeType)
  const objectPath = `objects/${sha256}${extension}`
  const absolutePath = resolve(corpusRoot, objectPath)
  if (!existsSync(absolutePath)) await writeFile(absolutePath, bytes)
  return {
    order,
    sourceUrl,
    object: {
      path: objectPath,
      sha256,
      bytes: bytes.length,
      mimeType,
      width: dimensions.width,
      height: dimensions.height,
    },
  }
}

async function mapWithConcurrency(values, concurrency, callback) {
  const output = new Array(values.length)
  let next = 0
  async function worker() {
    while (true) {
      const index = next++
      if (index >= values.length) return
      output[index] = await callback(values[index], index)
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, worker))
  return output
}

const records = JSON.parse(await readFile(inputPath, 'utf8'))
await mkdir(objectRoot, { recursive: true })
const failures = []
const chapters = []
for (const record of records) {
  const pages = await mapWithConcurrency(record.images ?? [], 6, async (page, order) => {
    try {
      return await capturePage(record, page, order)
    } catch (error) {
      failures.push({ chapterId: record.id, order, sourceUrl: page.url, error: String(error) })
      return undefined
    }
  })
  chapters.push({
    id: record.id,
    actualUrl: record.actualUrl,
    title: record.title,
    readerKind: record.kind,
    capturedAtUtc: new Date().toISOString(),
    pageCount: pages.filter(Boolean).length,
    pages: pages.filter(Boolean),
  })
}

const result = {
  schemaVersion: 1,
  corpusRoot,
  capturedAtUtc: new Date().toISOString(),
  status: failures.length === 0 ? 'complete-images' : 'capture-failed',
  failures,
  chapters,
}
await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`)
process.stdout.write(`${JSON.stringify({ status: result.status, chapters: chapters.length, objects: chapters.reduce((sum, chapter) => sum + chapter.pages.length, 0), failures: failures.length, outputPath }, null, 2)}\n`)
if (failures.length > 0) process.exitCode = 1
