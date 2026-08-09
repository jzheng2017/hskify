import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { evaluateImageRuntimeBenchmark } from './document-chapter-gates.mjs'

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : undefined
if (invokedPath === fileURLToPath(import.meta.url)) {
  const path = process.argv[2]
  if (!path) throw new Error('usage: node image-runtime-gates.mjs <raw-samples.json>')
  const input = JSON.parse(readFileSync(path, 'utf8'))
  const result = evaluateImageRuntimeBenchmark(input.imageRuntime)
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
  if (result.status !== 'passed') process.exitCode = 1
}
