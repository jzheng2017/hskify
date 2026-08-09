import { mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'

import { writeJsonSync } from './browser-harness.mjs'

export const NATIVE_EVIDENCE_ENV = 'HSKIFY_BENCH_EVIDENCE_PATH'

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds))
}

export function resolveNativeEvidencePath({ configuredPath, outputDirectory }) {
  const environmentPath = process.env[NATIVE_EVIDENCE_ENV]
  const selected = environmentPath ?? configuredPath ?? resolve(outputDirectory, 'native-evidence.json')
  if (typeof selected !== 'string' || selected.trim().length === 0) {
    throw new Error(`${NATIVE_EVIDENCE_ENV} and nativeEvidencePath must be non-empty strings.`)
  }
  return resolve(selected)
}

export function initializeNativeEvidence(path) {
  mkdirSync(dirname(path), { recursive: true })
  writeJsonSync(path, { samples: [] })
  process.env[NATIVE_EVIDENCE_ENV] = path
}

export function validateNativeEvidence(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || !Array.isArray(value.samples)) {
    throw new Error('Native benchmark evidence must be an object containing a samples array.')
  }
  const seen = new Set()
  for (const sample of value.samples) {
    if (sample?.kind !== 'document' && sample?.kind !== 'image') {
      throw new Error('Every native benchmark sample must have kind document or image.')
    }
    if (typeof sample.jobId !== 'string' || sample.jobId.length === 0) {
      throw new Error('Every native benchmark sample must contain a non-empty jobId.')
    }
    const key = `${sample.kind}\0${sample.jobId}`
    if (seen.has(key)) throw new Error(`Duplicate native ${sample.kind} jobId: ${sample.jobId}.`)
    seen.add(key)
  }
  return value
}

export function readNativeEvidence(path) {
  return validateNativeEvidence(JSON.parse(readFileSync(path, 'utf8')))
}

export function nativeSample(evidence, kind, jobId) {
  return evidence.samples.find((sample) => sample.kind === kind && sample.jobId === jobId)
}

export async function waitForNativeSample(
  path,
  { kind, jobId, ready, timeoutMs, pollIntervalMs = 50 },
) {
  const deadline = Date.now() + timeoutMs
  let lastError
  while (Date.now() < deadline) {
    try {
      const sample = nativeSample(readNativeEvidence(path), kind, jobId)
      if (sample && ready(sample)) return sample
    } catch (error) {
      // The native writer replaces one small JSON file. A read can briefly
      // overlap that write, so retain the parse error and retry until timeout.
      lastError = error
    }
    await delay(pollIntervalMs)
  }
  const detail = lastError instanceof Error ? ` Last read: ${lastError.message}` : ''
  throw new Error(`Timed out waiting for native ${kind} evidence for ${jobId}.${detail}`)
}
