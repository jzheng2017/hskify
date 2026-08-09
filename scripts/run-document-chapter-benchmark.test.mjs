import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'
import test from 'node:test'

import {
  createDocumentBenchmarkServer,
  mergeNativeDocumentEvidence,
  representativeChapterBlocks,
  representativeChapterMarkup,
  waitForNativeDocumentEvidence,
} from './run-document-chapter-benchmark.mjs'

function counters() {
  return {
    vision: 0,
    ocr: 0,
    projector: 0,
    segmentation: 0,
    inpainting: 0,
    patch: 0,
    font: 0,
  }
}

test('representative document fixture is exactly 300 blocks and 100,000 characters', () => {
  const blocks = representativeChapterBlocks()
  assert.equal(blocks.length, 300)
  assert.equal([...blocks.join('\n\n')].length, 100_000)
  assert.equal(blocks.every((block) => Buffer.byteLength(block, 'utf8') <= 16 * 1024), true)
  assert.equal(blocks.every((block) => /^[A-Za-z0-9 :,.]+$/u.test(block)), true)

  const markup = representativeChapterMarkup()
  assert.equal((markup.match(/<h1>/gu) ?? []).length, 1)
  assert.equal((markup.match(/<p>/gu) ?? []).length, 299)
  assert.match(markup, /data-benchmark="hskify-document-300x100000"/u)
})

test('local document benchmark server serves the representative chapter', async () => {
  const reader = await createDocumentBenchmarkServer()
  try {
    const response = await fetch(`http://127.0.0.1:${reader.port}/chapter`)
    assert.equal(response.status, 200)
    assert.match(response.headers.get('content-type') ?? '', /^text\/html; charset=utf-8$/u)

    const markup = await response.text()
    assert.equal((markup.match(/<h1>/gu) ?? []).length, 1)
    assert.equal((markup.match(/<p>/gu) ?? []).length, 299)
    assert.match(markup, /data-benchmark="hskify-document-300x100000"/u)

    const health = await fetch(`http://127.0.0.1:${reader.port}/health`)
    assert.equal(health.status, 204)
  } finally {
    await new Promise((resolvePromise) => reader.server.close(resolvePromise))
  }
})

test('native benchmark evidence merges only through an exact job id', () => {
  const merged = mergeNativeDocumentEvidence(
    [{ sampleIndex: 1, jobId: 'job-1', sourceSha256: 'a'.repeat(64) }],
    {
      samples: [{
        kind: 'document',
        jobId: 'job-1',
        tokenizerIdentity: 'tokenizer-sha256',
        dispatches: [{ reason: 'visible', itemIds: ['block-1'], tokenCount: 10 }],
        runtimeInitializations: counters(),
        runtimeInvocations: counters(),
      }],
    },
  )
  assert.equal(merged[0].nativeEvidenceMatched, true)
  assert.equal(merged[0].tokenizerIdentity, 'tokenizer-sha256')
  assert.deepEqual(merged[0].runtimeInvocations, counters())

  assert.equal(
    mergeNativeDocumentEvidence([{ sampleIndex: 1, jobId: 'different-job' }], {
      samples: [{
        kind: 'document',
        jobId: 'job-1',
        tokenizerIdentity: 'tokenizer-sha256',
        dispatches: [],
        runtimeInitializations: counters(),
        runtimeInvocations: counters(),
      }],
    })[0].nativeEvidenceMatched,
    false,
  )
})

test('duplicate native job evidence is rejected', () => {
  assert.throws(
    () => mergeNativeDocumentEvidence([], {
      samples: [
        { kind: 'document', jobId: 'job-1' },
        { kind: 'document', jobId: 'job-1' },
      ],
    }),
    /Duplicate native document jobId/u,
  )
})

test('waits through an in-progress native evidence write until a post-first dispatch exists', async () => {
  const directory = mkdtempSync(resolve(tmpdir(), 'hskify-document-evidence-'))
  const path = resolve(directory, 'native-evidence.json')
  try {
    writeFileSync(path, '{', 'utf8')
    const waiting = waitForNativeDocumentEvidence(path, 'job-1', 2_000)
    setTimeout(() => writeFileSync(path, JSON.stringify({
      samples: [{
        kind: 'document',
        jobId: 'job-1',
        tokenizerIdentity: 'tokenizer',
        dispatches: [
          { reason: 'visible', itemIds: ['one'], tokenCount: 20 },
          { reason: 'ordered', itemIds: ['two'], tokenCount: 20 },
        ],
        runtimeInitializations: counters(),
        runtimeInvocations: counters(),
      }],
    }), 'utf8'), 25)
    const sample = await waiting
    assert.equal(sample.kind, 'document')
    assert.equal(sample.dispatches.length, 2)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
