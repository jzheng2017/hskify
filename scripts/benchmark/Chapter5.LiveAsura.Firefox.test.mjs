import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildLiveTranslationProof,
  validateLiveChapterUrl,
} from './Chapter5.LiveAsura.Firefox.mjs'

test('live chapter URL is explicit, credential-free, and domain-agnostic', () => {
  assert.equal(
    validateLiveChapterUrl(
      'https://reader.example.test/series/30-years-since-the-prologue/chapter/5#page',
    ),
    'https://reader.example.test/series/30-years-since-the-prologue/chapter/5',
  )
  assert.throws(() => validateLiveChapterUrl('/chapter/5'), /explicit absolute/u)
  assert.throws(
    () => validateLiveChapterUrl('https://user:secret@reader.example.test/chapter/5'),
    /credential-free/u,
  )
})

test('translation proof correlates one English region with patch-before-text', () => {
  const update = {
    type: 'regionReady',
    region: {
      id: 'region-1',
      sourceEnglish: 'Are you ready?',
      displayedChinese: '你准备好了吗？',
      hsk: { strictlyValid: true },
      patch: { blobId: 'patch-1', mimeType: 'image/png' },
    },
  }
  const routes = {
    jobs: [{ jobId: 'job-1', pageIndex: 3, updates: [update] }],
  }
  const dom = {
    patches: [
      {
        patchId: 'patch-1',
        complete: true,
        naturalWidth: 120,
        naturalHeight: 60,
      },
    ],
    regions: [{ regionId: 'region-1', text: '你准备好了吗？', hskValid: 'true' }],
    events: [
      { index: 4, type: 'patchDomCommitted', patchId: 'patch-1' },
      { index: 5, type: 'selectableTextDomCommitted', regionId: 'region-1' },
    ],
  }
  const proof = buildLiveTranslationProof(dom, routes)
  assert.equal(proof.passed, true)
  assert.equal(proof.domOrdering.patchBeforeText, true)

  dom.events[0].index = 6
  assert.equal(buildLiveTranslationProof(dom, routes).passed, false)
})
