import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildLiveTranslationProof,
  resolveExpectedImageCount,
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

test('live chapter image count is reader-derived unless explicitly pinned', () => {
  assert.equal(resolveExpectedImageCount(undefined, 21), 21)
  assert.equal(resolveExpectedImageCount(21, 21), 21)
  assert.throws(() => resolveExpectedImageCount(undefined, 0), /no translatable/u)
  assert.throws(() => resolveExpectedImageCount(1.5, 21), /positive integer/u)
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

test('translation proof records a non-strict HSK result without hiding browser completion', () => {
  const update = {
    type: 'regionReady',
    region: {
      id: 'region-literary',
      sourceEnglish: 'A goddess descended from another realm.',
      displayedChinese: '一位女神从另一个世界来到这里。',
      hsk: {
        strictlyValid: false,
        repairState: 'rejected',
        aboveLevelTokens: ['女神'],
      },
      patch: { blobId: 'patch-literary', mimeType: 'image/png' },
    },
  }
  const proof = buildLiveTranslationProof(
    {
      patches: [
        {
          patchId: 'patch-literary',
          complete: true,
          naturalWidth: 120,
          naturalHeight: 60,
        },
      ],
      regions: [
        {
          regionId: 'region-literary',
          text: '一位女神从另一个世界来到这里。',
          hskValid: 'false',
        },
      ],
      events: [
        { index: 1, type: 'patchDomCommitted', patchId: 'patch-literary' },
        {
          index: 2,
          type: 'selectableTextDomCommitted',
          regionId: 'region-literary',
        },
      ],
    },
    {
      jobs: [
        {
          jobId: 'job-literary',
          pageIndex: 0,
          updates: [
            {
              ...update,
              region: {
                ...update.region,
                hsk: {
                  strictlyValid: false,
                  repairState: 'pending',
                  aboveLevelTokens: ['å¥³ç¥ž'],
                },
              },
            },
            {
              type: 'regionRefined',
              regionId: 'region-literary',
              displayedChinese: 'ä¸€ä½å¥³ç¥žä»Žå¦ä¸€ä¸ªä¸–ç•Œæ¥åˆ°è¿™é‡Œã€‚',
              hsk: {
                strictlyValid: false,
                repairState: 'rejected',
                aboveLevelTokens: ['å¥³ç¥ž'],
              },
            },
          ],
        },
      ],
    },
  )

  assert.equal(proof.passed, true)
  assert.equal(proof.hskStrictlyValid, false)
  assert.equal(proof.hskAssessment.repairState, 'rejected')
  assert.equal(proof.hskAssessment.aboveLevelTokens.length, 1)
})
