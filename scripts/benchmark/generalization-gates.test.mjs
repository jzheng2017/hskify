import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import test from 'node:test'
import { evaluateGeneralization, evaluateVisibleLatency, firstReadableLatency, REDESIGN_FINGERPRINT } from './generalization-gates.mjs'

function fixture() {
  const chapters = [], runs = [], files = new Map()
  for (const modality of ['image', 'document']) for (const split of ['development', 'blind']) {
    for (let index = 0; index < (modality === 'image' ? 13 : 12); index++) {
      const id = `${modality}-${split}-${index}`, path = `${id}.json`
      const annotation = { chapterId: id, sourceSha256: 'a'.repeat(64), items: [{ id: 'item', eligible: index !== 0, critical: index === 1 }] }
      const bytes = Buffer.from(JSON.stringify(annotation)); files.set(path, bytes)
      chapters.push({ id, modality, split, provider: `${modality}-${split}-provider-${index % 3}`, series: id, readerClass: 'reader',
        negativeControl: index === 0, strata: { artwork: 'distinct', font: 'font', genre: 'genre', layout: 'layout' },
        annotation: { path, sha256: createHash('sha256').update(bytes).digest('hex'), frozenAt: '2026-09-30T00:00:00Z' } })
      for (const mode of ['natural', ...Array.from({ length: 6 }, (_, i) => `strict-${i + 1}`)]) runs.push({ chapterId: id, mode, sourceSha256: annotation.sourceSha256,
        items: [{ id: 'item', disposition: index === 0 ? 'excluded' : 'translated', strictLexicalValid: true,
          assessment: { method: 'independent-bilingual', reviewers: ['human-a', 'human-b'], faithful: true, grammarAppropriate: true },
          restorationCorrupted: false, staleOverlay: false, renderingDefects: 0, ocr: { referenceCharacters: 100, editDistance: 1 } }] })
    }
  }
  return { manifest: { schemaVersion: 1, chapters }, evidence: { buildFingerprint: REDESIGN_FINGERPRINT, evaluatedAt: '2026-10-01T00:00:00Z', runs }, read: path => files.get(path) }
}
function evaluate(value) { return evaluateGeneralization(value.manifest, value.evidence, value.read) }

test('quality gates accept complete frozen development and blind evidence in every language mode', () => {
  const result = evaluate(fixture())
  assert.equal(result.status, 'pass', JSON.stringify(result.failures))
  assert.equal(result.metrics.length, 28)
  assert.equal(result.metrics.every(metric => metric.coverage === 1 && metric.faithfulRate === 1), true)
})
test('quality gates reject provider leakage, annotation changes and incomplete independent review', () => {
  const value = fixture()
  value.manifest.chapters[13].provider = value.manifest.chapters[0].provider
  value.manifest.chapters[1].annotation.sha256 = 'b'.repeat(64)
  value.evidence.runs[7].items[0].assessment.method = 'model-self-score'
  const result = evaluate(value)
  assert.equal(result.status, 'fail')
  assert.ok(result.failures.some(message => message.includes('held out')))
  assert.ok(result.failures.some(message => message.includes('digest mismatch')))
  assert.ok(result.failures.some(message => message.includes('independent bilingual')))
})
test('silent omissions, stale output, strict violations and critical semantic failures cannot pass', () => {
  const value = fixture()
  value.evidence.runs[7].items = []
  value.evidence.runs[8].items[0].strictLexicalValid = false
  value.evidence.runs[9].items[0].staleOverlay = true
  value.evidence.runs[10].items[0].assessment.faithful = false
  const failures = evaluate(value).failures.join('\n')
  for (const message of ['silent omission', 'strict lexical violation', 'source integrity', 'critical semantic']) assert.ok(failures.includes(message))
})
test('an empty corpus cannot establish quality', () => {
  assert.equal(evaluateGeneralization({ schemaVersion: 1, chapters: [] }, { runs: [] }, () => Buffer.from('{}')).status, 'fail')
})

function latencyFixture() {
  const samples = []
  for (const modality of ['image', 'document']) for (const mode of ['natural', 'strict']) for (let i = 0; i <= 30; i++) {
    const performanceMs = 100 + (mode === 'natural' ? 4_000 : 9_000)
    samples.push({ id: `${modality}/${mode}/${i}`, modality, mode, cacheState: i === 30 ? 'cold' : ['resident-miss', 'partial-hit', 'full-hit'][i % 3],
      startPosition: i % 2 ? 'middle' : 'top', startPerformanceMs: 100, gpu: 'NVIDIA GeForce RTX 4080 SUPER', packaged: true, buildAttestationVerified: true,
      extensionFingerprint: REDESIGN_FINGERPRINT, nativeFingerprint: REDESIGN_FINGERPRINT, browser: { name: 'firefox', version: '151.0' },
      events: [{ type: modality === 'image' ? 'selectableTextDomCommitted' : 'documentBlockDomCommitted', state: 'translated', performanceMs, visible: true, readable: true }],
      sourceItems: [{id: "visible", visibleAtStart: true}, {id: "background", visibleAtStart: false}],
      itemCompletions: [{id: "visible", visibleAtStart: true, performanceMs }, {id: "background", visibleAtStart: false, performanceMs: performanceMs + 100 }] })
  }
  return samples
}
test('latency gates use actual visible commits for both modalities and modes', () => {
  const result = evaluateVisibleLatency(latencyFixture())
  assert.equal(result.status, 'pass', JSON.stringify(result.failures))
  assert.deepEqual(result.metrics.map(metric => metric.p95Ms), [4_000, 9_000, 4_000, 9_000])
})
test('dispatches, pending skeletons, failures and offscreen Chinese do not count as readable text', () => {
  const sample = latencyFixture()[0]
  sample.events = [
    { type: 'dispatch', performanceMs: 101, visible: true, readable: true },
    { type: 'documentBlockDomCommitted', state: 'pending', performanceMs: 102, visible: true, readable: true },
    { type: 'selectableTextDomCommitted', sourcePreserving: true, performanceMs: 103, visible: true, readable: true },
    { type: 'selectableTextDomCommitted', performanceMs: 104, visible: false, readable: true },
  ]
  assert.equal(firstReadableLatency(sample), Infinity)
  assert.equal(evaluateVisibleLatency([sample]).status, 'fail')
})
test('latency gates reject missing warm samples, mismatched packages and background work finishing first', () => {
  const samples = latencyFixture()
  samples[0].nativeFingerprint = 'old'
  samples[1].itemCompletions[1].performanceMs = 101
  const failures = evaluateVisibleLatency(samples.slice(1)).failures.join('\n')
  assert.ok(failures.includes('30 warm'))
  assert.ok(failures.includes('visible work'))
  assert.equal(evaluateVisibleLatency(samples).status, 'fail')
})

test('latency gates reject omitted viewport work and use the earliest commit even if evidence is reordered', () => {
  const samples = latencyFixture()
  samples[0].itemCompletions.shift()
  assert.ok(evaluateVisibleLatency(samples).failures.some(message => message.includes('accounting')))
  const sample = latencyFixture()[0]
  sample.events.push({...sample.events[0], performanceMs: 200})
  assert.equal(firstReadableLatency(sample), 100)
})
