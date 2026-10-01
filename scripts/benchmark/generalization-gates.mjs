import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const REDESIGN_FINGERPRINT = 'hskify-windows-x86_64-msvc-cuda13.1-sm89-2026-10-01-r10'
const modalities = ['image', 'document']
const modes = ['natural', ...Array.from({ length: 6 }, (_, i) => `strict-${i + 1}`)]
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex')
const groupBy = (values, key) => {
  const groups = new Map()
  for (const value of values) {
    const name = key(value)
    if (!groups.has(name)) groups.set(name, [])
    groups.get(name).push(value)
  }
  return groups
}

/** An annotation is evidence only when its frozen bytes and source identity match. */
export function evaluateGeneralization(manifest, evidence, readAnnotation) {
  const failures = []
  const require = (condition, message) => { if (!condition) failures.push(message) }
  require(manifest.schemaVersion === 1, 'Manifest schema must be 1.')
  require(evidence.buildFingerprint === REDESIGN_FINGERPRINT, 'Evaluation must use the current packaged build.')
  const chapters = Array.isArray(manifest.chapters) ? manifest.chapters : []
  require(new Set(chapters.map(c => c.id)).size === chapters.length, 'Chapter IDs must be unique.')
  for (const modality of modalities) {
    const minimum = modality === 'image' ? 13 : 12
    const development = chapters.filter(c => c.modality === modality && c.split === 'development')
    const blind = chapters.filter(c => c.modality === modality && c.split === 'blind')
    require(development.length >= minimum && blind.length >= minimum, `${modality}: requires ${minimum} development and ${minimum} blind chapters.`)
    const providers = new Set(development.map(c => c.provider))
    const series = new Set(development.map(c => c.series))
    require(new Set(blind.map(c => c.provider)).size >= 3, `${modality}: requires at least three blind providers.`)
    require(blind.every(c => !providers.has(c.provider) && !series.has(c.series)), `${modality}: whole providers and series must be held out.`)
    require(chapters.some(c => c.modality === modality && c.negativeControl === true), `${modality}: missing negative controls.`)
  }
  const annotations = new Map()
  for (const chapter of chapters) {
    require(modalities.includes(chapter.modality) && ['development', 'blind'].includes(chapter.split), `${chapter.id}: invalid modality or split.`)
    require(['id', 'provider', 'series', 'readerClass'].every(key => typeof chapter[key] === 'string' && chapter[key].length > 0), `${chapter.id}: incomplete chapter identity.`)
    require(['artwork', 'font', 'genre', 'layout'].every(key => typeof chapter.strata?.[key] === 'string' && chapter.strata[key].length > 0), `${chapter.id}: missing evaluation strata.`)
    try {
      const frozenAt = Date.parse(chapter.annotation?.frozenAt)
      const evaluatedAt = Date.parse(evidence.evaluatedAt)
      require(Number.isFinite(frozenAt) && frozenAt < evaluatedAt, `${chapter.id}: annotations must be frozen before evaluation.`)
      const bytes = readAnnotation(chapter.annotation.path)
      require(sha256(bytes) === chapter.annotation.sha256, `${chapter.id}: frozen annotation digest mismatch.`)
      const annotation = JSON.parse(String(bytes))
      require(annotation.chapterId === chapter.id && /^[a-f0-9]{64}$/u.test(annotation.sourceSha256 ?? ''), `${chapter.id}: invalid annotated source identity.`)
      require(Array.isArray(annotation.items) && new Set(annotation.items.map(item => item.id)).size === annotation.items.length, `${chapter.id}: annotation items must be unique.`)
      annotations.set(chapter.id, annotation)
    } catch (error) { failures.push(`${chapter.id}: annotation unavailable: ${error.message}`) }
  }
  const runs = Array.isArray(evidence.runs) ? evidence.runs : []
  const byRun = groupBy(runs, run => `${run.chapterId}/${run.mode}`)
  const rows = []
  for (const chapter of chapters) for (const mode of modes) {
    const matches = byRun.get(`${chapter.id}/${mode}`) ?? []
    require(matches.length === 1, `${chapter.id}/${mode}: requires exactly one evaluation run.`)
    const run = matches[0], annotation = annotations.get(chapter.id)
    if (!run || !annotation || !Array.isArray(annotation.items)) continue
    require(run.sourceSha256 === annotation.sourceSha256, `${chapter.id}/${mode}: source differs from the frozen annotation.`)
    const results = Array.isArray(run.items) ? run.items : []
    const byItem = groupBy(results, item => item.id)
    const known = new Set(annotation.items.map(item => item.id))
    require(results.every(item => known.has(item.id)), `${chapter.id}/${mode}: contains unannotated results.`)
    for (const gold of annotation.items) {
      const matches = byItem.get(gold.id) ?? []
      if (!gold.eligible) {
        require(matches.every(item => item.disposition === 'excluded'), `${chapter.id}/${mode}/${gold.id}: translated an ineligible control.`)
        continue
      }
      require(matches.length === 1, `${chapter.id}/${mode}/${gold.id}: silent omission or duplicate result.`)
      const item = matches[0]
      if (!item) continue
      require(['translated', 'failed'].includes(item.disposition), `${chapter.id}/${mode}/${gold.id}: eligible content cannot be excluded.`)
      const translated = item.disposition === 'translated'
      const assessors = item.assessment?.reviewers
      const independent = item.assessment?.method === 'independent-bilingual' && Array.isArray(assessors) && new Set(assessors).size >= 2 && assessors.every(id => typeof id === 'string' && id.length > 0)
      require(!translated || independent, `${chapter.id}/${mode}/${gold.id}: missing independent bilingual assessment.`)
      require(!translated || typeof item.assessment?.faithful === 'boolean' && typeof item.assessment?.grammarAppropriate === 'boolean', `${chapter.id}/${mode}/${gold.id}: incomplete semantic or grammar assessment.`)
      require(!translated || mode === 'natural' || item.strictLexicalValid === true, `${chapter.id}/${mode}/${gold.id}: strict lexical violation.`)
      require(item.restorationCorrupted === false && item.staleOverlay === false && Number.isInteger(item.renderingDefects) && item.renderingDefects >= 0, `${chapter.id}/${mode}/${gold.id}: missing source and rendering checks.`)
      require(!item.restorationCorrupted && !item.staleOverlay, `${chapter.id}/${mode}/${gold.id}: source integrity failure.`)
      require(!gold.critical || translated && independent && item.assessment?.faithful === true, `${chapter.id}/${mode}/${gold.id}: critical semantic regression.`)
      if (chapter.modality === 'image') require(Number.isInteger(item.ocr?.referenceCharacters) && item.ocr.referenceCharacters > 0 && Number.isInteger(item.ocr?.editDistance) && item.ocr.editDistance >= 0, `${chapter.id}/${mode}/${gold.id}: missing OCR error measurement.`)
      rows.push({ ...item, translated, faithful: translated && independent && item.assessment?.faithful === true,
        grammarAppropriate: translated && independent && item.assessment?.grammarAppropriate === true,
        group: `${chapter.split}/${chapter.modality}/${chapter.readerClass}/${mode}` })
    }
  }
  const metrics = []
  for (const [group, items] of groupBy(rows, row => row.group)) {
    const translated = items.filter(item => item.translated)
    const coverage = translated.length / items.length
    const faithfulRate = translated.length ? translated.filter(item => item.faithful).length / translated.length : 0
    require(coverage >= (group.endsWith('/natural') ? .95 : .90), `${group}: translated coverage below target.`)
    require(faithfulRate >= .95, `${group}: faithful accepted translations below 95%.`)
    const reference = items.reduce((sum, item) => sum + (item.ocr?.referenceCharacters ?? 0), 0)
    metrics.push({ group, eligible: items.length, translated: translated.length, coverage, faithfulRate,
      abstentionRate: 1 - coverage,
      grammarAppropriateRate: translated.length ? translated.filter(item => item.grammarAppropriate).length / translated.length : 0,
      strictLexicalCompliance: translated.length ? translated.filter(item => item.strictLexicalValid === true).length / translated.length : 0,
      ocrCharacterErrorRate: reference ? items.reduce((sum, item) => sum + item.ocr.editDistance, 0) / reference : undefined,
      renderingDefects: items.reduce((sum, item) => sum + item.renderingDefects, 0) })
  }
  require(metrics.length > 0, 'No evaluated eligible story items.')
  return { status: failures.length ? 'fail' : 'pass', failures, metrics }
}

export function firstReadableLatency(sample) {
  const events = sample.events?.filter(event => event.performanceMs >= sample.startPerformanceMs && event.visible === true && event.readable === true &&
    (event.type === 'selectableTextDomCommitted' && event.sourcePreserving !== true || event.type === 'documentBlockDomCommitted' && event.state === 'translated'))
  return events?.length ? events.reduce((earliest, event) => Math.min(earliest, event.performanceMs), Infinity) - sample.startPerformanceMs : Infinity
}

/** Resident latency is derived from committed selectable viewport text, never dispatch timestamps. */
export function evaluateVisibleLatency(samples) {
  const failures = [], metrics = []
  for (const modality of modalities) for (const mode of ['natural', 'strict']) {
    const all = samples.filter(sample => sample.modality === modality && sample.mode === mode)
    const warm = all.filter(sample => sample.cacheState !== 'cold')
    const key = `${modality}/${mode}`
    if (warm.length < 30) failures.push(`${key}: requires at least 30 warm samples.`)
    for (const state of ['cold', 'resident-miss', 'partial-hit', 'full-hit']) if (!all.some(sample => sample.cacheState === state)) failures.push(`${key}: missing ${state} measurements.`)
    if (!warm.some(sample => sample.startPosition === 'middle')) failures.push(`${key}: missing starts halfway through chapters.`)
    if (new Set(all.map(sample => sample.id)).size !== all.length) failures.push(`${key}: sample IDs must be unique.`)
    for (const sample of all) {
      if (sample.extensionFingerprint !== REDESIGN_FINGERPRINT || sample.nativeFingerprint !== REDESIGN_FINGERPRINT || sample.browser?.name !== 'firefox' || !sample.browser.version || sample.gpu !== 'NVIDIA GeForce RTX 4080 SUPER' || sample.packaged !== true || sample.buildAttestationVerified !== true) failures.push(`${sample.id}: unverified packaged Firefox/native/hardware identity.`)
      if (!Number.isFinite(sample.startPerformanceMs) || !Number.isFinite(firstReadableLatency(sample))) failures.push(`${sample.id}: no measured first readable result.`)
      const completions = sample.itemCompletions ?? []
      const expected = sample.sourceItems ?? []
      const byItem = groupBy(completions, item => item.id)
      const expectedIds = new Set(expected.map(item => item.id))
      if (!expected.length || expectedIds.size !== expected.length ||
        expected.some(item => typeof item.id !== 'string' || !item.id || typeof item.visibleAtStart !== 'boolean' ||
          byItem.get(item.id)?.length !== 1 || byItem.get(item.id)?.[0].visibleAtStart !== item.visibleAtStart) ||
        completions.some(item => !expectedIds.has(item.id))) failures.push(`${sample.id}: incomplete source/completion accounting.`)
      const visible = completions.filter(item => item.visibleAtStart)
      const background = completions.filter(item => !item.visibleAtStart)
      if (!visible.length || completions.some(item => !Number.isFinite(item.performanceMs)) || background.some(item => item.performanceMs < visible.reduce((latest, item) => Math.max(latest, item.performanceMs), -Infinity))) failures.push(`${sample.id}: visible work did not finish before background work.`)
    }
    const sorted = warm.map(firstReadableLatency).sort((a, b) => a - b)
    const p95Ms = sorted.length ? sorted[Math.ceil(sorted.length * .95) - 1] : Infinity
    if (p95Ms > (mode === 'natural' ? 5_000 : 10_000)) failures.push(`${key}: p95 first-readable-result target exceeded.`)
    metrics.push({ group: key, warmSamples: warm.length, p95Ms, cacheStates: [...groupBy(all, sample => sample.cacheState)].map(([cacheState, samples]) => ({ cacheState, samples: samples.length, firstReadableMs: samples.map(firstReadableLatency) })) })
  }
  return { status: failures.length ? 'fail' : 'pass', failures, metrics }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [, , manifestPath, evidencePath, outputPath] = process.argv
  if (!manifestPath || !evidencePath || !outputPath) throw new Error('Usage: node scripts/benchmark/generalization-gates.mjs MANIFEST EVIDENCE OUTPUT')
  const evidence = JSON.parse(readFileSync(evidencePath, 'utf8'))
  const quality = evaluateGeneralization(JSON.parse(readFileSync(manifestPath, 'utf8')), evidence, path => readFileSync(resolve(dirname(manifestPath), path)))
  const latency = evaluateVisibleLatency(evidence.latencySamples ?? [])
  const result = { status: quality.status === 'pass' && latency.status === 'pass' ? 'pass' : 'fail', quality, latency }
  writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n')
  console.log(JSON.stringify({ status: result.status, qualityFailures: quality.failures.length, latencyFailures: latency.failures.length }))
  if (result.status !== 'pass') process.exitCode = 1
}
