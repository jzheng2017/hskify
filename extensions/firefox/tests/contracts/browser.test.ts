import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  parseBrowserJobCreated,
  parseBrowserJobRequest,
  parseBrowserJobResult,
  parseBrowserJobStatus,
  parseBrowserJobStatusSequence,
  parseBrowserSetupStatus,
  parseErrorResponse,
  parseHealthResponse,
  parseLookupResult,
  parseNativeHandshakeRequest,
  parseNativeReadyResponse,
  parseRetranslateRequest,
} from '../../src/contracts/browser'

const fixtureRoot = resolve(import.meta.dirname, '../../../../fixtures/contracts')

async function fixture(name: string): Promise<unknown> {
  return JSON.parse(await readFile(resolve(fixtureRoot, name), 'utf8')) as unknown
}

describe('shared protocol v1 fixtures', () => {
  it('accepts the valid request and result', async () => {
    expect(parseBrowserJobRequest(await fixture('job-request.valid.json')).protocolVersion).toBe(1)
    expect(parseBrowserJobResult(await fixture('job-result.complete.json')).regions).toHaveLength(2)
  })

  it('accepts monotonic status sequences', async () => {
    for (const name of [
      'progress.success.json',
      'progress.failure.json',
      'progress.cancellation.json',
      'progress.reconnect.json',
    ]) {
      expect(parseBrowserJobStatusSequence(await fixture(name)).length).toBeGreaterThan(0)
    }
  })

  it('accepts native, setup, and lookup fixtures', async () => {
    expect(parseNativeHandshakeRequest(await fixture('native-request.valid.json')).type).toBe(
      'start-or-discover-daemon',
    )
    expect(parseNativeReadyResponse(await fixture('native-ready.valid.json')).type).toBe('ready')
    expect(parseHealthResponse(await fixture('health.ready.json')).status).toBe('ready')
    expect(parseBrowserJobCreated(await fixture('job-created.valid.json')).jobId).toBe(
      'fixture-job-0001',
    )
    expect(parseRetranslateRequest(await fixture('retranslate.valid.json')).settings.hskLevel).toBe(
      1,
    )
    expect(parseBrowserSetupStatus(await fixture('setup.ready.json')).state).toBe('ready')
    expect(parseLookupResult(await fixture('lookup.valid.json')).tokens[0]?.simplified).toBe('离开')
    expect(parseErrorResponse(await fixture('error.valid.json')).code).toBe('IMAGE_TOO_LARGE')
  })

  it('rejects semantic protocol violations', async () => {
    const invalidRequest = await fixture('invalid/job-request.protocol-version.json')
    const invalidResult = await fixture('invalid/job-result.out-of-range-point.json')
    const invalidStatus = await fixture('invalid/progress.terminal-mismatch.json')

    expect(() => parseBrowserJobRequest(invalidRequest)).toThrow(/protocolVersion/)
    expect(() => parseBrowserJobResult(invalidResult)).toThrow(/textPolygon/)
    expect(() => parseBrowserJobStatus(invalidStatus)).toThrow(/stage/)
  })
})
