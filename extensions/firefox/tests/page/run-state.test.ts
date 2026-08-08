import { describe, expect, it } from 'vitest'

import { ChapterRunState } from '../../src/page/run-state'

describe('chapter image run state', () => {
  it('derives chapter completion from every registered image reaching a terminal state', () => {
    const state = new ChapterRunState<string>()
    expect(state.register('first')).toBe(true)
    expect(state.register('second')).toBe(true)
    expect(state.register('first')).toBe(false)

    state.start('first')
    state.complete('first')
    expect(state.snapshot()).toMatchObject({
      total: 2,
      completed: 1,
      unresolved: 1,
      allResolved: false,
    })

    state.start('second')
    state.fail('second')
    expect(state.snapshot()).toEqual({
      total: 2,
      queued: 0,
      running: 0,
      completed: 1,
      failed: 1,
      resolved: 2,
      unresolved: 0,
      allResolved: true,
    })
  })

  it('allows an explicit retry only after an image has failed', () => {
    const state = new ChapterRunState<string>()
    state.register('image')
    expect(state.manualRetryQueued('image')).toBe(false)
    state.start('image')
    state.fail('image')

    expect(state.manualRetryQueued('image')).toBe(true)
    expect(state.phase('image')).toBe('queued')
  })

  it('rejects contradictory lifecycle transitions instead of corrupting counters', () => {
    const state = new ChapterRunState<string>()
    state.register('image')
    expect(() => state.complete('image')).toThrow(/queued/u)
    state.start('image')
    expect(() => state.start('image')).toThrow(/running/u)
    state.complete('image')
    expect(() => state.fail('image')).toThrow(/complete/u)
  })
})
