import { describe, expect, it, vi } from 'vitest'

import {
  chooseMandarinVoice,
  MandarinSpeaker,
  type SpeechRuntime,
} from '../../src/selection/speech'

function voice(
  name: string,
  lang: string,
  options: Partial<Pick<SpeechSynthesisVoice, 'default' | 'localService'>> = {},
): SpeechSynthesisVoice {
  return {
    default: options.default ?? false,
    lang,
    localService: options.localService ?? true,
    name,
    voiceURI: name,
  }
}

class TestUtterance extends EventTarget {
  lang = ''
  pitch = 1
  rate = 1
  text: string
  voice: SpeechSynthesisVoice | null = null
  volume = 1

  constructor(text: string) {
    super()
    this.text = text
  }
}

function testRuntime(voices: SpeechSynthesisVoice[] = []) {
  const utterances: TestUtterance[] = []
  const events = new EventTarget()
  const runtime: SpeechRuntime = {
    getVoices: () => voices,
    addVoicesChangedListener: (listener) =>
      events.addEventListener('voiceschanged', listener),
    removeVoicesChangedListener: (listener) =>
      events.removeEventListener('voiceschanged', listener),
    createUtterance: (text) => {
      const utterance = new TestUtterance(text)
      utterances.push(utterance)
      return utterance as unknown as SpeechSynthesisUtterance
    },
    speak: vi.fn(),
    cancel: vi.fn(),
  }
  return {
    runtime,
    utterances,
    voices,
    voicesChanged: () => events.dispatchEvent(new Event('voiceschanged')),
  }
}

describe('Mandarin speech', () => {
  it('prefers a natural Mainland Mandarin voice', () => {
    const voices = [
      voice('English Natural', 'en-US'),
      voice('Taiwan Neural', 'zh-TW'),
      voice('Mainland Desktop', 'zh-CN', { default: true }),
      voice('Mainland Natural', 'zh-CN'),
      voice('Remote Mainland Neural', 'zh-CN', { localService: false }),
    ]

    expect(chooseMandarinVoice(voices)).toBe(voices[3])
    expect(chooseMandarinVoice([voices[0]!, voices[1]!, voices[4]!])).toBeUndefined()
  })

  it('recognizes a downloaded Windows natural voice even without a quality suffix', () => {
    const desktop = voice('Microsoft Huihui Desktop', 'zh-CN', { default: true })
    const downloadedNatural = voice('Microsoft Yunxi', 'zh-CN')

    expect(chooseMandarinVoice([desktop, downloadedNatural])).toBe(downloadedNatural)
  })

  it('configures clear Mandarin playback and resets after it ends', () => {
    const selectedVoice = voice('Mainland Natural', 'zh-CN')
    const { runtime, utterances } = testRuntime([selectedVoice])
    const speaker = new MandarinSpeaker(runtime)
    const states: string[] = []
    const selectedVoices: Array<{ name: string; lang: string; localService: boolean }> = []

    speaker.toggle(' 你好！ ', (speaking, selectedVoice) => {
      states.push(speaking)
      if (selectedVoice) selectedVoices.push(selectedVoice)
    })

    expect(runtime.speak).toHaveBeenCalledTimes(1)
    expect(utterances[0]).toMatchObject({
      text: '你好！',
      voice: selectedVoice,
      lang: 'zh-CN',
      rate: 0.94,
      pitch: 1,
      volume: 1,
    })
    expect(states).toEqual(['speaking'])
    expect(selectedVoices).toEqual([
      {
        name: 'Mainland Natural',
        lang: 'zh-CN',
        localService: true,
      },
    ])

    utterances[0]?.dispatchEvent(new Event('end'))
    expect(states).toEqual(['speaking', 'idle'])
  })

  it('stops the previous playback and toggles the active control', () => {
    const mandarin = voice('Mainland', 'zh-CN')
    const { runtime } = testRuntime([mandarin])
    const speaker = new MandarinSpeaker(runtime)
    const firstStates: string[] = []
    const secondStates: string[] = []
    const firstListener = (state: string) => firstStates.push(state)
    const secondListener = (state: string) => secondStates.push(state)

    speaker.toggle('你好', firstListener)
    speaker.stop(secondListener)
    expect(runtime.cancel).not.toHaveBeenCalled()
    expect(firstStates).toEqual(['speaking'])

    speaker.toggle('再见', secondListener)

    expect(runtime.cancel).toHaveBeenCalledTimes(1)
    expect(firstStates).toEqual(['speaking', 'idle'])
    expect(secondStates).toEqual(['speaking'])

    speaker.toggle('再见', secondListener)
    expect(runtime.cancel).toHaveBeenCalledTimes(2)
    expect(secondStates).toEqual(['speaking', 'idle'])
    expect(runtime.speak).toHaveBeenCalledTimes(2)
  })

  it('reports unavailable speech without trying to play', () => {
    const speaker = new MandarinSpeaker(null)
    const state = vi.fn()

    expect(speaker.isAvailable()).toBe(false)
    speaker.toggle('你好', state)
    expect(state).toHaveBeenCalledWith('unavailable')
  })

  it('reports an asynchronous synthesis failure', () => {
    const selectedVoice = voice('Mainland', 'zh-CN')
    const { runtime, utterances } = testRuntime([selectedVoice])
    const speaker = new MandarinSpeaker(runtime)
    const states: string[] = []

    speaker.toggle('你好', (state) => states.push(state))
    utterances[0]?.dispatchEvent(new Event('error'))

    expect(states).toEqual(['speaking', 'error'])
  })

  it('waits for Firefox to load a local Mandarin voice', async () => {
    const fixture = testRuntime()
    const speaker = new MandarinSpeaker(fixture.runtime)
    const states: string[] = []

    speaker.toggle('你好', (state) => states.push(state))
    expect(states).toEqual(['loading'])

    fixture.voices.push(voice('Mainland', 'zh-CN'))
    fixture.voicesChanged()
    await vi.waitFor(() => expect(fixture.runtime.speak).toHaveBeenCalledTimes(1))
    expect(states).toEqual(['loading', 'speaking'])
  })

  it('can stop voice loading without cancelling another speech queue', async () => {
    const fixture = testRuntime()
    const speaker = new MandarinSpeaker(fixture.runtime)
    const states: string[] = []
    const listener = (state: string) => states.push(state)

    speaker.toggle('你好', listener)
    speaker.stop(listener)
    fixture.voices.push(voice('Mainland', 'zh-CN'))
    fixture.voicesChanged()
    await Promise.resolve()

    expect(states).toEqual(['loading', 'idle'])
    expect(fixture.runtime.cancel).not.toHaveBeenCalled()
    expect(fixture.runtime.speak).not.toHaveBeenCalled()
  })

  it('allows a retry when a local voice appears after the loading timeout', async () => {
    vi.useFakeTimers()
    try {
      const fixture = testRuntime()
      const speaker = new MandarinSpeaker(fixture.runtime)
      const states: string[] = []
      const listener = (state: string) => states.push(state)

      speaker.toggle('你好', listener)
      await vi.advanceTimersByTimeAsync(2_000)
      expect(states).toEqual(['loading', 'unavailable'])

      fixture.voices.push(voice('Mainland', 'zh-CN'))
      speaker.toggle('你好', listener)
      expect(states).toEqual(['loading', 'unavailable', 'speaking'])
    } finally {
      vi.useRealTimers()
    }
  })
})
