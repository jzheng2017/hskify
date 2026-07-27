export type SpeechState = 'idle' | 'loading' | 'speaking' | 'unavailable' | 'error'
export type SpeechVoiceMetadata = {
  name: string
  lang: string
  localService: boolean
}
export type SpeechStateListener = (
  state: SpeechState,
  voice?: SpeechVoiceMetadata,
) => void

export type SpeechRuntime = {
  getVoices(): SpeechSynthesisVoice[]
  addVoicesChangedListener(listener: () => void): void
  removeVoicesChangedListener(listener: () => void): void
  createUtterance(text: string): SpeechSynthesisUtterance
  speak(utterance: SpeechSynthesisUtterance): void
  cancel(): void
}

export type TextSpeaker = {
  isAvailable(): boolean
  toggle(text: string, onStateChange: SpeechStateListener): void
  stop(owner?: SpeechStateListener): void
}

type ActivePlayback = {
  text: string
  utterance: SpeechSynthesisUtterance | undefined
  cancelLoading: (() => void) | undefined
  onStateChange: SpeechStateListener
}

function normalizedLanguage(voice: SpeechSynthesisVoice): string {
  return voice.lang.trim().replaceAll('_', '-').toLowerCase()
}

function voiceScore(voice: SpeechSynthesisVoice): number {
  if (!voice.localService) return Number.NEGATIVE_INFINITY
  const language = normalizedLanguage(voice)
  let score = Number.NEGATIVE_INFINITY
  try {
    const locale = new Intl.Locale(language).maximize()
    if (
      (locale.language === 'zh' || locale.language === 'cmn') &&
      locale.script === 'Hans' &&
      locale.region === 'CN'
    ) {
      score = language === 'zh-cn' || language === 'cmn-cn' ? 4_000 : 3_900
    }
  } catch {
    if (
      /^(?:zh|cmn)(?:-(?:hans-)?cn|-cn-hans)?$/u.test(language) ||
      /^(?:zh|cmn)-hans$/u.test(language)
    ) {
      score = 3_800
    }
  }
  if (!Number.isFinite(score)) return score

  const name = `${voice.name} ${voice.voiceURI}`.toLowerCase()
  // Windows' downloadable natural voices are sometimes exposed by Firefox
  // without a "Natural" or "Neural" suffix. Prefer their stable voice names
  // over the older Desktop/SAPI Mandarin voices when both are installed.
  if (
    /\b(?:xiaohan|xiaomeng|xiaomo|xiaoqiu|xiaorui|xiaoshuang|xiaoxiao|xiaoxuan|xiaoyan|xiaoyi|xiaoyou|yunfeng|yunhao|yunjian|yunxia|yunxi|yunyang|yunze)\b/u.test(
      name,
    )
  ) {
    score += 500
  }
  if (/\b(?:natural|neural)\b/u.test(name)) score += 350
  if (/\bpremium\b/u.test(name)) score += 250
  if (/\benhanced\b/u.test(name)) score += 200
  if (/\bmandarin\b/u.test(name) || /普通话|普通話|國語|国语/u.test(name)) score += 100
  if (voice.default) score += 25
  return score
}

export function chooseMandarinVoice(
  voices: readonly SpeechSynthesisVoice[],
): SpeechSynthesisVoice | undefined {
  let best: SpeechSynthesisVoice | undefined
  let bestScore = Number.NEGATIVE_INFINITY
  for (const voice of voices) {
    const score = voiceScore(voice)
    if (score > bestScore) {
      best = voice
      bestScore = score
    }
  }
  return best
}

function browserSpeechRuntime(): SpeechRuntime | null {
  if (
    typeof window === 'undefined' ||
    typeof window.speechSynthesis === 'undefined' ||
    typeof SpeechSynthesisUtterance === 'undefined'
  ) {
    return null
  }
  const synthesis = window.speechSynthesis
  return {
    getVoices: () => synthesis.getVoices(),
    addVoicesChangedListener: (listener) =>
      synthesis.addEventListener('voiceschanged', listener),
    removeVoicesChangedListener: (listener) =>
      synthesis.removeEventListener('voiceschanged', listener),
    createUtterance: (text) => new SpeechSynthesisUtterance(text),
    speak: (utterance) => synthesis.speak(utterance),
    cancel: () => synthesis.cancel(),
  }
}

export class MandarinSpeaker implements TextSpeaker {
  private active: ActivePlayback | undefined

  constructor(private readonly runtime: SpeechRuntime | null = browserSpeechRuntime()) {}

  isAvailable(): boolean {
    return this.runtime !== null
  }

  toggle(text: string, onStateChange: SpeechStateListener): void {
    const spokenText = text.trim()
    if (!this.runtime || !spokenText) {
      onStateChange('unavailable')
      return
    }
    if (
      this.active?.text === spokenText &&
      this.active.onStateChange === onStateChange
    ) {
      this.stop()
      return
    }

    this.stop()
    const active: ActivePlayback = {
      text: spokenText,
      utterance: undefined,
      cancelLoading: undefined,
      onStateChange,
    }
    this.active = active
    const voice = this.currentVoice()
    if (voice) {
      this.start(active, voice)
      return
    }
    onStateChange('loading')
    void this.waitForVoice(active).then((loadedVoice) => {
      if (this.active !== active) return
      if (!loadedVoice) {
        this.active = undefined
        onStateChange('unavailable')
        return
      }
      this.start(active, loadedVoice)
    })
  }

  stop(owner?: SpeechStateListener): void {
    const active = this.active
    if (!active || !this.runtime) return
    if (owner && active.onStateChange !== owner) return
    this.active = undefined
    active.cancelLoading?.()
    try {
      if (active.utterance) this.runtime.cancel()
    } finally {
      active.onStateChange('idle')
    }
  }

  private currentVoice(): SpeechSynthesisVoice | undefined {
    if (!this.runtime) return undefined
    try {
      return chooseMandarinVoice(this.runtime.getVoices())
    } catch {
      return undefined
    }
  }

  private waitForVoice(active: ActivePlayback): Promise<SpeechSynthesisVoice | undefined> {
    const runtime = this.runtime
    if (!runtime) return Promise.resolve(undefined)
    return new Promise((resolve) => {
      let settled = false
      let timeout: ReturnType<typeof globalThis.setTimeout> | undefined
      const finish = (voice: SpeechSynthesisVoice | undefined): void => {
        if (settled) return
        settled = true
        if (timeout !== undefined) globalThis.clearTimeout(timeout)
        try {
          runtime.removeVoicesChangedListener(changed)
        } catch {
          // The owning window may have gone away while voices were loading.
        }
        active.cancelLoading = undefined
        resolve(voice)
      }
      const changed = (): void => {
        const voice = this.currentVoice()
        if (voice) finish(voice)
      }
      active.cancelLoading = () => finish(undefined)
      try {
        runtime.addVoicesChangedListener(changed)
      } catch {
        finish(undefined)
        return
      }
      timeout = globalThis.setTimeout(() => finish(this.currentVoice()), 2_000)
      changed()
    })
  }

  private start(active: ActivePlayback, voice: SpeechSynthesisVoice): void {
    const runtime = this.runtime
    if (!runtime || this.active !== active) return
    let utterance: SpeechSynthesisUtterance
    try {
      utterance = runtime.createUtterance(active.text)
      utterance.voice = voice
      utterance.lang = voice.lang || 'zh-CN'
      utterance.rate = 0.94
      utterance.pitch = 1
      utterance.volume = 1
    } catch {
      this.active = undefined
      active.onStateChange('error')
      return
    }

    const finish = (state: 'idle' | 'error'): void => {
      if (this.active !== active) return
      this.active = undefined
      active.onStateChange(state)
    }
    utterance.addEventListener('end', () => finish('idle'), { once: true })
    utterance.addEventListener('error', () => finish('error'), { once: true })
    active.utterance = utterance
    active.onStateChange('speaking', {
      name: voice.name,
      lang: voice.lang,
      localService: voice.localService,
    })
    try {
      runtime.speak(utterance)
    } catch {
      if (this.active === active) {
        this.active = undefined
        active.onStateChange('error')
      }
    }
  }
}
