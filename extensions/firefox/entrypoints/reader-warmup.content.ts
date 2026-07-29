import { looksLikeSequentialArtReader } from '../src/discovery/images'
import { sendBackgroundMessage } from '../src/messaging/messages'

const PROBE_LIFETIME_MS = 20_000

export default defineContentScript({
  matches: ['http://*/*', 'https://*/*'],
  runAt: 'document_idle',
  main() {
    let finished = false
    let scheduled: number | undefined

    const stop = (): void => {
      if (finished) return
      finished = true
      observer.disconnect()
      window.removeEventListener('load', schedule, true)
      if (scheduled !== undefined) window.clearTimeout(scheduled)
      window.clearTimeout(lifetime)
    }

    const inspect = (): void => {
      scheduled = undefined
      if (finished || !looksLikeSequentialArtReader()) return
      stop()
      void sendBackgroundMessage({ type: 'engine:warmup' }).catch(() => {
        // The popup exposes actionable setup errors. A speculative page probe
        // remains silent so ordinary browsing is never interrupted.
      })
    }

    const schedule = (): void => {
      if (finished || scheduled !== undefined) return
      scheduled = window.setTimeout(inspect, 100)
    }

    const observer = new MutationObserver(schedule)
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'data-src', 'data-url', 'style', 'class'],
    })
    window.addEventListener('load', schedule, true)
    const lifetime = window.setTimeout(stop, PROBE_LIFETIME_MS)
    schedule()
  },
})
