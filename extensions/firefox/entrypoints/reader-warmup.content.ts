import { classifyChapter } from '../src/discovery/chapter'
import { sendBackgroundMessage } from '../src/messaging/messages'

const PROBE_LIFETIME_MS = 20_000
const INSPECTION_INTERVAL_MS = 1_000

export default defineContentScript({
  matches: ['http://*/*', 'https://*/*'],
  runAt: 'document_idle',
  main() {
    let finished = false
    let scheduled: number | undefined
    let inspecting = false
    let lastInspectionStarted = Number.NEGATIVE_INFINITY
    let contentKind: 'image' | 'document' | undefined
    let discoveryActive = true

    const stopDiscovery = (): void => {
      if (!discoveryActive) return
      discoveryActive = false
      observer.disconnect()
      window.removeEventListener('load', schedule, true)
      window.clearTimeout(lifetime)
    }

    const stop = (): void => {
      if (finished) return
      finished = true
      stopDiscovery()
      if (scheduled !== undefined) window.clearTimeout(scheduled)
    }

    const inspect = async (): Promise<void> => {
      scheduled = undefined
      if (finished) return
      if (inspecting) {
        schedule()
        return
      }
      inspecting = true
      lastInspectionStarted = performance.now()
      try {
        if (!contentKind) {
          const classification = await classifyChapter(document).catch(() => undefined)
          if (finished) return
          if (classification?.kind === 'document' || classification?.kind === 'image')
            contentKind = classification.kind
          if (contentKind) stopDiscovery()
        }
        if (contentKind) {
          const status = await sendBackgroundMessage({ type: 'engine:warmup', contentKind })
          if (status.state === 'ready') stop()
        }
      } catch {
        // Setup can still be missing or in progress. Keep the detected kind
        // and retry silently so installation does not strand this page.
      } finally {
        inspecting = false
      }
      if (!finished && contentKind) schedule()
    }

    const schedule = (): void => {
      if (finished || scheduled !== undefined) return
      const delay = Math.max(
        100,
        INSPECTION_INTERVAL_MS - (performance.now() - lastInspectionStarted),
      )
      scheduled = window.setTimeout(() => void inspect(), delay)
    }

    const observer = new MutationObserver((records) => {
      if (
        records.some(
          (record) =>
            !(
              record.target instanceof Element ? record.target : record.target.parentElement
            )?.closest('[data-hskify-owned]'),
        )
      )
        schedule()
    })
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
