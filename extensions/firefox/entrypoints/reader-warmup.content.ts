import { looksLikeSequentialArtReader } from '../src/discovery/images'
import { discoverPageSurfaces } from '../src/discovery/surfaces'
import { detectDocumentChapter } from '../src/document/extraction'
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
    let documentInspected = false
    let documentRejected = false

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
          if (!documentInspected) {
            documentInspected = true
            const documentDetection = await detectDocumentChapter(document).catch(() => undefined)
            if (finished) return
            if (documentDetection?.kind === 'document') contentKind = 'document'
            documentRejected = documentDetection?.kind === 'rejected'
          }
          if (
            !contentKind &&
            !documentRejected &&
            (looksLikeSequentialArtReader() || discoverPageSurfaces().surfaces.some(
                (surface) => surface.kind !== 'image' &&
                  (surface.continuous || surface.width >= 500) && surface.height >= 700,
            ))
          ) contentKind = 'image'
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
      if (!finished) schedule()
    }

    const schedule = (): void => {
      if (finished || scheduled !== undefined) return
      const delay = Math.max(
        100,
        INSPECTION_INTERVAL_MS - (performance.now() - lastInspectionStarted),
      )
      scheduled = window.setTimeout(() => void inspect(), delay)
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
