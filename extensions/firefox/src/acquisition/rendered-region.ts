import { sendBackgroundMessage } from '../messaging/messages'

const captures = new WeakMap<Document, Promise<void>>()

/** Wait for the source tab without polling or holding a capture transaction open. */
export async function waitForVisibleSource(element: Element, signal?: AbortSignal): Promise<void> {
  const doc = element.ownerDocument
  if (signal?.aborted) throw new DOMException('Source cancelled.', 'AbortError')
  if (!doc.hidden) return
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      doc.removeEventListener('visibilitychange', changed)
      signal?.removeEventListener('abort', aborted)
    }
    const changed = () => {
      if (!doc.hidden) {
        cleanup()
        resolve()
      }
    }
    const aborted = () => {
      cleanup()
      reject(new DOMException('Source cancelled.', 'AbortError'))
    }
    doc.addEventListener('visibilitychange', changed)
    signal?.addEventListener('abort', aborted, { once: true })
    changed()
    if (signal?.aborted) aborted()
  })
}

function captureViewport(element: Element):
  | {
      document: Document
      window: Window
      rect: {
        left: number
        top: number
        right: number
        bottom: number
        width: number
        height: number
      }
    }
  | undefined {
  const ownerWindow = element.ownerDocument.defaultView
  if (!ownerWindow) return undefined
  let win: Window = ownerWindow
  const source = element.getBoundingClientRect()
  let left = source.left,
    top = source.top
  try {
    while (win.frameElement) {
      if (
        left < 0 ||
        top < 0 ||
        left + source.width > win.innerWidth ||
        top + source.height > win.innerHeight
      )
        return undefined
      const frame: Element = win.frameElement
      const box = frame.getBoundingClientRect()
      const parent: Window | null = frame.ownerDocument.defaultView
      if (!parent) return undefined
      for (let node: Element | null = frame; node; node = node.parentElement) {
        const style = parent.getComputedStyle(node)
        if (
          style.transform !== 'none' ||
          (style.zoom && style.zoom !== '1' && style.zoom !== 'normal')
        )
          return undefined
      }
      left += box.left + frame.clientLeft
      top += box.top + frame.clientTop
      win = parent
    }
    return {
      document: win.document,
      window: win,
      rect: {
        left,
        top,
        right: left + source.width,
        bottom: top + source.height,
        width: source.width,
        height: source.height,
      },
    }
  } catch {
    return undefined
  }
}

function nextPaint(win: Window, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const aborted = () => {
      win.cancelAnimationFrame(frame)
      signal?.removeEventListener('abort', aborted)
      reject(new DOMException('Source cancelled.', 'AbortError'))
    }
    const frame = win.requestAnimationFrame(() => {
      signal?.removeEventListener('abort', aborted)
      resolve()
    })
    signal?.addEventListener('abort', aborted, { once: true })
    if (signal?.aborted) aborted()
  })
}

/** Capture the rendered source while extension overlays are transactionally hidden. */
export async function captureRenderedRegion(
  element: Element,
  width: number,
  height: number,
  signal?: AbortSignal,
): Promise<ArrayBuffer | undefined> {
  await waitForVisibleSource(element, signal)
  const viewport = captureViewport(element)
  if (!viewport || !element.isConnected || viewport.document.hidden || signal?.aborted)
    return undefined
  const doc = viewport.document,
    win = viewport.window
  let rect = viewport.rect
  // A partially offscreen rendered surface cannot be reconstructed from a visible-tab capture.
  if (
    rect.left < 0 ||
    rect.top < 0 ||
    rect.right > win.innerWidth ||
    rect.bottom > win.innerHeight ||
    rect.width <= 0 ||
    rect.height <= 0
  )
    return undefined
  const previous = captures.get(doc) ?? Promise.resolve()
  let release!: () => void
  const completion = new Promise<void>((resolve) => {
    release = resolve
  })
  captures.set(doc, completion)
  await previous
  const queued = captureViewport(element)
  if (queued) rect = queued.rect
  const sourceDocument = element.ownerDocument
  const overlayNodes = new Set([
    ...doc.querySelectorAll<HTMLElement>('[data-hskify-owned]'),
    ...sourceDocument.querySelectorAll<HTMLElement>('[data-hskify-owned]'),
  ])
  const overlays = [...overlayNodes].map((node) => ({
    node,
    value: node.style.getPropertyValue('visibility'),
    priority: node.style.getPropertyPriority('visibility'),
  }))
  try {
    if (
      !queued ||
      signal?.aborted ||
      !element.isConnected ||
      doc.hidden ||
      rect.left < 0 ||
      rect.top < 0 ||
      rect.right > win.innerWidth ||
      rect.bottom > win.innerHeight
    )
      return undefined
    for (const { node } of overlays) node.style.setProperty('visibility', 'hidden', 'important')
    await nextPaint(win, signal)
    const dataUrl = await sendBackgroundMessage({
      type: 'source:capture-region',
      pageUrl: win.location.href,
      rect: {
        x: rect.left + win.scrollX,
        y: rect.top + win.scrollY,
        width: rect.width,
        height: rect.height,
      },
    })
    const current = captureViewport(element)?.rect
    if (!current) return undefined
    if (
      signal?.aborted ||
      !element.isConnected ||
      doc.hidden ||
      Math.max(
        Math.abs(current.left - rect.left),
        Math.abs(current.top - rect.top),
        Math.abs(current.width - rect.width),
        Math.abs(current.height - rect.height),
      ) > 0.5
    )
      return undefined
    const blob = await (await fetch(dataUrl)).blob()
    const bitmap = await win.createImageBitmap(blob)
    try {
      const raster = doc.createElement('canvas')
      raster.width = width
      raster.height = height
      const context = raster.getContext('2d')
      if (!context) return undefined
      context.drawImage(bitmap, 0, 0, width, height)
      const probe = doc.createElement('canvas')
      probe.width = 64
      probe.height = 64
      const probeContext = probe.getContext('2d')
      if (!probeContext) return undefined
      probeContext.drawImage(raster, 0, 0, 64, 64)
      const pixels = probeContext.getImageData(0, 0, 64, 64).data
      if (!pixels.some((value, index) => value !== pixels[index % 4])) return undefined
      const encoded = await new Promise<Blob | null>((resolve) =>
        raster.toBlob(resolve, 'image/png'),
      )
      return encoded?.arrayBuffer()
    } finally {
      bitmap.close()
    }
  } catch {
    return undefined
  } finally {
    for (const { node, value, priority } of overlays) {
      if (value) node.style.setProperty('visibility', value, priority)
      else node.style.removeProperty('visibility')
    }
    release()
    if (captures.get(doc) === completion) captures.delete(doc)
  }
}
