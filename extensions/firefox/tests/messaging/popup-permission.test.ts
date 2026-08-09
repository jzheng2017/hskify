import { afterEach, describe, expect, it, vi } from 'vitest'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

afterEach(() => {
  window.dispatchEvent(new Event('unload'))
  document.body.replaceChildren()
  vi.unstubAllGlobals()
  vi.resetModules()
})

describe('popup permission gesture', () => {
  it('does not replace the page status with setup-ready text while polling', async () => {
    document.body.innerHTML = `
      <select id="hsk-level"><option value="5" selected>5</option></select>
      <select id="learning-mode"><option value="natural" selected>Natural</option></select>
      <select id="reading-direction"><option value="ltr" selected>LTR</option></select>
      <span id="content-kind"></span>
      <div id="image-settings"></div>
      <button id="translate-all">All</button>
      <button id="cancel">Cancel</button>
      <span id="status-title"></span>
      <span id="status-detail"></span>
      <progress id="status-progress"></progress>
      <button id="setup-primary" hidden></button>
    `
    const blockedState = deferred<{
      ok: true
      value: {
        state: 'complete'
        contentKind: 'image'
        current: number
        total: number
        message: string
        hskLevel: 5
        learningMode: 'natural'
        readingDirection: 'ltr'
      }
    }>()
    let stateCalls = 0
    const sendMessage = vi.fn(async (message: { type: string }) => {
      if (message.type === 'setup:status') {
        return {
          ok: true,
          value: { state: 'ready', modelId: 'qwen3.5-4b', message: 'Ready' },
        }
      }
      if (message.type === 'popup:prepare') {
        return { ok: true, value: undefined }
      }
      if (message.type === 'engine:warmup') {
        return {
          ok: true,
          value: { state: 'ready', modelId: 'qwen3.5-4b', message: 'Ready' },
        }
      }
      if (message.type === 'popup:state') {
        stateCalls += 1
        if (stateCalls > 1) return blockedState.promise
        return {
          ok: true,
          value: {
            state: 'complete',
            contentKind: 'image',
            current: 1,
            total: 1,
            message: 'Done',
            hskLevel: 5,
            learningMode: 'natural',
            readingDirection: 'ltr',
          },
        }
      }
      throw new Error(`Unexpected message ${message.type}`)
    })
    vi.stubGlobal('browser', {
      runtime: { sendMessage },
      permissions: { request: vi.fn() },
      storage: {
        local: {
          async get() {
            return {}
          },
          async set() {},
        },
      },
    })

    await import('../../entrypoints/popup/main')
    await vi.waitFor(() =>
      expect(document.querySelector('#status-title')?.textContent).toBe('Translation complete'),
    )
    await new Promise((resolve) => window.setTimeout(resolve, 1_050))

    expect(stateCalls).toBeGreaterThan(1)
    expect(document.querySelector('#status-title')?.textContent).toBe('Translation complete')
    expect(document.querySelector('#status-detail')?.textContent).toBe(
      'The translated text is ready.',
    )
    blockedState.resolve({
      ok: true,
      value: {
        state: 'complete',
        contentKind: 'image',
        current: 1,
        total: 1,
        message: 'Done',
        hskLevel: 5,
        learningMode: 'natural',
        readingDirection: 'ltr',
      },
    })
  })

  it('starts immediately after the page is prepared', async () => {
    document.body.innerHTML = `
      <select id="hsk-level"><option value="5" selected>5</option></select>
      <select id="learning-mode"><option value="natural" selected>Natural</option></select>
      <select id="reading-direction"><option value="ltr" selected>LTR</option></select>
      <span id="content-kind"></span>
      <div id="image-settings"></div>
      <button id="translate-all">All</button>
      <button id="cancel">Cancel</button>
      <span id="status-title"></span>
      <span id="status-detail"></span>
      <progress id="status-progress"></progress>
      <button id="setup-primary" hidden></button>
    `
    const order: string[] = []
    const sendMessage = vi.fn(async (message: { type: string }) => {
      if (message.type === 'setup:status') {
        return {
          ok: true,
          value: {
            state: 'ready',
            modelId: 'qwen3.5-4b',
            message: 'Ready',
          },
        }
      }
      if (message.type === 'popup:prepare') {
        return { ok: true, value: undefined }
      }
      if (message.type === 'engine:warmup') {
        order.push('warmup')
        return {
          ok: true,
          value: { state: 'ready', modelId: 'qwen3.5-4b', message: 'Ready' },
        }
      }
      if (message.type === 'popup:state') {
        return {
          ok: true,
          value: {
            state: 'idle',
            contentKind: 'image',
            current: 0,
            total: 0,
            message: 'Ready',
            hskLevel: 5,
            learningMode: 'natural',
            readingDirection: 'ltr',
          },
        }
      }
      if (message.type === 'popup:start') {
        order.push('start')
        return {
          ok: true,
          value: {
            state: 'running',
            contentKind: 'image',
            current: 0,
            total: 1,
            message: 'Queued',
          },
        }
      }
      throw new Error(`Unexpected message ${message.type}`)
    })
    const storage: Record<string, unknown> = {}
    vi.stubGlobal('browser', {
      runtime: { sendMessage },
      storage: {
        local: {
          async get(key: string) {
            return { [key]: storage[key] }
          },
          async set(values: Record<string, unknown>) {
            Object.assign(storage, values)
          },
        },
      },
    })

    await import('../../entrypoints/popup/main')
    const chapter = document.querySelector<HTMLButtonElement>('#translate-all')
    await vi.waitFor(() => expect(chapter?.disabled).toBe(false))
    expect(document.querySelector('#status-title')?.textContent).toBe('Ready')
    chapter?.click()
    chapter?.click()
    await vi.waitFor(() => expect(order).toEqual(['warmup', 'start']))
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'popup:start',
        scope: 'all',
        learningMode: 'natural',
        readingDirection: 'ltr',
      }),
    )
  })

  it('keeps a failed start inside the popup without opening a setup tab', async () => {
    document.body.innerHTML = `
      <select id="hsk-level"><option value="5" selected>5</option></select>
      <select id="learning-mode"><option value="natural" selected>Natural</option></select>
      <select id="reading-direction"><option value="ltr" selected>LTR</option></select>
      <span id="content-kind"></span>
      <div id="image-settings"></div>
      <button id="translate-all">All</button>
      <button id="cancel">Cancel</button>
      <span id="status-title"></span>
      <span id="status-detail"></span>
      <progress id="status-progress"></progress>
      <button id="setup-primary" hidden></button>
    `
    const sendMessage = vi.fn(async (message: { type: string }) => {
      if (message.type === 'setup:status') {
        return {
          ok: false,
          error: {
            code: 'COMPANION_UNAVAILABLE',
            message: 'The local translation engine is not installed.',
            retryable: true,
          },
        }
      }
      throw new Error(`Unexpected message ${message.type}`)
    })
    vi.stubGlobal('browser', {
      runtime: { sendMessage },
      permissions: { request: vi.fn() },
      storage: {
        local: {
          async get() {
            return {}
          },
          async set() {},
        },
      },
    })

    await import('../../entrypoints/popup/main')
    const action = document.querySelector<HTMLButtonElement>('#setup-primary')
    await vi.waitFor(() => expect(action?.textContent).toBe('Try again'))
    expect(document.querySelector<HTMLButtonElement>('#translate-all')?.disabled).toBe(true)
    action?.click()
    await vi.waitFor(() =>
      expect(
        sendMessage.mock.calls.filter(([message]) => message.type === 'setup:status'),
      ).toHaveLength(2),
    )
  })

  it('starts model setup and renders measured byte progress', async () => {
    document.body.innerHTML = `
      <select id="hsk-level"><option value="5" selected>5</option></select>
      <select id="learning-mode"><option value="natural" selected>Natural</option></select>
      <select id="reading-direction"><option value="ltr" selected>LTR</option></select>
      <span id="content-kind"></span>
      <div id="image-settings"></div>
      <button id="translate-all">All</button>
      <button id="cancel">Cancel</button>
      <span id="status-title"></span>
      <span id="status-detail"></span>
      <progress id="status-progress"></progress>
      <button id="setup-primary" hidden></button>
    `
    const sendMessage = vi.fn(async (message: { type: string }) => {
      if (message.type === 'setup:status') {
        return {
          ok: true,
          value: {
            state: 'missing-models',
            modelId: 'qwen3.5-4b',
            totalBytes: 2048,
            completedBytes: 0,
            requiredDiskBytes: 4096,
            message: 'Models are missing.',
          },
        }
      }
      if (message.type === 'setup:start') {
        return {
          ok: true,
          value: {
            state: 'downloading',
            modelId: 'qwen3.5-4b',
            currentFile: 'Qwen.gguf',
            completedBytes: 1024,
            totalBytes: 2048,
            requiredDiskBytes: 4096,
            message: 'Downloading Qwen.gguf.',
          },
        }
      }
      throw new Error(`Unexpected message ${message.type}`)
    })
    vi.stubGlobal('browser', {
      runtime: { sendMessage },
      permissions: { request: vi.fn() },
      storage: {
        local: {
          async get() {
            return {}
          },
          async set() {},
        },
      },
    })

    await import('../../entrypoints/popup/main')
    const action = document.querySelector<HTMLButtonElement>('#setup-primary')
    await vi.waitFor(() => expect(action?.textContent).toBe('Set up translation'))
    action?.click()
    await vi.waitFor(() =>
      expect(document.querySelector('#status-detail')?.textContent).toContain('50%'),
    )
    expect(document.querySelector<HTMLProgressElement>('#status-progress')?.value).toBe(0.5)
  })
})
