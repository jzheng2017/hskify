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
  it('requests exact origins directly on click before starting async content work', async () => {
    document.body.innerHTML = `
      <select id="hsk-level"><option value="5" selected>5</option></select>
      <button id="translate-visible">Visible</button>
      <button id="translate-all">All</button>
      <button id="cancel">Cancel</button>
      <span id="status-title"></span>
      <span id="status-detail"></span>
      <progress id="status-progress"></progress>
    `
    const permission = deferred<boolean>()
    const order: string[] = []
    const sendMessage = vi.fn(async (message: { type: string }) => {
      if (message.type === 'popup:prepare') {
        return {
          ok: true,
          value: {
            visibleOrigins: ['https://cdn.test/*'],
            allOrigins: ['https://cdn.test/*'],
          },
        }
      }
      if (message.type === 'popup:state') {
        return {
          ok: true,
          value: {
            state: 'idle',
            current: 0,
            total: 0,
            message: 'Ready',
            hskLevel: 5,
          },
        }
      }
      if (message.type === 'popup:start') {
        order.push('start')
        return {
          ok: true,
          value: {
            state: 'running',
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
      permissions: {
        request: vi.fn((request: browser.permissions.Permissions) => {
          order.push(`permission:${request.origins?.join(',')}`)
          return permission.promise
        }),
      },
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
    const visible = document.querySelector<HTMLButtonElement>('#translate-visible')
    await vi.waitFor(() => expect(visible?.disabled).toBe(false))
    visible?.click()
    visible?.click()
    expect(order).toEqual(['permission:https://cdn.test/*'])
    expect(browser.permissions.request).toHaveBeenCalledTimes(1)
    expect(sendMessage.mock.calls.some(([message]) => message.type === 'popup:start')).toBe(false)

    permission.resolve(true)
    await vi.waitFor(() => expect(order).toEqual([
      'permission:https://cdn.test/*',
      'start',
    ]))
  })
})
