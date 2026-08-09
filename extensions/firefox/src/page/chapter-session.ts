const PAGE_SESSION_KEY = 'hskify.pageSessionId'

export function createPageSessionId(reuseStored: boolean): string {
  const url = new URL(location.href)
  url.hash = ''
  const pageKey = `${PAGE_SESSION_KEY}:${url.href}`
  const existing = reuseStored ? sessionStorage.getItem(pageKey) : null
  if (existing) return existing
  const generated = typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  sessionStorage.setItem(pageKey, generated)
  return generated
}
