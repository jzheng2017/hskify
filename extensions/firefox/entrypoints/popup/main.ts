import type { BrowserSetupStatus, HskLevel } from '../../src/contracts/browser'
import {
  RuntimeMessageError,
  sendBackgroundMessage,
  type PermissionPlan,
  type PopupState,
  type TranslationScope,
} from '../../src/messaging/messages'
import {
  DEFAULT_HSK_LEVEL,
  isHskLevel,
  loadHskLevel,
  saveHskLevel,
} from '../../src/messaging/settings'

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`The popup document is missing ${selector}.`)
  return element
}

const levelSelect = requiredElement<HTMLSelectElement>('#hsk-level')
const translateVisible = requiredElement<HTMLButtonElement>('#translate-visible')
const translateAll = requiredElement<HTMLButtonElement>('#translate-all')
const cancel = requiredElement<HTMLButtonElement>('#cancel')
const statusTitle = requiredElement<HTMLElement>('#status-title')
const statusDetail = requiredElement<HTMLElement>('#status-detail')
const statusProgress = requiredElement<HTMLProgressElement>('#status-progress')
const setupPrimary = requiredElement<HTMLButtonElement>('#setup-primary')

let preparedPermissions: PermissionPlan | undefined
let startInFlight = false
let refreshInFlight = false
let setupReady = false
let pagePreparationFailed = false
let setupAction: 'install' | 'download' | 'retry' | undefined

function selectedLevel(): HskLevel {
  const parsed = Number(levelSelect.value)
  return isHskLevel(parsed) ? parsed : DEFAULT_HSK_LEVEL
}

function setBusy(busy: boolean): void {
  const unavailable = busy || startInFlight
  translateVisible.disabled = unavailable || !setupReady || !preparedPermissions
  translateAll.disabled = unavailable || !setupReady || !preparedPermissions
  levelSelect.disabled = unavailable || !setupReady
  setupPrimary.disabled = unavailable
}

function renderState(state: PopupState): void {
  setupReady = true
  setupAction = undefined
  setupPrimary.hidden = true
  levelSelect.value = String(state.hskLevel)
  const active = state.state === 'running'
  cancel.hidden = !active
  setBusy(false)
  statusTitle.textContent =
    state.state === 'running'
      ? state.total > 0
        ? `Image ${Math.min(state.current + 1, state.total)} of ${state.total}`
        : 'Preparing page'
      : state.state === 'complete'
        ? 'Translation complete'
        : state.state === 'failed'
          ? 'Translation needs attention'
          : state.state === 'cancelled'
            ? 'Translation cancelled'
            : 'Ready'
  statusDetail.textContent = state.message
  statusProgress.hidden = !active
  if (active) statusProgress.removeAttribute('value')
}

function renderError(error: unknown): void {
  setBusy(false)
  cancel.hidden = true
  statusProgress.hidden = true
  statusTitle.textContent = 'Could not start'
  statusDetail.textContent =
    error instanceof RuntimeMessageError || error instanceof Error
      ? error.message
      : 'The extension action failed.'
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KiB', 'MiB', 'GiB'] as const
  let value = bytes / 1024
  let unit: (typeof units)[number] = units[0]
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024
    unit = units[index] ?? unit
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${unit}`
}

function renderCompanionMissing(error: unknown): void {
  setupReady = false
  setupAction = 'install'
  preparedPermissions = undefined
  cancel.hidden = true
  statusProgress.hidden = true
  setupPrimary.hidden = false
  setupPrimary.textContent = 'Install local engine'
  statusTitle.textContent = 'Local engine required'
  statusDetail.textContent =
    error instanceof Error
      ? error.message
      : 'Install the local translation engine, then return here.'
  setBusy(false)
}

function renderSetup(status: BrowserSetupStatus): void {
  setupReady = status.state === 'ready'
  cancel.hidden = true
  statusDetail.textContent = status.message
  setupPrimary.hidden = true
  setupAction = undefined

  if (status.state === 'ready') {
    statusProgress.hidden = true
    return
  }

  preparedPermissions = undefined
  statusTitle.textContent =
    status.state === 'missing-models'
      ? 'Local models required'
      : status.state === 'downloading'
        ? 'Downloading local models'
        : status.state === 'verifying'
          ? 'Verifying local models'
          : 'Model setup needs attention'

  if (status.state === 'missing-models' || status.state === 'failed') {
    setupAction = status.state === 'failed' ? 'retry' : 'download'
    setupPrimary.textContent =
      status.state === 'failed' ? 'Retry model download' : 'Download local models'
    setupPrimary.hidden = false
  }

  const hasProgress =
    status.completedBytes !== undefined &&
    status.totalBytes !== undefined &&
    status.totalBytes > 0
  statusProgress.hidden = status.state !== 'downloading' && status.state !== 'verifying'
  if (!statusProgress.hidden) {
    if (hasProgress) {
      statusProgress.value = status.completedBytes! / status.totalBytes!
      statusDetail.textContent = `${status.message} ${formatBytes(status.completedBytes!)} of ${formatBytes(status.totalBytes!)}`
    } else {
      statusProgress.removeAttribute('value')
    }
  }
  setBusy(status.state === 'downloading' || status.state === 'verifying')
}

async function finishStart(
  scope: TranslationScope,
  hskLevel: HskLevel,
  permissionRequest: Promise<boolean>,
): Promise<void> {
  try {
    const granted = await permissionRequest
    if (!granted) {
      throw new RuntimeMessageError(
        'IMAGE_PERMISSION_DENIED',
        'Firefox image access was denied. The page was left unchanged.',
        true,
      )
    }
    await saveHskLevel(hskLevel)
    const state = await sendBackgroundMessage({
      type: 'popup:start',
      scope,
      hskLevel,
    })
    startInFlight = false
    renderState({ ...state, hskLevel })
  } catch (error) {
    startInFlight = false
    renderError(error)
  }
}

function startFromClick(scope: TranslationScope): void {
  if (startInFlight) return
  const plan = preparedPermissions
  if (!plan) {
    renderError(new Error('Image hosts are still being inspected. Please try again.'))
    return
  }
  const hskLevel = selectedLevel()
  const origins = scope === 'visible' ? plan.visibleOrigins : plan.allOrigins
  startInFlight = true
  setBusy(true)
  statusTitle.textContent = 'Preparing page'
  statusDetail.textContent =
    origins.length > 0
      ? 'Waiting for Firefox image access…'
      : 'Finding supported manga images…'
  statusProgress.hidden = false
  statusProgress.removeAttribute('value')
  let permissionRequest: Promise<boolean>
  try {
    // Keep this invocation directly in the click stack. Firefox will reject an
    // optional host prompt after asynchronous background/content work.
    permissionRequest =
      origins.length > 0
        ? browser.permissions.request({ origins })
        : Promise.resolve(true)
  } catch (error) {
    startInFlight = false
    renderError(error)
    return
  }
  void finishStart(scope, hskLevel, permissionRequest)
}

translateVisible.addEventListener('click', () => startFromClick('visible'))
translateAll.addEventListener('click', () => startFromClick('all'))
cancel.addEventListener('click', async () => {
  try {
    const state = await sendBackgroundMessage({ type: 'popup:cancel' })
    renderState({ ...state, hskLevel: selectedLevel() })
  } catch (error) {
    renderError(error)
  }
})
levelSelect.addEventListener('change', () => void saveHskLevel(selectedLevel()))

setupPrimary.addEventListener('click', async () => {
  if (!setupAction || startInFlight) return
  startInFlight = true
  setBusy(true)
  try {
    if (setupAction === 'install') {
      await sendBackgroundMessage({ type: 'setup:open-installer' })
      statusDetail.textContent =
        'Install the companion from the product bundle, then reopen this popup.'
      return
    }
    renderSetup(await sendBackgroundMessage({ type: 'setup:start' }))
  } catch (error) {
    if (error instanceof RuntimeMessageError && error.code === 'COMPANION_UNAVAILABLE') {
      renderCompanionMissing(error)
    } else {
      renderError(error)
      setupAction = 'retry'
      setupPrimary.textContent = 'Retry model download'
      setupPrimary.hidden = false
    }
  } finally {
    startInFlight = false
    setBusy(false)
  }
})

async function refresh(): Promise<void> {
  if (startInFlight) return
  try {
    renderState(await sendBackgroundMessage({ type: 'popup:state' }))
  } catch (error) {
    const hskLevel = await loadHskLevel()
    levelSelect.value = String(hskLevel)
    renderError(error)
  }
}

async function prepareReadyPage(): Promise<void> {
  if (!preparedPermissions && !pagePreparationFailed) {
    statusTitle.textContent = 'Preparing page'
    statusDetail.textContent = 'Inspecting supported image hosts…'
    try {
      preparedPermissions = await sendBackgroundMessage({ type: 'popup:prepare' })
    } catch (error) {
      pagePreparationFailed = true
      renderError(error)
      return
    }
  }
  if (preparedPermissions) await refresh()
}

async function refreshAll(): Promise<void> {
  if (refreshInFlight || startInFlight) return
  refreshInFlight = true
  try {
    const status = await sendBackgroundMessage({ type: 'setup:status' })
    renderSetup(status)
    if (status.state === 'ready') await prepareReadyPage()
  } catch (error) {
    if (error instanceof RuntimeMessageError && error.code === 'COMPANION_UNAVAILABLE') {
      renderCompanionMissing(error)
    } else {
      renderError(error)
    }
  } finally {
    refreshInFlight = false
  }
}

void loadHskLevel().then((level) => {
  levelSelect.value = String(level)
})
void refreshAll()
const refreshTimer = window.setInterval(() => void refreshAll(), 1_000)
window.addEventListener('unload', () => window.clearInterval(refreshTimer), { once: true })
