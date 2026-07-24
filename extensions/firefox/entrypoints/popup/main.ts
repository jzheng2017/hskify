import type { HskLevel } from '../../src/contracts/browser'
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

let preparedPermissions: PermissionPlan | undefined
let startInFlight = false

function selectedLevel(): HskLevel {
  const parsed = Number(levelSelect.value)
  return isHskLevel(parsed) ? parsed : DEFAULT_HSK_LEVEL
}

function setBusy(busy: boolean): void {
  const unavailable = busy || startInFlight
  translateVisible.disabled = unavailable || !preparedPermissions
  translateAll.disabled = unavailable || !preparedPermissions
  levelSelect.disabled = unavailable
}

function renderState(state: PopupState): void {
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

async function prepare(): Promise<void> {
  setBusy(false)
  statusTitle.textContent = 'Preparing page'
  statusDetail.textContent = 'Inspecting supported image hosts…'
  try {
    preparedPermissions = await sendBackgroundMessage({ type: 'popup:prepare' })
    setBusy(false)
    await refresh()
  } catch (error) {
    renderError(error)
  }
}

void prepare()
const refreshTimer = window.setInterval(() => void refresh(), 1_000)
window.addEventListener('unload', () => window.clearInterval(refreshTimer), { once: true })
