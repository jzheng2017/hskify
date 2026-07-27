import type { BrowserSetupStatus, HskLevel } from '../../src/contracts/browser'
import {
  RuntimeMessageError,
  sendBackgroundMessage,
  type PermissionPlan,
  type PopupState,
} from '../../src/messaging/messages'
import {
  DEFAULT_HSK_LEVEL,
  DEFAULT_NAME_TRANSLATION,
  isHskLevel,
  isNameTranslation,
  loadHskLevel,
  loadNameTranslation,
  saveHskLevel,
  saveNameTranslation,
  type NameTranslation,
} from '../../src/messaging/settings'

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`The popup document is missing ${selector}.`)
  return element
}

const levelSelect = requiredElement<HTMLSelectElement>('#hsk-level')
const nameTranslationSelect = requiredElement<HTMLSelectElement>('#name-translation')
const translateAll = requiredElement<HTMLButtonElement>('#translate-all')
const cancel = requiredElement<HTMLButtonElement>('#cancel')
const statusTitle = requiredElement<HTMLElement>('#status-title')
const statusDetail = requiredElement<HTMLElement>('#status-detail')
const statusProgress = requiredElement<HTMLProgressElement>('#status-progress')
const setupPrimary = requiredElement<HTMLButtonElement>('#setup-primary')
const productName = document.querySelector<HTMLElement>('#product-name')

if (productName) productName.textContent = browser.runtime.getManifest().name

let preparedPermissions: PermissionPlan | undefined
let startInFlight = false
let refreshInFlight = false
let setupReady = false
let pagePreparationFailed = false
let setupAction: 'reconnect' | 'download' | 'retry' | undefined

function selectedLevel(): HskLevel {
  const parsed = Number(levelSelect.value)
  return isHskLevel(parsed) ? parsed : DEFAULT_HSK_LEVEL
}

function selectedNameTranslation(): NameTranslation {
  return isNameTranslation(nameTranslationSelect.value)
    ? nameTranslationSelect.value
    : DEFAULT_NAME_TRANSLATION
}

function setBusy(busy: boolean): void {
  const unavailable = busy || startInFlight
  translateAll.disabled = unavailable || !setupReady || !preparedPermissions
  levelSelect.disabled = unavailable || !setupReady
  nameTranslationSelect.disabled = unavailable || !setupReady
  setupPrimary.disabled = unavailable
}

function renderState(state: PopupState): void {
  setupReady = true
  setupAction = undefined
  setupPrimary.hidden = true
  levelSelect.value = String(state.hskLevel)
  nameTranslationSelect.value = state.nameTranslation
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
  statusDetail.textContent =
    state.state === 'running'
      ? 'Hskify is translating this chapter.'
      : state.state === 'complete'
        ? 'The translated text is ready.'
        : state.state === 'failed'
          ? 'Some images could not be translated. Try again from the page.'
        : state.state === 'cancelled'
          ? 'Anything unfinished was left unchanged.'
          : state.message
  statusProgress.hidden = !active
  if (active) statusProgress.removeAttribute('value')
}

function renderError(error: unknown): void {
  setBusy(false)
  cancel.hidden = true
  statusProgress.hidden = true
  statusTitle.textContent = 'Could not start'
  statusDetail.textContent =
    error instanceof RuntimeMessageError && error.code === 'IMAGE_PERMISSION_DENIED'
      ? 'Allow Hskify to read the page images, then try again.'
      : 'Hskify couldn’t complete that action. Please try again.'
}

function renderCompanionMissing(_error: unknown): void {
  setupReady = false
  setupAction = 'reconnect'
  preparedPermissions = undefined
  cancel.hidden = true
  statusProgress.hidden = true
  setupPrimary.hidden = false
  setupPrimary.textContent = 'Try again'
  statusTitle.textContent = 'Hskify couldn’t start'
  statusDetail.textContent = 'Please try again.'
  setBusy(false)
}

function renderSetup(status: BrowserSetupStatus): void {
  setupReady = status.state === 'ready'
  cancel.hidden = true
  statusDetail.textContent =
    status.state === 'missing-models'
      ? 'Download the files Hskify needs to translate on this computer.'
      : status.state === 'downloading'
        ? 'Getting the translation files…'
        : status.state === 'verifying'
          ? 'Almost ready…'
          : status.state === 'failed'
            ? 'Setup could not be completed. Please try again.'
            : 'Hskify is ready.'
  setupPrimary.hidden = true
  setupAction = undefined

  if (status.state === 'ready') {
    statusProgress.hidden = true
    return
  }

  preparedPermissions = undefined
  statusTitle.textContent =
    status.state === 'missing-models'
      ? 'One-time download needed'
      : status.state === 'downloading'
        ? 'Setting up Hskify'
        : status.state === 'verifying'
          ? 'Finishing setup'
          : 'Setup needs attention'

  if (status.state === 'missing-models' || status.state === 'failed') {
    setupAction = status.state === 'failed' ? 'retry' : 'download'
    setupPrimary.textContent =
      status.state === 'failed' ? 'Try setup again' : 'Set up translation'
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
      const percent = Math.round((status.completedBytes! / status.totalBytes!) * 100)
      statusDetail.textContent = `Getting the translation files… ${percent}%`
    } else {
      statusProgress.removeAttribute('value')
    }
  }
  setBusy(status.state === 'downloading' || status.state === 'verifying')
}

async function finishStart(
  hskLevel: HskLevel,
  nameTranslation: NameTranslation,
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
    await Promise.all([saveHskLevel(hskLevel), saveNameTranslation(nameTranslation)])
    const state = await sendBackgroundMessage({
      type: 'popup:start',
      scope: 'all',
      hskLevel,
      nameTranslation,
    })
    startInFlight = false
    renderState({ ...state, hskLevel, nameTranslation })
  } catch (error) {
    startInFlight = false
    renderError(error)
  }
}

function startChapter(): void {
  if (startInFlight) return
  const plan = preparedPermissions
  if (!plan) {
    renderError(new Error('Image hosts are still being inspected. Please try again.'))
    return
  }
  const hskLevel = selectedLevel()
  const nameTranslation = selectedNameTranslation()
  const origins = plan.allOrigins
  startInFlight = true
  setBusy(true)
  statusTitle.textContent = 'Preparing chapter'
  statusDetail.textContent =
    origins.length > 0
      ? 'Waiting for permission to read the images…'
      : 'Finding the chapter images…'
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
  void finishStart(hskLevel, nameTranslation, permissionRequest)
}

translateAll.addEventListener('click', startChapter)
cancel.addEventListener('click', async () => {
  try {
    const state = await sendBackgroundMessage({ type: 'popup:cancel' })
    renderState({
      ...state,
      hskLevel: selectedLevel(),
      nameTranslation: selectedNameTranslation(),
    })
  } catch (error) {
    renderError(error)
  }
})
levelSelect.addEventListener('change', () => void saveHskLevel(selectedLevel()))
nameTranslationSelect.addEventListener('change', () =>
  void saveNameTranslation(selectedNameTranslation()),
)

setupPrimary.addEventListener('click', async () => {
  if (!setupAction || startInFlight) return
  startInFlight = true
  setBusy(true)
  try {
    if (setupAction === 'reconnect') {
      renderSetup(await sendBackgroundMessage({ type: 'setup:status' }))
      return
    }
    renderSetup(await sendBackgroundMessage({ type: 'setup:start' }))
  } catch (error) {
    if (error instanceof RuntimeMessageError && error.code === 'COMPANION_UNAVAILABLE') {
      renderCompanionMissing(error)
    } else {
      renderError(error)
      setupAction = 'retry'
      setupPrimary.textContent = 'Try setup again'
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
    const [hskLevel, nameTranslation] = await Promise.all([
      loadHskLevel(),
      loadNameTranslation(),
    ])
    levelSelect.value = String(hskLevel)
    nameTranslationSelect.value = nameTranslation
    renderError(error)
  }
}

async function prepareReadyPage(): Promise<void> {
  if (!preparedPermissions && !pagePreparationFailed) {
    statusTitle.textContent = 'Preparing chapter'
    statusDetail.textContent = 'Finding the chapter images…'
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
    if (status.state === 'ready') {
      setupReady = true
      setupAction = undefined
      setupPrimary.hidden = true
      await prepareReadyPage()
    } else {
      renderSetup(status)
    }
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
void loadNameTranslation().then((preference) => {
  nameTranslationSelect.value = preference
})
void refreshAll()
const refreshTimer = window.setInterval(() => void refreshAll(), 1_000)
window.addEventListener('unload', () => window.clearInterval(refreshTimer), { once: true })
