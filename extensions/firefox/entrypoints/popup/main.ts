import type { HskLevel } from '../../src/contracts/browser'
import {
  RuntimeMessageError,
  sendBackgroundMessage,
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

function selectedLevel(): HskLevel {
  const parsed = Number(levelSelect.value)
  return isHskLevel(parsed) ? parsed : DEFAULT_HSK_LEVEL
}

function setBusy(busy: boolean): void {
  translateVisible.disabled = busy
  translateAll.disabled = busy
  levelSelect.disabled = busy
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

async function start(scope: TranslationScope): Promise<void> {
  const hskLevel = selectedLevel()
  setBusy(true)
  statusTitle.textContent = 'Preparing page'
  statusDetail.textContent = 'Finding supported manga images…'
  statusProgress.hidden = false
  statusProgress.removeAttribute('value')
  try {
    await saveHskLevel(hskLevel)
    const state = await sendBackgroundMessage({
      type: 'popup:start',
      scope,
      hskLevel,
    })
    renderState({ ...state, hskLevel })
  } catch (error) {
    renderError(error)
  }
}

translateVisible.addEventListener('click', () => void start('visible'))
translateAll.addEventListener('click', () => void start('all'))
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
  try {
    renderState(await sendBackgroundMessage({ type: 'popup:state' }))
  } catch (error) {
    const hskLevel = await loadHskLevel()
    levelSelect.value = String(hskLevel)
    renderError(error)
  }
}

void refresh()
const refreshTimer = window.setInterval(() => void refresh(), 1_000)
window.addEventListener('unload', () => window.clearInterval(refreshTimer), { once: true })
