import { registerBackgroundHandlers } from '../src/messaging/background'

export default defineBackground(() => {
  registerBackgroundHandlers()
})
