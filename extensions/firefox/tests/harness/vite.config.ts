import { fileURLToPath } from 'node:url'

import { defineConfig } from 'vite'

const root = fileURLToPath(new URL('.', import.meta.url))
const extensionRoot = fileURLToPath(new URL('../..', import.meta.url))

export default defineConfig({
  root,
  server: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
    fs: {
      allow: [extensionRoot],
    },
  },
})
