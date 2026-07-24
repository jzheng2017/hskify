import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  expect: {
    timeout: 5_000,
  },
  fullyParallel: false,
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'firefox-renderer-harness',
      use: {
        ...devices['Desktop Firefox'],
      },
    },
  ],
  webServer: {
    command:
      'node ../../node_modules/vite/bin/vite.js --config tests/harness/vite.config.ts',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: false,
    timeout: 30_000,
  },
})
