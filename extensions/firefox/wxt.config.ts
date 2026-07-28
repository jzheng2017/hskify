import { defineConfig } from 'wxt'

export default defineConfig({
  srcDir: '.',
  outDir: '.output',
  manifestVersion: 3,
  manifest: ({ mode }) => ({
    name: mode === 'development' ? 'Hskify Dev' : 'Hskify',
    description: 'Translate English manga lettering into selectable, HSK-controlled Chinese.',
    version: '0.1.0',
    permissions: [
      'activeTab',
      'scripting',
      'storage',
      'nativeMessaging',
      'webRequest',
      'webRequestBlocking',
    ],
    host_permissions: ['http://*/*', 'https://*/*'],
    browser_specific_settings: {
      gecko: {
        id: 'hsk-manga-translator@local.hskify',
        strict_min_version: '142.0',
        data_collection_permissions: {
          required: ['none'],
        },
      },
    },
  }),
})
