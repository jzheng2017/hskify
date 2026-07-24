import { defineConfig } from 'wxt'

export default defineConfig({
  srcDir: '.',
  outDir: '.output',
  manifest: {
    name: 'HSK Manga Translator',
    description: 'Translate English manga lettering into selectable, HSK-controlled Chinese.',
    version: '0.1.0',
    permissions: ['activeTab', 'scripting', 'storage', 'nativeMessaging'],
    host_permissions: ['http://127.0.0.1/*'],
    optional_host_permissions: ['http://*/*', 'https://*/*'],
    browser_specific_settings: {
      gecko: {
        id: 'hsk-manga-translator@local.mangalations',
        strict_min_version: '128.0',
      },
    },
  },
})
