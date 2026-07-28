import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { defineConfig, type Plugin } from 'vite'

const root = fileURLToPath(new URL('.', import.meta.url))
const repositoryRoot = fileURLToPath(new URL('../../../..', import.meta.url))
const manifestPath = resolve(repositoryRoot, 'fixtures/real-reader-corpus/manifest.json')
const corpusRoot = resolve(
  process.env.HSKIFY_REAL_READER_CORPUS ?? resolve(repositoryRoot, 'local-corpus/real-reader-v1'),
)

type CorpusCase = {
  id: string
  object: {
    path: string
    mimeType: string
  }
}

function localCorpusPlugin(): Plugin {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as { cases: CorpusCase[] }
  const cases = new Map(manifest.cases.map((item) => [item.id, item]))
  return {
    name: 'hskify-local-real-reader-corpus',
    configureServer(server) {
      server.middlewares.use('/__real-reader/', (request, response) => {
        const caseId = decodeURIComponent((request.url ?? '').split('?')[0] ?? '').replace(
          /^\/+/u,
          '',
        )
        const item = cases.get(caseId)
        if (!item) {
          response.statusCode = 404
          response.end(`Unknown real-reader corpus case: ${caseId}`)
          return
        }
        const objectPath = resolve(corpusRoot, item.object.path)
        try {
          response.setHeader('content-type', item.object.mimeType)
          response.setHeader('cache-control', 'no-store')
          response.end(readFileSync(objectPath))
        } catch {
          response.statusCode = 503
          response.setHeader('content-type', 'text/plain; charset=utf-8')
          response.end(
            `Missing local real-reader object: ${objectPath}\nRun "node scripts/real-reader-corpus.mjs verify --selection smoke".`,
          )
        }
      })
    },
  }
}

export default defineConfig({
  root,
  plugins: [localCorpusPlugin()],
  server: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
    fs: {
      allow: [repositoryRoot],
    },
  },
})
