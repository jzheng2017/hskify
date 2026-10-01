import { describe, expect, it } from 'vitest'

import { sha256Hex } from '../../src/acquisition/hash'
import {
  canonicalDocumentText,
  detectDocumentChapter,
  normalizeDocumentText,
} from '../../src/document'

const sentence =
  'The lantern light followed Mara through the old archive while she considered the promise that had brought her home. '

function setPage(body: string, title = 'Chapter 7: The Archive'): void {
  document.head.replaceChildren()
  document.body.innerHTML = body
  document.title = title
}

function paragraphs(count = 8, repeats = 2): string {
  return Array.from(
    { length: count },
    (_, index) => `<p>${index + 1}. ${sentence.repeat(repeats)}</p>`,
  ).join('')
}

describe('light-novel document extraction', () => {
  it('extracts semantic story structure from a clone and never annotates or mutates live DOM', async () => {
    setPage(`
      <header><nav>Home Library Login</nav></header>
      <main>
        <article id="chapter" class="novel chapter-content">
          <h1>Chapter 7: The Archive</h1>
          <h2>A promise after midnight</h2>
          <p>${sentence.repeat(2)}</p>
          <p id="inline-prose">The letter was
            <em>still sealed</em><br>when Mara returned.</p>
          <blockquote>${sentence.repeat(2)}</blockquote>
          <ol><li>${sentence.repeat(2)}</li><li>${sentence.repeat(2)}</li></ol>
          <p>${sentence.repeat(2)}</p>
          <figure>
            <img src="https://images.example.test/archive.webp" alt="The old archive at night">
            <figcaption>An illustration of the archive</figcaption>
          </figure>
          <hr>
          <p>${sentence.repeat(2)}</p>
        </article>
        <aside class="comments"><p>${sentence.repeat(20)}</p></aside>
      </main>
    `)
    const before = document.documentElement.outerHTML

    const detection = await detectDocumentChapter(document)

    expect(detection, JSON.stringify(detection)).toMatchObject({ kind: 'document' })
    if (detection.kind !== 'document') return
    expect(document.documentElement.outerHTML).toBe(before)
    expect(document.querySelector('[data-hskify-source-marker]')).toBeNull()
    expect(detection.chapter.sourceRoot).toBe(document.querySelector('#chapter'))
    expect(detection.chapter.snapshot.blocks.map((block) => block.kind)).toContain('heading')
    expect(detection.chapter.snapshot.blocks.map((block) => block.kind)).toContain('dialogue')
    expect(detection.chapter.snapshot.blocks.map((block) => block.kind)).toContain('caption')
    expect(
      detection.chapter.snapshot.blocks.find((block) => block.text.startsWith('The letter was'))
        ?.text,
    ).toBe('The letter was still sealed\nwhen Mara returned.')
    expect(detection.chapter.structure.some((item) => item.type === 'image')).toBe(true)
    expect(detection.chapter.structure.some((item) => item.type === 'separator')).toBe(true)
    expect(detection.chapter.snapshot.blocks.some((block) => block.text.includes('Login'))).toBe(
      false,
    )
    expect(detection.chapter.snapshot.blocks.some((block) => block.text.includes('comments'))).toBe(
      false,
    )

    const expected = await sha256Hex(
      new TextEncoder().encode(canonicalDocumentText(detection.chapter.snapshot.blocks)).buffer,
    )
    expect(detection.chapter.snapshot.sourceSha256).toBe(expected)
  })

  it('accepts div-heavy prose after Readability identifies the article', async () => {
    setPage(
      `
      <div class="site-shell">
        <div id="story">
          <div><h1>Chapter 19</h1></div>
          ${Array.from(
            { length: 8 },
            (_, index) => `<div>${index + 1}. ${sentence.repeat(2)}</div>`,
          ).join('')}
        </div>
      </div>
    `,
      'Chapter 19',
    )

    const detection = await detectDocumentChapter(document)

    expect(detection, JSON.stringify(detection)).toMatchObject({ kind: 'document' })
    if (detection.kind !== 'document') return
    expect(detection.chapter.snapshot.blocks.length).toBeGreaterThanOrEqual(5)
    expect(detection.chapter.snapshot.characterCount).toBeGreaterThanOrEqual(1_000)
  })

  it('maps an exact chapter heading beside the article into the safe story root', async () => {
    setPage(
      `<main id="story-shell">
        <h1 id="chapter-title">Chapter 21: The Silent Gate</h1>
        <article id="chapter-body">${paragraphs()}</article>
      </main>`,
      'Chapter 21: The Silent Gate',
    )

    const detection = await detectDocumentChapter(document)

    expect(detection.kind).toBe('document')
    if (detection.kind !== 'document') return
    const title = detection.chapter.snapshot.blocks.find((block) => block.kind === 'heading')
    expect(title?.text).toBe('Chapter 21: The Silent Gate')
    expect(detection.chapter.sourceElements.get(title!.itemId)).toBe(
      document.querySelector('#chapter-title'),
    )
    expect(detection.chapter.sourceRoot).toBe(document.querySelector('#story-shell'))
  })

  it('retains identical prose blocks with distinct stable IDs', async () => {
    const duplicate = sentence.repeat(2)
    setPage(
      `<article id="story"><h1>Chapter 2</h1>${Array(8).fill(`<p>${duplicate}</p>`).join('')}</article>`,
      'Chapter 2',
    )

    const first = await detectDocumentChapter(document)
    const second = await detectDocumentChapter(document)

    expect(first.kind).toBe('document')
    expect(second.kind).toBe('document')
    if (first.kind !== 'document' || second.kind !== 'document') return
    const prose = first.chapter.snapshot.blocks.filter((block) => block.kind === 'prose')
    expect(prose).toHaveLength(8)
    expect(new Set(prose.map((block) => block.itemId)).size).toBe(8)
    expect(first.chapter.snapshot.blocks.map((block) => block.itemId)).toEqual(
      second.chapter.snapshot.blocks.map((block) => block.itemId),
    )
    expect(first.chapter.snapshot.sourceSha256).toBe(second.chapter.snapshot.sourceSha256)
  })

  it('rejects non-English content and manga pages with only incidental prose', async () => {
    setPage(
      `<article id="story"><h1>第一章</h1>${Array(8)
        .fill(`<p>${'他走过安静的庭院，想起了多年前的约定。'.repeat(15)}</p>`)
        .join('')}</article>`,
      '第一章',
    )
    expect(await detectDocumentChapter(document)).toMatchObject({
      kind: 'not-document',
      reason: 'not-predominantly-english',
    })

    setPage(`
      <main>
        <p>A short reader hint that does not form a novel chapter.</p>
        <img src="https://images.example.test/page-1.webp" alt="Page one">
        <img src="https://images.example.test/page-2.webp" alt="Page two">
        <img src="https://images.example.test/page-3.webp" alt="Page three">
      </main>
    `)
    const manga = await detectDocumentChapter(document)
    expect(manga.kind).toBe('not-document')
  })

  it('accepts a hybrid story as document and leaves its illustrations structural', async () => {
    setPage(
      `<article id="hybrid"><h1>Chapter with plates</h1>${paragraphs()}<img src="https://images.example.test/plate.jpg" alt="A chapter plate"></article>`,
      'Chapter with plates',
    )
    const detection = await detectDocumentChapter(document)
    expect(detection.kind).toBe('document')
    if (detection.kind !== 'document') return
    expect(detection.chapter.structure.filter((item) => item.type === 'image')).toHaveLength(1)
  })

  it('rejects a body-level mapping even when scattered prose passes length thresholds', async () => {
    setPage(paragraphs(9, 2))
    const detection = await detectDocumentChapter(document)
    expect(detection).toMatchObject({ kind: 'not-document', reason: 'unsafe-content-root' })
  })

  it('abstains from an oversized non-language block', async () => {
    const huge = 'A'.repeat(16 * 1024 + 1)
    setPage(
      `<article id="story"><h1>Oversized</h1><p>${huge}</p>${paragraphs(5, 2)}</article>`,
      'Oversized',
    )
    const detection = await detectDocumentChapter(document)
    expect(detection).toEqual({ kind: 'not-document', reason: 'not-predominantly-english' })
  })

  it('rejects more than 2,000 mapped text blocks rather than dropping the tail', async () => {
    setPage(
      `<article id="story"><h1>Many fragments</h1>${Array.from(
        { length: 2_001 },
        (_, index) => `<p>Fragment ${index}. ${sentence}</p>`,
      ).join('')}</article>`,
      'Many fragments',
    )
    const detection = await detectDocumentChapter(document)
    expect(detection).toEqual({ kind: 'rejected', reason: 'too-many-blocks' })
  })

  it('rejects a complete document request envelope over 1 MiB without truncating', async () => {
    const largeBlock = 'A complete English sentence with stable words and punctuation. '.repeat(250)
    expect(new TextEncoder().encode(largeBlock).byteLength).toBeLessThan(16 * 1024)
    setPage(
      `<article id="story"><h1>Large chapter</h1>${Array(80)
        .fill(`<p>${largeBlock}</p>`)
        .join('')}</article>`,
      'Large chapter',
    )
    const detection = await detectDocumentChapter(document)
    expect(detection.kind).toBe('rejected')
  })

  it('normalizes whitespace deterministically while retaining meaningful line breaks', () => {
    expect(normalizeDocumentText('  One\u00a0  two\r\n  three\n\n\n four  ')).toBe(
      'One two\nthree\n\nfour',
    )
  })
})
