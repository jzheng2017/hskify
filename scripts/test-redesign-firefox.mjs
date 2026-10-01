import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { createServer } from 'node:http'
import { mkdirSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const require = createRequire(resolve('extensions/firefox/package.json'))
const { build } = require('esbuild')
const { firefox } = require('@playwright/test')
const { representativeChapterMarkup } = await import('./run-document-chapter-benchmark.mjs')
const bundle = await build({ stdin: { contents: `
export {detectDocumentChapter} from './extensions/firefox/src/document/extraction';
export {mountDocumentReader} from './extensions/firefox/src/document/reader';
export {discoverPageSurfaces,LiveSurfaceDiscovery} from './extensions/firefox/src/discovery/surfaces';
export {calculateImageGeometry} from './extensions/firefox/src/rendering/geometry';
export {ChapterController} from './extensions/firefox/src/page/controller';
export {SelectableRenderer} from './extensions/firefox/src/rendering/renderer';
`, resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, platform: 'browser', format: 'iife', globalName: 'production' })
const server = createServer((_request, response) => { response.writeHead(200, {'content-type': 'text/html'}); response.end('<!doctype html><html><head></head><body></body></html>') })
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
const browser = await firefox.launch({headless: true, firefoxUserPrefs: {'webgl.disabled': false, 'webgl.force-enabled': true}})
const page = await browser.newPage({viewport: {width: 1000, height: 800}})
const cases = []
await page.exposeFunction('__captureRenderedPixels', async rect => 'data:image/png;base64,' + (await page.screenshot({type:'png', clip:{x:rect.x,y:rect.y,width:rect.width,height:rect.height}, timeout:10_000})).toString('base64'))
try {
  await page.goto('http://127.0.0.1:' + server.address().port)
  await page.addScriptTag({content: bundle.outputFiles[0].text})
  await page.evaluate(() => {
    window.paragraph = 'Mara opened the letter beside the old window. She did not recognize the handwriting, but she remembered the promise that her brother had made before he left the village. The rain covered his footsteps while she waited for him to return. '
    window.makeChapter = (extra = '', count = 8) => {
      document.body.innerHTML = '<article id="story"><h1>The Letter</h1>' + extra + Array.from({length: count}, (_, i) => '<p data-order="' + i + '">' + window.paragraph + '</p>').join('') + '</article>'
      return document.getElementById('story')
    }
    window.translation = sourceText => ({sourceText, baseChinese: '她看见了那封信。', displayedChinese: '她看见了那封信。', pinyin: 'tā kàn jiàn le nà fēng xìn', termination: 'stop', protectedNames: [], hsk: {requestedLevel: 3, learningMode: 'natural', strictlyValid: true, levelCoverage: 1, aboveLevelTokens: [], teachingTerms: [], repairState: 'not-needed'}})
    window.chapter = async () => {
      const result = await production.detectDocumentChapter(document)
      if (result.kind !== 'document') throw new Error(JSON.stringify(result))
      return result.chapter
    }
  })
  async function check(name, fn) {
    try { const evidence = await page.evaluate(fn); cases.push({name, status: 'pass', evidence}); console.log('PASS ' + name) }
    catch (error) { cases.push({name, status: 'fail', error: error.message}); console.log('FAIL ' + name + ': ' + error.message) }
  }
  await check('Complete coverage including direct section text, inline media, nested lists and original node identity', async () => {
    const root = makeChapter('<section id="direct">The final secret was written here.</section><p id="media">Before the picture <img id="plate" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"> after the picture.</p><ul><li id="outer">First item<ul><li id="inner">Nested story item</li></ul>Last clause.</li></ul>')
    const originals = [...root.querySelectorAll('*')], textNode = root.querySelector('[data-order]').firstChild
    const chapter = await window.chapter()
    if (!chapter.snapshot.blocks.some(b => b.text.includes('final secret'))) throw new Error('Direct section omitted')
    const reader = production.mountDocumentReader(chapter)
    for (const b of chapter.snapshot.blocks) reader.installBlock(b.itemId, translation(b.text))
    if (originals.some(node => !node.isConnected) || !textNode.isConnected) throw new Error('Original node disconnected')
    reader.destroy()
    if (textNode.data !== paragraph || !document.getElementById('inner').textContent.includes('Nested story')) throw new Error('Restoration changed story')
    return {items: chapter.snapshot.blocks.length}
  })
  await check('An empty external child never adopts translated Chinese as original', async () => {
    makeChapter(); const chapter = await window.chapter(), node = document.querySelector('[data-order]')
    const reader = production.mountDocumentReader(chapter), block = chapter.snapshot.blocks.find(b => b.kind === 'prose')
    reader.installBlock(block.itemId, translation(block.text)); node.append(document.createElement('span'))
    await new Promise(r => setTimeout(r, 0)); reader.destroy()
    if (node.textContent !== paragraph) throw new Error('Original corrupted: ' + node.textContent)
    return true
  })
  await check('Queued external edits are drained before an owned result write', async () => {
    makeChapter(); const chapter = await window.chapter(), node = document.querySelector('[data-order]')
    const original = node.firstChild, reader = production.mountDocumentReader(chapter), block = chapter.snapshot.blocks.find(b => b.kind === 'prose')
    original.data = 'The site replaced this source.'
    try { reader.installBlock(block.itemId, translation(block.text)) } catch {}
    if (!reader.isDestroyed || !node.textContent.includes('site replaced')) throw new Error('Queued edit lost or stale result installed')
    return true
  })
  await check('Benign lazy image src changes do not cancel prose', async () => {
    makeChapter('<img id="lazy">'); const chapter = await window.chapter(), reader = production.mountDocumentReader(chapter)
    document.getElementById('lazy').src = '/new-image.png'; await new Promise(r => setTimeout(r, 0))
    if (reader.isDestroyed) throw new Error('Benign image change cancelled reader')
    reader.destroy(); return true
  })
  await check('Removing an ancestor terminates the detached reader', async () => {
    const root = makeChapter(), shell = document.createElement('div'); root.replaceWith(shell); shell.append(root)
    const reader = production.mountDocumentReader(await window.chapter()); shell.remove()
    await new Promise(r => setTimeout(r, 0))
    if (!reader.isDestroyed) throw new Error('Detached reader remained active')
    return true
  })
  await check('Spanish prose abstains instead of using Latin-letter prevalence', async () => {
    document.body.innerHTML = '<article><h1>La carta</h1>' + '<p>La lluvia caía sobre el pueblo mientras ella esperaba a su hermano. No reconocía la letra de la carta, pero recordaba la promesa que él había hecho antes de marcharse. Caminó hasta la ventana y miró hacia la montaña.</p>'.repeat(10) + '</article>'
    const result = await production.detectDocumentChapter(document)
    if (result.kind === 'document') throw new Error('Spanish accepted')
    return result
  })
  await check('Uncertain short prose supports explicit region selection', async () => {
    document.body.innerHTML = '<section id="short"><p>Have a seat. May the gods protect you.</p></section>'
    const result = await production.detectDocumentChapter(document, {root: document.getElementById('short')})
    if (result.kind !== 'document') throw new Error('Selected short source rejected')
    return true
  })
  await check('Long paragraphs publish independent sentence groups without splitting original text nodes', async () => {
    const root = makeChapter('<p id="long">' + paragraph.repeat(20) + '</p>'), node = document.getElementById('long').firstChild
    const chapter = await window.chapter(), groups = chapter.snapshot.blocks.filter(b => b.parentBlockId === chapter.snapshot.blocks.find(b => b.text.startsWith('Mara')).parentBlockId)
    if (groups.length < 2 || groups.some(b => b.text.length > 600)) throw new Error('Long paragraph not bounded')
    const reader = production.mountDocumentReader(chapter)
    reader.installBlock(groups.at(-1).itemId, translation(groups.at(-1).text))
    if (root.querySelectorAll('[data-hskify-state=translated]').length !== 1 || !node.isConnected) throw new Error('Publication/ownership failed')
    reader.destroy()
    if (node.data !== paragraph.repeat(20)) throw new Error('Split restoration corrupted source')
    return {groups: groups.length}
  })
  await check('Changes during asynchronous source hashing reject the snapshot', async () => {
    const root = makeChapter(), crypto = {subtle: {digest: async (...args) => {root.append(document.createTextNode('New ending added during hashing.')); return window.crypto.subtle.digest(...args)}}}
    const result = await production.detectDocumentChapter(document, {crypto})
    if (result.kind === 'document') throw new Error('Changed source snapshot accepted')
    return result
  })
  await check('Generic discovery rejects images inside buttons', async () => {
    document.body.innerHTML = '<button><img src="/button.png" style="width:600px;height:900px"></button>'
    const image = document.querySelector('img')
    Object.defineProperties(image, {naturalWidth: {value:600}, naturalHeight:{value:900}, complete:{value:true}})
    if (production.discoverPageSurfaces().surfaces.length) throw new Error('Button admitted')
    return true
  })
  await check('Reused visible canvases emit content revisions; blank canvases abstain', async () => {
    document.body.innerHTML = '<canvas width="600" height="900" style="width:400px;height:600px"></canvas>'
    const canvas = document.querySelector('canvas'), context = canvas.getContext('2d')
    if (await production.discoverPageSurfaces().surfaces[0].capture()) throw new Error('Blank capture accepted')
    context.fillStyle = '#fff'; context.fillRect(0,0,600,900); context.fillStyle = '#111'; context.fillText('First page',50,80)
    const events = [], registry = new production.LiveSurfaceDiscovery(e => events.push(e))
    registry.start(); registry.setActive(true)
    await new Promise(r => setTimeout(r, 1200)); context.fillStyle = '#f00'; context.fillRect(100,100,200,200)
    await new Promise(r => setTimeout(r, 1200)); registry.stop()
    if (!events.some(e => e.type === 'updated' && e.candidate.sourceRevision)) throw new Error('Canvas content revision not emitted')
    return {updates: events.filter(e=>e.type==='updated').length}
  })
  await check('Incremental discovery avoids style reads on unrelated empty elements', async () => {
    document.body.innerHTML = '<div>'.repeat(10000) + '</div>'.repeat(10000)
    const original = window.getComputedStyle; let reads = 0
    window.getComputedStyle = (...args) => { reads++; return original(...args) }
    const started = performance.now()
    production.discoverPageSurfaces()
    window.getComputedStyle = original
    if (reads > 10) throw new Error(reads + ' unnecessary style reads')
    return {durationMs: performance.now() - started, styleReads: reads}
  })
  await check('Window and nested reading anchors stay fixed with site scroll anchoring disabled', async () => {
    document.head.innerHTML = '<style>body{margin:0;overflow-anchor:none}article{width:650px;margin:auto;font:18px/1.5 serif;overflow-anchor:none}p{margin:18px 0}</style>'
    makeChapter('',300); const anchor = document.querySelector('[data-order="150"]'); anchor.scrollIntoView()
    const before = anchor.getBoundingClientRect().top, chapter = await window.chapter(), reader = production.mountDocumentReader(chapter)
    const delta = Math.abs(anchor.getBoundingClientRect().top - before)
    reader.destroy()
    const root = document.getElementById('story'), scroller = document.createElement('div'); scroller.style.cssText='height:500px;overflow-y:auto;overflow-anchor:none'
    root.replaceWith(scroller); scroller.append(root); window.scrollTo(0,0); scroller.scrollTop=anchor.offsetTop-scroller.offsetTop
    const nestedBefore = anchor.getBoundingClientRect().top, nested = production.mountDocumentReader(await window.chapter())
    const nestedDelta = Math.abs(anchor.getBoundingClientRect().top-nestedBefore); nested.destroy()
    if (delta > 2 || nestedDelta > 2) throw new Error('Reading anchor moved: ' + JSON.stringify({delta,nestedDelta}))
    return {delta,nestedDelta}
  })
  await check('Initial document focus identifies the viewport halfway through a chapter before job creation', async () => {
    document.head.innerHTML = '<style>p{margin:20px 0;min-height:100px}</style>'; makeChapter('', 40)
    document.querySelector('[data-order="20"]').scrollIntoView()
    const chapter = await window.chapter(), reader = production.mountDocumentReader(chapter)
    await reader.waitForInitialFocus(new AbortController().signal)
    const visible = reader.visibleItemIds(), initial = chapter.snapshot.blocks[0].itemId
    if (!visible.length || visible.includes(initial) || !visible.some(id => chapter.sourceElements.get(id)?.dataset.order === '20')) throw new Error('Initial viewport did not match reading position: ' + JSON.stringify(visible))
    reader.destroy(); document.head.innerHTML = ''; window.scrollTo(0,0)
    return {visibleItems: visible.length, firstVisibleOrder: chapter.sourceElements.get(visible[0]).dataset.order}
  })
  await check('Computed CSS calc and edge offsets preserve object-fit geometry', async () => {
    const canvas = document.createElement('canvas'); canvas.width = 400; canvas.height = 200
    const ctx = canvas.getContext('2d'); ctx.fillStyle='white'; ctx.fillRect(0,0,400,200); ctx.fillStyle='black'; ctx.fillRect(10,10,40,20)
    document.body.innerHTML = '<img id="positioned" style="width:600px;height:500px;object-fit:contain;object-position:calc(100% - 20px) calc(100% - 30px)">'
    const image = document.getElementById('positioned'); image.src=canvas.toDataURL(); await image.decode()
    const rect = production.calculateImageGeometry(image, image, 400, 200).image
    if (Math.abs(rect.left - -20) > .5 || Math.abs(rect.top - 170) > .5) throw new Error('calc geometry misaligned: ' + JSON.stringify(rect))
    image.style.objectPosition='right 20px bottom 30px'
    const edge = production.calculateImageGeometry(image,image,400,200).image
    if (Math.abs(edge.left - rect.left) > .5 || Math.abs(edge.top - rect.top) > .5) throw new Error('Computed edge offsets changed geometry')
    return {computedPosition:getComputedStyle(image).objectPosition}
  })
  await check('An edit between extraction and mounting cannot be translated or overwritten', async () => {
    const root = makeChapter(), chapter = await window.chapter(), node = root.querySelector('[data-order]').firstChild
    node.data = 'The site changed the sentence before mounting.'
    let accepted = false
    try { const reader = production.mountDocumentReader(chapter); accepted = true; reader.destroy() } catch {}
    if (accepted || node.data !== 'The site changed the sentence before mounting.') throw new Error('Stale snapshot mounted')
    return true
  })
  await check('CSS-hidden story elements are excluded while every visible eligible text slot is mapped', async () => {
    makeChapter('<section style="display:none">Hidden Spanish prose should not be translated.</section>')
    const chapter = await window.chapter()
    if (chapter.snapshot.blocks.some(block => block.text.includes('Hidden Spanish'))) throw new Error('Hidden text included')
    return true
  })
  await check('Discovery does not choose a drawing context on publisher canvases', async () => {
    document.head.innerHTML = ''; document.body.innerHTML = '<canvas width="600" height="400" style="width:600px;height:400px"></canvas>'
    production.discoverPageSurfaces()
    if (!document.querySelector('canvas').getContext('2d')) throw new Error('Discovery locked the drawing API')
    return true
  })
  await check('Default WebGL framebuffer capture falls back to rendered pixels and restores hidden overlays', async () => {
    document.head.innerHTML = ''; document.body.innerHTML = '<canvas width="600" height="400" style="width:600px;height:400px"></canvas><div data-hskify-owned="true" id="overlay" style="position:absolute;left:10px;top:10px;width:200px;height:100px;background:red;visibility:visible!important"></div>'
    window.scrollTo(0,0)
    const canvas = document.querySelector('canvas'), gl = canvas.getContext('webgl')
    if (!gl) throw new Error('WebGL unavailable in Firefox regression environment')
    await new Promise(resolve => requestAnimationFrame(() => {
      gl.clearColor(1,1,1,1); gl.clear(gl.COLOR_BUFFER_BIT); gl.enable(gl.SCISSOR_TEST); gl.scissor(30,30,200,80); gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT); resolve()
    }))
    await new Promise(resolve => requestAnimationFrame(resolve))
    let requests = 0, hidden = false
    window.browser = {runtime:{sendMessage:async message => {
      requests++; hidden = getComputedStyle(document.getElementById('overlay')).visibility === 'hidden'
      return {ok:true,value:await window.__captureRenderedPixels(message.rect)}
    }}}
    const result = await production.discoverPageSurfaces().surfaces[0].capture()
    const restored = document.getElementById('overlay').style.getPropertyValue('visibility') === 'visible' && document.getElementById('overlay').style.getPropertyPriority('visibility') === 'important'
    delete window.browser
    if (!result || requests !== 1 || !hidden || !restored) throw new Error(JSON.stringify({captured:!!result,requests,hidden,restored}))
    return {bytes:result.bytes.byteLength,requests,hidden,restored}
  })
  await check('Rendered capture in a same-origin frame uses top-tab coordinates and restores both overlay owners', async () => {
    document.head.innerHTML = ''; document.body.innerHTML = '<iframe style="position:absolute;left:70px;top:90px;width:650px;height:450px;border:4px solid black"></iframe><div id="root-overlay" data-hskify-owned="true" style="position:absolute;left:70px;top:90px;width:100px;height:100px;background:red"></div>'
    window.scrollTo(0,0)
    const frame = document.querySelector('iframe')
    frame.srcdoc = '<div id="art" style="width:500px;height:250px;background-image:linear-gradient(white,black),url(data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7)"></div><div id="child-overlay" data-hskify-owned="true" style="position:absolute;left:10px;top:10px;background:red;width:100px;height:100px"></div>'
    await new Promise(resolve => frame.addEventListener('load',resolve,{once:true}))
    let message, hidden = false
    window.browser = {runtime:{sendMessage:async request => {
      message = request
      hidden = getComputedStyle(document.getElementById('root-overlay')).visibility === 'hidden' && frame.contentWindow.getComputedStyle(frame.contentDocument.getElementById('child-overlay')).visibility === 'hidden'
      return {ok:true,value:await window.__captureRenderedPixels(request.rect)}
    }}}
    const candidate = production.discoverPageSurfaces().surfaces.find(source => source.kind === 'frame')
    const captured = candidate && await candidate.capture()
    delete window.browser
    if (!captured || message.pageUrl !== location.href || Math.abs(message.rect.x - 82) > .5 || Math.abs(message.rect.y - 102) > .5 || !hidden || getComputedStyle(document.getElementById('root-overlay')).visibility !== 'visible' || frame.contentWindow.getComputedStyle(frame.contentDocument.getElementById('child-overlay')).visibility !== 'visible') throw new Error('Frame capture coordinates or transaction failed: ' + JSON.stringify({captured:!!captured,message,hidden}))
    return {rect:message.rect,hidden,bytes:captured.bytes.byteLength}
  })
  await check('Long transparent source text cannot push a failure notice and Retry out of its region', async () => {
    document.head.innerHTML = ''; document.body.replaceChildren()
    const image = document.createElement('img'); image.width = 700; image.height = 640; image.style.cssText = 'display:block;width:700px;height:640px'
    image.src = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="700" height="640"><rect width="700" height="640" fill="white"/></svg>')
    document.body.append(image); scrollTo(0,0); await image.decode()
    const original = image.outerHTML, candidate = {id:'failure-source',kind:'image',element:image,owner:image,sourceUrl:image.currentSrc,sourceWidth:700,sourceHeight:640,domIndex:0,visible:true}
    let retries = 0
    const renderer = new production.SelectableRenderer({fetchFont:async () => {throw new Error('Source notices must not load translation fonts')},lookup:async () => ({selectedText:'',tokens:[]}),onRetryRegion:id => {if(id !== 'failed-item') throw new Error('Wrong retry item');retries++}})
    const view = renderer.begin(candidate,{jobId:'failure-job',sourceWidth:700,sourceHeight:640})
    try {
      const sourceText = 'The original words remain in the illustration. '.repeat(30)
      for (const [left,right] of [[.25,.75],[.4,.52]]) {
        view.installSourcePreservingRegion({itemId:'failed-item',itemOrder:0,sourceText,disposition:'failed',reason:'Strict lexical policy failed',textPolygon:[{x:left,y:.15},{x:right,y:.15},{x:right,y:.275},{x:left,y:.275}]})
        const shadow = [...view.wrapper.children].find(el => el.shadowRoot).shadowRoot
        const region = shadow.querySelector('.hskify-source-notice'), retry = region.querySelector('button')
        const bounds = region.getBoundingClientRect(), button = retry.getBoundingClientRect()
        if (region.scrollWidth > region.clientWidth + .5 || region.scrollHeight > region.clientHeight + .5 || button.left < bounds.left - .5 || button.right > bounds.right + .5 || button.top < bounds.top - .5 || button.bottom > bounds.bottom + .5) throw new Error('Failure notice or Retry is clipped')
        if (region.dataset.hskifySourceText !== sourceText || region.querySelector('.hskify-region-text').textContent !== sourceText) throw new Error('Original source metadata changed')
        retry.click()
      }
      if (retries !== 2) throw new Error('Retry callback lost')
    } finally {view.destroy()}
    if (image.outerHTML !== original) throw new Error('Original image changed')
    return {retries,sourceWidths:[350,84]}
  })
  await page.setContent(representativeChapterMarkup())
  await page.addScriptTag({content:bundle.outputFiles[0].text})
  await check('Thirty warm 300-block/100000-character extraction and mount samples preserve the performance bound', async () => {
    const samples = []
    for (let index = 0; index < 31; index++) {
      const start = performance.now(), result = await production.detectDocumentChapter(document)
      if (result.kind !== 'document' || result.chapter.snapshot.blocks.length !== 300 || result.chapter.snapshot.characterCount !== 100000) throw new Error('Representative source changed or was rejected')
      const reader = production.mountDocumentReader(result.chapter), duration = performance.now() - start
      reader.destroy(); if (index) samples.push(duration)
    }
    const sorted = [...samples].sort((a,b) => a-b), p95Ms = sorted[Math.ceil(samples.length * .95)-1]
    if (p95Ms >= 100) throw new Error('Extraction + mount p95 exceeded 100ms: ' + JSON.stringify({p95Ms,samples}))
    return {p95Ms,samples}
  })
  const failures = cases.filter(c => c.status === 'fail')
  mkdirSync('.cache/redesign', {recursive:true})
  writeFileSync('.cache/redesign/firefox-regressions.json', JSON.stringify({browser:await browser.version(), transport:'production-modules-in-firefox', cases},null,2))
  assert.equal(failures.length,0, JSON.stringify(failures))
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)) }
