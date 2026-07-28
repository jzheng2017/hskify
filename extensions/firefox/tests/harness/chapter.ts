export {}

const pageOne = document.querySelector<HTMLImageElement>('#page-1')
const pageTwo = document.querySelector<HTMLImageElement>('#page-2')
if (!pageOne || !pageTwo) throw new Error('Chapter fixture is incomplete.')

pageOne.src = '/__real-reader/webtoon-vigilante-1-page-20'
pageTwo.src = '/__real-reader/webtoon-rooftops-1-page-20'
await Promise.all([pageOne.decode(), pageTwo.decode()])
document.documentElement.dataset.chapterReady = 'true'
