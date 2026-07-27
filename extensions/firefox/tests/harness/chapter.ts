import pageOneUrl from '../../../../fixtures/images/synthetic-panel-a.png?url'
import pageTwoUrl from '../../../../fixtures/images/synthetic-panel-b.png?url'

const pageOne = document.querySelector<HTMLImageElement>('#page-1')
const pageTwo = document.querySelector<HTMLImageElement>('#page-2')
if (!pageOne || !pageTwo) throw new Error('Chapter fixture is incomplete.')

pageOne.src = pageOneUrl
pageTwo.src = pageTwoUrl
await Promise.all([pageOne.decode(), pageTwo.decode()])
document.documentElement.dataset.chapterReady = 'true'
