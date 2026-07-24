const lazyImages = document.querySelectorAll('img[data-lazy-src]')
setTimeout(() => {
  for (const image of lazyImages) {
    image.src = image.dataset.lazySrc
    image.removeAttribute('data-lazy-src')
  }
}, 350)

const replaceButton = document.querySelector('#replace-image')
replaceButton?.addEventListener('click', () => {
  const current = document.querySelector('#spa-image')
  if (!(current instanceof HTMLImageElement)) return
  const replacement = current.cloneNode(false)
  replacement.src = current.src.includes('panel-a')
    ? '../images/synthetic-panel-b.png'
    : '../images/synthetic-panel-a.png'
  current.replaceWith(replacement)
})

const navigationCount = document.querySelector('#navigation-count')
document.querySelector('.reader-link')?.addEventListener('click', (event) => {
  event.preventDefault()
  if (navigationCount) {
    navigationCount.textContent = String(Number(navigationCount.textContent ?? '0') + 1)
  }
})

for (const image of document.querySelectorAll('img[data-cross-origin-src]')) {
  const port = new URL(location.href).searchParams.get('cdnPort') ?? '4174'
  image.src = `http://127.0.0.1:${port}/fixtures/images/${image.dataset.crossOriginSrc}`
}

const syntheticComments = document.querySelector('#synthetic-comments')
if (syntheticComments) {
  for (let index = 0; index < 133; index += 1) {
    const comment = document.createElement('div')
    comment.className = 'synthetic-comment'
    const avatar = document.createElement('img')
    avatar.className = 'comment-avatar'
    avatar.src = `../images/synthetic-panel-wide.png?avatar=${index}`
    avatar.alt = `Synthetic commenter avatar ${index + 1}`
    const text = document.createElement('p')
    text.textContent = `Synthetic site-neutral comment ${index + 1}.`
    comment.append(avatar, text)
    syntheticComments.append(comment)
  }
}
