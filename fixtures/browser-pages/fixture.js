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
  replacement.src = current.src.includes('e278e36e')
    ? '../../local-corpus/real-reader-v1/objects/95bb904a4b9c10908d5f2de250d1dbd87ab91cfe5b47db8c2974504566b7144d.png'
    : '../../local-corpus/real-reader-v1/objects/e278e36ed24ad2f5a7ea7acaf4bbb92291d79dc26e5ff2d8ea85ffebee581ca7.jpg'
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
  image.src = `http://127.0.0.1:${port}/${image.dataset.crossOriginSrc}`
}

const syntheticComments = document.querySelector('#synthetic-comments')
if (syntheticComments) {
  for (let index = 0; index < 133; index += 1) {
    const comment = document.createElement('div')
    comment.className = 'synthetic-comment'
    const avatar = document.createElement('img')
    avatar.className = 'comment-avatar'
    avatar.src = `../../local-corpus/real-reader-v1/objects/95bb904a4b9c10908d5f2de250d1dbd87ab91cfe5b47db8c2974504566b7144d.png?avatar=${index}`
    avatar.alt = `Synthetic commenter avatar ${index + 1}`
    const text = document.createElement('p')
    text.textContent = `Synthetic site-neutral comment ${index + 1}.`
    comment.append(avatar, text)
    syntheticComments.append(comment)
  }
}
