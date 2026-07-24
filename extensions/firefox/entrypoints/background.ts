export default defineBackground(() => {
  // Runtime handlers are registered by the messaging module. Keeping the
  // entrypoint side-effect free makes Firefox background suspension safe.
})
