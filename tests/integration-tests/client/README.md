# Generated general-application client fixture

This directory is generated integration-test material for the broader reused
application RPC surface. It is not the Hskify Firefox browser contract and is
not shipped or mounted by `hsk-manga-browser-daemon`.

Do not use the generated project, history, provider, pipeline, scene, or event
documents in this directory to implement the extension. The current browser
surface is defined by [`docs/browser-contract.md`](../../../docs/browser-contract.md)
and the shared fixtures under `fixtures/contracts`.

Regenerate this client only when its own general-application integration test
requires it; regeneration does not change the Hskify build fingerprint or
browser routes.
