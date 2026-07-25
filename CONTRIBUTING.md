# Contributing to Hskify

Thank you for helping improve Hskify.

Before making a change, use the [documentation index](docs/README.md) to find
the relevant design notes. Changes that affect component ownership, browser
protocols, security checks, resource bounds, upstream Koharu code, or bundled
data should include corresponding documentation or an architecture decision
record.

Useful starting points:

- [Architecture overview](docs/architecture.md)
- [Maintainer guide](docs/maintainer-guide.md)
- [Firefox manual test checklist](docs/firefox-manual-test-checklist.md)
- [Licence inventory](docs/licence-inventory.md)

Keep Hskify-specific behavior in the Firefox extension,
`browser-companion`, or `hsk-control` where possible. Shared pipeline changes
belong at the nearest Koharu layer and need a regression test. The
[maintainer guide](docs/maintainer-guide.md) explains how to keep those changes
reviewable during upstream synchronization.

Production models and language datasets must not be added without an
item-by-item licence and redistribution review. Do not commit downloaded
artifacts or reader content.

## AI usage

AI-assisted contributions are welcome when the contributor:

- discloses material AI use;
- understands and reviews every submitted change;
- runs appropriate tests and reports their actual results; and
- does not submit generated content, code, or claims that have not been
  validated.

Contributors remain responsible for everything they submit.
