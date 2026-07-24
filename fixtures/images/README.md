# Synthetic browser fixtures

The SVG panels in this directory were created specifically for the HSK Manga
Translator browser tests. They contain no third-party manga artwork and may be
used as CC0 test fixtures.

They intentionally cover monochrome manga, colour webtoon, wide-panel,
irregular-bubble, rotated-lettering, vertical-bubble, and object-fit geometry
cases. Fixture mode substitutes the frozen browser result contract and a small
valid PNG clean-image payload; production image acquisition deliberately
accepts only PNG, JPEG, WebP, and GIF.
