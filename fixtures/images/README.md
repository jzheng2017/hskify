# Synthetic browser fixtures

The source SVGs and generated PNG/WebP panels in this directory were created
specifically for the HSK Manga Translator browser tests. They contain no
third-party manga artwork and may be used as CC0 test fixtures.

They intentionally cover monochrome manga, colour webtoon, wide-panel,
irregular-bubble, rotated-lettering, vertical-bubble, and object-fit geometry
cases. Browser pages reference only the real raster outputs, which pass the
same PNG/WebP signature, dimension, byte, and browser-decode checks as
production images. The long WebP is synthetic and models the measured shape of
a modern vertical reader without copying any site artwork.
