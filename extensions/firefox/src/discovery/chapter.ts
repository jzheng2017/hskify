import { detectDocumentChapter } from '../document/extraction'
import type { DocumentDetection } from '../document/types'
import { looksLikeSequentialArtReader } from './images'
import { discoverPageSurfaces } from './surfaces'

/** Shared by warmup and active translation; prose takes precedence over artwork. */
export async function classifyChapter(
  live: Document = document,
): Promise<DocumentDetection | { kind: 'image' }> {
  const prose = await detectDocumentChapter(live)
  if (prose.kind === 'document' || prose.kind === 'rejected') return prose
  if (
    looksLikeSequentialArtReader(live) ||
    discoverPageSurfaces(live).surfaces.some(
      (surface) =>
        surface.kind !== 'image' &&
        (surface.continuous || surface.width >= 500) &&
        surface.height >= 700,
    )
  )
    return { kind: 'image' }
  return prose
}
