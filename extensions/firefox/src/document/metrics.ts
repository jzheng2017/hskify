export const DOCUMENT_EXTRACTION_START_MARK = 'hskify-document-extraction-start'
export const DOCUMENT_EXTRACTION_END_MARK = 'hskify-document-extraction-end'
export const DOCUMENT_EXTRACTION_MEASURE = 'hskify-document-extraction'
export const DOCUMENT_SKELETON_MOUNTED_MARK = 'hskify-document-skeleton-mounted'
export const DOCUMENT_EXTRACTION_AND_SKELETON_MEASURE = 'hskify-document-extraction-and-skeleton'

export function markDocumentPerformance(
  performanceApi: Performance | undefined,
  name: string,
): void {
  try {
    performanceApi?.mark?.(name)
  } catch {
    // User Timing must never affect chapter detection or restoration.
  }
}

export function measureDocumentPerformance(
  performanceApi: Performance | undefined,
  name: string,
  start: string,
  end: string,
): void {
  try {
    performanceApi?.measure?.(name, start, end)
  } catch {
    // A browser may evict marks from its bounded performance buffer.
  }
}
