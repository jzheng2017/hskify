import type { DocumentSourceBlock, SourceSpanKind } from '../contracts/browser'

export type DocumentBlockKind =
  | 'title'
  | 'heading'
  | 'paragraph'
  | 'blockquote'
  | 'ordered-list-item'
  | 'unordered-list-item'
  | 'caption'

export type DocumentTextBlock = {
  itemId: string
  order: number
  kind: DocumentBlockKind
  text: string
  headingLevel?: 1 | 2 | 3 | 4 | 5 | 6
}

export type DocumentImageItem = {
  type: 'image'
  itemId: string
  order: number
  sourceUrl: string
  alt: string
  width?: number
  height?: number
}

export type DocumentSeparatorItem = {
  type: 'separator'
  itemId: string
  order: number
}

export type DocumentStructureItem =
  | ({ type: 'text' } & DocumentTextBlock)
  | DocumentImageItem
  | DocumentSeparatorItem

/** The exact ordered span sent to the native document pipeline. */
export type NativeDocumentBlock = DocumentSourceBlock & { sourceIndex: 0 }

export type DocumentSnapshot = {
  sourceUrl: string
  sourceSha256: string
  title?: string
  characterCount: number
  blocks: NativeDocumentBlock[]
}

export type DocumentChapter = {
  snapshot: DocumentSnapshot
  structure: DocumentStructureItem[]
  sourceRoot: HTMLElement
  /** Live source elements, keyed by translated item ID. */
  sourceElements: ReadonlyMap<string, HTMLElement>
}

export type DocumentRejectionReason =
  | 'readability-rejected'
  | 'too-few-blocks'
  | 'too-short'
  | 'not-predominantly-english'
  | 'unmapped-content'
  | 'unsafe-content-root'
  | 'too-many-blocks'
  | 'block-too-large'
  | 'input-too-large'

export type DocumentDetection =
  | { kind: 'document'; chapter: DocumentChapter }
  | { kind: 'not-document'; reason: DocumentRejectionReason }
  | {
      kind: 'rejected'
      reason: Extract<
        DocumentRejectionReason,
        'too-many-blocks' | 'block-too-large' | 'input-too-large'
      >
    }

export function nativeKindForDocumentBlock(kind: DocumentBlockKind): SourceSpanKind {
  switch (kind) {
    case 'title':
    case 'heading':
      return 'heading'
    case 'blockquote':
      return 'dialogue'
    case 'caption':
      return 'caption'
    case 'paragraph':
    case 'ordered-list-item':
    case 'unordered-list-item':
      return 'prose'
  }
}

export function toNativeDocumentBlocks(
  blocks: readonly DocumentTextBlock[],
): NativeDocumentBlock[] {
  return blocks.map((block, itemOrder) => ({
    itemId: block.itemId,
    sourceIndex: 0,
    itemOrder,
    kind: nativeKindForDocumentBlock(block.kind),
    provenance: 'dom',
    text: block.text,
  }))
}

/**
 * Canonical bytes used by both browser and native code for a document source
 * hash. Do not replace this with JSON serialization: field separators and
 * record separators are part of the wire contract.
 */
export function canonicalDocumentText(blocks: readonly NativeDocumentBlock[]): string {
  return [...blocks]
    .sort((left, right) => left.sourceIndex - right.sourceIndex || left.itemOrder - right.itemOrder)
    .map(
      (block) =>
        `${block.itemId}\u001f${block.sourceIndex}\u001f${block.itemOrder}\u001f${block.kind}\u001f${block.provenance}\u001f${block.text}\u001e`,
    )
    .join('')
}
