import {
  BUILD_FINGERPRINT,
  type DocumentSourceBlock,
  type LearningMode,
  type LookupRequest,
  type ReadingDirection,
} from '../contracts/browser'
import { mountDocumentReader, type DocumentChapter, type DocumentReader } from '../document'
import {
  sendBackgroundMessage,
  RuntimeMessageError,
  type PageState,
  type RecoveredDocumentJob,
  type TranslationScope,
} from '../messaging/messages'
import { ExplanationController } from '../selection/popover'
import { MandarinSpeaker } from '../selection/speech'
import { ChapterRunController } from './chapter-run-controller'

export class DocumentChapterMode {
  private readonly run: ChapterRunController
  private reader: DocumentReader | undefined
  private explanation: ExplanationController | undefined
  private jobId: string | undefined
  private lookupJobId: string | undefined
  private lastVisibleBlockIds: string[] = []
  private invalidated = false
  private readonly sourceBlocks: ReadonlyMap<string, DocumentSourceBlock>

  constructor(readonly chapter: DocumentChapter) {
    this.sourceBlocks = new Map(chapter.snapshot.blocks.map((block) => [block.itemId, block]))
    this.run = new ChapterRunController('document', () => this.restoreMode())
  }

  get isInvalidated(): boolean {
    return this.invalidated
  }

  async start(
    _scope: TranslationScope,
    hskLevel: 1 | 2 | 3 | 4 | 5 | 6,
    learningMode: LearningMode,
    _readingDirection: ReadingDirection,
  ): Promise<PageState> {
    const total = this.chapter.snapshot.blocks.length
    const settings = {
      sourceLanguage: 'en' as const,
      targetLanguage: 'zh-CN' as const,
      hskStandard: '2.0' as const,
      hskLevel,
      learningMode,
    }
    const token = await this.run.start(total, 'Preparing the light-novel reader')

    let reader: DocumentReader
    try {
      reader = mountDocumentReader(this.chapter, {
        onVisibleBlocksChanged: (itemIds) => this.updateFocus(itemIds),
        onInvalidated: () => this.invalidateSource(),
        attachTranslatedText: (element, itemId) => {
          this.explanation?.register(element, this.lookupJobId ?? '', itemId)
          return () => this.explanation?.unregister(element)
        },
      })
      this.reader = reader
      this.explanation = new ExplanationController(
        reader.interactionRoot,
        reader.lookupElement,
        (request: LookupRequest) => sendBackgroundMessage({ type: 'dictionary:lookup', request }),
        undefined,
        new MandarinSpeaker(),
      )

      const recovered = await this.run.recover(token, [
        {
          kind: 'document',
          sourceSha256: this.chapter.snapshot.sourceSha256,
          settings,
        },
      ])
      const recoveredDocument = recovered.find(
        (job): job is RecoveredDocumentJob =>
          job.kind === 'document' && job.sourceSha256 === this.chapter.snapshot.sourceSha256,
      )
      const submitted =
        recoveredDocument ??
        (await sendBackgroundMessage({
          type: 'job:submit-document',
          pageUrl: location.href,
          request: {
            buildFingerprint: BUILD_FINGERPRINT,
            pageSessionId: token.pageSessionId,
            sourceSha256: this.chapter.snapshot.sourceSha256,
            settings,
            blocks: this.chapter.snapshot.blocks,
          },
        }))
      this.run.assertCurrent(token)
      this.jobId = submitted.jobId
      this.lookupJobId = submitted.jobId
      this.run.registerJob(token, submitted.jobId)
      await sendBackgroundMessage({
        type: 'chapter:source',
        pageSessionId: token.pageSessionId,
        pageUrl: token.pageUrl,
        sourceIndex: 0,
      })
      this.updateFocus(this.lastVisibleBlockIds, true)

      const terminal = await this.run.stream(
        token,
        submitted.jobId,
        submitted.acknowledgedSequence,
        token.signal,
        (update) => {
          this.run.assertCurrent(token)
          switch (update.type) {
            case 'progress': {
              const counts = reader.counts()
              this.run.update({
                current: counts.translated + counts.preserved,
                total,
                key: submitted.jobId,
                status: update,
              })
              break
            }
            case 'documentBlockReady':
              this.assertBlockIdentity(update.block)
              reader.installBlock(update.block.itemId, update.block.text)
              break
            case 'documentBlockPreserved':
              this.assertBlockIdentity(update.block)
              reader.preserveBlock(
                update.block.itemId,
                update.block.sourceText,
                update.block.reason,
              )
              break
            case 'imageRegionReady':
            case 'imageRegionPreserved':
              throw new RuntimeMessageError(
                'JOB_MODALITY_MISMATCH',
                'An image update was returned for a document job.',
                false,
              )
          }
        },
      )
      this.run.assertCurrent(token)
      const counts = reader.counts()
      if (
        counts.pending !== 0 ||
        counts.translated !== terminal.translatedCount ||
        counts.preserved !== terminal.preservedCount
      ) {
        throw new RuntimeMessageError(
          'DOCUMENT_COMPLETION_MISMATCH',
          'The completed job does not account for every document block.',
          false,
        )
      }
      this.jobId = undefined
      return this.run.finish({
        current: counts.translated + counts.preserved,
        total,
      })
    } catch (error) {
      this.jobId = undefined
      if (error instanceof Error && error.name === 'AbortError') throw error
      this.run.fail(error instanceof Error ? error.message : 'The document translation failed.', {
        current: 0,
        total,
      })
      throw error
    }
  }

  cancel(): PageState {
    const counts = this.reader?.counts()
    if (this.jobId) this.updateFocus([], true, false)
    return this.run.cancel({
      current: counts ? counts.translated + counts.preserved : 0,
      total: this.chapter.snapshot.blocks.length,
    })
  }

  snapshot(): PageState {
    return this.run.snapshot()
  }

  destroy(): void {
    this.run.destroy()
  }

  private assertBlockIdentity(
    block: Pick<DocumentSourceBlock, 'itemId' | 'sourceIndex' | 'itemOrder' | 'kind'>,
  ): void {
    const source = this.sourceBlocks.get(block.itemId)
    if (
      !source ||
      source.sourceIndex !== block.sourceIndex ||
      source.itemOrder !== block.itemOrder ||
      source.kind !== block.kind
    ) {
      throw new RuntimeMessageError(
        'DOCUMENT_BLOCK_IDENTITY_MISMATCH',
        'A document update does not belong to its extracted source block.',
        false,
      )
    }
  }

  private updateFocus(itemIds: readonly string[], force = false, active = true): void {
    this.lastVisibleBlockIds = [...new Set(itemIds)].slice(0, 64)
    if (!this.jobId || (!force && !this.run.isRunning)) return
    void sendBackgroundMessage({
      type: 'job:focus',
      jobId: this.jobId,
      focus: { kind: 'document', visibleBlockIds: this.lastVisibleBlockIds, active },
    }).catch(() => undefined)
  }

  private invalidateSource(): void {
    if (!this.reader) return
    this.invalidated = true
    this.run.fail('The source chapter changed. The original page was restored.', {
      current: 0,
      total: this.chapter.snapshot.blocks.length,
    })
  }

  private restoreMode(): void {
    if (this.jobId) {
      void sendBackgroundMessage({
        type: 'job:focus',
        jobId: this.jobId,
        focus: { kind: 'document', visibleBlockIds: [], active: false },
      }).catch(() => undefined)
    }
    this.jobId = undefined
    this.lookupJobId = undefined
    this.explanation?.destroy()
    this.explanation = undefined
    this.reader?.destroy()
    this.reader = undefined
  }
}
