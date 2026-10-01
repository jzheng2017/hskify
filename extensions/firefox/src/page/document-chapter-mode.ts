import {
  BUILD_FINGERPRINT,
  type TranslationSettings,
  type DocumentSourceBlock,
  type LearningMode,
  type LookupRequest,
  type ReadingDirection,
  type TranslatedText,
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
import { ChapterRunController, type ChapterRunToken } from './chapter-run-controller'

type RetainedBlock = { context: string; text: TranslatedText }
export type RetainedDocument = {
  settings: TranslationSettings
  blocks: ReadonlyMap<string, RetainedBlock>
}
function sourceContext(blocks: readonly DocumentSourceBlock[], index: number): string {
  return blocks
    .slice(Math.max(0, index - 6), index + 7)
    .map((block) => block.itemId + '\u001f' + block.text)
    .join('\u001e')
}

/** The run owns lifecycle; this adapter owns source slots and publication. */
export class DocumentChapterMode {
  private readonly run: ChapterRunController
  private reader: DocumentReader | undefined
  private explanation: ExplanationController | undefined
  private readonly jobIds = new Set<string>()
  private lastVisibleBlockIds: string[] = []
  private invalidated = false
  private readonly sourceBlocks: ReadonlyMap<string, DocumentSourceBlock>
  private settings: TranslationSettings | undefined
  private retained: RetainedDocument | undefined
  private retryChain: Promise<void> = Promise.resolve()

  constructor(
    readonly chapter: DocumentChapter,
    retained?: RetainedDocument,
    private readonly onSourceChanged?: () => void,
  ) {
    this.sourceBlocks = new Map(chapter.snapshot.blocks.map((block) => [block.itemId, block]))
    this.retained = retained
    this.run = new ChapterRunController('document', () => this.restoreMode())
  }
  get isInvalidated(): boolean {
    return this.invalidated
  }
  retainedTranslations(): RetainedDocument | undefined {
    if (!this.settings) return this.retained
    const blocks = new Map<string, RetainedBlock>()
    const translated = this.reader?.translatedBlocks() ?? new Map()
    this.chapter.snapshot.blocks.forEach((block, index) => {
      const text = translated.get(block.itemId)
      if (text)
        blocks.set(block.itemId, {
          text,
          context: sourceContext(this.chapter.snapshot.blocks, index),
        })
    })
    return blocks.size ? { settings: this.settings, blocks } : this.retained
  }
  async start(
    _scope: TranslationScope,
    hskLevel: 1 | 2 | 3 | 4 | 5 | 6,
    learningMode: LearningMode,
    _readingDirection: ReadingDirection,
  ): Promise<PageState> {
    const previous = this.retainedTranslations()
    const settings: TranslationSettings = {
      sourceLanguage: 'en',
      targetLanguage: 'zh-CN',
      hskStandard: '2.0',
      hskLevel,
      learningMode,
    }
    this.settings = settings
    const total = this.chapter.snapshot.blocks.length
    const token = await this.run.start(total, 'Preparing the light-novel reader')
    try {
      const reader = mountDocumentReader(this.chapter, {
        onVisibleBlocksChanged: (ids) => this.updateFocus(ids),
        onInvalidated: () => this.invalidateSource(),
        onRetry: (id) => this.retry(token, id),
        attachTranslatedText: (element, itemId) => {
          this.explanation?.register(element, '', itemId)
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
      const retainedIds = new Set<string>()
      if (
        previous &&
        previous.settings.hskLevel === hskLevel &&
        previous.settings.learningMode === learningMode
      ) {
        this.chapter.snapshot.blocks.forEach((block, index) => {
          const value = previous.blocks.get(block.itemId)
          if (value && value.context === sourceContext(this.chapter.snapshot.blocks, index)) {
            reader.installBlock(block.itemId, value.text)
            retainedIds.add(block.itemId)
          }
        })
      }
      const pending = this.chapter.snapshot.blocks
        .filter((block) => !retainedIds.has(block.itemId))
        .map((block) => block.itemId)
      if (pending.length)
        await this.process(token, reader, settings, retainedIds.size ? pending : [])
      this.run.assertCurrent(token)
      return this.finish()
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') throw error
      if (!this.reader)
        this.run.fail(error instanceof Error ? error.message : 'The reader could not be mounted.', {
          current: 0,
          total,
        })
      else
        this.failPending(
          this.chapter.snapshot.blocks.map((block) => block.itemId),
          error,
        )
      return this.snapshot()
    }
  }
  cancel(): PageState {
    const counts = this.reader?.counts()
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

  private async process(
    token: ChapterRunToken,
    reader: DocumentReader,
    settings: TranslationSettings,
    retryItemIds: string[],
  ): Promise<void> {
    await reader.waitForInitialFocus(token.signal)
    this.run.assertCurrent(token)
    this.lastVisibleBlockIds = reader.visibleItemIds()
    const target = new Set(
      retryItemIds.length
        ? retryItemIds
        : this.chapter.snapshot.blocks.map((block) => block.itemId),
    )
    const recovered = retryItemIds.length
      ? []
      : await this.run.recover(token, [
          { kind: 'document', sourceSha256: this.chapter.snapshot.sourceSha256, settings },
        ])
    const resumed = recovered.find(
      (job): job is RecoveredDocumentJob =>
        job.kind === 'document' && job.sourceSha256 === this.chapter.snapshot.sourceSha256,
    )
    const submitted =
      resumed ??
      (await sendBackgroundMessage({
        type: 'job:submit-document',
        pageUrl: location.href,
        request: {
          clientRequestId: crypto.randomUUID(),
          retryItemIds,
          focus: {
            kind: 'document',
            active: !document.hidden,
            visibleBlockIds: this.lastVisibleBlockIds,
          },
          buildFingerprint: BUILD_FINGERPRINT,
          pageSessionId: token.pageSessionId,
          sourceSha256: this.chapter.snapshot.sourceSha256,
          settings,
          blocks: this.chapter.snapshot.blocks,
        },
      }))
    this.run.registerJob(token, submitted.jobId)
    this.jobIds.add(submitted.jobId)
    const published = new Set<string>()
    try {
      this.updateFocus(this.lastVisibleBlockIds)
      await sendBackgroundMessage({
        type: 'chapter:source',
        pageSessionId: token.pageSessionId,
        pageUrl: token.pageUrl,
        sourceIndex: 0,
      })
      const terminal = await this.run.stream(
        token,
        submitted.jobId,
        submitted.acknowledgedSequence,
        token.signal,
        (update) => {
          switch (update.type) {
            case 'progress':
              this.updateProgress(submitted.jobId, update)
              break
            case 'documentBlockReady':
              this.assertBlockIdentity(update.block, target)
              if (published.has(update.block.itemId))
                throw new Error('The job repeated a sentence group.')
              reader.installBlock(update.block.itemId, update.block.text)
              published.add(update.block.itemId)
              this.updateProgress()
              break
            case 'documentBlockPreserved':
              this.assertBlockIdentity(update.block, target)
              if (published.has(update.block.itemId))
                throw new Error('The job repeated a sentence group.')
              reader.preserveBlock(
                update.block.itemId,
                update.block.sourceText,
                update.block.reason,
              )
              published.add(update.block.itemId)
              this.updateProgress()
              break
            default:
              throw new RuntimeMessageError(
                'JOB_MODALITY_MISMATCH',
                'An image update was returned for a document job.',
                false,
              )
          }
        },
      )
      if (
        published.size !== target.size ||
        terminal.translatedCount + terminal.preservedCount !== published.size
      )
        throw new Error('The job did not account for every requested sentence group.')
    } catch (error) {
      this.run.cancelJob(submitted.jobId)
      throw error
    } finally {
      this.jobIds.delete(submitted.jobId)
    }
  }
  private retry(token: ChapterRunToken, id: string): void {
    this.retryChain = this.retryChain
      .catch(() => undefined)
      .then(async () => {
        const reader = this.reader,
          settings = this.settings
        if (!reader || !settings) return
        try {
          this.run.assertCurrent(token)
          await this.process(token, reader, settings, [id])
          this.finish()
        } catch (error) {
          if (!(error instanceof Error && error.name === 'AbortError'))
            this.failPending([id], error)
        }
      })
  }
  private failPending(ids: readonly string[], error: unknown): void {
    const reason = error instanceof Error ? error.message : 'Translation failed.'
    const translated = this.reader?.translatedBlocks()
    for (const id of ids) {
      if (translated?.has(id)) continue
      const source = this.sourceBlocks.get(id)
      if (source && this.reader && !this.reader.isDestroyed)
        this.reader.preserveBlock(id, source.text, reason)
    }
    this.finish()
  }
  private finish(): PageState {
    const counts = this.reader?.counts() ?? {
      translated: 0,
      preserved: 0,
      pending: this.chapter.snapshot.blocks.length,
    }
    return this.run.finish(
      { current: counts.translated + counts.preserved, total: this.chapter.snapshot.blocks.length },
      counts.preserved
        ? counts.preserved + ' sentence groups failed. Use Retry or Original.'
        : undefined,
      true,
    )
  }
  private updateProgress(
    key?: string,
    status?: Parameters<ChapterRunController['update']>[0]['status'],
  ): void {
    const counts = this.reader?.counts()
    if (!counts) return
    this.run.update({
      current: counts.translated + counts.preserved,
      total: this.chapter.snapshot.blocks.length,
      message:
        counts.translated +
        ' translated · ' +
        counts.preserved +
        ' failed · ' +
        counts.pending +
        ' pending',
      ...(key ? { key } : {}),
      ...(status ? { status } : {}),
    })
  }
  private assertBlockIdentity(
    block: Pick<
      DocumentSourceBlock,
      'itemId' | 'sourceIndex' | 'itemOrder' | 'kind' | 'parentBlockId' | 'subItemOrder'
    >,
    target: ReadonlySet<string>,
  ): void {
    const source = this.sourceBlocks.get(block.itemId)
    if (
      !source ||
      !target.has(block.itemId) ||
      source.sourceIndex !== block.sourceIndex ||
      source.itemOrder !== block.itemOrder ||
      source.kind !== block.kind ||
      source.parentBlockId !== block.parentBlockId ||
      source.subItemOrder !== block.subItemOrder
    ) {
      throw new RuntimeMessageError(
        'DOCUMENT_BLOCK_IDENTITY_MISMATCH',
        'A result does not belong to its extracted source group.',
        false,
      )
    }
  }
  private updateFocus(ids: readonly string[]): void {
    this.lastVisibleBlockIds = [...new Set(ids)].slice(0, 64)
    for (const jobId of this.jobIds)
      void sendBackgroundMessage({
        type: 'job:focus',
        jobId,
        focus: {
          kind: 'document',
          visibleBlockIds: document.hidden ? [] : this.lastVisibleBlockIds,
          active: !document.hidden,
        },
      }).catch(() => undefined)
  }
  private invalidateSource(): void {
    this.retained = this.retainedTranslations()
    this.invalidated = true
    this.run.fail('Source changed; updating the chapter.', {
      current: 0,
      total: this.chapter.snapshot.blocks.length,
    })
    this.onSourceChanged?.()
  }
  private restoreMode(): void {
    this.jobIds.clear()
    this.explanation?.destroy()
    this.explanation = undefined
    this.reader?.destroy()
    this.reader = undefined
  }
}
