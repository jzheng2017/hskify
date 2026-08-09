import type { JobUpdate } from '../contracts/browser'
import { RuntimeMessageError, sendBackgroundMessage } from '../messaging/messages'

export type TerminalJobUpdate = Extract<
  JobUpdate,
  { type: 'complete' | 'failed' | 'cancelled' }
>
export type StreamingJobUpdate = Exclude<JobUpdate, TerminalJobUpdate>

function abortError(): Error {
  const error = new Error('The operation was cancelled.')
  error.name = 'AbortError'
  return error
}

/** One replay-safe poll/install/ack loop shared by both chapter modes. */
export class JobUpdateStream {
  async run(
    jobId: string,
    after: number,
    signal: AbortSignal,
    install: (update: StreamingJobUpdate) => void | Promise<void>,
  ): Promise<Extract<TerminalJobUpdate, { type: 'complete' }>> {
    while (true) {
      if (signal.aborted) throw abortError()
      const batch = await sendBackgroundMessage({ type: 'job:updates', jobId, after })
      if (signal.aborted) throw abortError()
      if (batch.jobId !== jobId) {
        throw new RuntimeMessageError(
          'UPDATE_IDENTITY_MISMATCH',
          'The update batch does not belong to the active chapter source.',
          false,
        )
      }
      if (batch.updates.length === 0) continue
      let terminal: TerminalJobUpdate | undefined
      for (const update of batch.updates) {
        if (signal.aborted) throw abortError()
        if (update.type === 'complete' || update.type === 'failed' || update.type === 'cancelled') {
          terminal = update
        } else {
          await install(update)
        }
      }
      after = batch.nextSequence
      await sendBackgroundMessage({
        type: 'job:ack',
        jobId,
        sequence: after,
        ...(terminal ? { terminalType: terminal.type } : {}),
      })
      if (!terminal) continue
      if (terminal.type === 'failed') {
        throw new RuntimeMessageError(terminal.code, terminal.message, terminal.retryable)
      }
      if (terminal.type === 'cancelled') throw abortError()
      return terminal
    }
  }
}
