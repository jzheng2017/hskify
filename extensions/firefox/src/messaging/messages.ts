import type {
  BrowserJobRequest,
  BrowserJobResult,
  BrowserJobStatus,
  HskLevel,
  LookupRequest,
  LookupResult,
} from '../contracts/browser'

export type TranslationScope = 'visible' | 'all'

export type PopupStartMessage = {
  type: 'popup:start'
  scope: TranslationScope
  hskLevel: HskLevel
}

export type PopupCancelMessage = {
  type: 'popup:cancel'
}

export type PopupStateMessage = {
  type: 'popup:state'
}

export type ContentStartMessage = {
  type: 'content:start'
  scope: TranslationScope
  hskLevel: HskLevel
}

export type ContentCancelMessage = {
  type: 'content:cancel'
}

export type ContentStateMessage = {
  type: 'content:state'
}

export type SubmitImageMessage = {
  type: 'job:submit'
  pageSessionId: string
  pageIndex: number
  imageUrl: string
  pageOrigin: string
  naturalWidth: number
  naturalHeight: number
  sourceMimeType?: string
  sourceBytes?: ArrayBuffer
  hskLevel: HskLevel
  precedingContext?: BrowserJobRequest['precedingContext']
  fixtureMode: boolean
}

export type PollJobMessage = {
  type: 'job:poll'
  jobId: string
}

export type GetJobResultMessage = {
  type: 'job:result'
  jobId: string
}

export type CancelJobMessage = {
  type: 'job:cancel'
  jobId: string
}

export type RecoverJobsMessage = {
  type: 'jobs:recover'
  pageSessionId: string
}

export type LookupMessage = {
  type: 'dictionary:lookup'
  request: LookupRequest
  fixtureMode: boolean
}

export type FontMessage = {
  type: 'font:get'
  fontId: string
  fixtureMode: boolean
}

export type BackgroundRequest =
  | PopupStartMessage
  | PopupCancelMessage
  | PopupStateMessage
  | SubmitImageMessage
  | PollJobMessage
  | GetJobResultMessage
  | CancelJobMessage
  | RecoverJobsMessage
  | LookupMessage
  | FontMessage

export type ContentRequest = ContentStartMessage | ContentCancelMessage | ContentStateMessage

export type PageState = {
  state: 'idle' | 'running' | 'complete' | 'cancelled' | 'failed'
  current: number
  total: number
  stage?: string
  message: string
}

export type SubmittedJob = {
  jobId: string
  clientImageId: string
  sourceSha256: string
}

export type DeliveredJobResult = {
  result: BrowserJobResult
  cleanImage: ArrayBuffer
}

export type FontPayload = {
  fontId: string
  bytes: ArrayBuffer
}

export type RecoveredJob = {
  jobId: string
  clientImageId: string
  sourceSha256: string
  pageIndex: number
  fixtureMode: boolean
  status: BrowserJobStatus
}

export type PopupState = PageState & {
  hskLevel: HskLevel
}

export type MessageError = {
  code: string
  message: string
  retryable: boolean
}

export type MessageResponse<T> =
  | {
      ok: true
      value: T
    }
  | {
      ok: false
      error: MessageError
    }

export type MessageResultMap = {
  'popup:start': PageState
  'popup:cancel': PageState
  'popup:state': PopupState
  'job:submit': SubmittedJob
  'job:poll': BrowserJobStatus
  'job:result': DeliveredJobResult
  'job:cancel': undefined
  'jobs:recover': RecoveredJob[]
  'dictionary:lookup': LookupResult
  'font:get': FontPayload
}

export type RequestOfType<T extends BackgroundRequest['type']> = Extract<
  BackgroundRequest,
  { type: T }
>

export class RuntimeMessageError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable: boolean,
  ) {
    super(message)
    this.name = 'RuntimeMessageError'
  }
}

export async function sendBackgroundMessage<T extends BackgroundRequest['type']>(
  message: RequestOfType<T>,
): Promise<MessageResultMap[T]> {
  const response = (await browser.runtime.sendMessage(message)) as MessageResponse<
    MessageResultMap[T]
  >
  if (!response || typeof response !== 'object') {
    throw new RuntimeMessageError(
      'INVALID_BACKGROUND_RESPONSE',
      'The extension background returned an invalid response.',
      true,
    )
  }
  if (!response.ok) {
    throw new RuntimeMessageError(
      response.error.code,
      response.error.message,
      response.error.retryable,
    )
  }
  return response.value
}
