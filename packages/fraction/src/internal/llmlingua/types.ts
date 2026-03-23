export interface LlmlinguaRuntimeOptions {
  readonly model: string
  readonly modelFamily: "bert" | "xlm-roberta"
  readonly cacheDir?: string
  readonly revision?: string
  readonly device?: string
  readonly dtype?: string
  readonly downloadModelIfMissing?: boolean
  readonly artifactBaseUrl?: string
  readonly artifactFiles?: ReadonlyArray<string>
}

export interface LlmlinguaCompressOptions {
  readonly rate?: number
  readonly targetToken?: number
  readonly tokenToWord?: "mean" | "first"
  readonly forceTokens?: ReadonlyArray<string>
  readonly forceReserveDigit?: boolean
  readonly dropConsecutive?: boolean
  readonly chunkEndTokens?: ReadonlyArray<string>
}

export interface LlmlinguaPromptOptions extends LlmlinguaCompressOptions {
  readonly useContextLevelFilter?: boolean
  readonly useTokenLevelFilter?: boolean
  readonly targetContext?: number
  readonly contextLevelRate?: number
  readonly contextLevelTargetToken?: number
  readonly forceContextIds?: ReadonlyArray<number>
}

export interface LlmlinguaTokenizerLike {
  readonly special_tokens: ReadonlyArray<string>
  readonly model: {
    convert_ids_to_tokens(ids: number[] | bigint[]): ReadonlyArray<string>
  }
  readonly decoder: {
    decode(tokens: ReadonlyArray<string>): string
  }
  tokenize(
    text: string,
    options?: {
      readonly add_special_tokens?: boolean
    }
  ): ReadonlyArray<string>
  (
    text: string | ReadonlyArray<string>,
    options?: {
      readonly padding?: boolean | "max_length"
      readonly truncation?: boolean
      readonly max_length?: number
      readonly return_tensor?: boolean
    }
  ): Promise<{
    readonly input_ids: {
      readonly data: ArrayLike<number | bigint>
      readonly dims: ReadonlyArray<number>
    }
    readonly attention_mask: {
      readonly data: ArrayLike<number | bigint>
      readonly dims: ReadonlyArray<number>
    }
  }>
}

export interface LlmlinguaModelLike {
  readonly config?: {
    readonly max_position_embeddings?: number
    readonly model_type?: string | null
  }
  (
    input: {
      readonly input_ids: unknown
      readonly attention_mask: unknown
    }
  ): Promise<{
    readonly logits: {
      readonly data: ArrayLike<number>
      readonly dims: ReadonlyArray<number>
    }
  }>
  readonly dispose?: () => Promise<unknown> | void
}

export interface LlmlinguaSession {
  readonly tokenizer: LlmlinguaTokenizerLike
  readonly model: LlmlinguaModelLike
  readonly modelFamily: "bert" | "xlm-roberta"
  readonly maxSeqLength: number
  readonly dispose: () => Promise<void>
}

export interface LlmlinguaCompressor {
  readonly compress: (
    text: string,
    options?: LlmlinguaCompressOptions
  ) => Promise<{
    readonly content: string
    readonly retainedRatio?: number
    readonly tokenCountBefore: number
    readonly tokenCountAfter: number
  }>
  readonly compressPrompt: (
    context: string | ReadonlyArray<string>,
    options?: LlmlinguaPromptOptions
  ) => Promise<{
    readonly content: string
    readonly retainedRatio?: number
    readonly tokenCountBefore: number
    readonly tokenCountAfter: number
  }>
  readonly close: () => Promise<void>
}
