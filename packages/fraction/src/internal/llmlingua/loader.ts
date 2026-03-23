import { Effect } from "effect"

import { CompressionUnavailable } from "../errors"

import { ensureLlmlinguaLocalModelPath } from "./artifacts"
import { inferLlmlinguaModelFamily } from "./token-boundaries"
import type {
  LlmlinguaModelLike,
  LlmlinguaRuntimeOptions,
  LlmlinguaSession,
  LlmlinguaTokenizerLike
} from "./types"

export const loadLlmlinguaSession = async (
  options: LlmlinguaRuntimeOptions
): Promise<LlmlinguaSession> =>
  Effect.runPromise(
    Effect.gen(function* () {
      const transformers: any = yield* Effect.tryPromise({
        try: () => import("@huggingface/transformers"),
        catch: (cause) => cause
      })
      const modelPath = yield* Effect.tryPromise({
        try: () =>
          ensureLlmlinguaLocalModelPath({
            model: options.model,
            ...(options.cacheDir ? { cacheDir: options.cacheDir } : {}),
            ...(options.downloadModelIfMissing !== undefined
              ? { downloadModelIfMissing: options.downloadModelIfMissing }
              : {}),
            ...(options.artifactBaseUrl ? { artifactBaseUrl: options.artifactBaseUrl } : {}),
            ...(options.artifactFiles ? { artifactFiles: options.artifactFiles } : {})
          }),
        catch: (cause) => cause
      })

      yield* Effect.sync(() => {
        transformers.env.cacheDir = options.cacheDir
        transformers.env.allowLocalModels = true
        transformers.env.allowRemoteModels =
          /^https?:\/\//.test(modelPath) || modelPath === options.model
      })

      const pretrainedOptions = {
        ...(options.cacheDir ? { cache_dir: options.cacheDir } : {}),
        ...(options.revision ? { revision: options.revision } : {}),
        ...(modelPath !== options.model || options.downloadModelIfMissing === false
          ? { local_files_only: true }
          : {})
      }

      const config = (yield* Effect.tryPromise({
        try: () => transformers.AutoConfig.from_pretrained(modelPath, pretrainedOptions),
        catch: (cause) => cause
      })) as {
        readonly model_type?: string | null
        readonly max_position_embeddings?: number
      }
      const tokenizer = (yield* Effect.tryPromise({
        try: () => transformers.AutoTokenizer.from_pretrained(modelPath, pretrainedOptions),
        catch: (cause) => cause
      })) as LlmlinguaTokenizerLike
      const model = (yield* Effect.tryPromise({
        try: () =>
          transformers.AutoModelForTokenClassification.from_pretrained(modelPath, {
            ...pretrainedOptions,
            ...(options.device ? { device: options.device } : {}),
            ...(options.dtype ? { dtype: options.dtype } : {})
          }),
        catch: (cause) => cause
      })) as LlmlinguaModelLike

      return {
        tokenizer,
        model,
        modelFamily:
          options.modelFamily ?? inferLlmlinguaModelFamily(config?.model_type, options.model),
        maxSeqLength: Math.min(config?.max_position_embeddings ?? 512, 512),
        dispose: async () => {
          await model.dispose?.()
        }
      } satisfies LlmlinguaSession
    }).pipe(
      Effect.mapError((cause) =>
        cause instanceof CompressionUnavailable
          ? cause
          : new CompressionUnavailable({
              message: "Failed to load LLMLingua-2 model artifacts",
              model: options.model,
              cause
            })
      )
    )
  )
