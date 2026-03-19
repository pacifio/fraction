import { CompressionUnavailable } from "../errors"

import { ensureLlmlinguaLocalModelPath } from "./artifacts"
import { inferLlmlinguaModelFamily } from "./token-boundaries"
import type { LlmlinguaRuntimeOptions, LlmlinguaSession } from "./types"

export const loadLlmlinguaSession = async (
  options: LlmlinguaRuntimeOptions
): Promise<LlmlinguaSession> => {
  try {
    const transformers: any = await import("@huggingface/transformers")
    const modelPath = await ensureLlmlinguaLocalModelPath({
      model: options.model,
      ...(options.cacheDir ? { cacheDir: options.cacheDir } : {}),
      ...(options.downloadModelIfMissing !== undefined
        ? { downloadModelIfMissing: options.downloadModelIfMissing }
        : {}),
      ...(options.artifactBaseUrl ? { artifactBaseUrl: options.artifactBaseUrl } : {}),
      ...(options.artifactFiles ? { artifactFiles: options.artifactFiles } : {})
    })
    transformers.env.cacheDir = options.cacheDir
    transformers.env.allowLocalModels = true
    transformers.env.allowRemoteModels = /^https?:\/\//.test(modelPath) || modelPath === options.model

    const pretrainedOptions = {
      ...(options.cacheDir ? { cache_dir: options.cacheDir } : {}),
      ...(options.revision ? { revision: options.revision } : {}),
      ...(modelPath !== options.model || options.downloadModelIfMissing === false
        ? { local_files_only: true }
        : {})
    }

    const config = await transformers.AutoConfig.from_pretrained(modelPath, pretrainedOptions)
    const tokenizer = await transformers.AutoTokenizer.from_pretrained(modelPath, pretrainedOptions)
    const model = await transformers.AutoModelForTokenClassification.from_pretrained(
      modelPath,
      {
        ...pretrainedOptions,
        ...(options.device ? { device: options.device } : {}),
        ...(options.dtype ? { dtype: options.dtype } : {})
      }
    )

    return {
      tokenizer,
      model,
      modelFamily:
        options.modelFamily ?? inferLlmlinguaModelFamily(config?.model_type, options.model),
      maxSeqLength: Math.min(config?.max_position_embeddings ?? 512, 512),
      dispose: async () => {
        await model.dispose?.()
      }
    }
  } catch (cause) {
    throw new CompressionUnavailable({
      message: "Failed to load LLMLingua-2 model artifacts",
      model: options.model,
      cause
    })
  }
}
