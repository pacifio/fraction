import { homedir } from "node:os"
import { join } from "node:path"

import type { EmbeddingProvider } from "./types"
import { toFloat32Vector } from "./utils"

export interface TransformersEmbeddingProviderOptions {
  readonly model?: string
  readonly revision?: string
  readonly cacheDir?: string
  readonly pooling?: "none" | "mean" | "cls" | "first_token" | "eos" | "last_token"
  readonly normalize?: boolean
  readonly dtype?: string
  readonly device?: string
  readonly quantized?: boolean
}

type TensorLike = {
  readonly data?: Float32Array | ArrayLike<number>
  readonly dims?: ReadonlyArray<number>
  readonly tolist?: () => unknown
}

type FeatureExtractionPipelineLike = {
  (input: string | ReadonlyArray<string>, options?: Record<string, unknown>): Promise<unknown>
  readonly dispose?: () => void | Promise<void>
}

const DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2"
const DEFAULT_CACHE_DIR = join(homedir(), ".fraction", "models")

const flattenToNumbers = (value: unknown): Array<number> => {
  if (typeof value === "number") {
    return [value]
  }
  if (Array.isArray(value)) {
    return value.flatMap(flattenToNumbers)
  }
  return []
}

const rowCountFromDims = (dims: ReadonlyArray<number> | undefined, dataLength: number) => {
  if (!dims || dims.length === 0) {
    return 1
  }
  if (dims.length === 1) {
    return 1
  }
  const rows = dims[0] ?? 1
  return rows > 0 ? rows : Math.max(1, dataLength)
}

const vectorsFromTensor = (tensor: TensorLike) => {
  const dataSource = tensor.data ?? flattenToNumbers(tensor.tolist?.())
  const data = Array.from(dataSource)
  const rows = rowCountFromDims(tensor.dims, data.length)
  const width = rows > 0 ? Math.max(1, Math.floor(data.length / rows)) : data.length

  if (rows <= 1) {
    return [Float32Array.from(data)]
  }

  const vectors: Array<Float32Array> = []
  for (let index = 0; index < rows; index++) {
    vectors.push(Float32Array.from(data.slice(index * width, (index + 1) * width)))
  }
  return vectors
}

const vectorsFromUnknown = (value: unknown): ReadonlyArray<Float32Array> => {
  if (value instanceof Float32Array) {
    return [value]
  }
  if (Array.isArray(value)) {
    if (value.every((item) => typeof item === "number")) {
      return [Float32Array.from(value)]
    }
    return value.flatMap(vectorsFromUnknown)
  }
  if (value && typeof value === "object") {
    return vectorsFromTensor(value as TensorLike)
  }
  throw new TypeError("Embedding provider returned an unsupported embedding payload")
}

export const createTransformersEmbeddingProvider = (
  options: TransformersEmbeddingProviderOptions = {}
): EmbeddingProvider => {
  let extractorPromise: Promise<FeatureExtractionPipelineLike> | undefined

  const loadExtractor = async () => {
    const transformers: any = await import("@huggingface/transformers")
    transformers.env.cacheDir = options.cacheDir ?? DEFAULT_CACHE_DIR
    transformers.env.allowRemoteModels = true
    transformers.env.allowLocalModels = true
    const pipelineOptions: Record<string, unknown> = {
      ...(options.revision ? { revision: options.revision } : {}),
      ...(options.dtype ? { dtype: options.dtype } : {}),
      ...(options.device ? { device: options.device } : {}),
      ...(options.quantized !== undefined ? { quantized: options.quantized } : {})
    }

    return transformers.pipeline(
      "feature-extraction",
      options.model ?? DEFAULT_MODEL,
      pipelineOptions
    ) as Promise<FeatureExtractionPipelineLike>
  }

  const getExtractor = () => {
    extractorPromise ??= loadExtractor()
    return extractorPromise
  }

  const runEmbedMany = async (texts: ReadonlyArray<string>) => {
    if (texts.length === 0) {
      return [] as ReadonlyArray<Float32Array>
    }
    const extractor = await getExtractor()
    const output = await extractor(texts.length === 1 ? texts[0]! : texts, {
      pooling: options.pooling ?? "mean",
      normalize: options.normalize ?? true
    })

    const vectors = vectorsFromUnknown(output).map((vector) => toFloat32Vector(vector))
    if (vectors.length === texts.length) {
      return vectors
    }
    if (vectors.length === 1 && texts.length === 1) {
      return vectors
    }
    throw new TypeError(`Expected ${texts.length} embedding vectors but received ${vectors.length}`)
  }

  return {
    embed: async (text) => (await runEmbedMany([text]))[0]!,
    embedMany: runEmbedMany,
    close: async () => {
      if (!extractorPromise) {
        return
      }
      const extractor = await extractorPromise
      await extractor.dispose?.()
      extractorPromise = undefined
    }
  }
}
