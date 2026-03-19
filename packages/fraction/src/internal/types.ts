import { homedir } from "node:os"
import { join } from "node:path"

import { Schema } from "effect"

import {
  DEFAULT_LLMLINGUA_ARTIFACT_FILES,
  defaultLlmlinguaModelPath
} from "./llmlingua/artifacts"

export const MemoryIdSchema = Schema.String.pipe(Schema.brand("MemoryId"))
export type MemoryId = Schema.Schema.Type<typeof MemoryIdSchema>

export const ScopeSchema = Schema.Struct({
  namespace: Schema.optional(Schema.String),
  userId: Schema.optional(Schema.String),
  agentId: Schema.optional(Schema.String),
  runId: Schema.optional(Schema.String),
  appId: Schema.optional(Schema.String)
})
export type Scope = Schema.Schema.Type<typeof ScopeSchema>

export const MessageSchema = Schema.Struct({
  role: Schema.Literals(["user", "assistant", "system"]),
  content: Schema.String
})
export type Message = Schema.Schema.Type<typeof MessageSchema>

export const FilterLeafSchema = Schema.Struct({
  op: Schema.Literals(["eq", "contains"]),
  field: Schema.String,
  value: Schema.Json
})
export type FilterLeaf = Schema.Schema.Type<typeof FilterLeafSchema>

export type FilterExpr =
  | FilterLeaf
  | {
      readonly op: "and" | "or"
      readonly filters: ReadonlyArray<FilterExpr>
    }

export const RecallOptionsSchema = Schema.Struct({
  limit: Schema.optional(Schema.Number),
  minScore: Schema.optional(Schema.Number),
  includeMetadata: Schema.optional(Schema.Boolean)
})
export type RecallOptions = Schema.Schema.Type<typeof RecallOptionsSchema>

export const RememberOptionsSchema = Schema.Struct({
  mode: Schema.optional(Schema.Literals(["never", "manual", "always"])),
  source: Schema.optional(Schema.Literals(["user", "assistant", "both"])),
  awaitWrite: Schema.optional(Schema.Boolean)
})
export type RememberOptions = Schema.Schema.Type<typeof RememberOptionsSchema>

export const CompressionModeSchema = Schema.Literals(["llmlingua2", "heuristic", "provider", "off"])
export type CompressionMode = Schema.Schema.Type<typeof CompressionModeSchema>

export const CompressionUnavailablePolicySchema = Schema.Literals([
  "fallback-heuristic",
  "error"
])
export type CompressionUnavailablePolicy = Schema.Schema.Type<
  typeof CompressionUnavailablePolicySchema
>

export const LlmlinguaModelFamilySchema = Schema.Literals(["bert", "xlm-roberta"])
export type LlmlinguaModelFamily = Schema.Schema.Type<typeof LlmlinguaModelFamilySchema>

export const CompressionResultSchema = Schema.Struct({
  content: Schema.String,
  mode: CompressionModeSchema,
  source: Schema.Literals(["native", "remote", "fallback", "none"]),
  retainedRatio: Schema.optional(Schema.Number),
  tokenCountBefore: Schema.optional(Schema.Number),
  tokenCountAfter: Schema.optional(Schema.Number)
})
export type CompressionResult = Schema.Schema.Type<typeof CompressionResultSchema>

export const ExtractionResultSchema = Schema.Struct({
  content: Schema.String,
  entities: Schema.Array(Schema.String),
  eventAt: Schema.optional(Schema.String)
})
export type ExtractionResult = Schema.Schema.Type<typeof ExtractionResultSchema>

export interface CompressionProvider {
  readonly compress: (text: string) => CompressionResult | Promise<CompressionResult>
  readonly compressMany?: (
    texts: ReadonlyArray<string>
  ) => ReadonlyArray<CompressionResult> | Promise<ReadonlyArray<CompressionResult>>
  readonly close?: () => void | Promise<void>
}

export interface EmbeddingProvider {
  readonly embed: (
    text: string
  ) => Float32Array | ReadonlyArray<number> | Promise<Float32Array | ReadonlyArray<number>>
  readonly embedMany?: (
    texts: ReadonlyArray<string>
  ) =>
    | ReadonlyArray<Float32Array | ReadonlyArray<number>>
    | Promise<ReadonlyArray<Float32Array | ReadonlyArray<number>>>
  readonly close?: () => void | Promise<void>
}

export interface ExtractionProvider {
  readonly extract: (text: string) => ExtractionResult | Promise<ExtractionResult>
  readonly close?: () => void | Promise<void>
}

export const LlmlinguaConfigSchema = Schema.Struct({
  enabled: Schema.optional(Schema.Boolean),
  model: Schema.optional(Schema.String),
  modelFamily: Schema.optional(LlmlinguaModelFamilySchema),
  cacheDir: Schema.optional(Schema.String),
  revision: Schema.optional(Schema.String),
  batchSize: Schema.optional(Schema.Number),
  device: Schema.optional(Schema.String),
  dtype: Schema.optional(Schema.String),
  downloadModelIfMissing: Schema.optional(Schema.Boolean),
  artifactBaseUrl: Schema.optional(Schema.String),
  artifactFiles: Schema.optional(Schema.Array(Schema.String)),
  onUnavailable: Schema.optional(CompressionUnavailablePolicySchema),
  tokenToWord: Schema.optional(Schema.Literals(["mean", "first"])),
  chunkEndTokens: Schema.optional(Schema.Array(Schema.String)),
  forceTokens: Schema.optional(Schema.Array(Schema.String)),
  forceReserveDigit: Schema.optional(Schema.Boolean),
  dropConsecutive: Schema.optional(Schema.Boolean)
})
type LlmlinguaConfigSchemaInput = Schema.Schema.Type<typeof LlmlinguaConfigSchema>

export interface LlmlinguaConfig extends LlmlinguaConfigSchemaInput {
  readonly enabled: boolean
  readonly model: string
  readonly modelFamily: LlmlinguaModelFamily
  readonly cacheDir: string
  readonly batchSize: number
  readonly downloadModelIfMissing: boolean
  readonly artifactBaseUrl?: string
  readonly artifactFiles: ReadonlyArray<string>
  readonly onUnavailable: CompressionUnavailablePolicy
  readonly tokenToWord: "mean" | "first"
  readonly chunkEndTokens: ReadonlyArray<string>
  readonly forceTokens: ReadonlyArray<string>
  readonly forceReserveDigit: boolean
  readonly dropConsecutive: boolean
}

export const FractionConfigSchema = Schema.Struct({
  filename: Schema.optional(Schema.String),
  defaultNamespace: Schema.optional(Schema.String),
  topK: Schema.optional(Schema.Number),
  rrfK: Schema.optional(Schema.Number),
  embeddingDimensions: Schema.optional(Schema.Number),
  maxFactsPerInput: Schema.optional(Schema.Number),
  compressorType: Schema.optional(CompressionModeSchema),
  compressionRate: Schema.optional(Schema.Number),
  adaptiveCompression: Schema.optional(Schema.Boolean),
  compressionMinChars: Schema.optional(Schema.Number),
  duplicateSimilarity: Schema.optional(Schema.Number),
  lexicalWeight: Schema.optional(Schema.Number),
  vectorWeight: Schema.optional(Schema.Number),
  graphWeight: Schema.optional(Schema.Number),
  temporalWeight: Schema.optional(Schema.Number),
  llmlingua: Schema.optional(LlmlinguaConfigSchema)
})
type FractionConfigSchemaInput = Schema.Schema.Type<typeof FractionConfigSchema>

export interface FractionConfigInput extends FractionConfigSchemaInput {
  readonly compressionProvider?: CompressionProvider
  readonly embeddingProvider?: EmbeddingProvider
  readonly extractionProvider?: ExtractionProvider
}

export interface FractionConfig extends Required<FractionConfigSchemaInput> {
  readonly filename: string
  readonly defaultNamespace: string
  readonly topK: number
  readonly rrfK: number
  readonly embeddingDimensions: number
  readonly maxFactsPerInput: number
  readonly compressorType: CompressionMode
  readonly compressionRate: number
  readonly adaptiveCompression: boolean
  readonly compressionMinChars: number
  readonly duplicateSimilarity: number
  readonly lexicalWeight: number
  readonly vectorWeight: number
  readonly graphWeight: number
  readonly temporalWeight: number
  readonly llmlingua: LlmlinguaConfig
  readonly compressionProvider?: CompressionProvider
  readonly embeddingProvider?: EmbeddingProvider
  readonly extractionProvider?: ExtractionProvider
}

export const MemoryRecordSchema = Schema.Struct({
  id: MemoryIdSchema,
  rootId: Schema.String,
  parentId: Schema.optional(Schema.String),
  version: Schema.Number,
  content: Schema.String,
  rawContent: Schema.String,
  metadata: Schema.Record(Schema.String, Schema.Json),
  scope: ScopeSchema,
  createdAt: Schema.String,
  updatedAt: Schema.String,
  eventAt: Schema.optional(Schema.String),
  forgottenAt: Schema.optional(Schema.String),
  isLatest: Schema.Boolean
})
export type MemoryRecord = Schema.Schema.Type<typeof MemoryRecordSchema>

export const SearchResultSchema = Schema.Struct({
  memory: MemoryRecordSchema,
  score: Schema.Number,
  signals: Schema.Record(Schema.String, Schema.Number)
})
export type SearchResult = Schema.Schema.Type<typeof SearchResultSchema>

export const HistoryEventSchema = Schema.Struct({
  id: Schema.String,
  rootId: Schema.String,
  memoryId: Schema.String,
  event: Schema.Literals(["ADD", "UPDATE", "FORGET", "DELETE", "SKIP"]),
  content: Schema.String,
  createdAt: Schema.String
})
export type HistoryEvent = Schema.Schema.Type<typeof HistoryEventSchema>

export const DEFAULT_CONFIG: FractionConfig = {
  filename: ".fraction/fraction.sqlite",
  defaultNamespace: "default",
  topK: 10,
  rrfK: 30,
  embeddingDimensions: 384,
  maxFactsPerInput: 3,
  compressorType: "llmlingua2",
  compressionRate: 0.6,
  adaptiveCompression: true,
  compressionMinChars: 160,
  duplicateSimilarity: 0.98,
  lexicalWeight: 1.2,
  vectorWeight: 1,
  graphWeight: 0.8,
  temporalWeight: 1,
  llmlingua: {
    enabled: true,
    model: defaultLlmlinguaModelPath(join(homedir(), ".fraction", "models")),
    modelFamily: "bert",
    cacheDir: join(homedir(), ".fraction", "models"),
    batchSize: 16,
    downloadModelIfMissing: true,
    artifactFiles: [...DEFAULT_LLMLINGUA_ARTIFACT_FILES],
    onUnavailable: "fallback-heuristic",
    tokenToWord: "mean",
    chunkEndTokens: [".", "\n"],
    forceTokens: [],
    forceReserveDigit: false,
    dropConsecutive: false,
    ...(process.env.FRACTION_LLMLINGUA_ARTIFACT_BASE_URL
      ? { artifactBaseUrl: process.env.FRACTION_LLMLINGUA_ARTIFACT_BASE_URL }
      : {})
  }
}

export type MetadataFor<TSchema extends Schema.Top | undefined> = TSchema extends Schema.Top
  ? Schema.Schema.Type<TSchema>
  : Record<string, Schema.Json>

export type JsonRecord = Record<string, Schema.Json>
