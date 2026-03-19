import { Schema } from "effect"

import {
  AdapterError,
  CompressionError,
  CompressionUnavailable,
  EmbeddingError,
  ExtractionError,
  InvalidScope,
  MemoryNotFound,
  MigrationError,
  RetrievalError,
  StorageError
} from "./internal/errors"
import {
  AdapterBridgeService,
  CompressionService,
  createFractionLayer,
  createFractionRuntime,
  EmbeddingService,
  ExtractionService,
  type FractionRuntime,
  FractionConfigRef,
  MemoryRepo,
  MemoryService,
  MigrationService,
  parseConfig,
  RetrievalService
} from "./internal/services"
import {
  createHeuristicCompressionProvider,
  createLlmlingua2CompressionProvider,
  createTransformersEmbeddingProvider,
  prefetchLlmlinguaModel
} from "./internal/providers"
import { createLlmlingua2Compressor } from "./internal/llmlingua/compressor"
import {
  CompressionModeSchema,
  CompressionResultSchema,
  CompressionUnavailablePolicySchema,
  DEFAULT_CONFIG,
  ExtractionResultSchema,
  FilterLeafSchema,
  FractionConfigSchema,
  HistoryEventSchema,
  LlmlinguaConfigSchema,
  LlmlinguaModelFamilySchema,
  MemoryIdSchema,
  MemoryRecordSchema,
  MessageSchema,
  RecallOptionsSchema,
  RememberOptionsSchema,
  ScopeSchema,
  SearchResultSchema
} from "./internal/types"

export type { ManagedRuntime } from "effect"
export type { FractionRuntime } from "./internal/services"

export {
  AdapterError,
  AdapterBridgeService,
  CompressionError,
  CompressionModeSchema,
  CompressionResultSchema,
  CompressionService,
  CompressionUnavailable,
  CompressionUnavailablePolicySchema,
  createHeuristicCompressionProvider,
  createLlmlingua2CompressionProvider,
  createLlmlingua2Compressor,
  createTransformersEmbeddingProvider,
  prefetchLlmlinguaModel,
  createFractionLayer,
  createFractionRuntime,
  DEFAULT_CONFIG,
  EmbeddingError,
  EmbeddingService,
  ExtractionError,
  ExtractionService,
  ExtractionResultSchema,
  FilterLeafSchema,
  FractionConfigRef,
  FractionConfigSchema,
  HistoryEventSchema,
  InvalidScope,
  LlmlinguaConfigSchema,
  LlmlinguaModelFamilySchema,
  MemoryIdSchema,
  MemoryNotFound,
  MemoryRecordSchema,
  MemoryRepo,
  MemoryService,
  MessageSchema,
  MigrationError,
  MigrationService,
  parseConfig,
  RecallOptionsSchema,
  RememberOptionsSchema,
  RetrievalError,
  RetrievalService,
  ScopeSchema,
  SearchResultSchema,
  StorageError
}

export type {
  CompressionMode,
  CompressionProvider,
  CompressionResult,
  CompressionUnavailablePolicy,
  EmbeddingProvider,
  ExtractionProvider,
  ExtractionResult,
  FilterExpr,
  FilterLeaf,
  FractionConfig,
  FractionConfigInput,
  HistoryEvent,
  JsonRecord,
  LlmlinguaConfig,
  LlmlinguaModelFamily,
  MemoryId,
  MemoryRecord,
  Message,
  MetadataFor,
  RecallOptions,
  RememberOptions,
  Scope,
  SearchResult
} from "./internal/types"

export const FractionSchemas = {
  MemoryId: MemoryIdSchema,
  Scope: ScopeSchema,
  Message: MessageSchema,
  FractionConfig: FractionConfigSchema,
  CompressionMode: CompressionModeSchema,
  CompressionResult: CompressionResultSchema,
  CompressionUnavailablePolicy: CompressionUnavailablePolicySchema,
  LlmlinguaConfig: LlmlinguaConfigSchema,
  LlmlinguaModelFamily: LlmlinguaModelFamilySchema,
  ExtractionResult: ExtractionResultSchema,
  MemoryRecord: MemoryRecordSchema,
  HistoryEvent: HistoryEventSchema,
  SearchResult: SearchResultSchema,
  RecallOptions: RecallOptionsSchema,
  RememberOptions: RememberOptionsSchema,
  FilterLeaf: FilterLeafSchema
} as const

export const FractionErrors = {
  AdapterError,
  CompressionError,
  CompressionUnavailable,
  EmbeddingError,
  ExtractionError,
  InvalidScope,
  MemoryNotFound,
  MigrationError,
  RetrievalError,
  StorageError
} as const

export const schemaToTanStack = <TSchema extends Schema.Top & { readonly DecodingServices: never }>(
  schema: TSchema
): TSchema & { readonly "~standard": unknown } =>
  Schema.toStandardSchemaV1(schema) as TSchema & { readonly "~standard": unknown }

export const schemaToAiJsonSchema = <TSchema extends Schema.Top>(schema: TSchema) =>
  Schema.toJsonSchemaDocument(schema).schema
