import { Schema } from "effect"

import {
  AdapterError,
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
import { createTransformersEmbeddingProvider } from "./internal/providers"
import {
  DEFAULT_CONFIG,
  ExtractionResultSchema,
  FilterLeafSchema,
  FractionConfigSchema,
  HistoryEventSchema,
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
  createTransformersEmbeddingProvider,
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
  EmbeddingProvider,
  ExtractionProvider,
  ExtractionResult,
  FilterExpr,
  FilterLeaf,
  FractionConfig,
  FractionConfigInput,
  HistoryEvent,
  JsonRecord,
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
