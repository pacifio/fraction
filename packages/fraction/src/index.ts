import { Layer, ManagedRuntime, Schema } from "effect"

import {
  MemoryService,
  MigrationService,
  createFractionRuntime,
  parseConfig
} from "./internal/services"
import {
  createHeuristicCompressionProvider,
  createLlmlingua2CompressionProvider,
  createTransformersEmbeddingProvider,
  prefetchLlmlinguaModel,
  type HeuristicCompressionProviderOptions,
  type TransformersEmbeddingProviderOptions
} from "./internal/providers"
import { createLlmlingua2Compressor } from "./internal/llmlingua/compressor"
import type {
  CompressionProvider,
  CompressionResult,
  CompressionUnavailablePolicy,
  EmbeddingProvider,
  ExtractionProvider,
  FilterExpr,
  FractionConfig,
  FractionConfigInput,
  HistoryEvent,
  JsonRecord,
  MemoryRecord,
  Message,
  MetadataFor,
  RecallOptions,
  RememberOptions,
  LlmlinguaConfig,
  LlmlinguaModelFamily,
  Scope,
  SearchResult
} from "./internal/types"
import { normalizeScope } from "./internal/utils"

export type {
  CompressionProvider,
  CompressionResult,
  CompressionUnavailablePolicy,
  EmbeddingProvider,
  ExtractionProvider,
  FilterExpr,
  FractionConfig,
  FractionConfigInput,
  HistoryEvent,
  JsonRecord,
  MemoryRecord,
  Message,
  RecallOptions,
  RememberOptions,
  LlmlinguaConfig,
  LlmlinguaModelFamily,
  Scope,
  SearchResult
} from "./internal/types"
export type {
  HeuristicCompressionProviderOptions,
  TransformersEmbeddingProviderOptions
} from "./internal/providers"
export {
  createHeuristicCompressionProvider,
  createLlmlingua2Compressor,
  createLlmlingua2CompressionProvider,
  createTransformersEmbeddingProvider,
  prefetchLlmlinguaModel
}

export interface SearchOptions {
  readonly limit?: number
  readonly filter?: FilterExpr
}

export interface MemoryClient<TMetadata = Record<string, unknown>> {
  add(
    input: string | ReadonlyArray<Message>,
    scope?: Scope,
    metadata?: TMetadata
  ): Promise<MemoryRecord>
  addMany(
    inputs: ReadonlyArray<string>,
    scope?: Scope,
    metadata?: TMetadata
  ): Promise<ReadonlyArray<MemoryRecord>>
  search(
    query: string,
    scope?: Scope,
    options?: SearchOptions
  ): Promise<ReadonlyArray<SearchResult>>
  get(id: string): Promise<MemoryRecord>
  getAll(scope?: Scope): Promise<ReadonlyArray<MemoryRecord>>
  update(id: string, input: string, metadata?: TMetadata): Promise<MemoryRecord>
  forget(id: string): Promise<void>
  delete(id: string): Promise<void>
  deleteAll(scope?: Scope): Promise<void>
  history(id: string): Promise<ReadonlyArray<HistoryEvent>>
  reset(): Promise<void>
  close(): Promise<void>
}

export interface FractionClient<
  TMetadata = Record<string, unknown>
> extends MemoryClient<TMetadata> {
  readonly runtime: ManagedRuntime.ManagedRuntime<any, any>
  readonly config: FractionConfig
}

const decodeMetadata = <TSchema extends Schema.Top | undefined>(
  schema: TSchema | undefined,
  metadata: unknown
) => {
  if (schema === undefined || metadata === undefined) {
    return metadata as MetadataFor<TSchema> | undefined
  }
  return Schema.decodeUnknownSync(schema as TSchema & { readonly DecodingServices: never })(
    metadata
  ) as MetadataFor<TSchema>
}

class FractionClientImpl<
  TSchema extends Schema.Top | undefined,
  TMetadata
> implements FractionClient<TMetadata> {
  constructor(
    public readonly runtime: ManagedRuntime.ManagedRuntime<any, any>,
    public readonly config: FractionConfig,
    private readonly metadataSchema?: TSchema
  ) {}

  add(input: string | ReadonlyArray<Message>, scope?: Scope, metadata?: TMetadata) {
    const validated = decodeMetadata(this.metadataSchema, metadata) as JsonRecord | undefined
    return this.runtime.runPromise(
      MemoryService.use((service) =>
        service.add(input, normalizeScope(scope, this.config.defaultNamespace), validated)
      )
    )
  }

  addMany(inputs: ReadonlyArray<string>, scope?: Scope, metadata?: TMetadata) {
    const validated = decodeMetadata(this.metadataSchema, metadata) as JsonRecord | undefined
    return this.runtime.runPromise(
      MemoryService.use((service) =>
        service.addMany(inputs, normalizeScope(scope, this.config.defaultNamespace), validated)
      )
    )
  }

  search(query: string, scope?: Scope, options?: SearchOptions) {
    return this.runtime.runPromise(
      MemoryService.use((service) =>
        service.search(query, normalizeScope(scope, this.config.defaultNamespace), options)
      )
    )
  }

  get(id: string) {
    return this.runtime.runPromise(MemoryService.use((service) => service.get(id)))
  }

  getAll(scope?: Scope) {
    return this.runtime.runPromise(
      MemoryService.use((service) =>
        service.getAll(normalizeScope(scope, this.config.defaultNamespace))
      )
    )
  }

  update(id: string, input: string, metadata?: TMetadata) {
    const validated = decodeMetadata(this.metadataSchema, metadata) as JsonRecord | undefined
    return this.runtime.runPromise(
      MemoryService.use((service) => service.update(id, input, validated))
    )
  }

  forget(id: string) {
    return this.runtime.runPromise(MemoryService.use((service) => service.forget(id)))
  }

  delete(id: string) {
    return this.runtime.runPromise(MemoryService.use((service) => service.delete(id)))
  }

  deleteAll(scope?: Scope) {
    return this.runtime.runPromise(
      MemoryService.use((service) =>
        service.deleteAll(normalizeScope(scope, this.config.defaultNamespace))
      )
    )
  }

  history(id: string) {
    return this.runtime.runPromise(MemoryService.use((service) => service.history(id)))
  }

  reset() {
    return this.runtime.runPromise(MemoryService.use((service) => service.reset))
  }

  close() {
    return this.runtime.dispose()
  }
}

export interface OpenOptions<
  TSchema extends Schema.Top | undefined = undefined
> extends FractionConfigInput {
  readonly metadataSchema?: TSchema
  readonly memoMap?: Layer.MemoMap
}

export class Fraction {
  static async open<TSchema extends Schema.Top | undefined = undefined>(
    options?: OpenOptions<TSchema>
  ): Promise<FractionClient<MetadataFor<TSchema>>> {
    const config = parseConfig(options)
    const runtime = createFractionRuntime(
      config,
      options?.memoMap ? { memoMap: options.memoMap } : undefined
    )
    await runtime.runPromise(MigrationService.use((migration) => migration.run))
    return new FractionClientImpl<TSchema, MetadataFor<TSchema>>(
      runtime,
      config,
      options?.metadataSchema
    )
  }
}

export class Memory {
  static async open<TSchema extends Schema.Top | undefined = undefined>(
    options?: OpenOptions<TSchema>
  ): Promise<MemoryClient<MetadataFor<TSchema>>> {
    return Fraction.open(options)
  }
}
