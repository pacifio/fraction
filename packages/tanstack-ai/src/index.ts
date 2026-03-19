import { Effect, Schema } from "effect"
import {
  summarize,
  toolDefinition,
  type ChatMiddleware,
  type ChatMiddlewareConfig,
  type ChatMiddlewareContext,
  type ModelMessage,
  type ServerTool,
  type SummarizeAdapter
} from "@tanstack/ai"
import type { ManagedRuntime } from "effect"

import type {
  CompressionProvider,
  ExtractionProvider,
  FilterExpr,
  FractionClient,
  JsonRecord,
  RecallOptions,
  Scope
} from "fraction"
import {
  AdapterBridgeService,
  CompressionResultSchema,
  ExtractionResultSchema,
  type SearchResult,
  MemoryService
} from "fraction/effect"

type RuntimeLike =
  | FractionClient<any>
  | ManagedRuntime.ManagedRuntime<any, any>
  | { readonly runtime: ManagedRuntime.ManagedRuntime<any, any> }

type MaybePromise<T> = T | Promise<T>
type RememberMode = "never" | "manual" | "always"
type RememberSource = "user" | "assistant" | "both"

export interface TanStackRememberSettings {
  readonly mode?: RememberMode
  readonly source?: RememberSource
  readonly awaitWrite?: boolean
  readonly metadata?: JsonRecord
}

export interface TanStackFractionMemoryOptions {
  readonly memory: RuntimeLike
  readonly scope?:
    | Scope
    | ((
        ctx: ChatMiddlewareContext,
        config: ChatMiddlewareConfig
      ) => MaybePromise<Scope | undefined>)
  readonly query?: (
    ctx: ChatMiddlewareContext,
    config: ChatMiddlewareConfig
  ) => MaybePromise<string>
  readonly filter?: FilterExpr
  readonly recall?: RecallOptions
  readonly remember?: TanStackRememberSettings
}

export interface TanStackExtractionProviderOptions<
  TAdapter extends SummarizeAdapter<string, object> = SummarizeAdapter<string, object>
> {
  readonly adapter: TAdapter
  readonly modelOptions?: object
  readonly prompt?: string | ((input: { readonly text: string }) => string)
  readonly maxLength?: number
}

export interface TanStackCompressionProviderOptions<
  TAdapter extends SummarizeAdapter<string, object> = SummarizeAdapter<string, object>
> {
  readonly adapter: TAdapter
  readonly modelOptions?: object
  readonly prompt?: string | ((input: { readonly text: string }) => string)
  readonly maxLength?: number
}

const SearchMemoriesInputSchema = Schema.Struct({
  query: Schema.String,
  limit: Schema.optional(Schema.Number)
})

const RememberMemoryInputSchema = Schema.Struct({
  content: Schema.String,
  metadata: Schema.optional(Schema.Record(Schema.String, Schema.Json))
})

const ForgetMemoryInputSchema = Schema.Struct({
  id: Schema.String
})

const GetMemoryInputSchema = Schema.Struct({
  id: Schema.String
})

const getRuntime = (memory: RuntimeLike): ManagedRuntime.ManagedRuntime<any, any> => {
  if ("runtime" in memory) {
    return memory.runtime
  }
  return memory
}

const staticScope = (scope: TanStackFractionMemoryOptions["scope"]): Scope | undefined =>
  typeof scope === "function" ? undefined : scope

const toTanStackSchema = <TSchema extends Schema.Top>(schema: TSchema) =>
  Schema.toStandardJSONSchemaV1(schema)

const decodeExtractionResult = Schema.decodeUnknownSync(
  ExtractionResultSchema as typeof ExtractionResultSchema & { readonly DecodingServices: never }
)

const defaultExtractionPrompt = (text: string) =>
  [
    "Return JSON only.",
    "Schema:",
    '{"content":"string","entities":["string"],"eventAt":"optional ISO timestamp"}',
    "Extract a concise durable memory from the source text and list important named entities.",
    "Only include eventAt when the source clearly references a specific time.",
    "",
    text
  ].join("\n")

const CompressionOutputSchema = Schema.Struct({
  content: Schema.String,
  retainedRatio: Schema.optional(Schema.Number),
  tokenCountBefore: Schema.optional(Schema.Number),
  tokenCountAfter: Schema.optional(Schema.Number)
})

const decodeCompressionResult = Schema.decodeUnknownSync(
  CompressionOutputSchema as typeof CompressionOutputSchema & { readonly DecodingServices: never }
)

const defaultCompressionPrompt = (text: string) =>
  [
    "Return JSON only.",
    "Schema:",
    '{"content":"string","retainedRatio":"optional number","tokenCountBefore":"optional number","tokenCountAfter":"optional number"}',
    "Compress the source text into a shorter durable memory string while preserving key facts.",
    "",
    text
  ].join("\n")

const stripFence = (value: string) =>
  value
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/, "")
    .trim()

const modelMessageText = (message: Pick<ModelMessage, "content">): string => {
  if (typeof message.content === "string") {
    return message.content
  }
  if (!Array.isArray(message.content)) {
    return ""
  }
  return message.content
    .flatMap((part) => {
      if (part.type === "text") {
        return [part.content]
      }
      return []
    })
    .join("\n")
}

const queryFromMessages = (messages: ReadonlyArray<Pick<ModelMessage, "role" | "content">>) => {
  const lastUser = [...messages].reverse().find((message) => message.role === "user")
  if (lastUser) {
    const text = modelMessageText(lastUser).trim()
    if (text.length > 0) {
      return text
    }
  }
  return messages
    .map((message) => `${message.role}: ${modelMessageText(message)}`)
    .filter((value) => value.trim().length > 0)
    .join("\n")
}

const resolveScope = (
  options: TanStackFractionMemoryOptions,
  ctx: ChatMiddlewareContext,
  config: ChatMiddlewareConfig
) => (typeof options.scope === "function" ? options.scope(ctx, config) : options.scope)

const filteredResults = (results: ReadonlyArray<SearchResult>, minScore: number | undefined) =>
  minScore === undefined ? results : results.filter((result) => result.score >= minScore)

const formatResults = async (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  results: ReadonlyArray<SearchResult>,
  recall: RecallOptions | undefined
) =>
  runtime.runPromise(
    AdapterBridgeService.use((service) => Effect.succeed(service.formatContext(results, recall)))
  )

const recallWithResolvedScope = async (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  query: string,
  scope: Scope | undefined,
  options: Pick<TanStackFractionMemoryOptions, "recall" | "filter">
) => {
  const results = await runtime.runPromise(
    AdapterBridgeService.use((service) =>
      service.recall(query, scope, options.recall, options.filter)
    )
  )
  return filteredResults(results, options.recall?.minScore)
}

const buildConfigFromContext = (ctx: ChatMiddlewareContext): ChatMiddlewareConfig => ({
  messages: [...ctx.messages],
  systemPrompts: [...ctx.systemPrompts],
  tools: []
})

const rememberContent = async (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  options: TanStackFractionMemoryOptions,
  content: string,
  scope: Scope | undefined
) => {
  if (content.trim().length === 0) {
    return
  }
  await runtime.runPromise(
    AdapterBridgeService.use((service) =>
      service.rememberText(content, scope, options.remember?.metadata)
    )
  )
}

export const recall = async (
  options: TanStackFractionMemoryOptions & {
    readonly query: string
  }
) => {
  const runtime = getRuntime(options.memory)
  return recallWithResolvedScope(runtime, options.query, staticScope(options.scope), options)
}

export const remember = async (
  options: TanStackFractionMemoryOptions & {
    readonly content: string
    readonly metadata?: JsonRecord
  }
) => {
  const runtime = getRuntime(options.memory)
  return runtime.runPromise(
    AdapterBridgeService.use((service) =>
      service.rememberText(options.content, staticScope(options.scope), options.metadata)
    )
  )
}

export const formatFractionContext = async (
  options: TanStackFractionMemoryOptions & {
    readonly query: string
  }
) => {
  const runtime = getRuntime(options.memory)
  const results = await recall(options)
  return formatResults(runtime, results, options.recall)
}

export const createTanStackExtractionProvider = <TAdapter extends SummarizeAdapter<string, object>>(
  options: TanStackExtractionProviderOptions<TAdapter>
): ExtractionProvider => ({
  extract: async (text) => {
    const result = await summarize({
      adapter: options.adapter,
      text:
        typeof options.prompt === "function"
          ? options.prompt({ text })
          : (options.prompt ?? defaultExtractionPrompt(text)),
      style: "concise",
      ...(options.maxLength !== undefined ? { maxLength: options.maxLength } : {}),
      ...(options.modelOptions ? { modelOptions: options.modelOptions as never } : {})
    })
    return decodeExtractionResult(JSON.parse(stripFence(result.summary)))
  }
})

export const createTanStackCompressionProvider = <TAdapter extends SummarizeAdapter<string, object>>(
  options: TanStackCompressionProviderOptions<TAdapter>
): CompressionProvider => {
  const compress = async (text: string) => {
    const result = await summarize({
      adapter: options.adapter,
      text:
        typeof options.prompt === "function"
          ? options.prompt({ text })
          : (options.prompt ?? defaultCompressionPrompt(text)),
      style: "concise",
      ...(options.maxLength !== undefined ? { maxLength: options.maxLength } : {}),
      ...(options.modelOptions ? { modelOptions: options.modelOptions as never } : {})
    })
    const parsed = decodeCompressionResult(JSON.parse(stripFence(result.summary)))
    return {
      content: parsed.content,
      mode: "provider",
      source: "remote",
      ...(parsed.retainedRatio !== undefined ? { retainedRatio: parsed.retainedRatio } : {}),
      ...(parsed.tokenCountBefore !== undefined
        ? { tokenCountBefore: parsed.tokenCountBefore }
        : {}),
      ...(parsed.tokenCountAfter !== undefined ? { tokenCountAfter: parsed.tokenCountAfter } : {})
    } satisfies Schema.Schema.Type<typeof CompressionResultSchema>
  }

  return {
    compress,
    compressMany: async (texts: ReadonlyArray<string>) =>
      Promise.all(texts.map((text: string) => compress(text)))
  }
}

export const fractionTools = (
  options: TanStackFractionMemoryOptions
): ReadonlyArray<ServerTool<any, any, string>> => {
  const runtime = getRuntime(options.memory)
  const scope = staticScope(options.scope)

  return [
    toolDefinition({
      name: "searchMemories",
      description: "Search previously stored Fraction memories.",
      inputSchema: toTanStackSchema(SearchMemoriesInputSchema)
    }).server(async (input) => {
      const results = await recallWithResolvedScope(
        runtime,
        input.query,
        scope,
        options.filter === undefined
          ? { recall: { ...options.recall, limit: input.limit ?? options.recall?.limit } }
          : {
              recall: { ...options.recall, limit: input.limit ?? options.recall?.limit },
              filter: options.filter
            }
      )
      return {
        results: results.map((result) => ({
          id: result.memory.id,
          content: result.memory.content,
          score: result.score,
          metadata: result.memory.metadata
        }))
      }
    }),
    toolDefinition({
      name: "rememberMemory",
      description: "Persist a new memory into Fraction.",
      inputSchema: toTanStackSchema(RememberMemoryInputSchema)
    }).server(async (input) =>
      runtime.runPromise(
        AdapterBridgeService.use((service) =>
          service.rememberText(input.content, scope, input.metadata)
        )
      )
    ),
    toolDefinition({
      name: "forgetMemory",
      description: "Soft-delete a memory from Fraction recall results.",
      inputSchema: toTanStackSchema(ForgetMemoryInputSchema)
    }).server(async (input) => {
      await runtime.runPromise(MemoryService.use((service) => service.forget(input.id)))
      return { ok: true }
    }),
    toolDefinition({
      name: "getMemory",
      description: "Fetch a single memory by id from Fraction.",
      inputSchema: toTanStackSchema(GetMemoryInputSchema)
    }).server(async (input) =>
      runtime.runPromise(MemoryService.use((service) => service.get(input.id)))
    )
  ] as const
}

export const fractionMiddleware = (options: TanStackFractionMemoryOptions): ChatMiddleware => ({
  name: "fraction-memory",
  async onConfig(ctx, config) {
    const runtime = getRuntime(options.memory)
    const scope = await resolveScope(options, ctx, config)
    const query = options.query
      ? await options.query(ctx, config)
      : queryFromMessages(config.messages)
    const results = await recallWithResolvedScope(runtime, query, scope, options)
    const contextBlock = await formatResults(runtime, results, options.recall)
    if (contextBlock.trim().length === 0) {
      return
    }
    return {
      systemPrompts: [...config.systemPrompts, contextBlock]
    }
  },
  onFinish(ctx, info) {
    const rememberMode = options.remember?.mode ?? "always"
    if (rememberMode !== "always") {
      return
    }
    const source = options.remember?.source ?? "both"
    const content = [
      source === "user" || source === "both" ? `user: ${queryFromMessages(ctx.messages)}` : "",
      source === "assistant" || source === "both" ? `assistant: ${info.content}` : ""
    ]
      .filter((value) => value.trim().length > 0)
      .join("\n")
      .trim()
    if (content.length === 0) {
      return
    }
    const runtime = getRuntime(options.memory)
    const write = Promise.resolve(resolveScope(options, ctx, buildConfigFromContext(ctx))).then(
      (scope) => rememberContent(runtime, options, content, scope)
    )
    if (options.remember?.awaitWrite) {
      return write.then(() => {})
    }
    ctx.defer(write)
  }
})

export const withFractionMemory = fractionMiddleware
