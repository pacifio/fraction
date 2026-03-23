import { Effect, Schema } from "effect"
import { Output, embed, embedMany, generateText, jsonSchema, tool, wrapLanguageModel } from "ai"
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3Message,
  LanguageModelV3Middleware,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
  SharedV3ProviderOptions
} from "@ai-sdk/provider"
import type { ManagedRuntime } from "effect"
import type { EmbeddingModel, LanguageModel, ModelMessage } from "ai"

import type {
  CompressionProvider,
  EmbeddingProvider,
  ExtractionProvider,
  FilterExpr,
  FractionClient,
  JsonRecord,
  RecallOptions,
  Scope
} from "fraction"
import {
  AdapterError,
  AdapterBridgeService,
  ExtractionResultSchema,
  CompressionResultSchema,
  MemoryNotFound,
  type MemoryRecord,
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

export interface VercelRememberSettings {
  readonly mode?: RememberMode
  readonly source?: RememberSource
  readonly awaitWrite?: boolean
  readonly metadata?: JsonRecord
}

export interface VercelScopeResolverInput {
  readonly prompt: LanguageModelV3Prompt
  readonly params: LanguageModelV3CallOptions
}

export interface VercelFractionMemoryOptions {
  readonly memory: RuntimeLike
  readonly scope?: Scope | ((input: VercelScopeResolverInput) => MaybePromise<Scope | undefined>)
  readonly query?: (input: VercelScopeResolverInput) => MaybePromise<string>
  readonly filter?: FilterExpr
  readonly recall?: RecallOptions
  readonly remember?: VercelRememberSettings
}

export interface VercelHelperScopeOptions {
  readonly scopeContext?: VercelScopeResolverInput
}

export interface VercelEmbeddingProviderOptions {
  readonly model: EmbeddingModel
  readonly maxRetries?: number
  readonly providerOptions?: SharedV3ProviderOptions
  readonly headers?: Record<string, string>
  readonly maxParallelCalls?: number
}

export interface VercelExtractionProviderOptions {
  readonly model: LanguageModel
  readonly maxRetries?: number
  readonly providerOptions?: SharedV3ProviderOptions
  readonly headers?: Record<string, string>
  readonly prompt?: string | ((input: { readonly text: string }) => string)
}

export interface VercelCompressionProviderOptions {
  readonly model: LanguageModel
  readonly maxRetries?: number
  readonly providerOptions?: SharedV3ProviderOptions
  readonly headers?: Record<string, string>
  readonly prompt?: string | ((input: { readonly text: string }) => string)
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

type ScopeResolver = Exclude<VercelFractionMemoryOptions["scope"], Scope | undefined>
type VercelToolExecutionContext = {
  readonly messages?: ReadonlyArray<ModelMessage>
}

const hasDynamicScope = (
  scope: VercelFractionMemoryOptions["scope"]
): scope is ScopeResolver => typeof scope === "function"

const adapterError = (message: string) =>
  new AdapterError({
    message
  })

const effectSchemaToAiSchema = <TSchema extends Schema.Top>(schema: TSchema) =>
  jsonSchema<Schema.Schema.Type<TSchema>>(() => Schema.toJsonSchemaDocument(schema).schema, {
    validate: (value) => {
      try {
        const decoded = Schema.decodeUnknownSync(
          schema as TSchema & { readonly DecodingServices: never }
        )(value) as Schema.Schema.Type<TSchema>
        return { success: true as const, value: decoded }
      } catch (error) {
        return {
          success: false as const,
          error: error instanceof Error ? error : new Error(String(error))
        }
      }
    }
  })

const defaultExtractionPrompt = (text: string) =>
  [
    "Extract durable memory facts from the text.",
    "Return an object with:",
    "- content: a concise factual memory string",
    "- entities: a list of named entities, people, places, organizations, emails, or products",
    "- eventAt: optional ISO timestamp only when the text clearly references a specific time",
    "Keep content concise and omit filler.",
    "",
    text
  ].join("\n")

const CompressionOutputSchema = Schema.Struct({
  content: Schema.String,
  retainedRatio: Schema.optional(Schema.Number),
  tokenCountBefore: Schema.optional(Schema.Number),
  tokenCountAfter: Schema.optional(Schema.Number)
})

const defaultCompressionPrompt = (text: string) =>
  [
    "Compress the source text into a shorter durable memory string while preserving key facts.",
    "Return an object with:",
    "- content: compressed text",
    "- retainedRatio: optional decimal fraction of kept tokens",
    "- tokenCountBefore: optional token count before compression",
    "- tokenCountAfter: optional token count after compression",
    "Do not add commentary.",
    "",
    text
  ].join("\n")

const messageText = (message: LanguageModelV3Message): string => {
  if (message.role === "system") {
    return message.content
  }
  return message.content
    .flatMap((part) => {
      switch (part.type) {
        case "text":
          return [part.text]
        case "reasoning":
          return [part.text]
        default:
          return []
      }
    })
    .join("\n")
}

const promptToText = (prompt: LanguageModelV3Prompt) =>
  prompt
    .map((message) => `${message.role}: ${messageText(message)}`)
    .filter((value) => value.trim().length > 0)
    .join("\n")

const modelMessageText = (message: ModelMessage): string => {
  if (typeof message.content === "string") {
    return message.content
  }
  if (!Array.isArray(message.content)) {
    return ""
  }
  return message.content
    .flatMap((part) => {
      if (part.type === "text") {
        return [part.text]
      }
      if ("text" in part && typeof part.text === "string") {
        return [part.text]
      }
      return []
    })
    .join("\n")
}

const toResolverPrompt = (messages: ReadonlyArray<ModelMessage>): LanguageModelV3Prompt =>
  messages.reduce<LanguageModelV3Prompt>((prompt, message) => {
    const text = modelMessageText(message).trim()
    if (text.length === 0) {
      return prompt
    }
    if (message.role === "system") {
      prompt.push({ role: "system", content: text })
      return prompt
    }
    prompt.push({
      role: message.role === "assistant" || message.role === "tool" ? "assistant" : "user",
      content: [{ type: "text", text }]
    })
    return prompt
  }, [])

const queryFromPrompt = (prompt: LanguageModelV3Prompt) => {
  const lastUser = [...prompt].reverse().find((message) => message.role === "user")
  if (lastUser) {
    const text = messageText(lastUser).trim()
    if (text.length > 0) {
      return text
    }
  }
  return promptToText(prompt)
}

const injectMemoryContext = (
  prompt: LanguageModelV3Prompt,
  contextBlock: string
): LanguageModelV3Prompt => {
  if (contextBlock.trim().length === 0) {
    return prompt
  }
  const next = [...prompt]
  const systemIndex = next.findIndex((message) => message.role === "system")
  if (systemIndex >= 0) {
    const existing = next[systemIndex] as Extract<LanguageModelV3Message, { role: "system" }>
    next[systemIndex] = {
      ...existing,
      content: `${existing.content}\n\n${contextBlock}`
    }
    return next
  }
  next.unshift({ role: "system", content: contextBlock })
  return next
}

const resolveScope = async (
  options: VercelFractionMemoryOptions,
  input: VercelScopeResolverInput
) => (typeof options.scope === "function" ? options.scope(input) : options.scope)

const resolveNormalizedScope = (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  scope: Scope | undefined
) => runtime.runPromise(AdapterBridgeService.use((service) => service.resolveScope(scope)))

const resolveHelperScope = async (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  options: Pick<VercelFractionMemoryOptions, "scope">,
  scopeContext: VercelScopeResolverInput | undefined
) => {
  if (hasDynamicScope(options.scope)) {
    if (scopeContext === undefined) {
      throw adapterError(
        "Resolver-based scope requires scopeContext for direct Vercel helper calls."
      )
    }
    return resolveNormalizedScope(runtime, await options.scope(scopeContext))
  }
  return resolveNormalizedScope(runtime, options.scope)
}

const resolveToolScope = async (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  options: VercelFractionMemoryOptions,
  context: VercelToolExecutionContext | undefined
) => {
  if (!hasDynamicScope(options.scope)) {
    return resolveNormalizedScope(runtime, options.scope)
  }
  const prompt = toResolverPrompt(context?.messages ?? [])
  return resolveNormalizedScope(
    runtime,
    await options.scope({
      prompt,
      params: { prompt } as LanguageModelV3CallOptions
    })
  )
}

const sameScopeValue = (left: string | undefined, right: string | undefined) =>
  (left ?? undefined) === (right ?? undefined)

const memoryMatchesScope = (record: MemoryRecord, scope: Scope) =>
  sameScopeValue(record.scope.namespace, scope.namespace) &&
  sameScopeValue(record.scope.userId, scope.userId) &&
  sameScopeValue(record.scope.agentId, scope.agentId) &&
  sameScopeValue(record.scope.runId, scope.runId) &&
  sameScopeValue(record.scope.appId, scope.appId)

const getMemoryRecord = async (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  id: string
) =>
  runtime.runPromise(
    MemoryService.use((service) => service.get(id)).pipe(
      Effect.catchTag("MemoryNotFound", (_error: MemoryNotFound) => Effect.succeed(undefined))
    )
  )

const getScopedMemory = async (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  id: string,
  scope: Scope
) => {
  const record = await getMemoryRecord(runtime, id)
  if (record === undefined || !memoryMatchesScope(record, scope)) {
    throw new MemoryNotFound({ memoryId: id })
  }
  return record
}

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
  options: Pick<VercelFractionMemoryOptions, "recall" | "filter">
) => {
  const results = await runtime.runPromise(
    AdapterBridgeService.use((service) =>
      service.recall(query, scope, options.recall, options.filter)
    )
  )
  return filteredResults(results, options.recall?.minScore)
}

const generatedText = (content: ReadonlyArray<LanguageModelV3Content>) =>
  content
    .filter(
      (part): part is Extract<LanguageModelV3Content, { readonly type: "text" }> =>
        part.type === "text"
    )
    .map((part) => part.text)
    .join("")

const rememberConversation = async (
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  options: VercelFractionMemoryOptions,
  input: VercelScopeResolverInput,
  assistantText: string
) => {
  const rememberMode = options.remember?.mode ?? "always"
  if (rememberMode !== "always") {
    return
  }
  const source = options.remember?.source ?? "both"
  const userText = queryFromPrompt(input.prompt)
  const content = [
    source === "user" || source === "both" ? `user: ${userText}` : "",
    source === "assistant" || source === "both" ? `assistant: ${assistantText}` : ""
  ]
    .filter((value) => value.trim().length > 0)
    .join("\n")
    .trim()
  if (content.length === 0) {
    return
  }
  const scope = await resolveScope(options, input)
  const write = runtime.runPromise(
    AdapterBridgeService.use((service) =>
      service.rememberText(content, scope, options.remember?.metadata)
    )
  )
  if (options.remember?.awaitWrite) {
    await write
  } else {
    void write
  }
}

const consumeStreamForMemory = async (
  stream: ReadableStream<LanguageModelV3StreamPart>,
  runtime: ManagedRuntime.ManagedRuntime<any, any>,
  options: VercelFractionMemoryOptions,
  input: VercelScopeResolverInput
) => {
  const reader = stream.getReader()
  let text = ""
  let shouldRemember = false
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      if (value.type === "text-delta") {
        text += value.delta
      }
      if (value.type === "finish") {
        shouldRemember = value.finishReason.unified !== "error"
      }
      if (value.type === "error") {
        shouldRemember = false
      }
    }
  } finally {
    reader.releaseLock()
  }
  if (shouldRemember && text.trim().length > 0) {
    await rememberConversation(runtime, options, input, text)
  }
}

export const recall = async (
  options: Omit<VercelFractionMemoryOptions, "query"> & {
    readonly query: string
  } & VercelHelperScopeOptions
) => {
  const runtime = getRuntime(options.memory)
  const scope = await resolveHelperScope(runtime, options, options.scopeContext)
  return recallWithResolvedScope(runtime, options.query, scope, options)
}

export const remember = async (
  options: VercelFractionMemoryOptions & {
    readonly content: string
    readonly metadata?: JsonRecord
  } & VercelHelperScopeOptions
) => {
  const runtime = getRuntime(options.memory)
  const scope = await resolveHelperScope(runtime, options, options.scopeContext)
  return runtime.runPromise(
    AdapterBridgeService.use((service) =>
      service.rememberText(options.content, scope, options.metadata)
    )
  )
}

export const formatFractionContext = async (
  options: Omit<VercelFractionMemoryOptions, "query"> & {
    readonly query: string
  } & VercelHelperScopeOptions
) => {
  const runtime = getRuntime(options.memory)
  const scope = await resolveHelperScope(runtime, options, options.scopeContext)
  const results = await recallWithResolvedScope(runtime, options.query, scope, options)
  return formatResults(runtime, results, options.recall)
}

export const createVercelEmbeddingProvider = (
  options: VercelEmbeddingProviderOptions
): EmbeddingProvider => ({
  embed: async (text) => {
    const result = await embed({
      model: options.model,
      value: text,
      ...(options.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
      ...(options.providerOptions ? { providerOptions: options.providerOptions } : {}),
      ...(options.headers ? { headers: options.headers } : {})
    })
    return Float32Array.from(result.embedding)
  },
  embedMany: async (texts) => {
    if (texts.length === 0) {
      return []
    }
    const result = await embedMany({
      model: options.model,
      values: [...texts],
      ...(options.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
      ...(options.providerOptions ? { providerOptions: options.providerOptions } : {}),
      ...(options.headers ? { headers: options.headers } : {}),
      ...(options.maxParallelCalls !== undefined
        ? { maxParallelCalls: options.maxParallelCalls }
        : {})
    })
    return result.embeddings.map((embedding) => Float32Array.from(embedding))
  }
})

export const createVercelExtractionProvider = (
  options: VercelExtractionProviderOptions
): ExtractionProvider => ({
  extract: async (text) => {
    const result = await generateText({
      model: options.model,
      prompt:
        typeof options.prompt === "function"
          ? options.prompt({ text })
          : (options.prompt ?? defaultExtractionPrompt(text)),
      ...(options.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
      ...(options.providerOptions ? { providerOptions: options.providerOptions } : {}),
      ...(options.headers ? { headers: options.headers } : {}),
      output: Output.object({
        schema: effectSchemaToAiSchema(ExtractionResultSchema),
        name: "fraction_memory_extraction",
        description: "Structured durable memory extracted from raw conversation text"
      })
    })
    return result.output
  }
})

export const createVercelCompressionProvider = (
  options: VercelCompressionProviderOptions
): CompressionProvider => {
  const compress = async (text: string) => {
    const result = await generateText({
      model: options.model,
      prompt:
        typeof options.prompt === "function"
          ? options.prompt({ text })
          : (options.prompt ?? defaultCompressionPrompt(text)),
      ...(options.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
      ...(options.providerOptions ? { providerOptions: options.providerOptions } : {}),
      ...(options.headers ? { headers: options.headers } : {}),
      output: Output.object({
        schema: effectSchemaToAiSchema(CompressionOutputSchema),
        name: "fraction_memory_compression",
        description: "Compressed durable memory text"
      })
    })

    return {
      content: result.output.content,
      mode: "provider",
      source: "remote",
      ...(result.output.retainedRatio !== undefined
        ? { retainedRatio: result.output.retainedRatio }
        : {}),
      ...(result.output.tokenCountBefore !== undefined
        ? { tokenCountBefore: result.output.tokenCountBefore }
        : {}),
      ...(result.output.tokenCountAfter !== undefined
        ? { tokenCountAfter: result.output.tokenCountAfter }
        : {})
    } satisfies Schema.Schema.Type<typeof CompressionResultSchema>
  }

  return {
    compress,
    compressMany: async (texts: ReadonlyArray<string>) =>
      Promise.all(texts.map((text: string) => compress(text)))
  }
}

export const fractionTools = (options: VercelFractionMemoryOptions) => {
  const runtime = getRuntime(options.memory)
  return {
    searchMemories: tool({
      description: "Search previously stored Fraction memories.",
      inputSchema: effectSchemaToAiSchema(SearchMemoriesInputSchema),
      execute: async (input, context) => {
        const scope = await resolveToolScope(runtime, options, context)
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
      }
    }),
    rememberMemory: tool({
      description: "Persist a new memory into Fraction.",
      inputSchema: effectSchemaToAiSchema(RememberMemoryInputSchema),
      execute: async (input, context) => {
        const scope = await resolveToolScope(runtime, options, context)
        return runtime.runPromise(
          AdapterBridgeService.use((service) =>
            service.rememberText(input.content, scope, input.metadata)
          )
        )
      }
    }),
    forgetMemory: tool({
      description: "Soft-delete a memory from Fraction recall results.",
      inputSchema: effectSchemaToAiSchema(ForgetMemoryInputSchema),
      execute: async (input, context) => {
        const scope = await resolveToolScope(runtime, options, context)
        const record = await getScopedMemory(runtime, input.id, scope)
        await runtime.runPromise(MemoryService.use((service) => service.forget(record.id)))
        return { ok: true }
      }
    }),
    getMemory: tool({
      description: "Fetch a single memory by id from Fraction.",
      inputSchema: effectSchemaToAiSchema(GetMemoryInputSchema),
      execute: async (input, context) => {
        const scope = await resolveToolScope(runtime, options, context)
        return getScopedMemory(runtime, input.id, scope)
      }
    })
  }
}

export const withFractionMemory = (
  model: LanguageModelV3,
  options: VercelFractionMemoryOptions
): LanguageModelV3 => {
  const middleware: LanguageModelV3Middleware = {
    specificationVersion: "v3",
    transformParams: async ({ params }) => {
      const input = { params, prompt: params.prompt }
      const runtime = getRuntime(options.memory)
      const scope = await resolveScope(options, input)
      const query = options.query ? await options.query(input) : queryFromPrompt(params.prompt)
      const results = await recallWithResolvedScope(runtime, query, scope, options)
      const contextBlock = await formatResults(runtime, results, options.recall)
      return {
        ...params,
        prompt: injectMemoryContext(params.prompt, contextBlock)
      }
    },
    wrapGenerate: async ({ doGenerate, params }) => {
      const runtime = getRuntime(options.memory)
      const input = { params, prompt: params.prompt }
      const result = await doGenerate()
      const text = generatedText(result.content)
      if (result.finishReason.unified !== "error" && text.trim().length > 0) {
        await rememberConversation(runtime, options, input, text)
      }
      return result
    },
    wrapStream: async ({ doStream, params }) => {
      const runtime = getRuntime(options.memory)
      const input = { params, prompt: params.prompt }
      const result = await doStream()
      const [memoryStream, passthrough] = result.stream.tee()
      void consumeStreamForMemory(memoryStream, runtime, options, input)
      return {
        ...result,
        stream: passthrough
      }
    }
  }

  return wrapLanguageModel({
    model,
    middleware
  })
}
