import type { EmbeddingModelV3, LanguageModelV3 } from "@ai-sdk/provider"
import { afterEach, describe, expect, test } from "bun:test"
import { rmSync } from "node:fs"
import { join } from "node:path"

import { Fraction } from "fraction"

import {
  createVercelCompressionProvider,
  createVercelEmbeddingProvider,
  createVercelExtractionProvider,
  formatFractionContext,
  fractionTools,
  recall,
  remember
} from "../src/index"

const openClients: Array<{ close: () => Promise<void> }> = []
const createdPaths = new Set<string>()

afterEach(async () => {
  while (openClients.length > 0) {
    await openClients.pop()!.close()
  }
  for (const path of createdPaths) {
    rmSync(path, { force: true })
    rmSync(`${path}-shm`, { force: true })
    rmSync(`${path}-wal`, { force: true })
  }
  createdPaths.clear()
})

describe("@fraction/vercel-ai", () => {
  const promptForUser = (userId: string, text: string) =>
    [{
      role: "user" as const,
      content: [{ type: "text" as const, text: `${userId}:${text}` }]
    }]

  const scopeFromPrompt = ({
    prompt
  }: {
    readonly prompt: ReadonlyArray<{ readonly content?: unknown }>
  }) => {
    const first = prompt[0]
    const content = Array.isArray((first as { readonly content?: unknown })?.content)
      ? ((first as { readonly content: Array<{ readonly type?: string; readonly text?: string }> })
          .content
          .find((part) => part.type === "text")
          ?.text ?? "")
      : ""
    const [userId] = content.split(":")
    return { namespace: "test", userId }
  }

  const expectAdapterError = async (operation: Promise<unknown>) => {
    try {
      await operation
      throw new Error("Expected AdapterError")
    } catch (error) {
      expect(error).toMatchObject({ _tag: "AdapterError" })
    }
  }

  const expectMemoryNotFound = async (operation: Promise<unknown>, memoryId: string) => {
    try {
      await operation
      throw new Error("Expected MemoryNotFound")
    } catch (error) {
      expect(error).toMatchObject({
        _tag: "MemoryNotFound",
        memoryId
      })
    }
  }

  test("memory tools can remember and search", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-vercel-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const tools = fractionTools({ memory, scope: { namespace: "test" } })

    await tools.rememberMemory.execute?.(
      { content: "Sam likes Bun and Effect." },
      { toolCallId: "remember-1", messages: [] }
    )

    const search = (await tools.searchMemories.execute?.(
      { query: "Who likes Bun?" },
      { toolCallId: "search-1", messages: [] }
    )) as { results: Array<{ content: string }> } | undefined

    expect(search?.results.length).toBeGreaterThan(0)
    expect(search?.results[0]?.content.includes("Bun")).toBeTrue()
  })

  test("tools honor resolver-based scopes per request", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-vercel-scope-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const tools = fractionTools({
      memory,
      scope: scopeFromPrompt
    })

    await tools.rememberMemory.execute?.(
      { content: "Alice likes Bun." },
      { toolCallId: "remember-alice", messages: promptForUser("alice", "remember this") }
    )
    await tools.rememberMemory.execute?.(
      { content: "Bob prefers Python." },
      { toolCallId: "remember-bob", messages: promptForUser("bob", "remember this") }
    )

    const aliceSearch = (await tools.searchMemories.execute?.(
      { query: "What does Alice like?" },
      { toolCallId: "search-alice", messages: promptForUser("alice", "What do I like?") }
    )) as { results: Array<{ content: string }> } | undefined

    const bobSearch = (await tools.searchMemories.execute?.(
      { query: "What does Bob prefer?" },
      { toolCallId: "search-bob", messages: promptForUser("bob", "What do I prefer?") }
    )) as { results: Array<{ content: string }> } | undefined

    expect(aliceSearch?.results.some((result) => result.content.includes("Alice likes Bun"))).toBeTrue()
    expect(aliceSearch?.results.some((result) => result.content.includes("Bob prefers Python"))).toBeFalse()
    expect(bobSearch?.results.some((result) => result.content.includes("Bob prefers Python"))).toBeTrue()
    expect(bobSearch?.results.some((result) => result.content.includes("Alice likes Bun"))).toBeFalse()
  })

  test("helpers require scopeContext for resolver-based scopes and honor it when provided", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-vercel-helpers-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const scopeContext = {
      prompt: promptForUser("alice", "remember this"),
      params: { prompt: promptForUser("alice", "remember this") } as never
    }

    await remember({
      memory,
      scope: scopeFromPrompt,
      content: "Alice uses scoped helper memory.",
      scopeContext
    })

    const helperResults = await recall({
      memory,
      scope: scopeFromPrompt,
      query: "Who uses scoped helper memory?",
      scopeContext
    })

    const formatted = await formatFractionContext({
      memory,
      scope: scopeFromPrompt,
      query: "Who uses scoped helper memory?",
      scopeContext
    })

    expect(helperResults.some((result) => result.memory.content.includes("Alice uses scoped helper memory"))).toBeTrue()
    expect(formatted.includes("Alice uses scoped helper memory")).toBeTrue()

    await expectAdapterError(
      recall({
        memory,
        scope: scopeFromPrompt,
        query: "missing context"
      })
    )
  })

  test("by-id tools enforce resolved scope and return not found on cross-tenant access", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-vercel-by-id-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const alice = await memory.add("Alice private memory.", { namespace: "test", userId: "alice" })
    const bob = await memory.add("Bob private memory.", { namespace: "test", userId: "bob" })

    const tools = fractionTools({
      memory,
      scope: scopeFromPrompt
    })

    const ownRecord = (await tools.getMemory.execute?.(
      { id: alice.id },
      { toolCallId: "get-alice", messages: promptForUser("alice", "load my memory") }
    )) as { content: string } | undefined

    expect(ownRecord?.content).toContain("Alice private memory")

    await expectMemoryNotFound(
      tools.getMemory.execute!(
        { id: bob.id },
        { toolCallId: "get-bob-from-alice", messages: promptForUser("alice", "load bob memory") }
      ),
      bob.id
    )

    await expectMemoryNotFound(
      tools.forgetMemory.execute!(
        { id: bob.id },
        { toolCallId: "forget-bob-from-alice", messages: promptForUser("alice", "forget bob memory") }
      ),
      bob.id
    )

    const bobStillExists = await memory.get(bob.id)
    expect(bobStillExists.content).toContain("Bob private memory")
  })

  test("creates AI SDK embedding, extraction, and compression providers", async () => {
    const embeddingModel = {
      specificationVersion: "v3",
      provider: "test",
      modelId: "embedding-test",
      maxEmbeddingsPerCall: Infinity,
      supportsParallelCalls: true,
      doEmbed: async ({ values }: { values: Array<string> }) => ({
        embeddings: values.map((value: string) => [value.length, 1, 0]),
        usage: { tokens: values.length },
        response: {},
        warnings: []
      })
    } as unknown as EmbeddingModelV3

    const languageModel = {
      specificationVersion: "v3",
      provider: "test",
      modelId: "lm-test",
      supportedUrls: {},
      doGenerate: async () => ({
        content: [
          {
            type: "text",
            text: JSON.stringify({
              content: "Sam uses the Vercel provider factory.",
              entities: ["Sam", "Vercel"],
              eventAt: "2026-03-17T00:00:00.000Z"
            })
          }
        ],
        finishReason: { unified: "stop", raw: "stop" },
        usage: {
          inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 1, text: 1, reasoning: 0 }
        },
        warnings: [],
        response: { timestamp: new Date(), modelId: "lm-test" }
      }),
      doStream: async () => {
        throw new Error("doStream not implemented in test")
      }
    } as unknown as LanguageModelV3

    const compressionLanguageModel = {
      specificationVersion: "v3",
      provider: "test",
      modelId: "lm-compress-test",
      supportedUrls: {},
      doGenerate: async () => ({
        content: [
          {
            type: "text",
            text: JSON.stringify({
              content: "Sam uses the Vercel provider factory.",
              retainedRatio: 0.5,
              tokenCountBefore: 10,
              tokenCountAfter: 5
            })
          }
        ],
        finishReason: { unified: "stop", raw: "stop" },
        usage: {
          inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 1, text: 1, reasoning: 0 }
        },
        warnings: [],
        response: { timestamp: new Date(), modelId: "lm-compress-test" }
      }),
      doStream: async () => {
        throw new Error("doStream not implemented in test")
      }
    } as unknown as LanguageModelV3

    const embeddingProvider = createVercelEmbeddingProvider({ model: embeddingModel })
    const extractionProvider = createVercelExtractionProvider({ model: languageModel })
    const compressionProvider = createVercelCompressionProvider({ model: compressionLanguageModel })

    const embeddings = await embeddingProvider.embedMany?.(["alpha", "beta"])
    const extracted = await extractionProvider.extract("Sam uses the Vercel provider factory.")
    const compressed = await compressionProvider.compress(
      "Sam uses the Vercel provider factory for memory compression."
    )

    expect(embeddings?.length).toBe(2)
    expect(embeddings?.[0]?.[0]).toBe(5)
    expect(extracted.content).toBe("Sam uses the Vercel provider factory.")
    expect(extracted.entities.includes("Vercel")).toBeTrue()
    expect(compressed.content).toBe("Sam uses the Vercel provider factory.")
    expect(compressed.mode).toBe("provider")
  })
})
