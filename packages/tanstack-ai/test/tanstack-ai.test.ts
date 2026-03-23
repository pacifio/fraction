import type { SummarizeAdapter } from "@tanstack/ai"
import { afterEach, describe, expect, test } from "bun:test"
import { rmSync } from "node:fs"
import { join } from "node:path"

import { Fraction } from "fraction"

import {
  createTanStackCompressionProvider,
  createTanStackExtractionProvider,
  formatFractionContext,
  fractionMiddleware,
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

describe("@fraction/tanstack-ai", () => {
  const toolContextForUser = (userId: string) =>
    ({
      toolCallId: `tool:${userId}`,
      emitCustomEvent: async () => {}
    }) as const

  const middlewareContextForUser = (userId: string) => ({
    requestId: `req-${userId}`,
    streamId: `stream-${userId}`,
    phase: "init" as const,
    iteration: 0,
    chunkIndex: 0,
    abort: () => {},
    context: undefined,
    defer: () => {},
    provider: "test",
    model: "test-model",
    source: "server" as const,
    streaming: false,
    systemPrompts: [],
    toolNames: [],
    messageCount: 1,
    hasTools: false,
    currentMessageId: null,
    accumulatedContent: "",
    messages: [{ role: "user" as const, content: `${userId}: what do I know?` }],
    createId: (prefix: string) => `${prefix}-${userId}`
  })

  const configForUser = (userId: string) => ({
    messages: [{ role: "user" as const, content: `${userId}: what do I know?` }],
    systemPrompts: [],
    tools: []
  })

  const userIdFromConfig = (
    config: Readonly<{ messages: ReadonlyArray<{ content: unknown }> }>
  ) => {
    const content = config.messages[0]?.content
    return typeof content === "string" ? content.split(":")[0] ?? "" : ""
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

  test("tools can remember and search", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-tanstack-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const tools = fractionTools({ memory, scope: { namespace: "test" } })

    const rememberTool = tools.find((tool) => tool.name === "rememberMemory")
    const searchTool = tools.find((tool) => tool.name === "searchMemories")

    await rememberTool?.execute?.({ content: "Fraction uses Bun and Effect." })
    const search = await searchTool?.execute?.({ query: "What uses Effect?" })

    expect(search?.results.length).toBeGreaterThan(0)
    expect(search?.results[0]?.content.includes("Effect")).toBeTrue()
  })

  test("tools honor dynamic toolScope per execution", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-tanstack-toolscope-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const tools = fractionTools({
      memory,
      scope: { namespace: "test" },
      toolScope: (context) => ({
        namespace: "test",
        userId: (context.toolCallId ?? "").split(":")[1]
      })
    })

    const rememberTool = tools.find((tool) => tool.name === "rememberMemory")
    const searchTool = tools.find((tool) => tool.name === "searchMemories")

    await rememberTool?.execute?.({ content: "Alice uses TanStack tools." }, toolContextForUser("alice"))
    await rememberTool?.execute?.({ content: "Bob uses a different tenant." }, toolContextForUser("bob"))

    const aliceSearch = await searchTool?.execute?.(
      { query: "Who uses TanStack tools?" },
      toolContextForUser("alice")
    )
    const bobSearch = await searchTool?.execute?.(
      { query: "Who uses a different tenant?" },
      toolContextForUser("bob")
    )

    expect(
      aliceSearch?.results.some((result: { content: string }) =>
        result.content.includes("Alice uses TanStack tools")
      )
    ).toBeTrue()
    expect(
      aliceSearch?.results.some((result: { content: string }) =>
        result.content.includes("Bob uses a different tenant")
      )
    ).toBeFalse()
    expect(
      bobSearch?.results.some((result: { content: string }) =>
        result.content.includes("Bob uses a different tenant")
      )
    ).toBeTrue()
  })

  test("helpers require scopeContext for resolver-based scopes and honor it when provided", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-tanstack-helpers-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const ctx = middlewareContextForUser("alice")
    const config = configForUser("alice")

    await remember({
      memory,
      scope: (_scopeCtx, scopeConfig) => ({
        namespace: "test",
        userId: userIdFromConfig(scopeConfig)
      }),
      content: "Alice helper-scoped TanStack memory.",
      scopeContext: { ctx, config }
    })

    const results = await recall({
      memory,
      scope: (_scopeCtx, scopeConfig) => ({
        namespace: "test",
        userId: userIdFromConfig(scopeConfig)
      }),
      query: "Who has helper-scoped TanStack memory?",
      scopeContext: { ctx, config }
    })

    const formatted = await formatFractionContext({
      memory,
      scope: (_scopeCtx, scopeConfig) => ({
        namespace: "test",
        userId: userIdFromConfig(scopeConfig)
      }),
      query: "Who has helper-scoped TanStack memory?",
      scopeContext: { ctx, config }
    })

    expect(results.some((result) => result.memory.content.includes("Alice helper-scoped TanStack memory"))).toBeTrue()
    expect(formatted.includes("Alice helper-scoped TanStack memory")).toBeTrue()

    await expectAdapterError(
      recall({
        memory,
        scope: () => ({ namespace: "test", userId: "alice" }),
        query: "missing context"
      })
    )
  })

  test("tools fail clearly without toolScope when scope is middleware-only dynamic", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-tanstack-missing-toolscope-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const tools = fractionTools({
      memory,
      scope: () => ({ namespace: "test", userId: "alice" })
    })

    const searchTool = tools.find((tool) => tool.name === "searchMemories")

    await expectAdapterError(searchTool!.execute!({ query: "test" }, toolContextForUser("alice")))
  })

  test("by-id tools enforce tool scope and return not found on cross-tenant access", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-tanstack-by-id-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const alice = await memory.add("Alice tenant memory.", { namespace: "test", userId: "alice" })
    const bob = await memory.add("Bob tenant memory.", { namespace: "test", userId: "bob" })

    const tools = fractionTools({
      memory,
      scope: { namespace: "test" },
      toolScope: (context) => ({
        namespace: "test",
        userId: (context.toolCallId ?? "").split(":")[1]
      })
    })

    const getTool = tools.find((tool) => tool.name === "getMemory")
    const forgetTool = tools.find((tool) => tool.name === "forgetMemory")

    const ownRecord = await getTool?.execute?.({ id: alice.id }, toolContextForUser("alice"))
    expect(ownRecord?.content).toContain("Alice tenant memory")

    await expectMemoryNotFound(getTool!.execute!({ id: bob.id }, toolContextForUser("alice")), bob.id)

    await expectMemoryNotFound(
      forgetTool!.execute!({ id: bob.id }, toolContextForUser("alice")),
      bob.id
    )

    const bobStillExists = await memory.get(bob.id)
    expect(bobStillExists.content).toContain("Bob tenant memory")
  })

  test("middleware injects memory context into system prompts", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-tanstack-mw-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({
      filename,
      defaultNamespace: "test",
      compressorType: "heuristic"
    })
    openClients.push(memory)
    await memory.add("Rina prefers concise technical answers.", { namespace: "test" })

    const middleware = fractionMiddleware({ memory, scope: { namespace: "test" } })
    const config = await middleware.onConfig?.(
      {
        requestId: "req-1",
        streamId: "stream-1",
        phase: "init",
        iteration: 0,
        chunkIndex: 0,
        abort: () => {},
        context: undefined,
        defer: () => {},
        provider: "test",
        model: "test-model",
        source: "server",
        streaming: false,
        systemPrompts: [],
        toolNames: [],
        messageCount: 1,
        hasTools: false,
        currentMessageId: null,
        accumulatedContent: "",
        messages: [{ role: "user", content: "What style does Rina prefer?" }],
        createId: (prefix) => `${prefix}-1`
      },
      {
        messages: [{ role: "user", content: "What style does Rina prefer?" }],
        systemPrompts: [],
        tools: []
      }
    )

    expect(config?.systemPrompts?.length ?? 0).toBeGreaterThan(0)
    expect(config?.systemPrompts?.[0]?.includes("Relevant Memory")).toBeTrue()
  })

  test("creates TanStack summarize-backed extraction and compression providers", async () => {
    const adapter: SummarizeAdapter = {
      kind: "summarize",
      name: "test",
      model: "summary-model",
      "~types": { providerOptions: {} },
      summarize: async () => ({
        id: "summary-1",
        model: "summary-model",
        summary: JSON.stringify({
          content: "Rina prefers concise technical answers.",
          entities: ["Rina"],
          eventAt: "2026-03-17T00:00:00.000Z"
        }),
        usage: {
          promptTokens: 1,
          completionTokens: 1,
          totalTokens: 2
        }
      })
    }

    const provider = createTanStackExtractionProvider({ adapter })
    const compressionProvider = createTanStackCompressionProvider({
      adapter: {
        ...adapter,
        summarize: async () => ({
          id: "summary-2",
          model: "summary-model",
          summary: JSON.stringify({
            content: "Rina prefers concise technical answers.",
            retainedRatio: 0.5,
            tokenCountBefore: 12,
            tokenCountAfter: 6
          }),
          usage: {
            promptTokens: 1,
            completionTokens: 1,
            totalTokens: 2
          }
        })
      }
    })
    const extraction = await provider.extract("Rina prefers concise technical answers.")
    const compression = await compressionProvider.compress(
      "Rina prefers concise technical answers in Fraction."
    )

    expect(extraction.content).toBe("Rina prefers concise technical answers.")
    expect(extraction.entities).toEqual(["Rina"])
    expect(compression.content).toBe("Rina prefers concise technical answers.")
    expect(compression.mode).toBe("provider")
  })
})
