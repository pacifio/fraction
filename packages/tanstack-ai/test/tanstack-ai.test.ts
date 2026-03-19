import type { SummarizeAdapter } from "@tanstack/ai"
import { afterEach, describe, expect, test } from "bun:test"
import { rmSync } from "node:fs"
import { join } from "node:path"

import { Fraction } from "fraction"

import {
  createTanStackCompressionProvider,
  createTanStackExtractionProvider,
  fractionMiddleware,
  fractionTools
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
