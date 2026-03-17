import type { EmbeddingModelV3, LanguageModelV3 } from "@ai-sdk/provider"
import { afterEach, describe, expect, test } from "bun:test"
import { rmSync } from "node:fs"
import { join } from "node:path"

import { Fraction } from "fraction"

import {
  createVercelEmbeddingProvider,
  createVercelExtractionProvider,
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

describe("@fraction/vercel-ai", () => {
  test("memory tools can remember and search", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-vercel-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Fraction.open({ filename, defaultNamespace: "test" })
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

  test("creates AI SDK embedding and extraction providers", async () => {
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

    const embeddingProvider = createVercelEmbeddingProvider({ model: embeddingModel })
    const extractionProvider = createVercelExtractionProvider({ model: languageModel })

    const embeddings = await embeddingProvider.embedMany?.(["alpha", "beta"])
    const extracted = await extractionProvider.extract("Sam uses the Vercel provider factory.")

    expect(embeddings?.length).toBe(2)
    expect(embeddings?.[0]?.[0]).toBe(5)
    expect(extracted.content).toBe("Sam uses the Vercel provider factory.")
    expect(extracted.entities.includes("Vercel")).toBeTrue()
  })
})
