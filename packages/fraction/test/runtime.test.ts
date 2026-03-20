import { afterEach, describe, expect, mock, test } from "bun:test"
import { rmSync } from "node:fs"
import { join } from "node:path"

const createdDirectories = new Set<string>()

afterEach(() => {
  mock.restore()
  for (const directory of createdDirectories) {
    rmSync(directory, { recursive: true, force: true })
  }
  createdDirectories.clear()
})

describe("Fraction runtime", () => {
  test("closes the internal llmlingua provider on runtime dispose", async () => {
    let closeCalls = 0

    const providersUrl = new URL("../src/internal/providers.ts", import.meta.url)
    const providerFactory = () => ({
      createHeuristicCompressionProvider: (options: { readonly maxFactsPerInput?: number } = {}) => ({
        compress: (text: string) => ({
          content: text.split(/\s+/).slice(0, options.maxFactsPerInput ?? 3).join(" "),
          mode: "heuristic" as const,
          source: "native" as const
        }),
        compressMany: (texts: ReadonlyArray<string>) =>
          texts.map((text) => ({
            content: text.split(/\s+/).slice(0, options.maxFactsPerInput ?? 3).join(" "),
            mode: "heuristic" as const,
            source: "native" as const
          }))
      }),
      createLlmlingua2CompressionProvider: () => ({
        compress: async (text: string) => ({
          content: text.trim(),
          mode: "llmlingua2" as const,
          source: "native" as const
        }),
        compressMany: async (texts: ReadonlyArray<string>) =>
          texts.map((text) => ({
            content: text.trim(),
            mode: "llmlingua2" as const,
            source: "native" as const
          })),
        close: async () => {
          closeCalls += 1
        }
      }),
      createTransformersEmbeddingProvider: () => ({
        embed: async () => Float32Array.from([1, 0, 0, 0]),
        embedMany: async (texts: ReadonlyArray<string>) =>
          texts.map(() => Float32Array.from([1, 0, 0, 0])),
        close: async () => {}
      }),
      prefetchLlmlinguaModel: async () => {}
    })

    mock.module(providersUrl.href, providerFactory)
    mock.module(providersUrl.pathname, providerFactory)

    const effectUrl = new URL("../src/effect.ts", import.meta.url)
    const { MemoryService, createFractionRuntime } = await import(
      `${effectUrl.href}?runtime-close-test=${Date.now()}`
    )

    const root = join(process.cwd(), `.tmp-fraction-runtime-close-${Date.now()}`)
    const filename = join(root, "close-test.sqlite")
    createdDirectories.add(root)

    const runtime = createFractionRuntime({
      filename,
      defaultNamespace: "close-test",
      adaptiveCompression: false,
      compressorType: "llmlingua2"
    })

    await runtime.runPromise(
      MemoryService.use((memory: any) =>
        memory.add("Disposing the runtime should release the llmlingua provider.", {
          namespace: "close-test"
        })
      )
    )

    await runtime.dispose()

    expect(closeCalls).toBe(1)
  })
})
