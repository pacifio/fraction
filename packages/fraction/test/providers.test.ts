import { afterEach, describe, expect, test } from "bun:test"
import { rmSync } from "node:fs"
import { join } from "node:path"

import { createLlmlingua2Compressor } from "../src/internal/llmlingua/compressor"
import { loadLlmlinguaSession } from "../src/internal/llmlingua/loader"
import { CompressionUnavailable } from "../src/internal/errors"
import { createLlmlingua2CompressionProvider } from "../src/internal/providers"

const createdDirectories = new Set<string>()

const missingModelPath = (suffix: string) => {
  const root = join(process.cwd(), `.tmp-fraction-llmlingua-${suffix}-${Date.now()}`)
  createdDirectories.add(root)
  return join(root, "missing-model")
}

afterEach(() => {
  for (const directory of createdDirectories) {
    rmSync(directory, { recursive: true, force: true })
  }
  createdDirectories.clear()
})

describe("LLMLingua provider and loader", () => {
  test("maps loader failures to CompressionUnavailable", async () => {
    const model = missingModelPath("loader")

    let error: unknown

    try {
      await loadLlmlinguaSession({
        model,
        modelFamily: "bert",
        downloadModelIfMissing: false
      })
    } catch (caught) {
      error = caught
    }

    expect(error).toBeInstanceOf(CompressionUnavailable)
  })

  test("wraps single and batch compression failures as CompressionUnavailable", async () => {
    const model = missingModelPath("provider")
    const provider = createLlmlingua2CompressionProvider({
      model,
      modelFamily: "bert",
      downloadModelIfMissing: false
    })

    let singleError: unknown
    let batchError: unknown

    try {
      await provider.compress("LLMLingua provider failures should stay typed.")
    } catch (caught) {
      singleError = caught
    }

    try {
      await provider.compressMany?.([
        "Batch one should report CompressionUnavailable.",
        "Batch two should report CompressionUnavailable."
      ])
    } catch (caught) {
      batchError = caught
    }

    expect(singleError).toBeInstanceOf(CompressionUnavailable)
    expect(batchError).toBeInstanceOf(CompressionUnavailable)
  })

  test("provider close is idempotent and ignores failed initialization", async () => {
    const model = missingModelPath("provider-close")
    const provider = createLlmlingua2CompressionProvider({
      model,
      modelFamily: "bert",
      downloadModelIfMissing: false
    })

    try {
      await provider.compress("Force the provider to initialize and fail.")
    } catch {}

    await provider.close?.()
    await provider.close?.()
  })

  test("compressor close is idempotent and ignores failed initialization", async () => {
    const model = missingModelPath("compressor-close")
    const compressor = createLlmlingua2Compressor({
      model,
      modelFamily: "bert",
      downloadModelIfMissing: false
    })

    try {
      await compressor.compress("Force the compressor session to initialize and fail.")
    } catch {}

    await compressor.close()
    await compressor.close()
  })
})
