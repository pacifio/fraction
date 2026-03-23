import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { afterEach, describe, expect, test } from "bun:test"

import {
  DEFAULT_LLMLINGUA_ARTIFACT_FILES,
  ensureLlmlinguaLocalModelPath
} from "../src/internal/llmlingua/artifacts"
import { CompressionUnavailable } from "../src/internal/errors"

const tempDirs: Array<string> = []

afterEach(async () => {
  await Promise.all(
    tempDirs.splice(0).map((dir) =>
      rm(dir, {
        recursive: true,
        force: true
      })
    )
  )
})

describe("LLMLingua artifacts", () => {
  test("downloads configured artifacts into a local model directory", async () => {
    const root = await mkdtemp(join(tmpdir(), "fraction-llmlingua-artifacts-"))
    const modelDir = join(root, "model")
    tempDirs.push(root)

    const server = Bun.serve({
      port: 0,
      fetch(request) {
        const pathname = new URL(request.url).pathname.replace(/^\//, "")
        return new Response(`stub:${pathname}`)
      }
    })

    try {
      const resolved = await ensureLlmlinguaLocalModelPath({
        model: modelDir,
        downloadModelIfMissing: true,
        artifactBaseUrl: server.url.href
      })

      expect(resolved).toBe(modelDir)
      for (const file of DEFAULT_LLMLINGUA_ARTIFACT_FILES) {
        const target = Bun.file(join(modelDir, file))
        expect(await target.exists()).toBe(true)
        expect(await target.text()).toBe(`stub:${file}`)
      }
    } finally {
      void server.stop(true)
    }
  })

  test("fails when artifacts are missing and no base URL is configured", async () => {
    const root = await mkdtemp(join(tmpdir(), "fraction-llmlingua-artifacts-"))
    const modelDir = join(root, "model")
    tempDirs.push(root)

    let error: unknown

    try {
      await ensureLlmlinguaLocalModelPath({
        model: modelDir,
        downloadModelIfMissing: true
      })
    } catch (caught) {
      error = caught
    }

    expect(error).toBeInstanceOf(CompressionUnavailable)
  })
})
