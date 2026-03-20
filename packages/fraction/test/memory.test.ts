import { afterEach, describe, expect, test } from "bun:test"
import { Database } from "bun:sqlite"
import { existsSync, rmSync } from "node:fs"
import { join } from "node:path"
import { Schema } from "effect"

import { MemoryService, createFractionRuntime } from "../src/effect"
import { Memory } from "../src/index"

const baseOptions = (filename: string) =>
  ({
    filename,
    defaultNamespace: "test",
    compressorType: "heuristic" as const
  }) as const

const openClients: Array<{ close: () => Promise<void> }> = []
const openRuntimes: Array<{ dispose: () => Promise<void> }> = []
const openDatabases: Array<Database> = []
const createdPaths = new Set<string>()
const createdDirectories = new Set<string>()

const queryGet = <TRow>(
  db: Database,
  sql: string,
  params: ReadonlyArray<string | number> = []
) => db.query(sql).get(...params) as TRow | null

afterEach(async () => {
  while (openClients.length > 0) {
    await openClients.pop()!.close()
  }
  while (openRuntimes.length > 0) {
    await openRuntimes.pop()!.dispose()
  }
  while (openDatabases.length > 0) {
    openDatabases.pop()!.close()
  }
  for (const path of createdPaths) {
    rmSync(path, { force: true })
    rmSync(`${path}-shm`, { force: true })
    rmSync(`${path}-wal`, { force: true })
  }
  for (const directory of createdDirectories) {
    rmSync(directory, { recursive: true, force: true })
  }
  createdPaths.clear()
  createdDirectories.clear()
})

describe("Memory", () => {
  test("persists, searches, forgets, and records history", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open(baseOptions(filename))
    openClients.push(memory)

    const record = await memory.add("Alice is planning a trip to Tokyo next Tuesday.")
    expect(record.content.length).toBeGreaterThan(0)

    const results = await memory.search("Who is going to Tokyo?", { namespace: "test" })
    expect(results.length).toBeGreaterThan(0)
    expect(results[0]?.memory.id).toBe(record.id)

    const historyBeforeForget = await memory.history(record.id)
    expect(historyBeforeForget.some((event) => event.event === "ADD")).toBeTrue()

    await memory.forget(record.id)

    const resultsAfterForget = await memory.search("Tokyo", { namespace: "test" })
    expect(resultsAfterForget.length).toBe(0)

    const historyAfterForget = await memory.history(record.id)
    expect(historyAfterForget.some((event) => event.event === "FORGET")).toBeTrue()
  })

  test("deleteAll and reset clear stored memories", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-delete.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open(baseOptions(filename))
    openClients.push(memory)

    await memory.add("Alice likes coffee.", { namespace: "alpha" })
    await memory.add("Bob likes tea.", { namespace: "beta" })

    await memory.deleteAll({ namespace: "alpha" })
    expect((await memory.getAll({ namespace: "alpha" })).length).toBe(0)
    expect((await memory.getAll({ namespace: "beta" })).length).toBe(1)

    await memory.reset()
    expect((await memory.getAll({ namespace: "beta" })).length).toBe(0)
  })

  test("bootstraps migrations and parent directories for direct Effect runtime usage", async () => {
    const root = join(process.cwd(), `.tmp-fraction-runtime-${Date.now()}`)
    const filename = join(root, "nested", "effect-runtime.sqlite")
    createdDirectories.add(root)

    const runtime = createFractionRuntime({
      filename,
      defaultNamespace: "effect-runtime",
      compressorType: "heuristic"
    })
    openRuntimes.push(runtime)

    const record = await runtime.runPromise(
      MemoryService.use((memory) =>
        memory.add("Effect runtime writes work on a fresh database.", {
          namespace: "effect-runtime"
        })
      )
    )

    const results = await runtime.runPromise(
      MemoryService.use((memory) =>
        memory.search("fresh database", {
          namespace: "effect-runtime"
        })
      )
    )

    expect(existsSync(join(root, "nested"))).toBeTrue()
    expect(existsSync(filename)).toBeTrue()
    expect(record.content.length).toBeGreaterThan(0)
    expect(results.some((result) => result.memory.id === record.id)).toBeTrue()
  })

  test("creates parent directories before opening nested sqlite paths", async () => {
    const root = join(process.cwd(), `.tmp-fraction-open-${Date.now()}`)
    const filename = join(root, "app", "memory.sqlite")
    createdDirectories.add(root)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "nested-open",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    await memory.add("Nested sqlite paths now work without manual mkdir.", {
      namespace: "nested-open"
    })

    const results = await memory.search("manual mkdir", { namespace: "nested-open" })

    expect(existsSync(join(root, "app"))).toBeTrue()
    expect(existsSync(filename)).toBeTrue()
    expect(results.length).toBeGreaterThan(0)
  })

  test("validates metadata with Effect Schema when provided", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-schema.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      ...baseOptions(filename),
      metadataSchema: Schema.Struct({
        category: Schema.String,
        priority: Schema.Number
      })
    })
    openClients.push(memory)

    const record = await memory.add(
      "Schema-backed metadata entry.",
      { namespace: "test" },
      {
        category: "note",
        priority: 2
      }
    )

    expect(record.metadata.category).toBe("note")

    expect(() =>
      memory.add("Invalid metadata.", { namespace: "test" }, {
        category: "note",
        priority: "high"
      } as never)
    ).toThrow()
  })

  test("batches embeddings through embedMany when a provider exposes it", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-batch.sqlite`)
    createdPaths.add(filename)

    const batchCalls: Array<ReadonlyArray<string>> = []
    let singleCalls = 0
    const vectorFor = (text: string) =>
      Float32Array.from([text.length, Math.max(1, text.length % 7), 1, 0])

    const memory = await Memory.open({
      ...baseOptions(filename),
      embeddingProvider: {
        embed: (text) => {
          singleCalls += 1
          return vectorFor(text)
        },
        embedMany: (texts) => {
          batchCalls.push(texts)
          return texts.map((text) => vectorFor(text))
        }
      }
    })
    openClients.push(memory)

    const records = await memory.addMany(
      ["Ada works on Fraction's SDK.", "Linus prefers Bun-based tooling."],
      { namespace: "test" }
    )

    expect(records.length).toBe(2)
    expect(batchCalls.length).toBe(1)
    expect(batchCalls[0]?.length).toBe(2)
    expect(singleCalls).toBe(0)
  })

  test("batches embeddings from the stored fallback content when extraction is empty", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-batch-fallback-embedding.sqlite`)
    createdPaths.add(filename)

    const batchCalls: Array<ReadonlyArray<string>> = []
    const memory = await Memory.open({
      ...baseOptions(filename),
      extractionProvider: {
        extract: () => ({
          content: "   ",
          entities: [],
          eventAt: null
        })
      },
      embeddingProvider: {
        embed: (text) => Float32Array.from([text.length, 1, 0, 0]),
        embedMany: (texts) => {
          batchCalls.push(texts)
          return texts.map((text) => Float32Array.from([text.length, 1, 0, 0]))
        }
      }
    })
    openClients.push(memory)

    const records = await memory.addMany(
      ["Ada works on Fraction's SDK.", "Linus prefers Bun-based tooling."],
      { namespace: "test" }
    )

    expect(records.map((record) => record.content)).toEqual([
      "Ada works on Fraction's SDK.",
      "Linus prefers Bun-based tooling."
    ])
    expect(batchCalls).toEqual([
      ["Ada works on Fraction's SDK.", "Linus prefers Bun-based tooling."]
    ])
  })

  test("uses a configured extraction provider when present", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-extract.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      ...baseOptions(filename),
      extractionProvider: {
        extract: (text) => ({
          content: `fact: ${text.split(".")[0]!.trim()}`,
          entities: ["Grace Hopper"],
          eventAt: "2026-03-17T00:00:00.000Z"
        })
      }
    })
    openClients.push(memory)

    const record = await memory.add("Grace Hopper is reviewing the Fraction SDK today.", {
      namespace: "test"
    })

    expect(record.content).toBe("fact: Grace Hopper is reviewing the Fraction SDK today")
    expect(record.eventAt).toBe("2026-03-17T00:00:00.000Z")
  })

  test("retrieves interdependent project memories across shared entities and isolates scope", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-cluster.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "ops",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const kickoff = await memory.add(
      "Maya scheduled the Project Atlas kickoff with Ravi in Tokyo on March 20, 2026.",
      { namespace: "ops", userId: "maya" }
    )
    const budget = await memory.add(
      "Ravi owns the Atlas budget review and sends weekly risk updates to Maya.",
      { namespace: "ops", userId: "maya" }
    )
    const travel = await memory.add(
      "Ken will book Tokyo flights for Maya before the Atlas kickoff.",
      { namespace: "ops", userId: "maya" }
    )
    await memory.add("Maya and Ravi are discussing an unrelated Atlas event in Berlin.", {
      namespace: "other",
      userId: "maya"
    })

    const results = await memory.search(
      "When is Maya's Tokyo Atlas kickoff and who owns the budget?",
      { namespace: "ops", userId: "maya" },
      { limit: 5 }
    )

    const resultIds = new Set(results.map((result) => result.memory.id))
    expect(resultIds.has(kickoff.id)).toBeTrue()
    expect(resultIds.has(budget.id)).toBeTrue()
    expect(resultIds.has(travel.id)).toBeTrue()
    expect(results.every((result) => result.memory.scope.namespace === "ops")).toBeTrue()
  })

  test("keeps latest project version searchable while preserving linked history", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-versions.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "ops",
      compressorType: "heuristic"
    })
    openClients.push(memory)

    const original = await memory.add("Project Atlas launch is in Tokyo on March 20, 2026.", {
      namespace: "ops",
      agentId: "planner"
    })
    const updated = await memory.update(
      original.id,
      "Project Atlas launch moved to Singapore on April 2, 2026."
    )
    const followUp = await memory.add(
      "Maya informed Ravi that Atlas travel and hotel bookings now need to be for Singapore.",
      { namespace: "ops", agentId: "planner" }
    )

    const singaporeResults = await memory.search(
      "Where is the Atlas launch now and who was informed?",
      { namespace: "ops", agentId: "planner" },
      { limit: 5 }
    )

    const resultIds = new Set(singaporeResults.map((result) => result.memory.id))
    expect(resultIds.has(updated.id)).toBeTrue()
    expect(resultIds.has(followUp.id)).toBeTrue()

    const history = await memory.history(updated.id)
    expect(
      history.some((event) => event.event === "ADD" && event.memoryId === original.id)
    ).toBeTrue()
    expect(
      history.some((event) => event.event === "UPDATE" && event.memoryId === updated.id)
    ).toBeTrue()

    const allRecords = await memory.getAll({ namespace: "ops", agentId: "planner" })
    expect(
      allRecords.some((record) => record.id === original.id && record.isLatest === false)
    ).toBeTrue()
    expect(
      allRecords.some((record) => record.id === updated.id && record.isLatest === true)
    ).toBeTrue()
  })

  test("falls back to heuristic compression when llmlingua is unavailable", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-llml-fallback.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "test",
      adaptiveCompression: false,
      compressorType: "llmlingua2",
      llmlingua: {
        model: "fraction/non-existent-llmlingua-model",
        onUnavailable: "fallback-heuristic"
      }
    })
    openClients.push(memory)

    const rawText =
      "Avery is coordinating the quarterly planning review with Jordan and Priya in Singapore next Tuesday. The final agenda includes a finance checkpoint. The travel logistics review needs hotel confirmations. Jordan also needs to circulate the updated board memo before the meeting."
    const record = await memory.add(rawText)

    expect(record.content.length).toBeGreaterThan(0)
    expect(record.content.length).toBeLessThan(rawText.length)
  })

  test("falls back to heuristic compression for addMany when llmlingua is unavailable", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-llml-batch-fallback.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "test",
      adaptiveCompression: false,
      compressorType: "llmlingua2",
      llmlingua: {
        model: "fraction/non-existent-llmlingua-model",
        onUnavailable: "fallback-heuristic"
      }
    })
    openClients.push(memory)

    const records = await memory.addMany(
      [
        "Avery is coordinating the quarterly planning review with Jordan and Priya in Singapore next Tuesday.",
        "Jordan also needs to circulate the updated board memo before the meeting."
      ],
      { namespace: "test" }
    )

    expect(records.length).toBe(2)
    expect(records.every((record) => record.content.length > 0)).toBeTrue()
  })

  test("errors when llmlingua is unavailable and fallback is disabled", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-llml-error.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "test",
      adaptiveCompression: false,
      compressorType: "llmlingua2",
      llmlingua: {
        model: "fraction/non-existent-llmlingua-model",
        onUnavailable: "error"
      }
    })
    openClients.push(memory)

    let didThrow = false
    try {
      await memory.add(
        "The operations review for Fraction is scheduled with Maya, Ravi, and Ken next Thursday, and they need to finalize the Singapore travel plan before finance approval."
      )
    } catch {
      didThrow = true
    }

    expect(didThrow).toBeTrue()
  })

  test("errors for addMany when llmlingua is unavailable and fallback is disabled", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-llml-batch-error.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "test",
      adaptiveCompression: false,
      compressorType: "llmlingua2",
      llmlingua: {
        model: "fraction/non-existent-llmlingua-model",
        onUnavailable: "error"
      }
    })
    openClients.push(memory)

    let didThrow = false
    try {
      await memory.addMany(
        [
          "The operations review for Fraction is scheduled with Maya, Ravi, and Ken next Thursday.",
          "They need to finalize the Singapore travel plan before finance approval."
        ],
        { namespace: "test" }
      )
    } catch {
      didThrow = true
    }

    expect(didThrow).toBeTrue()
  })

  test("uses a custom compression provider ahead of local llmlingua", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-compression-provider.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "test",
      adaptiveCompression: false,
      compressionProvider: {
        compress: (text) => ({
          content: `compressed: ${text.split(".")[0]!.trim()}`,
          mode: "provider",
          source: "remote"
        })
      }
    })
    openClients.push(memory)

    const record = await memory.add("Mina owns the Fraction release checklist. She also coordinates QA.")
    expect(record.content).toBe("compressed: Mina owns the Fraction release checklist")
  })

  test("batches compression through compressMany when a provider exposes it", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-compression-batch.sqlite`)
    createdPaths.add(filename)

    const batchCalls: Array<ReadonlyArray<string>> = []
    let singleCalls = 0

    const memory = await Memory.open({
      filename,
      defaultNamespace: "test",
      adaptiveCompression: false,
      compressionProvider: {
        compress: (text) => {
          singleCalls += 1
          return {
            content: text.toUpperCase(),
            mode: "provider",
            source: "remote"
          }
        },
        compressMany: (texts) => {
          batchCalls.push(texts)
          return texts.map((text) => ({
            content: text.toUpperCase(),
            mode: "provider" as const,
            source: "remote" as const
          }))
        }
      }
    })
    openClients.push(memory)

    const records = await memory.addMany(
      [
        "Mina coordinates the release notes and changelog.",
        "Ravi validates the staging deployment checklist."
      ],
      { namespace: "test" }
    )

    expect(records.length).toBe(2)
    expect(batchCalls.length).toBe(1)
    expect(batchCalls[0]?.length).toBe(2)
    expect(singleCalls).toBe(0)
  })

  test("applies metadata filters before truncating retrieval candidates", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-filtered-retrieval.sqlite`)
    createdPaths.add(filename)

    const embed = (text: string) =>
      text === "common"
        ? Float32Array.from([1, 0, 0, 0])
        : text.includes("rare-signal")
          ? Float32Array.from([0, 1, 0, 0])
          : Float32Array.from([1, 0, 0, 0])

    const memory = await Memory.open({
      ...baseOptions(filename),
      embeddingProvider: {
        embed,
        embedMany: (texts) => texts.map((text) => embed(text))
      }
    })
    openClients.push(memory)

    for (let index = 0; index < 5; index++) {
      await memory.add(`common common common filler-${index}`, { namespace: "test" }, {
        topic: "other"
      })
    }

    const matching = await memory.add("common rare-signal", { namespace: "test" }, {
      topic: "keep"
    })

    const results = await memory.search(
      "common",
      { namespace: "test" },
      {
        limit: 1,
        filter: {
          op: "eq",
          field: "topic",
          value: "keep"
        }
      }
    )

    expect(results.length).toBe(1)
    expect(results[0]?.memory.id).toBe(matching.id)
  })

  test("hard delete removes full chain rows and decrements shared graph edges", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-delete-graph.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open(baseOptions(filename))
    openClients.push(memory)

    await memory.add("Alice met Bob in London.", { namespace: "test" })
    const original = await memory.add("Alice met Bob in Paris.", { namespace: "test" })
    const updated = await memory.update(original.id, "Alice met Bob in Paris on Friday.")

    const db = new Database(filename, { readonly: true })
    openDatabases.push(db)

    const beforeDelete = queryGet<{ weight: number }>(
      db,
      `SELECT entity_edges.weight AS weight
       FROM entity_edges
       JOIN entities AS source ON source.id = entity_edges.source_entity_id
       JOIN entities AS target ON target.id = entity_edges.target_entity_id
       WHERE source.normalized = ? AND target.normalized = ?`,
      ["alice", "bob"]
    )
    expect(beforeDelete?.weight).toBe(2)

    await memory.delete(updated.id)

    const rootCounts = queryGet<{
      memoryCount: number
      versionCount: number
      historyCount: number
    }>(
      db,
      `SELECT
         (SELECT COUNT(*) FROM memories WHERE root_id = ?) AS memoryCount,
         (SELECT COUNT(*) FROM memory_versions WHERE root_id = ?) AS versionCount,
         (SELECT COUNT(*) FROM memory_history WHERE root_id = ?) AS historyCount`,
      [original.rootId, original.rootId, original.rootId]
    )

    expect(rootCounts?.memoryCount).toBe(0)
    expect(rootCounts?.versionCount).toBe(0)
    expect(rootCounts?.historyCount).toBe(0)

    const afterDelete = queryGet<{ weight: number }>(
      db,
      `SELECT entity_edges.weight AS weight
       FROM entity_edges
       JOIN entities AS source ON source.id = entity_edges.source_entity_id
       JOIN entities AS target ON target.id = entity_edges.target_entity_id
       WHERE source.normalized = ? AND target.normalized = ?`,
      ["alice", "bob"]
    )
    expect(afterDelete?.weight).toBe(1)

    const parisEntity = queryGet<{ normalized: string }>(
      db,
      "SELECT normalized FROM entities WHERE normalized = ?",
      ["paris"]
    )
    expect(parisEntity).toBeNull()
  })

  test("update removes previous graph state and keeps only the latest version in graph tables", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-update-graph.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open(baseOptions(filename))
    openClients.push(memory)

    const original = await memory.add("Alice met Bob in Paris.", { namespace: "test" })
    const updated = await memory.update(original.id, "Alice met Carol in Rome.")

    const db = new Database(filename, { readonly: true })
    openDatabases.push(db)

    const originalGraphRows = queryGet<{ count: number }>(
      db,
      "SELECT COUNT(*) AS count FROM memory_entities WHERE memory_id = ?",
      [original.id]
    )
    expect(originalGraphRows?.count).toBe(0)

    const originalEdgeRows = queryGet<{ count: number }>(
      db,
      "SELECT COUNT(*) AS count FROM memory_entity_edges WHERE memory_id = ?",
      [original.id]
    )
    expect(originalEdgeRows?.count).toBe(0)

    const updatedGraphRows = queryGet<{ count: number }>(
      db,
      "SELECT COUNT(*) AS count FROM memory_entities WHERE memory_id = ?",
      [updated.id]
    )
    expect(updatedGraphRows?.count).toBeGreaterThan(0)

    const bobEntity = queryGet<{ normalized: string }>(
      db,
      "SELECT normalized FROM entities WHERE normalized = ?",
      ["bob"]
    )
    expect(bobEntity).toBeNull()

    const aliceCarolEdge = queryGet<{ weight: number }>(
      db,
      `SELECT entity_edges.weight AS weight
       FROM entity_edges
       JOIN entities AS source ON source.id = entity_edges.source_entity_id
       JOIN entities AS target ON target.id = entity_edges.target_entity_id
       WHERE source.normalized = ? AND target.normalized = ?`,
      ["alice", "carol"]
    )
    expect(aliceCarolEdge?.weight).toBe(1)
  })
})
