import { afterEach, describe, expect, test } from "bun:test"
import { rmSync } from "node:fs"
import { join } from "node:path"
import { Schema } from "effect"

import { Memory } from "../src/index"

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

describe("Memory", () => {
  test("persists, searches, forgets, and records history", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({ filename, defaultNamespace: "test" })
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

    const memory = await Memory.open({ filename, defaultNamespace: "test" })
    openClients.push(memory)

    await memory.add("Alice likes coffee.", { namespace: "alpha" })
    await memory.add("Bob likes tea.", { namespace: "beta" })

    await memory.deleteAll({ namespace: "alpha" })
    expect((await memory.getAll({ namespace: "alpha" })).length).toBe(0)
    expect((await memory.getAll({ namespace: "beta" })).length).toBe(1)

    await memory.reset()
    expect((await memory.getAll({ namespace: "beta" })).length).toBe(0)
  })

  test("validates metadata with Effect Schema when provided", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-schema.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "test",
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
      filename,
      defaultNamespace: "test",
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

  test("uses a configured extraction provider when present", async () => {
    const filename = join(process.cwd(), `.tmp-fraction-${Date.now()}-extract.sqlite`)
    createdPaths.add(filename)

    const memory = await Memory.open({
      filename,
      defaultNamespace: "test",
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

    const memory = await Memory.open({ filename, defaultNamespace: "ops" })
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

    const memory = await Memory.open({ filename, defaultNamespace: "ops" })
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
})
