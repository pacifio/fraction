import { mkdirSync, rmSync } from "node:fs"
import { join } from "node:path"
import { performance } from "node:perf_hooks"

import { Memory } from "../packages/fraction/src/index"

const benchmarkDir = join(process.cwd(), ".benchmark-fraction")
mkdirSync(benchmarkDir, { recursive: true })

const filename = join(benchmarkDir, `fraction-${Date.now()}.sqlite`)

const memory = await Memory.open({
  filename,
  defaultNamespace: "benchmark"
})

const samples = [
  "Rina prefers concise technical answers and travels frequently for work.",
  "Sam is migrating Fraction to Bun with Effect and SQLite.",
  "Nadia is allergic to peanuts and keeps a nut-free diet.",
  "The team plans to benchmark add and search latency on local memory workloads.",
  "Tokyo is Rina's next destination and she leaves next Tuesday morning."
]

const addStart = performance.now()
for (let index = 0; index < 200; index++) {
  const sample = samples[index % samples.length]!
  await memory.add(`${sample} [sample=${index}]`, { namespace: "benchmark", userId: "bench" })
}
const addDuration = performance.now() - addStart

const searchStart = performance.now()
for (let index = 0; index < 50; index++) {
  await memory.search("Who is traveling to Tokyo and what style do they prefer?", {
    namespace: "benchmark",
    userId: "bench"
  })
}
const searchDuration = performance.now() - searchStart

const rows = await memory.getAll({ namespace: "benchmark", userId: "bench" })

console.log({
  rows: rows.length,
  addMsTotal: Number(addDuration.toFixed(2)),
  addMsPerItem: Number((addDuration / 200).toFixed(3)),
  searchMsTotal: Number(searchDuration.toFixed(2)),
  searchMsPerQuery: Number((searchDuration / 50).toFixed(3))
})

await memory.close()
rmSync(filename, { force: true })
rmSync(`${filename}-shm`, { force: true })
rmSync(`${filename}-wal`, { force: true })
rmSync(benchmarkDir, { force: true, recursive: true })
