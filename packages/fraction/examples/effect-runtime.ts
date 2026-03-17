import { MemoryService, createFractionRuntime } from "../src/effect"

const runtime = createFractionRuntime({
  filename: ".fraction/effect-example.sqlite",
  defaultNamespace: "effect-example"
})

const record = await runtime.runPromise(
  MemoryService.use((memory) =>
    memory.add("Use the Effect runtime directly when you want Layer-based composition.", {
      namespace: "effect-example"
    })
  )
)

const results = await runtime.runPromise(
  MemoryService.use((memory) =>
    memory.search("What should I use for layer-based composition?", {
      namespace: "effect-example"
    })
  )
)

console.log({ record, topHit: results[0] })

await runtime.dispose()
