# Fraction

Effect-first local memory SDK for Bun.

## What It Includes

- Bun-native persistent memory store backed by SQLite
- Effect v4 services, layers, schemas, and tagged errors
- Promise-friendly `Memory` / `Fraction` client API
- Hybrid recall with lexical, vector, entity, and temporal scoring
- Metadata-schema validation with Effect `Schema`

## Install

```bash
bun add fraction
```

## Basic Usage

```ts
import { Memory } from "fraction"

const memory = await Memory.open({
  filename: ".fraction/app.sqlite",
  defaultNamespace: "app"
})

await memory.add("Rina prefers concise technical answers.")

const results = await memory.search("How should I answer Rina?")
console.log(results[0]?.memory.content)

await memory.close()
```

## Effect Runtime Usage

```ts
import { createFractionRuntime, MemoryService } from "fraction/effect"

const runtime = createFractionRuntime({
  filename: ".fraction/app.sqlite",
  defaultNamespace: "app"
})

const record = await runtime.runPromise(
  MemoryService.use((memory) =>
    memory.add("Use Bun for the runtime layer.", { namespace: "app" })
  )
)

console.log(record.id)
await runtime.dispose()
```

## Metadata Validation

```ts
import { Schema } from "effect"
import { Memory } from "fraction"

const memory = await Memory.open({
  filename: ".fraction/app.sqlite",
  metadataSchema: Schema.Struct({
    category: Schema.String,
    priority: Schema.Number
  })
})

await memory.add(
  "Ship the Bun benchmark.",
  { namespace: "app" },
  { category: "task", priority: 1 }
)
```

## Examples

- [basic.ts](./examples/basic.ts)
- [effect-runtime.ts](./examples/effect-runtime.ts)
