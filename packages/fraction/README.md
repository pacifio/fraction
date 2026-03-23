# Fraction

Effect-first local memory SDK for Bun.

## What It Includes

- Bun-native persistent memory store backed by SQLite
- Effect v4 services, layers, schemas, and tagged errors
- Promise-friendly `Memory` / `Fraction` client API
- Hybrid recall with lexical, vector, entity, and temporal scoring
- Metadata-schema validation with Effect `Schema`
- Local-first compression with built-in LLMLingua-2 hooks and heuristic fallback

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

## Compression

```ts
import { Memory } from "fraction"

const memory = await Memory.open({
  filename: ".fraction/app.sqlite",
  defaultNamespace: "app",
  llmlingua: {
    artifactBaseUrl:
      "https://raw.githubusercontent.com/your-org/your-model-repo/main/llmlingua-2-bert-base-multilingual-cased-meetingbank-onnx"
  }
})

await memory.add("Rina prefers concise technical answers and weekly planning summaries.")
await memory.close()
```

Helpful exports:

- `createLlmlingua2CompressionProvider(...)`
- `createLlmlingua2Compressor(...)`
- `prefetchLlmlinguaModel(...)`
- `createHeuristicCompressionProvider(...)`

To automate the Fraction-owned ONNX export and publish flow:

```bash
cd packages/fraction
bun run convert:llmlingua
bun run validate:llmlingua
export FRACTION_LLMLINGUA_ARTIFACT_DIR=/absolute/path/to/your/cloned-model-repo/llmlingua-2-bert-base-multilingual-cased-meetingbank-onnx
bun run release:llmlingua --skip-convert --skip-validate
```

Useful environment variables:

- `FRACTION_LLMLINGUA_SOURCE_MODEL`
- `FRACTION_LLMLINGUA_OUTPUT_DIR`
- `FRACTION_LLMLINGUA_ARTIFACT_BASE_URL`
- `FRACTION_LLMLINGUA_ARTIFACT_DIR`
- `FRACTION_LLMLINGUA_VENV_DIR`

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
