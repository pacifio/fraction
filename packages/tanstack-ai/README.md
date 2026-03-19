# @fraction/tanstack-ai

TanStack AI middleware and tools for Fraction memory.

## Install

```bash
bun add @fraction/tanstack-ai fraction @tanstack/ai
```

## What It Exposes

- `fractionMiddleware(options)`
- `withFractionMemory(options)`
- `fractionTools(options)`
- `remember(options)`
- `recall(options)`
- `formatFractionContext(options)`
- `createTanStackExtractionProvider(options)`
- `createTanStackCompressionProvider(options)`

## Example

```ts
import { chat } from "@tanstack/ai"
import { Fraction } from "fraction"
import { fractionMiddleware, fractionTools } from "@fraction/tanstack-ai"

const memory = await Fraction.open({
  filename: ".fraction/app.sqlite",
  defaultNamespace: "chat"
})

const result = await chat({
  messages: [{ role: "user", content: "What style does Rina prefer?" }],
  middleware: [
    fractionMiddleware({
      memory,
      scope: { namespace: "chat", userId: "u_123" }
    })
  ],
  tools: fractionTools({
    memory,
    scope: { namespace: "chat", userId: "u_123" }
  })
})

console.log(result)
await memory.close()
```

## Examples

- [tanstack-ai.ts](./examples/tanstack-ai.ts)
