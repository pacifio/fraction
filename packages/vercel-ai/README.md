# @fraction/vercel-ai

Vercel AI SDK adapter for Fraction memory.

## Install

```bash
bun add @fraction/vercel-ai fraction ai
```

## What It Exposes

- `withFractionMemory(model, options)`
- `fractionTools(options)`
- `remember(options)`
- `recall(options)`
- `formatFractionContext(options)`

## Example

```ts
import { openai } from "@ai-sdk/openai"
import { generateText } from "ai"
import { Fraction } from "fraction"
import { withFractionMemory } from "@fraction/vercel-ai"

const memory = await Fraction.open({
  filename: ".fraction/app.sqlite",
  defaultNamespace: "chat"
})

const model = withFractionMemory(openai("gpt-4.1"), {
  memory,
  scope: { namespace: "chat", userId: "u_123" }
})

const result = await generateText({
  model,
  prompt: "How should I respond to Rina?"
})

console.log(result.text)
await memory.close()
```

## Examples

- [vercel-ai.ts](./examples/vercel-ai.ts)
