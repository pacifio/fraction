import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"
import { Fraction } from "fraction"

import { fractionTools, withFractionMemory } from "../src/index"

const memory = await Fraction.open({
  filename: ".fraction/vercel-example.sqlite",
  defaultNamespace: "chat"
})

await memory.add("Rina prefers concise technical answers.", {
  namespace: "chat",
  userId: "rina"
})

const model = withFractionMemory(openai("gpt-4.1"), {
  memory,
  scope: { namespace: "chat", userId: "rina" }
})

const result = await generateText({
  model,
  prompt: "How should I answer Rina?",
  tools: fractionTools({
    memory,
    scope: { namespace: "chat", userId: "rina" }
  })
})

console.log(result.text)

await memory.close()
