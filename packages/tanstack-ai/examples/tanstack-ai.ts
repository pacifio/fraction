import { chat } from "@tanstack/ai"
import { Fraction } from "fraction"

import { fractionMiddleware, fractionTools } from "../src/index"

const memory = await Fraction.open({
  filename: ".fraction/tanstack-example.sqlite",
  defaultNamespace: "chat"
})

await memory.add("Rina prefers concise technical answers.", {
  namespace: "chat",
  userId: "rina"
})

const response = chat({
  messages: [{ role: "user", content: "How should I answer Rina?" }],
  middleware: [
    fractionMiddleware({
      memory,
      scope: { namespace: "chat", userId: "rina" }
    })
  ],
  tools: fractionTools({
    memory,
    scope: { namespace: "chat", userId: "rina" }
  })
})

console.log(response)

await memory.close()
