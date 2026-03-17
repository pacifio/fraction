import { Memory } from "../src/index"

const memory = await Memory.open({
  filename: ".fraction/example.sqlite",
  defaultNamespace: "example"
})

await memory.add("Rina prefers concise technical answers.", {
  namespace: "example",
  userId: "rina"
})
await memory.add("Rina is planning a trip to Tokyo next Tuesday.", {
  namespace: "example",
  userId: "rina"
})

const results = await memory.search("How should I answer Rina about Tokyo?", {
  namespace: "example",
  userId: "rina"
})

console.log(
  results.map((result) => ({
    id: result.memory.id,
    content: result.memory.content,
    score: result.score
  }))
)

await memory.close()
