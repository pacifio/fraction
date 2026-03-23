#!/usr/bin/env bun

import { createLlmlingua2Compressor } from "../src/internal/llmlingua/compressor"

const model =
  process.env.FRACTION_LLMLINGUA_VALIDATE_MODEL ??
  process.env.FRACTION_LLMLINGUA_OUTPUT_DIR ??
  ".tmp/llmlingua-2-bert-base-multilingual-cased-meetingbank-onnx"
const cacheDir = process.env.FRACTION_LLMLINGUA_CACHE_DIR ?? ".tmp/llmlingua-cache"

const assert = (condition: unknown, message: string): asserts condition => {
  if (!condition) {
    throw new Error(message)
  }
}

const longMeetingSample = [
  "Ana is leading the multilingual launch review for Fraction.",
  "The team agreed that billing rollout happens next Tuesday after the ops checklist is signed.",
  "Support needs the migration guide, release notes, and customer-specific risk flags before launch.",
  "Sam will own the benchmark rerun and publish the final numbers after QA signs off."
].join(" ")

const numericReasoningSample =
  "Project Atlas costs 12000 dollars in Q2 and 18000 dollars in Q3, so reserve both figures in the compressed output."

const compressor = createLlmlingua2Compressor({
  model,
  cacheDir,
  downloadModelIfMissing: false,
  modelFamily: "bert",
  batchSize: 4
})

try {
  const compressed = await compressor.compress(longMeetingSample, {
    rate: 0.6,
    forceTokens: ["Fraction", "Tuesday"]
  })
  assert(compressed.content.length > 0, "Compression returned empty content")
  assert(
    compressed.tokenCountAfter <= compressed.tokenCountBefore,
    "Compression increased token count"
  )
  assert(compressed.content.includes("Fraction"), "Forced token 'Fraction' was not preserved")
  assert(compressed.content.includes("Tuesday"), "Forced token 'Tuesday' was not preserved")

  const promptCompression = await compressor.compressPrompt(
    [longMeetingSample, numericReasoningSample],
    {
      rate: 0.6,
      targetToken: 80,
      forceTokens: ["12000", "18000"],
      forceReserveDigit: true,
      useContextLevelFilter: true
    }
  )
  assert(promptCompression.content.length > 0, "Prompt compression returned empty content")
  assert(
    promptCompression.tokenCountAfter <= promptCompression.tokenCountBefore,
    "Prompt compression increased token count"
  )
  assert(promptCompression.content.includes("12000"), "Digit preservation failed for 12000")
  assert(promptCompression.content.includes("18000"), "Digit preservation failed for 18000")

  console.log("LLMLingua export validation passed.")
  console.log(`Model: ${model}`)
  console.log(
    JSON.stringify(
      {
        single: compressed,
        prompt: promptCompression
      },
      null,
      2
    )
  )
} finally {
  await compressor.close()
}
