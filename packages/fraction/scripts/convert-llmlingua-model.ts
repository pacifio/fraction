#!/usr/bin/env bun

import {
  exportModelToOnnx,
  getLlmlinguaReleaseConfig,
  validatePublishedConfig,
  writeModelMetadata
} from "./lib/llmlingua-release"

const config = getLlmlinguaReleaseConfig()

console.log(`Converting ${config.sourceModel} to ONNX in ${config.outputDir}`)
console.log("Bootstrapping a local Python toolchain for Optimum export.")

validatePublishedConfig(config)
await exportModelToOnnx(config)
await writeModelMetadata(config)

console.log("Conversion complete.")
console.log(`Exported model folder: ${config.outputDir}`)
console.log(`Suggested next step: bun scripts/validate-llmlingua-model.ts`)
