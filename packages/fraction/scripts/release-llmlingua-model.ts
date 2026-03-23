#!/usr/bin/env bun

import { $ } from "bun"

import {
  exportModelToOnnx,
  getLlmlinguaReleaseConfig,
  syncFolderToArtifactDir,
  validatePublishedConfig,
  writeModelMetadata
} from "./lib/llmlingua-release"

const config = getLlmlinguaReleaseConfig()
const shouldSkipConvert = process.argv.includes("--skip-convert")
const shouldSkipValidate = process.argv.includes("--skip-validate")
const shouldSkipSync = process.argv.includes("--skip-sync")

validatePublishedConfig(config)

if (!shouldSkipConvert) {
  console.log(`Exporting ${config.sourceModel} to ${config.outputDir}`)
  await exportModelToOnnx(config)
  await writeModelMetadata(config)
} else {
  console.log(`Skipping ONNX export and reusing ${config.outputDir}`)
}

if (!shouldSkipValidate) {
  console.log(`Validating exported artifacts from ${config.outputDir}`)
  await $`env FRACTION_LLMLINGUA_VALIDATE_MODEL=${config.outputDir} bun scripts/validate-llmlingua-model.ts`
} else {
  console.log("Skipping local validation")
}

if (!shouldSkipSync) {
  console.log(`Syncing ${config.outputDir} into ${config.artifactDir ?? "<unset>"}`)
  await syncFolderToArtifactDir(config)
} else {
  console.log("Skipping artifact directory sync")
}

console.log("LLMLingua release flow completed.")
