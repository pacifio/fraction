import { existsSync } from "node:fs"
import { mkdir, stat } from "node:fs/promises"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"

import { CompressionUnavailable } from "../errors"

const DEFAULT_CACHE_DIR = join(homedir(), ".fraction", "models")
const DEFAULT_MODEL_DIRNAME = "llmlingua-2-bert-base-multilingual-cased-meetingbank-onnx"

export const DEFAULT_LLMLINGUA_ARTIFACT_FILES = [
  "config.json",
  "special_tokens_map.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.txt",
  "onnx/model.onnx"
] as const

const normalizeBaseUrl = (value: string) => value.replace(/\/+$/, "")

const isHttpUrl = (value: string) => /^https?:\/\//.test(value)

const isPathLike = (value: string) =>
  value.startsWith("./") ||
  value.startsWith("../") ||
  value.startsWith("/") ||
  value.startsWith("~") ||
  /^[A-Za-z]:[\\/]/.test(value)

const expandHome = (value: string) => (value.startsWith("~") ? join(homedir(), value.slice(1)) : value)

const hasRequiredArtifacts = async (
  modelDir: string,
  files: ReadonlyArray<string>
) => {
  for (const file of files) {
    const target = join(modelDir, file)
    if (!existsSync(target)) {
      return false
    }
    const info = await stat(target)
    if (!info.isFile()) {
      return false
    }
  }
  return true
}

const downloadArtifact = async (baseUrl: string, file: string, targetPath: string) => {
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/${file}`)
  if (!response.ok) {
    throw new Error(`Failed to download ${file}: ${response.status} ${response.statusText}`)
  }

  await mkdir(dirname(targetPath), { recursive: true })
  await Bun.write(targetPath, await response.bytes())
}

export interface LlmlinguaArtifactOptions {
  readonly model: string
  readonly cacheDir?: string
  readonly downloadModelIfMissing?: boolean
  readonly artifactBaseUrl?: string
  readonly artifactFiles?: ReadonlyArray<string>
}

export const defaultLlmlinguaModelPath = (cacheDir = DEFAULT_CACHE_DIR) =>
  join(cacheDir, DEFAULT_MODEL_DIRNAME)

export const ensureLlmlinguaLocalModelPath = async (
  options: LlmlinguaArtifactOptions
) => {
  const files = options.artifactFiles ?? DEFAULT_LLMLINGUA_ARTIFACT_FILES
  const useLocalPath = isPathLike(options.model)
  const modelPath = useLocalPath
    ? resolve(expandHome(options.model))
    : options.model

  if (!useLocalPath) {
    return modelPath
  }

  if (await hasRequiredArtifacts(modelPath, files)) {
    return modelPath
  }

  if (options.downloadModelIfMissing === false) {
    throw new CompressionUnavailable({
      message: "LLMLingua-2 artifacts are missing from the local model directory",
      model: modelPath
    })
  }

  if (!options.artifactBaseUrl || !isHttpUrl(options.artifactBaseUrl)) {
    throw new CompressionUnavailable({
      message:
        "LLMLingua-2 local model directory is missing artifacts and no artifactBaseUrl is configured",
      model: modelPath
    })
  }

  await mkdir(modelPath, { recursive: true })
  for (const file of files) {
    const targetPath = join(modelPath, file)
    if (existsSync(targetPath)) {
      continue
    }
    await downloadArtifact(options.artifactBaseUrl, file, targetPath)
  }

  return modelPath
}
