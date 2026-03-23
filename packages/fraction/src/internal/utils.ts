import { createHash, randomUUID } from "node:crypto"
import * as chrono from "chrono-node"

import type { FilterExpr, JsonRecord, Message, Scope } from "./types"

export const nowIso = () => new Date().toISOString()

export const makeMemoryId = () => randomUUID()

export const hashText = (value: string) => createHash("sha256").update(value).digest("hex")

export const normalizeScope = (scope: Scope | undefined, defaultNamespace: string): Scope => ({
  namespace: scope?.namespace ?? defaultNamespace,
  userId: scope?.userId,
  agentId: scope?.agentId,
  runId: scope?.runId,
  appId: scope?.appId
})

export const hasScope = (scope: Scope) =>
  Boolean(scope.namespace || scope.userId || scope.agentId || scope.runId || scope.appId)

export const scopeToJson = (scope: Scope) => JSON.stringify(scope)

export const jsonToScope = (value: string): Scope => JSON.parse(value) as Scope

export const metadataToJson = (value: JsonRecord | undefined) => JSON.stringify(value ?? {})

export const jsonToMetadata = (value: string): JsonRecord => JSON.parse(value) as JsonRecord

export const cosineSimilarity = (a: Float32Array, b: Float32Array) => {
  if (a.length !== b.length) {
    return 0
  }
  let dot = 0
  let normA = 0
  let normB = 0
  for (let index = 0; index < a.length; index++) {
    const av = a[index]!
    const bv = b[index]!
    dot += av * bv
    normA += av * av
    normB += bv * bv
  }
  if (normA === 0 || normB === 0) {
    return 0
  }
  return dot / Math.sqrt(normA * normB)
}

export const encodeVector = (vector: Float32Array) => Buffer.from(vector.buffer.slice(0))

export const decodeVector = (value: Uint8Array | ArrayBuffer | Buffer) => {
  const buffer =
    value instanceof ArrayBuffer
      ? value
      : value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength)
  return new Float32Array(buffer)
}

export const toFloat32Vector = (value: Float32Array | ArrayLike<number> | ReadonlyArray<number>) =>
  value instanceof Float32Array ? value : Float32Array.from(value)

const tokenize = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter(Boolean)

export const embedTextLocal = (value: string, dimensions: number) => {
  const vector = new Float32Array(dimensions)
  const tokens = tokenize(value)
  for (const token of tokens) {
    const digest = createHash("sha256").update(token).digest()
    for (let offset = 0; offset < digest.length; offset += 4) {
      const chunk = digest.readUInt32BE(offset)
      const index = chunk % dimensions
      const sign = chunk % 2 === 0 ? 1 : -1
      vector[index] = (vector[index] ?? 0) + sign
    }
  }
  let norm = 0
  for (const value of vector) {
    norm += value * value
  }
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let index = 0; index < vector.length; index++) {
      vector[index] = vector[index]! / norm
    }
  }
  return vector
}

export const splitSentences = (value: string) =>
  value
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean)

export const extractEntities = (value: string) => {
  const entities = new Set<string>()
  const capitalized = value.match(/\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/g) ?? []
  const emails = value.match(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi) ?? []
  for (const entity of [...capitalized, ...emails]) {
    if (entity.length > 1) {
      entities.add(entity.trim())
    }
  }
  return [...entities]
}

export const extractEventAt = (value: string) => chrono.parseDate(value)?.toISOString()

export const compressText = (value: string, maxFactsPerInput: number) => {
  const sentences = splitSentences(value)
  if (sentences.length <= maxFactsPerInput) {
    return value.trim()
  }
  const scored = sentences.map((sentence) => {
    const entityScore = extractEntities(sentence).length * 2
    const timeScore = extractEventAt(sentence) ? 3 : 0
    const numberScore = /\d/.test(sentence) ? 1 : 0
    const preferenceScore = /\b(love|like|prefer|favorite|allergic|live|moved|work|born)\b/i.test(
      sentence
    )
      ? 2
      : 0
    return {
      sentence,
      score: entityScore + timeScore + numberScore + preferenceScore
    }
  })
  return scored
    .sort((left, right) => right.score - left.score)
    .slice(0, maxFactsPerInput)
    .map((entry) => entry.sentence)
    .join(" ")
    .trim()
}

export const messagesToText = (input: string | ReadonlyArray<Message>) =>
  typeof input === "string"
    ? input
    : input.map((message) => `${message.role}: ${message.content}`).join("\n")

export const queryTextFromMessages = (messages: ReadonlyArray<Message>) => {
  const lastUser = [...messages]
    .reverse()
    .find((message) => message.role === "user" && message.content.trim().length > 0)
  return lastUser?.content ?? messages.map((message) => message.content).join("\n")
}

export const matchesFilter = (metadata: JsonRecord, filter: FilterExpr | undefined): boolean => {
  if (!filter) {
    return true
  }
  if ("field" in filter) {
    const actual = readField(metadata, filter.field)
    if (filter.op === "eq") {
      return actual === filter.value
    }
    if (filter.op === "contains") {
      return stringifyJsonLike(actual)
        .toLowerCase()
        .includes(stringifyJsonLike(filter.value).toLowerCase())
    }
    return true
  }
  if (filter.op === "and") {
    return filter.filters.every((item) => matchesFilter(metadata, item))
  }
  if (filter.op === "or") {
    return filter.filters.some((item) => matchesFilter(metadata, item))
  }
  return true
}

const stringifyJsonLike = (value: unknown) => {
  if (value === null || value === undefined) {
    return ""
  }
  if (typeof value === "string") {
    return value
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value)
  }
  return JSON.stringify(value)
}

const readField = (record: JsonRecord, path: string) =>
  path.split(".").reduce<unknown>((current, key) => {
    if (current && typeof current === "object" && key in current) {
      return (current as Record<string, unknown>)[key]
    }
    return undefined
  }, record)

export const rrfScore = (
  rankings: ReadonlyArray<ReadonlyArray<{ readonly id: string; readonly score: number }>>,
  weights: ReadonlyArray<number>,
  k: number
) => {
  const aggregate = new Map<string, { score: number; signals: Record<string, number> }>()
  rankings.forEach((ranking, rankingIndex) => {
    ranking.forEach((entry, index) => {
      const current = aggregate.get(entry.id) ?? { score: 0, signals: {} }
      const contribution = weights[rankingIndex]! / (k + index + 1)
      current.score += contribution
      current.signals[`signal_${rankingIndex}`] = entry.score
      aggregate.set(entry.id, current)
    })
  })
  return aggregate
}
