import { createHash } from "node:crypto"
import { existsSync, readFileSync, statSync } from "node:fs"
import { cpus, freemem, platform, release, totalmem } from "node:os"
import { basename, resolve } from "node:path"

export type BenchmarkTier = "strict-pipeline" | "shipped-defaults"
export type BenchmarkMode = "exact-config" | "best-effort-local" | "shipped-defaults"
export type CachePolicy = "cold" | "warm" | "mixed"
export type RuntimeTarget = "python" | "typescript"
export type ConversationStrategy = "speaker-partitioned-merge-v1"
export type WorkloadKind =
  | "synthetic-crud-retrieval"
  | "conversation-retrieval"
  | "conversation-qa"
  | "scale-sweep"
export type DatasetKind = "real-locomo" | "synthetic-generated" | "synthetic-fixture"

export interface NormalizedScope {
  readonly namespace?: string | undefined
  readonly userId?: string | undefined
  readonly agentId?: string | undefined
  readonly runId?: string | undefined
}

export interface BenchmarkConfigMapping {
  readonly common?: Record<string, unknown>
  readonly python?: Record<string, unknown>
  readonly typescript?: Record<string, unknown>
}

export interface SyntheticDatasetRef {
  readonly type: "synthetic-memory"
  readonly path: string
  readonly datasetLabel?: DatasetKind | undefined
}

export interface ConversationDatasetRef {
  readonly type: "conversation-manifest"
  readonly manifest: string
  readonly datasetLabel?: DatasetKind | undefined
  readonly required?: boolean | undefined
  readonly fallbackAllowed?: boolean | undefined
}

export type DatasetRef = SyntheticDatasetRef | ConversationDatasetRef

export interface BenchmarkWorkload {
  readonly kind: WorkloadKind
  readonly dataset: DatasetRef
  readonly useBatchAdd?: boolean
  readonly includeGetAll?: boolean
  readonly includeDeleteAll?: boolean
  readonly includeUpdate?: boolean
  readonly scaleSizes?: ReadonlyArray<number>
  readonly maxConversations?: number
  readonly maxQuestions?: number
}

export interface QaOptions {
  readonly enabled: boolean
  readonly answerModel?: string
  readonly judgeModel?: string
  readonly skipJudge?: boolean
}

export interface BenchmarkScenario {
  readonly id: string
  readonly name: string
  readonly mode: BenchmarkMode
  readonly cachePolicy: CachePolicy
  readonly workload: BenchmarkWorkload
  readonly conversationStrategy?: ConversationStrategy | undefined
  readonly config?: BenchmarkConfigMapping
  readonly topK?: number
  readonly warmupRuns?: number
  readonly measuredRuns?: number
  readonly answering?: QaOptions
  readonly judge?: QaOptions
}

export interface BenchmarkSuiteConfig {
  readonly suiteId: string
  readonly title: string
  readonly tier: BenchmarkTier
  readonly runtimeTargets: ReadonlyArray<RuntimeTarget>
  readonly warmupRuns: number
  readonly measuredRuns: number
  readonly topK: number
  readonly scopeMode: string
  readonly outputDir: string
  readonly scenarios: ReadonlyArray<BenchmarkScenario>
}

export interface SyntheticMemoryRecord {
  readonly id: string
  readonly content: string
  readonly scope: NormalizedScope
  readonly metadata?: Record<string, unknown>
}

export interface SyntheticQuery {
  readonly id: string
  readonly query: string
  readonly scope: NormalizedScope
  readonly filter?: NormalizedFilter
  readonly expectedIds: ReadonlyArray<string>
  readonly category: string
}

export interface SyntheticMutationUpdate {
  readonly id: string
  readonly scope: NormalizedScope
  readonly content: string
  readonly expectedContentIncludes?: string
}

export interface SyntheticMutationDelete {
  readonly id: string
  readonly scope: NormalizedScope
}

export interface SyntheticMutationDeleteAll {
  readonly scope: NormalizedScope
}

export interface SyntheticDataset {
  readonly datasetId: string
  readonly version: string
  readonly records: ReadonlyArray<SyntheticMemoryRecord>
  readonly queries: ReadonlyArray<SyntheticQuery>
  readonly mutations?: {
    readonly updates?: ReadonlyArray<SyntheticMutationUpdate>
    readonly deletes?: ReadonlyArray<SyntheticMutationDelete>
    readonly deleteAll?: ReadonlyArray<SyntheticMutationDeleteAll>
  }
}

export interface ConversationQuestion {
  readonly question: string
  readonly answer: string
  readonly category: string
}

export interface ConversationTurn {
  readonly role?: string
  readonly speaker?: string
  readonly content?: string
  readonly text?: string
}

export interface ConversationEntry {
  readonly id: string
  readonly speakerA: string
  readonly speakerB: string
  readonly turns: ReadonlyArray<ConversationTurn>
  readonly questions: ReadonlyArray<ConversationQuestion>
}

export interface ConversationDataset {
  readonly datasetId: string
  readonly version: string
  readonly conversations: ReadonlyArray<ConversationEntry>
}

export interface ConversationDatasetManifest {
  readonly datasetId: string
  readonly version: string
  readonly path: string
  readonly datasetLabel?: DatasetKind | undefined
  readonly generatorCommand?: ReadonlyArray<string>
}

export interface DatasetProvenance {
  readonly scenarioId: string
  readonly datasetKind: DatasetKind
  readonly datasetPath: string
  readonly datasetHash: string
  readonly datasetRequired: boolean
  readonly fallbackAllowed: boolean
  readonly fallbackUsed: boolean
  readonly manifestPath?: string | undefined
}

export interface NormalizedFilter {
  readonly field: string
  readonly op: "eq" | "contains"
  readonly value: string | number | boolean
}

export interface NormalizedMemoryResult {
  readonly id: string
  readonly content: string
  readonly rawContent?: string
  readonly scope: NormalizedScope
  readonly metadata: Record<string, unknown>
  readonly score?: number
  readonly createdAt?: string
  readonly updatedAt?: string
  readonly runtime: RuntimeTarget
  readonly mode: BenchmarkMode
}

export interface RetrievalMetricSummary {
  readonly hitAt1: number
  readonly hitAt3: number
  readonly hitAt5: number
  readonly recallAt5: number
  readonly mrr: number
  readonly ndcgAt5: number
  readonly totalQueries: number
  readonly labelAmbiguityRate: number
  readonly multiExpectedRate: number
  readonly emptyExpectedRate: number
  readonly expectedIdsPerQueryMean: number
}

export interface QaMetricSummary {
  readonly bleu1?: number | undefined
  readonly f1?: number | undefined
  readonly judgeLikert?: number | undefined
  readonly judgeBinary?: number | undefined
  readonly totalQuestions?: number | undefined
}

export interface DistributionSummary {
  readonly count: number
  readonly min: number
  readonly max: number
  readonly mean: number
  readonly p50: number
  readonly p95: number
  readonly stddev: number
}

export interface ScenarioTimings {
  readonly openMs?: DistributionSummary
  readonly addMs?: DistributionSummary
  readonly addManyMs?: DistributionSummary
  readonly updateMs?: DistributionSummary
  readonly searchMs?: DistributionSummary
  readonly getMs?: DistributionSummary
  readonly getAllMs?: DistributionSummary
  readonly deleteMs?: DistributionSummary
  readonly deleteAllMs?: DistributionSummary
}

export interface ScenarioThroughput {
  readonly addOpsPerSec?: number | undefined
  readonly searchOpsPerSec?: number | undefined
}

export interface ScenarioResourceUsage {
  readonly diskBytes?: number | undefined
  readonly peakRssBytes?: number | undefined
}

export interface RuntimeEnvironment {
  readonly runtime: RuntimeTarget
  readonly timestamp: string
  readonly gitCommit?: string | undefined
  readonly gitDirty?: boolean | undefined
  readonly os: string
  readonly osRelease: string
  readonly cpuModel: string
  readonly cpuCount: number
  readonly totalMemoryBytes: number
  readonly freeMemoryBytes: number
  readonly bunVersion?: string | undefined
  readonly pythonVersion?: string | undefined
  readonly nodeVersion?: string | undefined
  readonly datasetHash?: string | undefined
}

export interface ScenarioResult {
  readonly scenarioId: string
  readonly runtime: RuntimeTarget
  readonly tier: BenchmarkTier
  readonly mode: BenchmarkMode
  readonly status: "passed" | "failed" | "skipped"
  readonly config: Record<string, unknown>
  readonly environment: RuntimeEnvironment
  readonly timings: ScenarioTimings
  readonly throughput: ScenarioThroughput
  readonly resourceUsage: ScenarioResourceUsage
  readonly retrievalMetrics?: RetrievalMetricSummary | undefined
  readonly qaMetrics?: QaMetricSummary | undefined
  readonly parityWarnings: ReadonlyArray<string>
  readonly notes: ReadonlyArray<string>
  readonly artifacts: Record<string, string>
  readonly error?: string | undefined
}

export interface RuntimeBundle {
  readonly suiteId: string
  readonly runtime: RuntimeTarget
  readonly environment: RuntimeEnvironment
  readonly datasets?: ReadonlyArray<DatasetProvenance> | undefined
  readonly results: ReadonlyArray<ScenarioResult>
}

export interface ComparisonEntry {
  readonly scenarioId: string
  readonly python?: ScenarioResult | undefined
  readonly typescript?: ScenarioResult | undefined
  readonly notes: ReadonlyArray<string>
  readonly parityWarnings: ReadonlyArray<string>
  readonly deltas: Record<string, number>
  readonly deltaPercent: Record<string, number>
}

export interface ComparisonReport {
  readonly suiteId: string
  readonly title: string
  readonly tier: BenchmarkTier
  readonly generatedAt: string
  readonly datasets: ReadonlyArray<DatasetProvenance>
  readonly comparison: ReadonlyArray<ComparisonEntry>
}

export const resolveProjectPath = (value: string) => resolve(process.cwd(), value)

export const readJson = <T>(value: string): T => JSON.parse(readFileSync(value, "utf8")) as T

export const fileSha256 = (value: string) =>
  createHash("sha256").update(readFileSync(value)).digest("hex")

export const summarize = (samples: ReadonlyArray<number>): DistributionSummary => {
  if (samples.length === 0) {
    return {
      count: 0,
      min: 0,
      max: 0,
      mean: 0,
      p50: 0,
      p95: 0,
      stddev: 0
    }
  }
  const sorted = [...samples].sort((left, right) => left - right)
  const count = sorted.length
  const sum = sorted.reduce((total, value) => total + value, 0)
  const mean = sum / count
  const variance =
    sorted.reduce((total, value) => total + (value - mean) * (value - mean), 0) / count
  const quantile = (ratio: number) => sorted[Math.min(count - 1, Math.floor((count - 1) * ratio))] ?? 0
  return {
    count,
    min: sorted[0] ?? 0,
    max: sorted[count - 1] ?? 0,
    mean,
    p50: quantile(0.5),
    p95: quantile(0.95),
    stddev: Math.sqrt(variance)
  }
}

export const tokenize = (value: string) =>
  value
    .toLowerCase()
    .match(/\w+/g)
    ?.filter((token) => token.length > 0) ?? []

export const calculateBleu1 = (predicted: string, groundTruth: string) => {
  const predictedTokens = tokenize(predicted)
  const groundTruthTokens = tokenize(groundTruth)
  if (predictedTokens.length === 0 || groundTruthTokens.length === 0) {
    return 0
  }
  const counts = new Map<string, number>()
  for (const token of groundTruthTokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1)
  }
  let clipped = 0
  for (const token of predictedTokens) {
    const remaining = counts.get(token) ?? 0
    if (remaining > 0) {
      clipped += 1
      counts.set(token, remaining - 1)
    }
  }
  return clipped / predictedTokens.length
}

export const calculateF1 = (predicted: string, groundTruth: string) => {
  const predictedTokens = new Set(tokenize(predicted))
  const groundTruthTokens = new Set(tokenize(groundTruth))
  if (predictedTokens.size === 0 || groundTruthTokens.size === 0) {
    return 0
  }
  const common = [...predictedTokens].filter((token) => groundTruthTokens.has(token))
  if (common.length === 0) {
    return 0
  }
  const precision = common.length / predictedTokens.size
  const recall = common.length / groundTruthTokens.size
  return (2 * precision * recall) / (precision + recall)
}

const dcgAtK = (relevances: ReadonlyArray<number>, limit: number) =>
  relevances
    .slice(0, limit)
    .reduce((total, relevance, index) => total + relevance / Math.log2(index + 2), 0)

export const evaluateRetrieval = (
  rankedIds: ReadonlyArray<ReadonlyArray<string>>,
  expectedIds: ReadonlyArray<ReadonlyArray<string>>
): RetrievalMetricSummary => {
  const totalQueries = expectedIds.length
  const multiExpected = expectedIds.filter((ids) => ids.length > 1).length
  const emptyExpected = expectedIds.filter((ids) => ids.length === 0).length
  const totalExpectedIds = expectedIds.reduce((sum, ids) => sum + ids.length, 0)
  if (totalQueries === 0) {
    return {
      hitAt1: 0,
      hitAt3: 0,
      hitAt5: 0,
      recallAt5: 0,
      mrr: 0,
      ndcgAt5: 0,
      totalQueries: 0,
      labelAmbiguityRate: 0,
      multiExpectedRate: 0,
      emptyExpectedRate: 0,
      expectedIdsPerQueryMean: 0
    }
  }

  let hitAt1 = 0
  let hitAt3 = 0
  let hitAt5 = 0
  let recallAt5 = 0
  let mrr = 0
  let ndcgAt5 = 0

  for (let index = 0; index < totalQueries; index++) {
    const ranked = rankedIds[index] ?? []
    const expected = new Set(expectedIds[index] ?? [])
    const relevant: Array<number> = ranked.map((id) => (expected.has(id) ? 1 : 0))
    if (relevant[0] === 1) {
      hitAt1 += 1
    }
    if (relevant.slice(0, 3).some((value) => value === 1)) {
      hitAt3 += 1
    }
    if (relevant.slice(0, 5).some((value) => value === 1)) {
      hitAt5 += 1
    }
    const foundAt = ranked.findIndex((id) => expected.has(id))
    if (foundAt >= 0) {
      mrr += 1 / (foundAt + 1)
    }
    const hits = relevant.slice(0, 5).reduce((total, value) => total + Number(value), 0)
    recallAt5 += expected.size === 0 ? 0 : hits / expected.size
    const ideal = Array.from({ length: Math.min(5, expected.size) }, () => 1)
    const idealDcg = dcgAtK(ideal, 5)
    ndcgAt5 += idealDcg === 0 ? 0 : dcgAtK(relevant, 5) / idealDcg
  }

  return {
    hitAt1: hitAt1 / totalQueries,
    hitAt3: hitAt3 / totalQueries,
    hitAt5: hitAt5 / totalQueries,
    recallAt5: recallAt5 / totalQueries,
    mrr: mrr / totalQueries,
    ndcgAt5: ndcgAt5 / totalQueries,
    totalQueries,
    labelAmbiguityRate: multiExpected / totalQueries,
    multiExpectedRate: multiExpected / totalQueries,
    emptyExpectedRate: emptyExpected / totalQueries,
    expectedIdsPerQueryMean: totalExpectedIds / totalQueries
  }
}

export const deriveExpectedIdsFromAnswer = (
  turns: ReadonlyArray<{ readonly id: string; readonly text: string }>,
  answer: string
) => {
  const answerTokens = new Set(tokenize(answer))
  const hits = turns
    .map((turn) => ({
      id: turn.id,
      overlap: tokenize(turn.text).filter((token) => answerTokens.has(token)).length
    }))
    .filter((entry) => entry.overlap > 0)
    .sort((left, right) => right.overlap - left.overlap)
  const best = hits[0]?.overlap ?? 0
  return hits.filter((entry) => entry.overlap === best).map((entry) => entry.id)
}

export const normalizeScope = (scope: Partial<NormalizedScope> | undefined): NormalizedScope => {
  const normalized: {
    namespace?: string | undefined
    userId?: string | undefined
    agentId?: string | undefined
    runId?: string | undefined
  } = {}
  if (scope?.namespace !== undefined) {
    normalized.namespace = scope.namespace
  }
  if (scope?.userId !== undefined) {
    normalized.userId = scope.userId
  }
  if (scope?.agentId !== undefined) {
    normalized.agentId = scope.agentId
  }
  if (scope?.runId !== undefined) {
    normalized.runId = scope.runId
  }
  return normalized
}

export const scopeKey = (scope: Partial<NormalizedScope> | undefined) =>
  JSON.stringify(normalizeScope(scope))

export const matchesFilter = (
  metadata: Record<string, unknown>,
  filter: NormalizedFilter | undefined
) => {
  if (filter === undefined) {
    return true
  }
  const value = metadata[filter.field]
  if (filter.op === "eq") {
    return value === filter.value
  }
  const haystack =
    typeof value === "string" || typeof value === "number" || typeof value === "boolean"
      ? String(value)
      : JSON.stringify(value ?? null)
  return haystack.includes(String(filter.value))
}

export const approximateDiskBytes = (path: string) => {
  if (!existsSync(path)) {
    return 0
  }
  const stats = statSync(path)
  return stats.isFile() ? stats.size : 0
}

export const environmentFromProcess = (
  runtime: RuntimeTarget,
  extra: Partial<RuntimeEnvironment> = {}
): RuntimeEnvironment => ({
  runtime,
  timestamp: new Date().toISOString(),
  os: platform(),
  osRelease: release(),
  cpuModel: cpus()[0]?.model ?? "unknown",
  cpuCount: cpus().length,
  totalMemoryBytes: totalmem(),
  freeMemoryBytes: freemem(),
  nodeVersion: process.version,
  ...extra
})

export const safeSuiteScenarioName = (suiteId: string, scenarioId: string) =>
  `${basename(suiteId, ".json")}-${scenarioId}`.replace(/[^a-zA-Z0-9._-]/g, "-")
