import { mkdirSync, rmSync } from "node:fs"
import { join } from "node:path"
import { performance } from "node:perf_hooks"

import type { FilterExpr, Scope, SearchResult } from "../../packages/fraction/src/index"
import { Memory } from "../../packages/fraction/src/index"

import {
  approximateDiskBytes,
  calculateBleu1,
  calculateF1,
  deriveExpectedIdsFromAnswer,
  environmentFromProcess,
  evaluateRetrieval,
  fileSha256,
  readJson,
  resolveProjectPath,
  safeSuiteScenarioName,
  scopeKey,
  summarize,
  type BenchmarkScenario,
  type BenchmarkSuiteConfig,
  type ConversationDataset,
  type ConversationDatasetManifest,
  type ConversationTurn,
  type QaMetricSummary,
  type RuntimeBundle,
  type RuntimeEnvironment,
  type ScenarioResult,
  type SyntheticDataset,
  type SyntheticMemoryRecord,
  type SyntheticQuery
} from "./normalize"

const ANSWER_PROMPT = `You are an intelligent memory assistant. Answer questions using ONLY the provided memories.

RULES:
1. Analyze ALL memories from BOTH speakers to find the answer.
2. ALWAYS give your best answer. NEVER say "not mentioned", "not specified", "unknown", "unclear", or "no information". If you're unsure, give your best guess from the available context.
3. Convert relative time references ("last year", "two months ago") to specific dates/years using timestamps.
4. Be SPECIFIC: use exact names, numbers, dates, and places from the memories. Prefer concrete nouns over vague descriptions.
5. For "how many" questions, give a number. For "who" questions, give a name. For "when" questions, give a date.
6. For inferential questions ("what would", "what might", "is it likely"), reason from the memories and give a direct answer.
7. Your answer MUST be 10 words or fewer. No explanations — ONLY the direct answer.

Memories for user {speaker_1_user_id}:

{speaker_1_memories}

Memories for user {speaker_2_user_id}:

{speaker_2_memories}

Question: {question}

Answer (10 words max, direct answer only):`

const LLM_JUDGE_PROMPT = `You are evaluating the quality of an AI-generated answer compared to a ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {predicted}

Rate the quality of the generated answer on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially relevant but mostly incorrect
3 = Somewhat correct but missing key details
4 = Mostly correct with minor issues
5 = Perfectly correct and complete

Respond with ONLY a single integer (1-5), nothing else.`

const BINARY_JUDGE_PROMPT = `You are evaluating the quality of an AI-generated answer compared to a ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {predicted}

Does the generated answer correctly capture the key information from the ground truth?
Respond with ONLY 0 (incorrect) or 1 (correct), nothing else.`

interface ParsedArgs {
  readonly suite: string
  readonly output: string
}

interface OperationTimings {
  readonly openMs: Array<number>
  readonly addMs: Array<number>
  readonly addManyMs: Array<number>
  readonly updateMs: Array<number>
  readonly searchMs: Array<number>
  readonly getMs: Array<number>
  readonly getAllMs: Array<number>
  readonly deleteMs: Array<number>
  readonly deleteAllMs: Array<number>
}

const emptyOperationTimings = (): OperationTimings => ({
  openMs: [],
  addMs: [],
  addManyMs: [],
  updateMs: [],
  searchMs: [],
  getMs: [],
  getAllMs: [],
  deleteMs: [],
  deleteAllMs: []
})

const parseArgs = (): ParsedArgs => {
  const args = process.argv.slice(2)
  const get = (flag: string) => {
    const index = args.indexOf(flag)
    return index >= 0 ? args[index + 1] : undefined
  }
  const suite = get("--suite")
  const output = get("--output")
  if (!suite || !output) {
    throw new Error("Usage: bun benchmarks/parity/ts_runner.ts --suite <path> --output <path>")
  }
  return {
    suite: resolveProjectPath(suite),
    output: resolveProjectPath(output)
  }
}

const commandOutput = (command: ReadonlyArray<string>) => {
  const [cmd, ...args] = command
  if (!cmd) {
    return undefined
  }
  const result = Bun.spawnSync([cmd, ...args], {
    cwd: process.cwd(),
    stderr: "ignore",
    stdout: "pipe"
  })
  if (result.exitCode !== 0) {
    return undefined
  }
  return new TextDecoder().decode(result.stdout).trim()
}

const buildEnvironment = (datasetHash?: string): RuntimeEnvironment =>
  environmentFromProcess("typescript", {
    bunVersion: Bun.version,
    gitCommit: commandOutput(["git", "rev-parse", "HEAD"]),
    gitDirty: (commandOutput(["git", "status", "--porcelain"]) ?? "").length > 0,
    datasetHash
  })

const scenarioRuns = (suite: BenchmarkSuiteConfig, scenario: BenchmarkScenario) => ({
  warmup: scenario.warmupRuns ?? suite.warmupRuns,
  measured: scenario.measuredRuns ?? suite.measuredRuns
})

const scopeToTs = (scope: Partial<Scope> | undefined): Scope => ({
  namespace: scope?.namespace,
  userId: scope?.userId,
  agentId: scope?.agentId,
  runId: scope?.runId
})

const filterToTs = (filter: SyntheticQuery["filter"]): FilterExpr | undefined =>
  filter === undefined ? undefined : { field: filter.field, op: filter.op, value: filter.value }

const uniqueScopes = (records: ReadonlyArray<SyntheticMemoryRecord>) => {
  const scopes = new Map<string, Scope>()
  for (const record of records) {
    const scope = scopeToTs(record.scope)
    scopes.set(scopeKey(scope), scope)
  }
  return [...scopes.values()]
}

const createRuntimeConfig = (
  scenario: BenchmarkScenario,
  defaultNamespace: string,
  filename: string
) => {
  const common = scenario.config?.common ?? {}
  const runtime = scenario.config?.typescript ?? {}
  const compressorType =
    typeof runtime.compressorType === "string"
      ? (runtime.compressorType as "heuristic" | "llmlingua2" | "off")
      : undefined
  return {
    filename,
    defaultNamespace,
    ...(common.topK !== undefined ? { topK: Number(common.topK) } : {}),
    ...(common.compressionRate !== undefined
      ? { compressionRate: Number(common.compressionRate) }
      : {}),
    ...(compressorType !== undefined ? { compressorType } : {})
  }
}

const tempFilename = (suiteId: string, scenarioId: string, runIndex: number) =>
  join(
    process.cwd(),
    ".benchmark-parity",
    "ts",
    safeSuiteScenarioName(suiteId, scenarioId),
    `run-${runIndex}.sqlite`
  )

const cleanupSqlite = (filename: string) => {
  rmSync(filename, { force: true })
  rmSync(`${filename}-shm`, { force: true })
  rmSync(`${filename}-wal`, { force: true })
}

const loadSyntheticDataset = (path: string) => readJson<SyntheticDataset>(resolveProjectPath(path))

const normalizeConversationDataset = (
  manifest: ConversationDatasetManifest
): ConversationDataset => {
  const source = readJson<Record<string, { conversation: ReadonlyArray<{ role?: string; speaker?: string; content?: string; text?: string }>; questions?: ReadonlyArray<{ question: string; answer: string; category: string }> }>>(
    resolveProjectPath(manifest.path)
  )
  const conversations = Object.entries(source).map(([id, value]) => ({
    id,
    speakerA:
      value.conversation.find((turn) => (turn.role ?? turn.speaker ?? "").length > 0)?.role ??
      value.conversation.find((turn) => (turn.role ?? turn.speaker ?? "").length > 0)?.speaker ??
      "speaker_a",
    speakerB:
      value.conversation
        .map((turn) => turn.role ?? turn.speaker ?? "")
        .find((speaker, index, speakers) => speaker.length > 0 && speaker !== speakers[0]) ??
      "speaker_b",
    turns: value.conversation,
    questions: value.questions ?? []
  }))
  return {
    datasetId: manifest.datasetId,
    version: manifest.version,
    conversations
  }
}

const loadConversationDataset = (manifestPath: string) => {
  const manifest = readJson<ConversationDatasetManifest>(resolveProjectPath(manifestPath))
  return normalizeConversationDataset(manifest)
}

const searchIds = (results: ReadonlyArray<SearchResult>) =>
  results.map((result) => result.memory.id)

const mapExpectedIds = (
  expectedIds: ReadonlyArray<string>,
  fixtureToRuntimeIds: ReadonlyMap<string, string>
) => expectedIds.map((id) => fixtureToRuntimeIds.get(id) ?? id)

const buildSearchOptions = (limit: number, filter?: SyntheticQuery["filter"]) =>
  filter === undefined ? { limit } : { limit, filter: filterToTs(filter)! }

const addRecords = async (
  memory: Awaited<ReturnType<typeof Memory.open>>,
  records: ReadonlyArray<SyntheticMemoryRecord>,
  timings: OperationTimings,
  useBatchAdd: boolean | undefined
) => {
  const fixtureToRuntimeIds = new Map<string, string>()
  if (useBatchAdd) {
    const scopedGroups = new Map<string, { scope: Scope; values: Array<string> }>()
    for (const record of records) {
      const scope = scopeToTs(record.scope)
      const key = scopeKey(scope)
      const current = scopedGroups.get(key) ?? { scope, values: [] }
      current.values.push(record.content)
      scopedGroups.set(key, current)
    }
    for (const group of scopedGroups.values()) {
      const started = performance.now()
      await memory.addMany(group.values, group.scope)
      timings.addManyMs.push(performance.now() - started)
    }
    return fixtureToRuntimeIds
  }

  for (const record of records) {
    const started = performance.now()
    const created = await memory.add(record.content, scopeToTs(record.scope), record.metadata)
    timings.addMs.push(performance.now() - started)
    fixtureToRuntimeIds.set(record.id, created.id)
  }

  return fixtureToRuntimeIds
}

const openMemory = async (suite: BenchmarkSuiteConfig, scenario: BenchmarkScenario, runIndex: number) => {
  const filename = tempFilename(suite.suiteId, scenario.id, runIndex)
  mkdirSync(join(filename, ".."), { recursive: true })
  cleanupSqlite(filename)
  const openStarted = performance.now()
  const memory = await Memory.open(createRuntimeConfig(scenario, "parity", filename))
  return { memory, filename, openMs: performance.now() - openStarted }
}

const runSyntheticScenario = async (
  suite: BenchmarkSuiteConfig,
  scenario: BenchmarkScenario,
  environment: RuntimeEnvironment
): Promise<ScenarioResult> => {
  if (scenario.workload.dataset.type !== "synthetic-memory") {
    throw new Error("Synthetic scenario requires a synthetic-memory dataset")
  }
  const dataset = loadSyntheticDataset(scenario.workload.dataset.path)
  const runs = scenarioRuns(suite, scenario)
  const timings = emptyOperationTimings()
  const rankedIds: Array<ReadonlyArray<string>> = []
  const expectedIds: Array<ReadonlyArray<string>> = []
  const notes: Array<string> = []
  const parityWarnings =
    scenario.mode === "best-effort-local" && scenario.config?.typescript?.compressorType !== "off"
      ? ["TypeScript strict suite uses runtime-native local compression configuration."]
      : []
  let diskBytes = 0

  for (let warm = 0; warm < runs.warmup; warm++) {
    const { memory, filename } = await openMemory(suite, scenario, -1 - warm)
    try {
      await addRecords(memory, dataset.records, emptyOperationTimings(), scenario.workload.useBatchAdd)
      for (const query of dataset.queries) {
        await memory.search(
          query.query,
          scopeToTs(query.scope),
          buildSearchOptions(scenario.topK ?? suite.topK, query.filter)
        )
      }
    } finally {
      await memory.close()
      cleanupSqlite(filename)
    }
  }

  for (let runIndex = 0; runIndex < runs.measured; runIndex++) {
    const { memory, filename, openMs } = await openMemory(suite, scenario, runIndex)
    timings.openMs.push(openMs)
    try {
      const fixtureToRuntimeIds = await addRecords(
        memory,
        dataset.records,
        timings,
        scenario.workload.useBatchAdd
      )

      for (const query of dataset.queries) {
        const started = performance.now()
        const results = await memory.search(
          query.query,
          scopeToTs(query.scope),
          buildSearchOptions(scenario.topK ?? suite.topK, query.filter)
        )
        timings.searchMs.push(performance.now() - started)
        rankedIds.push(searchIds(results))
        expectedIds.push(mapExpectedIds(query.expectedIds, fixtureToRuntimeIds))
      }

      for (const record of dataset.records.slice(0, 3)) {
        const started = performance.now()
        await memory.get(fixtureToRuntimeIds.get(record.id) ?? record.id).catch(() => undefined)
        timings.getMs.push(performance.now() - started)
      }

      if (scenario.workload.includeGetAll) {
        for (const scope of uniqueScopes(dataset.records)) {
          const started = performance.now()
          await memory.getAll(scope)
          timings.getAllMs.push(performance.now() - started)
        }
      }

      if (scenario.workload.includeUpdate) {
        for (const mutation of dataset.mutations?.updates ?? []) {
          const started = performance.now()
          const updated = await memory.update(
            fixtureToRuntimeIds.get(mutation.id) ?? mutation.id,
            mutation.content
          )
          timings.updateMs.push(performance.now() - started)
          if (
            mutation.expectedContentIncludes &&
            !updated.content.includes(mutation.expectedContentIncludes)
          ) {
            notes.push(`Update expectation failed for ${mutation.id}`)
          }
        }
      }

      for (const mutation of dataset.mutations?.deletes ?? []) {
        const started = performance.now()
        await memory.delete(fixtureToRuntimeIds.get(mutation.id) ?? mutation.id)
        timings.deleteMs.push(performance.now() - started)
      }

      if (scenario.workload.includeDeleteAll) {
        for (const mutation of dataset.mutations?.deleteAll ?? []) {
          const started = performance.now()
          await memory.deleteAll(scopeToTs(mutation.scope))
          timings.deleteAllMs.push(performance.now() - started)
        }
      }

      diskBytes = Math.max(
        diskBytes,
        approximateDiskBytes(filename) +
          approximateDiskBytes(`${filename}-wal`) +
          approximateDiskBytes(`${filename}-shm`)
      )
    } finally {
      await memory.close()
      cleanupSqlite(filename)
    }
  }

  return {
    scenarioId: scenario.id,
    runtime: "typescript",
    tier: suite.tier,
    mode: scenario.mode,
    status: "passed",
    config: {
      ...scenario.config?.common,
      ...scenario.config?.typescript
    },
    environment,
    timings: {
      openMs: summarize(timings.openMs),
      addMs: summarize(timings.addMs),
      addManyMs: summarize(timings.addManyMs),
      updateMs: summarize(timings.updateMs),
      searchMs: summarize(timings.searchMs),
      getMs: summarize(timings.getMs),
      getAllMs: summarize(timings.getAllMs),
      deleteMs: summarize(timings.deleteMs),
      deleteAllMs: summarize(timings.deleteAllMs)
    },
    throughput: {
      addOpsPerSec:
        timings.addMs.length === 0
          ? undefined
          : timings.addMs.length / (timings.addMs.reduce((sum, value) => sum + value, 0) / 1000),
      searchOpsPerSec:
        timings.searchMs.length === 0
          ? undefined
          : timings.searchMs.length /
            (timings.searchMs.reduce((sum, value) => sum + value, 0) / 1000)
    },
    resourceUsage: {
      diskBytes,
      peakRssBytes: process.memoryUsage().rss
    },
    retrievalMetrics: evaluateRetrieval(rankedIds, expectedIds),
    parityWarnings,
    notes,
    artifacts: {}
  }
}

const turnText = (turn: ConversationTurn) => turn.text ?? turn.content ?? ""
const turnSpeaker = (turn: ConversationTurn) => turn.speaker ?? turn.role ?? "speaker"

const groupMemoriesBySpeaker = (
  results: ReadonlyArray<SearchResult>,
  speakerA: string,
  speakerB: string
) => {
  const speakerAMemories = results
    .filter((result) => result.memory.scope.userId === speakerA)
    .map((result) => `- ${result.memory.rawContent || result.memory.content}`)
    .join("\n") || "(no memories found)"
  const speakerBMemories = results
    .filter((result) => result.memory.scope.userId === speakerB)
    .map((result) => `- ${result.memory.rawContent || result.memory.content}`)
    .join("\n") || "(no memories found)"
  return { speakerAMemories, speakerBMemories }
}

const openAiJson = async (
  model: string,
  prompt: string,
  maxTokens: number
) => {
  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required for QA scenarios")
  }
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      messages: [{ role: "user", content: prompt }]
    })
  })
  if (!response.ok) {
    throw new Error(`OpenAI request failed with ${response.status}`)
  }
  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>
  }
  return data.choices?.[0]?.message?.content?.trim() ?? ""
}

const qaMetricsFromPairs = (
  rows: ReadonlyArray<{ answer: string; prediction: string; question: string; judgeLikert?: number; judgeBinary?: number }>
): QaMetricSummary => {
  if (rows.length === 0) {
    return { totalQuestions: 0 }
  }
  const bleu = rows.reduce((sum, row) => sum + calculateBleu1(row.prediction, row.answer), 0) / rows.length
  const f1 = rows.reduce((sum, row) => sum + calculateF1(row.prediction, row.answer), 0) / rows.length
  const likertRows = rows.filter((row) => row.judgeLikert !== undefined)
  const binaryRows = rows.filter((row) => row.judgeBinary !== undefined)
  return {
    bleu1: bleu,
    f1,
    judgeLikert:
      likertRows.length === 0
        ? undefined
        : likertRows.reduce((sum, row) => sum + (row.judgeLikert ?? 0), 0) / likertRows.length,
    judgeBinary:
      binaryRows.length === 0
        ? undefined
        : binaryRows.reduce((sum, row) => sum + (row.judgeBinary ?? 0), 0) / binaryRows.length,
    totalQuestions: rows.length
  }
}

const judgeLikert = async (question: string, answer: string, prediction: string, model: string) => {
  const text = await openAiJson(
    model,
    LLM_JUDGE_PROMPT.replace("{question}", question)
      .replace("{ground_truth}", answer)
      .replace("{predicted}", prediction),
    10
  )
  const match = text.match(/[1-5]/)
  return match ? Number(match[0]) : 3
}

const judgeBinary = async (question: string, answer: string, prediction: string, model: string) => {
  const text = await openAiJson(
    model,
    BINARY_JUDGE_PROMPT.replace("{question}", question)
      .replace("{ground_truth}", answer)
      .replace("{predicted}", prediction),
    10
  )
  const match = text.match(/[01]/)
  return match ? Number(match[0]) : 0.5
}

const runConversationScenario = async (
  suite: BenchmarkSuiteConfig,
  scenario: BenchmarkScenario,
  environment: RuntimeEnvironment,
  withQa: boolean
): Promise<ScenarioResult> => {
  if (scenario.workload.dataset.type !== "conversation-manifest") {
    throw new Error("Conversation scenario requires a conversation-manifest dataset")
  }
  const dataset = loadConversationDataset(scenario.workload.dataset.manifest)
  const runs = scenarioRuns(suite, scenario)
  const timings = emptyOperationTimings()
  const rankedIds: Array<ReadonlyArray<string>> = []
  const expectedIds: Array<ReadonlyArray<string>> = []
  const qaRows: Array<{ answer: string; prediction: string; question: string; judgeLikert?: number; judgeBinary?: number }> = []
  const notes: Array<string> = []
  const conversations = dataset.conversations.slice(0, scenario.workload.maxConversations ?? dataset.conversations.length)
  let diskBytes = 0

  const runOnce = async (runIndex: number, recordMetrics: boolean) => {
    const { memory, filename, openMs } = await openMemory(suite, scenario, runIndex)
    if (recordMetrics) {
      timings.openMs.push(openMs)
    }
    try {
      for (const conversation of conversations) {
        const turnFixtureToRuntimeIds = new Map<string, string>()
        for (let turnIndex = 0; turnIndex < conversation.turns.length; turnIndex++) {
          const turn = conversation.turns[turnIndex]
          if (!turn) {
            continue
          }
          const text = turnText(turn)
          if (text.trim().length === 0) {
            continue
          }
          const started = performance.now()
          const created = await memory.add(text, {
            namespace: conversation.id,
            userId: turnSpeaker(turn)
          })
          turnFixtureToRuntimeIds.set(`${conversation.id}-turn-${turnIndex}`, created.id)
          if (recordMetrics) {
            timings.addMs.push(performance.now() - started)
          }
        }

        const turnRecords = conversation.turns.map((turn, index) => ({
          id: `${conversation.id}-turn-${index}`,
          text: turnText(turn)
        }))

        for (const question of conversation.questions.slice(
          0,
          scenario.workload.maxQuestions ?? conversation.questions.length
        )) {
          const started = performance.now()
          const results = await memory.search(question.question, { namespace: conversation.id }, {
            limit: scenario.topK ?? suite.topK
          })
          if (recordMetrics) {
            timings.searchMs.push(performance.now() - started)
            rankedIds.push(searchIds(results))
            expectedIds.push(
              mapExpectedIds(deriveExpectedIdsFromAnswer(turnRecords, question.answer), turnFixtureToRuntimeIds)
            )
          }

          if (recordMetrics && withQa && scenario.answering?.enabled) {
            const grouped = groupMemoriesBySpeaker(results, conversation.speakerA, conversation.speakerB)
            const prompt = ANSWER_PROMPT
              .replaceAll("{speaker_1_user_id}", conversation.speakerA)
              .replaceAll("{speaker_1_memories}", grouped.speakerAMemories)
              .replaceAll("{speaker_2_user_id}", conversation.speakerB)
              .replaceAll("{speaker_2_memories}", grouped.speakerBMemories)
              .replaceAll("{question}", question.question)
            const prediction = await openAiJson(
              scenario.answering.answerModel ?? "gpt-4o",
              prompt,
              50
            )
            const row: {
              answer: string
              prediction: string
              question: string
              judgeLikert?: number
              judgeBinary?: number
            } = {
              answer: question.answer,
              prediction,
              question: question.question
            }
            if (!scenario.answering.skipJudge) {
              row.judgeLikert = await judgeLikert(
                question.question,
                question.answer,
                prediction,
                scenario.answering.judgeModel ?? "gpt-4o-mini"
              )
              row.judgeBinary = await judgeBinary(
                question.question,
                question.answer,
                prediction,
                scenario.answering.judgeModel ?? "gpt-4o-mini"
              )
            }
            qaRows.push(row)
          }
        }
      }

      diskBytes = Math.max(
        diskBytes,
        approximateDiskBytes(filename) +
          approximateDiskBytes(`${filename}-wal`) +
          approximateDiskBytes(`${filename}-shm`)
      )
    } finally {
      await memory.close()
      cleanupSqlite(filename)
    }
  }

  for (let warm = 0; warm < runs.warmup; warm++) {
    await runOnce(-1 - warm, false)
  }
  for (let runIndex = 0; runIndex < runs.measured; runIndex++) {
    await runOnce(runIndex, true)
  }

  if (scenario.mode === "best-effort-local") {
    notes.push("Conversation parity uses runtime-native local implementations with shared datasets and prompts.")
  }

  return {
    scenarioId: scenario.id,
    runtime: "typescript",
    tier: suite.tier,
    mode: scenario.mode,
    status: "passed",
    config: {
      ...scenario.config?.common,
      ...scenario.config?.typescript
    },
    environment,
    timings: {
      openMs: summarize(timings.openMs),
      addMs: summarize(timings.addMs),
      searchMs: summarize(timings.searchMs)
    },
    throughput: {
      addOpsPerSec:
        timings.addMs.length === 0
          ? undefined
          : timings.addMs.length / (timings.addMs.reduce((sum, value) => sum + value, 0) / 1000),
      searchOpsPerSec:
        timings.searchMs.length === 0
          ? undefined
          : timings.searchMs.length /
            (timings.searchMs.reduce((sum, value) => sum + value, 0) / 1000)
    },
    resourceUsage: {
      diskBytes,
      peakRssBytes: process.memoryUsage().rss
    },
    retrievalMetrics: evaluateRetrieval(rankedIds, expectedIds),
    qaMetrics: withQa ? qaMetricsFromPairs(qaRows) : undefined,
    parityWarnings:
      scenario.mode === "best-effort-local"
        ? ["Conversation parity is best-effort local because Py and TS do not share identical local model artifacts."]
        : [],
    notes,
    artifacts: {}
  }
}

const runScaleSweepScenario = async (
  suite: BenchmarkSuiteConfig,
  scenario: BenchmarkScenario,
  environment: RuntimeEnvironment
): Promise<ScenarioResult> => {
  if (scenario.workload.dataset.type !== "synthetic-memory") {
    throw new Error("Scale sweep requires a synthetic-memory dataset")
  }
  const dataset = loadSyntheticDataset(scenario.workload.dataset.path)
  const sizes = scenario.workload.scaleSizes ?? [10, 100, 1000]
  const timings = emptyOperationTimings()
  const notes: Array<string> = []
  let diskBytes = 0

  for (const [sizeIndex, size] of sizes.entries()) {
    const { memory, filename, openMs } = await openMemory(suite, scenario, sizeIndex)
    timings.openMs.push(openMs)
    try {
      const records: Array<SyntheticMemoryRecord> = []
      for (let index = 0; index < size; index++) {
        const base = dataset.records[index % dataset.records.length]
        if (!base) {
          continue
        }
        records.push({
          ...base,
          id: `${base.id}-copy-${index}`,
          content: `${base.content} [copy=${index}]`
        })
      }
      await addRecords(memory, records, timings, false)
      const probeQueries = dataset.queries.slice(0, 3)
      for (const query of probeQueries) {
        const started = performance.now()
        await memory.search(query.query, scopeToTs(query.scope), { limit: scenario.topK ?? suite.topK })
        timings.searchMs.push(performance.now() - started)
      }
      notes.push(`Measured scale size ${size}`)
      diskBytes = Math.max(
        diskBytes,
        approximateDiskBytes(filename) +
          approximateDiskBytes(`${filename}-wal`) +
          approximateDiskBytes(`${filename}-shm`)
      )
    } finally {
      await memory.close()
      cleanupSqlite(filename)
    }
  }

  return {
    scenarioId: scenario.id,
    runtime: "typescript",
    tier: suite.tier,
    mode: scenario.mode,
    status: "passed",
    config: {
      ...scenario.config?.common,
      ...scenario.config?.typescript
    },
    environment,
    timings: {
      openMs: summarize(timings.openMs),
      addMs: summarize(timings.addMs),
      searchMs: summarize(timings.searchMs)
    },
    throughput: {
      addOpsPerSec:
        timings.addMs.length === 0
          ? undefined
          : timings.addMs.length / (timings.addMs.reduce((sum, value) => sum + value, 0) / 1000),
      searchOpsPerSec:
        timings.searchMs.length === 0
          ? undefined
          : timings.searchMs.length /
            (timings.searchMs.reduce((sum, value) => sum + value, 0) / 1000)
    },
    resourceUsage: {
      diskBytes,
      peakRssBytes: process.memoryUsage().rss
    },
    parityWarnings:
      scenario.mode === "best-effort-local"
        ? ["Scale sweep compares runtime-native local stacks with shared workload sizes."]
        : [],
    notes,
    artifacts: {}
  }
}

const runScenario = async (
  suite: BenchmarkSuiteConfig,
  scenario: BenchmarkScenario,
  environment: RuntimeEnvironment
): Promise<ScenarioResult> => {
  try {
    switch (scenario.workload.kind) {
      case "synthetic-crud-retrieval":
        return await runSyntheticScenario(suite, scenario, environment)
      case "conversation-retrieval":
        return await runConversationScenario(suite, scenario, environment, false)
      case "conversation-qa":
        return await runConversationScenario(suite, scenario, environment, true)
      case "scale-sweep":
        return await runScaleSweepScenario(suite, scenario, environment)
    }
  } catch (error) {
    return {
      scenarioId: scenario.id,
      runtime: "typescript",
      tier: suite.tier,
      mode: scenario.mode,
      status: "failed",
      config: {
        ...scenario.config?.common,
        ...scenario.config?.typescript
      },
      environment,
      timings: {},
      throughput: {},
      resourceUsage: {},
      parityWarnings: [],
      notes: [],
      artifacts: {},
      error: error instanceof Error ? error.message : String(error)
    }
  }
}

const main = async () => {
  const args = parseArgs()
  const suite = readJson<BenchmarkSuiteConfig>(args.suite)
  const datasetHashes: Array<string> = []
  for (const scenario of suite.scenarios) {
    if (scenario.workload.dataset.type === "synthetic-memory") {
      datasetHashes.push(fileSha256(resolveProjectPath(scenario.workload.dataset.path)))
    } else {
      const manifest = readJson<ConversationDatasetManifest>(
        resolveProjectPath(scenario.workload.dataset.manifest)
      )
      datasetHashes.push(fileSha256(resolveProjectPath(manifest.path)))
    }
  }
  const environment = buildEnvironment(datasetHashes.sort().join(","))
  const results: Array<ScenarioResult> = []
  for (const scenario of suite.scenarios) {
    results.push(await runScenario(suite, scenario, environment))
  }

  mkdirSync(join(args.output, ".."), { recursive: true })
  const bundle: RuntimeBundle = {
    suiteId: suite.suiteId,
    runtime: "typescript",
    environment,
    results
  }
  await Bun.write(args.output, JSON.stringify(bundle, null, 2))
}

await main()
