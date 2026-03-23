import { existsSync, mkdirSync } from "node:fs"
import { join } from "node:path"

import { renderMarkdownReport } from "./report"
import {
  fileSha256,
  readJson,
  resolveProjectPath,
  type BenchmarkSuiteConfig,
  type ComparisonEntry,
  type ComparisonReport,
  type ConversationDatasetManifest,
  type DatasetKind,
  type DatasetProvenance,
  type RuntimeBundle,
  type RuntimeTarget,
  type ScenarioResult
} from "./normalize"

interface ParsedArgs {
  readonly suite: string
}

const pythonInvoker = (): Array<string> => {
  if (process.env.FRACTION_PARITY_USE_SYSTEM_PYTHON === "1") {
    return ["python3"]
  }
  return Bun.which("uv") ? ["uv", "run", "python"] : ["python3"]
}

const parseArgs = (): ParsedArgs => {
  const args = process.argv.slice(2)
  const index = args.indexOf("--suite")
  const suite = index >= 0 ? args[index + 1] : undefined
  if (!suite) {
    throw new Error("Usage: bun benchmarks/parity/orchestrate.ts --suite <path>")
  }
  return { suite: resolveProjectPath(suite) }
}

const datasetKindForManifest = (
  label: DatasetKind | undefined,
  manifest: ConversationDatasetManifest
): DatasetKind => label ?? manifest.datasetLabel ?? "real-locomo"

const canonicalDatasetError = (path: string) =>
  `Canonical parity suite requires the real LoCoMo dataset at ${path}; synthetic fallback is not allowed.`

const uniqueDatasets = (datasets: ReadonlyArray<DatasetProvenance>) => {
  const keyed = new Map<string, DatasetProvenance>()
  for (const dataset of datasets) {
    const key = JSON.stringify([
      dataset.scenarioId,
      dataset.datasetKind,
      dataset.datasetPath,
      dataset.datasetHash,
      dataset.datasetRequired,
      dataset.fallbackAllowed,
      dataset.fallbackUsed,
      dataset.manifestPath ?? ""
    ])
    keyed.set(key, dataset)
  }
  return [...keyed.values()]
}

const datasetHashSummary = (datasets: ReadonlyArray<DatasetProvenance>) =>
  uniqueDatasets(datasets)
    .map((dataset) => dataset.datasetHash)
    .sort()
    .join(",")

const collectDatasetProvenance = (suite: BenchmarkSuiteConfig): Array<DatasetProvenance> => {
  const datasets: Array<DatasetProvenance> = []
  for (const scenario of suite.scenarios) {
    const dataset = scenario.workload.dataset
    if (dataset.type === "synthetic-memory") {
      const resolvedPath = resolveProjectPath(dataset.path)
      if (!existsSync(resolvedPath)) {
        throw new Error(`Synthetic benchmark dataset is missing: ${dataset.path}`)
      }
      datasets.push({
        scenarioId: scenario.id,
        datasetKind: dataset.datasetLabel ?? "synthetic-fixture",
        datasetPath: dataset.path,
        datasetHash: fileSha256(resolvedPath),
        datasetRequired: true,
        fallbackAllowed: false,
        fallbackUsed: false
      })
      continue
    }

    const manifest = readJson<ConversationDatasetManifest>(resolveProjectPath(dataset.manifest))
    const resolvedPath = resolveProjectPath(manifest.path)
    if (!existsSync(resolvedPath)) {
      throw new Error(canonicalDatasetError(manifest.path))
    }
    datasets.push({
      scenarioId: scenario.id,
      datasetKind: datasetKindForManifest(dataset.datasetLabel, manifest),
      datasetPath: manifest.path,
      datasetHash: fileSha256(resolvedPath),
      datasetRequired: dataset.required ?? true,
      fallbackAllowed: false,
      fallbackUsed: false,
      manifestPath: dataset.manifest
    })
  }
  return datasets
}

const spawnRuntime = (runtime: RuntimeTarget, suitePath: string, outputPath: string) => {
  const command =
    runtime === "typescript"
      ? ["bun", "benchmarks/parity/ts_runner.ts", "--suite", suitePath, "--output", outputPath]
      : [...pythonInvoker(), "benchmarks/parity/python_runner.py", "--suite", suitePath, "--output", outputPath]
  const cmd = command[0]!
  const rest = command.slice(1)
  const result = Bun.spawnSync([cmd, ...rest], {
    cwd: process.cwd(),
    stdout: "inherit",
    stderr: "inherit"
  })
  if (result.exitCode !== 0) {
    throw new Error(`${runtime} runner failed with exit code ${result.exitCode}`)
  }
}

const enrichBundle = (
  bundle: RuntimeBundle,
  datasets: ReadonlyArray<DatasetProvenance>
): RuntimeBundle => ({
  ...bundle,
  datasets: uniqueDatasets(datasets)
})

const mapByScenario = (results: ReadonlyArray<ScenarioResult>) => {
  const map = new Map<string, ScenarioResult>()
  for (const result of results) {
    map.set(result.scenarioId, result)
  }
  return map
}

const metricDelta = (left: number | undefined, right: number | undefined) =>
  left === undefined || right === undefined ? undefined : right - left

const metricPercent = (left: number | undefined, right: number | undefined) =>
  left === undefined || right === undefined || left === 0 ? undefined : ((right - left) / left) * 100

const buildComparison = (
  suite: BenchmarkSuiteConfig,
  bundles: ReadonlyArray<RuntimeBundle>,
  datasets: ReadonlyArray<DatasetProvenance>
): ComparisonReport => {
  const uniqueDatasetList = uniqueDatasets(datasets)
  const python = bundles.find((bundle) => bundle.runtime === "python")
  const typescript = bundles.find((bundle) => bundle.runtime === "typescript")
  const pyByScenario = mapByScenario(python?.results ?? [])
  const tsByScenario = mapByScenario(typescript?.results ?? [])

  const comparison: Array<ComparisonEntry> = suite.scenarios.map((scenario) => {
    const py = pyByScenario.get(scenario.id)
    const ts = tsByScenario.get(scenario.id)
    const datasetWarnings = uniqueDatasetList
      .filter((dataset) => dataset.scenarioId === scenario.id && dataset.datasetKind !== "real-locomo")
      .map(
        (dataset) =>
          `Scenario ${scenario.id} ran on ${dataset.datasetKind} instead of the canonical LoCoMo dataset.`
      )
    return {
      scenarioId: scenario.id,
      python: py,
      typescript: ts,
      notes: [...(py?.notes ?? []), ...(ts?.notes ?? [])],
      parityWarnings: [...(py?.parityWarnings ?? []), ...(ts?.parityWarnings ?? []), ...datasetWarnings],
      deltas: {
        addP50Ms: metricDelta(py?.timings.addMs?.p50, ts?.timings.addMs?.p50) ?? 0,
        searchP50Ms: metricDelta(py?.timings.searchMs?.p50, ts?.timings.searchMs?.p50) ?? 0,
        hitAt5: metricDelta(py?.retrievalMetrics?.hitAt5, ts?.retrievalMetrics?.hitAt5) ?? 0,
        mrr: metricDelta(py?.retrievalMetrics?.mrr, ts?.retrievalMetrics?.mrr) ?? 0
      },
      deltaPercent: {
        addP50Ms: metricPercent(py?.timings.addMs?.p50, ts?.timings.addMs?.p50) ?? 0,
        searchP50Ms: metricPercent(py?.timings.searchMs?.p50, ts?.timings.searchMs?.p50) ?? 0,
        hitAt5: metricPercent(py?.retrievalMetrics?.hitAt5, ts?.retrievalMetrics?.hitAt5) ?? 0,
        mrr: metricPercent(py?.retrievalMetrics?.mrr, ts?.retrievalMetrics?.mrr) ?? 0
      }
    }
  })

  return {
    suiteId: suite.suiteId,
    title: suite.title,
    tier: suite.tier,
    generatedAt: new Date().toISOString(),
    datasets: uniqueDatasetList,
    comparison
  }
}

const main = async () => {
  const { suite: suitePath } = parseArgs()
  const suite = readJson<BenchmarkSuiteConfig>(suitePath)
  const datasets = collectDatasetProvenance(suite)

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
  const reportRoot = resolveProjectPath(join(suite.outputDir, suite.suiteId, timestamp))
  const rawDir = join(reportRoot, "raw")
  mkdirSync(rawDir, { recursive: true })

  const bundles: Array<RuntimeBundle> = []
  for (const runtime of suite.runtimeTargets) {
    const outputPath = join(rawDir, `${runtime}.json`)
    spawnRuntime(runtime, suitePath, outputPath)
    const bundle = enrichBundle(readJson<RuntimeBundle>(outputPath), datasets)
    await Bun.write(outputPath, JSON.stringify(bundle, null, 2))
    bundles.push(bundle)
  }

  const comparison = buildComparison(suite, bundles, datasets)
  const comparisonPath = join(reportRoot, "comparison.json")
  const summaryPath = join(reportRoot, "summary.md")
  const metadataPath = join(reportRoot, "metadata.json")

  await Bun.write(comparisonPath, JSON.stringify(comparison, null, 2))
  await Bun.write(summaryPath, renderMarkdownReport(comparison))
  await Bun.write(
    metadataPath,
    JSON.stringify(
      {
        suiteId: suite.suiteId,
        title: suite.title,
        tier: suite.tier,
        generatedAt: comparison.generatedAt,
        datasetHash: datasetHashSummary(datasets),
        datasets: uniqueDatasets(datasets),
        rawArtifacts: suite.runtimeTargets.map((runtime) => join(rawDir, `${runtime}.json`))
      },
      null,
      2
    )
  )

  console.log(JSON.stringify({ reportRoot, comparisonPath, summaryPath }, null, 2))
}

await main()
