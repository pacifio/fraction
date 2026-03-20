import type { ComparisonEntry, ComparisonReport, DatasetProvenance, ScenarioResult } from "./normalize"

const metricValue = (result: ScenarioResult | undefined, path: "addP50" | "searchP50" | "hitAt5" | "mrr" | "bleu1") => {
  if (!result || result.status !== "passed") {
    return undefined
  }
  switch (path) {
    case "addP50":
      return result.timings.addMs?.p50
    case "searchP50":
      return result.timings.searchMs?.p50
    case "hitAt5":
      return result.retrievalMetrics?.hitAt5
    case "mrr":
      return result.retrievalMetrics?.mrr
    case "bleu1":
      return result.qaMetrics?.bleu1
  }
}

const formatMetric = (value: number | undefined, digits = 3) =>
  value === undefined ? "n/a" : value.toFixed(digits)

const scenarioStatus = (entry: ComparisonEntry) => {
  const py = entry.python?.status ?? "missing"
  const ts = entry.typescript?.status ?? "missing"
  return `py=${py} ts=${ts}`
}

const tableRow = (columns: ReadonlyArray<string>) => `| ${columns.join(" | ")} |`

const datasetSummary = (dataset: DatasetProvenance) =>
  `${dataset.datasetKind} @ ${dataset.datasetPath}${dataset.fallbackUsed ? " (fallback used)" : ""}`

export const renderMarkdownReport = (report: ComparisonReport) => {
  const lines = [
    `# ${report.title}`,
    "",
    `- Suite: \`${report.suiteId}\``,
    `- Tier: \`${report.tier}\``,
    `- Generated: ${report.generatedAt}`,
    "",
    "## Dataset Provenance",
    "",
    tableRow(["Scenario", "Dataset", "Required", "Fallback"]),
    tableRow(["---", "---", "---", "---"]),
    ...report.datasets.map((dataset) =>
      tableRow([
        dataset.scenarioId,
        datasetSummary(dataset),
        dataset.datasetRequired ? "yes" : "no",
        dataset.fallbackAllowed ? (dataset.fallbackUsed ? "used" : "allowed") : "disabled"
      ])
    ),
    "",
    "## Scenario Summary",
    "",
    tableRow(["Scenario", "Status", "Py add p50", "TS add p50", "Py search p50", "TS search p50", "Py hit@5", "TS hit@5"]),
    tableRow(["---", "---", "---", "---", "---", "---", "---", "---"])
  ]

  for (const entry of report.comparison) {
    lines.push(
      tableRow([
        entry.scenarioId,
        scenarioStatus(entry),
        formatMetric(metricValue(entry.python, "addP50"), 2),
        formatMetric(metricValue(entry.typescript, "addP50"), 2),
        formatMetric(metricValue(entry.python, "searchP50"), 2),
        formatMetric(metricValue(entry.typescript, "searchP50"), 2),
        formatMetric(metricValue(entry.python, "hitAt5")),
        formatMetric(metricValue(entry.typescript, "hitAt5"))
      ])
    )
  }

  lines.push("", "## Detailed Comparison", "")

  for (const entry of report.comparison) {
    lines.push(`### ${entry.scenarioId}`, "")
    lines.push(`- Status: ${scenarioStatus(entry)}`)
    if (entry.notes.length > 0) {
      lines.push(`- Notes: ${entry.notes.join("; ")}`)
    }
    if (entry.parityWarnings.length > 0) {
      lines.push(`- Parity Warnings: ${entry.parityWarnings.join("; ")}`)
    }
    lines.push(
      `- Add p50: py=${formatMetric(metricValue(entry.python, "addP50"), 2)}ms ts=${formatMetric(metricValue(entry.typescript, "addP50"), 2)}ms`
    )
    lines.push(
      `- Search p50: py=${formatMetric(metricValue(entry.python, "searchP50"), 2)}ms ts=${formatMetric(metricValue(entry.typescript, "searchP50"), 2)}ms`
    )
    lines.push(
      `- Retrieval: py hit@5=${formatMetric(metricValue(entry.python, "hitAt5"))}, ts hit@5=${formatMetric(metricValue(entry.typescript, "hitAt5"))}, py MRR=${formatMetric(metricValue(entry.python, "mrr"))}, ts MRR=${formatMetric(metricValue(entry.typescript, "mrr"))}`
    )
    lines.push(
      `- QA BLEU-1: py=${formatMetric(metricValue(entry.python, "bleu1"))}, ts=${formatMetric(metricValue(entry.typescript, "bleu1"))}`
    )
    lines.push("")
  }

  return `${lines.join("\n")}\n`
}
