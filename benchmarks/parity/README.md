# Py vs TS Parity Benchmarks

This suite benchmarks the Python Fraction implementation against the Bun/TypeScript implementation with a shared scenario contract and normalized report format.

The built-in parity commands now use the canonical LoCoMo dataset at `benchmarks/dataset/locomo10.json`, matching the historical Python benchmark input contract.

## Tiers

- `strict-pipeline`
  - same datasets
  - same scenario contract
  - same normalization
  - same metrics
  - shared benchmark-only `off` compression baseline by default
  - canonical LoCoMo dataset
  - `topK=15`
  - `compressionRate=0.6`
- `shipped-defaults`
  - compare both runtimes as currently shipped
  - canonical LoCoMo dataset
  - `topK=15`
  - `compressionRate=0.6`

## Commands

```bash
uv sync
uv run python -m spacy download en_core_web_sm

bun run benchmark:parity:smoke
bun run benchmark:parity:strict
bun run benchmark:parity:defaults
bun run benchmark:parity:qa
```

Outputs are written under:

```text
benchmarks/parity/reports/<suite-id>/<timestamp>/
```

Each run emits:

- `raw/python.json`
- `raw/typescript.json`
- `comparison.json`
- `summary.md`
- `metadata.json`

## Dataset Handling

- all built-in benchmark commands use [datasets/locomo.manifest.json](./datasets/locomo.manifest.json)
- the manifest points to `benchmarks/dataset/locomo10.json`
- the parity harness does not auto-generate a synthetic LoCoMo fallback
- if `benchmarks/dataset/locomo10.json` is missing, canonical runs fail immediately

The historical Python benchmark also expects `benchmarks/dataset/locomo10.json` directly, so this keeps the parity harness aligned with the original benchmark setup instead of silently substituting generated data.

## Python Environment

The parity orchestrator prefers `uv run python` automatically when `uv` is installed. Use the repo root as the working directory so `uv` resolves the local project environment.

The Python runner requires the Python Fraction dependencies to be installed in the active environment. At minimum, the runner checks for:

- `pydantic`
- `spacy`
- `sentence_transformers`
- `usearch`
- `llmlingua`
- spaCy model `en_core_web_sm`

If those dependencies are missing, the suite still produces a report, but the Python scenarios will be marked as failed with an explicit dependency error.

## Scenario Contract

The suite schema is defined in [config.schema.json](./config.schema.json).

Each suite JSON defines:

- suite metadata
- runtime targets
- warmup and measured runs
- scenario list
- dataset references
- optional runtime-specific config overrides

The built-in suites are:

- [scenarios/smoke.json](./scenarios/smoke.json)
- [scenarios/strict.json](./scenarios/strict.json)
- [scenarios/defaults.json](./scenarios/defaults.json)
- [scenarios/qa.json](./scenarios/qa.json)

Suite intent:

- `smoke`
  - uses the canonical LoCoMo dataset with a small subset for a fast sanity run
- `strict`
  - uses the canonical LoCoMo dataset with the full retrieval workload
- `defaults`
  - uses the canonical LoCoMo dataset with shipped runtime defaults
- `qa`
  - uses the canonical LoCoMo dataset with the parity QA prompts and judges

## Notes

- `strict`, `defaults`, and `qa` all use `topK=15` and `compressionRate=0.6` to match the historical Python benchmark defaults.
- `strict` and `qa` currently default to a shared benchmark-only `off` compression path so Tier 1 measures the storage and retrieval pipeline without runtime-specific local model downloads.
- `defaults` keeps the real shipped product defaults, including runtime-native local compression.
- end-to-end QA requires `OPENAI_API_KEY`.
- the QA prompt and judge prompts are frozen in both runtimes to keep prompt parity stable.
- generated reports now include dataset provenance so each run records the exact dataset path and whether fallback was allowed or used.
