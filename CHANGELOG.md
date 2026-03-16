# Changelog

## 0.1.1

### Added
- **LLM-based fact extraction** — new `compressor_type="llm"` option that uses any LLM provider (via litellm) to extract key facts from text, similar to mem0/supermemory's approach
- **litellm integration** — pass any provider's model string (OpenAI, Anthropic, Ollama, etc.) via `llm_model` config
- **Binary LLM judge** — 0/1 evaluation scale matching mem0's benchmark methodology, alongside the existing 1-5 Likert scale
- **Comparative benchmark mode** — `--compressor_type compare` runs both LLMLingua-2 and LLM extraction back-to-back with a side-by-side results table
- **New config options**: `llm_model`, `llm_api_key`, `llm_api_base`, `llm_extraction_prompt`
- **New CLI flags**: `--compressor_type`, `--judge_mode`, `--extractor_model`

### Changed
- Replaced direct OpenAI dependency with litellm for LLM extraction (optional: `pip install fractionally[llm]`)
- Benchmark reports now include binary judge scores and compressor type
- Package renamed to `fractionally` on PyPI

## 0.1.0

Initial release.

- LLMLingua-2 token compression (zero LLM calls, sub-100ms ingestion)
- Hybrid retrieval: vector similarity + BM25 + entity graph + temporal boost via RRF
- Adaptive compression for short texts
- Relevance gate to skip filler turns
- High-level `Memory` API with auto-persistence
- Low-level `Fraction` API for manual control
- LoCoMo benchmark suite with BLEU-1, F1, and LLM judge metrics
- spaCy NER entity extraction
- USearch HNSW vector indexing
- SQLite change history tracking
