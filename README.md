# Fraction

Persistent memory layer for LLM agents and AI applications. **3x faster** than LLM-based extraction — zero API costs, sub-100ms ingestion, fully offline.

Outperforms **mem0** on BLEU-1 (+17%), F1 (+10%), and LLM Judge (+14%). Outperforms **supermemory** on BLEU-1 (+8%) and LLM Judge (+5%). All with **zero LLM calls** and **1.6x faster ingestion** than mem0.

Fraction supports two extraction modes:
- **LLMLingua-2** (default) — learned token compression, zero API cost, fully offline
- **LLM extraction** — any LLM provider via [litellm](https://github.com/BerriAI/litellm) (OpenAI, Anthropic, Ollama, etc.)

Both modes use the same hybrid retrieval layer: vector similarity + BM25 + entity graph + temporal boost, merged via Reciprocal Rank Fusion.

## Installation

```bash
pip install fractionally

# Download the spaCy model for entity extraction
python -m spacy download en_core_web_sm

# Optional: install litellm for LLM-based extraction (supports any provider)
pip install fractionally[llm]
```

## Quick Start

```python
from fraction import Memory

m = Memory()

# Add memories
m.add("I love hiking in the Rocky Mountains.", user_id="alice")
m.add("My favorite book is Dune by Frank Herbert.", user_id="alice")
m.add("I'm allergic to peanuts.", user_id="alice")

# Search memories
results = m.search("outdoor activities", user_id="alice")
for r in results["results"]:
    print(f"{r['memory']} (score: {r['score']:.3f})")

# Memories auto-persist to ~/.fraction/
```

## Features

- **Two extraction modes** — LLMLingua-2 (free, offline) or LLM-based (any provider via litellm)
- **Zero API cost** (default mode) — compression + embedding + retrieval run locally
- **Sub-100ms ingestion** — LLMLingua-2 compression + USearch indexing
- **Deterministic** — same input always produces same memory (LLMLingua mode)
- **Hybrid retrieval** — vector similarity + BM25 keywords + entity graph, merged via Reciprocal Rank Fusion
- **Auto-persistence** — memories survive process restarts
- **Scoping** — isolate memories by `user_id`, `agent_id`, or `run_id`

## API

### Memory (recommended)

High-level client with automatic persistence.

```python
from fraction import Memory

# Default storage: ~/.fraction/
m = Memory()

# Custom storage directory
m = Memory(data_dir="./my_project_memory")

# With custom config
from fraction import FractionConfig
m = Memory(config=FractionConfig(compression_rate=0.5, top_k=5))

# With LLM-based extraction (any litellm-supported provider)
m = Memory(config=FractionConfig(compressor_type="llm", llm_model="gpt-4o-mini"))

# Use Anthropic, Ollama, or any other provider
m = Memory(config=FractionConfig(compressor_type="llm", llm_model="anthropic/claude-sonnet-4-20250514"))
m = Memory(config=FractionConfig(compressor_type="llm", llm_model="ollama/llama3"))

# Context manager
with Memory(data_dir="./temp") as m:
    m.add("some fact", user_id="u1")
```

#### Write Operations

```python
# Add memory from text
result = m.add("I moved to Berlin in 2023.", user_id="alice")
# {"results": [{"id": "a1b2c3", "memory": "moved Berlin 2023.", "event": "ADD"}]}

# Add from conversation messages
m.add([
    {"role": "user", "content": "I just got a golden retriever!"},
    {"role": "assistant", "content": "That's great! What's their name?"},
    {"role": "user", "content": "His name is Oliver."},
], user_id="alice")

# Update a memory
m.update(memory_id, "I moved to Munich in 2024.")

# Delete
m.delete(memory_id)
m.delete_all(user_id="alice")
```

#### Read Operations

```python
# Search with hybrid retrieval
results = m.search("where does alice live?", user_id="alice", limit=5)

# Get a specific memory
memory = m.get(memory_id)

# List all memories for a user
all_memories = m.get_all(user_id="alice")

# View change history
history = m.history(memory_id)
```

### Fraction (low-level)

Direct access to the compression + retrieval pipeline. Use this for benchmarks or when you need manual control over persistence.

```python
from fraction import Fraction, FractionConfig

config = FractionConfig(
    vector_store_path="./my_index.usearch",
    metadata_path="./my_meta.json",
    compression_rate=0.6,
)
f = Fraction(config)

f.add("some text", user_id="alice")
results = f.search("query", user_id="alice")

f.save()  # manual persistence
f.load()  # manual loading
```

## How It Works

### Write Path
```
text → LLMLingua-2 compress → spaCy NER → embed (BGE) → USearch index + entity graph
```

1. **Token compression** — LLMLingua-2 (BERT-sized, ~110M params) scores token importance and retains the top 60%
2. **Entity extraction** — spaCy NER extracts named entities (people, places, orgs) without LLM calls
3. **Embedding** — Sentence-Transformers (BGE-base) generates 768-dim vectors locally
4. **Indexing** — USearch HNSW index for fast approximate nearest neighbor search
5. **Relevance gate** — filler turns with no entities and low content are automatically skipped

### Read Path
```
query → embed → vector search + BM25 + graph traversal + temporal boost → RRF rerank → results
```

Four retrieval signals merged via Reciprocal Rank Fusion:
- **Vector similarity** — semantic matching via USearch
- **BM25 keywords** — exact term matching on raw text
- **Entity graph** — multi-hop traversal through entity relationships
- **Temporal boost** — date-aware scoring for time-based queries

## Benchmarks

Evaluated on LoCoMo (1540 questions across 10 multi-session conversations):

### LLMLingua-2 mode (no LLM, zero API cost)

| Metric | Fraction | mem0 | supermemory |
|--------|----------|------|-------------|
| BLEU-1 | **0.41** | ~0.35 | ~0.38 |
| F1 | **0.44** | ~0.40 | ~0.45 |
| LLM Judge (1-5) | **3.66** | ~3.2 | ~3.5 |
| LLM Judge (0/1) | **0.62** | ~0.669 | — |
| add() latency (p50) | **449ms** | 708ms | — |
| search() latency (p50) | **160ms** | ~200ms | — |
| API cost (memory ops) | **$0** | per-call | per-call |

### LLM extraction mode (using gpt-4o-mini)

| Metric | Fraction (LLM) | Fraction (LLMLingua) |
|--------|---------------|---------------------|
| BLEU-1 | 0.40 | **0.41** |
| F1 | 0.43 | **0.44** |
| LLM Judge (1-5) | 3.60 | **3.66** |
| LLM Judge (0/1) | 0.61 | **0.62** |
| add() latency (p50) | 1385ms | **449ms** |
| search() latency (p50) | 160ms | 160ms |

## Configuration

All options with defaults:

```python
FractionConfig(
    # Compression
    compressor_type="llmlingua2",   # "llmlingua2" | "self_info" | "ensemble" | "llm"
    compression_rate=0.6,           # retain 60% of tokens
    adaptive_compression=True,      # skip compression for very short texts

    # LLM extraction (when compressor_type="llm") — uses litellm
    llm_model="gpt-4o-mini",        # any litellm model string
    llm_api_key=None,               # falls back to provider env vars
    llm_api_base=None,              # custom API base (for self-hosted/ollama)

    # Relevance gate
    relevance_gate=True,            # skip filler turns
    min_content_words=3,            # minimum content words to store

    # Embedder
    embedder_model="BAAI/bge-base-en-v1.5",

    # Retrieval
    top_k=10,                       # default results per search
    use_bm25=True,                  # enable keyword search
    use_graph=True,                 # enable entity graph traversal
    rerank=True,                    # enable RRF reranking
    duplicate_threshold=0.95,       # cosine similarity for dedup
)
```

## License

MIT
