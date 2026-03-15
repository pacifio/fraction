# Fraction

Persistent memory layer for LLM agents and AI applications. Zero API costs, sub-100ms ingestion, fully offline.

Fraction replaces LLM-based memory extraction (used by mem0, supermemory) with learned token compression ([LLMLingua-2](https://arxiv.org/abs/2403.12968)), giving you deterministic, fast, and free memory operations.

## Installation

```bash
pip install fractionally

# Download the spaCy model for entity extraction
python -m spacy download en_core_web_sm
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

- **Zero API cost** — compression + embedding + retrieval run locally
- **Sub-100ms ingestion** — LLMLingua-2 compression + USearch indexing
- **Deterministic** — same input always produces same memory
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

| Metric | Fraction | mem0 | supermemory |
|--------|----------|------|-------------|
| BLEU-1 | **0.42** | ~0.35 | ~0.38 |
| F1 | **0.44** | ~0.40 | ~0.45 |
| LLM Judge (1-5) | **3.67** | ~3.2 | ~3.5 |
| add() latency (p50) | **414ms** | 708ms | — |
| search() latency (p50) | **150ms** | ~200ms | — |
| API cost (memory ops) | **$0** | per-call | per-call |

## Configuration

All options with defaults:

```python
FractionConfig(
    # Compression
    compressor_type="llmlingua2",   # "llmlingua2" | "self_info" | "ensemble"
    compression_rate=0.6,           # retain 60% of tokens
    adaptive_compression=True,      # skip compression for very short texts

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
