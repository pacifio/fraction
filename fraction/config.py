"""Configuration for Fraction memory system."""

from pydantic import BaseModel


class FractionConfig(BaseModel):
    """All configuration for a Fraction instance. Sensible defaults that work out-of-the-box."""

    # Compressor
    compressor_type: str = "llmlingua2"  # "llmlingua2" | "self_info" | "ensemble" | "llm"
    compression_rate: float = 0.6  # retain 60% of tokens (higher = more context)
    adaptive_compression: bool = True  # scale rate by input length (short texts get less compression)
    importance_threshold: float = 0.0

    # LLM extractor (used when compressor_type="llm")
    # Uses litellm — pass any provider's model string (e.g. "gpt-4o-mini", "anthropic/claude-sonnet-4-20250514", "ollama/llama3")
    llm_model: str = "gpt-4o-mini"  # litellm model string for fact extraction
    llm_api_key: str | None = None  # API key (falls back to provider env vars, e.g. OPENAI_API_KEY)
    llm_api_base: str | None = None  # custom API base URL (for self-hosted/ollama)
    llm_extraction_prompt: str | None = None  # custom extraction prompt (optional)

    # Relevance gate — skip filler turns that add noise to retrieval
    relevance_gate: bool = True
    min_content_words: int = 3  # minimum non-stopword words in compressed text to be stored

    # Embedder
    embedder_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # Vector store
    vector_store_path: str = "fraction_index.usearch"
    metadata_path: str = "fraction_metadata.json"
    metric: str = "cos"
    dtype: str = "f32"

    # Retrieval
    top_k: int = 10
    use_bm25: bool = True
    use_graph: bool = True
    rerank: bool = True
    duplicate_threshold: float = 0.95  # cosine similarity above this = duplicate
    min_score_ratio: float = 0.0  # drop results with score < top_score * this ratio (0 = disabled)

    # RRF
    rrf_k: int = 30  # lower k = sharper top-rank preference
    rrf_weights: dict = None  # per-signal weights, e.g. {"vector": 1.0, "bm25": 1.2}

    # Graph
    entity_similarity_threshold: float = 0.7

    # Storage
    history_db_path: str = "fraction_history.db"

    # Session defaults
    default_user_id: str = "default"
