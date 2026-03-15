"""High-level Memory client for agents and AI applications.

Drop-in memory layer with automatic persistence. Designed to match
the simplicity of mem0/supermemory while running fully offline.

Usage:
    from fraction import Memory

    m = Memory()
    m.add("I love hiking in the Rocky Mountains.", user_id="alice")
    results = m.search("outdoor activities", user_id="alice")

    # Memories auto-persist — no save() calls needed.
"""

import os
from pathlib import Path

from fraction.config import FractionConfig
from fraction.memory import Fraction


_DEFAULT_DATA_DIR = os.path.join(Path.home(), ".fraction")


class Memory:
    """Persistent memory layer for LLM agents and applications.

    Wraps Fraction with automatic persistence, managed storage directories,
    and a familiar API surface (compatible with mem0/supermemory patterns).

    Args:
        data_dir: Directory for persistent storage. Defaults to ~/.fraction/
        config: Optional FractionConfig for advanced tuning.
        auto_save: If True (default), persist after every write operation.

    Examples:
        # Basic usage
        m = Memory()
        m.add("User prefers dark mode.", user_id="alice")
        results = m.search("UI preferences", user_id="alice")

        # Per-project storage
        m = Memory(data_dir="./my_agent_memory")

        # With custom config
        from fraction import FractionConfig
        m = Memory(config=FractionConfig(compression_rate=0.5))

        # Context manager
        with Memory() as m:
            m.add("fact", user_id="u1")
            results = m.search("query", user_id="u1")
    """

    def __init__(
        self,
        data_dir: str = None,
        config: FractionConfig = None,
        auto_save: bool = True,
    ):
        self._data_dir = data_dir or _DEFAULT_DATA_DIR
        os.makedirs(self._data_dir, exist_ok=True)

        self._config = config or FractionConfig()
        # Point storage paths into the managed data directory
        self._config.vector_store_path = os.path.join(self._data_dir, "index.usearch")
        self._config.metadata_path = os.path.join(self._data_dir, "metadata.json")
        self._config.history_db_path = os.path.join(self._data_dir, "history.db")

        self._auto_save = auto_save
        self._fraction = Fraction(self._config)

        # Auto-load existing state if present
        if os.path.exists(self._config.metadata_path):
            self._fraction.load()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.save()

    # ── Write operations ──────────────────────────────────────────────

    def add(
        self,
        messages,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        metadata: dict = None,
    ) -> dict:
        """Add memories from text, message list, or conversation dict.

        Args:
            messages: str, list[str], or list[dict] with role/content keys.
            user_id: Scope memory to a user.
            agent_id: Scope memory to an agent.
            run_id: Scope memory to a conversation/run.
            metadata: Arbitrary key-value metadata to attach.

        Returns:
            {"results": [{"id": str, "memory": str, "event": "ADD"|"UPDATE"|"SKIP"}]}
        """
        result = self._fraction.add(
            messages, user_id=user_id, agent_id=agent_id,
            run_id=run_id, metadata=metadata,
        )
        if self._auto_save:
            self._fraction.save()
        return result

    def update(self, memory_id: str, data: str) -> dict:
        """Update a memory's content. Re-compresses and re-embeds.

        Args:
            memory_id: ID of the memory to update.
            data: New text content.

        Returns:
            {"id": str, "memory": str, "event": "UPDATE"} or {"error": str}
        """
        result = self._fraction.update(memory_id, data)
        if self._auto_save:
            self._fraction.save()
        return result

    def delete(self, memory_id: str) -> dict:
        """Delete a memory by ID.

        Returns:
            {"id": str, "event": "DELETE"} or {"error": str}
        """
        result = self._fraction.delete(memory_id)
        if self._auto_save:
            self._fraction.save()
        return result

    def delete_all(
        self,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
    ) -> dict:
        """Delete all memories matching the given scope.

        Returns:
            {"deleted": int}
        """
        result = self._fraction.delete_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id,
        )
        if self._auto_save:
            self._fraction.save()
        return result

    # ── Read operations ───────────────────────────────────────────────

    def search(
        self,
        query: str,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        limit: int = None,
        filters: dict = None,
    ) -> dict:
        """Search for relevant memories using hybrid retrieval.

        Combines vector similarity, BM25 keyword matching, entity graph
        traversal, and temporal boosting via Reciprocal Rank Fusion.

        Args:
            query: Natural language search query.
            user_id: Filter to memories from this user.
            agent_id: Filter to memories from this agent.
            run_id: Filter to memories from this run.
            limit: Max results (default: config.top_k).
            filters: Additional metadata filters.

        Returns:
            {"results": [{"id", "memory", "score", "metadata", "created_at"}], "latency_ms": float}
        """
        return self._fraction.search(
            query, user_id=user_id, agent_id=agent_id,
            run_id=run_id, limit=limit, filters=filters,
        )

    def get(self, memory_id: str) -> dict | None:
        """Get a single memory by ID.

        Returns:
            {"id", "memory", "metadata", "created_at"} or None
        """
        return self._fraction.get(memory_id)

    def get_all(
        self,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        limit: int = 100,
    ) -> dict:
        """List all memories with optional scope filtering.

        Returns:
            {"results": [{"id", "memory", "metadata", "created_at"}]}
        """
        return self._fraction.get_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit,
        )

    def history(self, memory_id: str) -> list[dict]:
        """Get the change history for a memory.

        Returns:
            List of {"id", "memory_id", "event", "old_content", "new_content", "user_id", "timestamp"}
        """
        return self._fraction.history(memory_id)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self):
        """Manually persist all state to disk."""
        self._fraction.save()

    def reset(self):
        """Clear all memories and persisted state."""
        self._fraction.reset()

    @property
    def data_dir(self) -> str:
        """Path to the storage directory."""
        return self._data_dir

    @property
    def config(self) -> FractionConfig:
        """Current configuration."""
        return self._config
