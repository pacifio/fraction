"""USearch-backed vector store for Fraction.

Single-file persistence, memory-mapped serving, i8 quantization support.
Replaces mem0's 28-backend abstraction with one fast, embedded backend.
"""

import json
import os

import numpy as np

from fraction.config import FractionConfig
from fraction.types import SearchResult


class USearchVectorStore:
    """Vector store using USearch HNSW index."""

    def __init__(self, config: FractionConfig):
        from usearch.index import Index

        self._Index = Index
        self.config = config
        self.index = Index(
            ndim=config.embedding_dim,
            metric=config.metric,
            dtype=config.dtype,
        )
        self.metadata: dict[int, dict] = {}  # key -> payload
        self._next_key: int = 0
        self._deleted: set[int] = set()

        # Load existing index if available
        if os.path.exists(config.vector_store_path) and os.path.exists(config.metadata_path):
            self.load()

    def insert(self, vector: list[float], payload: dict) -> int:
        """Insert a vector with metadata. Returns the assigned key."""
        key = self._next_key
        self._next_key += 1
        arr = np.array(vector, dtype=np.float32)
        self.index.add(key, arr)
        self.metadata[key] = payload
        return key

    def insert_batch(self, vectors: list[list[float]], payloads: list[dict]) -> list[int]:
        """Batch insert vectors with metadata."""
        keys = list(range(self._next_key, self._next_key + len(vectors)))
        self._next_key += len(vectors)
        arr = np.array(vectors, dtype=np.float32)
        self.index.add(np.array(keys), arr)
        for key, payload in zip(keys, payloads):
            self.metadata[key] = payload
        return keys

    def search(self, query_vector: list[float], limit: int = 10, filters: dict = None) -> list[SearchResult]:
        """Search for nearest neighbors, optionally filtered by metadata."""
        if len(self.metadata) == 0:
            return []

        # Over-fetch to account for deleted/filtered results
        fetch_limit = min(limit * 3, len(self.metadata))
        arr = np.array(query_vector, dtype=np.float32)
        matches = self.index.search(arr, fetch_limit)

        results = []
        for key, distance in zip(matches.keys, matches.distances):
            key = int(key)
            if key in self._deleted:
                continue
            if key not in self.metadata:
                continue

            payload = self.metadata[key]

            # Apply metadata filters
            if filters and not self._matches_filters(payload, filters):
                continue

            # Convert distance to similarity score (cosine: similarity = 1 - distance)
            score = 1.0 - float(distance) if self.config.metric == "cos" else -float(distance)

            results.append(SearchResult(
                id=payload.get("id", str(key)),
                content=payload.get("content", ""),
                score=score,
                metadata=payload,
                created_at=payload.get("created_at"),
            ))

            if len(results) >= limit:
                break

        return results

    def get(self, memory_id: str) -> dict | None:
        """Get metadata by memory ID."""
        for key, payload in self.metadata.items():
            if key not in self._deleted and payload.get("id") == memory_id:
                return payload
        return None

    def get_key_by_id(self, memory_id: str) -> int | None:
        """Get internal key by memory ID."""
        for key, payload in self.metadata.items():
            if key not in self._deleted and payload.get("id") == memory_id:
                return key
        return None

    def update(self, memory_id: str, vector: list[float], payload: dict) -> bool:
        """Update an existing entry (delete + re-insert with same ID)."""
        key = self.get_key_by_id(memory_id)
        if key is None:
            return False
        # Mark old as deleted and insert new
        self._deleted.add(key)
        del self.metadata[key]
        new_key = self._next_key
        self._next_key += 1
        arr = np.array(vector, dtype=np.float32)
        self.index.add(new_key, arr)
        self.metadata[new_key] = payload
        return True

    def delete(self, memory_id: str) -> bool:
        """Soft-delete a memory by ID."""
        key = self.get_key_by_id(memory_id)
        if key is None:
            return False
        self._deleted.add(key)
        del self.metadata[key]
        return True

    def list_all(self, filters: dict = None, limit: int = 100) -> list[dict]:
        """List all memories, optionally filtered."""
        results = []
        for key, payload in self.metadata.items():
            if key in self._deleted:
                continue
            if filters and not self._matches_filters(payload, filters):
                continue
            results.append(payload)
            if len(results) >= limit:
                break
        return results

    def count(self) -> int:
        """Number of active (non-deleted) entries."""
        return len(self.metadata) - len(self._deleted & set(self.metadata.keys()))

    def save(self):
        """Persist index and metadata to disk."""
        self.index.save(self.config.vector_store_path)
        state = {
            "metadata": {str(k): v for k, v in self.metadata.items()},
            "next_key": self._next_key,
            "deleted": list(self._deleted),
        }
        with open(self.config.metadata_path, "w") as f:
            json.dump(state, f, default=str)

    def load(self):
        """Load index and metadata from disk."""
        if os.path.exists(self.config.vector_store_path):
            self.index = self._Index(
                ndim=self.config.embedding_dim,
                metric=self.config.metric,
                dtype=self.config.dtype,
            )
            self.index.load(self.config.vector_store_path)
        if os.path.exists(self.config.metadata_path):
            with open(self.config.metadata_path, "r") as f:
                state = json.load(f)
            self.metadata = {int(k): v for k, v in state.get("metadata", {}).items()}
            self._next_key = state.get("next_key", 0)
            self._deleted = set(state.get("deleted", []))

    def reset(self):
        """Clear all data."""
        self.index = self._Index(
            ndim=self.config.embedding_dim,
            metric=self.config.metric,
            dtype=self.config.dtype,
        )
        self.metadata.clear()
        self._next_key = 0
        self._deleted.clear()
        # Remove persisted files
        for path in [self.config.vector_store_path, self.config.metadata_path]:
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _matches_filters(payload: dict, filters: dict) -> bool:
        """Check if payload matches all filter criteria."""
        for key, value in filters.items():
            if key not in payload:
                return False
            if payload[key] != value:
                return False
        return True
