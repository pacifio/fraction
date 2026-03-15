"""Tests for USearch vector store."""

import os
import tempfile

import numpy as np
import pytest

from fraction.config import FractionConfig
from fraction.vector_store import USearchVectorStore


@pytest.fixture
def store():
    """Create a temporary vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = FractionConfig(
            vector_store_path=os.path.join(tmpdir, "test.usearch"),
            metadata_path=os.path.join(tmpdir, "test_meta.json"),
            embedding_dim=8,
        )
        yield USearchVectorStore(config)


def random_vec(dim=8):
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


class TestUSearchVectorStore:
    def test_insert_and_search(self, store):
        vec = random_vec()
        key = store.insert(vec, {"id": "mem1", "content": "hello world", "user_id": "alice"})
        assert key == 0

        results = store.search(vec, limit=1)
        assert len(results) == 1
        assert results[0].id == "mem1"
        assert results[0].score > 0.9  # should be very similar (same vector)

    def test_metadata_filtering(self, store):
        v1 = random_vec()
        v2 = random_vec()
        store.insert(v1, {"id": "m1", "content": "foo", "user_id": "alice"})
        store.insert(v2, {"id": "m2", "content": "bar", "user_id": "bob"})

        results = store.search(v1, limit=10, filters={"user_id": "alice"})
        assert all(r.metadata.get("user_id") == "alice" for r in results)

    def test_delete(self, store):
        vec = random_vec()
        store.insert(vec, {"id": "m1", "content": "to delete", "user_id": "u1"})
        assert store.count() == 1

        store.delete("m1")
        assert store.get("m1") is None

    def test_update(self, store):
        vec1 = random_vec()
        vec2 = random_vec()
        store.insert(vec1, {"id": "m1", "content": "original", "user_id": "u1"})

        store.update("m1", vec2, {"id": "m1", "content": "updated", "user_id": "u1"})
        payload = store.get("m1")
        assert payload["content"] == "updated"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FractionConfig(
                vector_store_path=os.path.join(tmpdir, "test.usearch"),
                metadata_path=os.path.join(tmpdir, "test_meta.json"),
                embedding_dim=8,
            )
            store1 = USearchVectorStore(config)
            vec = random_vec()
            store1.insert(vec, {"id": "m1", "content": "persisted", "user_id": "u1"})
            store1.save()

            store2 = USearchVectorStore(config)
            payload = store2.get("m1")
            assert payload is not None
            assert payload["content"] == "persisted"

    def test_list_all(self, store):
        for i in range(5):
            store.insert(random_vec(), {"id": f"m{i}", "content": f"item {i}", "user_id": "alice"})
        items = store.list_all()
        assert len(items) == 5

    def test_reset(self, store):
        store.insert(random_vec(), {"id": "m1", "content": "x", "user_id": "u1"})
        store.reset()
        assert store.count() == 0
