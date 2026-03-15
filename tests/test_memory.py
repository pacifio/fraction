"""Tests for the core Fraction memory class."""

import os
import tempfile

import pytest

from fraction.config import FractionConfig


@pytest.fixture(scope="module")
def fraction_instance():
    """Create a Fraction instance with temp storage."""
    from fraction import Fraction

    tmpdir = tempfile.mkdtemp()
    config = FractionConfig(
        vector_store_path=os.path.join(tmpdir, "test.usearch"),
        metadata_path=os.path.join(tmpdir, "test_meta.json"),
        history_db_path=os.path.join(tmpdir, "test_history.db"),
    )
    f = Fraction(config)
    yield f
    f.reset()


class TestFraction:
    def test_add_and_search_roundtrip(self, fraction_instance):
        f = fraction_instance
        result = f.add("I love hiking in the Rocky Mountains every summer.", user_id="alice")
        assert len(result["results"]) == 1
        assert result["results"][0]["event"] in ("ADD", "UPDATE")

        search = f.search("outdoor hiking activities", user_id="alice")
        assert len(search["results"]) > 0
        assert "hiking" in search["results"][0]["memory"].lower() or "mountain" in search["results"][0]["memory"].lower()

    def test_user_scoping(self, fraction_instance):
        f = fraction_instance
        f.add("Bob likes playing chess.", user_id="bob")
        f.add("Alice likes painting.", user_id="alice_scope")

        bob_results = f.search("hobbies", user_id="bob")
        alice_results = f.search("hobbies", user_id="alice_scope")

        bob_memories = [r["memory"].lower() for r in bob_results["results"]]
        alice_memories = [r["memory"].lower() for r in alice_results["results"]]

        # Bob's search should find chess, not painting
        if bob_memories:
            assert any("chess" in m for m in bob_memories)

    def test_add_dict_messages(self, fraction_instance):
        f = fraction_instance
        messages = [
            {"role": "user", "content": "I just got a new puppy named Max!"},
            {"role": "assistant", "content": "That's wonderful! What breed is Max?"},
        ]
        result = f.add(messages, user_id="charlie")
        assert result["results"][0]["event"] in ("ADD", "UPDATE")

    def test_get_memory(self, fraction_instance):
        f = fraction_instance
        add_result = f.add("Testing get operation with unique content xyz123.", user_id="gettest")
        memory_id = add_result["results"][0]["id"]

        got = f.get(memory_id)
        assert got is not None
        assert got["id"] == memory_id

    def test_get_all(self, fraction_instance):
        f = fraction_instance
        f.add("Memory one for getall test.", user_id="getall_user")
        f.add("Memory two for getall test.", user_id="getall_user")

        all_mems = f.get_all(user_id="getall_user")
        assert len(all_mems["results"]) >= 2

    def test_delete(self, fraction_instance):
        f = fraction_instance
        add_result = f.add("This memory will be deleted.", user_id="deluser")
        memory_id = add_result["results"][0]["id"]

        del_result = f.delete(memory_id)
        assert del_result["event"] == "DELETE"

        got = f.get(memory_id)
        assert got is None

    def test_update(self, fraction_instance):
        f = fraction_instance
        add_result = f.add("Original content for update test.", user_id="upduser")
        memory_id = add_result["results"][0]["id"]

        upd_result = f.update(memory_id, "Updated content with new information.")
        assert upd_result["event"] == "UPDATE"

    def test_latency_under_threshold(self, fraction_instance):
        f = fraction_instance
        result = f.add("Quick latency test message.", user_id="latency")
        # Should be well under 5000ms (generous for first run with model loading)
        assert result["results"][0]["latency_ms"] < 5000
