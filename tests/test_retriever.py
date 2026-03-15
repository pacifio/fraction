"""Tests for the hybrid retriever."""

import os
import tempfile

import pytest

from fraction.config import FractionConfig


@pytest.fixture(scope="module")
def fraction_with_data():
    """Fraction instance pre-loaded with test data."""
    from fraction import Fraction

    tmpdir = tempfile.mkdtemp()
    config = FractionConfig(
        vector_store_path=os.path.join(tmpdir, "test.usearch"),
        metadata_path=os.path.join(tmpdir, "test_meta.json"),
        history_db_path=os.path.join(tmpdir, "test_history.db"),
    )
    f = Fraction(config)
    # Add diverse memories
    f.add("I went to Paris last summer for a two-week vacation.", user_id="u1")
    f.add("My favorite programming language is Python.", user_id="u1")
    f.add("I have a meeting with Dr. Smith on Monday at 3pm.", user_id="u1")
    f.add("The best Italian restaurant in town is called Luigi's.", user_id="u1")
    f.add("I'm training for a marathon next spring.", user_id="u1")
    yield f
    f.reset()


class TestHybridRetriever:
    def test_vector_search_finds_semantic_match(self, fraction_with_data):
        results = fraction_with_data.search("travel and holidays", user_id="u1")
        memories = [r["memory"].lower() for r in results["results"]]
        assert any("paris" in m or "vacation" in m for m in memories)

    def test_bm25_finds_keyword_match(self, fraction_with_data):
        results = fraction_with_data.search("Python programming", user_id="u1")
        memories = [r["memory"].lower() for r in results["results"]]
        assert any("python" in m for m in memories)

    def test_results_have_scores(self, fraction_with_data):
        results = fraction_with_data.search("food and restaurants", user_id="u1")
        for r in results["results"]:
            assert "score" in r
            assert isinstance(r["score"], float)

    def test_limit_respected(self, fraction_with_data):
        results = fraction_with_data.search("anything", user_id="u1", limit=2)
        assert len(results["results"]) <= 2


class TestRRF:
    def test_reciprocal_rank_fusion(self):
        from fraction.retriever import HybridRetriever

        list1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        list2 = [("b", 0.95), ("c", 0.85), ("d", 0.75)]

        merged = HybridRetriever.reciprocal_rank_fusion([list1, list2], k=60)
        ids = [doc_id for doc_id, _ in merged]

        # "b" appears in both lists so should rank highest
        assert ids[0] == "b"

    def test_rrf_single_list(self):
        from fraction.retriever import HybridRetriever

        single = [("x", 1.0), ("y", 0.5)]
        merged = HybridRetriever.reciprocal_rank_fusion([single])
        assert merged[0][0] == "x"
