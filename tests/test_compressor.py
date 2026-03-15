"""Tests for the token compression module."""

import pytest


@pytest.fixture(scope="module")
def llmlingua_scorer():
    from fraction.compressor import LLMLingua2Scorer
    return LLMLingua2Scorer()


class TestLLMLingua2Scorer:
    def test_compress_returns_fragment(self, llmlingua_scorer):
        text = "I really love hiking in the beautiful Rocky Mountains every summer vacation."
        result = llmlingua_scorer.compress(text, rate=0.5)
        assert result.original_text == text
        assert len(result.compressed_text) > 0
        assert result.compressed_text != text  # should be shorter
        assert 0 < result.compression_ratio <= 1.0

    def test_compress_preserves_entities(self, llmlingua_scorer):
        text = "John Smith works at Google in San Francisco since January 2024."
        result = llmlingua_scorer.compress(text, rate=0.6)
        compressed = result.compressed_text.lower()
        # Key entities should survive compression
        assert any(name in compressed for name in ["john", "smith", "google", "san francisco"])

    def test_compress_empty_text(self, llmlingua_scorer):
        result = llmlingua_scorer.compress("", rate=0.5)
        assert result.compressed_text == ""
        assert result.compression_ratio == 1.0

    def test_compress_deterministic(self, llmlingua_scorer):
        text = "The quick brown fox jumps over the lazy dog near the river."
        r1 = llmlingua_scorer.compress(text, rate=0.5)
        r2 = llmlingua_scorer.compress(text, rate=0.5)
        assert r1.compressed_text == r2.compressed_text

    def test_score_returns_token_pairs(self, llmlingua_scorer):
        text = "Machine learning is transforming healthcare."
        scores = llmlingua_scorer.score(text)
        assert len(scores) > 0
        for token, score in scores:
            assert isinstance(token, str)
            assert isinstance(score, (int, float))

    def test_compression_rate_respected(self, llmlingua_scorer):
        text = " ".join(["The student studied hard for the final exam."] * 5)
        result = llmlingua_scorer.compress(text, rate=0.3)
        # Allow 20% tolerance
        assert result.compression_ratio < 0.5
