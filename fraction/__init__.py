"""Fraction: Persistent memory layer for LLM agents and applications.

Two extraction modes:
- LLMLingua-2 (default): Zero LLM calls, sub-100ms ingestion, deterministic.
- LLM extraction: OpenAI-powered fact extraction for higher quality.

Usage:
    from fraction import Memory

    m = Memory()
    m.add("I love hiking in the Rocky Mountains.", user_id="alice")
    results = m.search("outdoor activities", user_id="alice")
"""

from fraction.client import Memory
from fraction.config import FractionConfig
from fraction.memory import Fraction
from fraction.types import CompressedFragment, MemoryItem, SearchResult

__version__ = "0.1.1"
__all__ = [
    "Memory",
    "Fraction",
    "FractionConfig",
    "MemoryItem",
    "SearchResult",
    "CompressedFragment",
]
