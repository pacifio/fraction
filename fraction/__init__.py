"""Fraction: Persistent memory layer for LLM agents and applications.

Zero LLM calls. Sub-100ms ingestion. Deterministic extraction.
Uses LLMLingua-2 token compression + USearch vector indexing.

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

__version__ = "0.1.0"
__all__ = [
    "Memory",
    "Fraction",
    "FractionConfig",
    "MemoryItem",
    "SearchResult",
    "CompressedFragment",
]
