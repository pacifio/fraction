"""Data models for Fraction memory system."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class CompressedFragment(BaseModel):
    """Result of token compression on a text input."""
    original_text: str
    compressed_text: str
    compression_ratio: float
    token_scores: list[tuple[str, float]] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class MemoryItem(BaseModel):
    """A single memory stored in Fraction."""
    id: str
    content: str
    content_raw: Optional[str] = None
    importance_scores: list[float] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hash: str = ""


class SearchResult(BaseModel):
    """A single search result from Fraction retrieval."""
    id: str
    content: str
    score: float
    metadata: dict = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class MemoryEvent(BaseModel):
    """Record of a memory operation."""
    id: str
    memory_id: str
    event: str  # ADD, UPDATE, DELETE
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
