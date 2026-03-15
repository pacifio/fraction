"""Utility functions for Fraction."""

import hashlib
import uuid
from datetime import datetime, timezone


def generate_id() -> str:
    """Generate a unique memory ID."""
    return uuid.uuid4().hex[:16]


def content_hash(text: str) -> str:
    """MD5 hash of text content for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def now_utc() -> datetime:
    """Current UTC datetime."""
    return datetime.now(timezone.utc)


def format_messages(messages) -> str:
    """Convert various message formats to a single text string.

    Accepts:
    - str: returned as-is
    - list[dict]: each dict has 'role' and 'content' keys
    - list[str]: joined with newlines
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts)
    return str(messages)
