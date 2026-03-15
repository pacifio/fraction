"""SQLite-backed history tracking for memory operations.

Mirrors mem0's memory/storage.py pattern: tracks ADD, UPDATE, DELETE events
with timestamps and content snapshots.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class MemoryHistory(Base):
    __tablename__ = "memory_history"

    id = Column(String, primary_key=True)
    memory_id = Column(String, index=True, nullable=False)
    event = Column(String, nullable=False)  # ADD, UPDATE, DELETE
    old_content = Column(Text, nullable=True)
    new_content = Column(Text, nullable=True)
    user_id = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SQLiteHistory:
    """Track memory change history in SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def add_event(
        self,
        event_id: str,
        memory_id: str,
        event: str,
        old_content: str = None,
        new_content: str = None,
        user_id: str = None,
    ):
        """Record a memory operation event."""
        with self._Session() as session:
            entry = MemoryHistory(
                id=event_id,
                memory_id=memory_id,
                event=event,
                old_content=old_content,
                new_content=new_content,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
            )
            session.add(entry)
            session.commit()

    def get_history(self, memory_id: str) -> list[dict]:
        """Get change history for a specific memory."""
        with self._Session() as session:
            entries = (
                session.query(MemoryHistory)
                .filter(MemoryHistory.memory_id == memory_id)
                .order_by(MemoryHistory.timestamp.asc())
                .all()
            )
            return [
                {
                    "id": e.id,
                    "memory_id": e.memory_id,
                    "event": e.event,
                    "old_content": e.old_content,
                    "new_content": e.new_content,
                    "user_id": e.user_id,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                }
                for e in entries
            ]

    def reset(self):
        """Clear all history."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        # Remove file
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
            Base.metadata.create_all(self.engine)
            self._Session = sessionmaker(bind=self.engine)
