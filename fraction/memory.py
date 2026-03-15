"""Core Fraction class — the public API.

Pipeline at write time:
  messages -> LLMLingua-2 compress -> entity extract -> embed -> USearch + graph

Pipeline at read time:
  query -> embed -> hybrid retrieval (vector + BM25 + graph) -> RRF rerank -> results

Zero LLM calls. Sub-100ms ingestion. Deterministic extraction.
"""

import time
from datetime import datetime, timezone

from fraction.compressor import build_compressor
from fraction.config import FractionConfig
from fraction.embedder import SentenceTransformerEmbedder
from fraction.entity import EntityExtractor
from fraction.graph import EntityGraph
from fraction.retriever import HybridRetriever
from fraction.storage import SQLiteHistory
from fraction.types import MemoryItem, SearchResult
from fraction.utils import content_hash, format_messages, generate_id, now_utc
from fraction.vector_store import USearchVectorStore


_GATE_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'it', 'its',
    'this', 'that', 'and', 'or', 'but', 'not', 'so', 'if', 'as',
    'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
    'they', 'them', 'their', 'am', 'are', 'yes', 'no', 'yeah',
    'okay', 'ok', 'oh', 'well', 'just', 'like', 'really', 'very',
    'too', 'also', 'then', 'than', 'more', 'much', 'now', 'here',
    'there', 'what', 'how', 'why', 'who', 'which', 'when', 'where',
    'all', 'any', 'some', 'get', 'got', 'know', 'think', 'right',
    'good', 'great', 'nice', 'sure', 'thanks', 'thank', 'sounds',
    'wow', 'hmm', 'haha', 'lol', 'hey', 'hi', 'hello', 'bye',
})


class Fraction:
    """Fractional memory chain for long-term LLM context management.

    Usage:
        from fraction import Fraction

        f = Fraction()
        f.add("I love hiking in the Rocky Mountains.", user_id="alice")
        results = f.search("outdoor activities", user_id="alice")
    """

    def __init__(self, config: FractionConfig = None):
        self.config = config or FractionConfig()
        self.compressor = build_compressor(self.config)
        self.embedder = SentenceTransformerEmbedder(self.config.embedder_model)

        # Update embedding dim from actual model if needed
        actual_dim = self.embedder.dimension
        if self.config.embedding_dim != actual_dim:
            self.config.embedding_dim = actual_dim

        self.vector_store = USearchVectorStore(self.config)
        self.entity_extractor = EntityExtractor()
        self.graph = EntityGraph()
        self._history_store = SQLiteHistory(self.config.history_db_path)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            entity_graph=self.graph,
            entity_extractor=self.entity_extractor,
            config=self.config,
        )
        # Paths for graph persistence
        self._graph_path = self.config.vector_store_path.replace(".usearch", "_graph.json")

    def add(
        self,
        messages,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        metadata: dict = None,
    ) -> dict:
        """Add memories from messages.

        Args:
            messages: str, list[str], or list[dict] with role/content keys
            user_id: scope to user
            agent_id: scope to agent
            run_id: scope to conversation run
            metadata: additional key-value metadata

        Returns:
            {"results": [{"id": str, "memory": str, "event": "ADD"|"UPDATE"}]}
        """
        t_start = time.perf_counter()
        text = format_messages(messages)
        user_id = user_id or self.config.default_user_id
        metadata = metadata or {}

        # 1. Compress text — THE KEY STEP (replaces mem0's LLM extraction)
        compressed = self.compressor.compress(
            text, rate=self.config.compression_rate, adaptive=self.config.adaptive_compression,
        )
        memory_text = compressed.compressed_text

        # 2. Extract entities (spaCy NER — no LLM)
        entities = self.entity_extractor.extract(memory_text)
        entity_names = [e["text"] for e in entities]

        # 2.5 Relevance gate — skip filler turns with no entities and low content
        if self.config.relevance_gate and not entity_names:
            content_words = [
                w for w in memory_text.split()
                if len(w) > 2 and w.lower() not in _GATE_STOPWORDS
            ]
            if len(content_words) < self.config.min_content_words:
                return {"results": [{"id": generate_id(), "memory": memory_text, "event": "SKIP"}]}

        # 3. Generate embedding
        embedding = self.embedder.embed(memory_text)

        # 4. Check for duplicates (cosine > threshold = duplicate)
        existing = self.vector_store.search(
            embedding, limit=1, filters={"user_id": user_id}
        )

        event = "ADD"
        memory_id = generate_id()
        old_content = None

        if existing and existing[0].score >= self.config.duplicate_threshold:
            # UPDATE existing memory
            event = "UPDATE"
            memory_id = existing[0].id
            old_content = existing[0].content
            # Merge: keep longer/more informative version
            if len(memory_text) > len(old_content):
                # New content is more informative — update
                pass
            else:
                # Existing is fine — skip
                return {"results": [{"id": memory_id, "memory": old_content, "event": "NOOP"}]}

        now = now_utc()
        h = content_hash(memory_text)

        # 5. Build payload
        payload = {
            "id": memory_id,
            "content": memory_text,
            "content_raw": text,
            "entities": entity_names,
            "importance_scores": [s for _, s in compressed.token_scores],
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "hash": h,
            **metadata,
        }

        # 6. Store in vector store
        if event == "UPDATE":
            self.vector_store.update(memory_id, embedding, payload)
            self.retriever.remove_from_corpus(memory_id)
            self.graph.remove_memory(memory_id)
        else:
            self.vector_store.insert(embedding, payload)

        # 7. Update BM25 corpus (index raw text for better keyword matching)
        self.retriever.add_to_corpus(memory_id, text)

        # 8. Update entity graph
        entity_node_ids = []
        for ent in entities:
            nid = self.graph.add_entity(ent["text"], ent["label"], memory_id)
            entity_node_ids.append(nid)
        # Add edges between co-occurring entities
        for i in range(len(entity_node_ids)):
            for j in range(i + 1, len(entity_node_ids)):
                self.graph.add_relationship(
                    entity_node_ids[i], entity_node_ids[j],
                    "co_occurs", memory_id,
                )
        # Add relationship triples from dependency parse
        triples = self.entity_extractor.extract_relationships(memory_text)
        for subj, rel, obj in triples:
            subj_id = self.graph.find_entity(subj)
            obj_id = self.graph.find_entity(obj)
            if subj_id and obj_id:
                self.graph.add_relationship(subj_id, obj_id, rel, memory_id)

        # 9. Log event
        self._history_store.add_event(
            event_id=generate_id(),
            memory_id=memory_id,
            event=event,
            old_content=old_content,
            new_content=memory_text,
            user_id=user_id,
        )

        t_elapsed = (time.perf_counter() - t_start) * 1000

        return {
            "results": [{
                "id": memory_id,
                "memory": memory_text,
                "event": event,
                "entities": entity_names,
                "compression_ratio": compressed.compression_ratio,
                "latency_ms": round(t_elapsed, 2),
            }]
        }

    def search(
        self,
        query: str,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        limit: int = None,
        filters: dict = None,
    ) -> dict:
        """Search for relevant memories.

        Returns:
            {"results": [{"id": str, "memory": str, "score": float, ...}]}
        """
        t_start = time.perf_counter()
        user_id = user_id or self.config.default_user_id
        limit = limit or self.config.top_k

        scope = dict(filters) if filters else {}
        if user_id:
            scope["user_id"] = user_id
        if agent_id:
            scope["agent_id"] = agent_id
        if run_id:
            scope["run_id"] = run_id

        results = self.retriever.retrieve(query, user_id=user_id, limit=limit, filters=scope)

        t_elapsed = (time.perf_counter() - t_start) * 1000

        return {
            "results": [
                {
                    "id": r.id,
                    "memory": r.content,
                    "score": round(r.score, 4),
                    "metadata": r.metadata,
                    "created_at": r.created_at,
                }
                for r in results
            ],
            "latency_ms": round(t_elapsed, 2),
        }

    def get(self, memory_id: str) -> dict | None:
        """Get a single memory by ID."""
        payload = self.vector_store.get(memory_id)
        if not payload:
            return None
        return {
            "id": payload.get("id"),
            "memory": payload.get("content"),
            "metadata": payload,
            "created_at": payload.get("created_at"),
        }

    def get_all(
        self,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        limit: int = 100,
    ) -> dict:
        """List all memories with optional scope filtering."""
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        items = self.vector_store.list_all(filters=filters if filters else None, limit=limit)
        return {
            "results": [
                {
                    "id": item.get("id"),
                    "memory": item.get("content"),
                    "metadata": item,
                    "created_at": item.get("created_at"),
                }
                for item in items
            ]
        }

    def update(self, memory_id: str, data: str) -> dict:
        """Update a memory's content. Re-compresses and re-embeds."""
        existing = self.vector_store.get(memory_id)
        if not existing:
            return {"error": "Memory not found", "id": memory_id}

        old_content = existing.get("content", "")

        # Re-compress
        compressed = self.compressor.compress(
            data, rate=self.config.compression_rate, adaptive=self.config.adaptive_compression,
        )
        new_content = compressed.compressed_text

        # Re-extract entities
        entities = self.entity_extractor.extract(new_content)
        entity_names = [e["text"] for e in entities]

        # Re-embed
        embedding = self.embedder.embed(new_content)

        now = now_utc()
        payload = {
            **existing,
            "content": new_content,
            "content_raw": data,
            "entities": entity_names,
            "updated_at": now.isoformat(),
            "hash": content_hash(new_content),
        }

        self.vector_store.update(memory_id, embedding, payload)

        # Update retriever corpus (index raw text)
        self.retriever.remove_from_corpus(memory_id)
        self.retriever.add_to_corpus(memory_id, data)

        # Update graph
        self.graph.remove_memory(memory_id)
        for ent in entities:
            self.graph.add_entity(ent["text"], ent["label"], memory_id)

        # Log event
        self._history_store.add_event(
            event_id=generate_id(),
            memory_id=memory_id,
            event="UPDATE",
            old_content=old_content,
            new_content=new_content,
            user_id=existing.get("user_id"),
        )

        return {
            "id": memory_id,
            "memory": new_content,
            "event": "UPDATE",
        }

    def delete(self, memory_id: str) -> dict:
        """Delete a memory by ID."""
        existing = self.vector_store.get(memory_id)
        if not existing:
            return {"error": "Memory not found", "id": memory_id}

        self.vector_store.delete(memory_id)
        self.retriever.remove_from_corpus(memory_id)
        self.graph.remove_memory(memory_id)

        self._history_store.add_event(
            event_id=generate_id(),
            memory_id=memory_id,
            event="DELETE",
            old_content=existing.get("content"),
            user_id=existing.get("user_id"),
        )

        return {"id": memory_id, "event": "DELETE"}

    def delete_all(
        self,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
    ) -> dict:
        """Delete all memories matching scope."""
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        items = self.vector_store.list_all(filters=filters if filters else None, limit=10000)
        count = 0
        for item in items:
            mid = item.get("id")
            if mid:
                self.vector_store.delete(mid)
                self.retriever.remove_from_corpus(mid)
                self.graph.remove_memory(mid)
                count += 1

        return {"deleted": count}

    def history(self, memory_id: str) -> list[dict]:
        """Get change history for a memory."""
        return self._history_store.get_history(memory_id)

    def save(self):
        """Persist all state to disk."""
        self.vector_store.save()
        self.graph.save(self._graph_path)

    def load(self):
        """Load persisted state from disk."""
        self.vector_store.load()
        self.graph.load(self._graph_path)
        # Rebuild BM25 corpus from stored memories (prefer raw text)
        all_items = self.vector_store.list_all(limit=100000)
        for item in all_items:
            mid = item.get("id")
            content = item.get("content_raw") or item.get("content", "")
            if mid and content:
                self.retriever.add_to_corpus(mid, content)

    def reset(self):
        """Clear all data."""
        self.vector_store.reset()
        self.graph.reset()
        self._history_store.reset()
        self.retriever._bm25_corpus.clear()
        self.retriever._bm25_dirty = True
