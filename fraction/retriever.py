"""Hybrid retriever: vector similarity + BM25 keyword + graph traversal with RRF reranking.

The March 2026 diagnostic study shows retrieval method is the dominant accuracy factor
(20-point spread across methods vs 3-8 across extraction strategies). This module is
where Fraction's real accuracy leverage lives.
"""

import re
from collections import defaultdict

from fraction.config import FractionConfig
from fraction.types import SearchResult


_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'it', 'its',
    'this', 'that', 'and', 'or', 'but', 'not', 'so', 'if', 'as',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
    'they', 'them', 'their', 'am', 'are',
})


def _tokenize(text: str) -> list[str]:
    """Tokenize text with punctuation removal and stopword filtering."""
    return [t for t in re.findall(r'\w+', text.lower()) if t not in _STOPWORDS]


_TEMPORAL_QUERY_PATTERN = re.compile(
    r'\b(when|what date|what time|how long|how many (?:days|weeks|months|years)|'
    r'since when|last time|first time|before|after)\b', re.IGNORECASE
)

_TEMPORAL_CONTENT_PATTERN = re.compile(
    r'(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|'  # YYYY-MM-DD
    r'(?:\d{1,2}[-/]\d{1,2}[-/]\d{4})|'  # MM/DD/YYYY
    r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+\d{1,2}(?:,?\s*\d{4})?)|'  # Month DD, YYYY
    r'(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+\d{4})?)|'  # DD Month YYYY
    r'(?:\b\d{4}\b)|'  # Standalone year
    r'(?:(?:last|next|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))|'
    r'(?:\d+\s+(?:days?|weeks?|months?|years?)\s+ago)',
    re.IGNORECASE
)


_QUERY_DATE_PATTERN = re.compile(
    r'(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|'  # YYYY-MM-DD
    r'(?:\d{1,2}[-/]\d{1,2}[-/]\d{4})|'  # MM/DD/YYYY
    r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+\d{1,2})?(?:,?\s*\d{4})?)|'  # Month [DD][, YYYY]
    r'(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:\s+\d{4})?)|'  # DD Month [YYYY]
    r'(?:\b\d{4}\b)',  # Standalone year
    re.IGNORECASE
)


def _extract_query_dates(query: str) -> list[str]:
    """Extract date/time references from a query string."""
    return [m.strip() for m in _QUERY_DATE_PATTERN.findall(query) if m.strip()]


class HybridRetriever:
    """Combines vector similarity, BM25 keyword search, and graph traversal."""

    def __init__(self, vector_store, embedder, entity_graph, entity_extractor, config: FractionConfig):
        self.vector_store = vector_store
        self.embedder = embedder
        self.graph = entity_graph
        self.entity_extractor = entity_extractor
        self.config = config
        # BM25 corpus: memory_id -> tokenized content
        self._bm25_corpus: dict[str, list[str]] = {}
        self._bm25_index = None
        self._bm25_ids: list[str] = []
        self._bm25_dirty = True

    def add_to_corpus(self, memory_id: str, content: str):
        """Add a memory to the BM25 corpus."""
        self._bm25_corpus[memory_id] = _tokenize(content)
        self._bm25_dirty = True

    def remove_from_corpus(self, memory_id: str):
        """Remove a memory from the BM25 corpus."""
        self._bm25_corpus.pop(memory_id, None)
        self._bm25_dirty = True

    def _rebuild_bm25(self):
        """Rebuild BM25 index from corpus."""
        if not self._bm25_dirty:
            return
        from rank_bm25 import BM25Okapi
        self._bm25_ids = list(self._bm25_corpus.keys())
        corpus = [self._bm25_corpus[mid] for mid in self._bm25_ids]
        if corpus:
            self._bm25_index = BM25Okapi(corpus)
        else:
            self._bm25_index = None
        self._bm25_dirty = False

    def retrieve(
        self,
        query: str,
        user_id: str = None,
        limit: int = 10,
        filters: dict = None,
    ) -> list[SearchResult]:
        """Hybrid retrieval with RRF reranking across all signals.

        1. Embed query -> vector search (USearch)
        2. Tokenize query -> BM25 search
        3. Extract entities from query -> graph traversal
        4. Reciprocal Rank Fusion across all result sets
        5. Return top-k merged results
        """
        # Build scope filters
        scope_filters = dict(filters) if filters else {}
        if user_id:
            scope_filters["user_id"] = user_id

        result_lists = []
        signal_weights = []
        weights_cfg = self.config.rrf_weights or {}

        # 1. Vector search
        query_embedding = self.embedder.embed(query)
        vector_results = self._vector_search(query_embedding, limit * 2, scope_filters)
        if vector_results:
            result_lists.append(vector_results)
            signal_weights.append(weights_cfg.get("vector", 1.0))

        # 2. BM25 search
        if self.config.use_bm25 and self._bm25_corpus:
            bm25_results = self._bm25_search(query, limit * 2, scope_filters)
            if bm25_results:
                result_lists.append(bm25_results)
                signal_weights.append(weights_cfg.get("bm25", 1.2))

        # 3. Graph search
        if self.config.use_graph:
            graph_results = self._graph_search(query, limit * 2, scope_filters)
            if graph_results:
                result_lists.append(graph_results)
                signal_weights.append(weights_cfg.get("graph", 0.8))

        # 4. Temporal boost — when query asks about time, boost date-rich memories
        if _TEMPORAL_QUERY_PATTERN.search(query):
            temporal_results = self._temporal_search(query, limit * 2, scope_filters)
            if temporal_results:
                result_lists.append(temporal_results)
                signal_weights.append(weights_cfg.get("temporal", 1.0))

        if not result_lists:
            return []

        # 5. RRF reranking
        if self.config.rerank and len(result_lists) > 1:
            merged = self.reciprocal_rank_fusion(
                result_lists, k=self.config.rrf_k, weights=signal_weights,
            )
        else:
            # Single signal — just use it directly
            merged = result_lists[0]

        # 5. Apply score-based cutoff to filter low-relevance results
        if merged and self.config.min_score_ratio > 0:
            top_score = merged[0][1]
            threshold = top_score * self.config.min_score_ratio
            merged = [(mid, s) for mid, s in merged if s >= threshold]

        # 6. Build final SearchResult objects
        final = []
        seen_ids = set()
        for memory_id, score in merged[:limit]:
            if memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)
            payload = self.vector_store.get(memory_id)
            if payload:
                final.append(SearchResult(
                    id=memory_id,
                    content=payload.get("content", ""),
                    score=score,
                    metadata=payload,
                    created_at=payload.get("created_at"),
                ))
        return final

    def _vector_search(
        self, query_embedding: list[float], limit: int, filters: dict
    ) -> list[tuple[str, float]]:
        """Search USearch index."""
        results = self.vector_store.search(query_embedding, limit=limit, filters=filters)
        return [(r.id, r.score) for r in results]

    def _bm25_search(
        self, query: str, limit: int, filters: dict
    ) -> list[tuple[str, float]]:
        """Search BM25 index."""
        self._rebuild_bm25()
        if not self._bm25_index or not self._bm25_ids:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25_index.get_scores(query_tokens)

        # Pair with IDs and filter
        scored = []
        for i, (mid, score) in enumerate(zip(self._bm25_ids, scores)):
            if score <= 0:
                continue
            # Apply scope filter
            if filters:
                payload = self.vector_store.get(mid)
                if payload and not all(payload.get(k) == v for k, v in filters.items()):
                    continue
            scored.append((mid, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _graph_search(
        self, query: str, limit: int, filters: dict
    ) -> list[tuple[str, float]]:
        """Search entity graph for memories connected to query entities.

        Scores by: number of matching query entities and hop distance.
        Direct entity matches score higher than multi-hop connections.
        """
        query_entities = self.entity_extractor.extract_names(query)
        if not query_entities:
            return []

        # Collect memory scores: weight by entity match count and hop distance
        memory_scores: dict[str, float] = defaultdict(float)
        for entity_text in query_entities:
            node_id = self.graph.find_entity(entity_text)
            if not node_id:
                continue
            # Direct memories from this entity (distance 0 -> score 1.0)
            for mid in self.graph.nodes[node_id]["memory_ids"]:
                memory_scores[mid] += 1.0
            # Related memories via graph traversal (distance 1 -> 0.5, distance 2 -> 0.33)
            related = self.graph.get_related(node_id, hops=2)
            for r in related:
                hop_score = 1.0 / (1 + r["distance"])
                for mid in r["memory_ids"]:
                    memory_scores[mid] += hop_score

        # Apply scope filters and build scored list
        scored = []
        for mid, score in memory_scores.items():
            if filters:
                payload = self.vector_store.get(mid)
                if payload and not all(payload.get(k) == v for k, v in filters.items()):
                    continue
            scored.append((mid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _temporal_search(
        self, query: str, limit: int, filters: dict
    ) -> list[tuple[str, float]]:
        """Rank memories by temporal relevance to the query.

        Two-tier scoring:
        - Memories containing dates/times that match the query get a high score (2.0 per match)
        - Memories with other temporal patterns get a low density-based score
        """
        # Extract specific date references from the query for targeted matching
        query_dates = _extract_query_dates(query)

        scored = []
        all_items = self.vector_store.list_all(filters=filters, limit=limit * 5)
        for item in all_items:
            mid = item.get("id")
            if not mid:
                continue
            content = item.get("content_raw") or item.get("content", "")
            content_lower = content.lower()

            score = 0.0

            # High-value: exact date matches from query
            for date_str in query_dates:
                if date_str.lower() in content_lower:
                    score += 2.0

            # Low-value: general temporal pattern density
            matches = _TEMPORAL_CONTENT_PATTERN.findall(content)
            if matches:
                score += len(matches) * 0.1 / max(len(content.split()), 1)

            if score > 0:
                scored.append((mid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: list[list[tuple[str, float]]],
        k: int = 30,
        weights: list[float] = None,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion across multiple ranked lists.

        score(d) = sum(weight_i / (k + rank_i(d))) for each list i where d appears.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        if weights is None:
            weights = [1.0] * len(result_lists)

        for result_list, weight in zip(result_lists, weights):
            for rank, (doc_id, _) in enumerate(result_list):
                rrf_scores[doc_id] += weight / (k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
