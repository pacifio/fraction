"""Lightweight in-memory entity relationship graph.

No external graph database required. Stores entity nodes and relationship edges
for multi-hop traversal during retrieval. Persists as JSON alongside USearch index.
"""

import json
import os
from collections import defaultdict
from difflib import SequenceMatcher

from fraction.utils import generate_id


class EntityGraph:
    """In-memory entity relationship graph."""

    def __init__(self):
        # node_id -> {"text": str, "type": str, "memory_ids": set, "aliases": set}
        self.nodes: dict[str, dict] = {}
        # list of {"source": id, "target": id, "relation": str, "memory_id": str}
        self.edges: list[dict] = []
        # Fast lookup: entity text (lowered) -> node_id
        self._text_to_id: dict[str, str] = {}
        # Adjacency list for fast traversal
        self._adjacency: dict[str, set[str]] = defaultdict(set)

    def add_entity(self, text: str, entity_type: str, memory_id: str) -> str:
        """Add or merge an entity node. Returns node ID."""
        # Try to find existing entity (exact or fuzzy match)
        existing_id = self.find_entity(text)
        if existing_id:
            self.nodes[existing_id]["memory_ids"].add(memory_id)
            self.nodes[existing_id]["aliases"].add(text)
            return existing_id

        node_id = generate_id()
        self.nodes[node_id] = {
            "text": text,
            "type": entity_type,
            "memory_ids": {memory_id},
            "aliases": {text},
        }
        self._text_to_id[text.lower()] = node_id
        return node_id

    def add_relationship(self, source_id: str, target_id: str, relation: str, memory_id: str):
        """Add a directed edge between two entity nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        self.edges.append({
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "memory_id": memory_id,
        })
        self._adjacency[source_id].add(target_id)
        self._adjacency[target_id].add(source_id)

    def find_entity(self, text: str, threshold: float = 0.85) -> str | None:
        """Find existing entity by exact or fuzzy text match."""
        lower = text.lower()
        # Exact match
        if lower in self._text_to_id:
            return self._text_to_id[lower]
        # Fuzzy match
        for existing_text, node_id in self._text_to_id.items():
            if SequenceMatcher(None, lower, existing_text).ratio() >= threshold:
                return node_id
        return None

    def get_related(self, entity_id: str, hops: int = 2) -> list[dict]:
        """BFS traversal up to N hops. Returns connected nodes with distance."""
        if entity_id not in self.nodes:
            return []

        visited = {entity_id: 0}
        queue = [entity_id]
        results = []

        while queue:
            current = queue.pop(0)
            current_dist = visited[current]
            if current_dist >= hops:
                continue

            for neighbor in self._adjacency.get(current, set()):
                if neighbor not in visited:
                    visited[neighbor] = current_dist + 1
                    queue.append(neighbor)
                    results.append({
                        "node_id": neighbor,
                        "text": self.nodes[neighbor]["text"],
                        "type": self.nodes[neighbor]["type"],
                        "distance": current_dist + 1,
                        "memory_ids": list(self.nodes[neighbor]["memory_ids"]),
                    })

        return results

    def search(self, query_entities: list[str], limit: int = 5) -> list[str]:
        """Find memory IDs connected to query entities via graph traversal."""
        memory_ids = set()
        for entity_text in query_entities:
            node_id = self.find_entity(entity_text)
            if not node_id:
                continue
            # Direct memories
            memory_ids.update(self.nodes[node_id]["memory_ids"])
            # Related memories (1-2 hops)
            related = self.get_related(node_id, hops=2)
            for r in related:
                memory_ids.update(r["memory_ids"])
            if len(memory_ids) >= limit:
                break
        return list(memory_ids)[:limit]

    def get_memory_entities(self, memory_id: str) -> list[dict]:
        """Get all entities associated with a memory."""
        return [
            {"node_id": nid, "text": node["text"], "type": node["type"]}
            for nid, node in self.nodes.items()
            if memory_id in node["memory_ids"]
        ]

    def remove_memory(self, memory_id: str):
        """Remove a memory's associations from the graph."""
        # Remove from nodes
        empty_nodes = []
        for nid, node in self.nodes.items():
            node["memory_ids"].discard(memory_id)
            if not node["memory_ids"]:
                empty_nodes.append(nid)
        # Clean up orphaned nodes
        for nid in empty_nodes:
            text = self.nodes[nid]["text"].lower()
            del self.nodes[nid]
            if text in self._text_to_id and self._text_to_id[text] == nid:
                del self._text_to_id[text]
            if nid in self._adjacency:
                del self._adjacency[nid]
            for adj_set in self._adjacency.values():
                adj_set.discard(nid)
        # Remove edges
        self.edges = [e for e in self.edges if e["memory_id"] != memory_id]

    def save(self, path: str):
        """Persist graph to JSON."""
        # Convert sets to lists for JSON serialization
        nodes_serial = {}
        for nid, node in self.nodes.items():
            nodes_serial[nid] = {
                **node,
                "memory_ids": list(node["memory_ids"]),
                "aliases": list(node["aliases"]),
            }
        state = {
            "nodes": nodes_serial,
            "edges": self.edges,
            "text_to_id": self._text_to_id,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str):
        """Load graph from JSON."""
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            state = json.load(f)
        self.nodes = {}
        for nid, node in state.get("nodes", {}).items():
            self.nodes[nid] = {
                **node,
                "memory_ids": set(node["memory_ids"]),
                "aliases": set(node["aliases"]),
            }
        self.edges = state.get("edges", [])
        self._text_to_id = state.get("text_to_id", {})
        # Rebuild adjacency
        self._adjacency = defaultdict(set)
        for edge in self.edges:
            self._adjacency[edge["source"]].add(edge["target"])
            self._adjacency[edge["target"]].add(edge["source"])

    def reset(self):
        """Clear all graph data."""
        self.nodes.clear()
        self.edges.clear()
        self._text_to_id.clear()
        self._adjacency.clear()
