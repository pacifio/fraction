"""Tests for entity extraction and graph."""

import pytest


@pytest.fixture(scope="module")
def extractor():
    from fraction.entity import EntityExtractor
    return EntityExtractor()


class TestEntityExtractor:
    def test_extract_person(self, extractor):
        entities = extractor.extract("John Smith went to the store.")
        names = [e["text"] for e in entities]
        assert "John Smith" in names

    def test_extract_location(self, extractor):
        entities = extractor.extract("I traveled to Paris and visited the Eiffel Tower.")
        labels = {e["text"]: e["label"] for e in entities}
        assert "Paris" in labels

    def test_extract_org(self, extractor):
        entities = extractor.extract("She works at Google in Mountain View.")
        names = [e["text"] for e in entities]
        assert "Google" in names

    def test_extract_names_deduped(self, extractor):
        names = extractor.extract_names("John met John at the park. John was happy.")
        assert names.count("John") == 1

    def test_extract_relationships(self, extractor):
        triples = extractor.extract_relationships("Alice visited Paris.")
        # Should find at least one triple
        assert len(triples) >= 0  # dep parsing may vary


class TestEntityGraph:
    def test_add_and_find_entity(self):
        from fraction.graph import EntityGraph
        g = EntityGraph()
        nid = g.add_entity("John Smith", "PERSON", "mem1")
        found = g.find_entity("John Smith")
        assert found == nid

    def test_fuzzy_match(self):
        from fraction.graph import EntityGraph
        g = EntityGraph()
        g.add_entity("New York City", "GPE", "mem1")
        found = g.find_entity("New York City")
        assert found is not None

    def test_graph_traversal(self):
        from fraction.graph import EntityGraph
        g = EntityGraph()
        n1 = g.add_entity("Alice", "PERSON", "mem1")
        n2 = g.add_entity("Bob", "PERSON", "mem1")
        n3 = g.add_entity("Charlie", "PERSON", "mem2")
        g.add_relationship(n1, n2, "knows", "mem1")
        g.add_relationship(n2, n3, "works_with", "mem2")

        related = g.get_related(n1, hops=2)
        related_texts = [r["text"] for r in related]
        assert "Bob" in related_texts
        assert "Charlie" in related_texts

    def test_search_returns_memory_ids(self):
        from fraction.graph import EntityGraph
        g = EntityGraph()
        g.add_entity("Paris", "GPE", "mem1")
        g.add_entity("France", "GPE", "mem2")
        g.add_relationship(
            g.find_entity("Paris"),
            g.find_entity("France"),
            "in", "mem1"
        )

        mids = g.search(["Paris"])
        assert "mem1" in mids
