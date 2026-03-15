"""Entity extraction using spaCy NER. No LLM calls.

Replaces mem0's LLM-based entity extraction in graph_memory.py with
deterministic, free, sub-10ms algorithmic extraction.
"""


class EntityExtractor:
    """Extract named entities and relationships using spaCy."""

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(model)

    def extract(self, text: str) -> list[dict]:
        """Extract named entities with types.

        Returns:
            [{"text": "Paris", "label": "GPE", "start": 10, "end": 15}, ...]
        """
        doc = self.nlp(text)
        entities = []
        seen = set()
        for ent in doc.ents:
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
        return entities

    def extract_names(self, text: str) -> list[str]:
        """Extract just entity text strings (deduplicated)."""
        return [e["text"] for e in self.extract(text)]

    def extract_relationships(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (subject, relation, object) triples using dependency parsing.

        Uses spaCy dependency parse to find nsubj -> verb -> dobj patterns.
        """
        doc = self.nlp(text)
        triples = []
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [
                    child for child in token.children
                    if child.dep_ in ("nsubj", "nsubjpass")
                ]
                objects = [
                    child for child in token.children
                    if child.dep_ in ("dobj", "attr", "prep")
                ]
                for subj in subjects:
                    # Get full subject span (including compound nouns)
                    subj_text = self._get_span_text(subj)
                    for obj in objects:
                        if obj.dep_ == "prep":
                            # Follow preposition to its object
                            for pobj in obj.children:
                                if pobj.dep_ == "pobj":
                                    obj_text = self._get_span_text(pobj)
                                    rel = f"{token.lemma_} {obj.text}"
                                    triples.append((subj_text, rel, obj_text))
                        else:
                            obj_text = self._get_span_text(obj)
                            triples.append((subj_text, token.lemma_, obj_text))
        return triples

    @staticmethod
    def _get_span_text(token) -> str:
        """Get the full text of a token including its compound children."""
        parts = []
        for child in token.children:
            if child.dep_ in ("compound", "amod", "det"):
                parts.append(child.text)
        parts.append(token.text)
        return " ".join(parts)
