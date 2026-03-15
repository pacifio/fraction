"""Embedding layer for Fraction. Local-only, no API calls."""

from abc import ABC, abstractmethod


class EmbedderBase(ABC):
    """Protocol for text embedding."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        ...


class SentenceTransformerEmbedder(EmbedderBase):
    """Default embedder using sentence-transformers. Runs locally on CPU."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension
