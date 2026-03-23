"""Embedding layer for Fraction. Local-only, no API calls."""

from abc import ABC, abstractmethod
import os


_MODEL_CACHE: dict[str, object] = {}
_DIMENSION_CACHE: dict[str, int] = {}


class _suppress_loader_noise:
    def __enter__(self):
        if os.environ.get("FRACTION_EMBEDDER_VERBOSE") == "1":
            self._active = False
            return
        self._active = True
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)
        self._null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null_fd, 1)
        os.dup2(self._null_fd, 2)

    def __exit__(self, exc_type, exc, tb):
        if not getattr(self, "_active", False):
            return
        try:
            os.dup2(self._stdout_fd, 1)
            os.dup2(self._stderr_fd, 2)
        finally:
            os.close(self._stdout_fd)
            os.close(self._stderr_fd)
            os.close(self._null_fd)


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

        cached_model = _MODEL_CACHE.get(model_name)
        cached_dimension = _DIMENSION_CACHE.get(model_name)
        if cached_model is None or cached_dimension is None:
            with _suppress_loader_noise():
                cached_model = SentenceTransformer(model_name)
            cached_dimension = cached_model.get_sentence_embedding_dimension()
            _MODEL_CACHE[model_name] = cached_model
            _DIMENSION_CACHE[model_name] = cached_dimension

        self.model = cached_model
        self._model_name = model_name
        self._dimension = cached_dimension

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension
