from __future__ import annotations

from typing import Protocol

from sentence_transformers import SentenceTransformer


class EmbeddingProvider(Protocol):
    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str, normalize: bool) -> None:
        self._model_name = model_name
        self._normalize = normalize
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        try:
            model = self._get_model()
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self._normalize,
            )
            return vectors.tolist()
        except Exception as exc:
            raise RuntimeError(f"Embedding failed: {exc}") from exc

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embed([query], batch_size=1)
        if not vectors:
            raise RuntimeError("Embedding returned empty result for query.")
        return vectors[0]
