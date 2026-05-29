from __future__ import annotations

import logging
from typing import Protocol

import torch
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


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

    def reset(self) -> None:
        raise NotImplementedError


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str, normalize: bool) -> None:
        self._model_name = model_name
        self._normalize = normalize
        self._model: SentenceTransformer | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Embedding device: %s", self._device)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            logger.info("Loaded embedding model %s on %s", self._model_name, self._device)
        return self._model

    def embed(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed a list of texts into vectors.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size for encoding. Uses model default if None.
            show_progress: Whether to show a progress bar.

        Returns:
            List of embedding vectors as float lists.

        Raises:
            RuntimeError: If encoding fails.

        Notes:
            Runs on GPU if CUDA is available, CPU otherwise.
            Model is lazy-loaded on first call.
        """
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
        """
        Embed a single query string.

        Args:
            query: Query string to embed.

        Returns:
            Embedding vector as a float list.

        Raises:
            RuntimeError: If embedding returns empty result.

        Notes:
            Convenience wrapper around embed() for single queries.
        """
        vectors = self.embed([query], batch_size=1)
        if not vectors:
            raise RuntimeError("Embedding returned empty result for query.")
        return vectors[0]

    def reset(self) -> None:
        """
        Release the model from GPU memory and reset the CUDA context.

        Args:
            None.

        Returns:
            None.

        Notes:
            Intended to be called between documents in batch ingestion.
        """
        if self._model is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            logger.info("Embedding model released and CUDA context reset")