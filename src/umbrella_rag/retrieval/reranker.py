from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol

from sentence_transformers import CrossEncoder

from ..config import Settings, load_settings
from .retriever import RetrievedChunk


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RerankedChunk:
    id: int
    text: str
    source: str
    page: int
    section: str
    section_title: str
    chunk_type: str
    spec_number: str
    series_id: str
    similarity: float
    rerank_score: float


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RerankedChunk]:
        raise NotImplementedError


class CrossEncoderReranker:
    def __init__(self, model_name: str, top_k_default: int) -> None:
        self._model_name = model_name
        self._top_k_default = top_k_default
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RerankedChunk]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not chunks:
            return []

        limit = top_k if top_k is not None else self._top_k_default
        if limit <= 0:
            raise ValueError("top_k must be a positive integer.")

        pairs = [(query, chunk.text) for chunk in chunks]
        try:
            scores = self._get_model().predict(pairs)
        except Exception as exc:
            raise RuntimeError(f"Reranker failed: {exc}") from exc

        reranked = [
            RerankedChunk(
                id=chunk.id,
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                section=chunk.section or "",
                section_title=chunk.section_title,
                chunk_type=chunk.chunk_type,
                spec_number=chunk.spec_number,
                series_id=chunk.series_id,
                similarity=chunk.similarity,
                rerank_score=float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)

        RERANK_CONFIDENCE_THRESHOLD = -3.0
        if reranked and reranked[0].rerank_score < RERANK_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low reranker confidence (top_score=%.2f, query=%r) — "
                "falling back to similarity ordering",
                reranked[0].rerank_score,
                query[:80],
            )
            reranked.sort(key=lambda item: item.similarity, reverse=True)

        return reranked[:limit]


def build_reranker(settings: Settings | None = None) -> CrossEncoderReranker:
    resolved = settings or load_settings()
    return CrossEncoderReranker(
        model_name=resolved.reranker_model_name,
        top_k_default=resolved.reranker_top_k,
    )
