from __future__ import annotations

from dataclasses import dataclass
import logging

from ..config import Settings, load_settings
from ..db.pgvector_client import PgVectorClient, PgVectorPool
from ..embeddings.encoder import EmbeddingProvider, SentenceTransformerEmbeddingProvider


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
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
    rerank_score: float | None = None


class Retriever:
    def __init__(
        self,
        db_client: PgVectorClient,
        embedder: EmbeddingProvider,
        top_k_default: int,
        pool: PgVectorPool | None = None,
    ) -> None:
        self._db_client = db_client
        self._embedder = embedder
        self._top_k_default = top_k_default
        self._pool = pool

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        limit = top_k if top_k is not None else self._top_k_default
        if limit <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_vector = self._embedder.embed_query(query)
        rows = self._db_client.search(query_vector, limit)
        if not rows:
            return []

        results = [
            RetrievedChunk(
                id=row["id"],
                text=row.get("text", ""),
                source=row.get("source", ""),
                page=row.get("page", 0) or 0,
                section=row.get("section", "") or "",
                section_title=row.get("section_title", "") or "",
                chunk_type=row.get("chunk_type", "") or "",
                spec_number=row.get("spec_number", "") or "",
                series_id=row.get("series_id", "") or "",
                similarity=float(row.get("similarity", 0.0) or 0.0),
                rerank_score=None,
            )
            for row in rows
        ]
        logger.debug("Retrieved %s chunks for query: %s", len(results), query)
        return results

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()


def build_retriever(settings: Settings | None = None) -> Retriever:
    resolved = settings or load_settings()
    pool = PgVectorPool(resolved)
    db_client = PgVectorClient(pool, resolved.table_name)
    embedder = SentenceTransformerEmbeddingProvider(
        resolved.embedding_model_name,
        resolved.embedding_normalize,
    )
    return Retriever(
        db_client=db_client,
        embedder=embedder,
        top_k_default=resolved.retrieval_top_k,
        pool=pool,
    )
