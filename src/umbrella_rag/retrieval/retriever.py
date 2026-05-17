from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, load_settings
from ..db.pgvector_client import PgVectorClient, PgVectorPool
from ..embeddings.encoder import EmbeddingProvider, SentenceTransformerEmbeddingProvider


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    text: str
    source: str
    page: int
    section: str | None
    similarity: float


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

        return [
            RetrievedChunk(
                id=row["id"],
                text=row["text"],
                source=row["source"],
                page=row["page"],
                section=row["section"],
                similarity=row["similarity"],
            )
            for row in rows
        ]

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
