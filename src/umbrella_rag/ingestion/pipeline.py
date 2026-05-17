from __future__ import annotations

from typing import Any

from ..config import Settings, load_settings
from ..db.pgvector_client import PgVectorClient, PgVectorPool
from ..embeddings.encoder import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from .chunker import chunk_pages
from .parser import parse_pdf


def embed_chunks(
    chunks: list[dict[str, Any]],
    embedder: EmbeddingProvider,
    batch_size: int,
    show_progress: bool,
) -> list[list[float]]:
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        return []
    return embedder.embed(texts, batch_size=batch_size, show_progress=show_progress)


def run_ingestion(
    pdf_path: str,
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
    db_client: PgVectorClient | None = None,
    pool: PgVectorPool | None = None,
) -> int:
    resolved = settings or load_settings()

    created_pool = False
    if db_client is None:
        if pool is None:
            pool = PgVectorPool(resolved)
            created_pool = True
        db_client = PgVectorClient(pool, resolved.table_name)

    if embedder is None:
        embedder = SentenceTransformerEmbeddingProvider(
            resolved.embedding_model_name,
            resolved.embedding_normalize,
        )

    pages, skipped = parse_pdf(pdf_path)
    print(f"Parsed {len(pages)} useful pages, skipped {skipped} pages")

    chunks = chunk_pages(
        pages,
        chunk_size=resolved.chunk_size_words,
        overlap=resolved.chunk_overlap_words,
        min_remaining_words=resolved.chunk_min_remainder_words,
    )
    print(f"Chunks created: {len(chunks)}")

    embeddings = embed_chunks(
        chunks,
        embedder=embedder,
        batch_size=resolved.embedding_batch_size,
        show_progress=resolved.embedding_show_progress,
    )
    print(f"Embeddings created: {len(embeddings)}")

    if not embeddings:
        print("No embeddings created; skipping database insert.")
        return 0

    db_client.insert_chunks(chunks, embeddings)
    print("Storage complete.")

    if created_pool and pool is not None:
        pool.close()

    return len(chunks)
