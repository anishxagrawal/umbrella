from __future__ import annotations

import logging
from typing import Any

from ..config import Settings, load_settings
from ..db.pgvector_client import PgVectorClient, PgVectorPool
from ..embeddings.encoder import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from .chunker import chunk_pages
from .parser import parse_pdf


logger = logging.getLogger(__name__)


def embed_chunks(
    chunks: list[dict[str, Any]],
    embedder: EmbeddingProvider,
    batch_size: int,
    show_progress: bool,
) -> list[list[float]]:
    """
    Embed chunk texts using the provided embedding provider.

    Args:
        chunks: Chunk dicts containing a text field.
        embedder: Embedding provider instance.
        batch_size: Batch size to use for embedding.
        show_progress: Whether to display progress in the embedder.

    Returns:
        List of embedding vectors aligned with chunks.

    Notes:
        Returns an empty list if there are no chunk texts.
    """
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
    """
    Ingest a single PDF file into the vector database.

    Args:
        pdf_path: Absolute path to a PDF file to ingest.
        settings: Optional Settings instance. Loaded from env if not provided.
        embedder: Optional embedding provider. Created from settings if not provided.
        db_client: Optional PgVectorClient instance.
        pool: Optional PgVectorPool instance.

    Returns:
        Number of chunks ingested.

    Notes:
        Creates pool and embedder only if not provided.
    """
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
    logger.info("Parsed %s useful pages, skipped %s pages", len(pages), skipped)

    chunks = chunk_pages(
        pages,
        chunk_size=resolved.chunk_size_words,
        overlap=resolved.chunk_overlap_words,
        min_remaining_words=resolved.chunk_min_remainder_words,
    )
    for chunk in chunks:
        for key, value in list(chunk.items()):
            if isinstance(value, str):
                chunk[key] = value.replace("\x00", " ")
    logger.info("Chunks created: %s", len(chunks))

    embeddings = embed_chunks(
        chunks,
        embedder=embedder,
        batch_size=resolved.embedding_batch_size,
        show_progress=resolved.embedding_show_progress,
    )
    logger.info("Embeddings created: %s", len(embeddings))

    if not embeddings:
        logger.warning("No embeddings created; skipping database insert.")
        return 0

    db_client.insert_chunks(chunks, embeddings)
    logger.info("Storage complete.")

    if created_pool and pool is not None:
        pool.close()

    return len(chunks)


def run_ingestion_batch(
    pdf_paths: list[str],
    settings: Settings | None = None,
    embedder: EmbeddingProvider | None = None,
) -> dict[str, int | str]:
    """
    Ingest multiple PDF files sequentially, continuing on individual failures.

    Args:
        pdf_paths: List of absolute paths to PDF files to ingest.
        settings: Optional Settings instance. Loaded from env if not provided.
        embedder: Optional embedding provider. Created from settings if not provided.
                  Shared across all documents to avoid reloading the model per file.

    Returns:
        Dict mapping each pdf_path to either:
        - int: number of chunks ingested (success)
        - str: error message (failure)

    Notes:
        Pool is created once and shared across all documents.
        Embedder is created once and shared.
        Failures are logged and ingestion continues to the next file.
    """
    resolved = settings or load_settings()
    results: dict[str, int | str] = {}
    total_chunks = 0
    succeeded = 0
    failed = 0

    pool = PgVectorPool(resolved)
    db_client = PgVectorClient(pool, resolved.table_name)

    shared_embedder = embedder
    if shared_embedder is None:
        shared_embedder = SentenceTransformerEmbeddingProvider(
            resolved.embedding_model_name,
            resolved.embedding_normalize,
        )

    try:
        for pdf_path in pdf_paths:
            logger.info("Starting ingestion for %s", pdf_path)
            try:
                chunk_count = run_ingestion(
                    pdf_path,
                    settings=resolved,
                    embedder=shared_embedder,
                    db_client=db_client,
                    pool=pool,
                )
                results[pdf_path] = chunk_count
                total_chunks += chunk_count
                succeeded += 1
                logger.info("Completed ingestion for %s: chunks=%s", pdf_path, chunk_count)
            except Exception as exc:
                results[pdf_path] = str(exc)
                failed += 1
                logger.warning("Failed ingestion for %s: %s", pdf_path, exc)
            finally:
                if hasattr(shared_embedder, "reset"):
                    shared_embedder.reset()
    finally:
        pool.close()

    logger.info(
        "Batch ingestion complete: total=%s succeeded=%s failed=%s chunks=%s",
        len(pdf_paths),
        succeeded,
        failed,
        total_chunks,
    )
    return results
