"""Re-embed fixed summary chunks and rebuild the ivfflat index."""
from __future__ import annotations

import logging

from umbrella_rag.config import load_settings
from umbrella_rag.db.pgvector_client import PgVectorClient, PgVectorPool
from umbrella_rag.embeddings.encoder import SentenceTransformerEmbeddingProvider


logger = logging.getLogger(__name__)


def _fetch_summaries(db_client: PgVectorClient) -> list[tuple[int, str]]:
    query = """
        SELECT id, text
        FROM telecom_chunks
        WHERE is_summary = true
          AND text LIKE 'Summary of section%'
        ORDER BY id
    """
    with db_client._pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return [(int(row[0]), str(row[1])) for row in rows]


def _update_embeddings(
    db_client: PgVectorClient,
    rows: list[tuple[int, str]],
    embedder: SentenceTransformerEmbeddingProvider,
    batch_size: int,
) -> int:
    if not rows:
        return 0

    ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]
    embeddings = embedder.embed(texts, batch_size=batch_size, show_progress=False)

    total = 0
    with db_client._pool.connection() as conn:
        with conn.cursor() as cur:
            for start in range(0, len(ids), 50):
                end = start + 50
                batch_records = [
                    (embeddings[idx], ids[idx])
                    for idx in range(start, min(end, len(ids)))
                ]
                cur.executemany(
                    "UPDATE telecom_chunks SET embedding = %s WHERE id = %s",
                    batch_records,
                )
                total += len(batch_records)
                logger.info("Updated %s/%s summaries", total, len(ids))
            conn.commit()
    return total


def _rebuild_index(db_client: PgVectorClient) -> None:
    with db_client._pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS telecom_chunks_embedding_idx")
            cur.execute(
                """
                CREATE INDEX telecom_chunks_embedding_idx
                ON telecom_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 200)
                """
            )
            conn.commit()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    settings = load_settings()
    pool = PgVectorPool(settings)
    db_client = PgVectorClient(pool, settings.table_name)
    embedder = SentenceTransformerEmbeddingProvider(
        settings.embedding_model_name,
        settings.embedding_normalize,
    )

    try:
        rows = _fetch_summaries(db_client)
        logger.info("Found %s summaries to re-embed", len(rows))
        if not rows:
            return

        updated = _update_embeddings(
            db_client,
            rows,
            embedder,
            batch_size=settings.embedding_batch_size,
        )
        logger.info("Re-embedding complete — %s rows updated", updated)

        logger.info("Rebuilding index...")
        _rebuild_index(db_client)
        logger.info("Index rebuilt — done")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
