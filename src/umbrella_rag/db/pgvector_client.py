from __future__ import annotations

from contextlib import contextmanager
import logging
import time
from typing import Any, Iterator

from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2 import pool, sql

from ..config import Settings


logger = logging.getLogger(__name__)


class PgVectorPool:
    def __init__(self, settings: Settings) -> None:
        """
        Initialize a threaded connection pool for pgvector.

        Args:
            settings: Application settings containing database connection info.

        Returns:
            None.

        Notes:
            Uses ThreadedConnectionPool for concurrent access.
        """
        self._pool = pool.ThreadedConnectionPool(
            minconn=settings.db_pool_min,
            maxconn=settings.db_pool_max,
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
        )
        self._retry_attempts = settings.db_retry_attempts
        self._retry_backoff_seconds = settings.db_retry_backoff_seconds

    @contextmanager
    def connection(self) -> Iterator[psycopg2.extensions.connection]:
        """
        Yield a pgvector-registered connection from the pool.

        Args:
            None.

        Returns:
            Iterator over a psycopg2 connection.

        Notes:
            Ensures the connection is returned to the pool.
        """
        conn = None
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                conn = self._pool.getconn()
                register_vector(conn)
                yield conn
                return
            except Exception as exc:
                last_exc = exc
                if conn is not None:
                    self._pool.putconn(conn)
                    conn = None
                if attempt >= self._retry_attempts:
                    break
                logger.warning(
                    "Database connection attempt %s/%s failed: %s",
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                time.sleep(self._retry_backoff_seconds * attempt)
        raise RuntimeError(f"Database connection failed: {last_exc}") from last_exc

    def close(self) -> None:
        """
        Close all pooled connections.

        Args:
            None.

        Returns:
            None.

        Notes:
            Safe to call multiple times.
        """
        self._pool.closeall()


class PgVectorClient:
    def __init__(self, pool: PgVectorPool, table_name: str) -> None:
        """
        Initialize the pgvector client.

        Args:
            pool: Connection pool to use.
            table_name: Database table name for chunks.

        Returns:
            None.

        Notes:
            Stateless; safe for concurrent use.
        """
        self._pool = pool
        self._table_name = table_name

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        source_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Run a vector similarity search without filters.

        Args:
            query_vector: Embedding vector to search against.
            top_k: Maximum number of results to return.
            source_filter: Optional list of source filenames to filter on.

        Returns:
            List of result dicts with chunk metadata and similarity.

        Notes:
            Uses parameterized queries only.
        """
        base_query = sql.SQL(
            """
            SELECT id, text, source, page, section,
                   section_title, chunk_type, chunk_index, total_chunks,
                   is_summary, depth, chapter, parent_section,
                   grandparent_section, spec_number, series_id,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {table}
            """
        ).format(table=sql.Identifier(self._table_name))

        where_clauses: list[sql.SQL] = []
        params: list[Any] = [query_vector]

        if source_filter:
            where_clauses.append(sql.SQL("source = ANY(%s)"))
            params.append(source_filter)

        if where_clauses:
            base_query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)

        query = base_query + sql.SQL(" ORDER BY embedding <=> %s::vector LIMIT %s")
        params.extend([query_vector, top_k])

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "page": row[3],
                "section": row[4],
                "section_title": row[5],
                "chunk_type": row[6],
                "chunk_index": row[7],
                "total_chunks": row[8],
                "is_summary": row[9],
                "depth": row[10],
                "chapter": row[11],
                "parent_section": row[12],
                "grandparent_section": row[13],
                "spec_number": row[14],
                "series_id": row[15],
                "similarity": float(row[16]),
            }
            for row in rows
        ]

    def search_with_filter(
        self,
        query_vector: list[float],
        top_k: int,
        series_id: str | None = None,
        chunk_type: str | None = None,
        is_summary: bool | None = None,
    ) -> list[dict]:
        """
        Vector similarity search with optional metadata filters.

        Args:
            query_vector: Embedding vector to search against.
            top_k: Maximum number of results to return.
            series_id: If provided, filter to chunks from this series only.
            chunk_type: If provided, filter to this chunk type only.
            is_summary: If provided, filter to summary or non-summary chunks.

        Returns:
            List of result dicts with chunk metadata and similarity.

        Notes:
            Filters are ANDed together. None means no filter on that field.
        """
        base_query = sql.SQL(
            """
            SELECT id, text, source, page, section,
                   section_title, chunk_type, chunk_index, total_chunks,
                   is_summary, depth, chapter, parent_section,
                   grandparent_section, spec_number, series_id,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {table}
            """
        ).format(table=sql.Identifier(self._table_name))

        where_clauses: list[sql.SQL] = []
        params: list[Any] = [query_vector]

        if series_id is not None:
            where_clauses.append(sql.SQL("series_id = %s"))
            params.append(series_id)
        if chunk_type is not None:
            where_clauses.append(sql.SQL("chunk_type = %s"))
            params.append(chunk_type)
        if is_summary is not None:
            where_clauses.append(sql.SQL("is_summary = %s"))
            params.append(is_summary)

        if where_clauses:
            base_query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)

        query = base_query + sql.SQL(" ORDER BY embedding <=> %s::vector LIMIT %s")
        params.extend([query_vector, top_k])

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "page": row[3],
                "section": row[4],
                "section_title": row[5],
                "chunk_type": row[6],
                "chunk_index": row[7],
                "total_chunks": row[8],
                "is_summary": row[9],
                "depth": row[10],
                "chapter": row[11],
                "parent_section": row[12],
                "grandparent_section": row[13],
                "spec_number": row[14],
                "series_id": row[15],
                "similarity": float(row[16]),
            }
            for row in rows
        ]

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """
        Insert chunk records and their embeddings.

        Args:
            chunks: Chunk dicts to insert.
            embeddings: Embedding vectors aligned with chunks.

        Returns:
            None.

        Raises:
            ValueError: If chunks and embeddings lengths differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch.")
        if not chunks:
            return

        query = sql.SQL(
            """
            INSERT INTO {table} (
                text, source, page, section, embedding,
                section_title, chunk_type, chunk_index, total_chunks,
                is_summary, depth, chapter, parent_section,
                grandparent_section, spec_number, series_id
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            """
        ).format(table=sql.Identifier(self._table_name))

        records = [
            (
                chunk.get("text", ""),
                chunk.get("source", ""),
                int(chunk.get("page", 0) or 0),
                chunk.get("section", ""),
                embedding,
                chunk.get("section_title", ""),
                chunk.get("chunk_type", ""),
                int(chunk.get("chunk_index", 0) or 0),
                int(chunk.get("total_chunks", 0) or 0),
                bool(chunk.get("is_summary", False)),
                int(chunk.get("depth", 0) or 0),
                chunk.get("chapter", ""),
                chunk.get("parent_section", ""),
                chunk.get("grandparent_section", ""),
                chunk.get("spec_number", ""),
                chunk.get("series_id", ""),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, records)
            conn.commit()
