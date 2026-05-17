from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2 import pool, sql

from ..config import Settings


class PgVectorPool:
    def __init__(self, settings: Settings) -> None:
        self._pool = pool.ThreadedConnectionPool(
            minconn=settings.db_pool_min,
            maxconn=settings.db_pool_max,
            host=settings.db_host,
            port=settings.db_port,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
        )

    @contextmanager
    def connection(self) -> Iterator[psycopg2.extensions.connection]:
        conn = None
        try:
            conn = self._pool.getconn()
            register_vector(conn)
            yield conn
        except Exception as exc:
            raise RuntimeError(f"Database connection failed: {exc}") from exc
        finally:
            if conn is not None:
                self._pool.putconn(conn)

    def close(self) -> None:
        self._pool.closeall()


class PgVectorClient:
    def __init__(self, pool: PgVectorPool, table_name: str) -> None:
        self._pool = pool
        self._table_name = table_name

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        query = sql.SQL(
            """
            SELECT id, text, source, page, section,
                     1 - (embedding <=> %s::vector) AS similarity
            FROM {table}
                 ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(table=sql.Identifier(self._table_name))

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (query_vector, query_vector, top_k))
                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "page": row[3],
                "section": row[4],
                "similarity": float(row[5]),
            }
            for row in rows
        ]

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch.")
        if not chunks:
            return

        query = sql.SQL(
            """
            INSERT INTO {table} (text, source, page, section, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """
        ).format(table=sql.Identifier(self._table_name))

        records = [
            (
                chunk["text"],
                chunk["source"],
                chunk["page"],
                chunk["section"],
                embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, records)
            conn.commit()
