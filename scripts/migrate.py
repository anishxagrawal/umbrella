"""
Reproducible database setup script.
Run once to create the telecom_chunks table and ivfflat index.
Safe to re-run - uses CREATE TABLE IF NOT EXISTS.

Usage:
    python scripts/migrate.py
"""
from __future__ import annotations

import logging
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS telecom_chunks (
    id                  SERIAL PRIMARY KEY,
    text                TEXT NOT NULL,
    source              TEXT,
    page                INTEGER,
    section             TEXT,
    embedding           vector(1024),
    section_title       TEXT,
    chunk_type          TEXT,
    chunk_index         INTEGER,
    total_chunks        INTEGER,
    is_summary          BOOLEAN DEFAULT FALSE,
    depth               INTEGER,
    chapter             TEXT,
    parent_section      TEXT,
    grandparent_section TEXT,
    spec_number         TEXT,
    series_id           TEXT,
    parent_chunk_id     INTEGER REFERENCES telecom_chunks(id)
);

CREATE INDEX IF NOT EXISTS telecom_chunks_embedding_idx
ON telecom_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);
"""


def main() -> None:
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(DDL)
        logger.info("Migration complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
