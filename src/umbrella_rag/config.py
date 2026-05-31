from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _as_int(name: str) -> int:
    return int(_require_env(name))


def _as_bool(name: str) -> bool:
    value = _require_env(name).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def _as_float(name: str) -> float:
    return float(_require_env(name))


def _as_csv_set(name: str) -> frozenset[str]:
    value = _require_env(name)
    items = [item.strip() for item in value.split(",") if item.strip()]
    return frozenset(items)


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    db_pool_min: int
    db_pool_max: int
    db_retry_attempts: int
    db_retry_backoff_seconds: float
    table_name: str
    embedding_model_name: str
    embedding_normalize: bool
    retrieval_top_k: int
    rerank_confidence_threshold: float
    reranker_model_name: str
    reranker_top_k: int
    chunk_size_words: int
    chunk_overlap_words: int
    chunk_min_remainder_words: int
    embedding_batch_size: int
    embedding_show_progress: bool
    flat_table_sources: frozenset[str]
    flat_table_chunk_size: int
    flat_table_overlap: int
    flat_table_min_remaining: int
    groq_api_key: str
    groq_model_name: str
    groq_max_tokens: int
    groq_temperature: float
    groq_system_prompt: str


def load_settings() -> Settings:
    return Settings(
        db_host=_require_env("DB_HOST"),
        db_port=_as_int("DB_PORT"),
        db_name=_require_env("DB_NAME"),
        db_user=_require_env("DB_USER"),
        db_password=_require_env("DB_PASSWORD"),
        db_pool_min=_as_int("DB_POOL_MIN"),
        db_pool_max=_as_int("DB_POOL_MAX"),
        db_retry_attempts=_as_int("DB_RETRY_ATTEMPTS"),
        db_retry_backoff_seconds=_as_float("DB_RETRY_BACKOFF_SECONDS"),
        table_name=_require_env("TELECOM_TABLE_NAME"),
        embedding_model_name=_require_env("EMBEDDING_MODEL_NAME"),
        embedding_normalize=_as_bool("EMBEDDING_NORMALIZE"),
        retrieval_top_k=_as_int("RETRIEVAL_TOP_K"),
        rerank_confidence_threshold=_as_float("RERANK_CONFIDENCE_THRESHOLD"),
        reranker_model_name=_require_env("RERANKER_MODEL_NAME"),
        reranker_top_k=_as_int("RERANKER_TOP_K"),
        chunk_size_words=_as_int("CHUNK_SIZE_WORDS"),
        chunk_overlap_words=_as_int("CHUNK_OVERLAP_WORDS"),
        chunk_min_remainder_words=_as_int("CHUNK_MIN_REMAINDER_WORDS"),
        embedding_batch_size=_as_int("EMBEDDING_BATCH_SIZE"),
        embedding_show_progress=_as_bool("EMBEDDING_SHOW_PROGRESS"),
        flat_table_sources=_as_csv_set("FLAT_TABLE_SOURCES"),
        flat_table_chunk_size=_as_int("FLAT_TABLE_CHUNK_SIZE"),
        flat_table_overlap=_as_int("FLAT_TABLE_OVERLAP"),
        flat_table_min_remaining=_as_int("FLAT_TABLE_MIN_REMAINING"),
        groq_api_key=_require_env("GROQ_API_KEY"),
        groq_model_name=_require_env("GROQ_MODEL_NAME"),
        groq_max_tokens=_as_int("GROQ_MAX_TOKENS"),
        groq_temperature=float(_require_env("GROQ_TEMPERATURE")),
        groq_system_prompt=_require_env("GROQ_SYSTEM_PROMPT"),
    )
