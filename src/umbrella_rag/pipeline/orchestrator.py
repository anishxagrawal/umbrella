from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable

from ..retrieval.expander import expand_query


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """Structured chunk returned by the RCA orchestrator.

    Attributes:
        id: Primary key of the chunk record.
        text: Chunk text content.
        source: Source document filename.
        page: Source page number.
        section: Section identifier.
        section_title: Human-readable section title.
        chunk_type: Chunk classification label.
        spec_number: 3GPP specification number.
        series_id: 3GPP series identifier.
        similarity: Dense retrieval similarity score.
        rerank_score: Optional reranker score.
    """

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


@dataclass(frozen=True)
class RCAResponse:
    """Structured end-to-end RCA response.

    Attributes:
        query: Original query string.
        expanded_query: Query after glossary expansion.
        answer: Final generated RCA answer.
        chunks: Ranked chunks used to produce the answer.
        retrieval_count: Number of chunks retrieved before reranking.
        reranked_count: Number of chunks returned after reranking.
        model: LLM model name used for generation.
        latency_ms: Total pipeline latency in milliseconds.
    """

    query: str
    expanded_query: str
    answer: str
    chunks: list[RetrievedChunk]
    retrieval_count: int
    reranked_count: int
    model: str
    latency_ms: float


class RCAOrchestrator:
    """Coordinate the full retrieve, rerank, and generate flow."""

    def __init__(
        self,
        retriever: Any,
        reranker: Any,
        generator: Any,
        expander_fn: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the orchestrator with injected dependencies.

        Args:
            retriever: Configured retriever instance.
            reranker: Configured reranker instance.
            generator: Configured generator instance.
            expander_fn: Optional query expansion function.

        Returns:
            None.

        Notes:
            Dependencies are injected and reused across requests.
            When no expander is provided, the default glossary expander is used.
        """
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._expander_fn = expander_fn or expand_query

    def run(self, query: str) -> RCAResponse:
        """Run the full RCA pipeline for a query.

        Args:
            query: Raw query string from an engineer.

        Returns:
            Structured RCA response containing the generated answer and chunks.

        Raises:
            RuntimeError: If any stage of the pipeline fails.

        Notes:
            The method is thread-safe as long as injected dependencies are.
            Latency is measured across expansion, retrieval, reranking, and generation.
        """
        start_time = time.perf_counter()
        try:
            expanded_query = self._expander_fn(query)
            retrieved_chunks = self._retriever.retrieve(expanded_query)
            reranked_chunks = self._reranker.rerank(expanded_query, retrieved_chunks)
            generator_response = self._generator.generate(expanded_query, reranked_chunks)
        except Exception as exc:
            logger.exception("RCA pipeline failed for query=%r", query)
            raise RuntimeError(f"RCA pipeline failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response_chunks = [_to_response_chunk(chunk) for chunk in reranked_chunks]
        answer = getattr(generator_response, "answer", "")
        model = getattr(generator_response, "model", "")

        logger.info(
            "RCA pipeline: query=%s expanded=%s retrieved=%s reranked=%s latency=%.0fms",
            query,
            expanded_query,
            len(retrieved_chunks),
            len(reranked_chunks),
            latency_ms,
        )
        for chunk in response_chunks:
            logger.debug(
                "RCA chunk: id=%s similarity=%.4f rerank_score=%s source=%s page=%s section=%s",
                chunk.id,
                chunk.similarity,
                "None" if chunk.rerank_score is None else f"{chunk.rerank_score:.4f}",
                chunk.source,
                chunk.page,
                chunk.section,
            )

        return RCAResponse(
            query=query,
            expanded_query=expanded_query,
            answer=answer,
            chunks=response_chunks,
            retrieval_count=len(retrieved_chunks),
            reranked_count=len(reranked_chunks),
            model=model,
            latency_ms=latency_ms,
        )

    def health(self) -> dict[str, bool]:
        """Check health of the orchestrator components.

        Returns:
            Mapping of component names to boolean health states.

        Notes:
            The method never raises and returns False for any failing component.
        """
        return {
            "retriever": _check_retriever_health(self._retriever),
            "reranker": _check_reranker_health(self._reranker),
            "generator": _check_generator_health(self._generator),
        }


def _safe_str(value: Any) -> str:
    """Convert a value to a safe string representation.

    Args:
        value: Input value that may be None.

    Returns:
        A non-null string representation.

    Notes:
        None values are converted to an empty string.
    """
    if value is None:
        return ""
    return str(value)


def _to_response_chunk(chunk: Any) -> RetrievedChunk:
    """Convert a retriever or reranker chunk into the orchestrator response shape.

    Args:
        chunk: Chunk object from the retrieval or reranking layer.

    Returns:
        A normalized RetrievedChunk dataclass instance.

    Notes:
        Missing metadata fields are normalized to empty strings.
    """
    return RetrievedChunk(
        id=int(getattr(chunk, "id")),
        text=_safe_str(getattr(chunk, "text", "")),
        source=_safe_str(getattr(chunk, "source", "")),
        page=int(getattr(chunk, "page", 0) or 0),
        section=_safe_str(getattr(chunk, "section", "")),
        section_title=_safe_str(getattr(chunk, "section_title", "")),
        chunk_type=_safe_str(getattr(chunk, "chunk_type", "")),
        spec_number=_safe_str(getattr(chunk, "spec_number", "")),
        series_id=_safe_str(getattr(chunk, "series_id", "")),
        similarity=float(getattr(chunk, "similarity", 0.0) or 0.0),
        rerank_score=(
            None
            if getattr(chunk, "rerank_score", None) is None
            else float(getattr(chunk, "rerank_score"))
        ),
    )


def _check_retriever_health(retriever: Any) -> bool:
    """Check whether the retriever can reach the database.

    Args:
        retriever: Retriever instance to inspect.

    Returns:
        True when the retriever exposes a usable connection pool.

    Notes:
        This check is intentionally lightweight and never raises.
    """
    pool = getattr(retriever, "_pool", None)
    if pool is None:
        return hasattr(retriever, "retrieve")

    try:
        with pool.connection():
            return True
    except Exception:
        return False


def _check_reranker_health(reranker: Any) -> bool:
    """Check whether the reranker model is ready.

    Args:
        reranker: Reranker instance to inspect.

    Returns:
        True when the reranker already has a loaded model or can expose one.

    Notes:
        This check avoids reranking any query and never raises.
    """
    if getattr(reranker, "_model", None) is not None:
        return True
    return hasattr(reranker, "rerank")


def _check_generator_health(generator: Any) -> bool:
    """Check whether the generator is initialized.

    Args:
        generator: Generator instance to inspect.

    Returns:
        True when the generator exposes a provider or generate method.

    Notes:
        This check does not call the LLM provider and never raises.
    """
    provider = getattr(generator, "_provider", None)
    if provider is not None:
        return hasattr(provider, "generate") or hasattr(provider, "_client")
    return hasattr(generator, "generate")
