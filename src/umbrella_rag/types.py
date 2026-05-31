from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkDTO:
    """Shared data transfer object for a retrieved and optionally reranked chunk.

    This is the canonical chunk type shared across retrieval, reranking,
    generation, and orchestration layers. No layer should define its own
    chunk dataclass - all must import from here.

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
        rerank_score: Optional reranker score. None before reranking.
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
