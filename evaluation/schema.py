from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalSample:
    id: str
    question: str
    gold_answer: str
    gold_source_spec: str
    category: str
    options: dict[str, str]


@dataclass(frozen=True)
class RetrievedChunkInfo:
    source: str
    page: int
    section: str
    section_title: str
    text: str
    similarity: float
    rerank_score: float | None


@dataclass(frozen=True)
class EvalResult:
    sample: EvalSample
    generated_answer: str
    retrieved_chunks: list[RetrievedChunkInfo]
    routed_to: list[str]
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class MetricScore:
    metric_name: str
    score: float
    reasoning: str
    passed: bool


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    total_samples: int
    failed_samples: int
    metric_scores: dict[str, list[MetricScore]]
    summary: dict[str, float]
    avg_latency_ms: float
    created_at: str


def make_chunk_info(chunk_dict: dict) -> RetrievedChunkInfo:
    return RetrievedChunkInfo(
        source=chunk_dict.get("source", ""),
        page=int(chunk_dict.get("page", 0) or 0),
        section=chunk_dict.get("section", "") or "",
        section_title=chunk_dict.get("section_title", "") or "",
        text=chunk_dict.get("text", ""),
        similarity=float(chunk_dict.get("similarity", 0.0) or 0.0),
        rerank_score=(
            None if chunk_dict.get("rerank_score") is None
            else float(chunk_dict["rerank_score"])
        ),
    )
