from __future__ import annotations

from evaluation.schema import EvalResult, MetricScore


def _chunk_score(chunk: object) -> float:
    score = getattr(chunk, "rerank_score", None)
    if score is None:
        return float(getattr(chunk, "similarity", 0.0) or 0.0)
    return float(score)


def score_reranker_hit(result: EvalResult, top_k: int = 5) -> MetricScore:
    """
    Sort chunks by rerank_score desc (None -> use similarity).
    Check if any of top_k chunks has source == gold_source_spec.
    score=1.0 if hit, 0.0 if not. passed = score == 1.0.
    """
    gold_source = result.sample.gold_source_spec
    sorted_chunks = sorted(
        result.retrieved_chunks,
        key=_chunk_score,
        reverse=True,
    )
    hit = any(chunk.source == gold_source for chunk in sorted_chunks[:top_k])
    score = 1.0 if hit else 0.0
    reasoning = f"expected {gold_source}, hit_in_top_{top_k}={hit}"
    return MetricScore(
        metric_name="reranker_hitrate",
        score=score,
        reasoning=reasoning,
        passed=score == 1.0,
    )


def batch_reranker_hitrate(
    results: list[EvalResult], top_k: int = 5
) -> list[MetricScore]:
    """Apply to each non-errored result."""
    scores: list[MetricScore] = []
    for result in results:
        if result.error:
            continue
        scores.append(score_reranker_hit(result, top_k=top_k))
    return scores
