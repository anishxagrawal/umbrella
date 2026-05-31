from __future__ import annotations

from evaluation.schema import EvalResult, MetricScore


def score_routing(result: EvalResult) -> MetricScore:
    """
    score=1.0 if gold_source_spec in result.routed_to else 0.0.
    passed = score == 1.0
    reasoning = f"expected {gold_source_spec}, found {routed_to}"
    """
    gold_source = result.sample.gold_source_spec
    routed_to = result.routed_to
    hit = gold_source in routed_to
    score = 1.0 if hit else 0.0
    reasoning = f"expected {gold_source}, found {routed_to}"
    return MetricScore(
        metric_name="routing_accuracy",
        score=score,
        reasoning=reasoning,
        passed=score == 1.0,
    )


def batch_routing_accuracy(results: list[EvalResult]) -> list[MetricScore]:
    """Apply score_routing to each non-errored result."""
    scores: list[MetricScore] = []
    for result in results:
        if result.error:
            continue
        scores.append(score_routing(result))
    return scores
