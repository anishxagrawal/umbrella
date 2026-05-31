from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from evaluation.schema import EvalResult, EvalReport, MetricScore


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def build_report(
    results: list[EvalResult],
    metric_scores: dict[str, list[MetricScore]],
    run_id: str | None = None,
) -> EvalReport:
    """
    run_id defaults to UTC now: YYYYMMDD_HHMMSS
    summary = {metric_name: mean(scores)}
    avg_latency_ms = mean of non-errored latencies
    failed_samples = count where error is not None
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    failed_samples = sum(1 for result in results if result.error is not None)
    ok_latencies = [result.latency_ms for result in results if result.error is None]
    avg_latency_ms = _mean(ok_latencies)

    summary: dict[str, float] = {}
    for metric_name, scores in metric_scores.items():
        summary[metric_name] = _mean([score.score for score in scores])

    created_at = datetime.now(timezone.utc).isoformat()

    return EvalReport(
        run_id=run_id,
        total_samples=len(results),
        failed_samples=failed_samples,
        metric_scores=metric_scores,
        summary=summary,
        avg_latency_ms=avg_latency_ms,
        created_at=created_at,
    )


def save_report(report: EvalReport, results_dir: str | Path) -> Path:
    """Save to results_dir/run_{run_id}.json. Create dir if needed."""
    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"run_{report.run_id}.json"
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def _format_row(left: str, right: str, width: int) -> str:
    padded_left = left.ljust(width - len(right) - 2)
    return f"| {padded_left}{right} |"


def print_report(report: EvalReport) -> None:
    """
    Print formatted table using only stdlib.
    """
    title = f"Umbrella RAG Evaluation - {report.run_id}"
    width = max(50, len(title) + 4)

    top = "+" + "-" * (width - 2) + "+"
    print(top)
    print("| " + title.center(width - 4) + " |")
    print("+" + "-" * (width - 2) + "+")

    samples_line = f"Samples: {report.total_samples} ({report.failed_samples} failed)"
    latency_line = f"Avg latency: {report.avg_latency_ms:.0f} ms"
    print(_format_row(samples_line, "", width - 2))
    print(_format_row(latency_line, "", width - 2))

    print("+" + "-" * (width - 2) + "+")
    header = "Metric               Score   Passed"
    print("| " + header.ljust(width - 4) + " |")
    print("| " + "-" * (width - 4) + " |")

    for metric_name, scores in report.metric_scores.items():
        total = len(scores)
        passed = sum(1 for score in scores if score.passed)
        mean_score = report.summary.get(metric_name, 0.0)
        row = f"{metric_name:<20} {mean_score:>6.3f}   {passed}/{total}"
        print("| " + row.ljust(width - 4) + " |")

    print(top)
