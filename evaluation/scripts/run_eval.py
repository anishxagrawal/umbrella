from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.dataset import filter_to_corpus, load_filtered, load_teleqna, save_filtered
from evaluation.metrics.correctness import batch_correctness
from evaluation.metrics.reranker_hitrate import batch_reranker_hitrate
from evaluation.metrics.routing_accuracy import batch_routing_accuracy
from evaluation.report import build_report, print_report, save_report
from evaluation.runner import EvalRunner, load_results, save_results
from evaluation.schema import EvalResult


def _resolve_results_file(results_file: str | None, results_dir: str | Path) -> Path:
    if results_file:
        return Path(results_file)
    return Path(results_dir) / "results.json"


def _load_samples(args: argparse.Namespace) -> list:
    filtered_path = Path(args.filtered_file) if args.filtered_file else None
    if filtered_path and filtered_path.exists():
        return load_filtered(filtered_path)

    if not args.teleqna_file:
        raise ValueError("--teleqna-file is required when no filtered file exists")

    raw = load_teleqna(args.teleqna_file)
    samples = filter_to_corpus(raw, max_per_spec=args.max_per_spec)
    if filtered_path:
        save_filtered(samples, filtered_path)
    return samples


def _load_existing_results(path: Path) -> list[EvalResult]:
    if path.exists():
        return load_results(path)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Umbrella RAG evaluation")
    parser.add_argument("--teleqna-file", help="path to raw TeleQnA JSON")
    parser.add_argument("--filtered-file", help="path to pre-filtered samples JSON")
    parser.add_argument("--results-file", help="path to save/resume raw results JSON")
    parser.add_argument("--results-dir", default="evaluation/results")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--max-per-spec", type=int)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    samples = _load_samples(args)
    results_path = _resolve_results_file(args.results_file, args.results_dir)
    existing_results = _load_existing_results(results_path)

    runner = EvalRunner(api_url=args.api_url, delay_between_requests=args.delay)
    results = runner.run_all(samples, existing_results=existing_results)

    save_results(results, results_path)

    metric_scores: dict[str, list] = {}
    metric_scores["routing_accuracy"] = batch_routing_accuracy(results)
    metric_scores["reranker_hitrate"] = batch_reranker_hitrate(results)

    if not args.skip_correctness:
        metric_scores["correctness"] = batch_correctness(results)

    if not args.skip_ragas:
        from evaluation.metrics.ragas_metrics import score_ragas

        metric_scores.update(score_ragas(results))

    report = build_report(results, metric_scores)
    save_report(report, args.results_dir)
    print_report(report)


if __name__ == "__main__":
    main()
