from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import time
from typing import Iterable

import requests

from evaluation.schema import EvalResult, EvalSample, RetrievedChunkInfo, make_chunk_info


logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout_seconds: int = 60,
        delay_between_requests: float = 0.5,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._delay_between_requests = delay_between_requests

    def run_sample(self, sample: EvalSample) -> EvalResult:
        """
        POST /rca with {"query": sample.question}.
        Parse chunks via make_chunk_info().
        routed_to = sorted unique sources across all chunks.
        Measure latency from request start.
        On any exception populate error field, return EvalResult with
        empty chunks and routed_to, latency_ms=0.0.
        """
        try:
            start = time.perf_counter()
            response = requests.post(
                f"{self._api_url}/rca",
                json={"query": sample.question},
                timeout=self._timeout_seconds,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            payload = response.json()
            raw_chunks = payload.get("chunks") or []
            chunks: list[RetrievedChunkInfo] = [
                make_chunk_info(chunk) for chunk in raw_chunks
            ]
            routed_to = sorted({chunk.source for chunk in chunks if chunk.source})
            return EvalResult(
                sample=sample,
                generated_answer=str(payload.get("answer", "")),
                retrieved_chunks=chunks,
                routed_to=routed_to,
                latency_ms=float(payload.get("latency_ms", latency_ms)),
                error=None,
            )
        except Exception as exc:
            return EvalResult(
                sample=sample,
                generated_answer="",
                retrieved_chunks=[],
                routed_to=[],
                latency_ms=0.0,
                error=str(exc),
            )

    def run_all(
        self,
        samples: list[EvalSample],
        existing_results: list[EvalResult] | None = None,
    ) -> list[EvalResult]:
        """
        Skip samples whose id already appears in existing_results.
        Log progress every 10 samples.
        Sleep delay_between_requests between calls.
        Return existing + new results.
        """
        results = list(existing_results or [])
        seen_ids = {result.sample.id for result in results}

        for index, sample in enumerate(samples, start=1):
            if sample.id in seen_ids:
                continue
            result = self.run_sample(sample)
            results.append(result)
            seen_ids.add(sample.id)
            if index % 10 == 0:
                logger.info("Processed %s samples", index)
            time.sleep(self._delay_between_requests)

        return results


def _serialize_results(results: Iterable[EvalResult]) -> list[dict]:
    return [asdict(result) for result in results]


def save_results(results: list[EvalResult], path: str | Path) -> None:
    """Serialize list[EvalResult] to JSON. Handle dataclass nesting."""
    target = Path(path)
    payload = _serialize_results(results)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_results(path: str | Path) -> list[EvalResult]:
    """Deserialize JSON back to list[EvalResult]."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Results file must be a JSON list")

    results: list[EvalResult] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sample_raw = item.get("sample") or {}
        chunks_raw = item.get("retrieved_chunks") or []
        sample = EvalSample(
            id=str(sample_raw.get("id", "")),
            question=str(sample_raw.get("question", "")),
            gold_answer=str(sample_raw.get("gold_answer", "")),
            gold_source_spec=str(sample_raw.get("gold_source_spec", "")),
            category=str(sample_raw.get("category", "")),
            options={
                str(key): str(value)
                for key, value in (sample_raw.get("options") or {}).items()
            },
        )
        chunks = [make_chunk_info(chunk) for chunk in chunks_raw if isinstance(chunk, dict)]
        results.append(
            EvalResult(
                sample=sample,
                generated_answer=str(item.get("generated_answer", "")),
                retrieved_chunks=chunks,
                routed_to=[str(value) for value in (item.get("routed_to") or [])],
                latency_ms=float(item.get("latency_ms", 0.0) or 0.0),
                error=None if item.get("error") is None else str(item.get("error")),
            )
        )
    return results
