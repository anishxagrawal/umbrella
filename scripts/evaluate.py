"""
CLI evaluator that calls the /rca HTTP endpoint.

Usage:
    python scripts/evaluate.py --query "example question"
    python scripts/evaluate.py --file queries.txt
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import httpx


logger = logging.getLogger(__name__)


def _read_queries(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON queries file must contain a list of objects")
        queries: list[str] = []
        for item in data:
            if not isinstance(item, dict) or "query" not in item:
                raise ValueError("Each JSON item must be an object with a 'query' field")
            query = str(item["query"]).strip()
            if query:
                queries.append(query)
        return queries

    raw = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in raw if line.strip()]


def _iter_queries(args: argparse.Namespace) -> Iterable[str]:
    if args.file:
        return _read_queries(Path(args.file))
    return [args.query]


def _post_query(client: httpx.Client, url: str, query: str) -> dict[str, object]:
    response = client.post(url, json={"query": query}, timeout=60.0)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Umbrella RAG via /rca")
    parser.add_argument("--url", default="http://localhost:8000/rca")
    parser.add_argument("--query", help="Single query string")
    parser.add_argument("--file", help="Path to newline-delimited queries")
    parser.add_argument("--queries-file", dest="file", help="Path to JSON or text queries")
    parser.add_argument("--json", action="store_true", help="Print raw JSON responses")
    args = parser.parse_args()

    if not args.query and not args.file:
        parser.error("Provide --query or --file")

    logging.basicConfig(level=logging.INFO)
    queries = list(_iter_queries(args))
    logger.info("Evaluating %s queries", len(queries))

    with httpx.Client() as client:
        for query in queries:
            result = _post_query(client, args.url, query)
            if args.json:
                logger.info("Result for %r: %s", query, json.dumps(result))
            else:
                answer = result.get("answer", "")
                latency = result.get("latency_ms", "")
                logger.info("Query=%r latency_ms=%s", query, latency)
                logger.info("Answer: %s", answer)


if __name__ == "__main__":
    main()
