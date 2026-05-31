from __future__ import annotations

import os
import time

import httpx

from evaluation.schema import EvalResult, MetricScore


PROMPT_TEMPLATE = """You are evaluating a RAG system answer for a telecom engineering question.

Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {generated_answer}

Score the generated answer from 0.0 to 1.0:
- 1.0: Correct and complete, matches gold answer meaning
- 0.75: Mostly correct, minor omissions
- 0.5: Partially correct, key facts present but incomplete
- 0.25: Mostly wrong, one relevant detail
- 0.0: Completely wrong or hallucinated

Respond with exactly two lines:
SCORE: <float>
REASONING: <one sentence>
"""


def _call_groq(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Groq API returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return str(content).strip()


def _parse_response(raw: str) -> tuple[float | None, str | None]:
    score = None
    reasoning = None
    for line in raw.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                score = None
        if line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return score, reasoning


def score_correctness(result: EvalResult) -> MetricScore:
    """
    Fill PROMPT_TEMPLATE, call Groq, parse SCORE and REASONING lines.
    On parse failure: score=0.0, reasoning=raw_response, passed=False.
    passed = score >= 0.5
    """
    prompt = PROMPT_TEMPLATE.format(
        question=result.sample.question,
        gold_answer=result.sample.gold_answer,
        generated_answer=result.generated_answer,
    )
    raw_response = ""
    try:
        raw_response = _call_groq(prompt)
        score, reasoning = _parse_response(raw_response)
    except Exception as exc:
        score = 0.0
        reasoning = str(exc)

    if score is None:
        score = 0.0
        reasoning = raw_response

    return MetricScore(
        metric_name="correctness",
        score=score,
        reasoning=reasoning or "",
        passed=score >= 0.5,
    )


def batch_correctness(
    results: list[EvalResult],
    delay_seconds: float = 1.0,
) -> list[MetricScore]:
    """Apply to non-errored results. Sleep delay_seconds between calls."""
    scores: list[MetricScore] = []
    for result in results:
        if result.error:
            continue
        scores.append(score_correctness(result))
        time.sleep(delay_seconds)
    return scores
