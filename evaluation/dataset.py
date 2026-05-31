from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from evaluation.schema import EvalSample


SPEC_KEYWORD_MAP: dict[str, list[str]] = {
    "38104-j40.pdf": [
        "38.104",
        "BS threshold",
        "reference sensitivity",
        "base station",
        "emission",
    ],
    "38133-i90.pdf": [
        "38.133",
        "RRM",
        "RSRP",
        "RSRQ",
        "SINR",
        "measurement threshold",
        "A3 event",
        "handover threshold",
    ],
    "38213-j30.pdf": [
        "38.213",
        "PDCCH",
        "DCI",
        "CORESET",
        "search space",
        "physical layer control",
    ],
    "38214-i40.pdf": [
        "38.214",
        "PDSCH",
        "PUSCH",
        "MCS",
        "CQI",
        "scheduling",
        "throughput",
    ],
    "38300-j20.pdf": [
        "38.300",
        "NR architecture",
        "CU-DU",
        "F1 interface",
        "protocol stack",
    ],
    "38321-j10.pdf": [
        "38.321",
        "MAC",
        "HARQ",
        "BSR",
        "RACH",
        "PHR",
        "timing advance",
    ],
    "38331-hg0.pdf": [
        "38.331",
        "RRC",
        "handover command",
        "MIB",
        "SIB",
        "reconfiguration",
    ],
    "38413-j00.pdf": [
        "38.413",
        "NGAP",
        "AMF",
        "PDU session",
        "path switch",
        "NG handover",
        "N2",
    ],
    "38423-i40.pdf": [
        "38.423",
        "XnAP",
        "Xn interface",
        "SN addition",
        "UE context release",
    ],
}


def load_teleqna(path: str | Path) -> dict:
    """Load raw TeleQnA JSON. Returns raw dict."""
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def detect_spec(question_text: str, category: str) -> str | None:
    """
    Case-insensitive keyword match across SPEC_KEYWORD_MAP.
    Return spec with most keyword hits. Return None if zero hits.
    """
    text = f"{question_text} {category}".lower()
    best_spec: str | None = None
    best_hits = 0

    for spec, keywords in SPEC_KEYWORD_MAP.items():
        hits = 0
        for keyword in keywords:
            if keyword.lower() in text:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_spec = spec

    if best_hits == 0:
        return None
    return best_spec


def _iter_questions(raw: dict | list) -> Iterable[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("questions", "data", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return []


def _resolve_answer(answer: str, options: dict[str, str]) -> str:
    if answer in options:
        return options[answer]
    return answer


def filter_to_corpus(
    raw: dict,
    max_per_spec: int | None = None,
) -> list[EvalSample]:
    """
    Filter raw TeleQnA to questions matching your 9 specs.
    For gold_answer: if answer is a key like 'a' and options exist,
    resolve to full option text. Otherwise use answer as-is.
    If max_per_spec set, cap per spec. Sort by gold_source_spec then id.
    """
    results: list[EvalSample] = []
    per_spec_counts: dict[str, int] = {}

    for entry in _iter_questions(raw):
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question", "")).strip()
        category = str(entry.get("category", "")).strip()
        if not question:
            continue
        spec = detect_spec(question, category)
        if spec is None:
            continue

        if max_per_spec is not None:
            current = per_spec_counts.get(spec, 0)
            if current >= max_per_spec:
                continue
            per_spec_counts[spec] = current + 1

        options = entry.get("options")
        if not isinstance(options, dict):
            options = {}

        answer = str(entry.get("answer", "")).strip()
        gold_answer = _resolve_answer(answer, options)

        sample = EvalSample(
            id=str(entry.get("id", "")).strip(),
            question=question,
            gold_answer=gold_answer,
            gold_source_spec=spec,
            category=category,
            options={str(k): str(v) for k, v in options.items()},
        )
        results.append(sample)

    results.sort(key=lambda item: (item.gold_source_spec, item.id))
    return results


def save_filtered(samples: list[EvalSample], path: str | Path) -> None:
    """Save list[EvalSample] as JSON."""
    target = Path(path)
    payload = [asdict(sample) for sample in samples]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_filtered(path: str | Path) -> list[EvalSample]:
    """Load JSON back to list[EvalSample]."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Filtered dataset must be a JSON list")

    results: list[EvalSample] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        results.append(
            EvalSample(
                id=str(item.get("id", "")),
                question=str(item.get("question", "")),
                gold_answer=str(item.get("gold_answer", "")),
                gold_source_spec=str(item.get("gold_source_spec", "")),
                category=str(item.get("category", "")),
                options={
                    str(key): str(value)
                    for key, value in (item.get("options") or {}).items()
                },
            )
        )
    return results
