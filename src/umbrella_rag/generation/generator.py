from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from groq import Groq

from ..config import Settings, load_settings
from ..types import ChunkDTO


@dataclass(frozen=True)
class SourceReference:
    source: str
    page: int
    section: str | None
    rerank_score: float


@dataclass(frozen=True)
class RCAResponse:
    answer: str
    sources: list[SourceReference]
    model: str
    query: str
    context_chunks: int


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GroqChatProvider:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
    ) -> None:
        self._client = Groq(api_key=api_key)
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as exc:
            raise RuntimeError(f"Groq API call failed: {exc}") from exc

        content = _extract_message_content(response)
        if not content:
            raise RuntimeError("Groq API returned an empty response.")
        return content


class Generator:
    def __init__(self, provider: LLMProvider, model_name: str) -> None:
        self._provider = provider
        self._model_name = model_name

    def generate(self, query: str, chunks: list[ChunkDTO]) -> RCAResponse:
        if not query or not query.strip():
            return RCAResponse(
                answer="Query must be a non-empty string.",
                sources=[],
                model=self._model_name,
                query=query,
                context_chunks=0,
            )

        limited = chunks[:5]
        if not limited:
            return RCAResponse(
                answer=(
                    "The provided context does not contain enough information to "
                    "answer this question."
                ),
                sources=[],
                model=self._model_name,
                query=query,
                context_chunks=0,
            )

        prompt = _build_prompt(query, limited)
        try:
            raw_response = _call_llm(self._provider, prompt)
            answer = _parse_response(raw_response)
            if not answer:
                return RCAResponse(
                    answer="Groq API returned an empty or malformed response.",
                    sources=[],
                    model=self._model_name,
                    query=query,
                    context_chunks=len(limited),
                )
            sources = _extract_sources(answer, limited)
            return RCAResponse(
                answer=answer,
                sources=sources,
                model=self._model_name,
                query=query,
                context_chunks=len(limited),
            )
        except Exception as exc:
            return RCAResponse(
                answer=_format_error_message(exc),
                sources=[],
                model=self._model_name,
                query=query,
                context_chunks=len(limited),
            )


def build_generator(settings: Settings | None = None) -> Generator:
    resolved = settings or load_settings()
    provider = GroqChatProvider(
        api_key=resolved.groq_api_key,
        model_name=resolved.groq_model_name,
        max_tokens=resolved.groq_max_tokens,
        temperature=resolved.groq_temperature,
        system_prompt=resolved.groq_system_prompt,
    )
    return Generator(provider=provider, model_name=resolved.groq_model_name)


def _build_prompt(query: str, chunks: list[ChunkDTO]) -> str:
    lines = ["", "Context:"]
    for chunk in chunks:
        section_label = _format_section(chunk.section)
        lines.extend(
            [
                "---",
                chunk.text,
                (
                    f"[Source: {chunk.source}, Page: {chunk.page}, "
                    f"Section: {section_label}]"
                ),
            ]
        )

    lines.extend(["", f"Engineer Question: {query}", "", "RCA Analysis:"])
    return "\n".join(lines)


def _call_llm(provider: LLMProvider, prompt: str) -> str:
    return provider.generate(prompt)


def _parse_response(raw: str) -> str:
    return raw.strip()


def _extract_message_content(response: object) -> str:
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if not content:
            return ""
        return str(content).strip()
    except Exception:
        return ""


def _format_error_message(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    if "rate limit" in lowered or "429" in lowered:
        return "Rate limit exceeded when calling the Groq API."
    if "timeout" in lowered:
        return "Groq API request timed out."
    if "authentication" in lowered or "unauthorized" in lowered or "api key" in lowered:
        return "Groq API authentication failed. Check GROQ_API_KEY."
    if "malformed" in lowered or "empty" in lowered:
        return "Groq API returned an empty or malformed response."
    return f"Groq API call failed: {message}"


def _extract_sources(answer: str, chunks: list[ChunkDTO]) -> list[SourceReference]:
    citations = _parse_citations(answer)
    if not citations:
        return []

    index = _build_source_index(chunks)
    results: list[SourceReference] = []
    seen: set[tuple[str, int, str | None]] = set()

    for citation in citations:
        key = _normalize_key(
            citation["source"],
            citation["page"],
            citation["section"],
        )
        if key in seen:
            continue
        reference = index.get(key)
        if reference is None:
            continue
        results.append(reference)
        seen.add(key)

    return results


def _parse_citations(answer: str) -> list[dict[str, object]]:
    pattern = re.compile(
        r"\[Source:\s*(?P<source>.*?),\s*Page:\s*(?P<page>\d+),\s*"
        r"Section:\s*(?P<section>.*?)\]"
    )
    matches = []
    for match in pattern.finditer(answer):
        source = match.group("source").strip()
        page = int(match.group("page"))
        section_raw = match.group("section").strip()
        section = _normalize_section(section_raw)
        matches.append({"source": source, "page": page, "section": section})
    return matches


def _build_source_index(chunks: list[ChunkDTO]) -> dict[tuple[str, int, str | None], SourceReference]:
    index: dict[tuple[str, int, str | None], SourceReference] = {}
    for chunk in chunks:
        key = _normalize_key(chunk.source, chunk.page, chunk.section)
        existing = index.get(key)
        if existing is None or chunk.rerank_score > existing.rerank_score:
            index[key] = SourceReference(
                source=chunk.source,
                page=chunk.page,
                section=chunk.section,
                rerank_score=chunk.rerank_score,
            )
    return index


def _format_section(section: str | None) -> str:
    if section is None or not str(section).strip():
        return "None"
    return str(section).strip()


def _normalize_section(section: str | None) -> str | None:
    if section is None:
        return None
    normalized = str(section).strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in {"none", "n/a", "null"}:
        return None
    return normalized


def _normalize_key(
    source: str,
    page: int,
    section: str | None,
) -> tuple[str, int, str | None]:
    normalized_source = source.strip().lower()
    normalized_section = _normalize_section(section)
    return (normalized_source, page, normalized_section)
