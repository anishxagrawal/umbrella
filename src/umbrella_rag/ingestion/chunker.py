from __future__ import annotations

from typing import Any


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    min_remaining_words: int,
) -> list[dict[str, Any]]:
    """Split pages into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")

    chunks: list[dict[str, Any]] = []

    for page in pages:
        words = page["text"].split()

        if len(words) <= chunk_size:
            chunks.append(
                {
                    "text": page["text"],
                    "source": page["source"],
                    "page": page["page"],
                    "section": page["section"],
                }
            )
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                {
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                    "section": page["section"],
                }
            )

            start += chunk_size - overlap
            if len(words) - start < min_remaining_words:
                break

    return chunks
