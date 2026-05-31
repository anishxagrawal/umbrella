from __future__ import annotations

from bisect import bisect_right
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

SECTION_PATTERN = re.compile(
    r"^(\d{1,2}(?:\.\d{1,3}){0,4})\s+([A-Z][^\n]{3,80})$",
    re.MULTILINE,
)
ANNEX_PATTERN = re.compile(
    r"^(Annex\s+[A-Z](?:\.\d{1,3})*)\s*"
    r"(?:\(normative\)|\(informative\))?\s*:?\s*([^\n]{3,80})$",
    re.MULTILINE,
)
TABLE_PATTERN = re.compile(
    r"^(Table\s+\d+[\.\-]\d+[\.\-]?\d*)\s*:\s*([^\n]{3,80})$",
    re.MULTILINE,
)
SPEC_FILENAME_PATTERN = re.compile(r"^(\d{2})(\d{3})")

DEFINITION_KEYWORDS = {"definition", "abbreviation", "glossary", "acronym", "terminology"}
PARAMETER_KEYWORDS = {"parameter", "threshold", "value", "timer", "constant", "ie", "field"}
PROCEDURE_KEYWORDS = {
    "procedure",
    "process",
    "step",
    "flow",
    "handover",
    "setup",
    "release",
    "establishment",
    "reconfiguration",
    "failure",
    "recovery",
}
EXAMPLE_KEYWORDS = {"example", "illustration", "scenario", "use case"}

REQUIRED_CHUNK_KEYS = {
    "text",
    "source",
    "page",
    "section",
    "section_title",
    "chunk_type",
    "chunk_index",
    "total_chunks",
    "is_summary",
    "depth",
    "chapter",
    "parent_section",
    "grandparent_section",
    "spec_number",
    "series_id",
}

SUMMARY_MAX_CHARS = 4000
SUMMARY_MIN_CHILD_WORDS = 15


def extract_spec_metadata(source: str) -> dict[str, str]:
    """
    Extract spec_number and series_id from a 3GPP filename.

    Args:
        source: Filename string such as "38331-hg0.pdf".

    Returns:
        Dict with keys spec_number (e.g. "38.331") and series_id (e.g. "38").
        Falls back to "unknown" if the pattern does not match.

    Notes:
        Pure function, safe for concurrent use.
    """
    match = SPEC_FILENAME_PATTERN.match(source)
    if match:
        series = match.group(1)
        number = match.group(2)
        return {"spec_number": f"{series}.{number}", "series_id": series}
    return {"spec_number": "unknown", "series_id": "unknown"}


def reconstruct_full_text(pages: list[dict[str, Any]]) -> tuple[str, list[tuple[int, int]]]:
    """
    Concatenate all page texts into one string, tracking char offsets per page.

    Args:
        pages: List of page dicts with keys text, page, source, section.

    Returns:
        Tuple of:
        - full_text: entire document as one string
        - page_map: list of (char_offset, page_number) tuples sorted by char_offset

    Notes:
        full_text is held in memory once and passed by reference downstream.
    """
    parts: list[str] = []
    page_map: list[tuple[int, int]] = []
    offset = 0

    for index, page in enumerate(pages):
        text = page.get("text", "")
        page_number = int(page.get("page", 0))
        page_map.append((offset, page_number))
        parts.append(text)
        offset += len(text)
        if index < len(pages) - 1:
            parts.append("\n")
            offset += 1

    return "".join(parts), page_map


def _resolve_page_number(start_char: int, page_map: list[tuple[int, int]]) -> int:
    """
    Resolve a character offset to a page number using the page map.

    Args:
        start_char: Character offset in the full document text.
        page_map: List of (char_offset, page_number) tuples.

    Returns:
        Page number where the offset belongs, or 0 if unknown.

    Notes:
        Uses bisect to find the last page offset less than or equal to start_char.
    """
    if not page_map:
        return 0
    offsets = [offset for offset, _ in page_map]
    index = bisect_right(offsets, start_char) - 1
    if index < 0:
        return page_map[0][1]
    return page_map[index][1]


def detect_sections(
    full_text: str,
    page_map: list[tuple[int, int]],
    source: str,
) -> list[dict[str, Any]]:
    """
    Find all section boundaries in the full document text.

    Args:
        full_text: Complete document text from reconstruct_full_text().
        page_map: Character offset to page number mapping.
        source: Filename string, used for metadata only.

    Returns:
        List of section dicts containing section_number, section_title, text,
        start_char, page, and source. Sorted by start_char.

    Notes:
        If no sections detected, returns one section for the entire document.
    """
    headings: list[dict[str, Any]] = []

    for match in SECTION_PATTERN.finditer(full_text):
        headings.append(
            {
                "start": match.start(),
                "end": match.end(),
                "section_number": match.group(1).strip(),
                "section_title": match.group(2).strip(),
            }
        )

    for match in ANNEX_PATTERN.finditer(full_text):
        headings.append(
            {
                "start": match.start(),
                "end": match.end(),
                "section_number": match.group(1).strip(),
                "section_title": match.group(2).strip(),
            }
        )

    headings.sort(key=lambda item: item["start"])

    if len(headings) < 3:
        logger.warning("Detected %s sections in %s", len(headings), source)

    if not headings:
        page_number = _resolve_page_number(0, page_map)
        return [
            {
                "section_number": "",
                "section_title": "Document",
                "text": full_text.strip(),
                "start_char": 0,
                "page": page_number,
                "source": source,
            }
        ]

    sections: list[dict[str, Any]] = []
    for index, heading in enumerate(headings):
        start = heading["end"]
        end = headings[index + 1]["start"] if index + 1 < len(headings) else len(full_text)
        section_text = full_text[start:end].strip()
        start_char = heading["start"]
        page_number = _resolve_page_number(start_char, page_map)
        sections.append(
            {
                "section_number": heading["section_number"],
                "section_title": heading["section_title"],
                "text": section_text,
                "start_char": start_char,
                "page": page_number,
                "source": source,
            }
        )

    return sections


def infer_hierarchy(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add parent-child relationship metadata to each section.

    Args:
        sections: List from detect_sections(), sorted by start_char.

    Returns:
        New list of section dicts with depth, chapter, parent_section,
        and grandparent_section added.

    Notes:
        Pure function with no side effects on input data.
    """
    enriched: list[dict[str, Any]] = []

    for section in sections:
        section_number = section.get("section_number", "")
        if not section_number:
            depth = 0
        elif section_number.startswith("Annex"):
            annex_id = section_number.replace("Annex", "", 1).strip()
            depth = annex_id.count(".") + 1 if annex_id else 1
        else:
            depth = section_number.count(".") + 1

        parent_section = section_number.rsplit(".", 1)[0] if "." in section_number else ""
        grandparent_section = (
            parent_section.rsplit(".", 1)[0] if "." in parent_section else ""
        )
        chapter = section_number.split(".")[0] if section_number else ""

        enriched_section = dict(section)
        enriched_section.update(
            {
                "depth": depth,
                "chapter": chapter,
                "parent_section": parent_section,
                "grandparent_section": grandparent_section,
            }
        )
        enriched.append(enriched_section)

    return enriched


def detect_chunk_type(section_title: str, section_text: str) -> str:
    """
    Classify a section into one of five chunk types.

    Args:
        section_title: Heading text of the section.
        section_text: Body text of the section.

    Returns:
        One of: "definition" | "parameter" | "procedure" | "example" | "general".

    Notes:
        Pure function, case-insensitive matching. Summary chunks are assigned
        by generate_summary_chunks().
    """
    title_lower = section_title.lower()
    text_lower = section_text.lower()

    if any(keyword in title_lower for keyword in DEFINITION_KEYWORDS):
        return "definition"
    if any(keyword in title_lower for keyword in PARAMETER_KEYWORDS):
        return "parameter"
    if TABLE_PATTERN.search(section_text):
        return "parameter"
    if any(keyword in title_lower for keyword in PROCEDURE_KEYWORDS):
        return "procedure"
    if "shall" in text_lower or "shall not" in text_lower or "is defined as" in text_lower:
        return "procedure"
    if any(keyword in title_lower for keyword in EXAMPLE_KEYWORDS):
        return "example"
    return "general"


def _build_chunk(
    *,
    text: str,
    source: str,
    page: int,
    section: str,
    section_title: str,
    chunk_type: str,
    chunk_index: int,
    total_chunks: int,
    is_summary: bool,
    depth: int,
    chapter: str,
    parent_section: str,
    grandparent_section: str,
    spec_number: str,
    series_id: str,
) -> dict[str, Any]:
    """
    Build a chunk dict with all required keys and defaults.

    Args:
        text: Chunk content.
        source: Filename source.
        page: Page number where the section starts.
        section: Section number.
        section_title: Section title.
        chunk_type: Chunk type classification.
        chunk_index: Index within section.
        total_chunks: Total chunks within section.
        is_summary: Whether this is a summary chunk.
        depth: Hierarchy depth.
        chapter: Top-level chapter identifier.
        parent_section: Parent section number.
        grandparent_section: Grandparent section number.
        spec_number: Spec number derived from filename.
        series_id: Series id derived from filename.

    Returns:
        Chunk dict with required schema and defaults for missing values.

    Notes:
        This function is pure and stateless.
    """
    return {
        "text": text or "",
        "source": source or "",
        "page": int(page) if page else 0,
        "section": section or "",
        "section_title": section_title or "",
        "chunk_type": chunk_type or "general",
        "chunk_index": int(chunk_index) if chunk_index is not None else 0,
        "total_chunks": int(total_chunks) if total_chunks is not None else 0,
        "is_summary": bool(is_summary),
        "depth": int(depth) if depth is not None else 0,
        "chapter": chapter or "",
        "parent_section": parent_section or "",
        "grandparent_section": grandparent_section or "",
        "spec_number": spec_number or "",
        "series_id": series_id or "",
    }


def apply_sliding_window(
    section: dict[str, Any],
    max_words: int,
    overlap: int,
    min_remaining: int,
    spec_meta: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Split an oversized section into overlapping word-window chunks.

    Args:
        section: Section dict with hierarchy metadata applied.
        max_words: Maximum words per chunk.
        overlap: Word overlap between consecutive chunks.
        min_remaining: Minimum words needed to create a trailing chunk.
        spec_meta: Dict with spec_number and series_id.

    Returns:
        List of chunk dicts for the section.

    Notes:
        Operates only within the section text and never crosses boundaries.
    """
    words = section.get("text", "").split()
    if not words:
        logger.warning("Section %s produced zero chunks (empty text)", section.get("section_number", ""))
        return []

    chunk_type = detect_chunk_type(
        section.get("section_title", ""),
        section.get("text", ""),
    )
    step = max_words - overlap
    chunks: list[dict[str, Any]] = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk_text = " ".join(words[start:end])
        chunks.append(
            _build_chunk(
                text=chunk_text,
                source=section.get("source", ""),
                page=section.get("page", 0),
                section=section.get("section_number", ""),
                section_title=section.get("section_title", ""),
                chunk_type=chunk_type,
                chunk_index=len(chunks),
                total_chunks=0,
                is_summary=False,
                depth=section.get("depth", 0),
                chapter=section.get("chapter", ""),
                parent_section=section.get("parent_section", ""),
                grandparent_section=section.get("grandparent_section", ""),
                spec_number=spec_meta.get("spec_number", ""),
                series_id=spec_meta.get("series_id", ""),
            )
        )

        start += step
        if len(words) - start < min_remaining:
            break

    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["total_chunks"] = total_chunks

    return chunks


def _first_n_words(text: str, count: int) -> str:
    """
    Return the first N words from text.

    Args:
        text: Input text to slice.
        count: Number of words to return.

    Returns:
        String containing the first N words.

    Notes:
        Returns an empty string if count is 0 or text is empty.
    """
    if count <= 0 or not text:
        return ""
    words = text.split()
    return " ".join(words[:count])



def generate_summary_chunks(
    sections_with_hierarchy: list[dict[str, Any]],
    spec_meta: dict[str, str],
    summary_words_per_child: int = 50,
) -> list[dict[str, Any]]:
    """
    Create one summary chunk per parent section.

    Args:
        sections_with_hierarchy: Output of infer_hierarchy().
        spec_meta: Dict with spec_number and series_id.
        summary_words_per_child: Number of words to take from each child snippet.

    Returns:
        List of summary chunk dicts.

    Notes:
        - Skips children with fewer than SUMMARY_MIN_CHILD_WORDS words to avoid
          embedding table-row fragments (e.g. 25-char IE name entries).
        - Hard caps summary text at SUMMARY_MAX_CHARS characters to prevent
          explosion when a section has hundreds of descendants.
        - Summary chunks are additional and do not replace leaf chunks.
    """
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for section in sections_with_hierarchy:
        parent_key = section.get("parent_section", "")
        if parent_key:
            children_by_parent.setdefault(parent_key, []).append(section)

    summary_chunks: list[dict[str, Any]] = []

    for section in sections_with_hierarchy:
        section_number = section.get("section_number", "")
        if section_number not in children_by_parent:
            continue
        children = children_by_parent[section_number]
        if not children:
            continue

        label = section_number or "Document"
        title = section.get("section_title", "")
        header = f"Section {label} - {title} covers: "

        summary_parts: list[str] = []
        total_chars = len(header)
        skipped = 0

        for child in children:
            child_text = child.get("text", "")
            child_words = child_text.split()

            if len(child_words) < SUMMARY_MIN_CHILD_WORDS:
                skipped += 1
                continue

            child_number = child.get("section_number", "")
            snippet = " ".join(child_words[:summary_words_per_child])
            entry = f"{child_number}: {snippet}".strip()

            if total_chars + len(entry) + 3 > SUMMARY_MAX_CHARS:
                summary_parts.append(
                    f"[{len(children) - len(summary_parts) - skipped} further subsections omitted]"
                )
                break

            summary_parts.append(entry)
            total_chars += len(entry) + 3

        if skipped > 0:
            logger.debug(
                "Section %s summary: skipped %s short children (table rows)",
                section_number,
                skipped,
            )

        if not summary_parts:
            summary_text = (
                f"Section {label} - {title}. "
                f"Contains {len(children)} subsections (table or parameter data)."
            )
        else:
            summary_text = header + " | ".join(summary_parts)

        first_child_page = int(children[0].get("page", 0))
        summary_chunks.append(
            _build_chunk(
                text=summary_text,
                source=section.get("source", ""),
                page=first_child_page,
                section=section_number,
                section_title=title,
                chunk_type="summary",
                chunk_index=0,
                total_chunks=1,
                is_summary=True,
                depth=section.get("depth", 0),
                chapter=section.get("chapter", ""),
                parent_section=section.get("parent_section", ""),
                grandparent_section=section.get("grandparent_section", ""),
                spec_number=spec_meta.get("spec_number", ""),
                series_id=spec_meta.get("series_id", ""),
            )
        )

    logger.debug("Generated %s summary chunks", len(summary_chunks))
    return summary_chunks


def chunk_document(
    pages: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    min_remaining_words: int,
) -> list[dict[str, Any]]:
    """
    Main entry point for hierarchical chunking of a 3GPP document.

    Args:
        pages: List of page dicts from parser.py.
        chunk_size: Maximum words per leaf chunk.
        overlap: Word overlap for sliding window fallback.
        min_remaining_words: Minimum words to create a trailing chunk.

    Returns:
        List of chunk dicts with required schema.

    Raises:
        ValueError: If chunk_size <= 0, overlap is invalid, or pages is empty.

    Notes:
        Stateless and thread-safe for concurrent container use.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1.")
    if not pages:
        raise ValueError("pages must not be empty.")

    source = pages[0].get("source", "")
    spec_meta = extract_spec_metadata(source)

    full_text, page_map = reconstruct_full_text(pages)
    sections = detect_sections(full_text, page_map, source)
    sections_with_hierarchy = infer_hierarchy(sections)

    leaf_chunks: list[dict[str, Any]] = []
    for section in sections_with_hierarchy:
        section_text = section.get("text", "")
        words = section_text.split()
        section_chunk_count = 0

        if not words:
            logger.warning(
                "Section %s produced zero chunks (empty text)",
                section.get("section_number", ""),
            )
            continue

        if len(words) <= chunk_size:
            chunk_type = detect_chunk_type(section.get("section_title", ""), section_text)
            leaf_chunks.append(
                _build_chunk(
                    text=section_text,
                    source=section.get("source", ""),
                    page=section.get("page", 0),
                    section=section.get("section_number", ""),
                    section_title=section.get("section_title", ""),
                    chunk_type=chunk_type,
                    chunk_index=0,
                    total_chunks=1,
                    is_summary=False,
                    depth=section.get("depth", 0),
                    chapter=section.get("chapter", ""),
                    parent_section=section.get("parent_section", ""),
                    grandparent_section=section.get("grandparent_section", ""),
                    spec_number=spec_meta.get("spec_number", ""),
                    series_id=spec_meta.get("series_id", ""),
                )
            )
            section_chunk_count = 1
        else:
            sliding_chunks = apply_sliding_window(
                section,
                chunk_size,
                overlap,
                min_remaining_words,
                spec_meta,
            )
            leaf_chunks.extend(sliding_chunks)
            section_chunk_count = len(sliding_chunks)

        if section_chunk_count == 0:
            logger.warning("Section %s produced zero chunks", section.get("section_number", ""))
        logger.debug(
            "Section %s produced %s chunks",
            section.get("section_number", ""),
            section_chunk_count,
        )

    summary_chunks = generate_summary_chunks(sections_with_hierarchy, spec_meta)
    all_chunks = leaf_chunks + summary_chunks

    logger.info(
        "Chunked %s: pages=%s sections=%s chunks=%s",
        source,
        len(pages),
        len(sections_with_hierarchy),
        len(all_chunks),
    )

    if logger.isEnabledFor(logging.DEBUG):
        _validate_chunks(all_chunks)

    return all_chunks


def _validate_chunks(chunks: list[dict[str, Any]]) -> None:
    """
    Internal validation of chunk schema.

    Args:
        chunks: List of chunk dicts to validate.

    Raises:
        ValueError if any chunk is missing a required key.

    Notes:
        Intended to be called only at DEBUG log level.
    """
    for index, chunk in enumerate(chunks):
        missing = REQUIRED_CHUNK_KEYS.difference(chunk.keys())
        if missing:
            raise ValueError(f"Chunk {index} missing keys: {sorted(missing)}")


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    min_remaining_words: int,
) -> list[dict[str, Any]]:
    """
    Backward-compatible wrapper around chunk_document().

    Args:
        pages: List of page dicts from parser.py.
        chunk_size: Maximum words per chunk.
        overlap: Word overlap between chunks.
        min_remaining_words: Minimum words to create a trailing chunk.

    Returns:
        List of chunk dicts with required schema.

    Notes:
        This signature is stable for pipeline.py compatibility.
    """
    return chunk_document(pages, chunk_size, overlap, min_remaining_words)


def chunk_as_flat_tables(
    pages: list[dict[str, Any]],
    chunk_size: int = 200,
    overlap: int = 40,
    min_remaining: int = 50,
) -> list[dict[str, Any]]:
    """
    Flat sliding-window chunker for table-heavy specs with no prose section structure.

    Args:
        pages: List of page dicts from parser.py.
        chunk_size: Maximum words per chunk.
        overlap: Word overlap between chunks.
        min_remaining: Minimum words to emit a trailing chunk.

    Returns:
        List of chunk dicts with required schema.

    Notes:
        Skips section detection entirely. Used for TS 38.133 which consists
        almost entirely of parameter tables whose row headers are mistakenly
        detected as section titles by the standard regex.
    """
    source = pages[0].get("source", "")
    spec_meta = extract_spec_metadata(source)
    full_text, page_map = reconstruct_full_text(pages)
    words = full_text.split()

    if not words:
        logger.warning("chunk_as_flat_tables: no words extracted from %s", source)
        return []

    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        char_offset = len(" ".join(words[:start]))
        page = _resolve_page_number(char_offset, page_map)

        chunks.append(
            _build_chunk(
                text=chunk_text,
                source=source,
                page=page,
                section="",
                section_title="",
                chunk_type="parameter",
                chunk_index=len(chunks),
                total_chunks=0,
                is_summary=False,
                depth=1,
                chapter="",
                parent_section="",
                grandparent_section="",
                spec_number=spec_meta.get("spec_number", ""),
                series_id=spec_meta.get("series_id", ""),
            )
        )
        start += step
        if len(words) - start < min_remaining:
            break

    total = len(chunks)
    for chunk in chunks:
        chunk["total_chunks"] = total

    logger.info(
        "Flat table chunking %s: words=%s chunks=%s",
        source, len(words), total,
    )
    return chunks
