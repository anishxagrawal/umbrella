from __future__ import annotations

import os
import re
from typing import Any

import fitz


def clean_text(text: str) -> str:
    """Remove repeating 3GPP headers and clean whitespace."""
    text = re.sub(r"3GPP\s*\n", "", text)
    text = re.sub(r"3GPP TS[\d\.\s\w\(\)\-]+\n", "", text)
    text = re.sub(r"Release \d+\s*\n", "", text)
    text = re.sub(r"Figure[\s\d\w\.\-]+:.*?\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_useless_page(text: str) -> bool:
    """Identify pages that add no knowledge value."""
    stripped = text.strip()
    if len(stripped) < 100:
        return True
    if stripped.count("...") > 10:
        return True
    if "RP-" in stripped and stripped.count("RP-") > 5:
        return True
    if "Postal address" in stripped or "Copyright Notification" in stripped:
        return True
    return False


def extract_section(text: str) -> str | None:
    """Try to find the section number at the start of the page."""
    match = re.search(r"^\s*(\d+\.\d+[\.\d]*)\s+\w", text, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def parse_pdf(pdf_path: str) -> tuple[list[dict[str, Any]], int]:
    """Parse a PDF and return cleaned pages with metadata."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    source = os.path.basename(pdf_path)
    pages: list[dict[str, Any]] = []
    skipped = 0

    for page_num, page in enumerate(doc):
        text = page.get_text()
        cleaned = clean_text(text)
        if is_useless_page(cleaned):
            skipped += 1
            continue

        section = extract_section(cleaned)
        pages.append(
            {
                "text": cleaned,
                "page": page_num + 1,
                "source": source,
                "section": section,
            }
        )

    return pages, skipped
