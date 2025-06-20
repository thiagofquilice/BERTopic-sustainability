#!/usr/bin/env python3
"""Prepare Guardian article content into paragraph-level JSON.

The raw Guardian export files contain HTML under a ``content`` field.  This
script extracts plain-text paragraphs, normalizes publication dates and writes a
consolidated JSON file suitable for the analysis scripts in this repository.
"""
from __future__ import annotations

import argparse
import json
import html
import re
from pathlib import Path
from datetime import datetime

from bs4 import BeautifulSoup

FOOTER_PHRASES = ["Change by degrees"]


def clean_paragraph(text: str) -> str:
    """Clean paragraph text."""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    for phrase in FOOTER_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_date(date_str: str) -> str:
    """Return YYYY-MM-DD from an ISO datetime string."""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return ""


def extract_paragraphs(html_str: str) -> list[str]:
    """Return cleaned paragraphs from HTML."""
    soup = BeautifulSoup(html_str, "html.parser")
    paragraphs = []
    for p in soup.find_all("p"):
        txt = clean_paragraph(p.get_text(separator=" ", strip=True))
        if txt:
            paragraphs.append(txt)
    return paragraphs


def process_file(path: Path) -> list[dict[str, object]]:
    """Return cleaned paragraph data from a raw Guardian export file."""

    articles = json.loads(path.read_text())
    processed = []
    for art in articles:
        html_content = art.get("content")
        if not html_content:
            continue
        paragraphs = extract_paragraphs(html_content)
        if not paragraphs:
            continue
        processed.append(
            {
                "id": art.get("id"),
                "paragraphs": paragraphs,
                "date": parse_date(art.get("webPublicationDate", "")),
            }
        )
    return processed


def main() -> None:
    """Command-line interface for preparing Guardian content."""

    ap = argparse.ArgumentParser(description="Prepare Guardian HTML content")
    ap.add_argument("input_files", nargs="+", help="Input JSON files")
    ap.add_argument("--output_file", required=True, help="Output JSON file")
    args = ap.parse_args()

    all_rows: list[dict[str, object]] = []
    for fname in args.input_files:
        rows = process_file(Path(fname))
        all_rows.extend(rows)

    Path(args.output_file).write_text(json.dumps(all_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
