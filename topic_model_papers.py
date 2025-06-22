#!/usr/bin/env python3
"""Fit a BERTopic model on a JSONL corpus of scientific papers.

This script reads a JSON-Lines file where each line contains a paper record with
at least ``paperId``, ``title`` and ``abstract`` fields. ``journal`` and ``year``
are ignored. It loads the file lazily, keeping only the text needed for topic
modeling. If the abstract is missing, the title is used instead. Records with
fewer than 50 characters are skipped. No text preprocessing is performed before
fitting the model.

The resulting topics are written to ``topics.csv`` and the mapping of document
IDs to their assigned topic is written to ``docs_topics.csv``. The BERTopic model
is saved to ``bertopic_model`` inside the output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from bertopic import BERTopic


def iter_texts(path: Path) -> Tuple[list[str], list[str]]:
    """Yield text and IDs from ``path``."""

    docs: list[str] = []
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get("abstract") or obj.get("title") or "").strip()
            if not text or len(text) <= 50:
                continue
            docs.append(text)
            ids.append(obj.get("paperId", ""))
    return docs, ids


def ensure_requirements(outdir: Path) -> None:
    """Write a simple requirements file if missing."""

    reqs = [
        "pandas",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "bertopic>=0.17",
    ]
    req = outdir / "requirements.txt"
    if not req.exists():
        req.write_text("\n".join(reqs))


def main() -> None:
    """CLI entry point."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL file")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_requirements(out_dir)

    docs, ids = iter_texts(Path(args.input))
    if not docs:
        print("No valid documents found.")
        return

    topic_model = BERTopic(verbose=True)
    topics, _ = topic_model.fit_transform(docs)

    # topics.csv with top 10 words
    rows = []
    for tid in sorted(set(topics)):
        words = topic_model.get_topic(tid)[:10]
        rows.append({"topic_id": tid, "top_words": " ".join(w for w, _ in words)})
    pd.DataFrame(rows).to_csv(out_dir / "topics.csv", index=False)

    # docs_topics.csv mapping paperId to topic
    pd.DataFrame({"paperId": ids, "topic_id": topics}).to_csv(
        out_dir / "docs_topics.csv", index=False
    )

    topic_model.save(out_dir / "bertopic_model")
    print(f"Saved model and CSV files to {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
