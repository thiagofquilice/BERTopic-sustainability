#!/usr/bin/env python3
"""Consolidate BERTopic models into a single CSV for comparison.

Given multiple directories containing saved BERTopic models, this script
loads each model and extracts the top words for every topic. The output is
one CSV file listing the model name, topic ID and the top words.

Example
-------
python consolidate_topics.py \
  --models results/guardian/guardian_bertopic_model \
  results/papers/papers_bertopic_model \
  --out_file consolidated_topics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from bertopic import BERTopic


def extract_topics(model_path: str, top_n: int) -> list[dict[str, str]]:
    """Return a list of topic dictionaries from ``model_path``."""

    model = BERTopic.load(model_path)
    records: list[dict[str, str]] = []
    for topic_id in model.get_topics().keys():
        words = [w for w, _ in model.get_topic(topic_id)[:top_n]]
        records.append(
            {
                "model": Path(model_path).name,
                "topic_id": topic_id,
                "top_words": ", ".join(words),
            }
        )
    return records


def iter_models(paths: Iterable[str], top_n: int) -> list[dict[str, str]]:
    """Extract topics from all model paths."""

    all_records: list[dict[str, str]] = []
    for path in paths:
        try:
            all_records.extend(extract_topics(path, top_n))
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"Could not load model at {path}: {exc}")
    return all_records


def main() -> None:
    """Command-line interface for topic consolidation."""

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to BERTopic model directories",
    )
    ap.add_argument("--out_file", required=True, help="CSV file to write")
    ap.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of words per topic",
    )
    args = ap.parse_args()

    records = iter_models(args.models, args.top_n)
    if not records:
        print("No topics extracted.")
        return

    df = pd.DataFrame(records)
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_file, index=False)
    print(f"Consolidated topics saved to {args.out_file}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
