#!/usr/bin/env python3
"""Generate a text representation of a BERTopic hierarchy."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bertopic import BERTopic


def locate_csv(model_path: Path) -> Path:
    """Return path to hierarchical_topics.csv near ``model_path``."""
    base = model_path
    if base.is_file():
        base = base.parent
    candidates = [base / "hierarchical_topics.csv", base.parent / "hierarchical_topics.csv"]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("Could not locate hierarchical_topics.csv")


def main() -> None:
    """CLI entry point for printing the topic tree."""
    ap = argparse.ArgumentParser(description="Visualize BERTopic hierarchy as text")
    ap.add_argument("--model_path", required=True, help="Saved BERTopic model directory, zip or pkl")
    ap.add_argument(
        "--hier_csv",
        help="Path to hierarchical_topics.csv. If omitted, search next to the model",
    )
    ap.add_argument("--output_file", required=True, help="File to write the tree")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    csv_path = Path(args.hier_csv) if args.hier_csv else locate_csv(model_path)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topic_model = BERTopic.load(str(model_path))
    hierarchical_topics = pd.read_csv(csv_path)
    tree = topic_model.get_topic_tree(hierarchical_topics)

    output_path.write_text(tree)
    print(f"Topic tree saved to {output_path}")


if __name__ == "__main__":
    main()
