#!/usr/bin/env python3
"""Analyze Guardian paragraphs with BERTopic.

This script is intended for researchers or journalists who want to quickly
discover themes in articles from *The Guardian*. Even if you are unfamiliar with
topic modeling, the process is largely automated. Provide a data file with
paragraphs, run the script, and it will group similar paragraphs together and
create several helpful files and visualizations.

Input must be a CSV or JSON file containing the columns ``id``, ``paragraphs``
and ``date``. Key command-line arguments are:

``--input_file`` – path to the CSV/JSON file.
``--date_format`` – format string for parsing the ``date`` column. Default
``%Y-%m-%d``.
``--out_dir`` – directory where all outputs are saved.
``--seed`` – random seed controlling the model initialization.

Results include the trained BERTopic model, topic distributions, hierarchical
visualizations and more. Use these files to interpret how articles discuss
sustainability-related topics over time.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import plotly.io as pio


def read_data(path: str, date_format: str) -> tuple[list[str], list[datetime], list[str]]:
    """Load Guardian data from CSV or JSONL."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=ext == ".jsonl")
    else:
        raise ValueError("Unsupported file type")
    needed = {"id", "paragraphs", "date"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {needed}")
    df = df.dropna(subset=["paragraphs", "date"])
    df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
    df = df.dropna(subset=["date"])
    texts = [" ".join(p) if isinstance(p, list) else str(p) for p in df["paragraphs"]]
    return texts, df["date"].tolist(), df["id"].tolist()


def ensure_requirements(outdir: Path) -> None:
    """Write a requirements file to ``outdir`` if it does not exist."""

    reqs = [
        "pandas",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "bertopic>=0.17",
        "plotly",
        "scipy",
        "statsmodels",
    ]
    req = outdir / "requirements.txt"
    if not req.exists():
        req.write_text("\n".join(reqs))


def main() -> None:
    """Command-line interface for fitting a BERTopic model."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--date_format", default="%Y-%m-%d")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_requirements(out_dir)

    try:
        texts, dates, doc_ids = read_data(args.input_file, args.date_format)
    except Exception as exc:
        print(f"Error reading data: {exc}")
        return
    if not texts:
        print("No valid documents found.")
        return

    np.random.seed(args.seed)
    embedding_model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    umap_model = UMAP(random_state=args.seed)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=False,
        verbose=True,
        umap_model=umap_model,
    )
    topic_model.fit(texts)

    tots = topic_model.topics_over_time(texts, timestamps=dates, global_tuning=True, nr_bins=20)
    hier, _ = topic_model.hierarchical_topics(texts)
    distr, _ = topic_model.approximate_distribution(texts)

    topic_model.save(out_dir / "guardian_bertopic_model")

    rep_docs = []
    for topic in topic_model.get_topics().keys():
        for doc in topic_model.get_representative_docs(topic):
            rep_docs.append({"topic": topic, "rep_doc": doc})
    pd.DataFrame(rep_docs).to_csv(out_dir / "representative_docs.csv", index=False)
    pd.DataFrame(distr, index=doc_ids).to_csv(out_dir / "topic_distribution.csv")
    tots.to_csv(out_dir / "topics_over_time.csv", index=False)
    pd.DataFrame(hier).to_csv(out_dir / "hierarchical_topics.csv", index=False)

    fig = topic_model.visualize_hierarchy()
    pio.write_html(fig, file=out_dir / "hierarchy.html", auto_open=False)
    print(f"Analysis complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
