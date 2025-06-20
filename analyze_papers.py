#!/usr/bin/env python3
"""Analyze scientific paper abstracts with BERTopic.

The script reads a CSV or JSON file of scientific publication metadata, fits a
BERTopic model on the abstract text and saves model outputs.  Expected input
columns are ``paper_id``, ``abstract`` and ``pub_year``.  Various CSV files and
HTML visualizations are written to the chosen output directory.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.io as pio


def read_data(path: str) -> tuple[list[str], list[int], list[str]]:
    """Load abstracts and publication years."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=ext == ".jsonl")
    else:
        raise ValueError("Unsupported file type")
    needed = {"paper_id", "abstract", "pub_year"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {needed}")
    df = df.dropna(subset=["abstract", "pub_year"])
    df["pub_year"] = df["pub_year"].astype(int)
    return df["abstract"].tolist(), df["pub_year"].tolist(), df["paper_id"].tolist()


def ensure_requirements(outdir: Path) -> None:
    """Create ``requirements.txt`` in ``outdir`` if missing."""

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
    """Entry point for the papers analysis command-line tool."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_requirements(out_dir)

    try:
        texts, years, ids = read_data(args.input_file)
    except Exception as exc:
        print(f"Error reading data: {exc}")
        return
    if not texts:
        print("No valid documents found.")
        return

    np.random.seed(args.seed)
    embedding_model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        calculate_probabilities=False,
        verbose=True,
        random_state=args.seed,
    )
    topic_model.fit(texts)

    tots = topic_model.topics_over_time(texts, timestamps=years, global_tuning=True, nr_bins=None)
    hier, _ = topic_model.hierarchical_topics(texts)
    distr, _ = topic_model.approximate_distribution(texts)

    topic_model.save(out_dir / "papers_bertopic_model")

    rep_docs = []
    for topic in topic_model.get_topics().keys():
        for doc in topic_model.get_representative_docs(topic):
            rep_docs.append({"topic": topic, "rep_doc": doc})
    pd.DataFrame(rep_docs).to_csv(out_dir / "representative_docs.csv", index=False)
    pd.DataFrame(distr, index=ids).to_csv(out_dir / "topic_distribution.csv")
    tots.to_csv(out_dir / "topics_over_time.csv", index=False)
    pd.DataFrame(hier).to_csv(out_dir / "hierarchical_topics.csv", index=False)

    fig = topic_model.visualize_hierarchy()
    pio.write_html(fig, file=out_dir / "hierarchy.html", auto_open=False)
    print(f"Analysis complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
