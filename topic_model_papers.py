#!/usr/bin/env python3
"""Fit a BERTopic model on a JSONL corpus of scientific papers.

Each line of the input file must contain at least ``paperId``, ``title`` or
``abstract`` and ``year``. The file is processed lazily, applying minimal text
cleaning (punctuation removal, stop word filtering and short word removal).

The command line now mirrors :mod:`analyze_guardian.py`.  You can control the
random seed with ``--seed`` and optionally restrict the analysis to specific
publication years using ``--years``.  After fitting the model several outputs
are written to ``out_dir``:

``papers_bertopic_model``
    Directory containing the saved BERTopic model.
``representative_docs.csv``
    Representative documents for each topic.
``topic_distribution.csv``
    Approximate topic distribution for each paper.
``topics_over_year.csv``
    Topic frequencies aggregated by publication year.
``hierarchical_topics.csv``
    Table describing the hierarchical topic structure.
``hierarchy.html``
    Interactive Plotly figure of the topic hierarchy.
``docs_topics.csv``
    Mapping of paper IDs and years to their assigned topic.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Tuple
from datetime import datetime

import numpy as np

import pandas as pd
from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    PartOfSpeech,
)
from sentence_transformers import SentenceTransformer
from umap import UMAP
import plotly.io as pio
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOPWORDS = set(ENGLISH_STOP_WORDS)
RE_PUNCT = re.compile(r"[^\w\s]")


def preprocess(text: str) -> str:
    """Lowercase ``text`` and remove stop words and punctuation."""

    text = text.lower()
    text = RE_PUNCT.sub(" ", text)
    tokens = [w for w in text.split() if len(w) >= 3 and w not in STOPWORDS]
    return " ".join(tokens)


def iter_texts(path: Path) -> Tuple[list[str], list[str], list[int]]:
    """Yield processed text, IDs and publication years from ``path``."""

    docs: list[str] = []
    ids: list[str] = []
    years: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("abstract") or obj.get("title") or ""
            year_val = obj.get("year")
            try:
                year = int(year_val)
            except (TypeError, ValueError):
                year = None
            if not text or len(text) <= 50 or year is None:
                continue
            docs.append(preprocess(text))
            ids.append(obj.get("paperId", ""))
            years.append(year)
    return docs, ids, years


def ensure_requirements(outdir: Path) -> None:
    """Write a simple requirements file if missing."""

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
    """CLI entry point."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL file")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Only analyze papers from these publication years",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_requirements(out_dir)

    docs, ids, years = iter_texts(Path(args.input))
    if not docs:
        print("No valid documents found.")
        return

    if args.years:
        yrs = set(args.years)
        filtered = [
            (d, i, y)
            for d, i, y in zip(docs, ids, years)
            if y in yrs
        ]
        if not filtered:
            print("No papers found for the selected years.")
            return
        docs, ids, years = map(list, zip(*filtered))

    np.random.seed(args.seed)

    embedding_model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
        "POS": PartOfSpeech("en_core_web_sm"),
    }

    umap_model = UMAP(random_state=args.seed)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=True,
        umap_model=umap_model,
    )
    topics, _ = topic_model.fit_transform(docs)

    dates = [datetime(year=y, month=1, day=1) for y in years]
    unique_years = sorted({y for y in years})

    tots = topic_model.topics_over_time(
        docs,
        timestamps=dates,
        global_tuning=False,
        nr_bins=len(unique_years),
    )
    hier = topic_model.hierarchical_topics(docs)
    distr, _ = topic_model.approximate_distribution(docs)

    rep_docs = []
    for topic in topic_model.get_topics().keys():
        for doc in topic_model.get_representative_docs(topic):
            rep_docs.append({"topic": topic, "rep_doc": doc})
    pd.DataFrame(rep_docs).to_csv(out_dir / "representative_docs.csv", index=False)

    pd.DataFrame(distr, index=ids).to_csv(out_dir / "topic_distribution.csv")
    tots.to_csv(out_dir / "topics_over_year.csv", index=False)
    pd.DataFrame(hier).to_csv(out_dir / "hierarchical_topics.csv", index=False)
    pd.DataFrame({"paperId": ids, "year": years, "topic_id": topics}).to_csv(
        out_dir / "docs_topics.csv",
        index=False,
    )

    topic_model.save(out_dir / "papers_bertopic_model")

    fig = topic_model.visualize_hierarchy()
    pio.write_html(fig, file=out_dir / "hierarchy.html", auto_open=False)
    print(f"Analysis complete. Results saved to {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
