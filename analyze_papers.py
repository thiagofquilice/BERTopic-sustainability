#!/usr/bin/env python3
"""Analyze scientific paper abstracts with BERTopic.

If you have a dataset of academic publications and want to uncover the common
themes, this script can help. It requires minimal expertise: supply a
spreadsheet or JSON file with article abstracts and the year of publication, and
BERTopic will cluster similar abstracts together.

The input file must contain ``paper_id``, ``abstract`` and ``pub_year`` columns.
Important arguments you can change:

``--input_file`` – path to the CSV/JSON data file.
``--out_dir`` – folder for saving the model and outputs.
``--seed`` – random seed so you can reproduce the same topics.
``--years`` – optional list of publication years to include.
``--start_year`` – first year of the range to keep.
``--end_year`` – last year of the range to keep.

The script stores the trained model, topic distributions, temporal trends and an
interactive hierarchy visualization inside ``out_dir``. These outputs let you
explore how research topics evolve over the years.

Change ``START_YEAR`` / ``END_YEAR`` above or pass ``--start_year`` /
``--end_year`` when running.

Quick start
-----------
python analyze_papers.py \
  --input_file data/papers_sample.json \
  --out_dir results/papers \
  --start_year 2000 --end_year 2025
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    PartOfSpeech,
)
from sentence_transformers import SentenceTransformer
from umap import UMAP
import plotly.io as pio

# ---- YEAR FILTER SETTINGS (edit here or via CLI) ------------------
START_YEAR = 2000   # first year to keep
END_YEAR = 2025     # last  year to keep
# -------------------------------------------------------------------


def filter_by_year(year: int | datetime, start: int, end: int) -> bool:
    """Return ``True`` if ``year`` falls within ``start`` and ``end``."""
    yr = year.year if hasattr(year, "year") else int(year)
    return start <= yr <= end


def read_data(path: str) -> tuple[list[str], list[int], list[str]]:
    """Load abstracts and publication years."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".jsonl", ".json"}:
        records = []
        # ``fix_count_papers.py`` writes JSONL using ``surrogatepass`` to
        # preserve potentially malformed Unicode. Opening the file with the
        # same error handler ensures we can read those bytes back without
        # raising ``UnicodeDecodeError``.
        with open(path, "r", encoding="utf-8", errors="surrogatepass") as fh:
            if ext == ".jsonl":
                for line in fh:
                    obj = json.loads(line)
                    records.append(
                        {
                            "paper_id": obj.get("paper_id"),
                            "abstract": obj.get("abstract"),
                            "pub_year": obj.get("pub_year"),
                        }
                    )
            else:
                data = json.load(fh)
                for obj in data:
                    records.append(
                        {
                            "paper_id": obj.get("paper_id"),
                            "abstract": obj.get("abstract"),
                            "pub_year": obj.get("pub_year"),
                        }
                    )
        df = pd.DataFrame.from_records(records)
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
    ap.add_argument("--start_year", type=int, default=START_YEAR)
    ap.add_argument("--end_year", type=int, default=END_YEAR)
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

    try:
        texts, years, ids = read_data(args.input_file)
    except Exception as exc:
        print(f"Error reading data: {exc}")
        return
    if not texts:
        print("No valid documents found.")
        return

    filtered = [
        (t, y, i)
        for t, y, i in zip(texts, years, ids)
        if filter_by_year(y, args.start_year, args.end_year)
    ]
    if args.years:
        yrs = set(args.years)
        filtered = [f for f in filtered if f[1] in yrs]
    if not filtered:
        print("No papers found for the selected years.")
        return

    texts, years, ids = map(list, zip(*filtered))

    # Prepare year labels for grouping by calendar year
    year_labels = pd.to_datetime(years, format="%Y").year.tolist()

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
    topic_model.fit(texts)

    tots = topic_model.topics_over_time(
        texts,
        timestamps=year_labels,
        global_tuning=False,
    )
    hier = topic_model.hierarchical_topics(texts)
    distr, _ = topic_model.approximate_distribution(texts)

    topic_model.save(out_dir / "papers_bertopic_model")

    rep_docs = []
    for topic in topic_model.get_topics().keys():
        for doc in topic_model.get_representative_docs(topic):
            rep_docs.append({"topic": topic, "rep_doc": doc})
    pd.DataFrame(rep_docs).to_csv(out_dir / "representative_docs.csv", index=False)
    pd.DataFrame(distr, index=ids).to_csv(out_dir / "topic_distribution.csv")
    tots.to_csv(out_dir / "topics_over_year.csv", index=False)
    pd.DataFrame(hier).to_csv(out_dir / "hierarchical_topics.csv", index=False)

    fig = topic_model.visualize_hierarchy()
    pio.write_html(fig, file=out_dir / "hierarchy.html", auto_open=False)
    print(f"Analysis complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
