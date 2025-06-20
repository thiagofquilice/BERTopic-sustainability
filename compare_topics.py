#!/usr/bin/env python3
"""Compare topics from two BERTopic models.

The script loads two trained BERTopic models along with their topic frequency
over time.  It computes cosine and Jaccard similarities between the topic
embeddings and top words, respectively, and performs a simple temporal
correlation analysis.  Results are stored as CSV files and HTML
visualizations in the specified output directory.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from statsmodels.tsa.stattools import grangercausalitytests


def load_model(path: str) -> BERTopic:
    """Load a BERTopic model from ``path``."""

    return BERTopic.load(path)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarity between two embedding matrices."""

    return cosine_similarity(a, b)


def jaccard_matrix(mod_a: BERTopic, mod_b: BERTopic, topn: int = 20) -> pd.DataFrame:
    """Calculate Jaccard similarity of top words between two models."""

    vocab_a = {t: {w for w, _ in mod_a.get_topic(t)[:topn]} for t in mod_a.get_topics()}
    vocab_b = {t: {w for w, _ in mod_b.get_topic(t)[:topn]} for t in mod_b.get_topics()}
    jac = np.zeros((len(vocab_a), len(vocab_b)))
    for i, ta in enumerate(vocab_a):
        for j, tb in enumerate(vocab_b):
            inter = len(vocab_a[ta] & vocab_b[tb])
            union = len(vocab_a[ta] | vocab_b[tb])
            jac[i, j] = inter / union if union else 0.0
    return pd.DataFrame(jac, index=[f"A_{t}" for t in vocab_a], columns=[f"B_{t}" for t in vocab_b])


def top_matches(mat: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
    """Return the top ``n`` matching topic pairs for each row of ``mat``."""

    pairs = []
    for i in range(mat.shape[0]):
        idx = np.argsort(mat[i])[::-1][:n]
        for j in idx:
            pairs.append((i, j, mat[i, j]))
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def main() -> None:
    """Run the topic comparison workflow from the command line."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--topics_a", required=True)
    ap.add_argument("--topics_b", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_a = load_model(args.model_a)
    model_b = load_model(args.model_b)

    mat = cosine_matrix(model_a.topic_embeddings_, model_b.topic_embeddings_)
    cos_df = pd.DataFrame(mat, index=[f"A_{i}" for i in range(mat.shape[0])], columns=[f"B_{j}" for j in range(mat.shape[1])])
    cos_df.to_csv(out_dir / "cosine_similarity.csv")

    print("Top cosine similarities:")
    for i, j, val in top_matches(mat)[:10]:
        print(f"A topic {i} ↔ B topic {j} : {val:.3f}")

    jac_df = jaccard_matrix(model_a, model_b)
    jac_df.to_csv(out_dir / "jaccard_similarity.csv")

    fig = px.imshow(cos_df, labels=dict(x="Model B", y="Model A", color="cosine"))
    fig.write_html(out_dir / "similarity_heatmap.html", auto_open=False)

    # Temporal series comparison using topics_over_time files
    ts_a = pd.read_csv(args.topics_a).pivot_table(index="Timestamp", columns="Topic", values="Frequency", aggfunc="sum").fillna(0)
    ts_b = pd.read_csv(args.topics_b).pivot_table(index="Timestamp", columns="Topic", values="Frequency", aggfunc="sum").fillna(0)

    temporal = []
    pairs = top_matches(mat)[:5]
    for i, j, val in pairs:
        if i >= ts_a.shape[1] or j >= ts_b.shape[1]:
            continue
        s1 = ts_a.iloc[:, i]
        s2 = ts_b.iloc[:, j]
        try:
            granger = grangercausalitytests(pd.concat([s1, s2], axis=1).dropna(), maxlag=6, verbose=False)
            pvals = [round(granger[l][0]['ssr_chi2test'][1], 4) for l in range(1, 7)]
        except Exception:
            pvals = [np.nan] * 6
        temporal.append({"topic_A": i, "topic_B": j, "max_corr": s1.corr(s2), "granger_pvals": pvals})
    pd.DataFrame(temporal).to_csv(out_dir / "temporal_correlation.csv", index=False)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
