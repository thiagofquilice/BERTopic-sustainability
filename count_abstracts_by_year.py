#!/usr/bin/env python3
"""Count paper abstracts by publication year."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """Lowercase ``text`` and remove English stop-words."""
    words = text.lower().split()
    filtered = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)


def read_data(path: str) -> pd.DataFrame:
    """Load abstracts and publication years from CSV/JSON/JSONL."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=ext == ".jsonl")
    else:
        raise ValueError("Unsupported file type")

    needed = {"abstract", "pub_year"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {needed}")

    df = df.dropna(subset=["abstract", "pub_year"])
    df["pub_year"] = df["pub_year"].astype(int)
    df["abstract"] = df["abstract"].astype(str).apply(clean_text)
    return df


def ensure_requirements(outdir: Path) -> None:
    """Ensure ``scikit-learn`` and ``pytest`` are listed in ``requirements.txt``."""
    req_file = outdir / "requirements.txt"
    if req_file.exists():
        lines = req_file.read_text().splitlines()
    else:
        lines = []
    updated = False
    for pkg in ["scikit-learn", "pytest"]:
        if pkg not in lines:
            lines.append(pkg)
            updated = True
    if updated:
        req_file.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_requirements(out_dir)

    df = read_data(args.input_file)
    df = df[(df["pub_year"] >= 2000) & (df["pub_year"] <= 2025)]
    if df.empty:
        print("No abstracts found in the 2000-2025 range.")
        return

    counts = df["pub_year"].value_counts().sort_index()
    out_path = out_dir / "abstract_counts.csv"
    counts.to_csv(out_path, header=["count"])

    print("Year   Count")
    for year, cnt in counts.items():
        print(f"{year:<6}{cnt:>6}")


if __name__ == "__main__":
    main()
