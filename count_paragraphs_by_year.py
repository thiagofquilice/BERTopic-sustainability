#!/usr/bin/env python3
"""Count paragraphs by publication year."""
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


def read_data(path: str, date_format: str) -> pd.DataFrame:
    """Load paragraphs and years from CSV/JSON/JSONL."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=ext == ".jsonl")
    else:
        raise ValueError("Unsupported file type")

    needed = {"paragraphs", "date"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {needed}")

    df = df.dropna(subset=["paragraphs", "date"])
    df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
    df = df.dropna(subset=["date"])

    records = []
    for _, row in df.iterrows():
        year = row["date"].year
        paragraphs = row["paragraphs"]
        if isinstance(paragraphs, list):
            for p in paragraphs:
                records.append({"year": year, "text": clean_text(str(p))})
        else:
            records.append({"year": year, "text": clean_text(str(paragraphs))})
    return pd.DataFrame(records)


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
    ap.add_argument("--date_format", default="%Y-%m-%d")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_requirements(out_dir)

    df = read_data(args.input_file, args.date_format)
    df = df[(df["year"] >= 2000) & (df["year"] <= 2025)]
    if df.empty:
        print("No paragraphs found in the 2000-2025 range.")
        return

    counts = df["year"].value_counts().sort_index()
    out_path = out_dir / "paragraph_counts.csv"
    counts.to_csv(out_path, header=["count"])

    print("Year   Count")
    for year, cnt in counts.items():
        print(f"{year:<6}{cnt:>6}")


if __name__ == "__main__":
    main()
