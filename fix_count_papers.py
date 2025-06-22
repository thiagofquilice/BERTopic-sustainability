#!/usr/bin/env python3
"""Clean a JSONL corpus of scientific papers and count papers by journal/year.

The script processes the input file line by line so it can handle large files
without loading everything into memory. It renames ``paperId`` to
``paper_id`` and ``year`` to ``pub_year``. Records missing an ``abstract`` or
publication year are skipped. Two summary TSV files are written with counts per
journal and per year.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the JSONL file")
    ap.add_argument(
        "--output",
        help="Path for the cleaned JSONL file",
    )
    ap.add_argument(
        "--journal_counts",
        help="TSV output for journal counts",
    )
    ap.add_argument(
        "--year_counts",
        help="TSV output for publication year counts",
    )
    return ap.parse_args()


def default_paths(input_path: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}.fixed.jsonl")
    )
    journal = (
        Path(args.journal_counts)
        if args.journal_counts
        else input_path.with_name(f"{input_path.stem}.journal_counts.tsv")
    )
    year = (
        Path(args.year_counts)
        if args.year_counts
        else input_path.with_name(f"{input_path.stem}.year_counts.tsv")
    )
    return out, journal, year


def process_file(in_path: Path, out_path: Path) -> tuple[int, Counter, Counter]:
    journal_counter: Counter[str] = Counter()
    year_counter: Counter[int] = Counter()
    total = 0

    # ``surrogatepass`` allows us to round-trip potentially malformed
    # Unicode that may appear in the data. Some records contain lone
    # surrogate characters which would otherwise raise ``UnicodeEncodeError``
    # on write. Using this error handler keeps those bytes intact.
    with in_path.open("r", encoding="utf-8", errors="surrogatepass") as inf, out_path.open(
        "w", encoding="utf-8", errors="surrogatepass"
    ) as outf:
        for line_no, line in enumerate(inf, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: malformed JSON on line {line_no}: {exc}", file=sys.stderr)
                continue

            if "paperId" in obj:
                obj["paper_id"] = obj.pop("paperId")
            if "year" in obj:
                obj["pub_year"] = obj.pop("year")

            abstract = obj.get("abstract")
            pub_year = obj.get("pub_year")
            if not abstract or not str(abstract).strip():
                continue
            if pub_year is None or str(pub_year).strip() == "":
                continue
            try:
                year_int = int(pub_year)
            except Exception:
                continue

            journal = obj.get("journal")
            if isinstance(journal, dict):
                name = journal.get("name") or "UNKNOWN"
            else:
                name = "UNKNOWN"

            journal_counter[name] += 1
            year_counter[year_int] += 1

            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1

    return total, journal_counter, year_counter


def write_tsv(path: Path, items: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", errors="surrogatepass") as fh:
        for key, count in items:
            fh.write(f"{key}\t{count}\n")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path, journal_path, year_path = default_paths(in_path, args)

    total, j_counts, y_counts = process_file(in_path, out_path)

    write_tsv(journal_path, sorted(j_counts.items(), key=lambda x: x[1], reverse=True))
    write_tsv(year_path, sorted(y_counts.items()))

    print(f"✅ Wrote {total} records to {out_path}")
    print("Year   Count")
    for year in sorted(y_counts):
        print(f"{year:<6}{y_counts[year]:>6}")


if __name__ == "__main__":
    main()
