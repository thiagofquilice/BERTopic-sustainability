from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from count_abstracts_by_year import clean_text, read_data as read_abstracts
from count_paragraphs_by_year import read_data as read_paragraphs


def create_paper_file(tmp_path: Path) -> Path:
    records = [
        {"paper_id": "a", "abstract": "The first", "pub_year": 1999},
        {"paper_id": "b", "abstract": "Second paper", "pub_year": 2000},
        {"paper_id": "c", "abstract": "Third", "pub_year": 2025},
        {"paper_id": "d", "abstract": "Fourth", "pub_year": 2026},
    ]
    path = tmp_path / "papers.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def create_guardian_file(tmp_path: Path) -> Path:
    records = [
        {"id": 1, "paragraphs": ["The start"], "date": "1999-01-02"},
        {"id": 2, "paragraphs": ["More news"], "date": "2000-05-10"},
        {"id": 3, "paragraphs": ["End"], "date": "2025-07-05"},
        {"id": 4, "paragraphs": ["Later"], "date": "2026-08-10"},
    ]
    path = tmp_path / "guardian.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh)
    return path


def test_year_filters(tmp_path: Path) -> None:
    paper_path = create_paper_file(tmp_path)
    df = read_abstracts(str(paper_path))
    df = df[(df["pub_year"] >= 2000) & (df["pub_year"] <= 2025)]
    assert set(df["pub_year"]) == {2000, 2025}

    guard_path = create_guardian_file(tmp_path)
    df_g = read_paragraphs(str(guard_path), "%Y-%m-%d")
    df_g = df_g[(df_g["year"] >= 2000) & (df_g["year"] <= 2025)]
    assert set(df_g["year"]) == {2000, 2025}


def test_clean_text() -> None:
    assert clean_text("The quick brown fox") == "quick brown fox"
