# BERTopic Sustainability

This repo contains example scripts for analyzing two text datasets using [BERTopic](https://github.com/MaartenGr/BERTopic). The datasets include Guardian news paragraphs and scientific paper abstracts. Each dataset is modeled separately and then compared.

## Contents
- `analyze_guardian.py` – Fit a BERTopic model on Guardian paragraphs.
- `analyze_papers.py` – Fit a BERTopic model on scientific paper abstracts.
- `compare_topics.py` – Compare two saved BERTopic models.
- `data/` – Small sample datasets (`guardian_sample.json`, `papers_sample.json`).

Install dependencies with:

```bash
pip install -r requirements.txt
```

Each script will also create a `requirements.txt` in the output directory if it does not already exist.



## Usage

```bash
python analyze_guardian.py --input_file data/guardian_sample.json --out_dir results/guardian
python analyze_papers.py --input_file data/papers_sample.json --out_dir results/papers
python compare_topics.py \
  --model_a results/guardian/guardian_bertopic_model \
  --model_b results/papers/papers_bertopic_model \
  --topics_a results/guardian/topics_over_time.csv \
  --topics_b results/papers/topics_over_time.csv \
  --out_dir results/compare
```

These commands assume you have installed the packages in `requirements.txt` and have the necessary dependencies for BERTopic.

## Preparing Guardian Data

Raw Guardian exports such as `guardian_news_sustainability_with_content_1to5.json` contain HTML snippets in a `content` field. To combine several of these files into a clean dataset you can run:

```bash
python prepare_guardian.py guardian_news_sustainability_with_content_*.json \
  --out_file data/guardian_all.json
```

`prepare_guardian.py` removes the HTML, breaks articles into paragraphs and writes a single JSON file. The output records use the keys `id`, `paragraphs` and `date`, matching the format expected by `analyze_guardian.py`.
