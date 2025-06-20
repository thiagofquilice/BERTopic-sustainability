# BERTopic Sustainability

This project demonstrates how to explore large amounts of text with
[BERTopic](https://github.com/MaartenGr/BERTopic). The example datasets include
Guardian news paragraphs and scientific paper abstracts. Each dataset is modeled
separately and the topics can then be compared.

## Contents
- `analyze_guardian.py` – Fit a BERTopic model on Guardian paragraphs. Arguments
  include `--input_file`, `--out_dir`, optional `--date_format` and `--seed`.
- `analyze_papers.py` – Fit a BERTopic model on scientific paper abstracts.
  Arguments include `--input_file`, `--out_dir` and optional `--seed`.
- `compare_topics.py` – Compare two saved BERTopic models using `--model_a`,
  `--model_b`, `--topics_a`, `--topics_b` and `--out_dir`.
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
  --output_file data/guardian_all.json
```

Here `input_files` is a list of one or more raw JSON exports from the Guardian
(wildcards are allowed). The `--output_file` argument specifies where to write
the cleaned and merged dataset. The resulting JSON contains the keys `id`,
`paragraphs` and `date`, ready for use with `analyze_guardian.py`.
