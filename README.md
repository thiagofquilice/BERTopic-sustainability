# BERTopic Sustainability


This project demonstrates how to explore large amounts of text with
[BERTopic](https://github.com/MaartenGr/BERTopic). BERTopic builds on
transformer-based embeddings and clustering to create interpretable topics with minimal
preprocessing. It offers interactive visualizations and multilingual support.
For a deeper overview see the [official BERTopic documentation](https://maartengr.github.io/BERTopic/index.html).
The example datasets include Guardian news paragraphs and scientific paper abstracts.
Each dataset is modeled separately and the topics can then be compared.


## Contents
- `analyze_guardian.py` – Fit a BERTopic model on Guardian paragraphs. Arguments
  include `--input_file`, `--out_dir`, optional `--date_format`, `--seed` and optional `--years`.
- `analyze_papers.py` – Fit a BERTopic model on scientific paper abstracts.
  Arguments include `--input_file`, `--out_dir`, optional `--seed` and optional `--years`.
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

## Data format

Your input files can be CSV, JSON or JSONL. The Guardian dataset must provide
columns named `id`, `paragraphs` and `date` where `paragraphs` is a list of
strings and `date` follows the format supplied by `--date_format` (default
`%Y-%m-%d`). The papers dataset must include `paper_id`, `abstract` and
`pub_year` columns.

## Running on a remote server

The snippet below demonstrates how to configure a virtual environment and run
`analyze_guardian.py` on a remote machine. Adjust the paths for your own setup.

1. **Create and activate a virtual environment**

   ```bash
   cd /home/thiago
   python3 -m venv guardian-venv
   source guardian-venv/bin/activate
   ```

2. **Install the requirements**

   ```bash
   pip install -r /home/thiago/requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Run the analysis**

   Activate the environment if needed and optionally start a `tmux` session so
   the command continues running after you disconnect:

   ```bash
   source /home/thiago/guardian-venv/bin/activate
   tmux new -s guardian
   ```

   Then execute the script:

   ```bash
   python /home/thiago/analyze_guardian.py \
       --input_file /home/thiago/processed_guardian_news.json \
       --out_dir /home/thiago/guardian_results
   ```

   You can now close the terminal or detach from `tmux` while the analysis runs.

4. **Reconnect to a running session**

   Detach from `tmux` with `Ctrl+b d` and reattach later with:

   ```bash
   tmux attach -t guardian
   ```

Once the environment is set up you can reuse it for future runs without
recreating the virtual environment or reinstalling dependencies.
