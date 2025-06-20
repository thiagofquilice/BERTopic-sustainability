# BERTopic Sustainability

This project demonstrates how to explore large amounts of text with
[BERTopic](https://github.com/MaartenGr/BERTopic). The example datasets include
Guardian news paragraphs and scientific paper abstracts. Each dataset is modeled
separately and the topics can then be compared.

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
python analyze_guardian.py --input_file data/guardian_sample.json --out_dir results/guardian --years 2020 2021
python analyze_papers.py --input_file data/papers_sample.json --out_dir results/papers --years 2019 2020
python compare_topics.py \
  --model_a results/guardian/guardian_bertopic_model \
  --model_b results/papers/papers_bertopic_model \
  --topics_a results/guardian/topics_over_time.csv \
  --topics_b results/papers/topics_over_time.csv \
  --out_dir results/compare
```

These commands assume you have installed the packages in `requirements.txt` and have the necessary dependencies for BERTopic.

## Preparing Guardian Data

Raw Guardian exports such as `guardian_news_sustainability_with_content_1to5.json` (not included due to size) contain HTML snippets in a `content` field. To combine several of these files into a clean dataset you can run:

```bash
python prepare_guardian.py --input_dir 2_news_json_files \
  --output_file data/guardian_all.json
```

By default the script reads all `.json` files inside `2_news_json_files`. You
can still pass individual paths as positional `input_files` if needed. The
`--output_file` argument specifies where to write the cleaned and merged dataset.
The resulting JSON contains the keys `id`, `paragraphs` and `date`, ready for
use with `analyze_guardian.py`.

## Extra step for richer topic labels
Some scripts optionally use POS filtering for cleaner keywords.
Install spaCy and download the English model once:

```bash
pip install -U spacy
python -m spacy download en_core_web_sm
```

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

   To filter by year:

   ```bash
   python analyze_guardian.py --input_file data/processed_guardian_news.json \
       --out_dir results/guardian --years 2019 2021
   python analyze_papers.py --input_file data/papers_sample.json \
       --out_dir results/papers --years 2019 2020
   ```

4. **Reconnect to a running session**

   Detach from `tmux` with `Ctrl+b d` and reattach later with:

   ```bash
   tmux attach -t guardian
   ```

Once the environment is set up you can reuse it for future runs without
recreating the virtual environment or reinstalling dependencies.
