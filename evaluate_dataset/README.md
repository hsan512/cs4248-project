# evaluate_dataset

## What we have done so far

- Built `run_sentiment_labeling.py` to use local vLLM (`openai/gpt-oss-120b`) as an LLM sentiment annotator.
- Added support for `sst` (SST-5 collapsed to 3 classes), and `imdb` (binary).
- Run labeling for `gpt-120b-medium` (MXFP4 quantization) on both datasets, and analyzed outputs with `analyze_label.py`. This is used for the Macro F1 score and accuracy used in the paper 


Current outputs in this folder:
- `sentiment_results/labeled_sst_gpt-120b-medium.csv`
- `sentiment_results/labeled_imdb_gpt-120b-medium.csv`

## Setup (vLLM GPT-OSS-120B)

1. Start the vLLM server with your SLURM script:

```bash
cd evaluate_dataset
sbatch scrum.sh
```

2. Align the API port between server and labeling script:
- `scrum.sh` serves on `--port 8050`
- `run_sentiment_labeling.py` currently uses `API_BASE_URL = "http://localhost:8000/v1"`

Use one consistent port (either change script to `8050`, or serve vLLM on `8000`).

3. Install Python dependencies in your run environment:

```bash
pip install openai pandas scikit-learn datasets tqdm
```

## Run labeling

```bash
cd evaluate_dataset
python run_sentiment_labeling.py --model gpt-120b-medium --datasets sst imdb
```

Other common runs:

```bash
python run_sentiment_labeling.py
python run_sentiment_labeling.py --datasets sst2
python run_sentiment_labeling.py --n-sst 2000 --n-imdb 5000 --model gpt-120b-high
```

Generated files (`sentiment_results/`):
- `labeled_<dataset>_<model>.csv`
- `metrics_summary_<model>.csv`
- `checkpoint_<dataset>_<model>.jsonl`

## Analyze outputs

```bash
cd evaluate_dataset
python analyze_label.py
```

Custom files:

```bash
python analyze_label.py --sst sentiment_results/labeled_sst_gpt-120b-medium.csv --imdb sentiment_results/labeled_imdb_gpt-120b-medium.csv
```

This prints detailed metrics and writes:
- `sentiment_results/analysis_summary.csv`

