# CS4248 Project – Tweet Sentiment Intensification

## Overview
This project is part of the CS4248 Natural Language Processing module at the National University of Singapore (NUS). The objective is to develop a system that intensifies the sentiment of a given tweet while preserving its original semantic meaning.

Unlike traditional sentiment classification, this work addresses a controlled text generation problem, where the model rewrites input text to produce a more emotionally expressive version conditioned on a target sentiment direction.

---

## Objectives
- Generate sentiment-intensified text conditioned on a target polarity (positive/negative)
- Preserve the semantic meaning and contextual intent of the original tweet
- Increase the strength and clarity of sentiment expression
- Evaluate model outputs using both automatic metrics and qualitative analysis

---

## Methodology

### 1. Text Preprocessing
The preprocessing pipeline is designed to handle noisy social media text. This includes normalization of URLs, user mentions, and special tokens, as well as handling emojis and emoticons to better preserve sentiment signals while reducing vocabulary sparsity.

### 2. Supervised Fine-Tuning (SFT)
The base generative model is fine-tuned on the sentiment intensification task, learning to rewrite input tweets into more emotionally expressive versions. Generation is conditioned on explicit sentiment direction tokens (e.g., [POS], [NEG]).

### 3. Reinforcement Learning (PPO-based)
We perform reinforcement fine-tuning using an Actor–Critic framework based on Proximal Policy Optimization (PPO). The policy is parameterized by the base model, optionally augmented with parameter-efficient adapters.

For each input, the model generates an output via stochastic decoding, and a scalar reward is derived from a sentiment classifier. To ensure training stability, the objective employs a clipped surrogate loss and Generalized Advantage Estimation (GAE), along with KL regularization against a reference model. This enables fine-grained token-level credit assignment and stabilizes policy updates.

### 4. Evaluation
We evaluate the system using a combination of:
- Classification metrics (e.g., F1 score)
- Generation metrics (e.g., BLEU, ROUGE, BERTScore)
- Perplexity for distribution alignment analysis
- Qualitative inspection of generated outputs

---

## Experimental Components

### Task-Adaptive Pretraining (TAPT)
We experiment with additional domain-specific pretraining on tweet data to assess whether adapting the language model to informal text improves downstream generation quality.

### Parameter-Efficient Fine-Tuning (LoRA)
We explore Low-Rank Adaptation (LoRA) as a parameter-efficient alternative to full fine-tuning, enabling reduced training cost while maintaining competitive performance.

---

## Baselines
Several baseline sentiment classifiers are included in [classifier/baselines/](classifier/baselines/) to benchmark against the main approach:

- **Logistic Regression** ([logistic_regression.py](classifier/baselines/logistic_regression.py)) – TF-IDF features with a linear classifier
- **Naive Bayes** ([naive_bayes.py](classifier/baselines/naive_bayes.py)) – Multinomial NB over bag-of-words features
- **RNN** ([rnn.py](classifier/baselines/rnn.py)) – Trainable embeddings fed into a recurrent classifier
- **GloVe + LSTM** ([glove_lstm.py](classifier/baselines/glove_lstm.py)) – Pretrained GloVe embeddings with an LSTM classifier

---
## Running Supporting Analysis Scripts
We recommend reading the README files for some of the submodule for detailed instructions on how to run the code and what each script does. Below is a brief overview of the main components.


### Classifier Analysis Utilities
Submodule README: [classifier_analysis/README.md](classifier_analysis/README.md)

This folder contains analysis scripts for understanding dataset characteristics and model prediction behavior.

How to run (from repo root):
```bash
python classifier_analysis/corpus_analysis.py
python classifier_analysis/results_analysis.py
```

What each script does:
- `classifier_analysis/corpus_analysis.py`: analyzes corpus-level patterns and highlights interesting examples.
- `classifier_analysis/results_analysis.py`: analyzes prediction outputs on the test set and reports useful examples/statistics.

### Evaluation Dataset Labeling Pipeline

LLM labelling using a GPT OSS 120B model with medium thinking was performed on the SST-5 and IMDB datasets. The labeling pipeline is implemented in the `evaluate_dataset` submodule, which includes scripts for running sentiment annotation and analyzing the results.

Submodule README: [evaluate_dataset/README.md](evaluate_dataset/README.md)

This folder contains the sentiment labeling and evaluation pipeline using a local vLLM-served model.

How to run labeling:
```bash
cd evaluate_dataset
python run_sentiment_labeling.py --model gpt-120b-medium --datasets sst imdb
```

How to run analysis:
```bash
cd evaluate_dataset
python analyze_label.py
```

What each script does:
- `evaluate_dataset/run_sentiment_labeling.py`: runs LLM-based sentiment annotation for configured datasets and writes labeled outputs + summary metrics into `evaluate_dataset/sentiment_results/`.
- `evaluate_dataset/analyze_label.py`: computes detailed evaluation metrics from labeled CSV outputs and writes `analysis_summary.csv`.
- `evaluate_dataset/scrum.sh`: launches the vLLM server job (port in this script must match `API_BASE_URL` in `run_sentiment_labeling.py`).



## Dependency Notes For Submodules

### LLM Labeling Pipeline
NOTE: The current data files are zipped to reduce repository size. You will need to unzip them before running the scripts.

The `evaluate_dataset` workflow requires:
```bash
pip install openai pandas scikit-learn datasets tqdm
```

Before running labeling, ensure your vLLM server is running and the API port is consistent between server startup (`scrum.sh`) and `run_sentiment_labeling.py`.



# Notes

The MLM pre-training phase utilizes the standard Hugging Face run_mlm.py script. We chose this reference implementation to ensure experimental consistency. You can find the original script in the Transformers examples repository.

---

## Acknowledgements
This project is developed as part of CS4248 (Natural Language Processing) at the National University of Singapore (NUS).
