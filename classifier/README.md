# Classifier Directory

This folder contains the sentiment classifier and preprocessing components used by the CS4248 project.
It includes model training, evaluation scripts, an ablation study for preprocessing, and baseline classifier implementations.

## Folder Structure

- `full_train.py`
  - Main training script for a RoBERTa-based sentiment classifier.
  - Uses `classifier.utils.clean_text.preprocess_pipeline` to preprocess tweet text before training.
  - Saves the best validation model to `./outputs/best_model`.

- `evaluate_classifier.py`
  - Loads a saved classifier from `./outputs/best_model`.
  - Preprocesses test data from `./data/test.csv` and computes accuracy, macro-F1, and a classification report.

- `evaluate_classifier_sst5.py`
  - Evaluates the saved classifier on the SST-5 test split from the `datasets` library.
  - Maps SST-5 labels to 3-way sentiment classes before evaluation.

- `ablation_preprocess.py`
  - Runs a preprocessing ablation study by disabling one preprocessing stage at a time.
  - Each variant trains the same RoBERTa model and stores metrics and predictions in a separate output folder.

- `baselines/`
  - Contains baseline classifier implementations for comparison.
  - Includes `logistic_regression.py`, `naive_bayes.py`, `rnn.py`, and `glove_lstm.py`.

- `utils/`
  - Contains helper preprocessing utilities.
  - `clean_text.py` defines the tweet cleaning pipeline, emoji preservation, and special token handling.

## Recommended Commands

Run these commands from the project root (`cs4248-project-main`):

```bash
# Train the main sentiment classifier
python classifier/full_train.py

# Evaluate the trained classifier on the local test set
python classifier/evaluate_classifier.py

# Evaluate the trained classifier on SST-5 (mapped to 3 sentiment classes)
python classifier/evaluate_classifier_sst5.py

# Run the preprocessing ablation study
python classifier/ablation_preprocess.py --train-csv ./data/train.csv --out-root ./outputs/ablation_preprocess --epochs 5
```

## Notes

- The training script currently writes a processed training file `train_processed.csv` in the project root.
- The classifier expects the saved model and tokenizer in `./outputs/best_model`.
- The preprocessing pipeline normalizes URLs, user mentions, hashtags, and emoji/emoticon content.

## Dependencies

Typical dependencies used by these scripts include:

- Python 3.8+
- `torch`
- `transformers`
- `pandas`
- `scikit-learn`
- `emot`
- `datasets`

Adjust your environment and install missing packages before running the scripts.
