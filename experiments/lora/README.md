# LoRA Experiments

This folder contains scripts for training and evaluating a LoRA (Low-Rank Adaptation) fine-tuned RoBERTa model for sentiment classification tasks.

## Folder Structure

- `lora_train.py`: Script to train a LoRA-adapted RoBERTa-base model on the local tweet dataset for 3-class sentiment classification (negative, neutral, positive).
- `evaluate_lora.py`: Script to evaluate the trained LoRA model, along with baseline comparisons (untrained base model and classifier-head-only trained model) on the local test dataset.
- `evaluate_lora_sst5.py`: Script to evaluate the trained LoRA model on the SST-5 dataset (mapped to 3 classes) for out-of-domain assessment.

## Prerequisites

- Python 3.8+
- Required packages: `transformers`, `peft`, `datasets`, `scikit-learn`, `emot`, `pandas`, `numpy`, `torch`
- Data files: `data/train.csv` and `data/test.csv` (with columns 'text' and 'sentiment')

## Running the Experiments

### 1. Train the LoRA Model

To train the LoRA-adapted model:

```bash
python lora_train.py
```

This script will:
- Load and preprocess the training data from `data/train.csv`
- Fine-tune RoBERTa-base with LoRA configuration (r=16, alpha=32, dropout=0.1)
- Perform early stopping based on macro F1 score
- Save the best model to `outputs/best_model_lora`

### 2. Evaluate on Local Test Dataset

To evaluate the trained model on the local test set:

```bash
python evaluate_lora.py
```

This script will:
- Load the test data from `data/test.csv`
- Evaluate three models: base RoBERTa, classifier-head-only trained, and LoRA fine-tuned
- Print a comparison report with accuracy and macro F1 scores

### 3. Evaluate on SST-5 Dataset

To evaluate the trained model on the SST-5 benchmark:

```bash
python evaluate_lora_sst5.py
```

This script will:
- Load the SST-5 test set from Hugging Face datasets
- Map SST-5 labels to 3 classes (very negative/negative → negative, neutral → neutral, positive/very positive → positive)
- Evaluate base RoBERTa and LoRA fine-tuned models
- Print evaluation metrics

## Output

- Trained model and tokenizer are saved to `outputs/best_model_lora`
- Evaluation results are printed to console with detailed classification reports

## Notes

- The scripts use RoBERTa-base as the base model
- Custom tokens `<USER>`, `<URL>`, `<TAG>` are added to the tokenizer
- Text preprocessing includes emoji and emoticon extraction with placeholders
- Training uses a batch size of 64 and early stopping with patience of 5 epochs</content>
<parameter name="filePath">/Users/ziruizhang/Downloads/cs4248-project-main/experiments/lora/README.md