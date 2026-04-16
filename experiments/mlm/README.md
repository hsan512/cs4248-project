# MLM Experiments

This folder contains scripts and data for experiments involving Masked Language Modeling (MLM) and downstream classification tasks.

## Folder Structure

- `data_corpus.txt`: The text corpus used for MLM training, generated from a sentiment analysis dataset. Contains preprocessed unique text lines.
- `mlm_generator.py`: Script to generate the MLM data corpus. Downloads data from Kaggle, preprocesses it, and writes to `data_corpus.txt`.
- `run_mlm.py`: Main script for fine-tuning a transformer model (e.g., BERT, RoBERTa) for masked language modeling on the corpus.
- `full_train.py`: Script for training a sequence classification model, likely using a model pretrained with MLM.
- `evaluate_classifier.py`: Script to evaluate the trained classifier on a test dataset, computing metrics like accuracy and F1-score.
- `evaluate_classifier_sst5.py`: Evaluation script specifically for the SST-5 (Stanford Sentiment Treebank) dataset.
- `special_token_added.py`: Script to add special tokens to the tokenizer or model.

## Dependencies

The scripts require the following Python packages (as indicated in `run_mlm.py`):
- transformers (from Hugging Face)
- albumentations >= 1.4.16
- accelerate >= 0.12.0
- torch >= 1.3
- datasets >= 2.14.0
- sentencepiece != 0.1.92
- protobuf
- evaluate
- scikit-learn
- pandas
- tqdm
- kagglehub (for data download)
- emot (for emoji processing)

Install dependencies using pip:
```
pip install transformers albumentations accelerate torch datasets sentencepiece protobuf evaluate scikit-learn pandas tqdm kagglehub emot
```

## Running the Experiments

1. **Generate MLM Data Corpus**:
   ```
   python mlm_generator.py
   ```
   This downloads the sentiment analysis dataset from Kaggle and creates `data_corpus.txt`.

2. **Run MLM Training**:
   ```
   python run_mlm.py --model_name_or_path <model_name> --train_file data_corpus.txt --output_dir <output_dir>
   ```
   Replace `<model_name>` with a Hugging Face model like `bert-base-uncased`, and `<output_dir>` with your desired output directory. Add other arguments as needed (see script for options).

3. **Train Classifier (Full Training)**:
   ```
   python full_train.py
   ```
   This trains a classification model. Ensure the data paths and model configurations are set correctly in the script.

4. **Evaluate Classifier**:
   ```
   python evaluate_classifier.py
   ```
   Evaluates the classifier on the test data. Modify the script to point to your trained model and test dataset.

5. **Evaluate on SST-5**:
   ```
   python evaluate_classifier_sst5.py
   ```
   Specific evaluation for SST-5 dataset.

6. **Add Special Tokens**:
   ```
   python special_token_added.py
   ```
   Adds special tokens to the model/tokenizer as needed.

Note: Ensure you have the necessary data files (e.g., train.csv, test.csv) in the appropriate locations, and configure paths in the scripts if needed. Some scripts may require GPU for training.