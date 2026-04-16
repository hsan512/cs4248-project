# Intensifier Folder

This folder contains scripts for training and evaluating an intensifier model that amplifies the sentiment in text inputs. The intensifier uses a combination of supervised fine-tuning (SFT) and reinforcement learning (RL) to generate intensified versions of sentiment-bearing texts.

## Folder Structure

- `senti_token_added.py`: Preprocesses the dataset by adding sentiment tokens ([POS] for positive, [NEG] for negative) to the text data. Downloads data from Kaggle and saves processed train.csv and test.csv to the data/ directory.
- `sft.py`: Performs supervised fine-tuning on a base language model (e.g., TinyLlama) using the processed dataset. Trains the model to generate intensified text based on sentiment tokens.
- `rl.py`: Applies reinforcement learning to further fine-tune the SFT model. Uses a reward model (classifier) to guide the generation towards more intensified outputs.
- `gpt_gen.py`: Generates intensified text using GPT-4o for comparison or baseline purposes. Processes test.csv and outputs intensified versions.
- `evaluate_intensifier.py`: Evaluates the intensifier models by computing metrics such as BLEU, ROUGE, BERTScore, and sentiment accuracy. Compares outputs from different models (base, SFT, RL, GPT).
- `evaluate_ppl.py`: Computes perplexity scores for generated texts using a reference model (e.g., Llama-3-8B). Evaluates how well the intensified texts align with natural language.

## Running Experiments

### Prerequisites
- Python 3.8+
- Required packages: Install dependencies via `pip install -r requirements.txt` (assuming a requirements.txt exists in the root directory).
- Hugging Face token for model access (set in scripts or environment).
- Access to Kaggle API for data download (for `senti_token_added.py`).

### Step-by-Step Guide

1. **Prepare Data**:
   Run the data preprocessing script to add sentiment tokens:
   ```
   python senti_token_added.py
   ```
   This will download the dataset, clean the text, and save `data/train.csv` and `data/test.csv` with sentiment tokens added.

2. **Supervised Fine-Tuning (SFT)**:
   Train the base model with SFT:
   ```
   python sft.py
   ```
   This trains the model and saves checkpoints to `./outputs/sft/`. Adjust parameters in the script as needed (e.g., model name, batch size).

3. **Reinforcement Learning (RL)**:
   Fine-tune the SFT model with RL:
   ```
   python rl.py
   ```
   Requires the SFT model path and a classifier model. Outputs to `./outputs/rl/`.

4. **Generate with GPT (Optional Baseline)**:
   Generate intensified text using GPT-4o:
   ```
   python gpt_gen.py
   ```
   Processes `test.csv` and saves outputs to `test_.csv`. Requires OpenAI API key.

5. **Evaluate Intensifier**:
   Evaluate the models' performance:
   ```
   python evaluate_intensifier.py
   ```
   Computes and prints various metrics comparing base model, SFT, RL, and GPT outputs.

6. **Evaluate Perplexity**:
   Compute perplexity for generated texts:
   ```
   python evaluate_ppl.py
   ```
   Requires generated CSV files in `eval_outputs/` and a reference model. Outputs perplexity scores.

### Notes
- Ensure CUDA is available for GPU acceleration if training large models.
- Model paths and configurations can be modified in the script configs (e.g., `RLConfig` in `rl.py`).
- Outputs are typically saved in `./outputs/` or `eval_outputs/` directories.
- For detailed configurations, refer to the dataclass configs within each script.