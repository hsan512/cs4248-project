# CS4248 Project – Tweet Sentiment Intensification

## Overview
This project is part of the **CS4248 Natural Language Processing** module at NUS.  
The goal is to build a model that can **intensify the sentiment of a given tweet** while preserving its original meaning.

Instead of simple classification, this is a **generative NLP task**, where the model learns to rewrite text to make it more emotionally expressive (e.g., turning “good” → “absolutely amazing”).

---

## Objectives
- Develop a model that can generate **sentiment-intensified text**
- Preserve the **semantic meaning** of the original input
- Improve the **expressiveness and strength** of sentiment
- Evaluate outputs using both **automatic metrics** and qualitative analysis

---

## Approach
Our pipeline consists of the following stages:

1. **Text Preprocessing**
   - Clean noisy social media text (URLs, usernames, etc.)
   - Normalize tokens for better model understanding

2. **Task-Adaptive Pretraining (TAPT)**
   - Further pretrain a language model on domain-specific tweet data
   - Helps the model better capture Twitter-style language

3. **Fine-tuning**
   - Train the model on the sentiment intensification task
   - Learn how to transform input text into intensified output

4. **Evaluation**
   - Measure performance using metrics such as F1, BLEU, or others
   - Analyze generated outputs qualitatively

---

## Key Features
- Domain-adapted language modeling (TAPT)
- Custom preprocessing for social media text
- Support for special tokens (e.g., `<URL>`, `<USER>`)
- Modular pipeline for experimentation and extension

---

## Project Structure
- `main.py` – Fine-tuning script  
- `evaluate.py` – Evaluation script  
- `clean_text.py` – Text preprocessing  
- `mlm_generator.py` – Generate corpus for TAPT  
- `special_token_added.py` – Add custom tokens  
- `run_mlm.py` – TAPT training script  
- `classifier/baselines/` – Baseline sentiment classifiers for comparison

---

## Baselines
Several baseline sentiment classifiers are included in [classifier/baselines/](classifier/baselines/) to benchmark against the main approach:

- **Logistic Regression** ([logistic_regression.py](classifier/baselines/logistic_regression.py)) – TF-IDF features with a linear classifier
- **Naive Bayes** ([naive_bayes.py](classifier/baselines/naive_bayes.py)) – Multinomial NB over bag-of-words features
- **RNN** ([rnn.py](classifier/baselines/rnn.py)) – Trainable embeddings fed into a recurrent classifier
- **GloVe + LSTM** ([glove_lstm.py](classifier/baselines/glove_lstm.py)) – Pretrained GloVe embeddings with an LSTM classifier


---

## Future Improvements
- Reinforcement Learning for better control of sentiment strength
- More robust evaluation metrics for generation quality
- Better handling of sarcasm and informal language
- Data augmentation for low-resource scenarios

---

## Acknowledgements
This project is developed as part of **CS4248 (Natural Language Processing)** at the  
**National University of Singapore (NUS)**.