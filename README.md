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

# Notes

The MLM pre-training phase utilizes the standard Hugging Face run_mlm.py script. We chose this reference implementation to ensure experimental consistency. You can find the original script in the Transformers examples repository.

---

## Acknowledgements
This project is developed as part of CS4248 (Natural Language Processing) at the National University of Singapore (NUS).