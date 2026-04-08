"""BiLSTM baseline with GloVe pretrained embeddings."""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from classifier.baselines.common import (
    load_data, Vocabulary, SentimentDataset,
    train_rnn, eval_rnn, get_device,
    GLOVE_PATH, EMBED_DIM, HIDDEN_DIM, MAX_LEN, BATCH_SIZE, EPOCHS_RNN, LR, DATA_DIR,
)


def load_glove(path, word2idx, embed_dim):
    """Load GloVe vectors for words in vocabulary."""
    embeddings = np.random.normal(0, 0.1, (len(word2idx), embed_dim))
    embeddings[0] = 0  # PAD
    found = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"  GloVe: loaded {found}/{len(word2idx)} word vectors")
    return torch.tensor(embeddings, dtype=torch.float32)


class GloVeLSTMClassifier(nn.Module):
    def __init__(self, pretrained_embeds, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeds, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(pretrained_embeds.shape[1], hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        hidden = torch.cat([h[0], h[1]], dim=1)
        return self.fc(self.dropout(hidden))


def run_glove_lstm(train_df, test_df, le, vocab):
    print("\n" + "=" * 50)
    print("BiLSTM + GloVe Pretrained Embeddings")
    print("=" * 50)

    if not os.path.exists(GLOVE_PATH):
        print(f"  [SKIP] GloVe file not found at {GLOVE_PATH}")
        print(f"  Download glove.6B.100d.txt from https://nlp.stanford.edu/projects/glove/")
        print(f"  and place it in {DATA_DIR}/")
        return

    glove_embeds = load_glove(GLOVE_PATH, vocab.word2idx, EMBED_DIM)

    train_ds = SentimentDataset(train_df["clean"].tolist(), train_df["label"].values, vocab, MAX_LEN)
    test_ds = SentimentDataset(test_df["clean"].tolist(), test_df["label"].values, vocab, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    DEVICE = get_device()
    num_classes = len(le.classes_)
    model = GloVeLSTMClassifier(glove_embeds, HIDDEN_DIM, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    best_state = None
    for epoch in range(EPOCHS_RNN):
        loss = train_rnn(model, train_loader, optimizer, criterion)
        y_pred, y_true = eval_rnn(model, test_loader)
        f1 = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)
        print(f"  Epoch {epoch+1}/{EPOCHS_RNN}  loss={loss:.4f}  acc={acc:.4f}  macro-f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(DEVICE)
    y_pred, y_true = eval_rnn(model, test_loader)
    print(f"\nBest results:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    from classifier.baselines.rnn import run_rnn

    train_df, test_df, le = load_data()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  Classes: {list(le.classes_)}")

    # Build vocab via the RNN baseline, then use it for GloVe
    vocab = run_rnn(train_df, test_df, le)
    run_glove_lstm(train_df, test_df, le, vocab)
