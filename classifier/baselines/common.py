import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from classifier.utils import preprocess_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

TEXT_COL = "text"
LABEL_COL = "sentiment"

GLOVE_PATH = os.path.join(DATA_DIR, "glove.6B.100d.txt")
EMBED_DIM = 100
HIDDEN_DIM = 128
MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS_RNN = 10
LR = 1e-3


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the datas
def simple_clean(text):
    cleaned_text, _, _, _ = preprocess_pipeline(text)
    return cleaned_text


def load_data():
    train_df = pd.read_csv(TRAIN_PATH).dropna(subset=[TEXT_COL, LABEL_COL])
    test_df = pd.read_csv(TEST_PATH).dropna(subset=[TEXT_COL, LABEL_COL])

    train_df["clean"] = train_df[TEXT_COL].apply(simple_clean)
    test_df["clean"] = test_df[TEXT_COL].apply(simple_clean)

    le = LabelEncoder()
    le.fit(train_df[LABEL_COL])
    train_df["label"] = le.transform(train_df[LABEL_COL])
    test_df["label"] = le.transform(test_df[LABEL_COL])

    return train_df, test_df, le


# vocab + dataset
class Vocabulary:
    def __init__(self, max_size=25000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.max_size = max_size

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        for word, _ in counter.most_common(self.max_size - 2):
            self.word2idx[word] = len(self.word2idx)

    def encode(self, text, max_len):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(w, 1) for w in tokens]
        ids += [0] * (max_len - len(ids))
        return ids


class SentimentDataset:
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        import torch
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# rnn stuffs
def train_rnn(model, train_loader, optimizer, criterion):
    DEVICE = get_device()
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


def eval_rnn(model, loader):
    DEVICE = get_device()
    model.eval()
    preds, trues = [], []
    import torch
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            trues.extend(y.numpy())
    return np.array(preds), np.array(trues)
