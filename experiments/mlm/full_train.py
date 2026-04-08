import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import pandas as pd
from sklearn.metrics import f1_score
import time

from emot import emot
emo = emot()

import tqdm
tqdm.tqdm.pandas()

# =========================
# Emoji / emoticon cleanup
# =========================

def is_bad_emoticon_context(text: str, start: int, end: int) -> bool:
    """
    Filter false-positive emoticons like:
    - 36.7 %)
    - 50%)
    - 12:)
    """
    window = text[max(0, start - 6): min(len(text), end + 6)]

    # examples: 36.7 %), 50%), 20 :)
    if re.search(r"\d+(?:\.\d+)?\s*%\)", window):
        return True

    if re.search(r"\d+(?:\.\d+)?\s*:\)", window):
        return True

    if re.search(r"\d+(?:\.\d+)?\s*:-\)", window):
        return True

    # immediate left-char heuristic
    left = text[start - 1] if start > 0 else ""
    if left.isdigit() or left == "%":
        return True

    return False


def extract_emojis_with_placeholders(text):
    if text is None:
        return ""

    text = str(text)
    found = []

    # --- detect unicode emojis ---
    emoji_info = emo.emoji(text)
    if emoji_info and "value" in emoji_info and "location" in emoji_info:
        for mean, loc in zip(emoji_info["mean"], emoji_info["location"]):
            label = f"*{mean}*" if mean else ""
            found.append({
                "start": loc[0],
                "end": loc[1],
                "label": label,
            })

    # --- detect emoticons ---
    emoticon_info = emo.emoticons(text)
    if emoticon_info and "mean" in emoticon_info and "location" in emoticon_info:
        for mean, loc in zip(emoticon_info["mean"], emoticon_info["location"]):
            if is_bad_emoticon_context(text, loc[0], loc[1]):
                continue
            label = f"*{mean}*" if mean else ""
            found.append({
                "start": loc[0],
                "end": loc[1],
                "label": label,
            })

    # sort by span, longer match first
    found.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    # remove overlaps
    filtered = []
    occupied_until = -1

    for item in found:
        if item["start"] < occupied_until:
            continue
        filtered.append(item)
        occupied_until = item["end"]

    # rebuild text with replacements
    pieces = []
    last = 0

    for item in filtered:
        start, end = item["start"], item["end"]
        label = item["label"]

        pieces.append(text[last:start])
        pieces.append(f" {label} ")
        last = end

    pieces.append(text[last:])
    new_text = "".join(pieces)

    # normalize spacing
    new_text = re.sub(r"\s+", " ", new_text).strip()

    return new_text

# ==============================
# Dataset
# ==============================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ==============================
# Unfreeze
# ==============================
def unfreeze(model):

    # Unfreeze backbone
    for param in model.roberta.parameters():
        param.requires_grad = True

    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

# ==============================
# Layer-wise LR decay
# ==============================
def build_optimizer(
    model,
    base_lr=2e-5,
    classifier_lr=1e-4,
    weight_decay=0.01,
    layerwise_decay=0.9,
):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # embeddings + encoder layers, from bottom -> top
    backbone_layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)

    optimizer_grouped_parameters = []

    # assign smaller LR to lower layers, larger LR to upper layers
    # top encoder layer gets base_lr, lower layers get progressively smaller LR
    num_layers = len(backbone_layers)

    for i, layer in enumerate(backbone_layers):
        layer_lr = base_lr * (layerwise_decay ** (num_layers - 1 - i))

        params_decay = []
        params_nodecay = []

        for name, param in layer.named_parameters():
            if not param.requires_grad:
                continue

            if any(nd in name for nd in no_decay):
                params_nodecay.append(param)
            else:
                params_decay.append(param)

        if params_decay:
            optimizer_grouped_parameters.append(
                {
                    "params": params_decay,
                    "lr": layer_lr,
                    "weight_decay": weight_decay,
                }
            )

        if params_nodecay:
            optimizer_grouped_parameters.append(
                {
                    "params": params_nodecay,
                    "lr": layer_lr,
                    "weight_decay": 0.0,
                }
            )

    # classifier head: usually use a higher LR than backbone
    classifier_decay = []
    classifier_nodecay = []

    for name, param in model.classifier.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay):
            classifier_nodecay.append(param)
        else:
            classifier_decay.append(param)

    if classifier_decay:
        optimizer_grouped_parameters.append(
            {
                "params": classifier_decay,
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            }
        )

    if classifier_nodecay:
        optimizer_grouped_parameters.append(
            {
                "params": classifier_nodecay,
                "lr": classifier_lr,
                "weight_decay": 0.0,
            }
        )

    return torch.optim.AdamW(optimizer_grouped_parameters)

# ==============================
# TRAIN FUNCTION
# ==============================
def train_model(texts, labels,
                model_name="./outputs/roberta-expanded",
                num_epochs=5,
                batch_size=64,
                patience=2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    labels = [label2id[label] for label in labels]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=0.1,
        stratify=labels,
        random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    val_dataset = TextDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(np.unique(labels)),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    best_val_f1 = 0
    patience_counter = 0

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.05 * total_steps)

    unfreeze(model)
    optimizer = build_optimizer(
        model,
        base_lr=2e-5,
        classifier_lr=1e-4,
        weight_decay=0.01,
        layerwise_decay=0.9,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")

        model.train()

        for batch in progress_bar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()

            labels = batch["labels"]
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(**model_inputs)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()

            if device.type == "cuda":
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}

                outputs = model(**model_inputs)
                preds = torch.argmax(outputs.logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = correct / total
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save_pretrained("./outputs/best_model_mlm")
            tokenizer.save_pretrained("./outputs/best_model_mlm")
            print("Saved Best Model")
        else:
            patience_counter += 1
            print(f"Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping")
            break

    print("Best Val Acc:", best_val_f1)

df = pd.read_csv(f"./data/train.csv", encoding="utf-8", encoding_errors="replace").dropna()

df["text"] = df["text"].astype(str).progress_apply(extract_emojis_with_placeholders)
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])

texts = df["text"].tolist()
labels = df["sentiment"].tolist()


# feel free to delete this part, it was here to check the processed texts easier
df_processed = pd.DataFrame({
    "text": texts,
    "sentiment": labels
})

df_processed.to_csv(f"train_processed.csv", index=False)
print ("Finished saving the new df.")


# train the model
start = time.time()
train_model(texts, labels)
print (f"Training time: {time.time()-start}")