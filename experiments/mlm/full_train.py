import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
import time

from classifier.utils.clean_text import preprocess_pipeline

import tqdm
tqdm.tqdm.pandas()


def preprocess_text(text):
    return preprocess_pipeline(text)[0]

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

    print("Best Val Macro-F1:", best_val_f1)

    # ============================================================
    # Final evaluation: reload best model → report val + test
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (best model from ./outputs/best_model_mlm)")
    print("=" * 60)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        "./outputs/best_model_mlm"
    ).to(device)
    best_model.eval()

    def _evaluate(loader, name):
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labs = batch.pop("labels")
                out = best_model(**batch)
                preds_all.extend(torch.argmax(out.logits, dim=1).cpu().tolist())
                labels_all.extend(labs.cpu().tolist())
        acc = accuracy_score(labels_all, preds_all)
        f1  = f1_score(labels_all, preds_all, average="macro")
        print(f"\n[{name}] Accuracy: {acc:.4f}   Macro-F1: {f1:.4f}")
        print(classification_report(
            labels_all, preds_all,
            target_names=[id2label[i] for i in sorted(id2label)]
        ))
        return acc, f1

    # ---- Validation ----
    _evaluate(val_loader, "VAL")

    # ---- Test set (data/test.csv) ----
    test_df = pd.read_csv("./data/test.csv",
                          encoding="utf-8", encoding_errors="replace").dropna()
    test_df["text"] = test_df["text"].astype(str).progress_apply(preprocess_text)
    test_df = test_df.dropna(subset=["text"]).drop_duplicates(subset=["text"])

    unknown = sorted(set(test_df["sentiment"]) - set(label2id.keys()))
    if unknown:
        raise ValueError(f"Unknown labels in test.csv: {unknown}")

    test_texts  = test_df["text"].tolist()
    test_labels = [label2id[l] for l in test_df["sentiment"].tolist()]
    test_loader = DataLoader(
        TextDataset(test_texts, test_labels, tokenizer),
        batch_size=batch_size, num_workers=4, pin_memory=True,
    )
    _evaluate(test_loader, "TEST")

df = pd.read_csv(f"./data/train.csv", encoding="utf-8", encoding_errors="replace").dropna()

df["text"] = df["text"].astype(str).progress_apply(preprocess_text)
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