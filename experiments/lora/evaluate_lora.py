import numpy as np
import torch
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel
from emot import emot

emo = emot()

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

def evaluate_model(model, tokenizer, texts, device, batch_size=64):
    """Run batched inference and return predictions."""
    model.eval()
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)

    return predictions


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading local tweet test dataset...")
    df = pd.read_csv("./data/test.csv")
    df["text"] = df["text"].apply(extract_emojis_with_placeholders)

    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("test.csv must contain 'text' and 'sentiment' columns.")

    df = df[["text", "sentiment"]].dropna().reset_index(drop=True)

    label2id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    id2label = {v: k for k, v in label2id.items()}

    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    df["label"] = df["sentiment"].map(label2id).astype(int)

    texts = df["text"].tolist()
    true_labels = df["label"].tolist()

    print(f"Loaded {len(true_labels)} evaluation samples from local dataset.\n")

    # ============================================================
    # 1. EVALUATE BASE MODEL
    # ============================================================
    model_name = "FacebookAI/roberta-base"
    print(f"Loading Base Model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add custom tokens
    new_tokens = ["<USER>", "<URL>", "<TAG>"]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model = base_model.to(device)

    print("Evaluating Base Model on local tweet dataset...")
    base_preds = evaluate_model(base_model, tokenizer, texts, device)

    base_acc = accuracy_score(true_labels, base_preds)
    base_f1 = f1_score(true_labels, base_preds, average="macro")

    # ============================================================
    # 1b. TRAIN CLASSIFIER HEAD ONLY (freeze backbone) AND EVALUATE
    # ============================================================
    print("\nTraining classifier head only (backbone frozen)...")

    train_df = pd.read_csv("./data/train.csv")
    if "text" not in train_df.columns or "sentiment" not in train_df.columns:
        raise ValueError("train.csv must contain 'text' and 'sentiment' columns.")
    train_df = train_df[["text", "sentiment"]].dropna().reset_index(drop=True)
    train_df["text"] = train_df["text"].apply(extract_emojis_with_placeholders)
    train_df["sentiment"] = train_df["sentiment"].astype(str).str.strip().str.lower()
    unknown = sorted(set(train_df["sentiment"]) - set(label2id.keys()))
    if unknown:
        raise ValueError(f"Unknown labels in train.csv: {unknown}")
    train_df["label"] = train_df["sentiment"].map(label2id).astype(int)
    train_df = train_df[train_df["text"].str.strip().astype(bool)].reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    train_ds = train_ds.train_test_split(test_size=0.1, seed=42)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Freeze backbone: only classifier head is trainable
    for name, param in base_model.named_parameters():
        param.requires_grad = name.startswith("classifier.")

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    print(f"Head-only trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    head_args = TrainingArguments(
        output_dir="./outputs/head_only_tmp",
        learning_rate=1e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        warmup_ratio=0.05,
        load_best_model_at_end=False,
        report_to="none",
    )

    head_trainer = Trainer(
        model=base_model,
        args=head_args,
        train_dataset=train_ds["train"],
        eval_dataset=train_ds["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    head_trainer.train()

    head_val_metrics = head_trainer.evaluate()
    print(f"Head-only val accuracy: {head_val_metrics['eval_accuracy']:.4f}, "
          f"macro F1: {head_val_metrics['eval_macro_f1']:.4f}")

    print("\nEvaluating Head-Only Model on local tweet test set...")
    head_preds = evaluate_model(base_model, tokenizer, texts, device)
    head_acc = accuracy_score(true_labels, head_preds)
    head_f1 = f1_score(true_labels, head_preds, average="macro")

    # Restore grads so LoRA attachment below is not affected by frozen flags
    for param in base_model.parameters():
        param.requires_grad = True

    # ============================================================
    # 2. EVALUATE MODEL (Base + LoRA)
    # ============================================================
    print("\nAttaching LoRA weights (./outputs/best_model_lora)...")
    model = PeftModel.from_pretrained(base_model, "./outputs/best_model_lora").to(device)

    print("Evaluating Model on local tweet dataset...")
    preds = evaluate_model(model, tokenizer, texts, device)

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="macro")

    # ============================================================
    # 3. PRINT COMPARISON REPORT
    # ============================================================
    print("\n" + "=" * 70)
    print("LOCAL DATASET EVALUATION REPORT (3-label tweet classification)")
    print("=" * 70)

    print(f"{'Metric':<12} | {'Base (Untrained Head)':<24} | {'Head-Only Trained':<18} | {'LoRA':<8}")
    print("-" * 80)
    print(f"{'Accuracy':<12} | {base_acc:<24.4f} | {head_acc:<18.4f} | {acc:<8.4f}")
    print(f"{'F1 Macro':<12} | {base_f1:<24.4f} | {head_f1:<18.4f} | {f1:<8.4f}")
    print("=" * 80)

    print("\n[Detailed Report: ]")
    print(
        classification_report(
            true_labels,
            preds,
            target_names=["Negative", "Neutral", "Positive"],
            digits=4,
        )
    )

if __name__ == "__main__":
    main()