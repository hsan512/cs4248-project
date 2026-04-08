import time
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType
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

class EarlyStoppingVisualizer(TrainerCallback):
    def __init__(self, patience):
        self.patience = patience
        self.patience_counter = 0
        self.best_metric = -float('inf')

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # The trainer adds 'eval_' prefix to your metric name
        current_score = metrics.get(f"eval_{args.metric_for_best_model}")
        
        if current_score is not None:
            if current_score > self.best_metric:
                self.best_metric = current_score
                self.patience_counter = 0
                print(f"\nNew Best {args.metric_for_best_model}: {current_score:.4f} (Patience Reset)")
            else:
                self.patience_counter += 1
                print(f"\nPatience: {self.patience_counter}/{self.patience}")

def main():
    print("Loading Tweet Dataset...")
    df = pd.read_csv("./data/train.csv")

    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("train.csv must contain 'text' and 'sentiment' columns.")

    df = df[["text", "sentiment"]].dropna().reset_index(drop=True)
    df["text"] = df["text"].apply(extract_emojis_with_placeholders)
    df = df.rename(columns={"sentiment": "label"})

    label2id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    id2label = {v: k for k, v in label2id.items()}

    df["label"] = df["label"].astype(str).str.strip().str.lower()

    unknown_labels = sorted(set(df["label"]) - set(label2id.keys()))
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")

    df["label"] = df["label"].map(label2id).astype(int)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    model_name = "FacebookAI/roberta-base"
    print(f"Loading Model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    # Add custom tokens
    new_tokens = ["<USER>", "<URL>", "<TAG>"]
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier", "embed_tokens"],
    )

    model = get_peft_model(model, lora_config)
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.print_trainable_parameters()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir="./outputs/best_model_lora",
        learning_rate=2e-4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        warmup_ratio=0.05,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), EarlyStoppingVisualizer(patience=5)],
    )

    start = time.time()
    trainer.train()
    print(f"Training time: {time.time() - start:.2f} seconds")

    print("Saving best LoRA adapter and tokenizer...")
    trainer.save_model("./outputs/best_model_lora")
    tokenizer.save_pretrained("./outputs/best_model_lora")


if __name__ == "__main__":
    main()