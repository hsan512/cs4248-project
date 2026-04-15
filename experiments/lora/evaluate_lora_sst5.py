import torch
import re
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from emot import emot
from classifier.utils.clean_text import preprocess_pipeline

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

# ==============================
# MAP SST5 TO SST3
# ==============================
def map_sst5_to_sst3(label):
    """
    Convert SST-5 label to SST-3:
    0,1 -> 0 (negative)
    2   -> 1 (neutral)
    3,4 -> 2 (positive)
    """
    if label.lower() == "very positive":
        return "positive"
    elif label.lower() == "very negative":
        return "negative"
    else:
        return label.lower()

# ==============================
# CONFIG
# ==============================
TEXT_COL = "text"
LABEL_COL = "label_text"

# ==============================
# MAIN
# ==============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading local sst5 test dataset...")
    ds = load_dataset("SetFit/sst5", split="test")
    df = ds.to_pandas().dropna()

    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(preprocess_pipeline)
    df = df.dropna(subset=[TEXT_COL]).drop_duplicates(subset=[TEXT_COL])
    texts = [extract_emojis_with_placeholders(t[0]) for t in df[TEXT_COL]]

    df[LABEL_COL] = df[LABEL_COL].astype(str).apply(map_sst5_to_sst3)
    labels = df[LABEL_COL].tolist()


    label2id = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    id2label = {v: k for k, v in label2id.items()}
    true_labels = [label2id[i] for i in labels]

    print(f"Loaded {len(true_labels)} evaluation samples from local dataset.\n")

    # ============================================================
    # EVALUATE BASE MODEL
    # ============================================================
    model_name = "FacebookAI/roberta-base"
    print(f"Loading Base Model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    print("Evaluating Base Model on local sst5 dataset...")
    base_preds = evaluate_model(base_model, tokenizer, texts, device)

    base_acc = accuracy_score(true_labels, base_preds)
    base_f1 = f1_score(true_labels, base_preds, average="macro")

    # ============================================================
    # EVALUATE MODEL (Base + LoRA)
    # ============================================================
    print("\nAttaching LoRA weights (./outputs/best_model_lora)...")
    model = PeftModel.from_pretrained(base_model, "./outputs/best_model_lora").to(device)

    print("Evaluating Model on local sst5 dataset...")
    preds = evaluate_model(model, tokenizer, texts, device)

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="macro")

    # ============================================================
    # PRINT COMPARISON REPORT
    # ============================================================
    print("\n" + "=" * 70)
    print("LOCAL DATASET EVALUATION REPORT (3-label sst5 classification)")
    print("=" * 70)

    print(f"{'Metric':<15} | {'Base (Untrained Head)':<24} | {'(LoRA)':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<15} | {base_acc:<24.4f} | {acc:<20.4f}")
    print(f"{'F1 Macro':<15} | {base_f1:<24.4f} | {f1:<20.4f}")
    print("=" * 70)

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