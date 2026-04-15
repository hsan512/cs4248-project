import torch
import re
from emot import emot
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from classifier.utils.clean_text import preprocess_pipeline
import tqdm
tqdm.tqdm.pandas()
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

# ==============================
# MAP LABEL FROM 5 LABELS TO 3 LABELS
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
MODEL_NAME = "./outputs/best_model_mlm"

TEXT_COL = "text"
LABEL_COL = "label_text"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Load tokenizer + label encoder
# ==============================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

id2label = model.config.id2label
label2id = model.config.label2id

# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# ==============================
# Prediction function
# ==============================
def predict_test(model, tokenizer, test_texts, batch_size=32):
    predictions = []
    # Force everything to string once
    clean_texts_list = [str(t) if t is not None else "" for t in test_texts]

    for i in tqdm.tqdm(range(0, len(clean_texts_list), batch_size), desc="Predicting"):
        # Use the cleaned list here!
        batch_texts = clean_texts_list[i:i+batch_size]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())

    return predictions

model.eval()

# ==============================
# Load test dataset
# ==============================
ds = load_dataset("SetFit/sst5", split="test")
df = ds.to_pandas().dropna()

df[TEXT_COL] = df[TEXT_COL].astype(str).progress_apply(preprocess_pipeline)
df = df.dropna(subset=[TEXT_COL]).drop_duplicates(subset=[TEXT_COL])
texts = [extract_emojis_with_placeholders(t[0]) for t in df[TEXT_COL]]

df[LABEL_COL] = df[LABEL_COL].astype(str).progress_apply(map_sst5_to_sst3)
labels = df[LABEL_COL].tolist()

y_true = [label2id[i] for i in labels]

# ==============================
# Run prediction
# ==============================
y_pred = predict_test(model, tokenizer, texts)

# ==============================
# Evaluation
# ==============================
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=[id2label[i] for i in sorted(id2label)]
))