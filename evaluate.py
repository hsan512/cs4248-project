import torch
import joblib
import kagglehub
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from clean_text import preprocess_pipeline
import tqdm
tqdm.tqdm.pandas()

# ==============================
# CONFIG
# ==============================
# change the name according to your saved model name
MODEL_NAME = "best_model"

TEXT_COL = "text"
LABEL_COL = "sentiment"

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
            max_length=128,
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
path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

df = pd.read_csv(f"{path}/test.csv", encoding="utf-8", encoding_errors="replace").dropna()

df["text"] = df["text"].astype(str).progress_apply(preprocess_pipeline)
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])

texts = [p[0] for p in df["text"]]
urls = [p[1] for p in df["text"]]
users = [p[2] for p in df["text"]]

labels = df["sentiment"].tolist()

y_true = [label2id[i] for i in df[LABEL_COL]]

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