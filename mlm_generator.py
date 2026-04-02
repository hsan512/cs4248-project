import pandas as pd
import kagglehub
from utils.clean_text import preprocess_pipeline
from tqdm import tqdm
tqdm.pandas()

path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

df = pd.read_csv(f"{path}/train.csv", encoding="utf-8", encoding_errors="replace").dropna()

df["text"] = df["text"].astype(str).progress_apply(preprocess_pipeline)
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])

path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

df_test = pd.read_csv(f"{path}/test.csv", encoding="utf-8", encoding_errors="replace").dropna()

df_test["text"] = df_test["text"].astype(str).progress_apply(preprocess_pipeline)

unique_texts = list(set([p[0] for p in df["text"]] + [p[0] for p in df_test["text"]]))

# 2. Write to the .txt file
with open("tapt_corpus.txt", "w", encoding="utf-8") as f:
    for line in unique_texts:
        # Ensure there are no empty lines or just whitespace
        if line.strip():
            f.write(line.strip() + "\n")

print(f"TAPT Corpus created with {len(unique_texts)} unique lines.")