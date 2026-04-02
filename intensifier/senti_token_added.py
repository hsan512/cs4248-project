import pandas as pd
import kagglehub
from utils.clean_text import preprocess_pipeline
from tqdm import tqdm
tqdm.pandas()

import random
random.seed(42)

_SENTIMENT_ = ["[POS]", "[NEG]"]
def add_sentiment_token(text, label):
    if label == "negative":
        text = "[NEG] " + text
    elif label == "positive":
        text = "[POS] " + text
    else:
        text = f"{random.choice(_SENTIMENT_)} {text}"
    return text

path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

df = pd.read_csv(f"{path}/train.csv", encoding="utf-8", encoding_errors="replace").dropna()

df[["text", "username", "url", "hashtags"]] = df["text"].astype(str).progress_apply(preprocess_pipeline).apply(pd.Series)
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
df["sentiment_text"] = df.progress_apply(lambda row: add_sentiment_token(row["text"], row["sentiment"]), axis=1)

df.to_csv("data/train.csv")

path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")

df_test = pd.read_csv(f"{path}/test.csv", encoding="utf-8", encoding_errors="replace").dropna()

df_test[["text", "username", "url", "hashtags"]] = df_test["text"].astype(str).progress_apply(preprocess_pipeline).apply(pd.Series)
df_test = df_test.dropna(subset=["text"]).drop_duplicates(subset=["text"])
df_test["sentiment_text"] = df_test.progress_apply(lambda row: add_sentiment_token(row["text"], row["sentiment"]), axis=1)

df_test.to_csv("data/test.csv")