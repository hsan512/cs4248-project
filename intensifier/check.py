import pandas as pd
from utils.clean_text import preprocess_pipeline

df_old = pd.read_csv("intensifier/test.csv")
df_old["sentiment_text"] = df_old["sentiment_text"].apply(preprocess_pipeline)
df_old["intensified_text"] = df_old["intensified_text"].apply(preprocess_pipeline)

df_old.to_csv("test.csv")