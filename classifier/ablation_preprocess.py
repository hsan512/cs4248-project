"""
Ablation study on the preprocessing pipeline used by full_train.py.

For each variant we disable exactly one preprocessing component (plus a
`full` baseline with everything on, and a `none` baseline with raw text),
train the same RoBERTa classifier from classifier.full_train.train_model,
and log test accuracy / macro-F1.

Components ablated:
    - ftfy    : ftfy.fix_text encoding repair
    - url     : replace URLs with <URL>
    - user    : replace @mentions / _underscore handles with <USER>
    - tag     : replace #hashtags with <TAG>
    - emoji   : preserve-then-restore unicode emoji / emoticon logic
"""

import os
import re
import time
import json
import argparse

import ftfy
import pandas as pd
import tqdm
tqdm.tqdm.pandas()

from classifier.full_train import train_model
from classifier.utils.clean_text import (
    extract_emojis_with_placeholders,
    restore_emojis,
)


# ---------------------------------------------------------------------------
# Per-component preprocessing, parametrised by what to DISABLE.
# Mirrors classifier.utils.clean_text.clean_text but with toggleable stages.
# ---------------------------------------------------------------------------

URL_PATTERN = r'https?\s*:\s*/\s*/\s*\S+|www[\.,]\S+|www\.\S+'
USER_PATTERN = r'(?<!\w)(?:@\s*[A-Za-z0-9_]+|_+[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_*)'
TAG_PATTERN = r'#\S+'


def make_preprocess(disable=None):
    """
    Build a preprocess_fn(text) -> str that disables the named component.
    `disable` may be one of: None, 'ftfy', 'url', 'user', 'tag', 'emoji',
    'all' (raw text — disable everything).
    """
    d = disable or ""

    def _pp(text):
        if d == "all":
            return str(text) if text is not None else ""

        text = str(text) if text is not None else ""

        if d != "ftfy":
            text = ftfy.fix_text(text)
        text = text.strip()

        # URL → <URL>
        if d != "url":
            text = re.sub(URL_PATTERN, " <URL> ", text)

        # Emoji / emoticon preserve-through-lowercasing
        emo_map = {}
        if d != "emoji":
            text, emo_map = extract_emojis_with_placeholders(text)

        # Username → <USER>
        if d != "user":
            text = re.sub(USER_PATTERN, " <USER> ", text)

        # Hashtag → <TAG>
        if d != "tag":
            text = re.sub(TAG_PATTERN, " <TAG> ", text)

        text = re.sub(r"\s+", " ", text).strip()

        if emo_map:
            text = restore_emojis(text, emo_map)

        return text

    return _pp


VARIANTS = [
    ("full",      None),     # everything on
    ("no_ftfy",   "ftfy"),
    ("no_url",    "url"),
    ("no_user",   "user"),
    ("no_tag",    "tag"),
    ("no_emoji",  "emoji"),
    ("raw",       "all"),    # nothing on
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_variant(name, disable, train_df, num_epochs, out_root):
    print("\n" + "=" * 72)
    print(f"[ABLATION] variant={name}  disable={disable}")
    print("=" * 72)

    pp = make_preprocess(disable)

    df = train_df.copy()
    df["text"] = df["text"].astype(str).progress_apply(pp)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().astype(bool)].drop_duplicates(subset=["text"])

    texts = df["text"].tolist()
    labels = df["sentiment"].tolist()

    variant_dir = os.path.join(out_root, name)
    os.makedirs(variant_dir, exist_ok=True)

    start = time.time()
    metrics = train_model(
        texts, labels,
        num_epochs=num_epochs,
        test_preprocess_fn=pp,
        output_dir=os.path.join(variant_dir, "best_model"),
        predictions_csv_path=os.path.join(variant_dir, "test_predictions.csv"),
    )
    elapsed = time.time() - start

    metrics.update({
        "variant": name,
        "disabled": disable or "",
        "num_train": len(texts),
        "elapsed_sec": round(elapsed, 1),
    })
    with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5,
                    help="epochs per variant (ablations usually run fewer than the full 10)")
    ap.add_argument("--train-csv", default="./data/train.csv")
    ap.add_argument("--out-root", default="./outputs/ablation_preprocess")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these variant names")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    train_df = pd.read_csv(
        args.train_csv, encoding="utf-8", encoding_errors="replace"
    ).dropna().reset_index(drop=True)

    variants = VARIANTS
    if args.only:
        variants = [v for v in VARIANTS if v[0] in set(args.only)]

    all_metrics = []
    for name, disable in variants:
        try:
            m = run_variant(name, disable, train_df, args.epochs, args.out_root)
        except Exception as e:
            print(f"[ABLATION] variant {name} FAILED: {e}")
            m = {"variant": name, "disabled": disable or "", "error": str(e)}
        all_metrics.append(m)

        pd.DataFrame(all_metrics).to_csv(
            os.path.join(args.out_root, "summary.csv"), index=False
        )

    print("\n" + "=" * 72)
    print("ABLATION SUMMARY")
    print("=" * 72)
    print(pd.DataFrame(all_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
