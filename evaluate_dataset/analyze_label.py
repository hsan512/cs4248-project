#!/usr/bin/env python3
"""
analyze_labels.py — Read labeled CSVs, compute metrics, show disagreements.

Usage:
    python analyze_labels.py
    python analyze_labels.py --sst path/to/labeled_sst.csv --imdb path/to/labeled_imdb.csv
"""

import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_SST  = "sentiment_results/labeled_sst_gpt-120b-medium.csv"
DEFAULT_IMDB = "sentiment_results/labeled_imdb_gpt-120b-medium.csv"
N_DISAGREE   = 10  # number of disagreement examples to show per dataset


# ============================================================================
# METRICS
# ============================================================================

def compute_all_metrics(gold, pred, label_space):
    """Return a dict of accuracy, macro/micro P/R/F1."""
    acc = accuracy_score(gold, pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        gold, pred, labels=label_space, average="macro", zero_division=0
    )
    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(
        gold, pred, labels=label_space, average="micro", zero_division=0
    )
    return {
        "Accuracy":        round(acc, 4),
        "Precision_macro": round(p_mac, 4),
        "Recall_macro":    round(r_mac, 4),
        "F1_macro":        round(f1_mac, 4),
        "Precision_micro": round(p_mic, 4),
        "Recall_micro":    round(r_mic, 4),
        "F1_micro":        round(f1_mic, 4),
    }


def analyze_dataset(df: pd.DataFrame, name: str, n_disagree: int = N_DISAGREE):
    """Analyze one labeled dataframe. Returns metrics dict + disagreement df."""

    # Drop rows with missing predictions or errors
    total = len(df)
    df = df.dropna(subset=["pred"]).copy()
    df = df[df["error"].isna() | (df["error"] == "")].copy()
    valid = len(df)

    # Determine label space from gold
    gold_labels = sorted(set(df["gold"]))
    has_neutral = "neutral" in gold_labels

    # For binary datasets: filter out neutral predictions (abstentions)
    n_neutral = int((df["pred"] == "neutral").sum())
    neutral_rate = n_neutral / valid if valid > 0 else 0

    if not has_neutral:
        df = df[df["pred"] != "neutral"].copy()
        label_space = ["negative", "positive"]
    else:
        label_space = ["negative", "neutral", "positive"]

    scored = len(df)

    print(f"\n{'='*70}")
    print(f"  DATASET: {name.upper()}")
    print(f"  Total rows: {total}  |  Valid predictions: {valid}  |  Scored: {scored}")
    if not has_neutral and n_neutral > 0:
        print(f"  Neutral predictions dropped: {n_neutral} ({neutral_rate:.2%} abstention rate)")
    print(f"{'='*70}")

    gold = df["gold"].tolist()
    pred = df["pred"].tolist()

    # --- Full metrics ---
    metrics = compute_all_metrics(gold, pred, label_space)
    metrics["Neutral_rate"] = round(neutral_rate, 4)
    print(f"\n  Label space: {label_space}")
    print()

    # Pretty table
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*20} {'-'*10}")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:>10.4f}")

    # --- Per-class report ---
    print(f"\n  Per-class breakdown:")
    report = classification_report(
        gold, pred, labels=label_space, digits=4, zero_division=0
    )
    for line in report.split("\n"):
        print(f"  {line}")

    # --- Confusion matrix ---
    cm = confusion_matrix(gold, pred, labels=label_space)
    cm_df = pd.DataFrame(cm, index=[f"gold:{l}" for l in label_space],
                         columns=[f"pred:{l}" for l in label_space])
    print(f"\n  Confusion Matrix:")
    for line in cm_df.to_string().split("\n"):
        print(f"    {line}")

    # --- Disagreement examples ---
    disagree = df[df["gold"] != df["pred"]].copy()
    print(f"\n  Disagreements: {len(disagree)} / {scored} ({len(disagree)/scored*100:.1f}%)")

    if len(disagree) > 0:
        # Sample diverse disagreements: try to get a mix of error types
        disagree["error_type"] = disagree["gold"] + " → " + disagree["pred"]
        # Stratified sample across error types
        sampled = disagree.groupby("error_type", group_keys=False).apply(
            lambda g: g.sample(n=min(3, len(g)), random_state=42)
        )
        sampled = sampled.head(n_disagree)

        print(f"\n  Sample disagreements (up to {n_disagree}):\n")
        for idx, row in sampled.iterrows():
            text = row["text"]
            if len(text) > 120:
                text = text[:120] + "..."
            print(f"    [{row['gold']:>8} → {row['pred']:<8}]  {text}")

    # --- Per-cell confusion matrix examples (untruncated, for 3-class datasets) ---
    if has_neutral:
        print(f"\n  Per-cell examples (untruncated, 2 per cell):")
        for g_label in label_space:
            for p_label in label_space:
                cell = df[(df["gold"] == g_label) & (df["pred"] == p_label)]
                if len(cell) == 0:
                    continue
                tag = "CORRECT" if g_label == p_label else "ERROR  "
                print(f"\n  --- [{tag}] gold={g_label} → pred={p_label}  (n={len(cell)}) ---")
                samples = cell.sample(n=min(2, len(cell)), random_state=42)
                for i, (_, row) in enumerate(samples.iterrows(), 1):
                    print(f"    [{i}] {row['text']}")

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst",  default=DEFAULT_SST)
    parser.add_argument("--imdb", default=DEFAULT_IMDB)
    parser.add_argument("--n-disagree", type=int, default=N_DISAGREE)
    args = parser.parse_args()

    all_rows = []

    # --- SST ---
    try:
        df_sst = pd.read_csv(args.sst)
        m = analyze_dataset(df_sst, "SST (SST-5 → 3-class)", args.n_disagree)
        m["dataset"] = "SST"
        all_rows.append(m)
    except FileNotFoundError:
        print(f"[skip] {args.sst} not found")

    # --- IMDB ---
    try:
        df_imdb = pd.read_csv(args.imdb)
        m = analyze_dataset(df_imdb, "IMDB", args.n_disagree)
        m["dataset"] = "IMDB"
        all_rows.append(m)
    except FileNotFoundError:
        print(f"[skip] {args.imdb} not found")

    # --- Combined summary table ---
    if all_rows:
        print(f"\n\n{'='*70}")
        print("  COMBINED SUMMARY")
        print(f"{'='*70}\n")
        summary = pd.DataFrame(all_rows)
        cols = ["dataset", "Accuracy", "Precision_macro", "Recall_macro",
                "F1_macro", "Precision_micro", "Recall_micro", "F1_micro",
                "Neutral_rate"]
        summary = summary[cols]
        print(summary.to_string(index=False))

        out_path = "sentiment_results/analysis_summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()