#!/usr/bin/env python3
"""
run_sentiment_labeling.py — LLM-as-labeller for SST-2 and IMDB.

Reuses the inference pattern from run_standard.py:
  - AsyncOpenAI against a local vLLM endpoint
  - responses.create with streaming
  - reasoning_effort
  - asyncio.Semaphore batching
  - robust JSON extraction (handles <think> blocks, fences, nested braces)
  - validation + retry loop

Outputs:
  - labeled_sst2_<model>.csv
  - labeled_imdb_<model>.csv
  - metrics_summary_<model>.csv   (accuracy / precision / recall / macro-F1)

Usage:
    python run_sentiment_labeling.py
    python run_sentiment_labeling.py --datasets sst2
    python run_sentiment_labeling.py --n-sst2 1000 --n-imdb 500
    python run_sentiment_labeling.py --model gpt-120b-medium
"""

import os
import re
import json
import asyncio
import argparse
from typing import List, Dict, Optional

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# ============================================================================
# CONFIGURATION  (mirrors run_standard.py)
# ============================================================================

OUTPUT_DIR    = "sentiment_results"
API_BASE_URL  = "http://localhost:8000/v1"
API_KEY       = "EMPTY"
BATCH_SIZE    = 128
MAX_OUTPUT_TOKENS = 32000
MAX_PARSE_RETRIES = 2
STREAM        = False   # set True to stream deltas, False for one-shot responses.create

MODEL_REGISTRY = {
    "gpt-120b-low": {
        "api_model_name": "openai/gpt-oss-120b",
        "reasoning_effort": "low",
    },
    "gpt-120b-medium": {
        "api_model_name": "openai/gpt-oss-120b",
        "reasoning_effort": "medium",
    },
    "gpt-120b-high": {
        "api_model_name": "openai/gpt-oss-120b",
        "reasoning_effort": "high",
    },
}

VALID_LABELS = {"positive", "neutral", "negative"}

# ============================================================================
# PROMPT — optimized for sentiment classification
# ============================================================================
#
# Design choices:
#   * Clear role + single task (reduces drift).
#   * Explicit rubric for each class (anchors the decision boundary).
#   * Tie-breaker rule: when mixed, judge the DOMINANT / OVERALL sentiment —
#     this is how both SST and IMDB annotators were instructed, so it aligns
#     the model with gold-label conventions.
#   * "Neutral" is reserved for genuinely balanced / factual text, not a
#     fallback for uncertainty — this prevents over-use of neutral.
#   * Forced JSON-only output for deterministic parsing.
#
SENTIMENT_PROMPT = """You are an expert sentiment annotator for movie reviews.

Classify the OVERALL sentiment of the review below into exactly one label:

  - "positive": the reviewer's dominant attitude toward the film is favorable
                (praise, enjoyment, recommendation, admiration).
  - "negative": the reviewer's dominant attitude is unfavorable
                (criticism, disappointment, dislike, warning others away).
  - "neutral" : the review is genuinely balanced or purely descriptive/factual,
                with no clear lean either way. Do NOT use "neutral" just because
                you are unsure — pick the dominant sentiment whenever one exists.

Rules:
  1. Judge the DOMINANT sentiment, not isolated phrases. Mixed reviews that
     still lean one way should get that leaning label.
  2. Sarcasm and irony: interpret the intended meaning, not the literal words.
  3. Ignore plot summary content; weigh only the reviewer's evaluative language.
  4. Output ONLY a JSON object, no explanation, no markdown.

Review:
\"\"\"{text}\"\"\"

Output format:
{{ "label": "positive" | "neutral" | "negative" }}"""


# ============================================================================
# JSON EXTRACTION (lifted from run_standard.py)
# ============================================================================

def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def _find_json_objects(text: str):
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth, start, in_str, escape = 1, i, False, False
            i += 1
            while i < len(text) and depth > 0:
                ch = text[i]
                if escape:
                    escape = False
                elif ch == '\\' and in_str:
                    escape = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == '{': depth += 1
                    elif ch == '}': depth -= 1
                i += 1
            if depth == 0:
                yield text[start:i]
        else:
            i += 1


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = _strip_think(text)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        for block in _find_json_objects(fence.group(1)):
            try:
                return json.loads(block)
            except Exception:
                pass
    for block in sorted(_find_json_objects(text), key=len, reverse=True):
        try:
            return json.loads(block)
        except Exception:
            continue
    return None


def extract_label(raw: str) -> Optional[str]:
    d = _extract_json(raw)
    if d and isinstance(d.get("label"), str):
        lab = d["label"].strip().lower()
        if lab in VALID_LABELS:
            return lab
    # Fallback: bare keyword scan (last-ditch)
    if raw:
        low = raw.lower()
        found = [l for l in VALID_LABELS if l in low]
        if len(found) == 1:
            return found[0]
    return None


def validate_response(raw: str) -> bool:
    return extract_label(raw) is not None


# ============================================================================
# CHECKPOINT — per-dataset JSONL, append-only, crash-safe
# ============================================================================

class Checkpoint:
    def __init__(self, model_alias: str, dataset_name: str, output_dir: str = OUTPUT_DIR):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, f"checkpoint_{dataset_name}_{model_alias}.jsonl")
        self.output_dir = output_dir
        self.model_alias = model_alias
        self.dataset_name = dataset_name
        self._completed: Dict[int, dict] = {}  # row_idx -> successful entry
        self._latest: Dict[int, dict] = {}     # row_idx -> latest entry (success or error)
        self._lock = asyncio.Lock()

    def load(self) -> "Checkpoint":
        if not os.path.exists(self.path):
            return self
        with open(self.path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._latest[e["row_idx"]] = e
                if e.get("error") is None and e.get("pred") is not None:
                    self._completed[e["row_idx"]] = e
        print(f"[Checkpoint:{self.dataset_name}] resumed {len(self._completed)} completed rows")
        return self

    def is_done(self, row_idx: int) -> bool:
        return row_idx in self._completed

    def get(self, row_idx: int) -> Optional[dict]:
        # Prefer successful entry, fall back to latest (likely an error)
        return self._completed.get(row_idx) or self._latest.get(row_idx)

    async def save(self, entry: dict):
        async with self._lock:
            self._latest[entry["row_idx"]] = entry
            if entry.get("error") is None and entry.get("pred") is not None:
                self._completed[entry["row_idx"]] = entry
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as ex:
                print(f"[Checkpoint write fail] {ex}")


# ============================================================================
# LLM CLIENT (same pattern as run_standard.py)
# ============================================================================

class LLMClient:
    def __init__(self, model_config: dict, batch_size: int = BATCH_SIZE):
        self.client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.model = model_config["api_model_name"]
        self.reasoning_effort = model_config.get("reasoning_effort")
        self.sem = asyncio.Semaphore(batch_size)

    async def call(self, prompt: str) -> Dict:
        async with self.sem:
            try:
                kwargs = dict(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
                if self.reasoning_effort:
                    kwargs["reasoning"] = {"effort": self.reasoning_effort}

                if STREAM:
                    kwargs["stream"] = True
                    stream = await self.client.responses.create(**kwargs)
                    full_text = ""
                    async for event in stream:
                        if getattr(event, "type", None) == "response.output_text.delta":
                            full_text += event.delta
                        elif getattr(event, "type", None) == "response.completed":
                            resp_obj = getattr(event, "response", None)
                            if resp_obj and not full_text:
                                for out in getattr(resp_obj, "output", []):
                                    for part in getattr(out, "content", []):
                                        if getattr(part, "type", "") == "output_text":
                                            full_text += getattr(part, "text", "")
                    return {"raw": full_text, "error": None}

                # Non-streaming: single-shot responses.create
                resp = await self.client.responses.create(**kwargs)
                full_text = ""
                text_attr = getattr(resp, "output_text", None)
                if isinstance(text_attr, str) and text_attr:
                    full_text = text_attr
                else:
                    for out in (getattr(resp, "output", []) or []):
                        for part in (getattr(out, "content", []) or []):
                            if getattr(part, "type", "") == "output_text":
                                full_text += getattr(part, "text", "")
                return {"raw": full_text, "error": None}
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                # Print the FIRST few errors loudly so user can diagnose
                if not getattr(LLMClient, "_printed_errors", 0) or LLMClient._printed_errors < 5:
                    print(f"\n[LLM ERROR] {err_msg}")
                    LLMClient._printed_errors = getattr(LLMClient, "_printed_errors", 0) + 1
                return {"raw": "", "error": err_msg[:300]}

    async def call_with_validation(self, prompt: str, max_retries: int = MAX_PARSE_RETRIES) -> Dict:
        for _ in range(1 + max_retries):
            resp = await self.call(prompt)
            if resp["error"] is not None:
                return resp
            if validate_response(resp["raw"]):
                return resp
        resp["error"] = "parse_failure"
        return resp


# ============================================================================
# DATASET LOADING
# ============================================================================

def _concat_splits(load_one_split_fn, splits: List[str]) -> pd.DataFrame:
    frames = []
    for sp in splits:
        try:
            df = load_one_split_fn(sp)
            df["split"] = sp
            frames.append(df)
        except Exception as e:
            print(f"  [warn] failed to load split '{sp}': {e}")
    if not frames:
        raise RuntimeError("No splits loaded.")
    return pd.concat(frames, ignore_index=True)


def load_sst2(n: Optional[int]) -> pd.DataFrame:
    """
    SST-2 (GLUE). Binary gold: 0=negative, 1=positive.
    Concatenates train + validation. The GLUE test split has hidden labels
    (all -1), so it is excluded.
    """
    from datasets import load_dataset
    def _one(split):
        ds = load_dataset("glue", "sst2", split=split)
        df = pd.DataFrame({"text": ds["sentence"], "label_int": ds["label"]})
        df["gold"] = df["label_int"].map({0: "negative", 1: "positive"})
        return df.dropna(subset=["gold"]).reset_index(drop=True)
    df = _concat_splits(_one, ["train", "validation"])
    if n:
        df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    return df[["text", "gold", "split"]]


def load_sst(n: Optional[int]) -> pd.DataFrame:
    """
    SST (SST-5, fine-grained) — all splits concatenated: train + validation + test.
    Native 5-class labels collapsed into 3-class {negative, neutral, positive}:
        0 very negative, 1 negative   -> negative
        2 neutral                     -> neutral
        3 positive,      4 very pos.  -> positive
    Unlike SST-2 and IMDB, SST-5 DOES contain true neutral gold labels.
    """
    from datasets import load_dataset
    mapping = {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}
    def _one(split):
        ds = load_dataset("SetFit/sst5", split=split)
        text_col = "text" if "text" in ds.column_names else "sentence"
        label_col = "label" if "label" in ds.column_names else "label-coarse"
        df = pd.DataFrame({"text": ds[text_col], "label_int": ds[label_col]})
        df["gold"] = df["label_int"].map(mapping)
        return df.dropna(subset=["gold"]).reset_index(drop=True)
    df = _concat_splits(_one, ["train", "validation", "test"])
    if n:
        df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    return df[["text", "gold", "split"]]


def load_imdb(n: Optional[int]) -> pd.DataFrame:
    """
    IMDB — train + test concatenated (both labelled, 25k each).
    The 'unsupervised' split has no labels and is excluded.
    """
    from datasets import load_dataset
    def _one(split):
        ds = load_dataset("imdb", split=split)
        df = pd.DataFrame({"text": ds["text"], "label_int": ds["label"]})
        df["gold"] = df["label_int"].map({0: "negative", 1: "positive"})
        return df.dropna(subset=["gold"]).reset_index(drop=True)
    df = _concat_splits(_one, ["train", "test"])
    if n:
        df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    return df[["text", "gold", "split"]]


# ============================================================================
# LABELING PIPELINE
# ============================================================================

async def label_row(llm: LLMClient, ckpt: Checkpoint, row_idx: int, text: str, gold: str, split: str, pbar) -> None:
    try:
        prompt = SENTIMENT_PROMPT.format(text=text)
        resp = await llm.call_with_validation(prompt)
        pred = extract_label(resp.get("raw", "")) if resp.get("error") is None else None
        entry = {
            "row_idx": row_idx,
            "text": text,
            "gold": gold,
            "split": split,
            "pred": pred,
            "error": resp.get("error"),
            "raw": resp.get("raw", "")[:500],
        }
        await ckpt.save(entry)
    finally:
        pbar.update(1)


async def label_dataset(llm: LLMClient, ckpt: Checkpoint, df: pd.DataFrame, name: str) -> pd.DataFrame:
    n_total = len(df)
    texts = df["text"].tolist()
    golds = df["gold"].tolist()
    splits = df["split"].tolist()

    # Build task list ONLY for rows that still need labelling
    todo = [(i, texts[i], golds[i], splits[i]) for i in range(n_total) if not ckpt.is_done(i)]
    n_done = n_total - len(todo)
    print(f"  {n_done}/{n_total} already labelled, {len(todo)} to go")

    if todo:
        pbar = tqdm(total=len(todo), desc=f"  labeling {name}")
        try:
            tasks = [label_row(llm, ckpt, i, t, g, s, pbar) for (i, t, g, s) in todo]
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print(f"\n  [interrupted] checkpoint preserved at {ckpt.path}")
            raise
        finally:
            pbar.close()

    # Reconstruct full output frame from checkpoint (handles resumed rows too)
    out = df.copy()
    out["row_idx"] = range(len(out))
    preds, errors, raws = [], [], []
    for i in range(len(out)):
        e = ckpt.get(i)
        if e:
            preds.append(e.get("pred"))
            errors.append(e.get("error"))
            raws.append(e.get("raw", ""))
        else:
            preds.append(None)
            errors.append("missing")
            raws.append("")
    out["pred"] = preds
    out["error"] = errors
    out["raw_response"] = raws
    return out


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(df: pd.DataFrame, dataset_name: str) -> List[Dict]:
    """
    Returns metric rows. Behavior depends on whether the dataset has neutral
    in its gold labels:

      * Binary gold (SST-2, IMDB):
          - 'strict'   : neutral/None predictions counted as wrong.
          - 'filtered' : rows with non-binary predictions excluded.
      * 3-class gold (SST / SST-5 collapsed):
          - 'strict' only: full 3-class macro metrics (neutral is a real class).
    """
    gold = df["gold"].tolist()
    pred_raw = df["pred"].tolist()
    gold_set = set(gold)
    has_neutral_gold = "neutral" in gold_set

    neutral_rate = sum(1 for p in pred_raw if p == "neutral") / len(pred_raw)
    invalid_rate = sum(1 for p in pred_raw if p is None) / len(pred_raw)

    rows: List[Dict] = []

    # --- strict view (always reported) ---
    pred_strict = [p if p in VALID_LABELS else "invalid" for p in pred_raw]
    if has_neutral_gold:
        label_space = ["negative", "neutral", "positive"]
    else:
        label_space = sorted(set(gold) | set(pred_strict))
    acc_s = accuracy_score(gold, pred_strict)
    p_s, r_s, f_s, _ = precision_recall_fscore_support(
        gold, pred_strict, labels=label_space, average="macro", zero_division=0
    )
    rows.append({
        "dataset": dataset_name,
        "view": "strict (3-class)" if has_neutral_gold else "strict (neutral/invalid penalized)",
        "n": len(df),
        "accuracy": round(acc_s, 4),
        "precision_macro": round(p_s, 4),
        "recall_macro": round(r_s, 4),
        "f1_macro": round(f_s, 4),
        "neutral_rate": round(neutral_rate, 4),
        "invalid_rate": round(invalid_rate, 4),
    })

    # --- filtered view (only for binary datasets) ---
    if not has_neutral_gold:
        mask = [p in {"positive", "negative"} for p in pred_raw]
        n_kept = sum(mask)
        if n_kept > 0:
            gold_f = [g for g, m in zip(gold, mask) if m]
            pred_f = [p for p, m in zip(pred_raw, mask) if m]
            acc_f = accuracy_score(gold_f, pred_f)
            p_f, r_f, f_f, _ = precision_recall_fscore_support(
                gold_f, pred_f, labels=["negative", "positive"],
                average="macro", zero_division=0
            )
        else:
            acc_f = p_f = r_f = f_f = 0.0
        rows.append({
            "dataset": dataset_name,
            "view": "filtered (binary preds only)",
            "n": n_kept,
            "accuracy": round(acc_f, 4),
            "precision_macro": round(p_f, 4),
            "recall_macro": round(r_f, 4),
            "f1_macro": round(f_f, 4),
            "neutral_rate": round(neutral_rate, 4),
            "invalid_rate": round(invalid_rate, 4),
        })

    return rows


# ============================================================================
# MAIN
# ============================================================================

async def _main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Choices: {list(MODEL_REGISTRY)}")
    model_config = MODEL_REGISTRY[args.model]

    print(f"\n{'='*60}")
    print(f"  MODEL     : {args.model}  ({model_config['api_model_name']})")
    print(f"  REASONING : {model_config.get('reasoning_effort')}")
    print(f"  DATASETS  : {args.datasets}")
    print(f"{'='*60}\n")

    llm = LLMClient(model_config, args.batch_size)
    all_metrics: List[Dict] = []

    async def _run_one(name: str, loader, n_arg: int):
        print(f"\nLoading {name} ...")
        df = loader(n_arg)
        print(f"  {len(df)} examples")
        ckpt = Checkpoint(args.model, name).load()
        labeled = await label_dataset(llm, ckpt, df, name)
        out_path = os.path.join(OUTPUT_DIR, f"labeled_{name}_{args.model}.csv")
        labeled.to_csv(out_path, index=False)
        print(f"  -> {out_path}")
        all_metrics.extend(compute_metrics(labeled, name))

    if "sst2" in args.datasets:
        await _run_one("sst2", load_sst2, args.n_sst2)
    if "sst" in args.datasets:
        await _run_one("sst", load_sst, args.n_sst)
    if "imdb" in args.datasets:
        await _run_one("imdb", load_imdb, args.n_imdb)

    # Summary table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_summary_{args.model}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print(f"\nMetrics saved -> {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-labeller for SST, SST-2, and IMDB sentiment.")
    parser.add_argument("--model", default="gpt-120b-high", choices=list(MODEL_REGISTRY))
    parser.add_argument("--datasets", nargs="+", default=["sst", "sst2", "imdb"],
                        choices=["sst", "sst2", "imdb"])
    parser.add_argument("--n-sst",  type=int, default=0,
                        help="Number of SST (SST-5) examples. 0 = full test set. Default 0.")
    parser.add_argument("--n-sst2", type=int, default=0,
                        help="Number of SST-2 examples. 0 = full validation set. Default 0.")
    parser.add_argument("--n-imdb", type=int, default=0,
                        help="Number of IMDB examples. 0 = full test set (25k). Default 0.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming responses (default: off, single-shot).")
    args = parser.parse_args()
    global STREAM
    STREAM = args.stream
    if args.n_sst  == 0: args.n_sst  = None
    if args.n_sst2 == 0: args.n_sst2 = None
    if args.n_imdb == 0: args.n_imdb = None
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()